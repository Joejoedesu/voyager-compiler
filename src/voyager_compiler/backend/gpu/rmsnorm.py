"""RMSNorm's backward as one Triton kernel, for a training step's program.

A captured ``LlamaRMSNorm`` reaches a gradient program as core aten ops: the
forward ``xf = x.float()``, ``r = rsqrt(mean(xf ** 2, -1) + eps)``,
``y = w * (xf * r).to(bfloat16)``, and its backward as some twenty ops,
which Inductor runs as two kernels, each reading the gradient ``dy``: one
for the input gradient, one, with a final sum, for the weight gradient's.
``fuse_rmsnorm_backward`` replaces each such backward with
``rmsnorm_backward``, which on a GPU reads ``dy`` and ``x`` once for both
gradients.  It also sums the up to ``MAX_GRADIENTS`` gradients that add up
to ``dy``, the input gradients of the GEMMs reading ``y``, and adds the
input gradient to the residual stream's, as the program does.

The kernel rounds to bfloat16 where the program's ops do, so every
elementwise value is the program's, run eagerly, bit for bit; only its
sums, a row's dot product and the weight gradient's sum over the rows, add
in another order.  Compiled, the program is not bit for bit its eager run
either: Inductor fuses its ops, keeping some intermediates in float32, and
the fused program differs from the unfused one by about as much.
"""

import operator
from typing import List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl
from torch.fx import Graph, Node
from torch.library import Library, impl

__all__ = ["fuse_rmsnorm_backward", "rmsnorm_backward"]

aten = torch.ops.aten

# The most gradients one kernel sums into ``dy``.
MAX_GRADIENTS = 3
# The rows one program of the kernel takes, summing their weight gradient,
# and its warps.
ROWS_PER_PROGRAM = 16
NUM_WARPS = 8
# The partial weight gradients the final sum adds at a time.
SUM_ROWS = 128
# The columns of the weight gradient one program of the final sum takes.
SUM_COLUMNS = 32
# nvfp4.py defines the ``voyager_gpu`` namespace; this adds to it.
_LIBRARY = Library("voyager_gpu", "FRAGMENT")


@triton.jit
def _bf16(value):
    """``value`` rounded to bfloat16, back in float32."""
    return value.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _backward_kernel(
    dy0,
    dy1,
    dy2,
    x,
    rsqrt,
    weight,
    residual,
    dx,
    partial_dw,
    rows,
    cols,
    GRADIENTS: tl.constexpr,
    RESIDUAL: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """The input gradient of ``ROWS`` rows and their weight gradient's sum.

    Each row is read once, whole; ``dy`` is the sum of ``GRADIENTS``
    gradients, ``dy0`` first, and the input gradient adds to ``residual``'s
    row when ``RESIDUAL``.
    """
    n = tl.arange(0, BLOCK)
    mask = n < cols
    w = tl.load(weight + n, mask=mask, other=0.0).to(tl.float32)
    inv_cols = tl.div_rn(1.0, cols.to(tl.float32))
    dw = tl.zeros([BLOCK], dtype=tl.float32)
    first = tl.program_id(0) * ROWS
    for m in range(first, tl.minimum(first + ROWS, rows)):
        offsets = m * cols + n
        dy = tl.load(dy0 + offsets, mask=mask, other=0.0).to(tl.float32)
        if GRADIENTS > 1:
            dy = _bf16(dy + tl.load(dy1 + offsets, mask=mask, other=0.0))
        if GRADIENTS > 2:
            dy = _bf16(dy + tl.load(dy2 + offsets, mask=mask, other=0.0))
        xf = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(rsqrt + m)
        dw += _bf16(dy * _bf16(xf * r))
        g = _bf16(dy * w)
        # d(rsqrt)/d(xf): -0.5 r^3 * 2 xf / cols, times the row's g . xf.
        scale = tl.sum(g * xf, axis=0) * -0.5 * (r * r * r) * inv_cols
        gradient = _bf16(g * r + scale * (xf * 2.0))
        if RESIDUAL:
            gradient = gradient + tl.load(
                residual + offsets, mask=mask, other=0.0
            ).to(tl.float32)
        tl.store(dx + offsets, gradient.to(tl.bfloat16), mask=mask)
    tl.store(partial_dw + tl.program_id(0) * cols + n, dw, mask=mask)


@triton.jit
def _sum_kernel(
    partial_dw,
    dw,
    programs,
    cols,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
):
    """The bfloat16 weight gradient of ``COLUMNS`` columns.

    Each column's is the sum of its ``programs`` partial sums.
    """
    n = tl.program_id(0) * COLUMNS + tl.arange(0, COLUMNS)
    total = tl.zeros([COLUMNS], dtype=tl.float32)
    for start in range(0, programs, ROWS):
        p = start + tl.arange(0, ROWS)
        mask = (p[:, None] < programs) & (n[None, :] < cols)
        partial = tl.load(
            partial_dw + p[:, None] * cols + n[None, :], mask=mask, other=0.0
        )
        total += tl.sum(partial, axis=0)
    tl.store(dw + n, total.to(tl.bfloat16), mask=n < cols)


_LIBRARY.define(
    "rmsnorm_backward(Tensor[] gradients, Tensor x, Tensor rsqrt, "
    "Tensor weight, Tensor? residual) -> (Tensor, Tensor)"
)


@impl(_LIBRARY, "rmsnorm_backward", "CUDA")
def rmsnorm_backward_cuda(
    gradients: List[torch.Tensor],
    x: torch.Tensor,
    rsqrt: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``rmsnorm_backward`` of contiguous CUDA bfloat16 tensors on a GPU.

    Two Triton kernels: both gradients from one read of the rows, then the
    sum of the weight gradient's partial sums.
    """
    cols = x.shape[-1]
    rows = x.numel() // cols
    programs = triton.cdiv(rows, ROWS_PER_PROGRAM)
    dx = torch.empty_like(x)
    partial_dw = torch.empty(
        (programs, cols), dtype=torch.float32, device=x.device
    )
    # Unused pointers repeat a used one.
    dys = [*gradients, *gradients[:1] * (MAX_GRADIENTS - len(gradients))]
    _backward_kernel[(programs,)](
        *dys,
        x,
        rsqrt,
        weight,
        x if residual is None else residual,
        dx,
        partial_dw,
        rows,
        cols,
        len(gradients),
        residual is not None,
        ROWS_PER_PROGRAM,
        triton.next_power_of_2(cols),
        num_warps=NUM_WARPS,
        # The program rounds each product and sum on its own.
        enable_fp_fusion=False,
    )
    dw = torch.empty(cols, dtype=weight.dtype, device=x.device)
    _sum_kernel[(triton.cdiv(cols, SUM_COLUMNS),)](
        partial_dw, dw, programs, cols, SUM_ROWS, SUM_COLUMNS
    )
    return dx, dw


@torch.library.register_fake("voyager_gpu::rmsnorm_backward")
def _(gradients, x, rsqrt, weight, residual):
    return torch.empty_like(x), weight.new_empty(x.shape[-1])


def rmsnorm_backward(
    gradients: Sequence[torch.Tensor],
    x: torch.Tensor,
    rsqrt: torch.Tensor,
    weight: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm's input and weight gradients, as its decomposed backward.

    Contiguous CUDA bfloat16 tensors, each gradient shaped as ``x``, run
    in ``rmsnorm_backward_cuda``'s kernels; others run the decomposed
    backward's ops.

    Args:
        gradients: The gradients summing to the norm's output gradient
            ``dy``, from the first.
        x: The norm's input.
        rsqrt: The forward's ``rsqrt(mean(x.float() ** 2, -1) + eps)``,
            float32, shaped as ``x`` with a last dimension of 1.
        weight: The norm's weight, ``[x.shape[-1]]``.
        residual: The gradient the input gradient adds to, shaped as ``x``,
            or None.

    Returns:
        ``(dx, dw)``: the input gradient, plus ``residual``, and the weight
        gradient, in ``weight``'s dtype.
    """
    rows = [*gradients, x, *([] if residual is None else [residual])]
    if (
        x.is_cuda
        and len(gradients) <= MAX_GRADIENTS
        and all(t.shape == x.shape for t in rows)
        and weight.shape == x.shape[-1:]
        and all(t.dtype is torch.bfloat16 for t in [*rows, weight])
        and rsqrt.dtype is torch.float32
        and all(t.is_contiguous() for t in [*rows, weight, rsqrt])
    ):
        return torch.ops.voyager_gpu.rmsnorm_backward(
            list(gradients), x, rsqrt, weight, residual
        )
    dy = gradients[0]
    for gradient in gradients[1:]:
        dy = dy + gradient
    xf = x.float()
    dw = (dy * (xf * rsqrt).to(x.dtype)).sum(tuple(range(x.ndim - 1)))
    g = (dy * weight).float()
    scale = (g * xf).sum(-1, keepdim=True) * -0.5 * rsqrt.pow(3)
    dx = (g * rsqrt + scale / x.shape[-1] * (xf * 2.0)).to(x.dtype)
    return dx if residual is None else residual + dx, dw


def _input(node) -> Optional[Node]:
    """``node``'s first argument if ``node`` is a Node, else None."""
    return node.args[0] if isinstance(node, Node) and node.args else None


def _sole_user(node) -> Optional[Node]:
    """``node``'s only user if ``node`` is a Node, else None."""
    if not isinstance(node, Node) or len(node.users) != 1:
        return None
    return next(iter(node.users))


def _call(node, target, *args) -> bool:
    """Whether ``node`` calls ``target`` on ``args``, without keywords."""
    return (
        isinstance(node, Node)
        and node.op == "call_function"
        and node.target is target
        and node.args == args
        and not node.kwargs
    )


def _cast(node, source: Node, dtype: torch.dtype) -> bool:
    """Whether ``node`` is ``source.to(dtype)``."""
    return (
        isinstance(node, Node)
        and node.target is aten._to_copy.default
        and node.args == (source,)
        and node.kwargs.get("dtype") is dtype
        and set(node.kwargs) <= {"dtype", "layout", "device"}
        and node.kwargs.get("layout", torch.strided) is torch.strided
    )


def _match(rsqrt: Node) -> Optional[Tuple[Node, ...]]:
    """The RMSNorm computing ``rsqrt``, if its backward is the decomposed one.

    Every op must be the decomposed backward's, operand for operand.

    Returns:
        ``(x, weight, dy, dw, dx)``: the norm's input and weight, its output
        gradient, its weight gradient and its bfloat16 input gradient; None
        if any op differs.
    """
    # The forward: xf = x.float(); rsqrt(mean(xf ** 2, -1) + eps).
    eps = rsqrt.args[0]
    mean = _input(eps)
    square = _input(mean)
    xf = _input(square)
    x = _input(xf)
    if not (
        len(eps.args) == 2
        and isinstance(eps.args[1], float)
        and _call(eps, aten.add.Tensor, mean, eps.args[1])
        and _call(mean, aten.mean.dim, square, [-1], True)
        and _call(square, aten.pow.Tensor_Scalar, xf, 2)
        and _cast(xf, x, torch.float32)
        and "val" in x.meta
    ):
        return None
    shape = list(x.meta["val"].shape)
    cols = shape[-1]

    # The forward's weight * n and the backward's dy * n, with
    # n = (xf * rsqrt).to(bfloat16).
    normalized = [
        u for u in rsqrt.users if _call(u, aten.mul.Tensor, xf, rsqrt)
    ]
    if len(normalized) != 1:
        return None
    n = _sole_user(normalized[0])
    if not _cast(n, normalized[0], torch.bfloat16) or len(n.users) != 2:
        return None
    # The forward's product comes first.
    y, gn = sorted(n.users)
    weight, dy = _input(y), _input(gn)
    if not (
        _call(y, aten.mul.Tensor, weight, n)
        and _call(gn, aten.mul.Tensor, dy, n)
        and isinstance(weight, Node)
        and isinstance(dy, Node)
        and len(dy.users) == 2
    ):
        return None

    # dw = (dy * n).sum(rows).
    total_n = _sole_user(gn)
    dw = _sole_user(total_n)
    leading = list(range(len(shape) - 1))
    if not (
        _call(total_n, aten.sum.dim_IntList, gn, leading, True)
        and _call(dw, aten.view.default, total_n, [cols])
    ):
        return None

    # dx = g * rsqrt + sum(g * xf) * -0.5 * rsqrt ** 3 / cols * (xf * 2.0),
    # with g = (dy * weight).float().
    gw = [u for u in dy.users if _call(u, aten.mul.Tensor, dy, weight)]
    if len(gw) != 1:
        return None
    g = _sole_user(gw[0])
    if not _cast(g, gw[0], torch.float32) or len(g.users) != 2:
        return None
    gx = [u for u in g.users if _call(u, aten.mul.Tensor, g, xf)]
    gr = [u for u in g.users if _call(u, aten.mul.Tensor, g, rsqrt)]
    if len(gx) != 1 or len(gr) != 1:
        return None
    dot = _sole_user(gx[0])
    half = _sole_user(dot)
    cube = _sole_user(half)
    power = cube.args[-1] if cube is not None and cube.args else None
    if not (
        _call(dot, aten.sum.dim_IntList, gx[0], [len(shape) - 1], True)
        and _call(half, aten.mul.Scalar, dot, -0.5)
        and _call(cube, aten.mul.Tensor, half, power)
        and _call(power, aten.pow.Tensor_Scalar, _input(power), 3)
    ):
        return None
    base = _input(power)
    while isinstance(base, Node) and base.target is aten.alias.default:
        base = base.args[0]
    expand = _sole_user(cube)
    divided = _sole_user(expand)
    term = _sole_user(divided)
    double = term.args[-1] if term is not None and term.args else None
    total = _sole_user(term)
    dx = _sole_user(total)
    if not (
        base is rsqrt
        and _call(expand, aten.expand.default, cube, shape)
        and _call(divided, aten.div.Scalar, expand, cols)
        and _call(term, aten.mul.Tensor, divided, double)
        and _call(double, aten.mul.Scalar, _input(double), 2.0)
        and _call(_input(double), aten.pow.Tensor_Scalar, xf, 1.0)
        and _call(total, aten.add.Tensor, gr[0], term)
        and _cast(dx, total, torch.bfloat16)
    ):
        return None
    return x, weight, dy, dw, dx


def _gradients(dy: Node) -> List[Node]:
    """The at most ``MAX_GRADIENTS`` gradients a chain of adds sums to ``dy``.

    The chain leans left, so ``(a + b) + c`` is ``[a, b, c]``; an add another
    op also reads is not split.
    """
    gradients = [dy]
    while len(gradients) < MAX_GRADIENTS:
        add = gradients[0]
        if not (
            add.target is aten.add.Tensor
            and len(add.args) == 2
            and not add.kwargs
            and (add is dy or len(add.users) == 1)
            and all(isinstance(a, Node) for a in add.args)
        ):
            break
        gradients[:1] = add.args
    return gradients


def fuse_rmsnorm_backward(graph: Graph) -> None:
    """Replace each decomposed RMSNorm backward with one ``rmsnorm_backward``.

    The graph is rewritten in place, leaving the ops replaced to dead-code
    elimination.  A backward is replaced only if every op of it is the
    decomposed backward's (``_match``).  The adds summing its output
    gradient (``_gradients``) go into the kernel, and so does the add of its
    input gradient to another gradient, the residual stream's, when that is
    the input gradient's only user and the other gradient is ready before
    the backward.

    Args:
        graph: A gradient program's graph.
    """
    for rsqrt in graph.find_nodes(
        op="call_function", target=aten.rsqrt.default
    ):
        match = _match(rsqrt)
        if match is None:
            continue
        x, weight, dy, dw, dx = match
        first = min(dw, dx)
        add = _sole_user(dx)
        residual = None
        if add is not None and len(add.args) == 2:
            residual = add.args[0] if add.args[1] is dx else add.args[1]
        if not (
            (
                _call(add, aten.add.Tensor, residual, dx)
                or _call(add, aten.add.Tensor, dx, residual)
            )
            and isinstance(residual, Node)
            and residual < first
        ):
            residual = None
        with graph.inserting_before(first):
            backward = graph.call_function(
                rmsnorm_backward,
                (_gradients(dy), x, rsqrt, weight, residual),
            )
            input_gradient, weight_gradient = (
                graph.call_function(operator.getitem, (backward, i))
                for i in range(2)
            )
        output = dx if residual is None else add
        output.replace_all_uses_with(input_gradient)
        dw.replace_all_uses_with(weight_gradient)

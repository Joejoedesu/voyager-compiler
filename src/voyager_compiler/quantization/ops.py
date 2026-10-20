"""The fake-quant and calibration ops, ``torch.ops.quantization.*``.

``voyager_compiler.ops`` holds the ops the hardware runs; these emulate
quantization in training and calibrate it.  Each op's reference is written
in torch ops: it is the op's CPU kernel and the oracle its CUDA kernel is
tested against.  On a GPU, ``mx_fake_quant``, ``hadamard_amax`` and
``bin_blocks`` run their Triton kernels (``kernels``) where those compute
the call exactly -- ``bin_blocks``' does any call; ``fake_quant`` and
``group_wise_affine_fake_quant`` have none.  The other calls run the
reference, a fake-quant's compiled, as eager GPU code runs.  The
fake-quants are ``triton_op``s, so that ``torch.compile`` traces their
bodies -- a kernel's launch, or the reference -- into the compiled program
rather than calling them as opaque kernels.  Each takes the
straight-through estimator as its backward.
"""

import importlib.util
from typing import List, Optional, Tuple

import torch
from torch._library.triton import set_wrap_triton_enabled
from torch.library import triton_op

from voyager_compiler.ops import (
    calculate_mx_qparam,
    compile_like_eager,
    hadamard_rotate,
    quantize_mx,
    repeat_per_block,
    vmap,
)
from voyager_compiler.quantization.mx_utils import _reshape_to_blocks

# The Triton kernels need triton, which ships only with the CUDA builds of
# torch; without it every call runs its reference.
kernels = None
if importlib.util.find_spec("triton") is not None:
    from voyager_compiler.quantization import kernels

__all__ = [
    "bin_blocks",
    "fake_quant",
    "group_wise_affine_fake_quant",
    "hadamard_amax",
    "mx_fake_quant",
]

#: Elements the reference ``bin_blocks`` widens to float64 at a time.
#: Widening a tensor and forming the two products is the pass's whole
#: memory cost, and a model's largest weight runs to half a billion
#: elements, so the pass is taken in slices rather than whole.
ACCUMULATE_CHUNK = 1 << 25


def _setup_straight_through(ctx, inputs, output):
    """Keep the op's argument count and its input's shape; only its first
    output, the fake-quantized tensor, takes a gradient."""
    ctx.num_arguments = len(inputs)
    ctx.input_shape = inputs[0].shape
    if isinstance(output, tuple):
        ctx.mark_non_differentiable(*output[1:])


def _straight_through(ctx, grad, *unused):
    """The straight-through estimator: the input takes the fake-quantized
    tensor's gradient unchanged, less the padding a rotation added, and no
    other argument takes one.

    Args:
        ctx: The context ``_setup_straight_through`` filled.
        grad: The fake-quantized tensor's gradient.
        *unused: The other outputs' gradients, which the outputs never
            take.
    """
    if grad.shape != ctx.input_shape:
        grad = grad[tuple(slice(size) for size in ctx.input_shape)]
    return grad, *(None,) * (ctx.num_arguments - 1)


def _fake_quant_reference(
    input: torch.Tensor, qmap: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Fake-quantize ``input`` against a per-tensor or per-channel scale.

    Args:
        input: Tensor to quantize.
        qmap: Lookup table, one entry per bfloat16 bit pattern.
        scale: Scale, broadcastable to ``input``.

    Returns:
        The fake-quantized tensor.
    """
    scale = scale.to(input.dtype)
    return vmap(input / scale, qmap) * scale


fake_quant = triton_op(
    "quantization::fake_quant", _fake_quant_reference, mutates_args=()
)
fake_quant.register_autograd(
    _straight_through, setup_context=_setup_straight_through
)


def _group_wise_affine_fake_quant_reference(
    input: torch.Tensor,
    axes: List[int],
    block_size: int,
    quant_min: float,
    quant_max: float,
    scale_qmap: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fake-quantize ``input`` with a scale and zero point per block.

    Each block maps its own ``[min, max]`` onto ``[quant_min, quant_max]``.

    Args:
        input: Tensor to quantize.
        axes: Axes the blocks run along.
        block_size: Elements per block.
        quant_min: Smallest code.
        quant_max: Largest code.
        scale_qmap: Lookup table the scale and zero point are quantized
            into, or None.

    Returns:
        ``(output, scale, zero_point)``: the fake-quantized tensor and the
        per-block parameters.
    """
    assert block_size > 0

    axes = [x + input.ndim if x < 0 else x for x in axes]

    reshaped_input, axes, _, _ = _reshape_to_blocks(input, axes, block_size)

    shared_exp_axes = [x + 1 for x in axes]

    min = torch.amin(reshaped_input, dim=shared_exp_axes)
    max = torch.amax(reshaped_input, dim=shared_exp_axes)

    sf = (max - min) / (quant_max - quant_min)
    sf = torch.where(sf > 0.0, sf, 1.0)
    # Quantize the scale using the codebook; a scale below its range
    # rounds to zero and the block is one level, as with no range.
    if scale_qmap is not None:
        sf = vmap(sf, scale_qmap)
        sf = torch.where(sf > 0.0, sf, 1.0)
    zp = -min / sf + quant_min
    if scale_qmap is not None:
        zp = vmap(zp, scale_qmap)

    expanded_sf = repeat_per_block(sf, input, block_size)
    expanded_zp = repeat_per_block(zp, input, block_size)

    q = torch.round(input / expanded_sf + expanded_zp)
    q = torch.clamp(q, quant_min, quant_max)
    return (q - expanded_zp) * expanded_sf, sf, zp


group_wise_affine_fake_quant = triton_op(
    "quantization::group_wise_affine_fake_quant",
    _group_wise_affine_fake_quant_reference,
    mutates_args=(),
)
group_wise_affine_fake_quant.register_autograd(
    _straight_through, setup_context=_setup_straight_through
)


def _quantize_mx_and_back(
    input,
    qmap,
    axes,
    block_size,
    quant_max,
    power_of_two,
    scale_qmap,
    seed,
    global_scale,
):
    """``input`` quantized by ``quantize_mx`` and scaled back: the part of
    ``mx_fake_quant`` between its rotations, taking its arguments."""
    scale, output = quantize_mx(
        input,
        qmap,
        axes,
        block_size,
        quant_max,
        power_of_two,
        scale_qmap=scale_qmap,
        seed=seed,
        global_scale=global_scale,
    )
    output = output * repeat_per_block(scale, output, block_size)
    if global_scale is not None:
        output = output * global_scale
    return output


def _mx_fake_quant_reference(
    input,
    qmap,
    axes,
    block_size,
    quant_max,
    power_of_two,
    scale_qmap,
    seed,
    global_scale,
    hadamard,
    hadamard_axis,
):
    """``mx_fake_quant`` in torch ops."""
    if hadamard is not None:
        input = hadamard_rotate(input, hadamard_axis, hadamard)
    output = _quantize_mx_and_back(
        input,
        qmap,
        axes,
        block_size,
        quant_max,
        power_of_two,
        scale_qmap,
        seed,
        global_scale,
    )
    if hadamard is not None:
        output = hadamard_rotate(output, hadamard_axis, hadamard.T)
    return output


def _run_mx_fake_quant(
    quantize,
    input,
    qmap,
    axes,
    block_size,
    quant_max,
    power_of_two,
    scale_qmap,
    seed,
    global_scale,
    hadamard,
    hadamard_axis,
):
    """``mx_fake_quant`` of a GPU call or of one being traced.

    The kernel computes a call it takes; a rotation it cannot fuse runs as
    torch ops around this function's own call on the rotated tensor; any
    other call runs ``quantize``.

    Args:
        quantize: Fake-quantizes a call that neither the kernel nor a
            rotation around it takes: ``_quantize_mx_and_back`` or its
            compiled form.
        input: Tensor to quantize; it and the arguments after it are the
            op's.
    """
    args = (
        input,
        qmap,
        axes,
        block_size,
        quant_max,
        power_of_two,
        scale_qmap,
        seed,
        global_scale,
    )
    if kernels is not None and kernels.mx_fake_quant_supported(
        *args, hadamard, hadamard_axis
    ):
        return kernels.mx_fake_quant(*args, hadamard, hadamard_axis)
    if hadamard is None:
        return quantize(*args)
    rotated = hadamard_rotate(input, hadamard_axis, hadamard)
    # Unrotated, the kernel may still take the rotated tensor.
    output = _run_mx_fake_quant(quantize, rotated, *args[1:], None, None)
    return hadamard_rotate(output, hadamard_axis, hadamard.T)


@triton_op("quantization::mx_fake_quant", mutates_args=())
def mx_fake_quant(
    input: torch.Tensor,
    qmap: torch.Tensor,
    axes: List[int],
    block_size: int,
    quant_max: float,
    power_of_two: bool,
    scale_qmap: Optional[torch.Tensor],
    seed: Optional[torch.Tensor],
    global_scale: Optional[torch.Tensor],
    hadamard: Optional[torch.Tensor],
    hadamard_axis: Optional[int],
) -> torch.Tensor:
    """Fake-quantize ``input`` in microscaling blocks.

    Each block's scale is its amax over ``quant_max``, or a power of two,
    divided by ``global_scale`` and quantized through ``scale_qmap`` when
    given; the elements are divided by it times ``global_scale``, rounded
    and scaled back.  With ``hadamard`` the input is first rotated along
    ``hadamard_axis`` (``hadamard_rotate``) and the result rotated back by
    the transpose, the rotated axis keeping its padding; the backward
    passes the gradient through unrotated, less the padding.  This body is
    what ``torch.compile`` traces (``_run_mx_fake_quant``).

    Args:
        input: Tensor to quantize.
        qmap: Lookup table the elements round to, or with a ``seed`` the
            ascending levels they round between stochastically.
        axes: Axes the blocks run along.
        block_size: Elements per block.
        quant_max: Largest magnitude ``qmap`` represents.
        power_of_two: Round each block scale to a power of two.
        scale_qmap: Lookup table the scales are quantized into, or None.
        seed: int64 scalar keying stochastic rounding, or None to round to
            the nearest value.
        global_scale: The tensor's global scale -- with ``hadamard``, the
            rotated tensor's -- as a 0-d tensor, or None.
        hadamard: The ``HADAMARD_SIZE`` x ``HADAMARD_SIZE`` matrix that
            rotates ``input``, or None.
        hadamard_axis: The axis ``hadamard`` rotates along.

    Returns:
        The fake-quantized tensor, laid out as ``input`` unless rotated, or
        when the kernel computes it with its rotation.
    """
    return _run_mx_fake_quant(
        _quantize_mx_and_back,
        input,
        qmap,
        axes,
        block_size,
        quant_max,
        power_of_two,
        scale_qmap,
        seed,
        global_scale,
        hadamard,
        hadamard_axis,
    )


mx_fake_quant.register_kernel("cpu")(_mx_fake_quant_reference)
mx_fake_quant.register_autograd(
    _straight_through, setup_context=_setup_straight_through
)


def _hadamard_amax_reference(
    input: torch.Tensor, axis: int, hadamard: torch.Tensor
) -> torch.Tensor:
    """``hadamard_amax`` in torch ops."""
    return torch.amax(torch.abs(hadamard_rotate(input, axis, hadamard)))


@triton_op("quantization::hadamard_amax", mutates_args=())
def hadamard_amax(
    input: torch.Tensor, axis: int, hadamard: torch.Tensor
) -> torch.Tensor:
    """The largest magnitude of ``hadamard_rotate(input, axis, hadamard)``,
    the global amax of a rotated ``mx_fake_quant``.  It has no backward.

    Args:
        input: Tensor to rotate.
        axis: Axis to rotate along.
        hadamard: The ``HADAMARD_SIZE`` x ``HADAMARD_SIZE`` matrix.

    Returns:
        A 0-d tensor in ``input``'s dtype.
    """
    if kernels is not None and kernels.hadamard_amax_supported(input, axis):
        return kernels.hadamard_amax(input, axis, hadamard)
    return _hadamard_amax_reference(input, axis, hadamard)


def _bin_blocks_reference(
    tensor: torch.Tensor,
    axis: int,
    block_size: int,
    quant_max: float,
    energy: Optional[torch.Tensor],
    edges: torch.Tensor,
    sums: List[torch.Tensor],
    counts: List[int],
    power_of_two: bool,
    scale_qmap: Optional[torch.Tensor],
    global_scale: Optional[torch.Tensor],
) -> None:
    """Add ``tensor``'s values, normalized by their blocks, to the
    histogram of each cell of a grid.

    The grid splits axis ``i`` into ``counts[i]`` equal runs, its cells in
    row-major order.  A block's scale is the one ``calculate_mx_qparam``
    takes over its cell, the last block of each run along ``axis`` padded
    short.  Each element weighs its cell's energy at its position in the
    run times its block's squared scale.  A value's bin is the number of
    interior edges at or below it, at most the last bin, and the bin adds
    the element's weight, weighted value and weighted square in float64.

    Args:
        tensor: The operand.
        axis: The axis its microscaling blocks run along.
        block_size: Elements per block, a power of two.
        quant_max: The value each block's largest magnitude normalizes to.
        energy: Each cell's weight per position along ``axis`` in its run,
            ``[cells, run]``, or None to weight every element by its
            block's squared scale alone.
        edges: The interior bin edges as float32, padded with +inf to a
            power of two above their count.
        sums: Each cell's ``[3, bins]`` float64 sums the terms are added
            to, in row-major order.
        counts: Runs along each axis.
        power_of_two: Floor each block scale to a power of two.
        scale_qmap: The 65536-entry table the block scales are quantized
            through, or None.
        global_scale: The tensor's global scale, a 0-d tensor, or None.
    """
    cells = [tensor]
    for dim, count in enumerate(counts):
        cells = [piece for cell in cells for piece in cell.chunk(count, dim)]
    if energy is None:
        energy = [None] * len(cells)
    edges = edges.double()
    bins = sums[0].shape[1]
    for cell, cell_energy, cell_sums in zip(cells, energy, sums):
        ordered = cell.transpose(axis, -1).float()
        # One scale serves both the normalization and the squared-scale
        # weight, so the two halves of the objective refer to the same
        # number -- the one deployment computes: its blocks pad out along
        # the contraction axis rather than running past it.
        scale = calculate_mx_qparam(
            ordered,
            axes=-1,
            block_size=block_size,
            quant_max=quant_max,
            force_scale_power_of_two=power_of_two,
            scale_qmap=scale_qmap,
            global_scale=global_scale,
        )
        if global_scale is not None:
            scale = scale * global_scale
        scale = repeat_per_block(scale, ordered, block_size)
        values = (ordered / scale).reshape(-1)
        if cell_energy is None:
            weights = torch.ones_like(values)
        else:
            # The contraction axis is last, so an element's channel is its
            # position modulo the channel count -- which is what tiling the
            # energy across the flattened tensor spells.
            weights = cell_energy.repeat(values.numel() // cell_energy.numel())
        weights = weights * (scale**2).reshape(-1)
        for start in range(0, values.numel(), ACCUMULATE_CHUNK):
            # Float64, so that the order the bins are summed in -- which on
            # a GPU the scheduler decides -- cannot move a level.  That
            # indifference to order is also what lets the pass be sliced.
            value = values[start : start + ACCUMULATE_CHUNK].double()
            weight = weights[start : start + ACCUMULATE_CHUNK].double()
            index = torch.bucketize(value, edges, right=True)
            index = index.clamp_(max=bins - 1)
            cell_sums[0].index_add_(0, index, weight)
            cell_sums[1].index_add_(0, index, weight * value)
            cell_sums[2].index_add_(0, index, weight * value * value)


bin_blocks = torch.library.custom_op(
    "quantization::bin_blocks", _bin_blocks_reference, mutates_args=("sums",)
)


@bin_blocks.register_fake
def _(*args) -> None:
    return None


# On a GPU a fake-quant's reference, or the part of ``mx_fake_quant`` its
# kernel does not take, runs compiled, as eager GPU code runs; a rotation
# around it stays eager matmuls.  ``bin_blocks`` runs its kernel, and
# ``hadamard_amax``'s body serves a GPU call as it is.
if kernels is not None:
    fake_quant.register_kernel("cuda")(
        compile_like_eager(_fake_quant_reference, dynamic=False)
    )
    group_wise_affine_fake_quant.register_kernel("cuda")(
        compile_like_eager(
            _group_wise_affine_fake_quant_reference, dynamic=False
        )
    )
    _compiled_quantize_mx_and_back = compile_like_eager(
        _quantize_mx_and_back, dynamic=False
    )

    @mx_fake_quant.register_kernel("cuda")
    def _(*args):
        """``mx_fake_quant`` of CUDA tensors, its kernel launched directly
        as a ``triton_op``'s own eager kernel launches it.  The tensors are
        detached, so that whether they take a gradient never enters the
        compiled fallback's guards."""
        args = [
            arg.detach() if isinstance(arg, torch.Tensor) else arg
            for arg in args
        ]
        with set_wrap_triton_enabled(False):
            return _run_mx_fake_quant(_compiled_quantize_mx_and_back, *args)

    bin_blocks.register_kernel("cuda")(kernels.bin_blocks)

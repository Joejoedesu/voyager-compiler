"""Real NVFP4 GEMMs for the linears of a converted graph.

``convert_pt2e`` leaves an NVFP4 linear -- ``fp4_e2m1`` elements in blocks of
16 along K, each block with an ``fp8_e4m3`` scale, under one global scale per
tensor -- as::

    t = clamp_min(amax(abs(x)) / (6 * 448), tiny)
    s, q = quantize_mx(x, ..., global_scale=t)
    linear_mx(q, W_q, b, input_scale=s, weight_scale=S_w, alpha=t_w * t)

with the codes and block scales held as bfloat16 values on their grids, and
``W_q``, ``S_w`` and ``t_w`` constants.  ``quantize_nvfp4`` repeats the
fake-quant's bfloat16 arithmetic, so it picks the same codes, and packs them
as torch's block-scaled GEMM reads them; ``nvfp4_linear`` is that GEMM.

The GEMM reads one scale per 16 elements of a row.  A weight quantized in
16x16 blocks keeps its codes; ``pack_nvfp4`` gives each of a block's 16 rows
the block's scale.  ``quantize_nvfp4_square`` casts a weight in 16x16 blocks
itself, packed for its GEMM and, transposed, for its input gradient's.
"""

import operator
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.fx import GraphModule, Node
from torch.library import Library, impl
from torch.nn.functional import ScalingType, SwizzleType
from torchao.prototype.mx_formats.kernels import (
    f4_unpacked_to_f32,
    f32_to_f4_unpacked,
    pack_uint4,
    unpack_uint4,
)
from torchao.prototype.mx_formats.utils import from_blocked, to_blocked

from voyager_compiler.codegen.node_info import get_arg_value
from voyager_compiler.ops import hadamard_rotate
from voyager_compiler.ops.quantized import (
    PHILOX_ROUNDS,
    _random_bytes,
)
from voyager_compiler.quantization.fake_quantize import (
    calculate_global_scale,
)
from voyager_compiler.shape_prop import fetch_attr

from .fp8 import GEMM_ALIGNMENT

QUANTIZE_MX = torch.ops.quantized_ops.quantize_mx.default
LINEAR_MX = torch.ops.quantized_ops.linear_mx.default

# NVFP4's block length and the largest fp4_e2m1 and fp8_e4m3 values.
BLOCK_SIZE = 16
FP4_MAX = 6.0
E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
# The ``meta["dtype"]`` of an NVFP4 ``quantize_mx``: its scales', its codes'.
NVFP4_DTYPES = ("fp8_e4m3", "fp4_e2m1")

# The compute capability majors torch runs block-scaled GEMMs on, and the
# CUDA version they need.
BLOCK_SCALED_MAJORS = (10, 12)
BLOCK_SCALED_CUDA = (12, 8)

# The buffer of a lowered graph holding the 1 its ``nvfp4_linear`` calls
# take as ``one``.
ONE = "nvfp4_one"

# The rows and the runs of 16 along K one program of the CUDA cast takes.
CAST_TILE = 64
CAST_RUNS = 4
# The runs of 16 along each axis one program of the CUDA pair cast takes.
PAIR_CAST_RUNS = 4
_LIBRARY = Library("voyager_gpu", "DEF")
# The casts' ops read their input's strides at run time, so Inductor may
# pass any layout: it writes a producer once and hands each cast a view.
CAST_TAGS = (torch.Tag.flexible_layout,)
# FP4_MAX and E4M3_MAX as the cast's kernels read them.
_FP4_MAX = tl.constexpr(FP4_MAX)
_E4M3_MAX = tl.constexpr(E4M3_MAX)
# The least global scale, as ``calculate_global_scale`` clamps it.
_FLOAT32_TINY = tl.constexpr(torch.finfo(torch.float32).tiny)
# float32's sign bit, as an int32.
_SIGN_BIT = tl.constexpr(-(2**31))
# The rounds of _random_bytes' Philox4x32, which tl.philox runs in the casts.
_PHILOX_ROUNDS = tl.constexpr(PHILOX_ROUNDS)


@dataclass
class Site:
    """One NVFP4 linear of a converted graph, in ``fp8.Site``'s terms.

    Attributes:
        gemm: The ``linear_mx``.
        output: ``gemm`` as well: it applies every scale itself.
        quantizes: The ``quantize_mx`` of its activation, alone.  Other
            sites may share it.
        kernel: The real GEMM, ``nvfp4_linear``.
        weight: The weight's codes, ``[N, K]`` (``get_attr``).
        weight_scale: Their block scales (``get_attr``): ``[N, K / 16]``, or
            ``[N / 16, K / 16]`` for 16x16 blocks.
        alpha: The product of the weight's and the activation's global
            scales.
        bias: The bias (``get_attr``), or None.
    """

    gemm: Node
    output: Node
    quantizes: Tuple[Node]
    kernel: Callable
    weight: Node
    weight_scale: Node
    alpha: Node
    bias: Optional[Node]


def match_site(model: GraphModule, linear: Node) -> Union[Site, str]:
    """The NVFP4 site ``linear`` heads, or why it is not one.

    Args:
        model: The converted graph.
        linear: A ``linear_mx`` of it.

    Returns:
        The ``Site``, or a short reason it cannot run as a real NVFP4 GEMM.
    """
    codes, weight = linear.args[:2]
    quantize = codes.args[0] if codes.target is operator.getitem else codes
    if quantize.target is not QUANTIZE_MX:
        return "input is not quantized"
    if quantize.meta.get("dtype") != NVFP4_DTYPES:
        return f"input is {quantize.meta.get('dtype')}, not NVFP4"
    if quantize.args[2:5] != ((-1,), BLOCK_SIZE, FP4_MAX):
        return "input is not in NVFP4's blocks of 16 along K"
    if get_arg_value(quantize, 9, "global_scale") is None:
        return "input has no global scale"
    value = quantize.args[0].meta.get("val")
    if value is None or value.dtype is not torch.bfloat16:
        return "input is not bfloat16"
    weight_scale = linear.kwargs["weight_scale"]
    if (
        weight.op != "get_attr"
        or weight.meta.get("dtype") != NVFP4_DTYPES[1]
        or weight_scale.meta.get("dtype") != NVFP4_DTYPES[0]
    ):
        return "weight is not a constant in NVFP4"
    n, k = fetch_attr(model, weight.target).shape
    # The GEMM counts K in bytes of two codes.
    if k // 2 % GEMM_ALIGNMENT or n % GEMM_ALIGNMENT:
        return f"weight ({n}, {k}): N is not a multiple of 16 or K of 32"
    bias = get_arg_value(linear, 2, "bias")
    if bias is not None and bias.op != "get_attr":
        return "bias is not a constant"
    return Site(
        linear,
        linear,
        (quantize,),
        NVFP4_LINEAR,
        weight,
        weight_scale,
        linear.kwargs["alpha"],
        bias,
    )


def _block_scales(
    amax: torch.Tensor, global_scale: torch.Tensor
) -> torch.Tensor:
    """Each block's scale from its amax, as ``quantize_mx`` computes it: the
    amax over 6 and over the global scale, rounded to ``fp8_e4m3`` in the
    amax's dtype, and 1 where that is 0."""
    scale = amax / FP4_MAX / global_scale
    scale = scale.clamp(max=E4M3_MAX).to(torch.float8_e4m3fn).to(amax.dtype)
    return torch.where(scale > 0, scale, 1.0)


@triton.jit
def _runs(
    x,
    h,
    m,
    k0,
    rows,
    stride_m,
    stride_k,
    ROTATE: tl.constexpr,
    TILE: tl.constexpr,
    RUNS: tl.constexpr,
):
    """A ``[TILE, RUNS, 16]`` tile of the cast's matrix: rows ``m``, ``RUNS``
    runs of 16 along K from ``k0``, each rotated by the 16x16 matrix at ``h``
    if ``ROTATE`` and rounded to bfloat16 as ``hadamard_rotate`` rounds it;
    values as float32."""
    k = k0 + tl.arange(0, RUNS * 16)
    mask = m[:, None] < rows
    tile = tl.load(
        x + m[:, None] * stride_m + k[None, :] * stride_k,
        mask=mask,
        other=0.0,
    )
    tile = tl.reshape(tile, (TILE * RUNS, 16))
    if ROTATE:
        run = tl.arange(0, 16)
        # +-1/4: exact in bfloat16.
        hm = tl.load(h + run[:, None] * 16 + run[None, :]).to(tl.bfloat16)
        tile = tl.dot(tile, hm).to(tl.bfloat16)
    return tl.reshape(tile.to(tl.float32), (TILE, RUNS, 16))


@triton.jit
def _amax_kernel(
    x,
    h,
    amax,
    rows,
    stride_m,
    stride_k,
    ROTATE: tl.constexpr,
    TILE: tl.constexpr,
    RUNS: tl.constexpr,
):
    """The cast's first pass: the largest magnitude, into ``amax``."""
    m = tl.program_id(0) * TILE + tl.arange(0, TILE)
    tile = _runs(
        x,
        h,
        m,
        tl.program_id(1) * RUNS * 16,
        rows,
        stride_m,
        stride_k,
        ROTATE,
        TILE,
        RUNS,
    )
    tl.atomic_max(amax, tl.max(tl.max(tl.max(tl.abs(tile), 2), 1), 0))


@triton.jit
def _fp4_code(value):
    """The fp4_e2m1 code of float32 ``value``, rounded to nearest even and
    saturating at 6, its sign in bit 3, negative zero's included."""
    m = tl.abs(value)
    code = (
        (m > 0.25).to(tl.int32)
        + (m >= 0.75).to(tl.int32)
        + (m > 1.25).to(tl.int32)
        + (m >= 1.75).to(tl.int32)
        + (m > 2.5).to(tl.int32)
        + (m >= 3.5).to(tl.int32)
        + (m > 5.0).to(tl.int32)
    )
    sign = (value.to(tl.int32, bitcast=True) < 0).to(tl.int32)
    return code | (sign << 3)


@triton.jit
def _cast_tile(
    tile,
    run_amax,
    m,
    first,
    rows,
    cols,
    amax,
    seed,
    codes,
    scales,
    global_scale,
    flat_m,
    flat_k,
    scale_col_tiles,
    STOCHASTIC: tl.constexpr,
    K_FLAT: tl.constexpr,
    TILE: tl.constexpr,
    RUNS: tl.constexpr,
):
    """Cast a ``[TILE, RUNS, 16]`` float32 tile -- rows ``m``, runs of 16
    from run ``first``, each scaled by the ``[TILE, RUNS]`` ``run_amax`` --
    with ``quantize_nvfp4``'s bfloat16 arithmetic op for op: each run's
    scale, its elements rounded to nearest or, if ``STOCHASTIC``, by
    ``_random_bytes``' Philox bits, packed two to a byte, and the scales
    stored in cuBLAS's swizzled layout.  The grid's first program stores
    the global scale ``t`` to ``global_scale``."""
    mask = m[:, None] < rows
    run = tl.arange(0, 16)
    # Eager torch divides by a Python number as a product with its
    # reciprocal, and by a tensor exactly.
    t = (tl.load(amax) * (1.0 / (_FP4_MAX * _E4M3_MAX))).to(tl.bfloat16)
    t = tl.maximum(t.to(tl.float32), _FLOAT32_TINY)
    if (tl.program_id(0) == 0) & (tl.program_id(1) == 0):
        tl.store(global_scale, t.to(tl.bfloat16))
    s = (run_amax * (1.0 / _FP4_MAX)).to(tl.bfloat16)
    s = tl.math.div_rn(s.to(tl.float32), t).to(tl.bfloat16).to(tl.float32)
    s = tl.minimum(s, _E4M3_MAX)
    s = s.to(tl.float8e4nv, fp_downcast_rounding="rtne").to(tl.float32)
    s = tl.where(s > 0, s, 1.0)
    denominator = (s * t).to(tl.bfloat16).to(tl.float32)
    elements = tl.math.div_rn(tile, denominator[:, :, None])
    elements = elements.to(tl.bfloat16).to(tl.float32)

    if STOCHASTIC:
        key = tl.load(seed)
        block = first + tl.arange(0, RUNS)
        if K_FLAT:
            # K runs along the flat index, so a run is one Philox draw.
            row_start = m[:, None].to(tl.int64) * flat_m
            index = row_start + (block * 16)[None, :].to(tl.int64)
        else:
            k = block[:, None] * 16 + run[None, :]
            index = (
                m[:, None, None].to(tl.int64) * flat_m
                + k[None, :, :].to(tl.int64) * flat_k
            )
        counter = index // 16
        c0 = (counter & 0xFFFFFFFF).to(tl.uint32)
        c1 = (counter >> 32).to(tl.uint32)
        zero = tl.zeros_like(c0)
        w0, w1, w2, w3 = tl.philox(key, c0, c1, zero, zero, _PHILOX_ROUNDS)
        if K_FLAT:
            which = run[None, None, :] // 4
            shift = ((run[None, None, :] % 4) * 8).to(tl.uint32)
            w0, w1 = w0[:, :, None], w1[:, :, None]
            w2, w3 = w2[:, :, None], w3[:, :, None]
        else:
            which = (index % 16) // 4
            shift = ((index % 4) * 8).to(tl.uint32)
        word = tl.where(
            which == 0,
            w0,
            tl.where(which == 1, w1, tl.where(which == 2, w2, w3)),
        )
        byte = ((word >> shift) & 0xFF).to(tl.float32)
        value = tl.minimum(tl.maximum(elements, -_FP4_MAX), _FP4_MAX)
        magnitude = tl.abs(value)
        gap = tl.where(magnitude < 2, 0.5, tl.where(magnitude < 4, 1.0, 2.0))
        near = tl.floor(magnitude / gap) * gap
        far = tl.where(near == magnitude, near, near + gap)
        rounded = tl.where(
            magnitude + byte / 256 * (far - near) >= far, far, near
        )
        sign = value.to(tl.int32, bitcast=True) & _SIGN_BIT
        elements = (rounded.to(tl.int32, bitcast=True) | sign).to(
            tl.float32, bitcast=True
        )

    low, high = tl.split(tl.reshape(_fp4_code(elements), (TILE, RUNS * 8, 2)))
    pair = first * 8 + tl.arange(0, RUNS * 8)
    tl.store(
        codes + m[:, None] * (cols // 2) + pair[None, :],
        (low | (high << 4)).to(tl.uint8),
        mask=mask,
    )

    # ``to_blocked``: 128 x 4 tiles of (row, run), each 32 rows of 4 row
    # groups x 4 runs.
    block = first + tl.arange(0, RUNS)
    tiles = (m[:, None] // 128) * scale_col_tiles + block[None, :] // 4
    within = (
        (m[:, None] % 32) * 16
        + ((m[:, None] % 128) // 32) * 4
        + block[None, :] % 4
    )
    tl.store(
        scales + tiles * 512 + within,
        s.to(tl.float8e4nv).to(tl.uint8, bitcast=True),
        mask=mask,
    )


@triton.jit
def _cast_kernel(
    x,
    h,
    amax,
    seed,
    codes,
    scales,
    global_scale,
    rows,
    cols,
    stride_m,
    stride_k,
    flat_m,
    flat_k,
    scale_col_tiles,
    ROTATE: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    K_FLAT: tl.constexpr,
    TILE: tl.constexpr,
    RUNS: tl.constexpr,
):
    """The cast's second pass: ``_cast_tile`` of each tile of ``_runs``."""
    m = tl.program_id(0) * TILE + tl.arange(0, TILE)
    first = tl.program_id(1) * RUNS
    tile = _runs(
        x, h, m, first * 16, rows, stride_m, stride_k, ROTATE, TILE, RUNS
    )
    _cast_tile(
        tile,
        tl.max(tl.abs(tile), 2),
        m,
        first,
        rows,
        cols,
        amax,
        seed,
        codes,
        scales,
        global_scale,
        flat_m,
        flat_k,
        scale_col_tiles,
        STOCHASTIC,
        K_FLAT,
        TILE,
        RUNS,
    )


@triton.jit
def _pair_tile(x, cols, RUNS_M: tl.constexpr, RUNS_K: tl.constexpr):
    """A ``[16 RUNS_M, 16 RUNS_K]`` bfloat16 tile of a contiguous matrix and
    its row and column indices."""
    m = tl.program_id(0) * RUNS_M * 16 + tl.arange(0, RUNS_M * 16)
    k = tl.program_id(1) * RUNS_K * 16 + tl.arange(0, RUNS_K * 16)
    return tl.load(x + m[:, None] * cols + k[None, :]), m, k


@triton.jit
def _transpose_runs(
    tile,
    h,
    ROTATE: tl.constexpr,
    RUNS_M: tl.constexpr,
    RUNS_K: tl.constexpr,
):
    """``_pair_tile``'s transpose as a ``[16 RUNS_K, RUNS_M, 16]`` float32
    tile of runs of 16, each rotated by ``h`` if ``ROTATE`` and rounded to
    bfloat16 as ``hadamard_rotate`` rounds it."""
    transpose = tl.reshape(tl.trans(tile), (RUNS_K * RUNS_M * 16, 16))
    if ROTATE:
        run = tl.arange(0, 16)
        hm = tl.load(h + run[:, None] * 16 + run[None, :]).to(tl.bfloat16)
        transpose = tl.dot(transpose, hm).to(tl.bfloat16)
    return tl.reshape(transpose.to(tl.float32), (RUNS_K * 16, RUNS_M, 16))


@triton.jit
def _pair_amax_kernel(
    x,
    h,
    amax,
    transpose_amax,
    cols,
    ROTATE: tl.constexpr,
    RUNS_M: tl.constexpr,
    RUNS_K: tl.constexpr,
):
    """The pair cast's first pass: each layout's largest magnitude."""
    tile, _, _ = _pair_tile(x, cols, RUNS_M, RUNS_K)
    tl.atomic_max(amax, tl.max(tl.max(tl.abs(tile.to(tl.float32)), 1), 0))
    transpose = _transpose_runs(tile, h, ROTATE, RUNS_M, RUNS_K)
    tl.atomic_max(
        transpose_amax, tl.max(tl.max(tl.max(tl.abs(transpose), 2), 1), 0)
    )


@triton.jit
def _pair_cast_kernel(
    x,
    h,
    amax,
    transpose_amax,
    seed,
    transpose_seed,
    codes,
    scales,
    global_scale,
    transpose_codes,
    transpose_scales,
    transpose_global_scale,
    rows,
    cols,
    ROTATE: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    TRANSPOSE_STOCHASTIC: tl.constexpr,
    RUNS_M: tl.constexpr,
    RUNS_K: tl.constexpr,
):
    """The pair cast's second pass: ``_cast_tile`` of a
    ``[16 RUNS_M, 16 RUNS_K]`` tile's rows or, in the grid's second half,
    of its transpose's."""
    m = tl.program_id(0) * RUNS_M * 16 + tl.arange(0, RUNS_M * 16)
    k = tl.program_id(1) * RUNS_K * 16 + tl.arange(0, RUNS_K * 16)
    if tl.program_id(2) == 0:
        tile = _runs(
            x,
            h,
            m,
            k0=tl.program_id(1) * RUNS_K * 16,
            rows=rows,
            stride_m=cols,
            stride_k=1,
            ROTATE=False,
            TILE=RUNS_M * 16,
            RUNS=RUNS_K,
        )
        _cast_tile(
            tile,
            tl.max(tl.abs(tile), 2),
            m,
            first=tl.program_id(1) * RUNS_K,
            rows=rows,
            cols=cols,
            amax=amax,
            seed=seed,
            codes=codes,
            scales=scales,
            global_scale=global_scale,
            flat_m=cols,
            flat_k=1,
            scale_col_tiles=tl.cdiv(cols // 16, 4),
            STOCHASTIC=STOCHASTIC,
            K_FLAT=True,
            TILE=RUNS_M * 16,
            RUNS=RUNS_K,
        )
    else:
        # Named apart from the rows' tile: the two differ in shape unless
        # RUNS_M equals RUNS_K.
        transpose = _runs(
            x,
            h,
            k,
            k0=tl.program_id(0) * RUNS_M * 16,
            rows=cols,
            stride_m=1,
            stride_k=cols,
            ROTATE=ROTATE,
            TILE=RUNS_K * 16,
            RUNS=RUNS_M,
        )
        _cast_tile(
            transpose,
            tl.max(tl.abs(transpose), 2),
            k,
            first=tl.program_id(0) * RUNS_M,
            rows=cols,
            cols=rows,
            amax=transpose_amax,
            seed=transpose_seed,
            codes=transpose_codes,
            scales=transpose_scales,
            global_scale=transpose_global_scale,
            flat_m=rows,
            flat_k=1,
            scale_col_tiles=tl.cdiv(rows // 16, 4),
            STOCHASTIC=TRANSPOSE_STOCHASTIC,
            K_FLAT=True,
            TILE=RUNS_K * 16,
            RUNS=RUNS_M,
        )


@triton.jit
def _square_cast_kernel(
    x,
    amax,
    codes,
    scales,
    transpose_codes,
    transpose_scales,
    global_scale,
    rows,
    cols,
    RUNS_M: tl.constexpr,
    RUNS_K: tl.constexpr,
):
    """The square cast's second pass: ``_cast_tile`` of a
    ``[16 RUNS_M, 16 RUNS_K]`` tile's rows and of its transpose's, each run
    scaled by the amax of the 16x16 block it lies in.  Both layouts store
    the same global scale to ``global_scale``."""
    tile, m, k = _pair_tile(x, cols, RUNS_M, RUNS_K)
    tile = tile.to(tl.float32)
    blocks = tl.reshape(tl.abs(tile), (RUNS_M, 16, RUNS_K, 16))
    block_amax = tl.max(tl.max(blocks, 3), 1)
    run_amax = tl.broadcast_to(block_amax[:, None, :], (RUNS_M, 16, RUNS_K))
    _cast_tile(
        tl.reshape(tile, (RUNS_M * 16, RUNS_K, 16)),
        tl.reshape(run_amax, (RUNS_M * 16, RUNS_K)),
        m,
        first=tl.program_id(1) * RUNS_K,
        rows=rows,
        cols=cols,
        amax=amax,
        seed=amax,
        codes=codes,
        scales=scales,
        global_scale=global_scale,
        flat_m=cols,
        flat_k=1,
        scale_col_tiles=tl.cdiv(cols // 16, 4),
        STOCHASTIC=False,
        K_FLAT=True,
        TILE=RUNS_M * 16,
        RUNS=RUNS_K,
    )
    transpose_amax = tl.broadcast_to(
        tl.trans(block_amax)[:, None, :], (RUNS_K, 16, RUNS_M)
    )
    _cast_tile(
        tl.reshape(tl.trans(tile), (RUNS_K * 16, RUNS_M, 16)),
        tl.reshape(transpose_amax, (RUNS_K * 16, RUNS_M)),
        k,
        first=tl.program_id(0) * RUNS_M,
        rows=cols,
        cols=rows,
        amax=amax,
        seed=amax,
        codes=transpose_codes,
        scales=transpose_scales,
        global_scale=global_scale,
        flat_m=rows,
        flat_k=1,
        scale_col_tiles=tl.cdiv(rows // 16, 4),
        STOCHASTIC=False,
        K_FLAT=True,
        TILE=RUNS_K * 16,
        RUNS=RUNS_M,
    )


def _runs_dividing(length: int, most: int) -> int:
    """The most runs of 16, up to ``most`` and halving from it, that divide
    a ``length`` long axis."""
    runs = most
    while length // BLOCK_SIZE % runs:
        runs //= 2
    return runs


def _cast_outputs(
    rows: int, cols: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A cast's packed codes, ``[rows, cols / 2]``, its block scales,
    zeroed and padded to ``pack_nvfp4``'s swizzled tiles, and its 0-d
    bfloat16 global scale."""
    codes = torch.empty((rows, cols // 2), dtype=torch.uint8, device=device)
    scales = torch.zeros(
        triton.cdiv(rows, 128) * 128 * triton.cdiv(cols // BLOCK_SIZE, 4) * 4,
        dtype=torch.uint8,
        device=device,
    )
    global_scale = torch.empty((), dtype=torch.bfloat16, device=device)
    return codes, scales, global_scale


def _paired_castable(x: torch.Tensor) -> bool:
    """Whether the paired Triton casts take ``x``: a contiguous CUDA
    bfloat16 matrix whose sides are multiples of 16."""
    return (
        x.is_cuda
        and x.dtype is torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and x.shape[0] % BLOCK_SIZE == x.shape[1] % BLOCK_SIZE == 0
    )


_LIBRARY.define(
    "cast_nvfp4(Tensor x, Tensor? hadamard, Tensor? seed, int axis) "
    "-> (Tensor, Tensor, Tensor)",
    tags=CAST_TAGS,
)


@impl(_LIBRARY, "cast_nvfp4", "CUDA")
def cast_nvfp4(
    x: torch.Tensor,
    hadamard: Optional[torch.Tensor],
    seed: Optional[torch.Tensor],
    axis: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``quantize_nvfp4`` of a CUDA bfloat16 matrix as two Triton kernels:
    the amax, then the cast.

    Args:
        x: The matrix.
        hadamard: ``quantize_nvfp4``'s.
        seed: ``quantize_nvfp4``'s.
        axis: The axis the blocks run along, 0 or 1.

    Returns:
        ``quantize_nvfp4``'s outputs.
    """
    if axis == 0:
        cols, rows = x.shape
        stride_k, stride_m = x.stride()
        flat_m, flat_k = 1, x.shape[1]
    else:
        rows, cols = x.shape
        stride_m, stride_k = x.stride()
        flat_m, flat_k = x.shape[1], 1
    runs = _runs_dividing(cols, CAST_RUNS)
    grid = (triton.cdiv(rows, CAST_TILE), cols // BLOCK_SIZE // runs)
    amax = torch.zeros((), dtype=torch.float32, device=x.device)
    rotate = hadamard is not None
    matrix = hadamard if rotate else x
    _amax_kernel[grid](
        x, matrix, amax, rows, stride_m, stride_k, rotate, CAST_TILE, runs
    )
    codes, scales, global_scale = _cast_outputs(rows, cols, x.device)
    _cast_kernel[grid](
        x,
        matrix,
        amax,
        amax if seed is None else seed,
        codes,
        scales,
        global_scale,
        rows,
        cols,
        stride_m,
        stride_k,
        flat_m,
        flat_k,
        triton.cdiv(cols // BLOCK_SIZE, 4),
        rotate,
        seed is not None,
        flat_k == 1,
        CAST_TILE,
        runs,
    )
    return codes, scales, global_scale


@torch.library.register_fake("voyager_gpu::cast_nvfp4")
def _(x, hadamard, seed, axis):
    rows, cols = x.shape if axis == 1 else x.shape[::-1]
    return _cast_outputs(rows, cols, x.device)


def _pair_cast_layout(
    x: torch.Tensor,
) -> Tuple[List[int], Tuple[int, int], List[torch.Tensor]]:
    """How a cast of a matrix's rows and of its transpose's rows splits it,
    and the outputs it writes.

    Args:
        x: CUDA matrix ``[M, K]``, both multiples of 16.

    Returns:
        ``(runs, grid, outputs)``: the runs of 16 along M and along K one
        program takes, at most ``PAIR_CAST_RUNS`` and dividing each length;
        the grid of programs over M and K; and ``_cast_outputs`` for the
        rows, then for the transpose's rows.
    """
    rows, cols = x.shape
    runs = [_runs_dividing(length, PAIR_CAST_RUNS) for length in (rows, cols)]
    grid = (rows // BLOCK_SIZE // runs[0], cols // BLOCK_SIZE // runs[1])
    outputs = [
        *_cast_outputs(rows, cols, x.device),
        *_cast_outputs(cols, rows, x.device),
    ]
    return runs, grid, outputs


_LIBRARY.define(
    "cast_nvfp4_pair(Tensor x, Tensor? hadamard, Tensor? seed, "
    "Tensor? transpose_seed) -> (Tensor, Tensor, Tensor, Tensor, Tensor, "
    "Tensor)",
    tags=CAST_TAGS,
)


@impl(_LIBRARY, "cast_nvfp4_pair", "CUDA")
def cast_nvfp4_pair(
    x: torch.Tensor,
    hadamard: Optional[torch.Tensor],
    seed: Optional[torch.Tensor],
    transpose_seed: Optional[torch.Tensor],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """``cast_nvfp4`` of a CUDA bfloat16 matrix along its columns and of
    its transpose along its columns, as two Triton kernels over the matrix
    made contiguous: both amaxes from one read, then both casts.

    Args:
        x: The matrix.
        hadamard: ``quantize_nvfp4``'s, for the transpose.
        seed: The rows' ``quantize_nvfp4`` seed.
        transpose_seed: The transpose's.

    Returns:
        ``quantize_nvfp4_pair``'s outputs.
    """
    x = x.contiguous()
    rows, cols = x.shape
    runs, grid, outputs = _pair_cast_layout(x)
    amax = torch.zeros(2, dtype=torch.float32, device=x.device)
    rotate = hadamard is not None
    matrix = hadamard if rotate else x
    _pair_amax_kernel[grid](x, matrix, amax, amax[1:], cols, rotate, *runs)
    _pair_cast_kernel[(*grid, 2)](
        x,
        matrix,
        amax,
        amax[1:],
        amax if seed is None else seed,
        amax if transpose_seed is None else transpose_seed,
        *outputs,
        rows,
        cols,
        rotate,
        seed is not None,
        transpose_seed is not None,
        *runs,
    )
    return tuple(outputs)


@torch.library.register_fake("voyager_gpu::cast_nvfp4_pair")
def _(x, hadamard, seed, transpose_seed):
    _, _, outputs = _pair_cast_layout(x)
    return tuple(outputs)


def _square_cast_layout(
    x: torch.Tensor,
) -> Tuple[List[int], Tuple[int, int], List[torch.Tensor]]:
    """``_pair_cast_layout`` for ``cast_nvfp4_square``, its outputs in
    ``quantize_nvfp4_square``'s order: one global scale, last, for both
    layouts."""
    runs, grid, outputs = _pair_cast_layout(x)
    codes, scales, global_scale, transpose_codes, transpose_scales, _ = outputs
    outputs = [codes, scales, transpose_codes, transpose_scales, global_scale]
    return runs, grid, outputs


_LIBRARY.define(
    "cast_nvfp4_square(Tensor x) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
    tags=CAST_TAGS,
)


@impl(_LIBRARY, "cast_nvfp4_square", "CUDA")
def cast_nvfp4_square(x: torch.Tensor) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """``quantize_nvfp4_square`` of a CUDA bfloat16 matrix as two Triton
    kernels over the matrix made contiguous: the amax, then both layouts'
    casts from one read.

    Returns:
        ``quantize_nvfp4_square``'s outputs.
    """
    x = x.contiguous()
    rows, cols = x.shape
    runs, grid, outputs = _square_cast_layout(x)
    amax = torch.zeros((), dtype=torch.float32, device=x.device)
    # ``x`` stands in for the Hadamard matrix, unread without ROTATE.
    _amax_kernel[grid](
        x, x, amax, rows, cols, 1, False, runs[0] * BLOCK_SIZE, runs[1]
    )
    _square_cast_kernel[grid](x, amax, *outputs, rows, cols, *runs)
    return tuple(outputs)


@torch.library.register_fake("voyager_gpu::cast_nvfp4_square")
def _(x):
    _, _, outputs = _square_cast_layout(x)
    return tuple(outputs)


def quantize_nvfp4(
    input: torch.Tensor,
    axis: int = -1,
    seed: Optional[torch.Tensor] = None,
    hadamard: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``input`` to NVFP4 as Voyager's fake-quant does, packed for
    ``nvfp4_linear``.

    Repeats ``MXFakeQuantize`` and ``quantize_mx`` op for op in bfloat16: the
    random Hadamard transform, the global scale ``t`` from the tensor's amax,
    each block's scale -- its amax over 6 and over ``t``, rounded to
    ``fp8_e4m3``, and 1 where that is 0 -- and each element divided by its
    block scale times ``t`` and rounded to ``fp4_e2m1``.  A CUDA matrix, or
    a tensor blocked along its last axis, is cast by ``cast_nvfp4``'s two
    Triton kernels, the same arithmetic reading ``input`` twice in place.

    Args:
        input: bfloat16 tensor, a multiple of 16 long along ``axis``.
        axis: Axis the blocks run along, the GEMM's K.
        seed: int64 scalar keying stochastic rounding, by the random bits
            ``quantize`` draws for a tensor of ``input``'s shape; None
            rounds to nearest.
        hadamard: ``hadamard_matrix``, on ``input``'s device, to apply the
            random Hadamard transform along ``axis`` first; None does not.

    Returns:
        ``(codes, scales, t)``: the codes with ``axis`` moved last,
        ``[..., K / 2]``, and their block scales, as ``pack_nvfp4`` packs
        them over the codes' rows; and ``t``, a 0-d tensor.
    """
    ndim, axis = input.ndim, axis % input.ndim
    if (
        input.is_cuda
        and input.dtype is torch.bfloat16
        and input.shape[axis] % BLOCK_SIZE == 0
        and (axis == ndim - 1 or (ndim == 2 and axis == 0))
    ):
        codes, scales, global_scale = torch.ops.voyager_gpu.cast_nvfp4(
            input.reshape(-1, input.shape[-1]),
            hadamard,
            seed,
            1 if axis == ndim - 1 else 0,
        )
        shape = input.movedim(axis, -1).shape[:-1]
        return codes.view(*shape, -1), scales, global_scale
    if hadamard is not None:
        input = hadamard_rotate(input, axis, hadamard)
    global_scale = calculate_global_scale(input, FP4_MAX, E4M3_MAX)
    blocks = input.movedim(axis, -1).unflatten(-1, (-1, BLOCK_SIZE))
    scale = _block_scales(blocks.abs().amax(-1), global_scale)
    elements = blocks / (scale * global_scale).unsqueeze(-1)
    if seed is not None:
        random_bytes = _random_bytes(seed, input.shape, input.device)
        random_bytes = random_bytes.movedim(axis, -1).unflatten(
            -1, (-1, BLOCK_SIZE)
        )
        elements = _round_stochastically(elements, random_bytes)
    values = elements.flatten(-2)
    codes, scales = pack_nvfp4(values.flatten(0, -2), scale.flatten(0, -2))
    return codes.view(*values.shape[:-1], -1), scales, global_scale


def quantize_nvfp4_pair(
    input: torch.Tensor,
    seed: Optional[torch.Tensor] = None,
    transpose_seed: Optional[torch.Tensor] = None,
    hadamard: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """``quantize_nvfp4`` of a matrix along its last axis and of its
    transpose along its last axis, as ``quantize_mx_transpose`` quantizes
    them.  A contiguous CUDA bfloat16 matrix is cast by ``cast_nvfp4_pair``,
    reading it once per pass for both.

    Args:
        input: Matrix ``[M, K]``, ``M`` and ``K`` multiples of 16.
        seed: The row-wise cast's ``seed`` for ``quantize_nvfp4``.
        transpose_seed: The transpose's.
        hadamard: ``quantize_nvfp4``'s, for the transpose.

    Returns:
        ``quantize_nvfp4``'s outputs for the rows, then for the transpose.
    """
    if _paired_castable(input):
        return torch.ops.voyager_gpu.cast_nvfp4_pair(
            input, hadamard, seed, transpose_seed
        )
    return (
        *quantize_nvfp4(input, -1, seed),
        *quantize_nvfp4(input.mT, -1, transpose_seed, hadamard),
    )


def quantize_nvfp4_square(
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, ...]:
    """Quantize a weight to NVFP4 in 16x16 blocks as Voyager's fake-quant
    does, packed for its forward GEMM and, transposed, for its dgrad.

    ``quantize_nvfp4``'s arithmetic over 16x16 blocks: the global scale ``t``
    from the weight's amax, each block's scale from its amax, and each
    element divided by its block scale times ``t``.  A contiguous CUDA
    weight is cast by ``cast_nvfp4_square``, reading it once per pass for
    both layouts.

    Args:
        weight: bfloat16 ``[N, K]``, both multiples of 16.

    Returns:
        ``(codes, scales, transposed_codes, transposed_scales, t)``: the
        codes packed along K, ``[N, K / 2]``, and along N, ``[K, N / 2]``,
        each beside its block scales as ``pack_nvfp4`` packs them; and ``t``,
        a 0-d tensor.
    """
    if _paired_castable(weight):
        return torch.ops.voyager_gpu.cast_nvfp4_square(weight)
    global_scale = calculate_global_scale(weight, FP4_MAX, E4M3_MAX)
    blocks = weight.unflatten(-1, (-1, BLOCK_SIZE)).unflatten(
        -3, (-1, BLOCK_SIZE)
    )
    scale = _block_scales(blocks.abs().amax((-3, -1)), global_scale)
    element_scale = (scale * global_scale)[:, None, :, None]
    values = (blocks / element_scale).reshape(weight.shape)
    return (
        *pack_nvfp4(values, scale),
        *pack_nvfp4(values.mT, scale.mT),
        global_scale,
    )


def _round_stochastically(
    elements: torch.Tensor, random_bytes: torch.Tensor
) -> torch.Tensor:
    """Round each element to one of the two ``fp4_e2m1`` values around it by
    the rule and the float32 arithmetic of ``quantize``'s stochastic rounding.

    The two values come from the grid's spacing -- 0.5 below 2, 1 below 4 and
    2 up to 6 -- rather than from a search of its lookup table.

    Args:
        elements: Values already divided by their scales.
        random_bytes: Eight random bits per element.

    Returns:
        float32 values on the ``fp4_e2m1`` grid.
    """
    value = elements.float().clamp(-FP4_MAX, FP4_MAX)
    magnitude = value.abs()
    gap = torch.where(magnitude < 2, 0.5, torch.where(magnitude < 4, 1.0, 2.0))
    near = torch.floor(magnitude / gap) * gap
    far = torch.where(near == magnitude, near, near + gap)
    away = magnitude + random_bytes / 256 * (far - near) >= far
    return torch.copysign(torch.where(away, far, near), value)


def pack_nvfp4(
    values: torch.Tensor, scales: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 elements and their block scales in the layout ``nvfp4_linear``
    reads.

    Args:
        values: ``[R, K]`` elements, each rounded to the nearest ``fp4_e2m1``
            value (Voyager's codes are on that grid already).
        scales: Their block scales, values on the ``fp8_e4m3`` grid:
            ``[R, K / 16]``, one per 16 elements of a row, or
            ``[R / 16, K / 16]`` for 16x16 blocks, whose rows each read
            their block's.

    Returns:
        ``(codes, scales)``: ``[R, K / 2]`` uint8, two codes per byte along
        K, the even element's in the low nibble; and the scales' fp8_e4m3
        bytes as cuBLAS reads block scales, in 128x4 tiles over ``R`` rounded
        up to 128 rows and ``K / 16`` up to 4 columns, zero-padded.
    """
    rows, length = values.shape
    if scales.shape[0] != rows:
        scales = scales.repeat_interleave(BLOCK_SIZE, 0)[:rows]
    codes = pack_uint4(f32_to_f4_unpacked(values.float()))
    scales = to_blocked(scales.to(torch.float8_e4m3fn).view(torch.uint8))
    return codes, scales


def unpack_nvfp4(
    codes: torch.Tensor, scales: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The elements and block scales ``pack_nvfp4`` packed, in bfloat16.

    Args:
        codes: ``[..., K / 2]`` packed codes.
        scales: Their packed block scales.

    Returns:
        ``(values, block_scales)``: ``[..., K]`` and ``[..., K / 16]``.
    """
    values = f4_unpacked_to_f32(unpack_uint4(codes)).to(torch.bfloat16)
    *rows, length = values.shape
    blocks = from_blocked(
        scales, values.numel() // length, length // BLOCK_SIZE
    )
    block_scales = blocks.view(torch.float8_e4m3fn).reshape(*rows, -1)
    return values, block_scales.to(torch.bfloat16)


_LIBRARY.define(
    "nvfp4_linear(Tensor input, Tensor input_scale, Tensor weight, "
    "Tensor weight_scale, Tensor alpha, Tensor one, Tensor? bias) -> Tensor"
)


@impl(_LIBRARY, "nvfp4_linear", "CUDA")
def nvfp4_linear(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    one: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """``input @ weight.T * alpha + bias`` in bfloat16, on packed NVFP4
    operands, as one ``F.scaled_mm`` with block scales per 16 elements and
    ``alpha`` as the second-level (``TensorWise``) scale, applied to the
    float32 product inside the GEMM.

    The scalars are tensors the caller holds, so a call creates none.

    Args:
        input: ``[..., K / 2]`` codes, as ``quantize_nvfp4`` packs them.
        input_scale: Their packed block scales.
        weight: ``[N, K / 2]`` codes, as ``pack_nvfp4`` packs them.
        weight_scale: Their packed block scales.
        alpha: float32 scalar holding the bfloat16 product of the two global
            scales.
        one: float32 scalar 1, the weight's second-level scale, since
            ``alpha`` holds both global scales.
        bias: ``[N]`` in bfloat16, or None.

    Returns:
        ``[..., N]`` in bfloat16.
    """
    a = input.reshape(-1, input.shape[-1]).view(torch.float4_e2m1fn_x2)
    # The GEMM reads its second operand column-major: [K / 2, N] over
    # [N, K / 2] in memory.
    b = weight.view(torch.float4_e2m1fn_x2).t()
    scale_a = input_scale.view(torch.float8_e4m3fn)
    scale_b = weight_scale.view(torch.float8_e4m3fn)
    blockwise, swizzled = ScalingType.BlockWise1x16, SwizzleType.SWIZZLE_32_4_4
    tensorwise, unswizzled = ScalingType.TensorWise, SwizzleType.NO_SWIZZLE
    out = F.scaled_mm(
        a,
        b,
        [scale_a, alpha],
        [blockwise, tensorwise],
        [scale_b, one],
        [blockwise, tensorwise],
        swizzle_a=[swizzled, unswizzled],
        swizzle_b=[swizzled, unswizzled],
        bias=bias,
        output_dtype=torch.bfloat16,
    )
    return out.reshape(*input.shape[:-1], -1)


@torch.library.register_fake("voyager_gpu::nvfp4_linear")
def _(input, input_scale, weight, weight_scale, alpha, one, bias):
    shape = (*input.shape[:-1], weight.shape[0])
    return input.new_empty(shape, dtype=torch.bfloat16)


NVFP4_LINEAR = torch.ops.voyager_gpu.nvfp4_linear.default


def insert_gemm_scalars(
    model: GraphModule, alpha: Node, device: torch.device
) -> Tuple[Node, Node]:
    """Insert, at the graph's insertion point, the scalars an
    ``nvfp4_linear`` of ``model`` takes: ``alpha`` copied to float32 in the
    graph, where Inductor can fuse the copy into the op computing it, and a
    ``get_attr`` of the 1 all its GEMMs share, registered on first use.

    Args:
        model: The graph being lowered.
        alpha: The GEMM's bfloat16 product of global scales.
        device: The device ``model`` runs on.

    Returns:
        ``(alpha, one)``, as ``nvfp4_linear`` takes them.
    """
    if not hasattr(model, ONE):
        model.register_buffer(ONE, torch.ones((), device=device))
    graph = model.graph
    return (
        graph.call_function(
            torch.ops.aten._to_copy.default, (alpha,), {"dtype": torch.float32}
        ),
        graph.get_attr(ONE),
    )


def emulate_nvfp4_linear(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    one: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """``nvfp4_linear`` on any device, as the reference: the operands
    unpacked to bfloat16 and multiplied by ``linear_mx``, as the converted
    graph multiplies them.

    Args:
        input: As ``nvfp4_linear`` takes it.
        input_scale: As ``nvfp4_linear`` takes it.
        weight: As ``nvfp4_linear`` takes it.
        weight_scale: As ``nvfp4_linear`` takes it.
        alpha: As ``nvfp4_linear`` takes it.
        one: Taken for ``nvfp4_linear``'s signature only.
        bias: As ``nvfp4_linear`` takes it.

    Returns:
        ``[..., N]`` in bfloat16.
    """
    input, input_scale = unpack_nvfp4(input, input_scale)
    weight, weight_scale = unpack_nvfp4(weight, weight_scale)
    return torch.ops.quantized_ops.linear_mx(
        input,
        weight,
        bias,
        input_scale=input_scale,
        weight_scale=weight_scale,
        block_size=BLOCK_SIZE,
        alpha=alpha,
    )

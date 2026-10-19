"""The Triton kernels behind the CUDA calls of ``quantization.ops``.

``mx_fake_quant`` fake-quantizes a CUDA bfloat16 tensor in blocks along one
axis as ``quantization::mx_fake_quant``'s reference does -- block amax,
block scale (divided by a global scale and quantized through a scale table,
if given), division, rounding (to nearest through a lookup table, or
stochastically between levels), rescale, and with a random Hadamard
transform the rotation before and after -- repeating its float32 sums and
bfloat16 roundings op for op, so the output is bit-identical to it.
``hadamard_amax`` takes the rotation's global amax.  ``bin_blocks`` bins
block-normalized values into weighted moments in one pass where the
reference launches about twenty kernels; its bins are the reference's and
its sums differ only in the order they are added.  The ``*_supported``
predicates say which calls the kernels compute exactly.
``mx_fake_quant`` and ``hadamard_amax`` run inside ``triton_op`` bodies,
which ``torch.compile`` traces, so they launch through ``wrap_triton``;
``bin_blocks`` backs a plain custom op, which compile calls opaquely, so it
launches its kernel directly.  ``ops`` imports this module only when
triton, which ships with the CUDA builds of torch, is installed.
"""

import functools
import math
from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch.library import wrap_triton
from triton.language.extra import libdevice

from voyager_compiler.ops.quantized import (
    HADAMARD_SIZE,
    PHILOX_ROUNDS,
    QMAP_SIZE,
)
from voyager_compiler.quantization.mx_utils import FP32_MIN_NORMAL

#: Block sizes the fake-quant kernel takes.
BLOCK_SIZES = (8, 16, 32, 64)
#: Most levels stochastic rounding searches.
MAX_LEVELS = 256
#: Elements one Philox draw serves.
DRAW_SIZE = 16
#: Rows and columns of the tile one program fake-quantizes, its rows
#: widened to a whole block when the blocks run along them, and its warps.
TILE_ROWS, TILE_COLS, NUM_WARPS = 16, 64, 2
#: Elements one ``_bin_kernel`` program bins.
TILE = 2048
#: Copies of each bin ``bin_blocks`` spreads its adds over.  Neighbouring
#: elements add into different copies, so the many elements of one bin --
#: a ReLU output's zeros -- do not queue on one address; the copies are
#: summed after.
BIN_COPIES = 512

_DRAW_SIZE = tl.constexpr(DRAW_SIZE)
_HADAMARD_SIZE = tl.constexpr(HADAMARD_SIZE)
_PHILOX_ROUNDS = tl.constexpr(PHILOX_ROUNDS)
_MIN_NORMAL = tl.constexpr(FP32_MIN_NORMAL)


@triton.jit
def _bfloat16(x):
    """``x`` rounded to bfloat16, as float32."""
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _lookup(table, x):
    """The entry of a bfloat16 table for each of ``x``'s bfloat16 bit
    patterns, as ``vmap`` reads it, as float32."""
    bits = x.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32)
    return tl.load(table + (bits & 0xFFFF)).to(tl.float32)


@triton.jit
def _max_with_nan(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def _philox_byte(seed, counter, lane):
    """Byte ``lane % 4`` of word ``lane // 4`` of the Philox4x32 draw with
    ``counter`` under ``seed``, ``_random_bytes``' eight bits for the element
    at ``16 * counter + lane``; ``counter`` and ``lane`` broadcast."""
    c0 = (counter & 0xFFFFFFFF).to(tl.uint32)
    c1 = (counter >> 32).to(tl.uint32)
    zero = tl.zeros_like(c0)
    w0, w1, w2, w3 = tl.philox(seed, c0, c1, zero, zero, _PHILOX_ROUNDS)
    word = tl.where(
        lane < 8, tl.where(lane < 4, w0, w1), tl.where(lane < 12, w2, w3)
    )
    return (word >> ((lane % 4) * 8).to(tl.uint32)) & 0xFF


@triton.jit
def _flat_index(r, c, rows, cols, TRANSPOSED: tl.constexpr):
    """The flat index of element ``(r, c)`` of a ``[rows, cols]`` matrix, or
    of its transpose's ``(c, r)`` if ``TRANSPOSED``."""
    if TRANSPOSED:
        return c.to(tl.int64) * rows + r
    return r.to(tl.int64) * cols + c


@triton.jit
def _rotate(
    tile,
    h,
    ALONG_COLS: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    """``hadamard_rotate`` of a float32 tile holding a ``[ROWS, COLS]``
    matrix in row-major order, along its columns if ``ALONG_COLS``, else its
    rows: each run of ``HADAMARD_SIZE`` times the matrix at ``h`` in
    float32, or its transpose if ``TRANSPOSE``, rounded to bfloat16.

    An IEEE float32 ``tl.dot`` sums each element's products one by one in
    order, from zero, as cuBLAS's float32 GEMM does.

    Returns:
        The rotated tile as float32, in ``tile``'s shape.
    """
    i = tl.arange(0, _HADAMARD_SIZE)[:, None]
    j = tl.arange(0, _HADAMARD_SIZE)[None, :]
    # Element (i, j) of the matrix, and of its transpose.
    matrix, transpose = h + i * _HADAMARD_SIZE + j, h + j * _HADAMARD_SIZE + i
    if ALONG_COLS:
        runs = tl.reshape(tile, (ROWS * COLS // _HADAMARD_SIZE, _HADAMARD_SIZE))
        factor = tl.load(transpose if TRANSPOSE else matrix).to(tl.float32)
        rotated = tl.dot(runs, factor, input_precision="ieee")
    else:
        # The runs are columns, multiplied by the matrix's transpose from
        # the left.
        RUNS: tl.constexpr = ROWS // _HADAMARD_SIZE
        runs = tl.reshape(tile, (RUNS, _HADAMARD_SIZE, COLS))
        factor = tl.load(matrix if TRANSPOSE else transpose).to(tl.float32)
        factor = tl.broadcast_to(
            factor[None, :, :], (RUNS, _HADAMARD_SIZE, _HADAMARD_SIZE)
        )
        rotated = tl.dot(factor, runs, input_precision="ieee")
    return tl.reshape(_bfloat16(rotated), tile.shape)


@triton.jit
def _round_stochastically(
    v, byte, levels, NUM_LEVELS: tl.constexpr, SEARCH_STEPS: tl.constexpr
):
    """``_stochastic_round`` of float32 ``v`` between ``NUM_LEVELS``
    ascending float32 ``levels`` by its random ``byte``, as float32."""
    v = tl.maximum(v, tl.load(levels), propagate_nan=tl.PropagateNan.ALL)
    v = tl.minimum(
        v, tl.load(levels + NUM_LEVELS - 1), propagate_nan=tl.PropagateNan.ALL
    )
    # The levels below v, found as ``searchsorted`` finds them; a NaN sorts
    # after every level.
    below = tl.zeros(v.shape, tl.int32)
    for k in tl.static_range(SEARCH_STEPS):
        step = 1 << (SEARCH_STEPS - 1 - k)
        probe = tl.load(levels + tl.minimum(below + step, NUM_LEVELS) - 1)
        below = tl.where(probe < v, below + step, below)
    below = tl.where(v != v, NUM_LEVELS - 1, below)
    high = tl.load(levels + below)
    low = tl.where(high == v, high, tl.load(levels + tl.maximum(below - 1, 0)))
    near = tl.where(v < 0, high, low)
    far = tl.where(v < 0, low, high)
    noise = byte.to(tl.float32) * (1.0 / 256.0)
    # The product is rounded before the sum, as torch's separate ops round
    # it, not fused into one multiply-add.
    jitter = libdevice.mul_rn(noise, high - low)
    away = tl.abs(v) + jitter >= tl.abs(far)
    return tl.where(away, far, near)


@triton.jit
def _mx_fake_quant_kernel(
    x,
    out,
    table,
    scale_qmap,
    global_scale,
    seed,
    hadamard,
    rows,
    cols,
    quant_max,
    BLOCK: tl.constexpr,
    ALONG_COLS: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    DRAW_PER_BLOCK: tl.constexpr,
    NUM_LEVELS: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    TILE_COLS: tl.constexpr,
):
    """Fake-quantize one ``[TILE_ROWS, TILE_COLS]`` tile of the contiguous
    ``[rows, cols]`` matrix ``x`` into ``out``.

    Blocks of ``BLOCK`` elements run along the columns if ``ALONG_COLS``,
    else along the rows.  ``table`` is the lookup table nearest rounding
    reads, or with a ``seed`` the ``NUM_LEVELS`` levels stochastic rounding
    searches in ``SEARCH_STEPS`` halvings.  The random bytes follow the
    flat order of the tensor fake-quantized: the matrix, or its transpose if
    ``TRANSPOSED``.  ``DRAW_PER_BLOCK``: each block is 16 consecutive
    elements of that order starting at a multiple of 16, one Philox draw.
    With a ``hadamard`` matrix the tile is rotated along the blocks before
    the block amax and rotated back by the transpose before the store.
    """
    # The tile as [rows, blocks, BLOCK] or [blocks, BLOCK, cols]: (r, c)
    # is each element's position, (block_r, block_c) its block's first.
    lane = tl.arange(0, BLOCK)
    if ALONG_COLS:
        blocks: tl.constexpr = TILE_COLS // BLOCK
        block_r = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
        block_c = tl.program_id(1) * TILE_COLS + tl.arange(0, blocks) * BLOCK
        block_r, block_c = block_r[:, None, None], block_c[None, :, None]
        lane = lane[None, None, :]
        r, c = block_r, block_c + lane
        AXIS: tl.constexpr = 2
    else:
        blocks: tl.constexpr = TILE_ROWS // BLOCK
        block_r = tl.program_id(0) * TILE_ROWS + tl.arange(0, blocks) * BLOCK
        block_c = tl.program_id(1) * TILE_COLS + tl.arange(0, TILE_COLS)
        block_r, block_c = block_r[:, None, None], block_c[None, None, :]
        lane = lane[None, :, None]
        r, c = block_r + lane, block_c
        AXIS: tl.constexpr = 1
    mask = (r < rows) & (c < cols)
    offset = r.to(tl.int64) * cols + c
    tile = tl.load(x + offset, mask=mask, other=0.0).to(tl.float32)
    if hadamard is not None:
        tile = _rotate(tile, hadamard, ALONG_COLS, False, TILE_ROWS, TILE_COLS)

    # The block scale as ``calculate_mx_qparam`` forms it: the amax over
    # quant_max, a multiply by its float32 reciprocal as eager CUDA divides
    # by a scalar, divided by the global scale and looked up in the scale
    # table.  The elements are divided by it times the global scale.
    amax = tl.reduce(tl.abs(tile), AXIS, _max_with_nan, keep_dims=True)
    # torch.compile passes a float as float64.
    reciprocal = tl.math.div_rn(1.0, quant_max.to(tl.float32))
    scale = _bfloat16(amax * reciprocal)
    if global_scale is not None:
        t = tl.load(global_scale).to(tl.float32)
        scale = _bfloat16(tl.math.div_rn(scale, t))
    if scale_qmap is not None:
        scale = _lookup(scale_qmap, scale)
    scale = tl.where(scale > 0.0, scale, 1.0)
    element_scale = scale
    if global_scale is not None:
        element_scale = _bfloat16(scale * t)
    v = _bfloat16(tl.math.div_rn(tile, element_scale))

    if seed is not None:
        # Element i in flat order takes byte i % 16 of draw i // 16.
        if DRAW_PER_BLOCK:
            flat = _flat_index(block_r, block_c, rows, cols, TRANSPOSED)
        else:
            flat = _flat_index(r, c, rows, cols, TRANSPOSED)
            lane = (flat % _DRAW_SIZE).to(tl.int32)
        byte = _philox_byte(tl.load(seed), flat // _DRAW_SIZE, lane)
        q = _round_stochastically(v, byte, table, NUM_LEVELS, SEARCH_STEPS)
        q = _bfloat16(q)
    else:
        q = _lookup(table, v)
    output = _bfloat16(q * scale)
    if global_scale is not None:
        output = _bfloat16(output * t)
    if hadamard is not None:
        output = _rotate(
            output, hadamard, ALONG_COLS, True, TILE_ROWS, TILE_COLS
        )
    tl.store(out + offset, output.to(tl.bfloat16), mask=mask)


@triton.jit
def _hadamard_amax_kernel(
    x,
    hadamard,
    partial,
    rows,
    cols,
    ALONG_COLS: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    TILE_COLS: tl.constexpr,
):
    """The largest magnitude of one ``[TILE_ROWS, TILE_COLS]`` tile of the
    contiguous ``[rows, cols]`` matrix ``x``, its runs along the columns if
    ``ALONG_COLS``, else along the rows, rotated by ``hadamard``; into
    ``partial``, one element per program."""
    r = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)[:, None]
    c = tl.program_id(1) * TILE_COLS + tl.arange(0, TILE_COLS)[None, :]
    tile = tl.load(
        x + r.to(tl.int64) * cols + c,
        mask=(r < rows) & (c < cols),
        other=0.0,
    ).to(tl.float32)
    tile = _rotate(tile, hadamard, ALONG_COLS, False, TILE_ROWS, TILE_COLS)
    amax = tl.reduce(tl.abs(tile), None, _max_with_nan)
    index = tl.program_id(0) * tl.num_programs(1) + tl.program_id(1)
    tl.store(partial + index, amax.to(partial.dtype.element_ty))


@triton.jit
def _bin_kernel(
    x_ptr,
    energy_ptr,
    edges_ptr,
    scale_qmap_ptr,
    global_scale_ptr,
    row_cells_ptr,
    column_cells_ptr,
    copies_ptr,
    block_rows,
    blocks,
    length,
    inner,
    inner_tiles,
    cells,
    reciprocal,
    quant_max_exponent,
    BINS: tl.constexpr,
    EDGES: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
    COLUMNS: tl.constexpr,
    COPIES: tl.constexpr,
    WEIGHTED: tl.constexpr,
    POWER_OF_TWO: tl.constexpr,
    SCALE_QMAP: tl.constexpr,
    GLOBAL_SCALE: tl.constexpr,
    GRID: tl.constexpr,
):
    """Bin one tile: ``ROWS`` block rows by ``COLUMNS`` inner columns.

    The operand is read as ``[outer, length, inner]``, blocked along
    ``length``: block row ``r`` is block ``r % blocks`` of outer index
    ``r // blocks``.  Each of the tile's ``ROWS * COLUMNS`` blocks is one
    row of ``BLOCK`` elements below.  With a grid, a block's cell is its
    outer index's share of the cell index plus its column's.
    """
    tile = tl.program_id(0)
    block = tl.arange(0, ROWS * COLUMNS)
    row = tile // inner_tiles * ROWS + block // COLUMNS
    column = tile % inner_tiles * COLUMNS + block % COLUMNS
    position = (row % blocks)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    mask = (
        (row < block_rows)[:, None]
        & (column < inner)[:, None]
        & (position < length)
    )
    outer = (row // blocks).to(tl.int64)[:, None]
    offset = (outer * length + position) * inner + column[:, None]
    x = tl.load(x_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    cell = 0
    if GRID:
        cell = tl.load(
            row_cells_ptr + row // blocks, mask=row < block_rows, other=0
        ) + tl.load(column_cells_ptr + column, mask=column < inner, other=0)
        cell = cell.to(tl.int64)[:, None]

    # The block scale as calculate_mx_qparam forms it on a GPU: a power of
    # two from the maximum's exponent, or the maximum over quant_max -- a
    # multiply by its float32 reciprocal, as eager CUDA divides by a scalar
    # -- divided by the global scale and looked up in the scale codebook as
    # vmap looks it up.  The scale the elements are divided by is that one
    # times the global scale, as quantize_mx forms it.
    amax = tl.max(tl.abs(x), axis=1)
    if GLOBAL_SCALE:
        global_scale = tl.load(global_scale_ptr).to(tl.float32)
    if POWER_OF_TWO:
        amax = tl.where(amax == 0, _MIN_NORMAL, amax)
        exponent = libdevice.floor(libdevice.log2(amax)) - quant_max_exponent
        scale = libdevice.exp2(exponent)
    else:
        scale = amax * reciprocal
        if GLOBAL_SCALE:
            scale = tl.math.div_rn(scale, global_scale)
        if SCALE_QMAP:
            pattern = scale.to(tl.bfloat16).to(tl.int16, bitcast=True)
            entry = pattern.to(tl.int32) & 0xFFFF
            scale = tl.load(scale_qmap_ptr + entry).to(tl.float32)
    scale = tl.where(scale > 0, scale, 1.0)
    if GLOBAL_SCALE:
        scale = scale * global_scale
    value = tl.math.div_rn(x, tl.broadcast_to(scale[:, None], x.shape))

    # A value's bin is the number of interior edges at or below it, clamped
    # as the reference clamps; the edges are padded with +inf to EDGES, a
    # power of two, so the search halves a range of known size.
    index = tl.zeros(x.shape, dtype=tl.int32)
    step = EDGES // 2
    for _ in tl.static_range(SEARCH_STEPS):
        edge = tl.load(edges_ptr + index + step - 1)
        index = tl.where(edge <= value, index + step, index)
        step = step // 2
    index = tl.minimum(index, BINS - 1)

    # Each element is weighted by its block's squared scale as well, in the
    # energy's precision as the reference multiplies, before widening.
    square = tl.broadcast_to((scale * scale)[:, None], x.shape)
    if WEIGHTED:
        weight = tl.load(
            energy_ptr + cell * length + position, mask=mask, other=0.0
        )
        weight = weight * square
    else:
        weight = square
    weight = weight.to(tl.float64)
    value = value.to(tl.float64)
    weighted = weight * value

    # Neighbouring elements add into different copies of the bins, so a run
    # of equal values -- a ReLU's zeros, each block's maximum -- does not
    # queue on one address.
    element = block[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    copy = ((element + tile) % COPIES).to(tl.int64)
    target = copies_ptr + (copy * cells + cell) * (3 * BINS) + index
    tl.atomic_add(target, weight, mask=mask, sem="relaxed")
    # An exact zero adds nothing to the two value-weighted sums.
    nonzero = mask & (value != 0)
    tl.atomic_add(target + BINS, weighted, mask=nonzero, sem="relaxed")
    tl.atomic_add(
        target + 2 * BINS, weighted * value, mask=nonzero, sem="relaxed"
    )


def _matrix(
    input: torch.Tensor, axis: int
) -> Optional[Tuple[int, int, bool, bool]]:
    """How the kernels read ``input``.

    Returns:
        ``(rows, cols, along_cols, transposed)``: the contiguous matrix
        ``input``'s memory holds, whether the blocks along ``axis`` run along
        its columns, and whether ``input`` is its transpose.  None unless
        ``input`` is contiguous and blocked along its last axis or, as a
        matrix, along either, or is a contiguous matrix's transpose.
    """
    axis %= input.ndim
    if input.is_contiguous():
        cols = input.shape[-1]
        if axis == input.ndim - 1:
            return input.numel() // cols, cols, True, False
        if input.ndim == 2 and axis == 0:
            return *input.shape, False, False
    elif input.ndim == 2 and input.mT.is_contiguous():
        return *input.shape[::-1], axis == 0, True
    return None


def _is_lookup_table(table: torch.Tensor) -> bool:
    """Whether ``table`` is a ``QMAP_SIZE``-entry bfloat16 table."""
    return table.dtype is torch.bfloat16 and table.shape == (QMAP_SIZE,)


def mx_fake_quant_supported(
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
) -> bool:
    """Whether ``mx_fake_quant`` computes a call of the op exactly.

    It takes a CUDA bfloat16 tensor laid out as ``_matrix`` reads it, in
    blocks of a ``BLOCK_SIZES`` size along one axis, each block scaled by
    its amax rather than a power of two; a ``QMAP_SIZE``-entry bfloat16
    table and scale table, or rounding stochastically at most
    ``MAX_LEVELS`` float32 levels; and a rotation only along the blocks'
    axis when it is whole runs of ``HADAMARD_SIZE``, which
    ``hadamard_rotate`` pads nothing; and a global scale in the input's
    dtype, as ``calculate_global_scale`` computes it.

    Args:
        The op's arguments, as ``quantization::mx_fake_quant`` takes them;
        ``quant_max`` does not decide it.

    Returns:
        True if the kernel computes the call exactly.
    """
    if seed is not None:
        table_fits = (
            qmap.dtype is torch.float32
            and qmap.dim() == 1
            and qmap.numel() <= MAX_LEVELS
        )
    else:
        table_fits = _is_lookup_table(qmap)
    return (
        table_fits
        and input.is_cuda
        and input.dtype is torch.bfloat16
        and input.numel() > 0
        and len(axes) == 1
        and block_size in BLOCK_SIZES
        and not power_of_two
        and _matrix(input, axes[0]) is not None
        and (scale_qmap is None or _is_lookup_table(scale_qmap))
        and (global_scale is None or global_scale.dtype is input.dtype)
        and (
            hadamard is None
            or (
                hadamard_axis % input.ndim == axes[0] % input.ndim
                and input.shape[hadamard_axis] % HADAMARD_SIZE == 0
            )
        )
    )


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
    """``quantization::mx_fake_quant`` of a call ``mx_fake_quant_supported``
    takes, in one kernel.

    Args:
        The op's arguments.  ``power_of_two`` is False and ``hadamard_axis``
        the blocks' axis, as the call is supported.

    Returns:
        The fake-quantized tensor, laid out as ``input``.
    """
    output = torch.empty_like(input)
    rows, cols, along_cols, transposed = _matrix(input, axes[0])
    # Each block is one draw if it is 16 elements of the tensor's last
    # axis, whose length is a multiple of 16.
    draw_per_block = (
        seed is not None
        and block_size == DRAW_SIZE
        and along_cols != transposed
        and (rows if transposed else cols) % DRAW_SIZE == 0
    )
    tile_rows = TILE_ROWS if along_cols else max(TILE_ROWS, block_size)
    num_levels = qmap.numel() if seed is not None else 1
    grid = (triton.cdiv(rows, tile_rows), triton.cdiv(cols, TILE_COLS))
    wrap_triton(_mx_fake_quant_kernel)[grid](
        input,
        output,
        qmap,
        scale_qmap,
        global_scale,
        seed,
        hadamard,
        rows,
        cols,
        quant_max,
        BLOCK=block_size,
        ALONG_COLS=along_cols,
        TRANSPOSED=transposed,
        DRAW_PER_BLOCK=draw_per_block,
        NUM_LEVELS=num_levels,
        SEARCH_STEPS=(num_levels - 1).bit_length(),
        TILE_ROWS=tile_rows,
        TILE_COLS=TILE_COLS,
        num_warps=NUM_WARPS,
    )
    return output


def hadamard_amax_supported(input: torch.Tensor, axis: int) -> bool:
    """Whether ``hadamard_amax`` computes a call of the op exactly: a CUDA
    bfloat16 ``input`` laid out as ``_matrix`` reads it, rotated along an
    ``axis`` of whole runs of ``HADAMARD_SIZE``."""
    return (
        input.is_cuda
        and input.dtype is torch.bfloat16
        and input.numel() > 0
        and _matrix(input, axis) is not None
        and input.shape[axis] % HADAMARD_SIZE == 0
    )


def hadamard_amax(
    input: torch.Tensor, axis: int, hadamard: torch.Tensor
) -> torch.Tensor:
    """``quantization::hadamard_amax`` of a call ``hadamard_amax_supported``
    takes: one kernel's per-tile maxima, then their max.

    Args:
        The op's arguments.

    Returns:
        A 0-d tensor in ``input``'s dtype.
    """
    rows, cols, along_cols, _ = _matrix(input, axis)
    grid = (triton.cdiv(rows, TILE_ROWS), triton.cdiv(cols, TILE_COLS))
    partial = input.new_empty(grid)
    wrap_triton(_hadamard_amax_kernel)[grid](
        input,
        hadamard,
        partial,
        rows,
        cols,
        ALONG_COLS=along_cols,
        TILE_ROWS=TILE_ROWS,
        TILE_COLS=TILE_COLS,
        num_warps=NUM_WARPS,
    )
    return partial.amax()


@functools.lru_cache(maxsize=None)
def _cell_maps(shape, axis, counts, device):
    """Each outer row's and each inner column's share of a block's cell.

    The tensor reads as ``[outer, length, inner]``, with ``axis`` split
    into its ``counts[axis]`` runs and the run the last outer index.  A
    cell's row-major index is a sum of one term per axis, so the outer
    axes' terms and the inner axes' add.

    Args:
        shape: The tensor's shape.
        axis: The axis its blocks run along.
        counts: Runs along each axis.
        device: Where the maps live.

    Returns:
        ``(rows, columns)``: int32 shares over the outer rows and the inner
        columns.
    """
    strides = [math.prod(counts[d + 1 :]) for d in range(len(shape))]

    def terms(axes):
        total = torch.zeros((), dtype=torch.int64)
        for d in axes:
            run = torch.arange(shape[d]) // (shape[d] // counts[d])
            total = total[..., None] + run * strides[d]
        return total

    rows = terms(range(axis))[..., None]
    rows = rows + torch.arange(counts[axis]) * strides[axis]
    columns = terms(range(axis + 1, len(shape)))
    return (
        rows.flatten().int().to(device),
        columns.flatten().int().to(device),
    )


def bin_blocks(
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
    """``quantization::bin_blocks`` in one kernel, for any CUDA call.

    Args:
        The op's arguments.
    """
    tensor = tensor.contiguous()
    axis %= tensor.dim()
    runs = counts[axis]
    length = tensor.shape[axis] // runs
    outer = math.prod(tensor.shape[:axis]) * runs
    inner = math.prod(tensor.shape[axis + 1 :])
    cells = len(sums)
    row_cells = column_cells = edges
    if cells > 1:
        row_cells, column_cells = _cell_maps(
            tuple(tensor.shape), axis, tuple(counts), tensor.device
        )
    blocks = triton.cdiv(length, block_size)
    columns = min(triton.next_power_of_2(inner), 64)
    rows = max(TILE // (block_size * columns), 1)
    inner_tiles = triton.cdiv(inner, columns)
    bins = sums[0].shape[1]
    scratch = torch.zeros(
        BIN_COPIES, cells, 3, bins, dtype=torch.float64, device=tensor.device
    )
    _bin_kernel[(triton.cdiv(outer * blocks, rows) * inner_tiles,)](
        tensor,
        edges if energy is None else energy.contiguous(),
        edges,
        edges if scale_qmap is None else scale_qmap,
        edges if global_scale is None else global_scale,
        row_cells,
        column_cells,
        scratch,
        outer * blocks,
        blocks,
        length,
        inner,
        inner_tiles,
        cells,
        (torch.tensor(1.0) / quant_max).item(),
        float(math.floor(math.log2(quant_max))),
        BINS=bins,
        EDGES=edges.numel(),
        SEARCH_STEPS=edges.numel().bit_length() - 1,
        BLOCK=block_size,
        ROWS=rows,
        COLUMNS=columns,
        COPIES=BIN_COPIES,
        WEIGHTED=energy is not None,
        POWER_OF_TWO=power_of_two,
        SCALE_QMAP=scale_qmap is not None,
        GLOBAL_SCALE=global_scale is not None,
        GRID=cells > 1,
    )
    torch._foreach_add_(sums, list(scratch.sum(0)))

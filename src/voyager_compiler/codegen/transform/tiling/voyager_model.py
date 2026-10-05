"""Voyager matrix model construction; invoked through the mapping policy."""

import math
import torch
from interstellar import loop_enum as le
from voyager_compiler.codegen.transform.tiling.cost import (
    _step_classes,
    _sweep_cycles,
    strided_bank_walk,
)
from typing import Optional, Tuple
from .traversal import MappingTraversal
from voyager_compiler.codegen.node_info import (
    _pair,
    get_arg_value,
    is_bmm,
    is_conv2d,
    weight_transforms,
)
from voyager_compiler.codegen.transform.tiling.cost import (
    get_dtype_width,
    _node_dtype_bits,
)
from voyager_compiler.ops.layout import NCHW_TO_NHWC, unproject


def prepare_matrix(problem, tiler):
    anchor = problem.anchor
    layer = problem.layer
    out_dtype = problem.out_dtype
    fused_specs = problem.fused_specs
    constraint = problem.constraint
    has_tail = problem.has_tail
    single_k_tail_extra_pass = problem.single_k_tail_extra_pass
    split_k_tail_extra_pass = problem.split_k_tail_extra_pass
    tail_keeps_shape = problem.tail_keeps_shape
    scratch_regions = problem.scratch_regions
    outlier_rate = problem.outlier_rate
    outlier_pct = problem.outlier_pct
    out_outlier_pct = problem.out_outlier_pct
    boundary = problem.boundary
    if_bits = problem.if_bits
    fl_bits = problem.fl_bits
    of_bits = problem.of_bits
    if_scale_bits = problem.if_scale_bits
    fl_scale_bits = problem.fl_scale_bits
    of_scale_bits = problem.of_scale_bits
    # A scratchpad bank moves one store word per cycle, ``bank_width`` bytes:
    # a port the hardware fixes, not one that widens with the element.  So a
    # row of elements wider than the port's lanes (int6 attention operands on
    # the 4-bit NF4 port) takes more than one beat.  Without a bank width the
    # port is taken as one input row per cycle.
    sram_bandwidth = tiler.config.sram_bandwidth_bits(if_bits)
    stream = tiler.config.connection("matrix_vector_stream").bandwidth
    accum_bits = get_dtype_width(anchor.value.dtype)
    stream_bytes = stream.bytes_per_cycle(tiler.config, accum_bits)
    output_stream_beats = math.ceil(
        tiler.config.pe_array_size[1] * accum_bits / 8 / stream_bytes
    )

    batch = math.prod(anchor.value.shape[:-2]) if is_bmm(anchor) else 1

    weight = anchor.args[1]
    transposed, repeat = weight_transforms(weight)[1:3]
    weight_repeat = (
        math.prod(repeat[: max(0, len(weight.shape) - 2)]) if repeat else 1
    )

    rc = RuntimeCalculator(
        if_bits,
        fl_bits,
        of_bits,
        get_dtype_width(anchor.value.dtype),
        tiler.config.double_buffered_accum_buffer,
        sram_bandwidth,
        tiler.config.bytes_per_cycle,
        tiler.config.access_latency_cycles,
        tiler.config.bank_switch_cycles,
        tiler.config.output_slack,
        tiler.config.spmm_row_cycles,
        output_stream_beats,
        double_buffered_l2=(
            tiler.config.double_buffered_l2 and tiler.resident_bytes is None
        ),
        batch=batch,
        weight_batch=batch // weight_repeat,
        has_tail=has_tail,
        single_k_tail_extra_pass=single_k_tail_extra_pass,
        split_k_tail_extra_pass=split_k_tail_extra_pass,
        tail_keeps_shape=tail_keeps_shape,
        tail_specs=fused_specs,
        input_scale_width=if_scale_bits,
        weight_scale_width=fl_scale_bits,
        output_scale_width=of_scale_bits,
        scale_block_size=anchor.kwargs.get("block_size") or 1,
        outlier_rate=outlier_rate,
        bias_width=_node_dtype_bits(get_arg_value(anchor, 2, "bias", None), 0),
        stride=(layer.hstd, layer.wstd),
        bank_size=tiler.config.bank_size,
        weight_transposed=transposed,
    )
    rc.resident = tiler.resident_bytes is not None
    rc.stream_weights = tiler.stream_weights
    native_padding = (
        rc.resident
        and is_conv2d(anchor)
        and any(_pair(get_arg_value(anchor, 4, "padding", 0)))
    )
    if native_padding:
        dims = NCHW_TO_NHWC if anchor.meta.get("transposed", False) else None
        rc.resident_input_hw = unproject(anchor.args[0].value.shape, dims)[2:]
    # Match the direct-window ingress contract: transposed weights and
    # convolution halos/gaps retain whole loads outside the tile loop.
    dense_input = (
        native_padding
        or not is_conv2d(anchor)
        or (layer.hfil == layer.wfil == layer.hstd == layer.wstd == 1)
    )
    rc.resident_boundary = (
        boundary[0] and dense_input,
        boundary[1] and dense_input,
        boundary[2] and not transposed,
        boundary[3] and not transposed,
        boundary[4],
    )

    # Built up front rather than per attempt: each one reads the node, which
    # only the parent may do.  They close over it, so they cannot be pickled --
    # hence a forked worker rather than a spawned one.
    size_fn = make_size_fn(
        node=anchor,
        out_dtype=out_dtype,
        fused_specs=fused_specs,
        constraint=constraint,
        has_tail=has_tail,
        single_k_tail_extra_pass=single_k_tail_extra_pass,
        tail_keeps_shape=tail_keeps_shape,
        scratch_regions=1 if rc.resident else scratch_regions,
        num_slots=1 if rc.resident else tiler.config.num_slots,
        batch=batch,
        weight_batch=batch // weight_repeat,
        outlier_pct=outlier_pct,
        out_outlier_pct=out_outlier_pct,
        bank_width=tiler.config.bank_width,
        vector_lanes=tiler.config.vector_lanes,
        spmm_scale_rows_limit=tiler.config.spmm_scale_rows,
        resident_bytes=tiler.resident_bytes,
        stream_weights=tiler.stream_weights,
    )
    return size_fn, rc


# Positions of interstellar's (input, output, weight) byte triple.
_IF, _OF, _FL = 0, 1, 2

# The L3 loops each operand's tile spans.  A tile advances only when one of
# them turns, so an operand whose every entry is a single L3 step reads one
# tile for the whole sweep.
_IF_DIMS = (le.OX, le.OY, le.IC, le.ON)
_FL_DIMS = (le.OC, le.IC, le.FX, le.FY)
_OF_DIMS = (le.OC, le.OY, le.OX, le.ON)

_QUANTIZE_MX_OUTLIER = torch.ops.quantized_ops.quantize_mx_outlier.default


def spmm_scale_rows(mapping):
    """Weight-scale buffer rows ``mapping``'s tile takes in the SpMM unit:
    its K blocks (the L1 and L2 IC blockings; the PE level is the array)
    times its L1 OC passes -- ``C * K0`` in the toolchain's ``SpMM.h``."""
    ic, oc = mapping.loop_blockings[le.IC], mapping.loop_blockings[le.OC]
    return ic[1] * ic[2] * oc[1]


def output_is_psum(point, level):
    """Whether the output stored at ``level`` is still a partial sum: it is,
    while the IC reduction is incomplete *above* this level.  A partial sum is
    held at the accumulator's width, and carries no output scale yet."""
    num_levels = len(point.loop_blocking(le.IC))
    ic_above = 1
    for lvl in range(level + 1, num_levels):
        ic_above *= point.loop_blocking(le.IC)[lvl]
        ic_above *= point.loop_partitioning(le.IC)[lvl]
    return ic_above > 1


def make_size_fn(
    node,
    out_dtype=None,
    fused_specs=(),
    constraint=None,
    has_tail=False,
    single_k_tail_extra_pass=False,
    tail_keeps_shape=False,
    scratch_regions=1,
    num_slots=1,
    batch=1,
    weight_batch=1,
    outlier_pct=0.0,
    out_outlier_pct=0.0,
    bank_width=None,
    vector_lanes=None,
    *,
    spmm_scale_rows_limit,
    resident_bytes=None,
    stream_weights=False,
):
    """Build a ``Layer.size_fn``: the bytes a tile occupies at a byte-pool
    level.

    Interstellar hands over element counts and the mapping; everything that
    turns those into bytes is policy and lives here -- element widths,
    microscaling scale tensors (one scale per ``block_size`` values), the bias
    and fused post-op operands interstellar knows nothing about, and how all of
    them are packed into banks.

    Operands are grouped one per bank ideally:

        input+csr | input_scale | weight+weight_scale+bias
                  | output+output_scale+csr staging | each fused operand

    An outlier CSR the GEMM consumes rides the input's bank -- both are
    DMA-written and compute-read at the same depth -- and the CSR a fused
    tail emits stages in the output's, beside the dense pair it stores by
    the same route.

    Each such source is ping-ponged, so it costs ``num_slots`` whole banks --
    the two halves live in *separate* banks, which is what the planner does and
    what lets a load overlap the compute reading the other half.  A split
    reduction with a fused tail also accumulates into a scratch buffer the
    builders allocate exactly once (``_ScratchSpec``); it is charged a single
    bank-aligned region on top.  So is the finished tile a tail that needs
    a pass of its own (``single_k_tail_extra_pass``) parks even for a
    single round, unless it keeps the tile's shape (``tail_keeps_shape``)
    and runs that pass in place on the output slot.  Leaving either out is
    what let the tiler return tilings ``plan_memory`` could not place.  A
    split reduction whose tail reads that tile back from scratch may take
    ``scratch_regions`` of them: a second region lets the read-back pass
    ride the finalize commit instead of running bare, so it is charged
    whenever it fits beside the sources, and the tile falls back to one
    region -- priced as the bare pass -- when it does not.

    A bank cannot be split between groups, so each group rounds up to a whole
    bank -- which puts a floor of ``num_slots * len(groups) * bank_size`` on the
    tile, however small it is.  While the groups' banks exceed the budget the
    two *smallest* are merged (a tiny scale tensor would otherwise waste a
    whole bank), so a tile is admitted with the least sharing that fits it
    and the cost model prices that sharing.  Only whole sources merge, never
    a source's own two slots -- each slot must keep its own bank.

    Every operand is sized the way ``tensor_alloc_bytes`` sizes it for the
    planner -- payload, the slack a final store beat overshoots by, aligned to
    ``bank_width`` -- and each operand of a shared bank is padded before they
    are summed, because the planner lays them out one after another.  Sizing
    a tile at its raw payload instead lets a group whose payload lands on a
    bank boundary claim one more bank in the plan than the search charged.

    ``fused_specs`` are the ``(dims, dtype_bits)`` pairs from
    ``_fused_operand_specs``.  ``bank_size is None`` -> no banking: just sum.

    The returned ``size_fn`` exposes the partition itself as
    ``size_fn.compute_groups``: ``_finish_search`` replays it for the winning
    mapping and stamps the surviving role partition for the memory planner,
    so the plan realizes exactly the banking the fit check priced.
    """
    if isinstance(out_dtype, (list, tuple)):
        of_scale_dtype, of_dtype = out_dtype[-2], out_dtype[-1]
    else:
        of_scale_dtype, of_dtype = None, out_dtype
    # A CSR-producing tail returns (data, indices, indptr, scale, inliers);
    # a staged entry is an outlier value and its column index, each its own
    # buffer.
    out_csr_data_bits, out_csr_index_bits = (
        (get_dtype_width(out_dtype[0]), get_dtype_width(out_dtype[1]))
        if isinstance(out_dtype, (list, tuple)) and len(out_dtype) == 5
        else (0, 0)
    )
    if_bits = _node_dtype_bits(node.args[0])
    fl_bits = _node_dtype_bits(node.args[1])
    of_bits = get_dtype_width(of_dtype) if of_dtype else _node_dtype_bits(node)
    bias_bits = _node_dtype_bits(get_arg_value(node, 2, "bias", None), 0)
    if_scale_bits = _node_dtype_bits(node.kwargs.get("input_scale"), 0)
    fl_scale_bits = _node_dtype_bits(node.kwargs.get("weight_scale"), 0)
    of_scale_bits = get_dtype_width(of_scale_dtype) if of_scale_dtype else 0
    stage_bits = get_dtype_width(node.value.dtype)
    block_size = node.kwargs.get("block_size") or 1
    # A gathered-CSR entry: the outlier value and its column index, each
    # staged in its own buffer.
    csr_data_bits = _node_dtype_bits(node.kwargs.get("A_data"), 0)
    csr_index_bits = _node_dtype_bits(node.kwargs.get("A_indices"), 0)
    # A CSR consumer streams its weight tile through the SpMM unit.
    is_spmm = node.kwargs.get("A_data") is not None

    def _align(size):
        """Round ``size`` up to a whole ``bank_width`` store word."""
        if not bank_width:
            return size
        return math.ceil(size / bank_width) * bank_width

    def _alloc_bytes(count, bits):
        """Bytes an on-chip buffer of ``count`` ``bits``-wide elements takes.

        Mirrors ``tensor_alloc_bytes``, which is what the memory planner
        allocates through: the payload, plus the slack a final store beat of
        ``vector_lanes`` values overshoots the payload by, aligned to a whole
        store word.
        """
        if not bits or count <= 0:
            return 0.0
        size = math.ceil(count * bits / 8.0)
        if bank_width and vector_lanes:
            beat = math.ceil(vector_lanes * bits / 8.0)
            size += _align(beat) - beat
        return float(_align(size))

    def _scale_bytes(count, bits):
        """Bytes the scales of ``count`` values take: one per block."""
        return _alloc_bytes(count / block_size, bits)

    def compute_groups(
        counts, point, level, partitioning_accum, bank_size, num_banks
    ):
        """The bank partition of one candidate tile: the merged ``(bytes,
        kind, slots, roles)`` groups, the scratch region's bytes and the
        regions it takes, or ``(None, 0.0, 1)`` for a vetoed tile.
        ``roles`` is the set of operand
        roles sharing the bank (``"input"``/``"csr"``, ``"input_scale"``,
        ``"weight"``/``"weight_scale"``/``"bias"``, ``"output"``,
        ``("fused", i)``); a merge unions the two sets.  ``size_fn`` sums
        exactly these groups, so the stamped partition and the fit check can
        never disagree."""
        if_count, of_count, fl_count = counts
        is_psum = output_is_psum(point, level)

        def extent(d):
            """Output-dim extent here: one bank's worth, or the whole spatially
            replicated block when a partitioning is given."""
            e = 1
            for b in point.loop_blocking(d)[: level + 1]:
                e *= b
            if partitioning_accum is not None:
                e *= partitioning_accum[d]
            return e

        # Veto a tile off the pinned loop extents: an OC tile that splits an
        # attention head (the MHA relayout must store whole heads), a CSR
        # slice width the coupled ops agreed on.
        if constraint is not None and not constraint.allows(extent):
            return None, 0.0, 1
        # The typed memory instance owns the SpMM scale-store depth.
        if is_spmm and spmm_scale_rows(point) > spmm_scale_rows_limit:
            return None, 0.0, 1

        # The output is a partial sum in the anchor's dtype until IC is fully
        # reduced; only the final value carries an output scale.
        out_bits = stage_bits if is_psum else of_bits
        of_scale = 0.0 if is_psum else _scale_bytes(of_count, of_scale_bits)
        bias = _alloc_bytes(extent(le.OC), bias_bits)
        # A CSR the tail emits stages in the output's bank before its packed
        # stores: a block's (value, index) pairs and its row pointers.  The
        # quantize sees the output tile, so the budget follows it.
        staged_csr = 0.0
        if out_csr_data_bits:
            rows = of_count / max(1.0, extent(le.OC))
            staged = of_count * out_outlier_pct
            staged_csr = (
                _alloc_bytes(staged, out_csr_data_bits)
                + _alloc_bytes(staged, out_csr_index_bits)
                + _alloc_bytes(rows + 1, 32)
            )

        def slots(dims, distinct):
            """Banks an operand spanning ``dims`` needs: a second holds the
            next tile, so an operand with one tile over the whole sweep needs
            one.  The mapping covers a single batch element -- the builder
            loops the rest -- so the sweep's tile count is the L3 trips times
            ``distinct``, the operand's distinct tiles over that batch loop
            (``_batch_loads`` counts fetches the same way).  The builders'
            ``PipelinedKernel._num_slots`` resolves the depth from the full
            grid, so this never charges less than is allocated."""
            tiles = distinct
            for d in dims:
                tiles *= point.loop_blocking(d)[3]
            return num_slots if tiles > 1 else 1

        # The gathered CSR shares the input's bank: a block's (value, index)
        # pairs, at the fraction of the activation the stream is sized for.
        # Its staging ping-pongs exactly when the input does -- a sweep that
        # gathers once keeps one slot (``_SparseGemm.forward``).
        gathered = if_count * outlier_pct
        groups = [
            (
                _alloc_bytes(if_count, if_bits)
                + _alloc_bytes(gathered, csr_data_bits)
                + _alloc_bytes(gathered, csr_index_bits),
                _IF,
                slots(_IF_DIMS, batch),
                {"input", "csr"},
            ),
            (
                _scale_bytes(if_count, if_scale_bits),
                _IF,
                slots(_IF_DIMS, batch),
                {"input_scale"},
            ),
            (
                _alloc_bytes(fl_count, fl_bits)
                + _scale_bytes(fl_count, fl_scale_bits)
                + bias,
                _FL,
                slots(_FL_DIMS, weight_batch),
                {"weight", "weight_scale", "bias"},
            ),
            (
                _alloc_bytes(of_count, out_bits) + of_scale + staged_csr,
                _OF,
                slots(_OF_DIMS, batch),
                {"output"},
            ),
        ]
        for i, (dims, bits) in enumerate(fused_specs):
            count = 1
            for d in dims:
                count *= extent(d)
            distinct = batch if le.ON in dims else 1
            groups.append(
                (
                    _alloc_bytes(count, bits),
                    _OF,
                    slots(dims, distinct),
                    {("fused", i)},
                )
            )

        # An absent operand (no scale, no bias) occupies no bank.
        groups = [g for g in groups if g[0] > 0]

        # A reduction split across L3 steps accumulates into a scratch buffer
        # the builders allocate once for the whole kernel, not per ping-pong
        # half -- so it is charged one region, outside ``num_slots``.  A
        # shape-changing extra pass parks even a single round's tile.
        if has_tail and (
            point.loop_blocking(le.IC)[3] > 1
            or (single_k_tail_extra_pass and not tail_keeps_shape)
        ):
            scratch = _alloc_bytes(of_count, stage_bits)
        else:
            scratch = 0.0
        # Only a split reduction's read-back tail has a second region to
        # gain; a single staged round keeps one.
        regions = 1
        if scratch and point.loop_blocking(le.IC)[3] > 1:
            regions = scratch_regions

        if not bank_size:
            return groups, scratch, regions

        scratch_banks = math.ceil(scratch / bank_size) if scratch else 0

        def banks(groups):
            return sum(
                n * math.ceil(size / bank_size) for size, _, n, _ in groups
            )

        def merge_to_fit(budget):
            """The groups with their two smallest merged until their banks
            fit ``budget``, and whether they do -- one group may still not."""
            merged = list(groups)
            while len(merged) > 1 and banks(merged) > budget:
                merged.sort(key=lambda g: g[0])
                (s0, k0, n0, r0), (s1, k1, n1, r1) = merged[0], merged[1]
                # Charge the shared bank to the larger member's operand, and
                # to the deeper pipeline: one buffer holding both has to
                # ping-pong if either of them does.
                merged = [
                    (s0 + s1, k0 if s0 >= s1 else k1, max(n0, n1), r0 | r1)
                ] + merged[2:]
            return merged, banks(merged) <= budget

        # Banks left for the ping-ponged sources, after the scratch regions.
        # Without a bank count the partition is one source per bank, so
        # nothing merges.  A second scratch region is kept only when the
        # sources still fit beside it.
        while True:
            budget = (num_banks or math.inf) - regions * scratch_banks
            merged, fits = merge_to_fit(budget)
            if fits or regions == 1:
                return merged, scratch, regions
            regions -= 1

    def size_fn(counts, point, level, partitioning_accum, bank_size, num_banks):
        groups, scratch, regions = compute_groups(
            counts, point, level, partitioning_accum, bank_size, num_banks
        )
        if groups is None:
            return (float("inf"),) * 3

        if resident_bytes is not None and level == 2:
            # Operands already have whole-tensor allocations. Compute tiles
            # are views, not additional DMA slots. Only reduction workspace
            # adds storage to that provisional placement.
            workspace = regions * (
                math.ceil(scratch / bank_size) * bank_size
                if bank_size
                else scratch
            )
            staging = (
                _alloc_bytes(counts[_FL], fl_bits) if stream_weights else 0
            )
            if bank_size:
                staging = math.ceil(staging / bank_size) * bank_size
            return (resident_bytes, workspace, staging)

        out = [0.0, 0.0, 0.0]
        if not bank_size:
            for size, kind, n, _ in groups:
                out[kind] += n * size
            out[_OF] += regions * scratch
            return tuple(out)

        for size, kind, n, _ in groups:
            out[kind] += n * math.ceil(size / bank_size) * bank_size
        if scratch:
            out[_OF] += regions * math.ceil(scratch / bank_size) * bank_size
        return tuple(out)

    size_fn.compute_groups = compute_groups
    return size_fn


class RuntimeCalculator(MappingTraversal):
    """Runtime cost model for a 4-level hierarchy (PE / L1 / L2 / DRAM).

    A mapping is priced in cycles as its L3 grid sweep.  With a
    double-buffered L2 each step costs the slower of its DRAM transfers and
    its compute, otherwise their sum, and the sweep is framed by the
    un-overlapped first load and last store; a split reduction adds its
    accumulate steps and the tail pass that finishes each output tile.  A
    DRAM transfer costs its bytes at ``dram_bandwidth`` plus one access
    latency, block scales being a transfer of their own, and the batch loop
    outside the mapping shares a weight tile among the elements of one
    group.  Energy is interstellar's cost model, not this one.

    The compute of an L2 block is the busier of the matrix unit and the
    scratchpad bus.  The matrix unit charges the systolic passes, each weight
    tile costing the longer of its loading and the rows streamed through it,
    plus the back-pressure of a single-buffered accumulator against
    ``self.output_slack`` of buffering.  The bus charges the words every operand
    role moves, summed per bank group of the planner's partition with the
    busiest bank setting the pace: input and weight rows in whole beats,
    packed as the toolchain packs them, with the read interface's lost cycle
    whenever an aligned request follows an unaligned one; block scales one
    word each; the output, the fused tail operands, the bias and the
    reduction scratch; and the round trip a stream pays when it changes bank
    on its bank-aligned tile buffer.  A weight tile held across the spatial
    loops is fetched once.  The SpMM unit adds a turnaround per row visit and
    the outlier rows it gathers at the block's K depth, most of them a bank
    switch when the weight tile spans several banks.  A tail pass of its own
    costs the vector unit's lane rate or its bank words, whichever is
    slower.  The ramps of a sweep, the first buffer fill, the systolic skew
    and the last drain, are spread over the ops the sweep dispatches.

    Not modeled are the per-op costs (parameter load, deserialisation, the
    start/done handshake, the drain between uncommitted ops, the host's
    dispatch time), so an op that runs alone pays its ramps in full where
    the spreading charges a share; the cycle-exact datapath, the systolic
    skew being a constant of the array dims; a stream-breaking
    ``quantize_mx`` tail, charged as a staged region rather than a drained
    pass; a conv input tile's halo; more than one port width, every operand
    moving at ``sram_bandwidth`` over one bus per bank; the block scales'
    bank switches; and the outlier density of an individual tile, priced at
    the layer's average.
    """

    def __init__(
        self,
        input_dtype_width: int,
        weight_dtype_width: int,
        output_dtype_width: int,
        accum_dtype_width: int,
        double_buffered_accum_buffer: bool,
        sram_bandwidth: int,
        dram_bandwidth: int,
        dram_access_latency_cycles: float,
        bank_switch_cycles: float,
        output_slack: int,
        spmm_row_cycles: float,
        output_stream_beats: int,
        double_buffered_l2: bool = False,
        outlier_rate: float = 0.0,
        batch: int = 1,
        weight_batch: Optional[int] = None,
        has_tail: bool = False,
        single_k_tail_extra_pass: bool = False,
        split_k_tail_extra_pass: bool = False,
        tail_keeps_shape: bool = False,
        tail_specs=(),
        input_scale_width: int = 0,
        weight_scale_width: int = 0,
        output_scale_width: int = 0,
        scale_block_size: int = 1,
        bias_width: int = 0,
        stride: Tuple[int, int] = (1, 1),
        bank_size: Optional[int] = None,
        weight_transposed: bool = False,
    ):
        self.input_dtype_width = input_dtype_width
        self.weight_dtype_width = weight_dtype_width
        self.output_dtype_width = output_dtype_width
        self.accum_dtype_width = accum_dtype_width
        self.double_buffered_accum_buffer = double_buffered_accum_buffer
        self.sram_bandwidth = sram_bandwidth
        self.dram_bandwidth = dram_bandwidth
        self.dram_access_latency_cycles = dram_access_latency_cycles
        self.double_buffered_l2 = double_buffered_l2
        self.outlier_rate = outlier_rate
        self.batch = batch
        self.weight_batch = batch if weight_batch is None else weight_batch
        self.has_tail = has_tail
        self.single_k_tail_extra_pass = single_k_tail_extra_pass
        self.split_k_tail_extra_pass = split_k_tail_extra_pass
        self.tail_keeps_shape = tail_keeps_shape
        self.tail_specs = tuple(tail_specs)
        self.input_scale_width = input_scale_width
        self.weight_scale_width = weight_scale_width
        self.output_scale_width = output_scale_width
        self.scale_block_size = scale_block_size
        self.bias_width = bias_width
        self.stride = stride
        self.bank_size = bank_size
        self.weight_transposed = weight_transposed
        self.bank_switch_cycles = bank_switch_cycles
        self.output_slack = output_slack
        self.spmm_row_cycles = spmm_row_cycles
        self.output_stream_beats = output_stream_beats
        self.dram_bytes = {}

    def tail_tile_sizes(self, mapping):
        """DRAM bytes each fused tail operand streams for one output tile --
        one transfer apiece.  An operand is tiled along the output dims it is
        not broadcast over, so its tile is the output tile's extent there."""
        blockings = mapping.loop_blockings
        partitionings = mapping.loop_partitionings
        sizes = []
        for dims, bits in self.tail_specs:
            count = 1
            for d in dims:
                count *= blockings[d][1] * blockings[d][2] * partitionings[d][0]
            sizes.append(count * bits / 8)
        return sizes

    def _bank_cycles(self, words, bank_groups):
        """Cycles the busiest scratchpad bank spends moving ``words`` -- bus
        words per operand role -- when the roles sharing a bank
        (``bank_groups``: the search's partition as role sets, ``None`` =
        nothing shares) queue on its single port.  A role the partition does
        not name keeps a port of its own.
        """
        placed = set()
        busiest = 0
        for roles in bank_groups or ():
            busiest = max(busiest, sum(words.get(role, 0) for role in roles))
            placed.update(roles)
        loose = [count for role, count in words.items() if role not in placed]
        return max([busiest, *loose])

    def _request_words(self, requests, row_elems, bits, loop_bound, pitch):
        """Bus words ``requests`` fetches of one ``row_elems``-element row
        take.  A request is served in whole beats, so a row that is not a
        multiple of the port costs more than its bytes.  The controller packs
        the rows that fill whole beats into one request when the L1 loop
        that walks them, of ``loop_bound`` steps, divides into that many
        (the toolchain's ``get_packing_factor``); a ``loop_bound`` of 0 never
        packs.  Consecutive requests are ``pitch`` bytes apart, the row
        pitch of the tile buffer.  The SoC's read interface flushes the tail
        of a request that started inside a beat in a cycle of its own and
        holds the next request's first beat when that one starts on a beat
        boundary, so a pitch that is not a multiple of the beat cycles the
        requests through the beat offsets and costs one cycle per aligned
        request that follows an unaligned one."""
        if not bits or requests <= 0:
            return 0
        row_bits = row_elems * bits
        pf = math.lcm(row_bits, self.sram_bandwidth) // row_bits
        rows_per_request = pf if loop_bound and loop_bound % pf == 0 else 1
        request_words = math.ceil(
            rows_per_request * row_bits / self.sram_bandwidth
        )
        count = math.ceil(requests / rows_per_request)
        beat = self.sram_bandwidth // 8
        misalignment = rows_per_request * pitch % beat
        flushes = 0
        if misalignment:
            flushes = count * math.gcd(misalignment, beat) / beat
        return count * request_words + flushes

    def _bus_words(self, count, bits, bandwidth=None):
        """Bus words ``count`` elements of ``bits`` occupy at the bank's full
        width, or at ``bandwidth`` bytes per cycle."""
        if not bits or count <= 0:
            return 0
        return math.ceil(
            count * bits / 8 / (bandwidth or self.sram_bandwidth / 8)
        )

    def _load_words(self, mapping):
        """Bus words one L1 tile's every-round operands take, per role: the
        input and its block scales, the weight and its scales.  The input
        and the weight arrive one PE-array row per request
        (``_request_words``), packed several rows per request when the L1
        loop that walks them allows it -- never for a transposed weight;
        the weight scales one PE-array row of scales per request, never
        packed.  An input scale is delivered one
        per bus word however narrow it is; every outlier in the input tile
        gathers one more weight row on top of the dense weight tile."""
        ext = lambda loop: self._extent(mapping, loop, 1)
        rows = ext(le.OX) * ext(le.OY) * ext(le.ON)
        depth = ext(le.IC)
        taps = ext(le.FX) * ext(le.FY)
        gathered_rows = rows * depth * self.outlier_rate
        blockings = mapping.loop_blockings
        ic_unroll = mapping.loop_partitionings[le.IC][0]
        oc_unroll = mapping.loop_partitionings[le.OC][0]
        weight_loop = 0 if self.weight_transposed else blockings[le.OC][1]
        # The tile buffers are [rows, IC] for the input and [IC, OC] for the
        # weight and its scales: a request walks one row of each.
        ic3 = self._extent(mapping, le.IC, 2)
        oc3 = self._extent(mapping, le.OC, 2)
        input_pitch = ic3 * self.input_dtype_width // 8
        weight_pitch = oc3 * self.weight_dtype_width // 8
        scale_pitch = oc3 * self.weight_scale_width // 8
        words = {
            "input": self._request_words(
                rows * depth / ic_unroll,
                ic_unroll,
                self.input_dtype_width,
                blockings[le.IC][1],
                input_pitch,
            ),
            "weight": self._request_words(
                (depth * taps + gathered_rows) * ext(le.OC) / oc_unroll,
                oc_unroll,
                self.weight_dtype_width,
                weight_loop,
                weight_pitch,
            ),
            "weight_scale": self._request_words(
                depth * taps / self.scale_block_size * ext(le.OC) / oc_unroll,
                oc_unroll,
                self.weight_scale_width,
                0,
                scale_pitch,
            ),
        }
        if self.input_scale_width:
            words["input_scale"] = math.ceil(
                rows * depth / self.scale_block_size
            )
        return words

    def _tail_words(self, mapping, level):
        """Bus words the tail's operands take for one tile at ``level`` (1 =
        an L1 output tile, 2 = the whole L3 output tile), per role: the
        finished output with its block scales -- each scale leaves in a bus
        word of its own, however narrow, as the input scales arrive -- and
        each fused operand over the output dims it is tiled along.  The tail
        rides on the array's output one vector (a row of ``oc_dim`` values)
        at a time and fetches a fused operand in one unpacked request per
        vector, so an operand narrower than a bus word still costs a word
        per vector."""
        ext = lambda loop: self._extent(mapping, loop, level)
        out = ext(le.OC) * ext(le.OY) * ext(le.OX) * ext(le.ON)
        words = {"output": self._bus_words(out, self.output_dtype_width)}
        if self.output_scale_width:
            words["output"] += math.ceil(out / self.scale_block_size)
        oc_dim = mapping.loop_partitionings[le.OC][0]
        oc2 = self._extent(mapping, le.OC, 2)
        for i, (dims, bits) in enumerate(self.tail_specs):
            if le.OC in dims:
                rows = math.prod(ext(dim) for dim in dims if dim != le.OC)
                vectors = rows * math.ceil(ext(le.OC) / oc_dim)
                words[("fused", i)] = self._request_words(
                    vectors, oc_dim, bits, 0, math.ceil(oc2 * bits / 8)
                )
            else:
                words[("fused", i)] = self._bus_words(
                    math.prod(ext(dim) for dim in dims), bits
                )
        return words

    def _stream_switches(self, mapping, key_loops, walk_of, held_loops=()):
        """Compose L1 bank walks in the emitted L2 loop order.

        ``walk_of(idx)`` returns ``(switches, first_bank, last_bank)`` for
        one L1 request nest. Key loops change its addresses; other loops
        repeat it, unless the controller holds the operand across them.
        Fold repeats at their actual nesting depth, including loops between
        two key loops, so both rewinds and inter-block transitions survive.
        """
        blockings, orders = mapping.loop_blockings, mapping.loop_orders
        nest = sorted(
            (
                i
                for i in range(le.NUM)
                if blockings[i][2] > 1 and i not in held_loops
            ),
            key=lambda i: -orders[i][2],
        )

        def walk(depth, idx):
            if depth == len(nest):
                return walk_of(idx)
            loop = nest[depth]
            count = blockings[loop][2]
            if loop not in key_loops:
                inner, start, end = walk(depth + 1, idx)
                return count * inner + (count - 1) * (start != end), start, end
            total = 0
            first = last = None
            for index in range(count):
                idx[loop] = index
                inner, start, end = walk(depth + 1, idx)
                total += inner + (last is not None and start != last)
                if first is None:
                    first = start
                last = end
            return total, first, last

        return walk(0, {})[0]

    def _bank_switch_cycles(self, mapping):
        """Cycles one L3 step's input and weight streams lose to bank
        switches, per role (``self.bank_switch_cycles`` each).

        Follow the mapping's L1 input order and the weight FY/FX/IC/OC scan.
        Packing combines adjacent channel groups into a request exactly
        when the toolchain permits it, matching _request_words. Buffers
        start on banks; input scales are not included.
        A weight tile held across spatial loops is not refetched by them.
        """
        if not self.bank_size:
            return {}
        b = mapping.loop_blockings
        orders = mapping.loop_orders
        ic3 = self._extent(mapping, le.IC, 2)
        oc3 = self._extent(mapping, le.OC, 2)
        ic1 = self._extent(mapping, le.IC, 1)
        oc1 = self._extent(mapping, le.OC, 1)
        fy, fx = b[le.FY][1], b[le.FX][1]
        oy1, ox1 = b[le.OY][1], b[le.OX][1]
        hs, ws = self.stride
        y_in = (oy1 * b[le.OY][2] - 1) * hs + fy
        x_in = (ox1 * b[le.OX][2] - 1) * ws + fx
        pitch_in = ic3 * self.input_dtype_width / 8
        ic_dim = mapping.loop_partitionings[le.IC][0]
        oc_dim = mapping.loop_partitionings[le.OC][0]

        def packed_width(lanes, bits, count):
            row_bits = lanes * bits
            factor = math.lcm(row_bits, self.sram_bandwidth) // row_bits
            return lanes * (factor if count and count % factor == 0 else 1)

        input_chunk = packed_width(ic_dim, self.input_dtype_width, b[le.IC][1])
        input_order = sorted((le.IC, le.OY, le.OX), key=lambda i: -orders[i][1])

        def input_walk(idx):
            y0 = idx.get(le.OY, 0) * oy1 * hs
            x0 = idx.get(le.OX, 0) * ox1 * ws
            # Filter taps expand the spatial fetch; the controller disables
            # its separate L1 FX/FY/OC loops. Clip the last halo to the tile.
            sy, sx = (hs if fy == 1 else 1), (ws if fx == 1 else 1)
            ny = min(
                oy1 if fy == 1 else oy1 * hs + fy - 1, (y_in - 1 - y0) // sy + 1
            )
            nx = min(
                ox1 if fx == 1 else ox1 * ws + fx - 1, (x_in - 1 - x0) // sx + 1
            )
            width = input_chunk * self.input_dtype_width / 8
            scans = {
                le.IC: (ic1 // input_chunk, width),
                le.OY: (ny, sy * x_in * pitch_in),
                le.OX: (nx, sx * pitch_in),
            }
            loops = tuple(scans[i] for i in input_order)
            offset = (y0 * x_in + x0) * pitch_in
            offset += idx.get(le.IC, 0) * ic1 * self.input_dtype_width / 8
            return strided_bank_walk(loops, width, self.bank_size, offset)

        pitch_w = oc3 * self.weight_dtype_width / 8
        beat_w = oc1 * self.weight_dtype_width / 8
        weight_chunk = packed_width(
            oc_dim,
            self.weight_dtype_width,
            0 if self.weight_transposed else b[le.OC][1],
        )

        def weight_walk(idx):
            c0 = idx.get(le.IC, 0) * ic1
            k_off = idx.get(le.OC, 0) * beat_w
            width = weight_chunk * self.weight_dtype_width / 8
            loops = (
                (fy * fx, ic3 * pitch_w),
                (ic1, pitch_w),
                (oc1 // weight_chunk, width),
            )
            return strided_bank_walk(
                loops, width, self.bank_size, c0 * pitch_w + k_off
            )

        held = ()
        if b[le.IC][2] == 1:
            held = tuple(
                loop
                for loop in (le.OX, le.OY)
                if orders[loop][2] < orders[le.OC][2]
            )
        return {
            "input": self.bank_switch_cycles
            * self._stream_switches(mapping, (le.OX, le.OY, le.IC), input_walk),
            "weight": self.bank_switch_cycles
            * self._stream_switches(mapping, (le.OC, le.IC), weight_walk, held),
        }

    def _tail_bank_switch_cycles(self, mapping, tiled):
        """Fused-operand read latency, charged only on a finishing pass.

        A riding tail uses MatrixOps' filtered L2/L1 output loops. A
        separate vector pass scans the finished output tile in storage
        order. Broadcast dimensions have zero address stride. Multi-beat
        requests overlap part of the bank-switch drain (two beats lose
        seven cycles, versus eight for one), as in pool_bank_switch_cycles.
        """
        if not self.bank_size:
            return {}
        b, orders = mapping.loop_blockings, mapping.loop_orders
        dims_out = (le.ON, le.OY, le.OX, le.OC)
        oc_dim = mapping.loop_partitionings[le.OC][0]
        l1_order = sorted(dims_out, key=lambda i: -orders[i][1])
        result = {}
        for i, (dims, bits) in enumerate(self.tail_specs):
            strides = {}
            pitch = bits / 8
            for dim in reversed(dims_out):
                strides[dim] = pitch if dim in dims else 0
                if dim in dims:
                    pitch *= self._extent(mapping, dim, 2)
            lanes = oc_dim if le.OC in dims else 1
            width = math.ceil(lanes * bits / 8)

            def tile_walk(idx):
                offset = sum(
                    idx.get(d, 0) * self._extent(mapping, d, 1) * strides[d]
                    for d in dims_out
                )
                loops = tuple(
                    (b[d][1], strides[d] * mapping.loop_partitionings[d][0])
                    for d in l1_order
                )
                return strided_bank_walk(loops, width, self.bank_size, offset)

            if tiled:
                switches = self._stream_switches(
                    mapping, dims, tile_walk, (le.IC, le.FX, le.FY)
                )
            else:
                loops = tuple(
                    (
                        b[d][1] * b[d][2],
                        strides[d] * mapping.loop_partitionings[d][0],
                    )
                    for d in dims_out
                )
                switches = strided_bank_walk(loops, width, self.bank_size)[0]
            beats = math.ceil(width * 8 / self.sram_bandwidth)
            result[("fused", i)] = switches * max(
                1, self.bank_switch_cycles + 1 - beats
            )
        return result

    def _tail_stall(self, mapping, bank_groups, words, compute, bank):
        """Cycles the array loses to the tail's burst on a bank that feeds
        one of its every-round buffers.

        Those buffers ping-pong per L1 sweep (``blockings[IC][2]`` sweeps
        to a block), so each sweep's fetch must land inside the sweep before
        it.  The tail's words on such a bank do not spread over the block:
        they arrive together, and the bank's round-robin port lets them
        through at the tail's beats per request round for every grant of a
        stream that is always pending, or in the sweep's free cycles when
        the stream idles between its requests.  The sweeps the burst lands
        on are priced one by one, and what they cost beyond the block's
        price is the stall.  A tail buffer meets the stream's bank only on
        the steps whose ping-pong slots agree: every step when the stream
        is refetched with it, else every other.
        """
        if not bank_groups:
            return 0
        sweeps = mapping.loop_blockings[le.IC][2]
        steps = self._l3_blocks(mapping)
        if sweeps == 1 or steps == 1:
            return 0
        sweep_compute = compute / sweeps
        vectors = 1
        for loop in [le.OC, le.OY, le.OX]:
            vectors *= mapping.loop_blockings[loop][1]
        stream_dims = {
            "input": _IF_DIMS,
            "input_scale": _IF_DIMS,
            "weight": _FL_DIMS,
            "weight_scale": _FL_DIMS,
        }
        stall = 0.0
        for roles in bank_groups:
            streams = [role for role in roles if role in stream_dims]
            tails = []
            for role in roles:
                is_fused = isinstance(role, tuple) and role[0] == "fused"
                if (role == "output" or is_fused) and words.get(role, 0):
                    tails.append(role)
            if not streams or not tails:
                continue
            others = [
                role
                for role in roles
                if role not in streams and role not in tails
            ]
            stream_words = [words.get(role, 0) / sweeps for role in streams]
            busiest = max(stream_words)
            pending = sum(stream_words)
            spread = sum(words.get(role, 0) for role in others) / sweeps
            tail_words = sum(words[role] for role in tails)
            # The output's beats and its scales leave on two requesters,
            # every beat and every scale a request of its own; a fused
            # operand is fetched once per output vector.
            oc_dim = mapping.loop_partitionings[le.OC][0]
            requests = []
            for role in tails:
                if role == "output":
                    out = vectors * oc_dim
                    stores = self._bus_words(out, self.output_dtype_width)
                    requests.append(stores)
                    if self.output_scale_width:
                        requests.append(math.ceil(out / self.scale_block_size))
                elif le.OC in self.tail_specs[role[1]][0]:
                    requests.append(vectors)
                else:
                    requests.append(words[role])
            beats_per_grant = tail_words / max(requests)
            remaining = tail_words
            priced = 0.0
            for sweep in range(sweeps):
                free = max(0.0, sweep_compute - pending - spread)
                landed = min(remaining, max(beats_per_grant * busiest, free))
                if sweep == sweeps - 1:
                    landed = remaining
                remaining -= landed
                priced += max(sweep_compute, pending + spread + landed)
            refetched = [
                self._l3_loads(mapping, stream_dims[role]) == steps
                for role in streams
            ]
            fraction = 1.0 if any(refetched) else 0.5
            excess = max(0.0, priced - max(compute, bank))
            stall = max(stall, fraction * excess)
        return stall

    def matrix_cycles(self, mapping, bank_groups):
        """Cycles of one L3 grid step: the L2 sweep of weight-reuse tiles,
        each costing the busier of the matrix unit -- its systolic passes,
        plus the back-pressure a single-buffered accumulator takes while the
        tail drains each finished tile -- and the scratchpad bank with the
        most to move for it -- the every-round operands of its L1 sub-tiles,
        the accumulator read back and rewritten while the reduction is
        split, the finished tile and the tail's operands when it is not,
        the round trips a stream idles for when it changes bank, each
        summed with whatever shares its bank -- plus the stall the tail's
        burst inflicts where it shares a bank with one of the array's
        every-round buffers (``_tail_stall``) -- and, for an outlier GEMM,
        the SpMM unit, which must deliver the block's sparse correction
        before the vector pipeline releases any of its rows: per 64-column
        pass it walks every row of the block, paying ``self.spmm_row_cycles`` of
        turnaround plus that row's outliers, each gather a bank switch when
        the weight tile spans several banks -- plus the once-per-sweep
        overhead (buffer fill, systolic skew, the last parked tile's drain)
        spread over the ops a double-buffered L2 overlaps it with.  Also
        the reporting model's per-tile utilization denominator.

        Args:
            mapping: The interstellar mapping to price.
            bank_groups: Its bank partition as role sets (``bank_partition``),
                or ``None`` when nothing shares a bank.
        """
        blockings = mapping.loop_blockings
        orders = mapping.loop_orders
        partitionings = mapping.loop_partitionings

        # --- L1: weight-reuse tile timing ---
        sa_weight_loading_time = partitionings[le.IC][0]

        first_non_ox_oy_index = 6
        for i in range(le.NUM):
            if i == le.OX or i == le.OY:
                continue
            if orders[i][1] < first_non_ox_oy_index:
                first_non_ox_oy_index = orders[i][1]

        weight_reuse_tile_size = 1
        for i in range(le.NUM):
            if orders[i][1] < first_non_ox_oy_index:
                weight_reuse_tile_size *= blockings[i][1]
        weight_reuse_tile_time = max(
            sa_weight_loading_time, weight_reuse_tile_size
        )

        num_remaining_l1_tiles = 1
        for i in range(le.NUM):
            if orders[i][1] >= first_non_ox_oy_index:
                num_remaining_l1_tiles *= blockings[i][1]
        num_remaining_l1_tiles *= blockings[le.IC][2]
        computation_l1_time = weight_reuse_tile_time * num_remaining_l1_tiles

        # --- the finished L1 output tile: its vectors, and the bus beats the
        # tail spends on each -- storing it, and reading a fused operand
        # alongside when one is wider than the port ---
        num_k = blockings[le.IC][3]
        output_size = 1
        for loop in [le.OC, le.OY, le.OX]:
            output_size *= blockings[loop][1]
        oc_dim = partitionings[le.OC][0]
        output_width = (
            self.accum_dtype_width if num_k > 1 else self.output_dtype_width
        )
        store_cycles = math.ceil(output_width * oc_dim / self.sram_bandwidth)
        if num_k == 1 and self.output_scale_width:
            # The block scales leave on a requester of their own, one beat
            # per vector, on the output's bank.
            store_cycles += 1
        vector_beats = max(store_cycles, self.output_stream_beats)
        if num_k == 1 and not self.single_k_tail_extra_pass:
            for dims, bits in self.tail_specs:
                vector_beats = max(
                    vector_beats,
                    math.ceil(
                        bits
                        * (oc_dim if le.OC in dims else 1)
                        / self.sram_bandwidth
                    ),
                )

        # Without a bank to park the finished tile in, its vectors leave the
        # array one per step during the last reduction pass of each weight
        # tile -- as one burst of the whole tile when the OC passes are
        # adjacent (no filter loops at L1), else as OC1 bursts of one spatial
        # tile -- and the tail drains them at ``vector_beats`` apiece.  The
        # path absorbs ``self.output_slack`` of them; past that the array runs at
        # the tail's pace for the rest of the burst.  A double-buffered
        # accumulator parks a tile whose tail moves more than a bus word per
        # vector on some port (the toolchain's ``should_use_direct_path``);
        # a narrower tail rides the pass and pays like a single-buffered one.
        parked = self.double_buffered_accum_buffer
        if parked:
            widths = [output_width * oc_dim]
            for dims, bits in self.tail_specs:
                widths.append(bits * (oc_dim if le.OC in dims else 1))
            parked = max(widths) > self.sram_bandwidth
        if not parked:
            burst_vectors = weight_reuse_tile_size
            burst_cycles = weight_reuse_tile_time
            bursts = blockings[le.OC][1]
            if blockings[le.FX][1] * blockings[le.FY][1] == 1:
                burst_vectors *= bursts
                burst_cycles *= bursts
                bursts = 1
            computation_l1_time += bursts * max(
                0,
                burst_vectors * vector_beats - burst_cycles - self.output_slack,
            )

        # --- L2: outer spatial-tile loop ---
        l2_blocks = 1
        for i in range(le.NUM):
            if i != le.IC:
                l2_blocks *= blockings[i][2]

        # --- bus traffic of one L2 output block: the loads of its L1
        # sub-tiles, then what the vector unit moves for the block itself ---
        loads = self._load_words(mapping)
        words = {
            role: count * blockings[le.IC][2] for role, count in loads.items()
        }
        # With the whole reduction inside the block, a weight tile whose L2
        # loop is outside the spatial ones is fetched once and held across
        # them (the input is refetched every block), so its words are spread
        # over the blocks that reuse it.
        if blockings[le.IC][2] == 1:
            held = 1
            for loop in [le.OX, le.OY]:
                if orders[loop][2] < orders[le.OC][2]:
                    held *= blockings[loop][2]
            words["weight"] /= held
            words["weight_scale"] /= held
        # The bias is read once per output tile: spread over its rounds.
        words["bias"] = (
            self._bus_words(self._extent(mapping, le.OC, 1), self.bias_width)
            / num_k
        )
        output_elems = 1
        for loop in [le.OC, le.OY, le.OX, le.ON]:
            output_elems *= self._extent(mapping, loop, 1)
        if num_k > 1:
            # A split reduction reads the running partial back through the
            # same single-ported bank it writes the new one to.
            words["scratch"] = 2 * self._bus_words(
                output_elems, self.accum_dtype_width
            )
        elif self.single_k_tail_extra_pass and not self.tail_keeps_shape:
            # A staged single round parks the finished tile in scratch for
            # the tail's own pass (``vector_cycles``) to read back.
            words["scratch"] = self._bus_words(
                output_elems, self.output_dtype_width
            )
        elif self.single_k_tail_extra_pass:
            # An in-place pass reads the tile back from its output slot and
            # rewrites it: two bank visits beyond a riding tail's.
            words.update(self._tail_words(mapping, 1))
            words["output"] += 2 * self._bus_words(
                output_elems, self.output_dtype_width
            )
        else:
            words.update(self._tail_words(mapping, 1))
        # A stream that changes bank idles its port for a round trip each
        # time; spread the step's switches over its output blocks.
        switches = self._bank_switch_cycles(mapping)
        if num_k == 1 and (
            not self.single_k_tail_extra_pass or self.tail_keeps_shape
        ):
            switches.update(
                self._tail_bank_switch_cycles(
                    mapping, tiled=not self.single_k_tail_extra_pass
                )
            )
        for role, cycles in switches.items():
            words[role] += cycles / l2_blocks
        # The matrix unit and the vector unit are pipelined -- one drains a
        # tile while the other computes the next -- so a block costs the
        # busier of the two.
        bank = self._bank_cycles(words, bank_groups)
        block_time = max(computation_l1_time, bank)
        # The tail's words do not spread over the block: they land as a
        # burst on a few of its sweeps, and on a bank that feeds one of the
        # array's ping-pong buffers they can outrun the sweeps they land on.
        block_time += self._tail_stall(
            mapping, bank_groups, words, computation_l1_time, bank
        )

        # The SpMM unit runs the block alongside and the vector pipeline
        # waits for its correction on every output vector, so a block also
        # costs its pace: per PE-array-wide pass, every row's turnaround
        # plus its gathered weight rows.
        if self.outlier_rate:
            rows = 1
            for loop in [le.OX, le.OY, le.ON]:
                rows *= self._extent(mapping, loop, 1)
            k_block = self._extent(mapping, le.IC, 2)
            passes = blockings[le.OC][1]
            visits = passes * rows
            gathers = visits * k_block * self.outlier_rate
            # Gathers hit random K rows of the weight tile; when it spans
            # several banks most consecutive gathers change bank and pay the
            # read path's round trip, as the streams above do.
            switch = 0.0
            if self.bank_size:
                weight_tile_bytes = (
                    self._extent(mapping, le.OC, 2)
                    * k_block
                    * blockings[le.FX][1]
                    * blockings[le.FY][1]
                    * self.weight_dtype_width
                    / 8
                )
                banks = max(1, math.ceil(weight_tile_bytes / self.bank_size))
                switch = self.bank_switch_cycles * (1 - 1 / banks)
            spmm_block_time = visits * self.spmm_row_cycles + gathers * (
                1 + switch
            )
            block_time = max(block_time, spmm_block_time)

        # The first tile's loads overlap nothing; the last parked tile's drain
        # is a whole vector pass, while a single-buffered accumulator's drain
        # is already in its blocks' own time.
        buffer_fill = self._bank_cycles(loads, bank_groups)
        skew = partitionings[le.IC][0] + partitionings[le.OC][0] - 2
        drain = output_size * vector_beats if parked else 0
        overhead = buffer_fill + skew + drain
        steady = l2_blocks * block_time

        if not self.double_buffered_l2:
            return steady + overhead

        # Every op the sweep dispatches -- the L3 steps and the batch elements
        # looped outside the mapping -- overlaps the ramps of its neighbours.
        steps = self._l3_blocks(mapping) if num_k == 1 else num_k
        return steady + overhead / (steps * self.batch)

    def vector_cycles(self, mapping, bank_groups):
        """Vector-unit cycles to finish one L3 output tile.  Charged on the grid
        step that ends a K sweep, after that step's accumulation.

        The busier of the unit's own rate -- one lane group per cycle, sized
        by the widest element the tail touches (the partial sum it reads, not
        the narrower value a ``quantize_mx`` writes) at ``dram_bandwidth``,
        which is what ``vector_op_utilization`` charges for the same tail in
        the reporting model -- and the busiest bank: the accumulator read
        back, the finished tile and its scales written, each tail operand
        read, summed wherever the partition puts them together.
        """
        blockings = mapping.loop_blockings
        output_size = 1
        for loop in [le.OC, le.OY, le.OX]:
            output_size *= blockings[loop][1] * blockings[loop][2]
        oc_dim = mapping.loop_partitionings[le.OC][0]
        widths = [self.output_dtype_width, self.accum_dtype_width]
        widths += [bits for _, bits in self.tail_specs]
        lane_bytes = max(widths) / 8 * oc_dim
        lanes = output_size * math.ceil(lane_bytes / self.dram_bandwidth)
        words = self._tail_words(mapping, 2)
        for role, cycles in self._tail_bank_switch_cycles(
            mapping, tiled=False
        ).items():
            words[role] += cycles
        if blockings[le.IC][3] > 1 or (
            self.single_k_tail_extra_pass and not self.tail_keeps_shape
        ):
            words["scratch"] = self._bus_words(
                output_size * oc_dim, self.accum_dtype_width
            )
        elif self.single_k_tail_extra_pass:
            # The in-place pass reads the finished tile from its output slot.
            words["output"] += self._bus_words(
                output_size * oc_dim, self.output_dtype_width
            )
        return max(lanes, self._bank_cycles(words, bank_groups))

    def calculate_runtime(self, architecture, layer, mapping):
        blockings = mapping.loop_blockings
        partitionings = mapping.loop_partitionings

        # Elements of one L3 tile: levels 0-2 only, since [3] is the grid trip
        # count, not part of the tile.
        input_elems = (
            partitionings[le.IC][0]
            * blockings[le.IC][1]
            * blockings[le.IC][2]
            * blockings[le.OY][1]
            * blockings[le.OY][2]
            * blockings[le.OX][1]
            * blockings[le.OX][2]
        )
        if getattr(self, "resident_input_hw", None) is not None:
            # Native padded resident invocations load the actual unpadded
            # image, which may differ from the output extent (e.g. stride 2).
            input_elems = (
                partitionings[le.IC][0]
                * blockings[le.IC][1]
                * blockings[le.IC][2]
                * math.prod(self.resident_input_hw)
            )
        weight_elems = (
            partitionings[le.IC][0]
            * blockings[le.IC][1]
            * blockings[le.IC][2]
            * partitionings[le.OC][0]
            * blockings[le.OC][1]
            * blockings[le.OC][2]
            * blockings[le.FY][1]
            * blockings[le.FX][1]
        )
        output_elems = (
            partitionings[le.OC][0]
            * blockings[le.OC][1]
            * blockings[le.OC][2]
            * blockings[le.OY][1]
            * blockings[le.OY][2]
            * blockings[le.OX][1]
            * blockings[le.OX][2]
        )

        lat = self.dram_access_latency_cycles

        def transfer(*sizes):
            """Cycles to move each of ``sizes`` as its own DMA: one fixed
            access latency apiece plus the bytes.  A microscaling operand's
            block scales are such a DMA -- a few hundred bytes, a whole
            latency."""
            sizes = [s for s in sizes if s]
            return len(sizes) * lat + sum(sizes) / self.dram_bandwidth

        input_sizes = (
            input_elems * self.input_dtype_width / 8,
            input_elems / self.scale_block_size * self.input_scale_width / 8,
        )
        weight_sizes = (
            weight_elems * self.weight_dtype_width / 8,
            weight_elems / self.scale_block_size * self.weight_scale_width / 8,
        )
        output_sizes = (
            output_elems * self.output_dtype_width / 8,
            output_elems / self.scale_block_size * self.output_scale_width / 8,
        )
        input_load = transfer(*input_sizes)
        weight_load = transfer(*weight_sizes)
        store = transfer(*output_sizes)

        # A tail operand spans output dims alone, so its count already runs
        # over the output steps -- the only ones that read it.
        tail_sizes = self.tail_tile_sizes(mapping)
        tail_dmas = [
            (
                transfer(size),
                self._batch_loads(
                    mapping, dims, self.batch if le.ON in dims else 1
                ),
            )
            for (dims, _), size in zip(self.tail_specs, tail_sizes)
        ]

        bank_groups, scratch_slots = bank_partition(
            architecture, layer.size_fn, layer, mapping
        )
        matrix_cycles = self.matrix_cycles(mapping, bank_groups)
        vector_cycles = (
            self.vector_cycles(mapping, bank_groups) if self.has_tail else 0
        )

        input_steps = self._batch_loads(mapping, _IF_DIMS, self.batch)
        weight_steps = self._batch_loads(mapping, _FL_DIMS, self.weight_batch)

        # The mapping covers one batch element; the builder loops the rest.
        l3_blocks = self._l3_blocks(mapping) * self.batch
        num_k = blockings[le.IC][3]
        output_tiles = l3_blocks // num_k

        # Traffic the sweep moves, for a caller ranking by DRAM rather than by
        # time, and to check the reuse counts against a profile.  Every
        # candidate mapping is priced through here, so it describes the last
        # one scored -- read it straight after the call that priced the mapping
        # in question.
        self.dram_bytes = {
            "input": input_steps * sum(input_sizes),
            "weight": weight_steps * sum(weight_sizes),
            "output": output_tiles * sum(output_sizes),
            "tail": sum(t * s for (_, t), s in zip(tail_dmas, tail_sizes)),
        }

        if getattr(self, "resident", False):
            self.dram_bytes = dict.fromkeys(self.dram_bytes, 0)
            boundary = getattr(self, "resident_boundary", (False,) * 5)
            # A retained window loads on its first visit, not every time an
            # outer output-channel/batch loop revisits it. The byte footprint
            # stays pinned; these costs choose the boundary's compute grid.
            incoming = [
                s for s, enabled in zip(input_sizes, boundary[:2]) if enabled
            ]
            weights = [
                s for s, enabled in zip(weight_sizes, boundary[2:4]) if enabled
            ]
            unique_input = self.batch * math.prod(
                blockings[d][3] for d in _IF_DIMS
            )
            unique_weight = self.weight_batch * math.prod(
                blockings[d][3] for d in _FL_DIMS
            )
            self.dram_bytes["input"] = unique_input * sum(incoming)
            self.dram_bytes["weight"] = unique_weight * sum(weights)
            self.dram_bytes["output"] = (
                output_tiles * sum(output_sizes) if boundary[4] else 0
            )
            step = matrix_cycles
            if num_k > 1 or self.single_k_tail_extra_pass:
                step += vector_cycles / num_k
            # Rank the grid with exposed first loads/last stores and overlap.
            # Reuse and reduction phases are averaged for this search score;
            # the emitted loop's exact timing comes from estimate_schedule.
            total = _sweep_cycles(
                [
                    (store if boundary[4] else 0, output_tiles),
                    (transfer(*incoming), unique_input),
                    (transfer(*weights), unique_weight),
                ],
                l3_blocks,
                step,
            )
            if getattr(self, "stream_weights", False):
                # The bounded large-weight retry still has one staging slot.
                total += weight_steps * transfer(weight_sizes[0])
                self.dram_bytes["weight"] += weight_steps * weight_sizes[0]
            return total

        dmas = [
            (store, output_tiles),
            (input_load, input_steps),
            (weight_load, weight_steps),
            *tail_dmas,
        ]

        if not self.double_buffered_l2:
            total_time = l3_blocks * matrix_cycles + sum(t * c for c, t in dmas)
            if num_k > 1 or self.single_k_tail_extra_pass:
                total_time += output_tiles * vector_cycles
            return total_time

        if num_k == 1:
            # Every step finishes a tile: one schedule covers the sweep.  A
            # riding tail drains inside the matrix pass, and an in-place one
            # overlaps the next tile's (its bank words are in the block);
            # a staged one is a pass of its own, serial on the single
            # scratch region.
            step = matrix_cycles
            if self.single_k_tail_extra_pass and not self.tail_keeps_shape:
                step += vector_cycles
            return _sweep_cycles(dmas, l3_blocks, step)

        load = input_load + weight_load
        # The sweep's last step has no tile after it to prefetch, so it costs
        # compute alone: hold it out of the count and let the epilogue charge
        # it, with the tail and store that drain behind it.
        accum_steps = l3_blocks - 2 * (output_tiles - 1) - 1
        classes = _step_classes(tail_dmas, output_tiles)
        # Hold out the first output step in the same way -- the prologue fetches
        # its tail, with nothing running yet to hide it behind.  Taking what
        # that step owed leaves every tail fetch counted exactly once.
        first_tail = classes[-1][0]
        classes[-1] = (classes[-1][0], classes[-1][1] - 1)

        total_time = load + first_tail + accum_steps * max(load, matrix_cycles)
        # One window per remaining tile, spanning two grid steps: the matrix
        # unit finishes this tile and starts the next, while DRAM fits that
        # tile's tail read, its store and the next prefetch into the same span.
        # The busier side sets the price, and only the tail differs from one
        # window to the next -- hence one price per class.
        for tail, count in classes:
            prefetch = load + tail
            if self.split_k_tail_extra_pass and scratch_slots == 1:
                # The bare pass holds the control stream through both the
                # matrix and the vector pass, so the window's loads are
                # issued only then and nothing hides them.
                total_time += count * (
                    matrix_cycles + vector_cycles + prefetch + matrix_cycles
                )
                continue
            compute = max(matrix_cycles, prefetch) + matrix_cycles
            dma = max(matrix_cycles + vector_cycles, prefetch) + store + load
            total_time += count * max(compute, dma)
        total_time += matrix_cycles + vector_cycles + store
        return total_time


def bank_partition(architecture, size_fn, layer, mapping):
    """The role partition ``mapping``'s L2 fit is checked with.

    Rebuilds interstellar's scratchpad-level ``size_fn`` invocation
    (``cost_model.get_block_size``): the element counts from the blocking /
    partitioning products through L2, the bank geometry from the
    architecture -- and replays ``size_fn``'s own group construction, so the
    partition is exactly the one the fit check priced.  The runtime model
    prices every candidate mapping through it, and the winner's partition is
    stamped for the memory planner and the reporting model.

    Returns:
        ``(partition, scratch_slots)``: a list of role sets, one per bank
        group -- plus a ``{"scratch"}`` entry when the search charged the
        reduction scratch its own regions -- and how many regions it
        charged, the slot count the scratch is allocated with.  The
        partition is ``None`` when the architecture has no banked level or
        there is no ``size_fn`` (nothing checked the fit, so nothing shares
        a bank).
    """
    if size_fn is None:
        return None, 1
    level = 2
    buf = architecture.buffer(level)
    bank_size = buf.bank_size
    if not bank_size:
        return None, 1
    capacity = buf.capacity
    if isinstance(capacity, list):
        capacity = capacity[0]
    num_banks = capacity // bank_size

    blocking_accum = []
    partitioning_accum = []
    for i in range(le.NUM):
        blocking_accum.append(math.prod(mapping.loop_blocking(i)[: level + 1]))
        partitioning_accum.append(
            math.prod(mapping.loop_partitioning(i)[: level + 1])
        )
    partitioning = list(zip(*mapping.loop_partitionings))[level]

    from interstellar import cost_model

    counts = (
        cost_model.get_if_size(
            blocking_accum, partitioning_accum, partitioning, layer
        ),
        cost_model.get_of_size(
            blocking_accum, partitioning_accum, partitioning
        ),
        cost_model.get_fl_size(
            blocking_accum, partitioning_accum, partitioning
        ),
    )
    groups, scratch, regions = size_fn.compute_groups(
        counts, mapping, level, partitioning_accum, bank_size, num_banks
    )
    if groups is None:
        return None, 1
    partition = [roles for _, _, _, roles in groups]
    if scratch:
        partition.append({"scratch"})
    return partition, regions

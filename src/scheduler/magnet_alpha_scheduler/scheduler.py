# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from scheduler.types import (
    NUM_LOOPS,
    GemmWorkload,
    LevelAccessCounts,
    LocalScheduleResult,
    Loop,
    MappingPoint,
    OperationWorkload,
)

if TYPE_CHECKING:
    from voyager_compiler.codegen.target_backends.magnet_alpha.hardware import (
        MagnetAlphaHardwareConfig,
    )


MAGNET_ALPHA_RULE_BASED_SCHEDULER = "magnet_alpha_rule_based"


class MagnetAlphaSchedulingError(ValueError):
    pass


def mapping_point_to_alpha_factors(
    mapping: MappingPoint,
) -> dict[str, int]:
    """Export a canonical mapping using Alpha's cumulative factor form."""
    blockings = mapping.loop_blockings
    partitionings = mapping.loop_partitionings
    if mapping.num_levels != 3:
        raise ValueError("Alpha GEMM projection requires a 3-level mapping")
    return {
        "c0": partitionings[Loop.IC][0],
        "c1": blockings[Loop.IC][1],
        "c2": blockings[Loop.IC][2],
        "c3": 1,
        "k0": partitionings[Loop.OC][0],
        "k1": blockings[Loop.OC][1],
        "k2": blockings[Loop.OC][2],
        "k3": 1,
        "p0": blockings[Loop.OY][1],
        "p1": blockings[Loop.OY][1],
        "p2": blockings[Loop.OY][2],
        "p2s": 1,
        "p3": 1,
        "q0": blockings[Loop.OX][1],
        "q1": blockings[Loop.OX][1],
        "q2": blockings[Loop.OX][2],
        "q2s": 1,
        "q3": 1,
        "r": blockings[Loop.FY][1],
        "s": blockings[Loop.FX][1],
        "batch_size": blockings[Loop.ON][2],
    }


def _encode_loop_orders(
    loop_blockings,
    loop_partitionings,
    preferred_orders,
    forced_unit_loops=(),
) -> tuple[tuple[int, ...], ...]:
    """Encode Magnet Alpha loop sequences into canonical per-loop order rows."""
    blockings = tuple(tuple(row) for row in loop_blockings)
    partitionings = tuple(tuple(row) for row in loop_partitionings)
    if len(blockings) != NUM_LOOPS or len(partitionings) != NUM_LOOPS:
        raise ValueError("loop factors must contain one row per canonical loop")
    num_levels = len(blockings[0])
    if any(len(row) != num_levels for row in blockings) or any(
        len(row) != num_levels for row in partitionings
    ):
        raise ValueError("loop factor rows must have equal length")
    preferred_orders = tuple(tuple(order) for order in preferred_orders)
    if len(preferred_orders) != num_levels:
        raise ValueError("preferred loop orders must match mapping levels")
    if not forced_unit_loops:
        forced_unit_loops = (frozenset(),) * num_levels
    forced_unit_loops = tuple(frozenset(loops) for loops in forced_unit_loops)
    if len(forced_unit_loops) != num_levels:
        raise ValueError("forced unit loops must match mapping levels")

    level_orders = []
    for level, preferred_order in enumerate(preferred_orders):
        preferred_order = tuple(Loop(loop) for loop in preferred_order)
        if len(set(preferred_order)) != len(preferred_order):
            raise ValueError(
                f"preferred loop order at level {level} has duplicates"
            )
        active = {
            Loop(loop)
            for loop in range(NUM_LOOPS)
            if blockings[loop][level] != 1 or partitionings[loop][level] != 1
        } | {Loop(loop) for loop in forced_unit_loops[level]}
        missing = active - set(preferred_order)
        if missing:
            names = ", ".join(sorted(loop.name for loop in missing))
            raise ValueError(
                f"preferred loop order at level {level} omits {names}"
            )
        encoded = [NUM_LOOPS - 1] * NUM_LOOPS
        for order, loop in enumerate(
            loop for loop in preferred_order if loop in active
        ):
            encoded[loop] = order
        level_orders.append(encoded)

    return tuple(
        tuple(level[loop] for level in level_orders)
        for loop in range(NUM_LOOPS)
    )


def project_gemm_layer(
    layer: GemmWorkload,
    hardware_config,
    input_dtype_width,
) -> GemmWorkload:
    """Project software-resident GEMM channels onto physical matrix lanes.

    Graph padding first creates legal resident channel extents. Alpha can
    issue an aligned half-vector of INT8 output channels, while its dedicated
    image-input path can mask a declared partial input vector. The returned
    layer describes physical issued work. The resident layer, physical layer,
    and hardware lanes contain the complete padding relationship.
    """
    input_lanes = hardware_config.input_lanes(input_dtype_width)
    output_lanes = hardware_config.output_lanes(input_dtype_width)
    resident_c = layer.nifm
    resident_k = layer.nofm

    if resident_c % input_lanes:
        if not hardware_config.supports_partial_matrix_input(resident_c):
            raise MagnetAlphaSchedulingError(
                f"resident IC={resident_c} is not divisible by "
                f"IC0={input_lanes}"
            )
        physical_c = math.ceil(resident_c / input_lanes) * input_lanes
    else:
        physical_c = resident_c

    if resident_k % output_lanes:
        if not hardware_config.supports_partial_matrix_output(
            resident_k,
            input_dtype_width,
        ):
            raise MagnetAlphaSchedulingError(
                f"resident OC={resident_k} is neither divisible by "
                f"OC0={output_lanes} nor a supported partial output"
            )
        physical_k = math.ceil(resident_k / output_lanes) * output_lanes
    else:
        physical_k = resident_k

    if physical_c == resident_c and physical_k == resident_k:
        physical_layer = layer
    else:
        physical_layer = replace(
            layer,
            nifm=physical_c,
            nofm=physical_k,
        )
    return physical_layer


@dataclass(frozen=True)
class MagnetAlphaNmpWorkload:
    """Validated inputs for constructing one Alpha NMP mapping."""

    p1: int
    q1: int
    c1: int
    k1: int

    def __post_init__(self):
        if min(self.p1, self.q1, self.c1, self.k1) <= 0:
            raise ValueError("NMP traversal factors must be positive")


@dataclass(frozen=True)
class _MagnetAlphaNmpDecision:
    """One classified NMP operation and its validated traversal."""

    operation: str
    traversal: MagnetAlphaNmpWorkload


def _divisors_desc(value: int) -> list[int]:
    divisors = []
    for candidate in range(1, math.isqrt(value) + 1):
        if value % candidate != 0:
            continue
        divisors.append(candidate)
        paired = value // candidate
        if paired != candidate:
            divisors.append(paired)
    return sorted(divisors, reverse=True)


def _choose_local_factors(
    layer,
    hardware_config,
    input_lanes,
    output_lanes,
    q_alignment=1,
):
    collector_depth = hardware_config.output_collector.depth_words
    channel_groups = layer.nifm // input_lanes
    output_groups = layer.nofm // output_lanes
    input_depth = hardware_config.local_input_depth

    # C1/K1 describe one streamed Alpha call, not resident local-weight
    # capacity. Keep both complete and choose only the spatial traversal here.
    for q1 in _divisors_desc(layer.wofm):
        if q1 % q_alignment:
            continue
        if q1 > collector_depth:
            continue
        q2 = layer.wofm // q1
        can_flatten_p = (
            layer.wfil == 1
            and layer.hfil == 1
            and layer.wstd == 1
            and layer.hstd == 1
            and q2 == 1
        )
        p_candidates = _divisors_desc(layer.hofm) if can_flatten_p else [1]
        for p1 in p_candidates:
            if p1 * q1 > collector_depth:
                continue
            input_width = (q1 - 1) * layer.wstd + layer.wfil
            input_height = (p1 - 1) * layer.hstd + layer.hfil
            spatial_input_words = input_width * input_height
            if spatial_input_words * channel_groups <= input_depth:
                return p1, q1, channel_groups, output_groups

    raise MagnetAlphaSchedulingError(
        "No Magnet Alpha streamed call fits the input-buffer and collector "
        "constraints"
    )


_MATRIX_PREFERRED_ORDERS = (
    (Loop.OC, Loop.IC, Loop.FX, Loop.FY, Loop.OX, Loop.OY, Loop.ON),
    (Loop.OX, Loop.OY, Loop.IC, Loop.FX, Loop.FY, Loop.OC, Loop.ON),
    (Loop.IC, Loop.OX, Loop.OY, Loop.OC, Loop.ON, Loop.FX, Loop.FY),
)


def _validate_coverage(layer, blockings, partitionings):
    for loop, extent in enumerate(layer.sizes):
        covered = math.prod(blockings[loop]) * math.prod(partitionings[loop])
        if covered != extent:
            loop_name = Loop(loop).name
            raise MagnetAlphaSchedulingError(
                f"{loop_name} covers {covered}, expected {extent}"
            )


def schedule(
    layer,
    hardware_config: MagnetAlphaHardwareConfig,
    input_dtype_width: int,
    *,
    q_alignment: int = 1,
):
    """Build a deterministic Alpha-compatible three-level mapping.

    The local spatial tile is shared by the activation-buffer and collector
    views, so this first policy enforces ``P0=P1`` and ``Q0=Q1``. Remaining
    spatial and channel traversal is placed at the global-memory level.
    """
    if isinstance(q_alignment, bool):
        raise MagnetAlphaSchedulingError(
            "Q alignment must be a positive integer"
        )
    normalized_q_alignment = int(q_alignment)
    if normalized_q_alignment <= 0 or normalized_q_alignment != q_alignment:
        raise MagnetAlphaSchedulingError(
            "Q alignment must be a positive integer"
        )
    q_alignment = normalized_q_alignment

    input_lanes = hardware_config.input_lanes(input_dtype_width)
    output_lanes = hardware_config.output_lanes(input_dtype_width)

    input_word_bits = hardware_config.pe.input_word_bits
    if (
        input_word_bits is not None
        and input_lanes * input_dtype_width != input_word_bits
    ):
        raise MagnetAlphaSchedulingError(
            "Input lanes and dtype do not fill one PE input word"
        )
    if layer.nifm % input_lanes != 0:
        raise MagnetAlphaSchedulingError(
            f"IC={layer.nifm} is not divisible by IC0={input_lanes}"
        )
    if layer.nofm % output_lanes != 0:
        raise MagnetAlphaSchedulingError(
            f"OC={layer.nofm} is not divisible by OC0={output_lanes}"
        )

    p1, q1, c1, k1 = _choose_local_factors(
        layer,
        hardware_config,
        input_lanes,
        output_lanes,
        q_alignment,
    )
    c_groups = layer.nifm // input_lanes
    k_groups = layer.nofm // output_lanes
    c2 = c_groups // c1
    k2 = k_groups // k1
    p2 = layer.hofm // p1
    q2 = layer.wofm // q1

    blockings = [[1, 1, 1] for _ in range(NUM_LOOPS)]
    partitionings = [[1, 1, 1] for _ in range(NUM_LOOPS)]
    partitionings[Loop.IC][0] = input_lanes
    partitionings[Loop.OC][0] = output_lanes

    blockings[Loop.FX][1] = layer.wfil
    blockings[Loop.FY][1] = layer.hfil
    blockings[Loop.OX][1] = q1
    blockings[Loop.OY][1] = p1
    blockings[Loop.OC][1] = k1
    blockings[Loop.IC][1] = c1

    blockings[Loop.OX][2] = q2
    blockings[Loop.OY][2] = p2
    blockings[Loop.OC][2] = k2
    blockings[Loop.IC][2] = c2
    blockings[Loop.ON][2] = layer.nimg

    blockings = [tuple(values) for values in blockings]
    partitionings = [tuple(values) for values in partitionings]
    _validate_coverage(layer, blockings, partitionings)
    orders = _encode_loop_orders(
        blockings,
        partitionings,
        _MATRIX_PREFERRED_ORDERS,
    )
    mapping = MappingPoint(
        orders,
        blockings,
        partitionings,
    )
    return mapping


_ALPHA_UINT16_MAX = (1 << 16) - 1
_ALPHA_UINT32_MAX = (1 << 32) - 1
_ALPHA_INT16_MAX = (1 << 15) - 1
_ALPHA_WEIGHT_DIMENSION_MAX_EXTENT = 1 << 8
_ALPHA_BIAS_SCALE_GROUPS = 8


def _require_single_call_field_fit(name: str, value: int, maximum: int) -> None:
    if value <= 0 or value > maximum:
        raise MagnetAlphaSchedulingError(
            f"Alpha GEMM {name}={value} exceeds the encoded range "
            f"1..{maximum}"
        )


def _validate_single_call_mapping(
    mapping: MappingPoint,
    hardware_config: MagnetAlphaHardwareConfig,
    input_dtype_width: int,
) -> None:
    """Reject mappings not representable as one encoded Alpha GEMM call."""
    factors = mapping_point_to_alpha_factors(mapping)
    illegal = {
        name: factors[name]
        for name in ("c2", "c3", "k2", "k3")
        if factors[name] != 1
    }
    if illegal:
        raise MagnetAlphaSchedulingError(
            "Alpha requires C2=C3=K2=K3=1 for one "
            f"GEMM call; found {illegal}"
        )
    p0 = factors["p0"]
    p1 = factors["p1"]
    p2 = factors["p2"]
    q0 = factors["q0"]
    q1 = factors["q1"]
    q2 = factors["q2"]
    c1 = factors["c1"]
    k1 = factors["k1"]
    batch = factors["batch_size"]
    kernel_channel_words = factors["r"] * factors["s"] * c1

    # GenJsonConfig emits one PE configuration per batch item rather than
    # representing N in the per-PE traversal counters. This interface has one PE,
    # and the compiler does not yet split N into separate executable calls,
    # so only batch one is a complete one-call mapping.
    if batch != 1:
        raise MagnetAlphaSchedulingError(
            "Alpha one-PE GEMM supports only batch one; "
            f"found batch_size={batch}"
        )

    _require_single_call_field_fit(
        "R*S*C1 weight dimension",
        kernel_channel_words,
        _ALPHA_WEIGHT_DIMENSION_MAX_EXTENT,
    )
    _require_single_call_field_fit(
        "K1 bias/scale groups",
        k1,
        _ALPHA_BIAS_SCALE_GROUPS,
    )
    if p1 % p0 or q1 % q0:
        raise MagnetAlphaSchedulingError(
            "Alpha GEMM requires integral P1/P0 and Q1/Q0"
        )

    temporal_tiling_factor = p2 * q2
    weights_per_spatial_tile = kernel_channel_words * k1
    num_wgt_tiles = (
        temporal_tiling_factor
        * (p1 // p0)
        * (q1 // q0)
        * weights_per_spatial_tile
    )
    num_to_read_0 = weights_per_spatial_tile
    num_to_read_1 = p1 * q1 * weights_per_spatial_tile
    mac_iterations = p1 * q1 * weights_per_spatial_tile * p2 * q2
    _require_single_call_field_fit(
        "temporal_tiling_factor",
        temporal_tiling_factor,
        _ALPHA_UINT16_MAX,
    )
    _require_single_call_field_fit(
        "num_wgt_tiles",
        num_wgt_tiles,
        _ALPHA_UINT16_MAX,
    )
    _require_single_call_field_fit(
        "num_to_read_0",
        num_to_read_0,
        _ALPHA_UINT16_MAX,
    )
    _require_single_call_field_fit(
        "num_to_read_1",
        num_to_read_1,
        _ALPHA_UINT16_MAX,
    )
    # GenJsonConfig stores ``mac_iterations - 1`` in NVUINT32.
    _require_single_call_field_fit(
        "MAC iterations",
        mac_iterations,
        _ALPHA_UINT32_MAX + 1,
    )

    input_lanes = hardware_config.input_lanes(input_dtype_width)
    output_lanes = hardware_config.output_lanes(input_dtype_width)
    local_weight_word_bits = input_lanes * input_dtype_width
    global_memory = hardware_config.global_memory
    global_weight = global_memory.memory(global_memory.weight_memory)
    if global_weight.bits_per_bank_entry % local_weight_word_bits:
        raise MagnetAlphaSchedulingError(
            "Alpha global weight line is not an integral number of local "
            "weight vectors"
        )
    vectors_per_global_line = (
        global_weight.bits_per_bank_entry // local_weight_word_bits
    )
    if vectors_per_global_line <= 0 or output_lanes % vectors_per_global_line:
        raise MagnetAlphaSchedulingError(
            "Alpha output lanes cannot be filled by integral global "
            "weight lines"
        )
    lines_per_weight_tile = output_lanes // vectors_per_global_line
    global_weight_sweep_lines = weights_per_spatial_tile * lines_per_weight_tile
    global_weight_entries = (
        global_weight.bank_count * global_weight.depth_per_bank
    )
    # GBWeightDimType has one more bit than its physical address. The reset
    # and advance values are also serialized as signed 16-bit strides; the
    # positive advance makes 32767 the tighter encoded limit.
    _require_single_call_field_fit(
        "global weight sweep lines",
        global_weight_sweep_lines,
        min(2 * global_weight_entries, _ALPHA_INT16_MAX),
    )
    weight_payload_bits = (
        weights_per_spatial_tile * output_lanes * local_weight_word_bits
    )
    if weight_payload_bits > global_weight.capacity_bits:
        raise MagnetAlphaSchedulingError(
            "Alpha GEMM weight payload exceeds global weight "
            f"capacity: required={weight_payload_bits} bits, "
            f"available={global_weight.capacity_bits} bits"
        )


def single_call_gemm_schedule_result(
    layer: GemmWorkload,
    hardware_config: MagnetAlphaHardwareConfig,
    input_dtype_width: int,
    *,
    resident_input_channels: int | None = None,
) -> LocalScheduleResult:
    """Return one complete physically legal Alpha GEMM target schedule."""
    physical_layer = project_gemm_layer(
        layer,
        hardware_config,
        input_dtype_width,
    )
    expected_resident = hardware_config.matrix_input_resident_channels(
        layer.nifm,
        input_dtype_width,
    )
    if resident_input_channels is None:
        resident_input_channels = expected_resident
    if isinstance(resident_input_channels, bool):
        raise MagnetAlphaSchedulingError(
            "resident input channels must be a positive integer"
        )
    normalized_resident_channels = int(resident_input_channels)
    if normalized_resident_channels != resident_input_channels:
        raise MagnetAlphaSchedulingError(
            "resident input channels must be a positive integer"
        )
    resident_input_channels = normalized_resident_channels
    if resident_input_channels != expected_resident:
        raise MagnetAlphaSchedulingError(
            "Alpha GEMM resident input channels disagree with the target "
            f"profile: declared={resident_input_channels}, "
            f"expected={expected_resident} for logical C={layer.nifm}"
        )
    issued_channels = physical_layer.nifm
    if not (
        layer.nifm <= resident_input_channels <= issued_channels
        and issued_channels % resident_input_channels == 0
    ):
        raise MagnetAlphaSchedulingError(
            "Alpha GEMM logical/resident/issued input channels are "
            f"inconsistent: {layer.nifm}/{resident_input_channels}/"
            f"{issued_channels}"
        )
    q_alignment = hardware_config.matrix_input_q_alignment(
        layer.nifm,
        input_dtype_width,
    )
    if q_alignment != issued_channels // resident_input_channels:
        raise MagnetAlphaSchedulingError(
            "Alpha GEMM packed-Q alignment disagrees with the target "
            "channel projection"
        )
    mapping = schedule(
        physical_layer,
        hardware_config,
        input_dtype_width,
        q_alignment=q_alignment,
    )
    _validate_single_call_mapping(
        mapping,
        hardware_config,
        input_dtype_width,
    )
    return LocalScheduleResult(
        mapping=mapping,
        backend=MAGNET_ALPHA_RULE_BASED_SCHEDULER,
        named_factors=mapping_point_to_alpha_factors(mapping),
    )


def single_call_gemm_schedule(
    layer: GemmWorkload,
    hardware_config: MagnetAlphaHardwareConfig,
    input_dtype_width: int,
    *,
    resident_input_channels: int | None = None,
) -> MappingPoint:
    """Return the mapping from one complete Alpha GEMM target schedule."""
    return single_call_gemm_schedule_result(
        layer,
        hardware_config,
        input_dtype_width,
        resident_input_channels=resident_input_channels,
    ).mapping


def schedule_nmp(workload: MagnetAlphaNmpWorkload):
    """Represent an Alpha NMP traversal as a two-level MappingPoint.

    Level zero is the implicit hardware execution level. Level one records
    the AoV-compatible ``P1/Q1/C1/K1`` traversal factors. Unit factors remain
    explicitly ordered so a value such as Softmax ``Q1=1`` is serialized.
    """
    blockings = [[1, 1] for _ in range(NUM_LOOPS)]
    partitionings = [[1, 1] for _ in range(NUM_LOOPS)]
    factors_by_loop = (
        (Loop.OY, workload.p1),
        (Loop.OX, workload.q1),
        (Loop.IC, workload.c1),
        (Loop.OC, workload.k1),
    )
    for loop, factor in factors_by_loop:
        blockings[loop][1] = factor

    preferred_orders = (
        tuple(Loop),
        tuple(loop for loop, _factor in factors_by_loop),
    )
    orders = _encode_loop_orders(
        blockings,
        partitionings,
        preferred_orders,
        (
            frozenset(),
            frozenset(loop for loop, _factor in factors_by_loop),
        ),
    )
    mapping = MappingPoint(orders, blockings, partitionings)
    return LocalScheduleResult(
        mapping=mapping,
        backend=MAGNET_ALPHA_RULE_BASED_SCHEDULER,
        named_factors={
            "p1": workload.p1,
            "q1": workload.q1,
            "c1": workload.c1,
            "k1": workload.k1,
        },
    )


def _activation_dims(shape, transposed=False):
    """Return N/C/H/W for the layout used by one Magnet Alpha operation."""
    if shape is None or len(shape) != 4:
        return None
    if transposed:
        batch, height, width, channels = shape
    else:
        batch, channels, height, width = shape
    return batch, channels, height, width


def _channel_groups(channels, lanes):
    return (channels + lanes - 1) // lanes


def _shape_numel(shape) -> int:
    if shape is None:
        return 0
    return math.prod(int(dimension) for dimension in shape)


def _pool_work_per_output(
    classification,
    input_shape,
    output_shape,
) -> int:
    kernel_size = classification.attribute("kernel_size")
    if isinstance(kernel_size, int):
        return kernel_size * kernel_size
    if isinstance(kernel_size, (tuple, list)) and kernel_size:
        return math.prod(int(value) for value in kernel_size)
    input_elements = _shape_numel(input_shape)
    output_elements = _shape_numel(output_shape)
    if output_elements == 0:
        return 1
    return max(1, math.ceil(input_elements / output_elements))


def _spatial_pair(value):
    if isinstance(value, int):
        return int(value), int(value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return None


def _depthwise_input_accesses(
    classification,
    input_shape,
    output_shape,
    weight_shape,
):
    """Count valid activation taps requested by one depthwise convolution."""
    transposed = classification.attribute("transposed", False)
    input_dims = _activation_dims(input_shape, transposed)
    output_dims = _activation_dims(output_shape, transposed)
    if (
        input_dims is None
        or output_dims is None
        or weight_shape is None
        or len(weight_shape) < 2
    ):
        return None

    stride = _spatial_pair(classification.attribute("stride", 1))
    padding = _spatial_pair(classification.attribute("padding", 0))
    dilation = _spatial_pair(classification.attribute("dilation", 1))
    if stride is None or padding is None or dilation is None:
        return None

    batch, _, input_height, input_width = input_dims
    _, output_channels, output_height, output_width = output_dims
    kernel_height, kernel_width = (
        int(weight_shape[-2]),
        int(weight_shape[-1]),
    )

    def valid_taps(
        input_extent,
        output_extent,
        kernel_extent,
        step,
        pad,
        spacing,
    ):
        return sum(
            0
            <= output_index * step - pad + kernel_index * spacing
            < input_extent
            for output_index in range(output_extent)
            for kernel_index in range(kernel_extent)
        )

    height_taps = valid_taps(
        input_height,
        output_height,
        kernel_height,
        stride[0],
        padding[0],
        dilation[0],
    )
    width_taps = valid_taps(
        input_width,
        output_width,
        kernel_width,
        stride[1],
        padding[1],
        dilation[1],
    )
    return batch * output_channels * height_taps * width_taps


def _estimate_nmp_performance(
    classification,
    workload,
    hardware_config,
):
    """Return scalar memory accesses and a streaming cycle estimate.

    The NMP operates directly on the global hierarchy, so level zero is an
    implicit execution level and level one records global scalar accesses.
    Compute and transfer are assumed to overlap; configured shared bandwidth
    groups decide which transfers must serialize.
    """
    operation = classification.kind
    input_shape = next(
        (
            operand.shape
            for operand in workload.operands_for_port("input")
            if operand.shape is not None
        ),
        None,
    )
    output_shape = workload.output.shape
    input_shape = input_shape or output_shape
    input_elements = _shape_numel(input_shape)
    output_elements = _shape_numel(output_shape)
    parameter_elements = sum(
        _shape_numel(operand.shape)
        for key in ("weight", "bias")
        for operand in workload.operands_for_port(key)
    )

    work_per_output = 1
    input_passes = 1
    if operation == "channel_shuffle_copy":
        output_dims = _activation_dims(
            output_shape,
            classification.attribute("transposed", False),
        )
        if output_dims is None:
            raise MagnetAlphaSchedulingError(
                "Alpha shuffle-copy performance requires a rank-4 output"
            )
        batch, _, output_height, output_width = output_dims
        block_channels = int(classification.attribute("block_channels", 0))
        copied_elements = batch * output_height * output_width * block_channels
        input_accesses = copied_elements
        output_elements = copied_elements
        weight_accesses = 0
    elif operation == "depthwise_conv2d":
        weight_shape = next(
            (
                operand.shape
                for operand in workload.operands_for_port("weight")
                if operand.shape is not None
            ),
            None,
        )
        if weight_shape is not None and len(weight_shape) >= 2:
            work_per_output = math.prod(weight_shape[-2:])
        input_accesses = _depthwise_input_accesses(
            classification,
            input_shape,
            output_shape,
            weight_shape,
        )
        if input_accesses is None:
            input_accesses = output_elements * work_per_output
        weight_accesses = output_elements * work_per_output
    elif operation == "pool2d":
        work_per_output = _pool_work_per_output(
            classification,
            input_shape,
            output_shape,
        )
        input_accesses = output_elements * work_per_output
        weight_accesses = 0
    elif operation in {"group_norm", "softmax"}:
        input_passes = 3
        input_accesses = input_elements * input_passes
        weight_accesses = parameter_elements
    else:
        input_accesses = output_elements
        weight_accesses = parameter_elements

    accesses = LevelAccessCounts(
        input=input_accesses,
        output=output_elements,
        weight=weight_accesses,
    )
    lanes = hardware_config.nmp_lanes()
    scalar_work = max(
        input_accesses,
        output_elements * work_per_output * input_passes,
    )
    compute_cycles = math.ceil(scalar_work / lanes)

    transfers = ()
    if hardware_config.hierarchy_bandwidths:
        accepted_bits = hardware_config.compute_unit(
            hardware_config.nmp_compute_unit
        ).accepted_input_bits
        default_bits = min(accepted_bits) if accepted_bits else 8
        input_bits = max(
            (
                operand.bits
                for name in ("input", "input1", "input2")
                for operand in workload.operands_for_port(name)
                if operand.bits is not None
            ),
            default=default_bits,
        )
        weight_bits = max(
            (
                operand.bits
                for operand in workload.operands_for_port("weight")
                if operand.bits is not None
            ),
            default=input_bits,
        )
        output_bits = workload.output.bits
        if output_bits is None:
            raise MagnetAlphaSchedulingError(
                f"Alpha {operation} scheduling requires the physical "
                "output width"
            )
        transfers = [
            (
                hardware_config.global_memory.name,
                hardware_config.nmp_compute_unit,
                "activation",
                input_accesses * input_bits,
            ),
            (
                hardware_config.nmp_compute_unit,
                hardware_config.global_memory.name,
                "output",
                output_elements * output_bits,
            ),
        ]
        if weight_accesses:
            transfers.append(
                (
                    hardware_config.global_memory.name,
                    hardware_config.nmp_compute_unit,
                    "weight",
                    weight_accesses * weight_bits,
                )
            )
    runtime = hardware_config.estimate_runtime_cycles(
        compute_cycles,
        transfers,
    )

    return (
        (LevelAccessCounts(), accesses),
        float(runtime),
    )


def _validate_nmp_operand_bits(
    classification,
    workload,
    hardware_config,
    *ports,
) -> None:
    """Reject missing or unsupported physical NMP operand widths."""
    nmp = hardware_config.compute_unit(hardware_config.nmp_compute_unit)
    accepted = nmp.accepted_input_bits
    for port in ports:
        operands = workload.operands_for_port(port)
        if not operands:
            raise MagnetAlphaSchedulingError(
                f"Alpha {classification.kind} scheduling requires the "
                f"physical {port} width"
            )
        for operand in operands:
            label = (
                port
                if not operand.occurrence
                else f"{port}{operand.occurrence!r}"
            )
            bits = operand.bits
            if bits is None:
                raise MagnetAlphaSchedulingError(
                    f"Alpha {classification.kind} scheduling requires the "
                    f"physical {label} width"
                )
            if accepted and bits not in accepted:
                raise MagnetAlphaSchedulingError(
                    f"Alpha {classification.kind} {label} uses "
                    f"{bits}-bit data; {nmp.name} accepts "
                    f"{sorted(accepted)}"
                )


def _build_nmp_workload(
    workload,
    hardware_config,
):
    """Recognize an Alpha NMP op and derive its traversal factors."""
    bindings = hardware_config.compute_unit_bindings()
    if "nmp" not in bindings:
        return None

    classification = workload.operation
    if classification is None:
        return None
    output_shape = workload.output.shape
    transposed = classification.attribute("transposed", False)
    output_dims = _activation_dims(output_shape, transposed)
    input_shape = next(
        (
            operand.shape
            for operand in workload.operands_for_port("input")
            if operand.shape is not None
        ),
        None,
    )
    input_dims = _activation_dims(input_shape, transposed)
    operation = classification.kind
    nmp = hardware_config.compute_unit(hardware_config.nmp_compute_unit)
    if not nmp.supports_operation(operation):
        return None
    lanes = hardware_config.nmp_lanes()

    if operation == "channel_shuffle_copy":
        _validate_nmp_operand_bits(
            classification,
            workload,
            hardware_config,
            "input",
        )
        if output_dims is None:
            return None
        batch, _, output_height, output_width = output_dims
        if batch != 1:
            raise MagnetAlphaSchedulingError(
                "Alpha shuffle-copy scheduling supports only batch one"
            )
        block_channels = classification.attribute("block_channels")
        if block_channels != lanes:
            raise MagnetAlphaSchedulingError(
                "Alpha shuffle-copy width must equal the NMP lane count: "
                f"block_channels={block_channels}, lanes={lanes}"
            )
        return _MagnetAlphaNmpDecision(
            operation,
            MagnetAlphaNmpWorkload(
                p1=output_height,
                q1=output_width,
                c1=1,
                k1=1,
            ),
        )

    if operation == "depthwise_conv2d":
        _validate_nmp_operand_bits(
            classification,
            workload,
            hardware_config,
            "input",
            "weight",
        )
        if input_dims is None or output_dims is None:
            return None
        groups = classification.attribute("groups", 1)
        _, input_channels, _, _ = input_dims
        _, output_channels, output_height, output_width = output_dims
        if groups == input_channels and groups != 1:
            if output_channels != input_channels:
                raise MagnetAlphaSchedulingError(
                    "Alpha depthwise NMP scheduling supports only channel "
                    "multiplier one"
                )
            channel_groups = _channel_groups(output_channels, lanes)
            return _MagnetAlphaNmpDecision(
                operation,
                MagnetAlphaNmpWorkload(
                    p1=output_height,
                    q1=output_width,
                    c1=channel_groups,
                    k1=channel_groups,
                ),
            )

    if operation == "pool2d":
        _validate_nmp_operand_bits(
            classification,
            workload,
            hardware_config,
            "input",
        )
        if input_dims is None or output_dims is None:
            return None
        _, input_channels, input_height, input_width = input_dims
        _, output_channels, _, _ = output_dims
        return _MagnetAlphaNmpDecision(
            operation,
            MagnetAlphaNmpWorkload(
                p1=input_height,
                q1=input_width,
                c1=_channel_groups(input_channels, lanes),
                k1=_channel_groups(output_channels, lanes),
            ),
        )

    if operation == "channel_shuffle":
        _validate_nmp_operand_bits(
            classification,
            workload,
            hardware_config,
            "input1",
            "input2",
        )
        if output_dims is None:
            return None
        groups = classification.attribute("groups")
        _, output_channels, output_height, output_width = output_dims
        if groups is None or groups <= 0 or output_channels % groups != 0:
            raise MagnetAlphaSchedulingError(
                "Channel shuffle requires a positive group count that "
                "divides the output channels"
            )
        channels_per_group = output_channels // groups
        return _MagnetAlphaNmpDecision(
            operation,
            MagnetAlphaNmpWorkload(
                p1=output_height,
                q1=output_width,
                c1=_channel_groups(channels_per_group, lanes),
                k1=_channel_groups(output_channels, lanes),
            ),
        )

    if operation == "upsample_nearest2d":
        _validate_nmp_operand_bits(
            classification,
            workload,
            hardware_config,
            "input",
        )
        if output_dims is None:
            return None
        _, channels, output_height, output_width = output_dims
        channel_groups = _channel_groups(channels, lanes)
        return _MagnetAlphaNmpDecision(
            operation,
            MagnetAlphaNmpWorkload(
                p1=output_height,
                q1=output_width,
                c1=channel_groups,
                k1=channel_groups,
            ),
        )

    if operation == "group_norm":
        _validate_nmp_operand_bits(
            classification,
            workload,
            hardware_config,
            "input",
        )
        if output_dims is None:
            return None
        _, channels, output_height, output_width = output_dims
        channel_groups = _channel_groups(channels, lanes)
        return _MagnetAlphaNmpDecision(
            operation,
            MagnetAlphaNmpWorkload(
                p1=output_height,
                q1=output_width,
                c1=channel_groups,
                k1=channel_groups,
            ),
        )

    if operation == "softmax":
        _validate_nmp_operand_bits(
            classification,
            workload,
            hardware_config,
            "input",
        )
        if input_shape is None or len(input_shape) < 2:
            return None
        dim = classification.attribute("dim", -1)
        if dim < 0:
            dim += len(input_shape)
        if dim != len(input_shape) - 1:
            raise MagnetAlphaSchedulingError(
                "Alpha Softmax scheduling supports only the last dimension"
            )
        channel_groups = _channel_groups(input_shape[-1], lanes)
        return _MagnetAlphaNmpDecision(
            operation,
            MagnetAlphaNmpWorkload(
                p1=input_shape[-2],
                q1=1,
                c1=channel_groups,
                k1=channel_groups,
            ),
        )

    return None


def try_schedule_operation(
    workload: OperationWorkload,
    hardware_config,
):
    """Schedule a supported Magnet Alpha NMP op, or return ``None``.

    The compiler-side target classifier owns FX recognition. This provider
    consumes only its frozen semantics, typed operands, and hardware facts.
    """
    if not isinstance(workload, OperationWorkload):
        raise TypeError("Magnet Alpha scheduler requires an OperationWorkload")
    decision = _build_nmp_workload(
        workload,
        hardware_config,
    )
    if decision is None:
        return None
    scheduled = schedule_nmp(decision.traversal)
    access_counts, runtime = _estimate_nmp_performance(
        workload.operation,
        workload,
        hardware_config,
    )
    return LocalScheduleResult(
        mapping=scheduled.mapping,
        backend=scheduled.backend,
        access_counts=access_counts,
        runtime=runtime,
        named_factors=scheduled.named_factors,
    )

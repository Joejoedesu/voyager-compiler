# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Interstellar provider adapter and its runtime models."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scheduler import interstellar

from . import INTERSTELLAR_SCHEDULER
from ..types import (
    LevelAccessCounts,
    LocalScheduleResult,
    MappingPoint,
    OperationWorkload,
)

if TYPE_CHECKING:
    from voyager_compiler.codegen.hardware import HardwareConfig


def _default_compilation_target() -> str:
    # Defer this import to keep the integration module independently importable.
    from voyager_compiler.codegen.targets import DEFAULT_COMPILATION_TARGET

    return DEFAULT_COMPILATION_TARGET


def from_interstellar_mapping(mapping) -> MappingPoint:
    """Copy an Interstellar mapping into the canonical representation."""
    return MappingPoint(
        mapping.loop_orders,
        mapping.loop_blockings,
        mapping.loop_partitionings,
    )


def to_interstellar_mapping(mapping: MappingPoint):
    """Create an Interstellar mapping at the provider boundary."""
    return interstellar.MappingPoint(
        mapping.loop_orders,
        mapping.loop_blockings,
        mapping.loop_partitionings,
    )


def _output_drain_cycles(
    output_size: int,
    input_dtype_width: int,
    output_dtype_width: int,
) -> int:
    """Return output-drain cycles in input-width transfer words."""
    output_bits = output_size * output_dtype_width
    return (output_bits + input_dtype_width - 1) // input_dtype_width


class RuntimeCalculator:
    def __init__(
        self,
        input_dtype_width: int,
        output_dtype_width: int,
        double_buffered_accum_buffer: bool,
        has_high_precision_vector_input: bool = False,
        has_sparse_op: bool = False,
    ):
        self.input_dtype_width = input_dtype_width
        self.output_dtype_width = output_dtype_width
        self.double_buffered_accum_buffer = double_buffered_accum_buffer
        self.has_sparse_op = has_sparse_op
        self.has_high_precision_vector_input = has_high_precision_vector_input
        if self.has_sparse_op:
            print(
                "Using sparse runtime calculator, "
                "which doubles the weight loading time"
            )

    def calculate_runtime(self, architecture, layer, mapping):
        le = interstellar.le

        blockings = mapping.loop_blockings
        orders = mapping.loop_orders
        partitionings = mapping.loop_partitionings

        # IC is unrolled vertically; replication not handled
        sa_weight_loading_time = partitionings[le.IC][0] + 2

        # index of first loop that isn't OX or OY
        first_non_ox_oy_index = 6
        for i in range(le.NUM):
            if i == le.OX or i == le.OY:
                continue
            if orders[i][1] < first_non_ox_oy_index:
                first_non_ox_oy_index = orders[i][1]

        # weight reuse tile: all loops before first non-(OX/OY) loop
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
        # include the reduction loop at the L2 level
        num_remaining_l1_tiles *= blockings[le.IC][2]

        computation_l1_time = weight_reuse_tile_time * num_remaining_l1_tiles

        input_buffer_loading_size = 1
        for loop in [le.IC, le.OY, le.OX]:
            input_buffer_loading_size *= blockings[loop][1]
        input_buffer_loading_time = input_buffer_loading_size

        weight_buffer_loading_size = 1
        for loop in [le.IC, le.OC, le.FY, le.FX]:
            weight_buffer_loading_size *= blockings[loop][1]
        # Include the unrolled reduction loop
        weight_buffer_loading_size *= partitionings[le.IC][0]
        # Assume that each value in the weight buffer is loaded in one cycle
        weight_buffer_loading_time = weight_buffer_loading_size
        # If there is a fused SpMM operation, weight loading time is longer
        if self.has_sparse_op:
            weight_buffer_loading_time *= 2

        output_size = 1
        for loop in [le.OC, le.OY, le.OX]:
            output_size *= blockings[loop][1]
        vector_unit_time = _output_drain_cycles(
            output_size,
            self.input_dtype_width,
            self.output_dtype_width,
        )

        requires_high_precision = (
            self.output_dtype_width > self.input_dtype_width
            or self.has_high_precision_vector_input
        )

        using_double_buffer_accum_buffer = (
            self.double_buffered_accum_buffer and requires_high_precision
        )

        if not using_double_buffer_accum_buffer:
            l1_time = max(
                computation_l1_time,
                input_buffer_loading_time,
                weight_buffer_loading_time,
            )
        else:
            l1_time = max(
                computation_l1_time,
                input_buffer_loading_time,
                weight_buffer_loading_time,
                vector_unit_time,
            )

        l2_blocks = 1
        for i in range(le.NUM):
            if i == le.IC:
                continue
            l2_blocks *= blockings[i][2]

        if self.double_buffered_accum_buffer:
            total_time = (
                max(
                    input_buffer_loading_time,
                    weight_buffer_loading_time,
                )
                + l2_blocks * l1_time
                + vector_unit_time
            )
        else:
            extra_vector_unit_time = (
                output_size if requires_high_precision else 0
            )
            total_time = (
                max(
                    input_buffer_loading_time,
                    weight_buffer_loading_time,
                )
                + l2_blocks * l1_time
                + extra_vector_unit_time
            )

        return total_time


InterstellarScheduleConstraintBuilder = Callable[[int, int], dict]


def build_interstellar_schedule_constraint(ic_dim, num_levels):
    """Build the generic Interstellar schedule constraints."""
    filter_level0 = {"blocking_size": 1, "partitioning_size": 1}
    schedule_hint = {
        "IC": {
            "level0": {
                "order": 1,
                "partitioning_size": ic_dim,
            },
            "level1": {"order": -1},
            "level2": {"order": 0},
        },
        "OC": {"level0": {"order": 0}},
        "FX": {"level0": filter_level0.copy()},
        "FY": {"level0": filter_level0.copy()},
    }

    for loop in ("FX", "FY"):
        schedule_hint[loop]["level2"] = {
            "blocking_size": 1,
            "partitioning_size": 1,
        }

    if num_levels == 4:
        schedule_hint["IC"]["level3"] = {"blocking_size": 1}
        for loop in ("FX", "FY"):
            schedule_hint[loop]["level3"] = {
                "blocking_size": 1,
                "partitioning_size": 1,
            }

    return {"schedule_hint": schedule_hint}


def _get_architecture_dimensions(
    target,
    configured_ic_dim,
    l2_cache_size,
    input_buffer_size,
    weight_buffer_size,
):
    """Validate and return generic legacy Interstellar capacities."""
    _require_generic_legacy_target(target)
    return (
        configured_ic_dim,
        l2_cache_size,
        input_buffer_size,
        weight_buffer_size,
    )


def _require_generic_legacy_target(target):
    """Reject target-specific use of the legacy scalar builders."""
    generic_target = _default_compilation_target()
    if target != generic_target:
        raise ValueError(
            "Legacy Interstellar resource builders support only target "
            f"{generic_target!r}; got {target!r}. Use "
            "build_architecture_and_schedule_from_hardware() with a typed "
            "HardwareConfig for target-specific scheduling."
        )


@dataclass(frozen=True)
class InterstellarHardwareConfig:
    """Physical Interstellar configuration built lazily for each dtype."""

    ic_dim: int
    oc_dim: int
    l2_cache_size: int
    input_buffer_size: int
    weight_buffer_size: int
    accum_buffer_size: int
    target: str = field(default_factory=_default_compilation_target)

    def build(self, input_dtype_width, output_dtype_width=None):
        """Build the Resource and Schedule for one logical input width."""
        return build_architecture_and_schedule(
            self.ic_dim,
            self.oc_dim,
            self.l2_cache_size,
            self.input_buffer_size,
            self.weight_buffer_size,
            self.accum_buffer_size,
            target=self.target,
            input_dtype_width=input_dtype_width,
        )


def _build_cached_architecture(
    hardware_config,
    architecture_cache,
    input_dtype_width,
    weight_dtype_width,
    output_dtype_width,
    schedule_constraint_builder=build_interstellar_schedule_constraint,
):
    cache_key = (
        input_dtype_width,
        weight_dtype_width,
        output_dtype_width,
    )
    if (
        schedule_constraint_builder
        is not build_interstellar_schedule_constraint
    ):
        cache_key += (schedule_constraint_builder,)
    if cache_key not in architecture_cache:
        architecture_cache[cache_key] = (
            build_architecture_and_schedule_from_hardware(
                hardware_config,
                input_dtype_width,
                output_dtype_width,
                weight_dtype_width=weight_dtype_width,
                schedule_constraint_builder=schedule_constraint_builder,
            )
        )
    return architecture_cache[cache_key]


def run_interstellar_scheduler(
    workload: OperationWorkload,
    hardware_config,
    provider_cache,
    scheduler_options,
):
    """Run the Interstellar provider and return a canonical result."""
    return run_interstellar_scheduler_with_schedule_constraint(
        workload,
        hardware_config,
        provider_cache,
        scheduler_options,
        schedule_constraint_builder=build_interstellar_schedule_constraint,
    )


def run_interstellar_scheduler_with_schedule_constraint(
    workload: OperationWorkload,
    hardware_config,
    provider_cache,
    _scheduler_options,
    *,
    schedule_constraint_builder: InterstellarScheduleConstraintBuilder,
):
    """Run Interstellar with one target-selected constraint builder."""
    if not callable(schedule_constraint_builder):
        raise TypeError("schedule_constraint_builder must be callable")
    # Defer this import to avoid the scheduler/codegen package cycle.
    from voyager_compiler.codegen.hardware import (
        HardwareConfig,
        TensorRole,
    )

    if not isinstance(workload, OperationWorkload):
        raise TypeError("Interstellar requires an OperationWorkload")
    if not isinstance(hardware_config, HardwareConfig):
        raise TypeError("Interstellar requires a HardwareConfig")
    layer = workload.gemm
    if layer is None:
        raise TypeError("Interstellar requires a GEMM workload")
    input_operand = workload.require_operand("input")
    weight_operand = workload.operand("weight")
    if weight_operand is None:
        weight_operand = workload.operand("other")
    if weight_operand is None:
        raise ValueError(
            "Interstellar requires a weight or matrix-RHS operand"
        )
    if (
        input_operand.bits is None
        or weight_operand.bits is None
        or workload.output.bits is None
    ):
        raise ValueError(
            "Interstellar requires resolved input, weight, and output widths"
        )
    input_dtype_width = input_operand.bits
    weight_dtype_width = weight_operand.bits
    output_dtype_width = workload.output.bits
    double_buffered_accum_buffer = (
        hardware_config.memory_is_double_buffered(
            hardware_config.local_memory,
            TensorRole.OUTPUT,
        )
    )

    if hardware_config.external_memory is not None:
        runtime_calculator = RuntimeCalculatorWithDRAM(
            input_dtype_width,
            weight_dtype_width,
            output_dtype_width,
            double_buffered_accum_buffer,
            hardware_config=hardware_config,
            has_high_precision_vector_input=(
                workload.has_high_precision_vector_input
            ),
            has_sparse_op=workload.has_sparse_op,
        )
    else:
        runtime_calculator = RuntimeCalculator(
            input_dtype_width,
            output_dtype_width,
            double_buffered_accum_buffer,
            workload.has_high_precision_vector_input,
            workload.has_sparse_op,
        )
    architecture_arguments = (
        hardware_config,
        provider_cache,
        input_dtype_width,
        weight_dtype_width,
        output_dtype_width,
    )
    if schedule_constraint_builder is build_interstellar_schedule_constraint:
        architecture, schedule = _build_cached_architecture(
            *architecture_arguments
        )
    else:
        architecture, schedule = _build_cached_architecture(
            *architecture_arguments,
            schedule_constraint_builder,
        )
    _, runtime, native_mapping, _ = interstellar.optimizer.opt_optimizer(
        architecture,
        layer,
        schedule,
        runtime_calculator.calculate_runtime,
        True,
    )
    _, _, access_list = interstellar.cost_model.get_cost(
        architecture,
        native_mapping,
        layer,
    )
    return LocalScheduleResult(
        mapping=from_interstellar_mapping(native_mapping),
        access_counts=tuple(
            LevelAccessCounts.from_sequence(level)
            for level in access_list
        ),
        runtime=runtime,
        backend=INTERSTELLAR_SCHEDULER,
    )


def _memory_capacity_elements(
    level,
    memory_index,
    input_dtype_width,
    weight_dtype_width,
    output_dtype_width,
    psum_dtype_width,
):
    """Return a conservative scalar capacity for one physical memory.

    Interstellar counts scalar operands rather than bytes. When multiple
    operand roles share one physical memory, normalize its bit capacity by
    the widest bound operand. This is conservative and keeps separate Alpha
    memory instances intact; a later byte-aware cost model can remove this
    compatibility conversion.
    """
    # Defer this import to keep the integration module independently importable.
    from voyager_compiler.codegen.hardware import TensorRole

    role_widths = (
        (level.input_memory, input_dtype_width),
        (level.output_memory, output_dtype_width),
        (level.weight_memory, weight_dtype_width),
    )
    memory_name = level.instances[memory_index].name
    widths = [
        width
        for bound_memory, width in role_widths
        if bound_memory == memory_name
    ]
    memory = level.instances[memory_index]
    if memory.usable_logical_capacity is not None:
        return memory.usable_logical_capacity
    if TensorRole.PARTIAL_SUM in memory.accepted_roles:
        widths.append(psum_dtype_width)
    if not widths:
        raise ValueError(
            f"Memory {memory_name} is not bound to a scheduler operand"
        )
    return memory.capacity_bits // max(widths)


def build_architecture_and_schedule_from_hardware(
    hardware_config: HardwareConfig,
    input_dtype_width: int,
    output_dtype_width: int,
    weight_dtype_width: int | None = None,
    *,
    schedule_constraint_builder: InterstellarScheduleConstraintBuilder = (
        build_interstellar_schedule_constraint
    ),
):
    """Translate a typed hardware profile into Interstellar resources."""
    if weight_dtype_width is None:
        weight_dtype_width = input_dtype_width
    input_lanes = hardware_config.input_lanes(input_dtype_width)
    levels = hardware_config.memory_levels
    capacities = [[1, 1, 1]]
    access_costs = [[1, 1, 1]]
    static_costs = [[0, 0, 0]]
    memory_partitions = [[0, 1, 2]]

    for level in levels:
        capacities.append(
            [
                _memory_capacity_elements(
                    level,
                    index,
                    input_dtype_width,
                    weight_dtype_width,
                    output_dtype_width,
                    hardware_config.output_collector.accumulator_bits,
                )
                for index in range(len(level.instances))
            ]
        )
        access_costs.append(
            [memory.access_cost for memory in level.instances]
        )
        static_costs.append([0 for _ in level.instances])
        memory_partitions.append(level.operand_partitions)

    architecture = interstellar.Resource(
        buf_capacity_list=capacities,
        buf_access_cost_list=access_costs,
        buf_unit_static_cost_list=static_costs,
        para_count_list=[
            input_lanes * hardware_config.output_lanes(input_dtype_width),
            *([1] * len(levels)),
        ],
        mac_capacity=0,
        partition_mode=[0] * (len(levels) + 1),
        memory_partitions=memory_partitions,
        invalid_underutilized=False,
    )

    num_levels = len(levels) + 1
    if not callable(schedule_constraint_builder):
        raise TypeError("schedule_constraint_builder must be callable")
    schedule_constraint = schedule_constraint_builder(
        input_lanes,
        num_levels,
    )
    schedule_constraint["schedule_hint"]["OC"]["level0"][
        "partitioning_size"
    ] = hardware_config.output_lanes(input_dtype_width)
    schedule_data = interstellar.extract_input.extract_schedule_info(
        schedule_constraint,
        num_levels,
    )
    schedule = interstellar.Schedule(
        schedule_data["schedule_hint"],
        schedule_data["partition_loops"],
    )
    return architecture, schedule


def build_architecture_and_schedule(
    ic_dim: int,
    oc_dim: int,
    l2_cache_size: int,
    input_buffer_size: int,
    weight_buffer_size: int,
    accum_buffer_size: int,
    *,
    target: str | None = None,
    input_dtype_width: int = 8,
):
    if target is None:
        target = _default_compilation_target()

    (
        effective_ic_dim,
        effective_l2_cache_size,
        effective_input_buffer_size,
        effective_weight_buffer_size,
    ) = _get_architecture_dimensions(
        target,
        ic_dim,
        l2_cache_size,
        input_buffer_size,
        weight_buffer_size,
    )

    architecture = interstellar.Resource(
        buf_capacity_list=[
            [1, 1, 1],
            [
                effective_input_buffer_size * effective_ic_dim,
                accum_buffer_size * oc_dim,
                effective_weight_buffer_size * oc_dim,
            ],
            [effective_l2_cache_size],
        ],
        buf_access_cost_list=[[1, 1, 1], [10, 10, 10], [100]],
        buf_unit_static_cost_list=[[0, 0, 0], [0, 0, 0], [0]],
        para_count_list=[effective_ic_dim * oc_dim, 1, 1],
        mac_capacity=0,
        partition_mode=[0, 0, 0],
        memory_partitions=[[0, 1, 2], [0, 1, 2], [0, 0, 0]],
        invalid_underutilized=False,
    )

    schedule_constraint = build_interstellar_schedule_constraint(
        effective_ic_dim,
        3,
    )
    schedule_constraint["schedule_hint"]["OC"]["level0"][
        "partitioning_size"
    ] = oc_dim
    schedule_data = interstellar.extract_input.extract_schedule_info(
        schedule_constraint, 3
    )
    schedule = interstellar.Schedule(
        schedule_data["schedule_hint"],
        schedule_data["partition_loops"],
    )
    return architecture, schedule


def _extract_layer_dims(first_node, key_to_shape, output_shape):
    """
    Build an interstellar.Layer from tiled shapes.
    Returns None to signal the layer should be skipped.
    key_to_shape: dict mapping kwarg key names to tiled shapes.
    output_shape: tiled shape of the output tensor.
    """
    # Local imports break the graph-tiling/scheduler adapter cycle.
    from voyager_compiler.codegen.mapping_utils import (
        FIRST_CONV_INPUT_CHANNELS,
        is_conv2d,
        is_depthwise_conv,
        is_matmul,
    )
    from voyager_compiler.codegen.passes.tiling import _conv2d_layout
    from voyager_compiler.codegen.passes.utils import get_arg_value, _pair

    if is_depthwise_conv(first_node):
        return None

    input_shape = key_to_shape.get("input")
    if input_shape is None:
        return None

    # Skip fully-connected layers (no spatial tiling needed)
    if math.prod(input_shape[:-1]) == 1:
        return None

    if is_conv2d(first_node):
        weight_shape = key_to_shape.get("weight")
        if weight_shape is None or len(weight_shape) != 4:
            return None

        transposed = first_node.meta.get("transposed", False)
        (
            kH,
            kW,
            input_channels,
            output_channels,
        ) = _conv2d_layout(weight_shape, True, not transposed)
        _, height, width, _ = _conv2d_layout(
            output_shape, False, not transposed
        )

        if input_channels in FIRST_CONV_INPUT_CHANNELS:
            return None

        stride_val = _pair(get_arg_value(first_node, 3, "stride", 1))
        stride_h, stride_w = stride_val
    else:
        # linear or matmul
        weight_shape = (
            key_to_shape.get("weight") or key_to_shape.get("other")
        )
        if weight_shape is None or len(weight_shape) != 2:
            return None

        # mirrors _build_gemm_shape_map in tiling.py
        weight_transposed = (
            is_matmul(first_node)
            ^ first_node.meta.get("transposed", False)
        )
        if weight_transposed:
            input_channels, output_channels = weight_shape  # (IC, OC)
        else:
            output_channels, input_channels = weight_shape  # transposed

        kH, kW = 1, 1
        height = 1
        width = math.prod(output_shape[:-1])
        stride_h, stride_w = 1, 1

    return interstellar.Layer(
        nifm=input_channels,
        nofm=output_channels,
        wofm=width,
        hofm=height,
        wfil=kW,
        hfil=kH,
        wstd=stride_w,
        hstd=stride_h,
    )


class RuntimeCalculatorWithDRAM:
    """
    Runtime cost model for a 4-level memory hierarchy:
      L0 = PE registers, L1 = scratchpad, L2 = on-chip SRAM, L3 = DRAM.

    Mirrors RuntimeCalculator for L0-L2, then wraps the L2 timing in an outer
    L3 loop and adds DRAM transfer latency (input + weight tile loaded from DRAM
    to L2 before each L3 iteration).

    The typed path reads DRAM bandwidth and double-buffer labels from
    ``HardwareConfig``. Transfers through single-buffered memories serialize
    with compute. Transfers through double-buffered memories overlap with
    compute, so their contribution is ``max(compute, transfer)``.
    """

    def __init__(
        self,
        input_dtype_width: int,
        weight_dtype_width: int,
        output_dtype_width: int,
        double_buffered_accum_buffer: bool,
        dram_bandwidth: int | None = None,
        double_buffered_l2: bool = False,
        has_sparse_op: bool = False,
        has_high_precision_vector_input: bool = False,
        hardware_config: HardwareConfig | None = None,
    ):
        self.input_dtype_width = input_dtype_width
        self.weight_dtype_width = weight_dtype_width
        self.output_dtype_width = output_dtype_width
        self.double_buffered_accum_buffer = double_buffered_accum_buffer
        self.dram_bandwidth = dram_bandwidth
        self.double_buffered_l2 = double_buffered_l2
        self.has_sparse_op = has_sparse_op
        self.has_high_precision_vector_input = has_high_precision_vector_input
        self.hardware_config = hardware_config
        if self.hardware_config is None and not self.dram_bandwidth:
            raise ValueError("dram_bandwidth must be positive")

    def calculate_runtime(self, architecture, layer, mapping):
        le = interstellar.le

        blockings = mapping.loop_blockings
        orders = mapping.loop_orders
        partitionings = mapping.loop_partitionings

        # --- L1: weight-reuse tile timing (identical to RuntimeCalculator) ---
        sa_weight_loading_time = partitionings[le.IC][0] + 2

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
            sa_weight_loading_time,
            weight_reuse_tile_size,
        )

        num_remaining_l1_tiles = 1
        for i in range(le.NUM):
            if orders[i][1] >= first_non_ox_oy_index:
                num_remaining_l1_tiles *= blockings[i][1]
        num_remaining_l1_tiles *= blockings[le.IC][2]
        computation_l1_time = weight_reuse_tile_time * num_remaining_l1_tiles

        input_buffer_loading_size = 1
        for loop in [le.IC, le.OY, le.OX]:
            input_buffer_loading_size *= blockings[loop][1]
        input_buffer_loading_time = input_buffer_loading_size

        weight_buffer_loading_size = 1
        for loop in [le.IC, le.OC, le.FY, le.FX]:
            weight_buffer_loading_size *= blockings[loop][1]
        weight_buffer_loading_size *= partitionings[le.IC][0]
        weight_buffer_loading_time = (
            weight_buffer_loading_size
            * self.weight_dtype_width
            / self.input_dtype_width
        )
        if self.has_sparse_op:
            weight_buffer_loading_time *= 2

        output_size = 1
        for loop in [le.OC, le.OY, le.OX]:
            output_size *= blockings[loop][1]
        vector_unit_time = _output_drain_cycles(
            output_size,
            self.input_dtype_width,
            self.output_dtype_width,
        )

        requires_high_precision = (
            self.output_dtype_width > self.input_dtype_width
            or self.has_high_precision_vector_input
        )

        using_double_buffer_accum_buffer = (
            self.double_buffered_accum_buffer and requires_high_precision
        )

        if not using_double_buffer_accum_buffer:
            l1_time = max(
                computation_l1_time,
                input_buffer_loading_time,
                weight_buffer_loading_time,
            )
        else:
            l1_time = max(
                computation_l1_time,
                input_buffer_loading_time,
                weight_buffer_loading_time,
                vector_unit_time,
            )

        # --- L2: outer spatial-tile loop ---
        l2_blocks = 1
        for i in range(le.NUM):
            if i != le.IC:
                l2_blocks *= blockings[i][2]

        if self.double_buffered_accum_buffer:
            per_l3_compute_time = (
                max(
                    input_buffer_loading_time,
                    weight_buffer_loading_time,
                )
                + l2_blocks * l1_time
                + vector_unit_time
            )
        else:
            extra_vector_unit_time = (
                output_size if requires_high_precision else 0
            )
            per_l3_compute_time = (
                max(
                    input_buffer_loading_time,
                    weight_buffer_loading_time,
                )
                + l2_blocks * l1_time
                + extra_vector_unit_time
            )

        # --- L3 (DRAM): outer tile loop + transfer latency ---
        # IC is pinned to blocking_size=1 at L3, so l3_blocks is purely spatial.
        # Each L3 iteration loads inputs and weights from DRAM, then writes
        # outputs back.
        l3_blocks = 1
        for i in range(le.NUM):
            if i != le.IC:
                l3_blocks *= blockings[i][3]

        # DRAM transfer size for one L3 block. Levels 0-2 define the tile;
        # level 3 belongs in l3_blocks, not the per-iteration tile size.
        dram_input_elements = (
            partitionings[le.IC][0]
            * blockings[le.IC][1] * blockings[le.IC][2]
            * blockings[le.OY][1] * blockings[le.OY][2]
            * blockings[le.OX][1] * blockings[le.OX][2]
        )
        dram_weight_elements = (
            partitionings[le.IC][0]
            * blockings[le.IC][1] * blockings[le.IC][2]
            * partitionings[le.OC][0]
            * blockings[le.OC][1] * blockings[le.OC][2]
            * blockings[le.FY][1]
            * blockings[le.FX][1]
        )
        dram_output_elements = (
            partitionings[le.OC][0]
            * blockings[le.OC][1] * blockings[le.OC][2]
            * blockings[le.OY][1] * blockings[le.OY][2]
            * blockings[le.OX][1] * blockings[le.OX][2]
        )
        if self.hardware_config is not None:
            # Defer this import to avoid the scheduler/codegen package cycle.
            from voyager_compiler.codegen.hardware import TensorRole

            external = self.hardware_config.external_memory
            global_memory = self.hardware_config.global_memory
            transfers = (
                (
                    external.name,
                    global_memory.name,
                    TensorRole.ACTIVATION,
                    dram_input_elements * self.input_dtype_width,
                ),
                (
                    external.name,
                    global_memory.name,
                    TensorRole.WEIGHT,
                    dram_weight_elements * self.weight_dtype_width,
                ),
                (
                    global_memory.name,
                    external.name,
                    TensorRole.OUTPUT,
                    dram_output_elements * self.output_dtype_width,
                ),
            )
            buffered = tuple(
                transfer
                for transfer in transfers
                if self.hardware_config.memory_is_double_buffered(
                    global_memory,
                    transfer[2],
                )
            )
            unbuffered = tuple(
                transfer
                for transfer in transfers
                if transfer not in buffered
            )
            buffered_time = (
                self.hardware_config.estimate_transfer_cycles(buffered)
                if buffered
                else 0
            )
            unbuffered_time = (
                self.hardware_config.estimate_transfer_cycles(unbuffered)
                if unbuffered
                else 0
            )
            per_l3_time = unbuffered_time + max(
                per_l3_compute_time,
                buffered_time,
            )
        else:
            dram_transfer_time = (
                dram_input_elements
                + (
                    dram_weight_elements
                    * self.weight_dtype_width
                    / self.input_dtype_width
                )
                + (
                    dram_output_elements
                    * self.output_dtype_width
                    / self.input_dtype_width
                )
            ) / self.dram_bandwidth
            if self.double_buffered_l2:
                per_l3_time = max(
                    dram_transfer_time,
                    per_l3_compute_time,
                )
            else:
                per_l3_time = (
                    dram_transfer_time + per_l3_compute_time
                )

        return l3_blocks * per_l3_time


def build_architecture_and_schedule_with_dram(
    ic_dim: int,
    oc_dim: int,
    l2_cache_size: int,
    input_buffer_size: int,
    weight_buffer_size: int,
    accum_buffer_size: int,
    dram_size: int,
    dram_access_cost: int = 1000,
    double_buffered_l2: bool = False,
    *,
    target: str | None = None,
    input_dtype_width: int = 8,
):
    """
    Build a 4-level interstellar Resource and Schedule with DRAM as L3.

    Extends build_architecture_and_schedule with a 4th level:
      L0 = PE registers, L1 = L1 scratchpad, L2 = on-chip SRAM, L3 = DRAM.

    IC is pinned to blocking_size=1 at L3 so that all IC accumulation completes
    on-chip and partial sums never spill to DRAM.

    When double_buffered_l2=True, two L3 tiles must fit in L2 simultaneously
    (one being computed, one being loaded), so the effective L2 capacity is
    halved.
    """
    if target is None:
        target = _default_compilation_target()

    (
        effective_ic_dim,
        effective_l2_cache_size,
        effective_input_buffer_size,
        effective_weight_buffer_size,
    ) = _get_architecture_dimensions(
        target,
        ic_dim,
        l2_cache_size,
        input_buffer_size,
        weight_buffer_size,
    )
    if double_buffered_l2:
        effective_l2_cache_size //= 2

    effective_dram_size = dram_size

    architecture = interstellar.Resource(
        buf_capacity_list=[
            [1, 1, 1],
            [
                effective_input_buffer_size * effective_ic_dim,
                accum_buffer_size * oc_dim,
                effective_weight_buffer_size * oc_dim,
            ],
            [effective_l2_cache_size],
            [effective_dram_size],
        ],
        buf_access_cost_list=[
            [1, 1, 1],
            [10, 10, 10],
            [100],
            [dram_access_cost],
        ],
        buf_unit_static_cost_list=[[0, 0, 0], [0, 0, 0], [0], [0]],
        para_count_list=[effective_ic_dim * oc_dim, 1, 1, 1],
        memory_partitions=[[0, 1, 2], [0, 1, 2], [0, 0, 0], [0, 0, 0]],
        mac_capacity=0,
        partition_mode=[0, 0, 0, 0],
        invalid_underutilized=False,
    )

    schedule_constraint = build_interstellar_schedule_constraint(
        effective_ic_dim,
        4,
    )
    schedule_constraint["schedule_hint"]["OC"]["level0"][
        "partitioning_size"
    ] = oc_dim
    schedule_data = interstellar.extract_input.extract_schedule_info(
        schedule_constraint, 4
    )
    schedule = interstellar.Schedule(
        schedule_data["schedule_hint"],
        schedule_data["partition_loops"],
    )
    return architecture, schedule


def _extract_layer_from_node(node):
    """
    Build an interstellar Layer from a node's current (pre-tiling) shapes.

    Unlike _extract_layer_dims (which takes already-tiled shapes from the memory
    mapping phase), this function reads shapes directly from the FX node and is
    meant to be called before any tiling has occurred.

    Returns None for layers that should be skipped (depthwise, FC with batch=1,
    RGB/grayscale first conv, unsupported weight shapes).
    """
    # Local imports break the graph-tiling/scheduler adapter cycle.
    from voyager_compiler.codegen.mapping_utils import (
        FIRST_CONV_INPUT_CHANNELS,
        is_conv2d,
        is_depthwise_conv,
        is_matmul,
    )
    from voyager_compiler.codegen.passes.tiling import _conv2d_layout
    from voyager_compiler.codegen.passes.utils import get_arg_value, _pair

    if is_depthwise_conv(node):
        return None

    if is_conv2d(node):
        if len(node.args) < 2:
            return None
        weight_shape = node.args[1].shape
        if len(weight_shape) != 4:
            return None

        transposed = node.meta.get("transposed", False)
        kH, kW, input_channels, output_channels = _conv2d_layout(
            weight_shape, True, not transposed
        )
        _, height, width, _ = _conv2d_layout(node.shape, False, not transposed)

        if input_channels in FIRST_CONV_INPUT_CHANNELS:
            return None

        stride_h, stride_w = _pair(get_arg_value(node, 3, "stride", 1))
    else:
        input_shape = node.args[0].shape
        if math.prod(input_shape[:-1]) == 1:
            return None

        if len(node.args) < 2:
            return None
        weight_shape = node.args[1].shape
        if len(weight_shape) != 2:
            return None

        weight_transposed = is_matmul(node) ^ node.meta.get("transposed", False)
        if weight_transposed:
            input_channels, output_channels = weight_shape
        else:
            output_channels, input_channels = weight_shape

        kH, kW = 1, 1
        height = 1
        width = math.prod(node.shape[:-1])
        stride_h, stride_w = 1, 1

    return interstellar.Layer(
        nifm=input_channels,
        nofm=output_channels,
        wofm=width,
        hofm=height,
        wfil=kW,
        hfil=kH,
        wstd=stride_w,
        hstd=stride_h,
    )


def run_interstellar_dram(
    node,
    architecture,
    schedule,
    dram_bandwidth: int,
    input_dtype_width: int = 8,
    weight_dtype_width: int = 8,
    output_dtype_width: int = 8,
    double_buffered_accum_buffer: bool = False,
    double_buffered_l2: bool = False,
):
    """
    Run Interstellar with four-level DRAM for one GEMM or convolution node.

    Extracts layer dims from the node's current (pre-tiling) shapes, runs the
    optimizer, logs the resulting L2 and L3 tile sizes, and stores the mapping
    in node.meta["interstellar_dram_tiling"] for future use.

    Returns the best MappingPoint, or None if the node is skipped.
    """
    logger = logging.getLogger(__name__)

    layer = _extract_layer_from_node(node)
    if layer is None:
        return None

    logger.info(
        f"[interstellar DRAM] {node.name}: "
        f"IC={layer.nifm} OC={layer.nofm} "
        f"H={layer.hofm} W={layer.wofm} "
        f"kH={layer.hfil} kW={layer.wfil}"
    )

    rc = RuntimeCalculatorWithDRAM(
        input_dtype_width,
        weight_dtype_width,
        output_dtype_width,
        double_buffered_accum_buffer,
        dram_bandwidth,
        double_buffered_l2=double_buffered_l2,
    )

    _, runtime, mapping, _ = interstellar.optimizer.opt_optimizer(
        architecture,
        layer,
        schedule,
        rc.calculate_runtime,
        verbose=False,
    )

    le = interstellar.le
    b = mapping.loop_blockings
    logger.info(
        f"[interstellar DRAM] {node.name} L1 tiles: "
        f"IC={b[le.IC][1]} OC={b[le.OC][1]} "
        f"OX={b[le.OX][1]} OY={b[le.OY][1]} ON={b[le.ON][1]}"
    )
    logger.info(
        f"[interstellar DRAM] {node.name} L2 tiles: "
        f"IC={b[le.IC][2]} OC={b[le.OC][2]} "
        f"OX={b[le.OX][2]} OY={b[le.OY][2]} ON={b[le.ON][2]}"
    )
    logger.info(
        f"[interstellar DRAM] {node.name} L3 (DRAM) tiles: "
        f"IC={b[le.IC][3]} OC={b[le.OC][3]} "
        f"OX={b[le.OX][3]} OY={b[le.OY][3]} ON={b[le.ON][3]}"
    )
    logger.info(f"[interstellar DRAM] {node.name} estimated runtime: {runtime}")
    logger.info(interstellar.utils.format_tiling(mapping))

    node.meta["interstellar_dram_tiling"] = mapping
    return mapping


def format_mapping(mapping) -> str:
    """Format a canonical mapping through the Interstellar adapter."""
    native_mapping = (
        to_interstellar_mapping(mapping)
        if isinstance(mapping, MappingPoint)
        else mapping
    )
    return interstellar.utils.format_tiling(native_mapping)


__all__ = [
    "INTERSTELLAR_SCHEDULER",
    "InterstellarHardwareConfig",
    "RuntimeCalculator",
    "RuntimeCalculatorWithDRAM",
    "_extract_layer_dims",
    "build_architecture_and_schedule",
    "build_architecture_and_schedule_from_hardware",
    "build_architecture_and_schedule_with_dram",
    "build_interstellar_schedule_constraint",
    "format_mapping",
    "from_interstellar_mapping",
    "run_interstellar_dram",
    "run_interstellar_scheduler",
    "run_interstellar_scheduler_with_schedule_constraint",
    "to_interstellar_mapping",
]

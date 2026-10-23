# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Registered Magnet Alpha GEMM scheduler providers."""

from dataclasses import replace

from scheduler.magnet_alpha_scheduler.evaluator import (
    MagnetAlphaRuntimeCalculator,
)
from scheduler.magnet_alpha_scheduler.scheduler import (
    MAGNET_ALPHA_RULE_BASED_SCHEDULER,
    MagnetAlphaSchedulingError,
    project_gemm_layer,
    single_call_gemm_schedule,
    single_call_gemm_schedule_result,
    try_schedule_operation,
)
from scheduler.types import SchedulerProviderSet


def _require_magnet_alpha_hardware(hardware_config, provider):
    bindings = hardware_config.compute_unit_bindings()
    if "matrix" not in bindings or "nmp" not in bindings:
        raise TypeError(
            f"{provider} requires matrix and NMP compute-unit bindings"
        )


def _accum_buffer_is_double_buffered(hardware_config) -> bool:
    local_memory = hardware_config.local_memory
    return local_memory.memory(local_memory.output_memory).double_buffered


def _gemm_transfers(
    access_counts,
    hardware_config,
    input_dtype_width,
    output_dtype_width,
):
    local = access_counts[1]
    global_accesses = access_counts[2]
    return (
        (
            hardware_config.global_memory.name,
            hardware_config.local_memory.name,
            "activation",
            local.input * input_dtype_width,
        ),
        (
            hardware_config.global_memory.name,
            hardware_config.local_memory.name,
            "weight",
            local.weight * input_dtype_width,
        ),
        (
            hardware_config.local_memory.name,
            hardware_config.global_memory.name,
            "output",
            global_accesses.output * output_dtype_width,
        ),
    )


def _gemm_facts(workload, provider):
    layer = workload.gemm
    if layer is None:
        raise TypeError(f"{provider} requires a GEMM workload")
    input_operand = workload.require_operand("input")
    if input_operand.bits is None or workload.output.bits is None:
        raise ValueError(
            f"{provider} requires resolved input and output widths"
        )
    return layer, input_operand.bits, workload.output.bits


def run_rule_based_gemm_scheduler(
    workload,
    hardware_config,
    _provider_cache,
    _scheduler_options,
):
    """Run the registered Magnet Alpha rule-based GEMM provider."""
    _require_magnet_alpha_hardware(
        hardware_config,
        MAGNET_ALPHA_RULE_BASED_SCHEDULER,
    )
    layer, input_dtype_width, output_dtype_width = _gemm_facts(
        workload,
        MAGNET_ALPHA_RULE_BASED_SCHEDULER,
    )
    physical_layer = project_gemm_layer(
        layer,
        hardware_config,
        input_dtype_width,
    )
    runtime_calculator = MagnetAlphaRuntimeCalculator(
        input_dtype_width,
        output_dtype_width,
        _accum_buffer_is_double_buffered(hardware_config),
        workload.has_high_precision_vector_input,
        workload.has_sparse_op,
        vector_size=hardware_config.input_lanes(input_dtype_width),
    )
    result = single_call_gemm_schedule_result(
        layer,
        hardware_config,
        input_dtype_width,
    )
    mapping = result.mapping
    runtime = runtime_calculator.calculate_runtime(
        physical_layer,
        mapping,
    )
    access_counts = runtime_calculator.calculate_access_counts(
        physical_layer,
        mapping,
    )
    runtime = hardware_config.estimate_runtime_cycles(
        runtime,
        _gemm_transfers(
            access_counts,
            hardware_config,
            input_dtype_width,
            output_dtype_width,
        ),
    )
    return replace(
        result,
        access_counts=access_counts,
        runtime=runtime,
    )


def alpha_gemm_tiling_candidate_fits(
    layer,
    full_layer,
    hardware_config,
    input_dtype_width,
    can_materialize_output_channel_tiles,
) -> bool:
    """Return whether one outer tile is an encodable Alpha GEMM call."""
    if (
        layer.nofm != full_layer.nofm
        and not can_materialize_output_channel_tiles
    ):
        return False
    try:
        single_call_gemm_schedule(
            layer,
            hardware_config,
            input_dtype_width,
        )
    except MagnetAlphaSchedulingError:
        return False
    return True


MAGNET_ALPHA_SCHEDULER_PROVIDERS = SchedulerProviderSet(
    default_backend=MAGNET_ALPHA_RULE_BASED_SCHEDULER,
    gemm_providers={
        MAGNET_ALPHA_RULE_BASED_SCHEDULER: run_rule_based_gemm_scheduler,
    },
    operation_provider=try_schedule_operation,
    gemm_tiling_candidate_fits=alpha_gemm_tiling_candidate_fits,
)


__all__ = [
    "MAGNET_ALPHA_SCHEDULER_PROVIDERS",
]

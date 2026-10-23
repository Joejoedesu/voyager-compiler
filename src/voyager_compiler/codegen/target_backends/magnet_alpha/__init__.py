# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Magnet Alpha compilation-target backend components."""

from scheduler.magnet_alpha_scheduler.providers import (
    MAGNET_ALPHA_SCHEDULER_PROVIDERS,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.graph import (
    apply_graph_rewrites,
    canonicalize_eadd_execution_ports,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.hardware import (
    ALPHA_HARDWARE_CONFIG,
    AOV_COMPAT_HARDWARE_CONFIG,
    MagnetAlphaHardwareConfig,
    create_alpha_hardware_config,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.hardware_pipeline import (
    MAGNET_ALPHA_HARDWARE_PIPELINE,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.lowering import (
    apply_memory_rewrites,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.memory import (
    AlphaScratchpadAllocator,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.operators import (
    INT8_CONV2D_ACCUMULATOR_OPERATOR,
    NMP_FIXED_REQUANTIZE_OPERATOR,
    PPU_FIXED_RELU_OPERATOR,
    PVS_QUANTIZE_OPERATOR,
    VSQ_CONV2D_OPERATOR,
    classify_scheduler_operation,
    run_tiled_conv2d,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.pe_lowering import (
    pad_matrix_output_operands,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.quantization import (
    apply_quantization_rewrites,
    cleanup_quantization_preparation,
    finalize_quantization_calibration,
    prepare_quantization,
    prepare_quantization_dynamic_qat,
    prepare_quantization_qat,
    prepare_quantization_recalibration,
    refresh_quantization_qat,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.serialization import (
    prepare_mapped_serialization_metadata,
    prepare_serialized_artifacts,
    validate_serialization_metadata,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.traits import (
    MAGNET_ALPHA_OPERATOR_TRAITS,
)
from voyager_compiler.codegen.target_backends.spec import TargetBackendSpec
from voyager_compiler.codegen.targets import MAGNET_ALPHA_COMPILATION_TARGET

MAGNET_ALPHA_EXECUTION_FUSION_PATTERNS = tuple(
    tuple(stage.targets for stage in pattern)
    for pattern in MAGNET_ALPHA_HARDWARE_PIPELINE
)


def create_scratchpad_allocator(hardware):
    """Create the model-wide allocator for Magnet Alpha physical memories."""
    return AlphaScratchpadAllocator(hardware)


BACKEND_SPEC = TargetBackendSpec(
    target=MAGNET_ALPHA_COMPILATION_TARGET,
    operator_traits=MAGNET_ALPHA_OPERATOR_TRAITS,
    hardware_config_type=MagnetAlphaHardwareConfig,
    scheduler_providers=MAGNET_ALPHA_SCHEDULER_PROVIDERS,
    apply_graph_rewrites=apply_graph_rewrites,
    apply_quantization_rewrites=apply_quantization_rewrites,
    apply_memory_rewrites=apply_memory_rewrites,
    prepare_quantization=prepare_quantization,
    finalize_quantization_calibration=(finalize_quantization_calibration),
    create_scratchpad_allocator=create_scratchpad_allocator,
    allow_qdq_fusion=False,
    supports_bufferized=False,
    # Alpha owns resident concat/stack writes and address legality.
    uses_generic_aggregate_copies=False,
    hardware_pipeline=MAGNET_ALPHA_HARDWARE_PIPELINE,
    execution_fusion_patterns=MAGNET_ALPHA_EXECUTION_FUSION_PATTERNS,
    scratchpad_validate_end_to_end_operations=(
        AlphaScratchpadAllocator.validate_end_to_end_operations
    ),
    scratchpad_has_operation_scope=(
        AlphaScratchpadAllocator.has_operation_scope
    ),
    scratchpad_allocate_operation_scope=(
        AlphaScratchpadAllocator.allocate_operation_scope
    ),
    scratchpad_validate_operation_scopes=(
        AlphaScratchpadAllocator.validate_operation_scopes
    ),
    prepare_execution_fusion=canonicalize_eadd_execution_ports,
    classify_scheduler_operation=classify_scheduler_operation,
    pad_matrix_output_operands=pad_matrix_output_operands,
    prepare_quantization_recalibration=(prepare_quantization_recalibration),
    prepare_quantization_dynamic_qat=(prepare_quantization_dynamic_qat),
    prepare_quantization_qat=prepare_quantization_qat,
    refresh_quantization_qat=refresh_quantization_qat,
    cleanup_quantization_preparation=(cleanup_quantization_preparation),
    run_tiled_conv2d=run_tiled_conv2d,
    prepare_serialized_artifacts=prepare_serialized_artifacts,
    prepare_serialization_metadata=(prepare_mapped_serialization_metadata),
    validate_serialization_metadata=(validate_serialization_metadata),
)


__all__ = [
    "ALPHA_HARDWARE_CONFIG",
    "AOV_COMPAT_HARDWARE_CONFIG",
    "BACKEND_SPEC",
    "MAGNET_ALPHA_EXECUTION_FUSION_PATTERNS",
    "MAGNET_ALPHA_HARDWARE_PIPELINE",
    "MagnetAlphaHardwareConfig",
    "create_alpha_hardware_config",
]

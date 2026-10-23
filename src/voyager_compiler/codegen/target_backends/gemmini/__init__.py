# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Immutable assembly for the Gemmini compilation target."""

from torch.fx import GraphModule

from scheduler.dispatch import GENERIC_SCHEDULER_PROVIDERS
from voyager_compiler.codegen.memory import ScratchpadAllocator
from voyager_compiler.codegen.target_backends.gemmini.hardware import (
    GEMMINI_COMPILATION_TARGET,
    GEMMINI_HARDWARE_CONFIG,
    GemminiHardwareConfig,
    create_gemmini_hardware_config,
)
from voyager_compiler.codegen.target_backends.gemmini.hardware_pipeline import (
    GEMMINI_EXECUTION_FUSION_PATTERNS,
    GEMMINI_HARDWARE_PIPELINE,
)
from voyager_compiler.codegen.target_backends.gemmini.lowering import (
    apply_memory_rewrites,
)
from voyager_compiler.codegen.target_backends.gemmini.quantization import (
    apply_quantization_rewrites,
    finalize_quantization_calibration,
    prepare_quantization,
)
from voyager_compiler.codegen.target_backends.gemmini.serialization import (
    prepare_canonical_serialization_metadata,
    validate_serialization_metadata,
)
from voyager_compiler.codegen.target_backends.gemmini.traits import (
    GEMMINI_OPERATOR_TRAITS,
)
from voyager_compiler.codegen.target_backends.spec import TargetBackendSpec


def _preserve_exact_graph(model: GraphModule) -> GraphModule:
    """Keep the target-owned canonical operation graph unchanged."""
    return model


def _create_scratchpad_allocator(
    hardware: GemminiHardwareConfig,
) -> ScratchpadAllocator:
    """Create the generic transient mapper required by the compile pipeline."""
    return ScratchpadAllocator(
        hardware.global_memory,
        hardware_config=hardware,
    )


BACKEND_SPEC = TargetBackendSpec(
    target=GEMMINI_COMPILATION_TARGET,
    operator_traits=GEMMINI_OPERATOR_TRAITS,
    hardware_config_type=GemminiHardwareConfig,
    scheduler_providers=GENERIC_SCHEDULER_PROVIDERS,
    apply_graph_rewrites=_preserve_exact_graph,
    apply_quantization_rewrites=apply_quantization_rewrites,
    apply_memory_rewrites=apply_memory_rewrites,
    prepare_quantization=prepare_quantization,
    finalize_quantization_calibration=finalize_quantization_calibration,
    create_scratchpad_allocator=_create_scratchpad_allocator,
    allow_qdq_fusion=False,
    supports_bufferized=False,
    uses_generic_aggregate_copies=True,
    hardware_pipeline=GEMMINI_HARDWARE_PIPELINE,
    execution_fusion_patterns=GEMMINI_EXECUTION_FUSION_PATTERNS,
    classify_scheduler_operation=None,
    prepare_serialized_artifacts=None,
    prepare_serialization_metadata=prepare_canonical_serialization_metadata,
    validate_serialization_metadata=validate_serialization_metadata,
)


__all__ = [
    "BACKEND_SPEC",
    "GEMMINI_EXECUTION_FUSION_PATTERNS",
    "GEMMINI_HARDWARE_PIPELINE",
    "GEMMINI_HARDWARE_CONFIG",
    "GemminiHardwareConfig",
    "create_gemmini_hardware_config",
]

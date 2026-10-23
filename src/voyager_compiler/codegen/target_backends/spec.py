# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Typed contracts for compilation-target backend registration."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from scheduler.types import SchedulerProviderSet
from voyager_compiler.codegen.hardware import HardwareConfig

BackendHook = Callable[..., Any]


def _empty_mapping() -> Mapping:
    return MappingProxyType({})


@dataclass(frozen=True)
class TargetOperatorTraits:
    """Selected-target facts consumed by otherwise generic passes."""

    gemm_operators: tuple = ()
    conv2d_operators: tuple = ()
    output_padding_epilogue_operators: tuple = ()
    output_padding_components: Mapping = field(default_factory=_empty_mapping)
    conv2d_input_channel_dims: Mapping = field(default_factory=_empty_mapping)
    prepared_domain_transfer_rules: tuple = ()
    transposed_operators: Mapping = field(default_factory=_empty_mapping)
    parameter_operand_index: Mapping = field(default_factory=_empty_mapping)
    non_allocating_operand_meta: tuple = ()
    folded_scale_operators: tuple = ()
    scale_bearing_conv2d_operators: tuple = ()
    tiled_conv2d_operators: tuple = ()
    conv2d_tiling_epilogue_operators: tuple = ()
    quantization_output_epilogue_operators: tuple = ()
    operand_role_overrides: Mapping = field(default_factory=_empty_mapping)
    execution_fusion_predecessor_inputs: Mapping = field(
        default_factory=_empty_mapping
    )

    def __post_init__(self):
        tuple_fields = (
            "gemm_operators",
            "conv2d_operators",
            "output_padding_epilogue_operators",
            "prepared_domain_transfer_rules",
            "non_allocating_operand_meta",
            "folded_scale_operators",
            "scale_bearing_conv2d_operators",
            "tiled_conv2d_operators",
            "conv2d_tiling_epilogue_operators",
            "quantization_output_epilogue_operators",
        )
        mapping_fields = (
            "output_padding_components",
            "conv2d_input_channel_dims",
            "transposed_operators",
            "parameter_operand_index",
            "operand_role_overrides",
            "execution_fusion_predecessor_inputs",
        )
        for name in tuple_fields:
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in mapping_fields:
            value = MappingProxyType(dict(getattr(self, name)))
            object.__setattr__(self, name, value)
        for (
            target,
            operand_index,
        ) in self.execution_fusion_predecessor_inputs.items():
            if isinstance(operand_index, bool) or not isinstance(
                operand_index,
                int,
            ):
                raise TypeError(
                    "execution fusion predecessor input for "
                    f"{target} must be an integer"
                )
            if operand_index < 0:
                raise ValueError(
                    "execution fusion predecessor input for "
                    f"{target} must not be negative"
                )
            schema = getattr(target, "_schema", None)
            arguments = getattr(schema, "arguments", ())
            if arguments and operand_index >= len(arguments):
                raise ValueError(
                    "execution fusion predecessor input for "
                    f"{target} exceeds its operand count"
                )


@dataclass(frozen=True)
class TargetBackendSpec:
    """One explicit, immutable target implementation registration."""

    target: str
    operator_traits: TargetOperatorTraits
    hardware_config_type: type[HardwareConfig]
    scheduler_providers: SchedulerProviderSet
    apply_graph_rewrites: BackendHook
    apply_quantization_rewrites: BackendHook
    apply_memory_rewrites: BackendHook
    prepare_quantization: BackendHook
    finalize_quantization_calibration: BackendHook
    create_scratchpad_allocator: BackendHook
    allow_qdq_fusion: bool
    supports_bufferized: bool
    uses_generic_aggregate_copies: bool
    # Target-owned hardware pipeline over ordinary PyTorch/quantized FX
    # semantics. Each entry is an ordered tuple of OpMatcher stages, exactly
    # the shape accepted by ``fuse_operator``.
    hardware_pipeline: tuple = ()
    # Target-lowered sequences that must form one execution record.  These
    # retain the historical tuple-of-targets representation and are matched
    # with ``require_all_pattern_stages=True`` by ``transform``.
    execution_fusion_patterns: tuple = ()
    scratchpad_validate_end_to_end_operations: BackendHook | None = None
    scratchpad_has_operation_scope: BackendHook | None = None
    scratchpad_allocate_operation_scope: BackendHook | None = None
    scratchpad_validate_operation_scopes: BackendHook | None = None
    prepare_execution_fusion: BackendHook | None = None
    classify_scheduler_operation: BackendHook | None = None
    pad_matrix_output_operands: BackendHook | None = None
    prepare_quantization_recalibration: BackendHook | None = None
    prepare_quantization_dynamic_qat: BackendHook | None = None
    prepare_quantization_qat: BackendHook | None = None
    refresh_quantization_qat: BackendHook | None = None
    cleanup_quantization_preparation: BackendHook | None = None
    run_tiled_conv2d: BackendHook | None = None
    prepare_serialized_artifacts: BackendHook | None = None
    prepare_serialization_metadata: BackendHook | None = None
    validate_serialization_metadata: BackendHook | None = None

    def __post_init__(self):
        if not self.target:
            raise ValueError("backend target must not be empty")
        if not isinstance(self.operator_traits, TargetOperatorTraits):
            raise TypeError(
                "operator_traits must be a TargetOperatorTraits instance"
            )
        if not isinstance(self.hardware_config_type, type) or not issubclass(
            self.hardware_config_type,
            HardwareConfig,
        ):
            raise TypeError(
                "hardware_config_type must derive from HardwareConfig"
            )
        if not isinstance(
            self.scheduler_providers,
            SchedulerProviderSet,
        ):
            raise TypeError(
                "scheduler_providers must be a SchedulerProviderSet"
            )
        required_hooks = (
            "apply_graph_rewrites",
            "apply_quantization_rewrites",
            "apply_memory_rewrites",
            "prepare_quantization",
            "finalize_quantization_calibration",
            "create_scratchpad_allocator",
        )
        optional_hooks = (
            "scratchpad_validate_end_to_end_operations",
            "scratchpad_has_operation_scope",
            "scratchpad_allocate_operation_scope",
            "scratchpad_validate_operation_scopes",
            "prepare_execution_fusion",
            "classify_scheduler_operation",
            "pad_matrix_output_operands",
            "prepare_quantization_recalibration",
            "prepare_quantization_dynamic_qat",
            "prepare_quantization_qat",
            "refresh_quantization_qat",
            "cleanup_quantization_preparation",
            "run_tiled_conv2d",
            "prepare_serialized_artifacts",
            "prepare_serialization_metadata",
            "validate_serialization_metadata",
        )
        for name in required_hooks:
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        for name in optional_hooks:
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None")
        operation_scope_hooks = (
            self.scratchpad_has_operation_scope,
            self.scratchpad_allocate_operation_scope,
            self.scratchpad_validate_operation_scopes,
        )
        if any(hook is not None for hook in operation_scope_hooks) and not all(
            hook is not None for hook in operation_scope_hooks
        ):
            raise ValueError(
                "operation-scoped scratchpad allocation requires has, "
                "allocate, and validate hooks"
            )
        hardware_pipeline = tuple(
            tuple(pattern) for pattern in self.hardware_pipeline
        )
        for pattern in hardware_pipeline:
            if not pattern:
                raise ValueError("hardware-pipeline patterns must not be empty")
            for stage in pattern:
                if not callable(
                    getattr(stage, "matches", None)
                ) or not isinstance(getattr(stage, "targets", None), tuple):
                    raise TypeError(
                        "hardware-pipeline stages must be OpMatcher-compatible"
                    )
        object.__setattr__(self, "hardware_pipeline", hardware_pipeline)
        object.__setattr__(
            self,
            "execution_fusion_patterns",
            tuple(self.execution_fusion_patterns),
        )


EMPTY_TARGET_OPERATOR_TRAITS = TargetOperatorTraits()


__all__ = [
    "EMPTY_TARGET_OPERATOR_TRAITS",
    "TargetBackendSpec",
    "TargetOperatorTraits",
]

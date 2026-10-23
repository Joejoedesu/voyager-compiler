# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compilation-target backend registration and dispatch."""

from collections.abc import Mapping
from types import MappingProxyType

from scheduler.dispatch import GENERIC_SCHEDULER_PROVIDERS
from voyager_compiler.codegen.hardware import HardwareConfig
from voyager_compiler.codegen.memory import ScratchpadAllocator
from voyager_compiler.codegen.target_backends import (
    gemmini,
    magnet_alpha,
)
from voyager_compiler.codegen.target_backends.spec import (
    EMPTY_TARGET_OPERATOR_TRAITS,
    TargetBackendSpec,
    TargetOperatorTraits,
)
from voyager_compiler.codegen.targets import (
    COMPILATION_TARGETS,
    DEFAULT_COMPILATION_TARGET,
)


def _build_backend_registry(*backends):
    registry = {}
    for backend in backends:
        if backend.target in registry:
            raise RuntimeError(
                "Duplicate compilation target backend: " f"{backend.target}"
            )
        registry[backend.target] = backend
    return MappingProxyType(registry)


# Architecture bindings live here; generic mapping only calls the factory.
_TARGET_BACKENDS = _build_backend_registry(
    gemmini.BACKEND_SPEC,
    magnet_alpha.BACKEND_SPEC,
)


def _validate_backend_targets():
    expected = set(COMPILATION_TARGETS) - {
        DEFAULT_COMPILATION_TARGET,
    }
    registered = set(_TARGET_BACKENDS)
    if registered != expected:
        missing = sorted(expected - registered)
        unexpected = sorted(registered - expected)
        raise RuntimeError(
            "Compilation target/backend registry mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


_validate_backend_targets()


TARGET_REWRITE_STAGES = (
    "graph",
    "quantization",
    "memory",
)


def _target_backend(target):
    if target == DEFAULT_COMPILATION_TARGET:
        return None
    backend = _TARGET_BACKENDS.get(target)
    if backend is None:
        raise ValueError(f"Unsupported compilation target: {target}")
    return backend


def get_target_backend_spec(target):
    """Return the selected typed backend entry, or ``None`` for generic."""
    return _target_backend(target)


def get_target_operator_traits(target):
    """Return operator traits for only the selected target."""
    backend = _target_backend(target)
    if backend is None:
        return EMPTY_TARGET_OPERATOR_TRAITS
    return backend.operator_traits


def get_target_scheduler_providers(target):
    """Return scheduler engines bound by the selected compiler target."""
    backend = _target_backend(target)
    if backend is None:
        return GENERIC_SCHEDULER_PROVIDERS
    return backend.scheduler_providers


def classify_target_scheduler_operation(
    operator,
    target,
    operands,
):
    """Freeze selected-target operation semantics for scheduler dispatch."""
    backend = _target_backend(target)
    if backend is None:
        return None
    classify = backend.classify_scheduler_operation
    if classify is None:
        return None
    return classify(operator, operands)


def validate_target_hardware_config(
    target,
    hardware_config,
):
    backend = _target_backend(target)
    if hardware_config is None:
        if backend is not None:
            raise TypeError(
                f"{target} target requires "
                f"{backend.hardware_config_type.__name__}"
            )
        return

    if not isinstance(hardware_config, HardwareConfig):
        raise TypeError("hardware_config must be a HardwareConfig instance")
    if hardware_config.target != target:
        raise ValueError(
            f"Compilation target {target!r} does not match "
            f"hardware_config.target {hardware_config.target!r}"
        )
    if backend is not None and not isinstance(
        hardware_config,
        backend.hardware_config_type,
    ):
        raise TypeError(
            f"{target} target requires "
            f"{backend.hardware_config_type.__name__}, got "
            f"{type(hardware_config).__name__}"
        )


def _apply_target_stage(model, target, stage, **kwargs):
    """Apply one explicitly ordered target-owned compiler stage."""
    if stage not in TARGET_REWRITE_STAGES:
        raise ValueError(f"Unknown target rewrite stage: {stage}")
    backend = _target_backend(target)
    if backend is None:
        return model
    if stage == "graph":
        rewrite = backend.apply_graph_rewrites
    elif stage == "quantization":
        rewrite = backend.apply_quantization_rewrites
    else:
        rewrite = backend.apply_memory_rewrites
    if rewrite is None:
        return model
    return rewrite(model, **kwargs)


def apply_target_graph_rewrites(
    model,
    target,
    hardware_config=None,
):
    """Apply the selected target's idempotent structural graph rewrites."""
    validate_target_hardware_config(target, hardware_config)
    return _apply_target_stage(model, target, "graph")


def prepare_target_execution_fusion(
    model,
    target,
    hardware_config=None,
):
    """Canonicalize the selected target graph immediately before fusion."""
    validate_target_hardware_config(target, hardware_config)
    backend = _target_backend(target)
    if backend is None or backend.prepare_execution_fusion is None:
        return model
    return backend.prepare_execution_fusion(model)


def lower_target_quantization(
    model,
    target,
    hardware_config=None,
):
    """Lower reference quantization into the selected target IR."""
    validate_target_hardware_config(target, hardware_config)
    return _apply_target_stage(model, target, "quantization")


def apply_target_memory_rewrites(
    model,
    target,
    hardware_config=None,
):
    """Prepare the selected target's physical-memory graph."""
    validate_target_hardware_config(target, hardware_config)
    return _apply_target_stage(
        model,
        target,
        "memory",
        hardware=hardware_config,
    )


def pad_target_matrix_output_operands(
    model,
    node,
    output_channels,
    output_padding,
    target,
):
    """Extend target-owned per-output operands after matrix K padding."""
    backend = _target_backend(target)
    if backend is None:
        return
    pad_operands = backend.pad_matrix_output_operands
    if pad_operands is not None:
        pad_operands(
            model,
            node,
            output_channels,
            output_padding,
        )


def prepare_target_quantization(
    model,
    target,
    region_qspec=None,
    hardware_config=None,
):
    """Apply target-owned preparation around generic PT2E observers."""
    validate_target_hardware_config(target, hardware_config)
    backend = _target_backend(target)
    if backend is None:
        return model
    return backend.prepare_quantization(
        model,
        region_qspec,
        hardware_config=hardware_config,
        operator_traits=backend.operator_traits,
    )


def finalize_target_quantization_calibration(model, target):
    """Finalize target-owned observer domains before PT2E conversion."""
    backend = _target_backend(target)
    if backend is not None:
        backend.finalize_quantization_calibration(model)
    return model


def cleanup_target_quantization_preparation(model, target):
    """Remove target-owned prepared state before generic PT2E conversion."""
    backend = _target_backend(target)
    if backend is None:
        return model
    cleanup = backend.cleanup_quantization_preparation
    return model if cleanup is None else cleanup(model)


def prepare_target_quantization_recalibration(
    model,
    target,
    *,
    reset_observers: bool,
) -> int:
    """Prepare an optional target-exact second calibration pass."""
    backend = _target_backend(target)
    if backend is None:
        return 0
    prepare = backend.prepare_quantization_recalibration
    return (
        0
        if prepare is None
        else prepare(
            model,
            reset_observers=reset_observers,
        )
    )


def prepare_target_quantization_qat(model, target) -> int:
    """Enter a target's exact, trainable fake-forward phase."""
    backend = _target_backend(target)
    if backend is None:
        return 0
    prepare = backend.prepare_quantization_qat
    return 0 if prepare is None else prepare(model)


def prepare_target_quantization_dynamic_qat(model, target) -> int:
    """Select a target's pre-exact, trainable fake-forward behavior."""
    backend = _target_backend(target)
    if backend is None:
        return 0
    prepare = backend.prepare_quantization_dynamic_qat
    return 0 if prepare is None else prepare(model)


def refresh_target_quantization_qat(model, target) -> int:
    """Refresh derived exact operands without changing the prepared graph."""
    backend = _target_backend(target)
    if backend is None:
        return 0
    refresh = backend.refresh_quantization_qat
    return 0 if refresh is None else refresh(model)


def run_target_tiled_conv2d(
    target,
    operator_target,
    args,
    scales,
    folded_coarse_scale,
    block_size,
    semantic_coarse_scales=(None, None),
    *,
    operator_kwargs=None,
):
    """Execute a tiled convolution through the selected target backend."""
    backend = _target_backend(target)
    if (
        backend is None
        or operator_target not in backend.operator_traits.tiled_conv2d_operators
    ):
        raise ValueError(
            f"Target {target!r} does not support tiled convolution "
            f"{operator_target}"
        )
    if backend.run_tiled_conv2d is None:
        raise ValueError(
            f"Target {target!r} does not provide tiled convolution"
        )
    if operator_kwargs is not None and not isinstance(
        operator_kwargs,
        Mapping,
    ):
        raise TypeError("tiled operator_kwargs must be a mapping")
    operator_kwargs = MappingProxyType(dict(operator_kwargs or {}))
    return backend.run_tiled_conv2d(
        operator_target,
        args,
        scales,
        folded_coarse_scale,
        block_size,
        semantic_coarse_scales,
        operator_kwargs,
    )


def target_allows_qdq_fusion(target):
    """Return whether generic Q/DQ folding is legal for the target."""
    backend = _target_backend(target)
    if backend is None:
        return True
    return backend.allow_qdq_fusion


def get_target_execution_fusion_patterns(target):
    """Return target-required physical execution-region patterns."""
    backend = _target_backend(target)
    if backend is None:
        return ()
    return backend.execution_fusion_patterns


def get_target_hardware_pipeline(target):
    """Return semantic FX matcher stages owned by ``target``'s hardware.

    The generic target remains a named source-owned profile even though it
    needs no backend hooks.  Concrete targets register their own view; a
    hardware pipeline is never inferred from a converter capture window.
    """

    backend = _target_backend(target)
    if backend is None:
        from voyager_compiler.codegen.target_backends.generic import (
            GENERIC_HARDWARE_PIPELINE,
        )

        return GENERIC_HARDWARE_PIPELINE
    return backend.hardware_pipeline


def target_supports_bufferized(target):
    """Return whether the target supports the bufferized terminal path."""
    backend = _target_backend(target)
    if backend is None:
        return True
    return backend.supports_bufferized


def target_uses_generic_aggregate_copies(target):
    """Return whether shared cat/stack copy materialization should run."""
    backend = _target_backend(target)
    if backend is None:
        return True
    return backend.uses_generic_aggregate_copies


def prepare_target_serialized_artifacts(
    model,
    target,
    storage_graph,
    storage_plane,
    storage_allocator,
) -> None:
    """Declare nonresident target artifacts before storage finalization."""
    backend = _target_backend(target)
    if backend is None:
        return
    prepare = backend.prepare_serialized_artifacts
    if prepare is not None:
        prepare(
            model,
            storage_graph,
            storage_plane,
            storage_allocator,
        )


def prepare_target_serialization_metadata(
    model,
    target,
    storage_allocator,
    hardware_config,
) -> None:
    """Prepare target wire facts after physical memory planning."""
    backend = _target_backend(target)
    if backend is None:
        return
    prepare = backend.prepare_serialization_metadata
    if prepare is not None:
        prepare(model, storage_allocator, hardware_config)


def validate_target_serialization_metadata(model, target) -> None:
    """Validate selected-target facts immediately before dump emission."""
    backend = _target_backend(target)
    if backend is None:
        return
    validate = backend.validate_serialization_metadata
    if validate is not None:
        validate(model)


def target_supports_operation_scoped_scratchpad(target) -> bool:
    """Return whether the selected backend registered scoped allocation."""
    backend = _target_backend(target)
    return (
        backend is not None
        and backend.scratchpad_has_operation_scope is not None
    )


def validate_target_end_to_end_scratchpad_operations(
    target,
    allocator,
    model,
) -> None:
    """Run the selected backend's registered whole-model operation gate."""
    backend = _target_backend(target)
    if (
        backend is not None
        and backend.scratchpad_validate_end_to_end_operations is not None
    ):
        backend.scratchpad_validate_end_to_end_operations(allocator, model)


def target_has_operation_scope(target, allocator, operation) -> bool:
    """Return whether one operation has a registered target-local arena."""
    backend = _target_backend(target)
    if backend is None or backend.scratchpad_has_operation_scope is None:
        return False
    result = backend.scratchpad_has_operation_scope(allocator, operation)
    if not isinstance(result, bool):
        raise TypeError("scratchpad_has_operation_scope must return a bool")
    return result


def allocate_target_operation_scope(target, allocator, operation):
    """Allocate one operation through the selected backend's explicit hook."""
    backend = _target_backend(target)
    if backend is None or backend.scratchpad_allocate_operation_scope is None:
        raise ValueError(
            f"Target {target!r} does not support operation-scoped allocation"
        )
    return backend.scratchpad_allocate_operation_scope(
        allocator,
        operation,
    )


def validate_target_operation_scopes(target, allocator, model) -> None:
    """Validate all operation-local arenas through target dispatch."""
    backend = _target_backend(target)
    if backend is None or backend.scratchpad_validate_operation_scopes is None:
        return
    backend.scratchpad_validate_operation_scopes(allocator, model)


def create_scratchpad_allocator(hardware: HardwareConfig):
    """Select a model-wide allocator for one hardware target."""
    if not isinstance(hardware, HardwareConfig):
        raise TypeError("hardware must be a HardwareConfig instance")
    validate_target_hardware_config(hardware.target, hardware)
    backend = _target_backend(hardware.target)
    if backend is not None:
        return backend.create_scratchpad_allocator(hardware)
    return ScratchpadAllocator(
        hardware.global_memory,
        hardware_config=hardware,
    )


__all__ = [
    "EMPTY_TARGET_OPERATOR_TRAITS",
    "TARGET_REWRITE_STAGES",
    "TargetBackendSpec",
    "TargetOperatorTraits",
    "allocate_target_operation_scope",
    "apply_target_graph_rewrites",
    "apply_target_memory_rewrites",
    "pad_target_matrix_output_operands",
    "create_scratchpad_allocator",
    "classify_target_scheduler_operation",
    "cleanup_target_quantization_preparation",
    "finalize_target_quantization_calibration",
    "prepare_target_quantization_recalibration",
    "prepare_target_quantization_dynamic_qat",
    "prepare_target_quantization_qat",
    "refresh_target_quantization_qat",
    "get_target_execution_fusion_patterns",
    "get_target_hardware_pipeline",
    "get_target_backend_spec",
    "get_target_operator_traits",
    "get_target_scheduler_providers",
    "prepare_target_serialized_artifacts",
    "prepare_target_serialization_metadata",
    "prepare_target_quantization",
    "prepare_target_execution_fusion",
    "lower_target_quantization",
    "target_allows_qdq_fusion",
    "target_has_operation_scope",
    "target_supports_operation_scoped_scratchpad",
    "target_supports_bufferized",
    "target_uses_generic_aggregate_copies",
    "validate_target_serialization_metadata",
    "validate_target_hardware_config",
    "validate_target_end_to_end_scratchpad_operations",
    "validate_target_operation_scopes",
    "run_target_tiled_conv2d",
]

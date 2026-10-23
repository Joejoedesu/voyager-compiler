# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Bind final Magnet Alpha FX operations to configured compute units."""

from torch.fx import GraphModule, Node

from scheduler.types import OperationClassification
from voyager_compiler.codegen.mapping_utils import is_gemm_op
from voyager_compiler.codegen.target_backends.magnet_alpha.graph import (
    apply_memory_rewrites as insert_memory_rewrites,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.hardware import (
    MagnetAlphaHardwareConfig,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.operators import (
    OPERATION_CLASSIFICATION_META,
    OPERATION_COMPUTE_UNIT_META,
    classify_scheduler_operation,
    magnet_alpha_operation_classification,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.traits import (
    MAGNET_ALPHA_OPERATOR_TRAITS,
)


def _graph_modules(model: GraphModule):
    yield from (
        module for module in model.modules() if isinstance(module, GraphModule)
    )


def _operation_binding(
    node: Node,
    hardware: MagnetAlphaHardwareConfig,
) -> tuple[str, OperationClassification] | None:
    classification = classify_scheduler_operation(node, use_bound=False)
    if classification is not None:
        unit = hardware.nmp_compute_unit
        if hardware.compute_unit(unit).supports_operation(classification.kind):
            return unit, classification

    if is_gemm_op(node, MAGNET_ALPHA_OPERATOR_TRAITS):
        unit = hardware.pe.name
        if hardware.compute_unit(unit).supports_operation("gemm"):
            return unit, magnet_alpha_operation_classification("gemm")
    return None


def bound_operation_classification(
    node: Node,
) -> tuple[str, OperationClassification] | None:
    """Return the final target binding attached to one FX operation."""
    magnet_alpha = node.meta.get("magnet_alpha", {})
    classification = magnet_alpha.get(OPERATION_CLASSIFICATION_META)
    unit = magnet_alpha.get(OPERATION_COMPUTE_UNIT_META)
    if classification is None and unit is None:
        return None
    if not isinstance(classification, OperationClassification):
        raise TypeError(
            f"{node.name}: invalid Magnet Alpha operation classification"
        )
    if not isinstance(unit, str) or not unit:
        raise TypeError(f"{node.name}: invalid Magnet Alpha compute unit")
    return unit, classification


def _clear_operation_binding(node: Node) -> None:
    magnet_alpha = node.meta.get("magnet_alpha")
    if not isinstance(magnet_alpha, dict):
        return
    previous_unit = magnet_alpha.pop(OPERATION_COMPUTE_UNIT_META, None)
    magnet_alpha.pop(OPERATION_CLASSIFICATION_META, None)
    if (
        previous_unit is not None
        and node.meta.get("compute_unit") == previous_unit
    ):
        node.meta.pop("compute_unit")
    if not magnet_alpha:
        node.meta.pop("magnet_alpha")


def bind_operation_classifications(
    model: GraphModule,
    hardware: MagnetAlphaHardwareConfig,
) -> GraphModule:
    """Classify every final FX call once before storage planning."""
    if not isinstance(hardware, MagnetAlphaHardwareConfig):
        raise TypeError(
            "Magnet Alpha operation binding requires MagnetAlphaHardwareConfig"
        )

    for graph_module in _graph_modules(model):
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            _clear_operation_binding(node)
            binding = _operation_binding(node, hardware)
            if binding is None:
                continue
            unit, classification = binding
            magnet_alpha = node.meta.setdefault("magnet_alpha", {})
            magnet_alpha[OPERATION_CLASSIFICATION_META] = classification
            magnet_alpha[OPERATION_COMPUTE_UNIT_META] = unit
            node.meta["compute_unit"] = unit

    validate_operation_classifications(model, hardware)
    return model


def validate_operation_classifications(
    model: GraphModule,
    hardware: MagnetAlphaHardwareConfig,
) -> None:
    """Validate that every attached binding is supported and current."""
    if not isinstance(hardware, MagnetAlphaHardwareConfig):
        raise TypeError(
            "Magnet Alpha operation validation requires MagnetAlphaHardwareConfig"
        )

    for graph_module in _graph_modules(model):
        for node in graph_module.graph.nodes:
            if node.op != "call_function":
                continue
            binding = bound_operation_classification(node)
            if binding is None:
                continue
            unit, classification = binding
            if node.meta.get("compute_unit") != unit:
                raise ValueError(
                    f"{node.name}: Magnet Alpha compute-unit metadata disagrees"
                )
            if not hardware.compute_unit(unit).supports_operation(
                classification.kind
            ):
                raise ValueError(
                    f"{node.name}: {unit} does not support "
                    f"{classification.kind}"
                )
            expected = _operation_binding(node, hardware)
            if expected != binding:
                raise ValueError(
                    f"{node.name}: stale Magnet Alpha operation classification"
                )


def apply_memory_rewrites(
    model: GraphModule,
    hardware: MagnetAlphaHardwareConfig,
) -> GraphModule:
    """Insert physical operations, then bind the resulting final FX graph."""
    model = insert_memory_rewrites(model, hardware)
    return bind_operation_classifications(model, hardware)


__all__ = [
    "OPERATION_CLASSIFICATION_META",
    "OPERATION_COMPUTE_UNIT_META",
    "apply_memory_rewrites",
    "bind_operation_classifications",
    "bound_operation_classification",
    "validate_operation_classifications",
]

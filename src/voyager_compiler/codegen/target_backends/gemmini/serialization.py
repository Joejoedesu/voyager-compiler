# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Validate shared Gemmini semantics before canonical protobuf emission."""

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.mapping_utils import (
    LOCAL_SCHEDULE,
    LOCAL_SCHEDULER_WORKLOAD,
    TARGET_CODEGEN_KWARGS,
)
from voyager_compiler.codegen.target_backends.gemmini.lowering import (
    _has_public_injection_shape,
    gemmini_injection_region,
    gemmini_workload_add_regions,
    gemmini_workload_average_regions,
    gemmini_workload_linear_regions,
    gemmini_workload_norm_regions,
    gemmini_workload_regions,
)


class GemminiSerializationError(RuntimeError):
    """Raised when final Gemmini FX cannot use the canonical helper ABI."""


_INT8_QMAP_ROLES = frozenset({"input_quantize", "output_quantize"})
_CODE_DEQUANTIZE_ROLES = frozenset(
    {
        "input_a_dequantize",
        "input_b_dequantize",
        "average_input_dequantize",
    }
)


def _node_input(node: Node, index: int, name: str):
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(name)


def _placeholder_binding(region, placeholder: Node) -> object:
    placeholders = tuple(
        node
        for node in region.submodule.graph.nodes
        if node.op == "placeholder"
    )
    try:
        index = placeholders.index(placeholder)
    except ValueError as error:
        raise GemminiSerializationError(
            f"{placeholder.name}: qmap is not a fused-region placeholder"
        ) from error
    if index < len(region.outer.args):
        return region.outer.args[index]
    name = str(placeholder.target)
    if name in region.outer.kwargs:
        return region.outer.kwargs[name]
    raise GemminiSerializationError(
        f"{region.outer.name}: missing binding for qmap placeholder {name}"
    )


def _get_attr(owner: GraphModule, node: object, label: str) -> object:
    if not isinstance(node, Node) or node.op != "get_attr":
        raise GemminiSerializationError(
            f"{label}: qmap must resolve from a final-FX get_attr"
        )
    value: object = owner
    for field in str(node.target).split("."):
        if not hasattr(value, field):
            raise GemminiSerializationError(
                f"{label}: qmap attribute {node.target!s} is missing"
            )
        value = getattr(value, field)
    return value


def _resolve_region_qmap(
    model: GraphModule,
    region,
    operand: object,
    label: str,
) -> object | None:
    if operand is None:
        return None
    if isinstance(operand, Node) and operand.op == "placeholder":
        operand = _placeholder_binding(region, operand)
        return _get_attr(model, operand, label)
    if isinstance(operand, Node) and operand.op == "get_attr":
        owner = (
            region.submodule
            if operand.graph is region.submodule.graph
            else model
        )
        return _get_attr(owner, operand, label)
    raise GemminiSerializationError(
        f"{label}: qmap must be None or a final-FX attribute"
    )


def _canonical_qmap(
    dtype: str | None,
    device: torch.device,
) -> torch.Tensor:
    # Lazy import avoids the package initialization cycle through codegen.
    from voyager_compiler.fake_quantize import get_quantization_map

    return get_quantization_map(dtype, device=device)


def _require_canonical_qmap(
    value: object,
    dtype: str | None,
    label: str,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise GemminiSerializationError(f"{label}: qmap must be a tensor")
    expected = _canonical_qmap(dtype, value.device)
    matches = (
        value.dtype == expected.dtype
        and value.shape == expected.shape
        and torch.equal(
            value.contiguous().view(torch.int16),
            expected.contiguous().view(torch.int16),
        )
    )
    if not matches:
        map_name = "identity" if dtype is None else dtype
        raise GemminiSerializationError(
            f"{label}: qmap is not the canonical {map_name} map"
        )


def _validate_quantize_qmap(
    model: GraphModule,
    region,
    node: Node,
    role: str,
) -> None:
    label = f"{region.outer.name}:{role}:{node.name}"
    value = _resolve_region_qmap(
        model,
        region,
        _node_input(node, 5, "qmap"),
        label,
    )
    _require_canonical_qmap(value, "int8", label)


def _validate_dequantize_qmaps(
    model: GraphModule,
    region,
    node: Node,
    role: str,
) -> None:
    label = f"{region.outer.name}:{role}:{node.name}"
    input_qmap = _resolve_region_qmap(
        model,
        region,
        _node_input(node, 5, "input_qmap"),
        f"{label}:input_qmap",
    )
    output_qmap = _resolve_region_qmap(
        model,
        region,
        _node_input(node, 6, "output_qmap"),
        f"{label}:output_qmap",
    )
    if role == "accumulator_dequantize":
        # PT2E's native accumulator boundary uses output_dtype=None, hence the
        # raw BF16 identity map. A float32 round-trip differs at NaN payloads.
        _require_canonical_qmap(
            input_qmap,
            None,
            f"{label}:input_qmap",
        )
        if output_qmap is not None:
            raise GemminiSerializationError(
                f"{label}: output_qmap is not supported by the helper ABI"
            )
        return
    if role in _CODE_DEQUANTIZE_ROLES:
        if input_qmap is not None or output_qmap is not None:
            raise GemminiSerializationError(
                f"{label}: qmaps are not supported by the helper ABI"
            )
        return
    raise GemminiSerializationError(
        f"{label}: no Gemmini qmap contract is declared for this DQ stage"
    )


def _validate_region_qmaps(model: GraphModule, region) -> None:
    roles = region.member_roles
    if len(roles) != len(region.members):
        raise GemminiSerializationError(
            f"{region.outer.name}: member roles do not match fused members"
        )
    for role, node in zip(roles, region.members, strict=True):
        if role in _INT8_QMAP_ROLES:
            _validate_quantize_qmap(model, region, node, role)
        elif node.target == torch.ops.quantized_ops.dequantize.default:
            _validate_dequantize_qmaps(model, region, node, role)


def _validate_terminal_dequantize_qmaps(
    model: GraphModule,
    node: Node,
) -> None:
    for index, name in ((5, "input_qmap"), (6, "output_qmap")):
        operand = _node_input(node, index, name)
        if operand is None:
            continue
        value = _get_attr(model, operand, f"{node.name}:{name}")
        if value is not None:
            raise GemminiSerializationError(
                f"{node.name}: terminal {name} is not supported by the "
                "helper ABI"
            )


def validate_gemmini_pipeline_qmaps(model: GraphModule) -> None:
    """Fail before serialization can erase target-significant qmap fields."""
    if _has_public_injection_shape(model):
        region = gemmini_injection_region(model)
        _validate_region_qmaps(model, region)
        _validate_terminal_dequantize_qmaps(
            model,
            region.terminal_dequantize,
        )
        return
    regions = (
        *gemmini_workload_regions(model),
        *gemmini_workload_add_regions(model),
        *gemmini_workload_linear_regions(model),
        *gemmini_workload_average_regions(model),
    )
    for region in regions:
        _validate_region_qmaps(model, region)


def _reject_local_control(node: Node) -> None:
    forbidden = {
        LOCAL_SCHEDULE: node.meta.get(LOCAL_SCHEDULE),
        LOCAL_SCHEDULER_WORKLOAD: node.meta.get(LOCAL_SCHEDULER_WORKLOAD),
        "l2_tiling": node.meta.get("l2_tiling"),
        "tiled_shapes": node.meta.get("tiled_shapes"),
        "tile_strides": node.meta.get("tile_strides"),
        "physical_scratchpad_map": node.meta.get("physical_scratchpad_map"),
    }
    present = tuple(
        name for name, value in forbidden.items() if value is not None
    )
    if present:
        raise GemminiSerializationError(
            f"{node.name}: target-local control reached serialization: "
            + ", ".join(present)
        )
    target_kwargs = node.meta.get(TARGET_CODEGEN_KWARGS, {})
    if not isinstance(target_kwargs, dict):
        raise GemminiSerializationError(
            f"{node.name}: target codegen kwargs must be a dictionary"
        )
    if target_kwargs:
        raise GemminiSerializationError(
            f"{node.name}: Gemmini target ABI fields are forbidden"
        )


def _serialization_regions(model: GraphModule):
    """Return controlled nodes and scheduler skips for the graph shape."""
    norm_regions = gemmini_workload_norm_regions(model)
    if norm_regions:
        names = ", ".join(region.outer.name for region in norm_regions)
        raise GemminiSerializationError(
            "tiled_norm_auto binding cannot serialize without target "
            f"HAS_NORMALIZATIONS support: {names}"
        )
    if _has_public_injection_shape(model):
        region = gemmini_injection_region(model)
        return (
            (region.outer, *region.members, region.terminal_dequantize),
            (region.outer, region.terminal_dequantize),
        )

    regions = (
        *gemmini_workload_regions(model),
        *gemmini_workload_add_regions(model),
        *gemmini_workload_linear_regions(model),
        *gemmini_workload_average_regions(model),
    )
    controlled_nodes = tuple(
        node for region in regions for node in (region.outer, *region.members)
    )
    scheduler_skips = tuple(region.outer for region in regions)
    return controlled_nodes, scheduler_skips


def prepare_canonical_serialization_metadata(
    model: GraphModule,
    storage_allocator,
    hardware_config,
) -> None:
    """Remove only empty generic-scheduler skip markers for Conv regions."""
    del storage_allocator, hardware_config
    validate_gemmini_pipeline_qmaps(model)
    controlled_nodes, scheduler_skips = _serialization_regions(model)
    for node in controlled_nodes:
        _reject_local_control(node)
    for node in scheduler_skips:
        if node.meta.get("scratchpad_map") != {}:
            raise GemminiSerializationError(
                f"{node.name}: missing empty Gemmini scheduler-skip marker"
            )
        node.meta.pop("scratchpad_map")


def validate_serialization_metadata(model: GraphModule) -> None:
    """Fail closed unless canonical FX preserves shared semantics only."""
    validate_gemmini_pipeline_qmaps(model)
    controlled_nodes, scheduler_skips = _serialization_regions(model)
    for node in controlled_nodes:
        _reject_local_control(node)
    for node in scheduler_skips:
        if node.meta.get("scratchpad_map") is not None:
            raise GemminiSerializationError(
                f"{node.name}: scratchpad marker reached canonical emission"
            )


__all__ = [
    "GemminiSerializationError",
    "prepare_canonical_serialization_metadata",
    "validate_gemmini_pipeline_qmaps",
    "validate_serialization_metadata",
]

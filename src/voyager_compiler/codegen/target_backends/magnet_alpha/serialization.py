# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Magnet Alpha compiler serialization facts and validation."""

from dataclasses import dataclass

import torch
from torch.fx import GraphModule, Node

from scheduler.types import Loop, OperationWorkload
from voyager_compiler.codegen.artifacts import (
    discard_supplemental_artifact,
    set_supplemental_artifact,
)
from voyager_compiler.codegen.execution_order import execution_order_for_model
from voyager_compiler.codegen.hardware import get_dtype_width
from voyager_compiler.codegen.mapping_utils import (
    LOCAL_SCHEDULE,
    LOCAL_SCHEDULER_WORKLOAD,
    SCHEDULE_FACTOR_PREFIX,
    TARGET_CODEGEN_KWARGS,
    is_nop,
)
from voyager_compiler.codegen.storage import (
    SerializedTensorRef,
    StorageGraph,
    ValueRef,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.graph import (
    is_shuffle_copy,
    shuffle_copy_movements,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.lowering import (
    bound_operation_classification,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.operators import (
    PVS_QUANTIZE_OPERATOR,
)

_SERIALIZED_ARTIFACT_BINDINGS = "magnet_alpha_serialized_artifact_bindings"
_BANK_PLACEMENTS_ARTIFACT = "magnet_alpha_bank_placements"
_NMP_CLASSIFICATIONS = frozenset(
    {
        "channel_shuffle",
        "channel_shuffle_copy",
        "depthwise_conv2d",
        "group_norm",
        "pool2d",
        "softmax",
        "upsample_nearest2d",
    }
)
_GEMM_RECORD_COVERAGE_FIELDS = (
    "padding_c",
    "padding_k",
    "materialized_input_valid_channels",
    "materialized_input_resident_channels",
    "weight_valid_input_channels",
)


class MagnetAlphaSerializationError(RuntimeError):
    """Raised when canonical Magnet Alpha serialization facts conflict."""


@dataclass(frozen=True)
class _SerializedArtifactBinding:
    """One operation field backed by a nonresident target payload."""

    operation: Node
    field: str
    logical_tensor: ValueRef
    plane: str
    artifact: str


def _serialization_operations(model: GraphModule) -> tuple[Node, ...]:
    """Return the top-graph operations that become serialized records."""
    return tuple(
        operation
        for operation in execution_order_for_model(model).nodes
        if operation.op in {"call_function", "call_module"}
        and isinstance(
            getattr(operation, "value", None),
            (torch.Tensor, tuple, list),
        )
    )


def _serialized_execution_nodes(
    model: GraphModule,
    operation: Node,
) -> tuple[Node, ...]:
    """Return executable nodes serialized inside one top-graph record."""
    if operation.op == "call_function":
        return (operation,)
    if operation.op != "call_module":
        return ()
    submodule = model.get_submodule(str(operation.target))
    return tuple(
        node
        for node in submodule.graph.nodes
        if (
            node.op == "call_function"
            and not node.meta.get("fused", False)
            and not is_nop(node)
        )
    )


def _gemm_record_target_kwargs(
    model: GraphModule,
    operation: Node,
    target_kwargs: dict,
    error_type: type[RuntimeError],
) -> dict:
    """Resolve schedule-coverage facts across one fused GEMM record.

    Local scheduling is bound to the top-level fused record, while physical
    convolution padding is owned by its executable inner convolution.  Merge
    the two views before validating that the schedule covers the physical
    call exactly.
    """
    resolved = dict(target_kwargs)
    if operation.op != "call_module":
        return resolved
    submodule = model.get_submodule(str(operation.target))
    for record_node in submodule.graph.nodes:
        if record_node.op != "call_function":
            continue
        execution_kwargs = record_node.meta.get(
            TARGET_CODEGEN_KWARGS,
            {},
        )
        for field in _GEMM_RECORD_COVERAGE_FIELDS:
            if field not in execution_kwargs:
                continue
            value = execution_kwargs[field]
            if field in resolved and resolved[field] != value:
                raise error_type(
                    f"{operation.name}: conflicting serialized {field}"
                )
            resolved[field] = value
    return resolved


def _normalized_operand_bits(width: int) -> int:
    return 4 if width <= 4 else 8


def _logical_value(storage_graph: StorageGraph, logical_tensor):
    logical_ref = storage_graph.ref(logical_tensor)
    value = getattr(logical_ref.node, "value", None)
    try:
        for index in logical_ref.path:
            value = value[index]
    except (IndexError, TypeError):
        return None
    return value


def _pad_tensor_axis(
    value: torch.Tensor,
    axis: int,
    extent: int,
) -> torch.Tensor:
    if extent < value.shape[axis]:
        raise MagnetAlphaSerializationError(
            "serialized tensor extent cannot shrink"
        )
    if extent == value.shape[axis]:
        return value
    shape = list(value.shape)
    shape[axis] = extent
    result = value.new_zeros(shape)
    slices = [slice(None)] * value.ndim
    slices[axis] = slice(0, value.shape[axis])
    result[tuple(slices)] = value
    return result


def _register_serialized_artifact(
    storage_graph: StorageGraph,
    storage_plane,
    bindings: list[_SerializedArtifactBinding],
    operation: Node,
    field: str,
    logical_tensor,
    value: torch.Tensor,
) -> None:
    logical_ref = storage_graph.ref(logical_tensor)
    logical = storage_graph.index.logical_tensors[logical_ref]
    artifact_name = f"{operation.name}.{field}"
    storage_graph.register_serialized_artifact(
        logical_ref,
        value,
        plane=storage_plane,
        name=artifact_name,
        dtype=logical.dtype,
        bits=_normalized_operand_bits(get_dtype_width(logical.dtype)),
    )
    bindings.append(
        _SerializedArtifactBinding(
            operation=operation,
            field=field,
            logical_tensor=logical_ref,
            plane=str(getattr(storage_plane, "value", storage_plane)),
            artifact=artifact_name,
        )
    )


def _declare_vector_scale_artifact(
    storage_graph: StorageGraph,
    storage_plane,
    bindings: list[_SerializedArtifactBinding],
    operation: Node,
    field: str,
    logical_tensor,
    *,
    value: torch.Tensor | None = None,
) -> None:
    """Declare a canonical raw rank-4 vector-scale payload."""
    logical_ref = storage_graph.ref(logical_tensor)
    if value is None:
        value = _logical_value(storage_graph, logical_ref)
    if not isinstance(value, torch.Tensor) or value.ndim != 4:
        raise MagnetAlphaSerializationError(
            f"{logical_ref.name}: vector scale must be rank 4"
        )
    _register_serialized_artifact(
        storage_graph,
        storage_plane,
        bindings,
        operation,
        field,
        logical_ref,
        value.contiguous(),
    )


def _contains_pvs(model: GraphModule, operation: Node) -> bool:
    if operation.target == PVS_QUANTIZE_OPERATOR:
        return True
    if operation.op != "call_module":
        return False
    submodule = model.get_submodule(str(operation.target))
    graph = getattr(submodule, "graph", None)
    return graph is not None and any(
        node.target == PVS_QUANTIZE_OPERATOR for node in graph.nodes
    )


def _declare_operation_scale_artifacts(
    model: GraphModule,
    storage_graph: StorageGraph,
    storage_plane,
    storage_allocator,
    bindings: list[_SerializedArtifactBinding],
) -> None:
    constraints = getattr(storage_allocator, "operation_constraints", None)
    if callable(constraints):
        for operation, constraint in constraints():
            if constraint.kind != "gemm":
                continue
            input_scale = constraint.implementation_operand("input_scale")
            if isinstance(input_scale, Node):
                _declare_vector_scale_artifact(
                    storage_graph,
                    storage_plane,
                    bindings,
                    operation,
                    "input_vector_scale_payload",
                    input_scale,
                )

            weight_scale = constraint.implementation_operand("weight_scale")
            if not isinstance(weight_scale, Node):
                continue
            encoded_k = operation.meta.get("magnet_alpha", {}).get(
                "weight_scale_encoded_output_channels"
            )
            scale_value = _logical_value(storage_graph, weight_scale)
            if not isinstance(encoded_k, int) or encoded_k <= 0:
                raise MagnetAlphaSerializationError(
                    f"{operation.name}: missing encoded weight-scale extent"
                )
            if not isinstance(scale_value, torch.Tensor):
                raise MagnetAlphaSerializationError(
                    f"{operation.name}: GEMM weight scale has no payload"
                )
            _declare_vector_scale_artifact(
                storage_graph,
                storage_plane,
                bindings,
                operation,
                "weight_vector_scale_payload",
                weight_scale,
                value=_pad_tensor_axis(scale_value, 0, encoded_k),
            )

    for operation in model.graph.nodes:
        if not _contains_pvs(model, operation):
            continue
        outputs = {
            output.path: output
            for output in storage_graph.index.node_outputs.get(operation, ())
        }
        fine_scale = outputs.get((1,))
        if fine_scale is None:
            raise MagnetAlphaSerializationError(
                f"{operation.name}: PVS output has no fine-scale component"
            )
        _declare_vector_scale_artifact(
            storage_graph,
            storage_plane,
            bindings,
            operation,
            "output_vector_scale_payload",
            fine_scale,
        )


def _declare_shuffle_copy_artifacts(
    model: GraphModule,
    storage_graph: StorageGraph,
    storage_plane,
    bindings: list[_SerializedArtifactBinding],
) -> None:
    nodes_by_name = {node.name: node for node in model.graph.nodes}
    for copy_node in model.graph.nodes:
        if not is_shuffle_copy(copy_node):
            continue
        metadata = copy_node.meta.get("magnet_alpha", {})
        parent_name = metadata.get("logical_parent")
        block_channels = metadata.get("block_channels")
        copy_index = metadata.get("copy_index")
        shuffle = nodes_by_name.get(parent_name)
        if (
            not isinstance(shuffle, Node)
            or not isinstance(block_channels, int)
            or block_channels <= 0
            or not isinstance(copy_index, int)
        ):
            raise MagnetAlphaSerializationError(
                f"{copy_node.name}: incomplete shuffle-copy declaration"
            )
        shape = getattr(shuffle, "shape", None)
        value = getattr(shuffle, "value", None)
        if (
            not isinstance(value, torch.Tensor)
            or shape is None
            or len(shape) != 4
            or shape[1] % block_channels
        ):
            raise MagnetAlphaSerializationError(
                f"{copy_node.name}: invalid shuffle-copy payload geometry"
            )
        movements = {
            movement.index: movement
            for movement in shuffle_copy_movements(shape[1] // block_channels)
        }
        movement = movements.get(copy_index)
        if movement is None:
            raise MagnetAlphaSerializationError(
                f"{copy_node.name}: unknown shuffle-copy movement "
                f"{copy_index}"
            )
        destination_channel = movement.destination_entry * block_channels
        block = value[
            :,
            destination_channel : destination_channel + block_channels,
        ].contiguous()
        if block.shape[1] != block_channels:
            raise MagnetAlphaSerializationError(
                f"{copy_node.name}: copy block is outside the logical "
                "shuffle output"
            )
        _register_serialized_artifact(
            storage_graph,
            storage_plane,
            bindings,
            copy_node,
            "copy_block_payload",
            copy_node,
            block,
        )


def _clear_serialized_artifact_bindings(model: GraphModule) -> None:
    """Release fields owned by an earlier storage-plan finalization."""
    for binding in model.meta.get(_SERIALIZED_ARTIFACT_BINDINGS, ()):
        target_kwargs = binding.operation.meta.get(TARGET_CODEGEN_KWARGS)
        if not isinstance(target_kwargs, dict):
            continue
        previous = target_kwargs.get(binding.field)
        if previous is not None and not isinstance(
            previous,
            SerializedTensorRef,
        ):
            raise MagnetAlphaSerializationError(
                f"{binding.operation.name}: prior {binding.field} is not "
                "a serialized artifact reference"
            )
        target_kwargs.pop(binding.field, None)
        if not target_kwargs:
            binding.operation.meta.pop(TARGET_CODEGEN_KWARGS, None)


def prepare_serialized_artifacts(
    model: GraphModule,
    storage_graph: StorageGraph,
    storage_plane,
    storage_allocator,
) -> None:
    """Declare canonical nonresident target payloads before finalization."""
    _clear_serialized_artifact_bindings(model)
    bindings: list[_SerializedArtifactBinding] = []
    _declare_operation_scale_artifacts(
        model,
        storage_graph,
        storage_plane,
        storage_allocator,
        bindings,
    )
    _declare_shuffle_copy_artifacts(
        model,
        storage_graph,
        storage_plane,
        bindings,
    )
    model.meta[_SERIALIZED_ARTIFACT_BINDINGS] = tuple(bindings)


def bind_serialized_artifacts(model: GraphModule) -> None:
    """Bind canonical nonresident payloads from the finalized plan."""
    plan = model.meta.get("storage_plan")
    bindings = model.meta.get(_SERIALIZED_ARTIFACT_BINDINGS, ())
    if plan is None and bindings:
        raise MagnetAlphaSerializationError(
            "Magnet Alpha serialized artifacts have no finalized storage plan"
        )
    for binding in bindings:
        reference = plan.serialized_ref(
            binding.logical_tensor,
            binding.plane,
            binding.artifact,
        )
        target_kwargs = binding.operation.meta.setdefault(
            TARGET_CODEGEN_KWARGS,
            {},
        )
        previous = target_kwargs.get(binding.field)
        if previous is not None and (
            not isinstance(previous, SerializedTensorRef)
            or previous.artifact is not reference.artifact
        ):
            raise MagnetAlphaSerializationError(
                f"{binding.operation.name}: conflicting finalized "
                f"{binding.field} artifact"
            )
        target_kwargs[binding.field] = reference


def _validate_serialized_artifacts(
    model: GraphModule,
    error_type: type[RuntimeError],
) -> None:
    plan = model.meta.get("storage_plan")
    bindings = model.meta.get(_SERIALIZED_ARTIFACT_BINDINGS, ())
    if plan is None and bindings:
        raise error_type(
            "Magnet Alpha serialized artifacts have no storage plan"
        )
    for binding in bindings:
        reference = binding.operation.meta.get(
            TARGET_CODEGEN_KWARGS,
            {},
        ).get(binding.field)
        expected = plan.serialized_artifact(
            binding.logical_tensor,
            binding.plane,
            binding.artifact,
        )
        if (
            not isinstance(reference, SerializedTensorRef)
            or reference.artifact is not expected
        ):
            raise error_type(
                f"{binding.operation.name}: {binding.field} was not bound "
                "to its finalized serialized artifact"
            )


def _validate_operation_constraints(
    model: GraphModule,
    storage_allocator,
    error_type: type[RuntimeError],
) -> None:
    constraints = getattr(storage_allocator, "operation_constraints", None)
    if not callable(constraints):
        return
    for operation, constraint in constraints():
        implementation = getattr(constraint, "implementation", None)
        execution_nodes = _serialized_execution_nodes(
            model,
            operation,
        )
        if implementation not in execution_nodes:
            name = getattr(implementation, "name", repr(implementation))
            raise error_type(
                f"{operation.name}: selected Magnet Alpha "
                f"implementation {name} is not serialized by its "
                "executable owner"
            )
        bound = bound_operation_classification(implementation)
        if bound is None:
            raise error_type(
                f"{operation.name}: selected implementation has no final "
                "FX classification"
            )
        unit, classification = bound
        if unit != getattr(
            constraint, "unit", None
        ) or classification != getattr(constraint, "classification", None):
            raise error_type(
                f"{operation.name}: allocator constraint disagrees with "
                "final FX classification"
            )
        if operation.meta.get("compute_unit") != unit:
            raise error_type(
                f"{operation.name}: executable owner has no bound compute "
                f"unit {unit!r}"
            )
        schedule = operation.meta.get(LOCAL_SCHEDULE)
        workload = operation.meta.get(LOCAL_SCHEDULER_WORKLOAD)
        if schedule is None or not isinstance(workload, OperationWorkload):
            raise error_type(
                f"{operation.name}: selected implementation has no bound "
                "schedule and workload"
            )
        if workload.operation != classification:
            raise error_type(
                f"{operation.name}: scheduler workload disagrees with final "
                "FX classification"
            )


def _bank_placement_payload(model: GraphModule, history) -> dict:
    allocations = []
    for record in sorted(history, key=lambda item: item.allocation_id):
        allocations.append(
            {
                "allocation_id": int(record.allocation_id),
                "physical_tensor": record.physical_tensor.name,
                "logical_tensors": sorted(
                    logical.name for logical in record.logical_tensors
                ),
                "role": record.role.value,
                "memory_name": record.memory_name,
                "memory_space": int(record.memory_space),
                "start": int(record.start),
                "end": int(record.end),
                "address": (
                    None if record.address is None else int(record.address)
                ),
                "birth_position": int(record.birth_position),
                "death_position": int(record.death_position),
                "primary": bool(record.primary),
                "banks": sorted({bank for bank, _entry in record.cells}),
            }
        )
    return {
        "schema_version": 1,
        "target": "magnet-alpha",
        "position_semantics": "zero_based_execution_order_index",
        "interval_semantics": "birth_inclusive_death_exclusive",
        "execution_order": [
            node.name for node in execution_order_for_model(model).nodes
        ],
        "allocations": allocations,
    }


def prepare_mapped_serialization_metadata(
    model: GraphModule,
    storage_allocator,
    hardware_config,
) -> None:
    """Bind canonical artifacts and validate the selected final FX facts."""
    discard_supplemental_artifact(model, _BANK_PLACEMENTS_ARTIFACT)
    bind_serialized_artifacts(model)
    _validate_operation_constraints(
        model,
        storage_allocator,
        MagnetAlphaSerializationError,
    )
    history = getattr(storage_allocator, "allocation_history", None)
    if history is not None:
        set_supplemental_artifact(
            model,
            _BANK_PLACEMENTS_ARTIFACT,
            _bank_placement_payload(model, history),
        )


def _nmp_activation_dims(
    operation: Node,
    workload: OperationWorkload,
    operand,
    role: str,
    error_type: type[RuntimeError],
) -> tuple[int, int, int, int]:
    classification = workload.operation
    shape = None if operand is None else operand.shape
    if (
        classification is None
        or shape is None
        or len(shape) != 4
        or any(dimension <= 0 for dimension in shape)
    ):
        raise error_type(
            f"{operation.name}: NMP {role} has no positive rank-4 "
            "scheduler extent"
        )
    if classification.attribute("transposed", False):
        batch, height, width, channels = shape
    else:
        batch, channels, height, width = shape
    return batch, channels, height, width


def _validate_nmp_schedule_coverage(
    operation: Node,
    workload: OperationWorkload,
    schedule,
    factors: dict[str, int],
    error_type: type[RuntimeError],
) -> None:
    """Check a bound NMP result without selecting or changing factors."""
    classification = workload.operation
    if (
        classification is None
        or classification.kind not in _NMP_CLASSIFICATIONS
    ):
        actual = None if classification is None else classification.kind
        raise error_type(
            f"{operation.name}: invalid NMP classification {actual!r}"
        )
    kind = classification.kind
    required = {"c1", "k1", "p1", "q1"}
    if not required.issubset(factors):
        raise error_type(
            f"{operation.name}: {kind} schedule is missing NMP factors"
        )
    if schedule.mapping is None:
        raise error_type(
            f"{operation.name}: {kind} has no bound scheduler mapping"
        )
    mapping_factors = {
        "c1": schedule.mapping.loop_coverage(Loop.IC),
        "k1": schedule.mapping.loop_coverage(Loop.OC),
        "p1": schedule.mapping.loop_coverage(Loop.OY),
        "q1": schedule.mapping.loop_coverage(Loop.OX),
    }
    mismatches = {
        name: (factors[name], expected)
        for name, expected in mapping_factors.items()
        if factors[name] != expected
    }
    if mismatches:
        raise error_type(
            f"{operation.name}: {kind} named factors disagree with its "
            f"scheduler mapping: {mismatches}"
        )

    if kind == "softmax":
        input_operand = workload.operand("input")
        input_shape = None if input_operand is None else input_operand.shape
        output_shape = workload.output.shape
        if (
            input_shape is None
            or len(input_shape) < 2
            or output_shape != input_shape
        ):
            raise error_type(f"{operation.name}: softmax extents are invalid")
        dim = classification.attribute("dim", -1)
        if dim < 0:
            dim += len(input_shape)
        if dim != len(input_shape) - 1:
            raise error_type(
                f"{operation.name}: softmax must use its last dimension"
            )
        expected_spatial = {"p1": input_shape[-2], "q1": 1}
        channel_relation = factors["c1"] == factors["k1"]
    else:
        output = _nmp_activation_dims(
            operation,
            workload,
            workload.output,
            "output",
            error_type,
        )
        expected_spatial = {"p1": output[2], "q1": output[3]}
        channel_relation = None

    if kind == "depthwise_conv2d":
        input_dims = _nmp_activation_dims(
            operation,
            workload,
            workload.operand("input"),
            "input",
            error_type,
        )
        groups = classification.attribute("groups", 1)
        if (
            input_dims[0] != output[0]
            or input_dims[1] != output[1]
            or output[1] != groups
        ):
            raise error_type(
                f"{operation.name}: depthwise workload has inconsistent "
                "batch, channels, or groups"
            )
        channel_relation = factors["c1"] == factors["k1"]
    elif kind == "pool2d":
        input_dims = _nmp_activation_dims(
            operation,
            workload,
            workload.operand("input"),
            "input",
            error_type,
        )
        if input_dims[:2] != output[:2]:
            raise error_type(
                f"{operation.name}: pool workload does not preserve N/C"
            )
        expected_spatial = {
            "p1": input_dims[2],
            "q1": input_dims[3],
        }
        channel_relation = factors["c1"] == factors["k1"]
    elif kind == "channel_shuffle":
        input1 = _nmp_activation_dims(
            operation,
            workload,
            workload.operand("input1"),
            "input1",
            error_type,
        )
        input2 = _nmp_activation_dims(
            operation,
            workload,
            workload.operand("input2"),
            "input2",
            error_type,
        )
        groups = classification.attribute("groups")
        if (
            groups != 2
            or input1 != input2
            or input1[0] != output[0]
            or input1[2:] != output[2:]
            or 2 * input1[1] != output[1]
        ):
            raise error_type(
                f"{operation.name}: channel-shuffle extents are invalid"
            )
        channel_relation = factors["k1"] == 2 * factors["c1"]
    elif kind == "channel_shuffle_copy":
        input_dims = _nmp_activation_dims(
            operation,
            workload,
            workload.operand("input"),
            "input",
            error_type,
        )
        block_channels = classification.attribute("block_channels")
        if (
            input_dims != output
            or output[0] != 1
            or not isinstance(block_channels, int)
            or block_channels <= 0
        ):
            raise error_type(
                f"{operation.name}: shuffle-copy extents are invalid"
            )
        channel_relation = factors["c1"] == factors["k1"] == 1
    elif kind == "upsample_nearest2d":
        input_dims = _nmp_activation_dims(
            operation,
            workload,
            workload.operand("input"),
            "input",
            error_type,
        )
        if (
            input_dims[:2] != output[:2]
            or output[2] < input_dims[2]
            or output[3] < input_dims[3]
        ):
            raise error_type(
                f"{operation.name}: nearest-upsample extents are invalid"
            )
        channel_relation = factors["c1"] == factors["k1"]
    elif kind == "group_norm":
        input_dims = _nmp_activation_dims(
            operation,
            workload,
            workload.operand("input"),
            "input",
            error_type,
        )
        if input_dims != output:
            raise error_type(
                f"{operation.name}: {kind} does not preserve its extent"
            )
        channel_relation = factors["c1"] == factors["k1"]

    spatial_mismatches = {
        name: (factors[name], expected)
        for name, expected in expected_spatial.items()
        if factors[name] != expected
    }
    if spatial_mismatches:
        raise error_type(
            f"{operation.name}: {kind} spatial coverage disagrees with "
            f"its workload: {spatial_mismatches}"
        )
    if channel_relation is False:
        raise error_type(
            f"{operation.name}: {kind} channel-factor relation is invalid"
        )


def _validate_gemm_schedule_coverage(
    operation: Node,
    workload: OperationWorkload,
    factors: dict[str, int],
    target_kwargs: dict,
    error_type: type[RuntimeError],
) -> None:
    if workload.gemm is None:
        raise error_type(
            f"{operation.name}: scheduled GEMM has no GEMM workload"
        )
    required = {
        "batch_size",
        "c0",
        "c1",
        "c2",
        "c3",
        "k0",
        "k1",
        "k2",
        "k3",
        "p1",
        "p2",
        "p3",
        "q1",
        "q2",
        "q3",
        "r",
        "s",
    }
    if not required.issubset(factors):
        raise error_type(
            f"{operation.name}: GEMM schedule is missing extent factors"
        )
    padding_c = target_kwargs.get("padding_c")
    padding_k = target_kwargs.get("padding_k")
    if (
        isinstance(padding_c, bool)
        or not isinstance(padding_c, int)
        or not 0 <= padding_c < factors["c0"]
        or isinstance(padding_k, bool)
        or not isinstance(padding_k, int)
        or not 0 <= padding_k < factors["k0"]
    ):
        raise error_type(
            f"{operation.name}: GEMM padding is missing or invalid"
        )
    coverage = {
        "N": factors["batch_size"],
        "C": (
            (factors["c0"] - padding_c)
            * factors["c1"]
            * factors["c2"]
            * factors["c3"]
        ),
        "K": (
            (factors["k0"] - padding_k)
            * factors["k1"]
            * factors["k2"]
            * factors["k3"]
        ),
        "P": factors["p1"] * factors["p2"] * factors["p3"],
        "Q": factors["q1"] * factors["q2"] * factors["q3"],
        "R": factors["r"],
        "S": factors["s"],
    }
    valid_input_channels = target_kwargs.get(
        "materialized_input_valid_channels"
    )
    resident_input_channels = target_kwargs.get(
        "materialized_input_resident_channels"
    )
    materialized_input = (
        valid_input_channels is not None or resident_input_channels is not None
    )
    if materialized_input:
        if (
            isinstance(valid_input_channels, bool)
            or not isinstance(valid_input_channels, int)
            or isinstance(resident_input_channels, bool)
            or not isinstance(resident_input_channels, int)
            or not 0 < valid_input_channels <= resident_input_channels
        ):
            raise error_type(
                f"{operation.name}: materialized GEMM input must declare "
                "paired positive valid/resident channel extents"
            )
        if workload.gemm.nifm != valid_input_channels:
            raise error_type(
                f"{operation.name}: materialized GEMM input valid C="
                f"{valid_input_channels} disagrees with workload C="
                f"{workload.gemm.nifm}"
            )
        weight_valid_channels = target_kwargs.get("weight_valid_input_channels")
        if weight_valid_channels != valid_input_channels:
            raise error_type(
                f"{operation.name}: materialized GEMM input valid C does "
                "not match its weight contract"
            )
        if coverage["C"] != resident_input_channels:
            raise error_type(
                f"{operation.name}: GEMM schedule resident C="
                f"{coverage['C']} disagrees with materialized input C="
                f"{resident_input_channels}"
            )
    expected = {
        "N": workload.gemm.nimg,
        "C": (
            resident_input_channels
            if materialized_input
            else workload.gemm.nifm
        ),
        "K": workload.gemm.nofm,
        "P": workload.gemm.hofm,
        "Q": workload.gemm.wofm,
        "R": workload.gemm.hfil,
        "S": workload.gemm.wfil,
    }
    mismatches = {
        dimension: (coverage[dimension], expected[dimension])
        for dimension in coverage
        if coverage[dimension] != expected[dimension]
    }
    if mismatches:
        raise error_type(
            f"{operation.name}: GEMM schedule coverage disagrees with its "
            f"workload: {mismatches}"
        )


def _validate_channel_shuffle_storage(
    operation: Node,
    target_kwargs: dict,
    error_type: type[RuntimeError],
) -> None:
    addresses = target_kwargs.get("output_addresses")
    instances = (
        target_kwargs.get("output0_memory_instance"),
        target_kwargs.get("output1_memory_instance"),
    )
    if (
        not isinstance(addresses, list)
        or len(addresses) != 2
        or any(
            isinstance(address, bool)
            or not isinstance(address, int)
            or address < 0
            for address in addresses
        )
        or any(
            not isinstance(instance, str) or not instance
            for instance in instances
        )
    ):
        raise error_type(
            f"{operation.name}: channel shuffle must serialize two "
            "addresses with exact memory instances"
        )


def validate_serialization_metadata(
    model: GraphModule,
    *,
    error_type: type[RuntimeError] = MagnetAlphaSerializationError,
) -> None:
    """Reject incomplete or contradictory canonical Magnet Alpha metadata."""
    _validate_serialized_artifacts(model, error_type)
    for operation in _serialization_operations(model):
        target_kwargs = operation.meta.get(TARGET_CODEGEN_KWARGS, {})
        schedule = operation.meta.get(LOCAL_SCHEDULE)
        factor_kwargs = {
            key: value
            for key, value in target_kwargs.items()
            if key.startswith(SCHEDULE_FACTOR_PREFIX)
        }
        if schedule is None:
            if factor_kwargs:
                raise error_type(
                    f"{operation.name}: unscheduled operation has schedule "
                    "fields"
                )
            if operation.meta.get("compute_unit") is not None:
                raise error_type(
                    f"{operation.name}: bound compute operation has no "
                    "local schedule"
                )
            continue
        if not schedule.named_factors:
            raise error_type(
                f"{operation.name}: scheduled operation has no named factors"
            )
        expected_factors = {
            f"{SCHEDULE_FACTOR_PREFIX}{name}": value
            for name, value in schedule.named_factors
        }
        if factor_kwargs != expected_factors:
            raise error_type(
                f"{operation.name}: serialized factors disagree with its "
                "bound local schedule"
            )
        workload = operation.meta.get(LOCAL_SCHEDULER_WORKLOAD)
        if not isinstance(workload, OperationWorkload):
            raise error_type(
                f"{operation.name}: scheduled operation has no workload"
            )
        factors = dict(schedule.named_factors)
        classification = workload.operation
        if classification is not None and classification.kind == "gemm":
            _validate_gemm_schedule_coverage(
                operation,
                workload,
                factors,
                _gemm_record_target_kwargs(
                    model,
                    operation,
                    target_kwargs,
                    error_type,
                ),
                error_type,
            )
        elif (
            classification is not None
            and classification.kind in _NMP_CLASSIFICATIONS
        ):
            _validate_nmp_schedule_coverage(
                operation,
                workload,
                schedule,
                factors,
                error_type,
            )
            if classification.kind == "channel_shuffle":
                _validate_channel_shuffle_storage(
                    operation,
                    target_kwargs,
                    error_type,
                )
        else:
            kind = None if classification is None else classification.kind
            raise error_type(
                f"{operation.name}: unsupported scheduled classification "
                f"{kind!r}"
            )


__all__ = [
    "MagnetAlphaSerializationError",
    "TARGET_CODEGEN_KWARGS",
    "bind_serialized_artifacts",
    "prepare_mapped_serialization_metadata",
    "prepare_serialized_artifacts",
    "validate_serialization_metadata",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Whole-graph logical-to-physical storage analysis."""

from __future__ import annotations

import math
import operator
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.execution_order import (
    ExecutionOrder,
    execution_positions,
    fx_execution_order,
    validate_execution_order,
)
from voyager_compiler.codegen.hardware import TensorRole, get_dtype_width
from voyager_compiler.codegen.storage.effects import (
    generic_storage_relation,
    node_value,
    normalize_getitem_index,
)
from voyager_compiler.codegen.storage.types import (
    LogicalTensor,
    PhysicalAccess,
    PhysicalPayloadRef,
    PhysicalTensor,
    PhysicalTensorComponent,
    PhysicalTensorId,
    SerializedTensorArtifact,
    SerializedTensorRef,
    StorageInterval,
    StoragePlane,
    StorageRelation,
    StorageRelationKind,
    StorageTransfer,
    ValueRef,
    ViewTransform,
)
from voyager_compiler.codegen.storage_predicates import (
    StorageEffectKind,
    forwards_aggregate_writes,
    storage_op_traits,
)


def _plane_name(plane: str | StoragePlane) -> str:
    return plane.value if isinstance(plane, StoragePlane) else str(plane)


def _trace_storage_relations(
    index: GraphIndex,
    relations: dict[ValueRef, StorageRelation],
    current: ValueRef,
    *,
    follow_kinds: Iterable[StorageRelationKind] | None,
    follow_copy: Callable[[ValueRef], bool] | None,
    follow_write_input: Callable[[ValueRef], bool] | None,
) -> tuple[ValueRef, ...]:
    sharing_kinds = set(
        follow_kinds
        if follow_kinds is not None
        else {
            StorageRelationKind.ALIAS,
            StorageRelationKind.VIEW,
            StorageRelationKind.WRITE_INTO,
            StorageRelationKind.COMPONENT,
        }
    )
    chain = []
    visited = set()
    while True:
        if current in visited:
            names = " -> ".join(ref.name for ref in (*chain, current))
            raise ValueError(f"Storage-relation cycle: {names}")
        visited.add(current)
        chain.append(current)
        relation = relations[current]
        if (
            relation.kind is StorageRelationKind.WRITE_INTO
            and follow_write_input is not None
            and follow_write_input(current)
        ):
            inputs = index.logical_tensors[current].inputs
            if len(inputs) != 1:
                break
            current = inputs[0]
            continue
        follows_source = relation.kind in sharing_kinds or (
            relation.kind is StorageRelationKind.COPY
            and follow_copy is not None
            and follow_copy(current)
        )
        if not follows_source or relation.source is None:
            break
        current = relation.source
    return tuple(chain)


@dataclass(frozen=True)
class _PhysicalComponentDeclaration:
    """A target payload declaration awaiting physical-ID resolution."""

    logical_tensor: ValueRef
    physical_tensor: PhysicalTensorId | None
    name: str
    shape: tuple[int, ...]
    dtype: object
    bits: int
    byte_size: int
    value: torch.Tensor
    backing_size_bytes: int
    layout: str


@dataclass(frozen=True)
class _SerializedArtifactDeclaration:
    """A nonresident target encoding awaiting physical-ID resolution."""

    logical_tensor: ValueRef
    name: str
    shape: tuple[int, ...]
    dtype: object
    bits: int
    byte_size: int
    value: torch.Tensor


def _tensor_value_paths(value, path=()):
    if isinstance(value, torch.Tensor):
        yield path, value
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tensor_value_paths(item, path + (index,))


class GraphIndex:
    """Stable logical tensor and source-topology index for one FX graph."""

    def __init__(self, model: GraphModule):
        self.model = model
        self.source_positions = {
            node: position for position, node in enumerate(model.graph.nodes)
        }
        self.node_outputs: dict[Node, tuple[ValueRef, ...]] = {}
        self.node_inputs: dict[Node, tuple[ValueRef, ...]] = {}
        self.logical_tensors: dict[ValueRef, LogicalTensor] = {}
        self._build()

    def _outputs_for(self, node: Node) -> tuple[ValueRef, ...]:
        if node.op == "output":
            return ()
        paths = tuple(
            path for path, _value in _tensor_value_paths(node_value(node))
        )
        if not paths and (
            getattr(node, "shape", None) is not None
            or node.meta.get("dtype") is not None
        ):
            paths = ((),)
        return tuple(ValueRef(node, path) for path in paths)

    def _input_refs(self, node: Node) -> tuple[ValueRef, ...]:
        if (
            node.target == operator.getitem
            and node.args
            and isinstance(node.args[0], Node)
            and len(node.args) > 1
            and isinstance(node.args[1], int)
        ):
            outputs = self.node_outputs.get(node.args[0], ())
            index = normalize_getitem_index(
                node.args[0],
                node.args[1],
                outputs,
            )
            selected = tuple(
                output for output in outputs if output.path[:1] == (index,)
            )
            if selected:
                return selected

        refs = []
        seen = set()
        for input_node in node.all_input_nodes:
            for ref in self.node_outputs.get(input_node, ()):
                if ref not in seen:
                    refs.append(ref)
                    seen.add(ref)
        return tuple(refs)

    @staticmethod
    def _shape_dtype(ref: ValueRef):
        declared_dtype = ref.node.meta.get("dtype")
        if ref.path and isinstance(declared_dtype, (tuple, list)):
            value_at_path = declared_dtype
            for index in ref.path:
                value_at_path = value_at_path[index]
            declared_dtype = value_at_path

        value = node_value(ref.node)
        for path, tensor in _tensor_value_paths(value):
            if path == ref.path:
                return (
                    tuple(tensor.shape),
                    declared_dtype or tensor.dtype,
                )

        shape = getattr(ref.node, "shape", ())
        return tuple(shape or ()), declared_dtype

    def _build(self) -> None:
        for node in self.model.graph.nodes:
            self.node_outputs[node] = self._outputs_for(node)

        for node in self.model.graph.nodes:
            self.node_inputs[node] = self._input_refs(node)

        users = defaultdict(set)
        for node, inputs in self.node_inputs.items():
            for ref in inputs:
                users[ref].add(node)

        for node, outputs in self.node_outputs.items():
            inputs = self.node_inputs[node]
            for ref in outputs:
                shape, dtype = self._shape_dtype(ref)
                self.logical_tensors[ref] = LogicalTensor(
                    ref=ref,
                    shape=shape,
                    dtype=dtype,
                    inputs=inputs,
                    users=tuple(
                        sorted(
                            users.get(ref, ()),
                            key=self.source_positions.__getitem__,
                        )
                    ),
                )

    def ref(self, node: Node, path: tuple[int, ...] = ()) -> ValueRef:
        candidate = ValueRef(node, path)
        if candidate not in self.logical_tensors:
            raise KeyError(f"{candidate.name} is not a logical tensor")
        return candidate


class StorageGraph:
    """Mutable relation builder that finalizes into a storage plan."""

    def __init__(
        self,
        model: GraphModule,
        planes: Iterable[str],
    ):
        self.index = GraphIndex(model)
        self._planes = tuple(
            dict.fromkeys(_plane_name(plane) for plane in planes)
        )
        if not self._planes:
            raise ValueError("Storage analysis requires at least one plane")
        self._relations: dict[str, dict[ValueRef, StorageRelation]] = {}
        self._auxiliary: dict[str, set[PhysicalTensorId]] = {}
        self._accesses: dict[
            str,
            list[
                tuple[
                    ValueRef | PhysicalTensorId,
                    PhysicalAccess,
                ]
            ],
        ] = {}
        self._transfers: list[tuple[ValueRef, str, str, str]] = []
        self._components: dict[str, list[_PhysicalComponentDeclaration]] = {}
        self._artifacts: dict[str, list[_SerializedArtifactDeclaration]] = {}
        for plane in self._planes:
            self._relations[plane] = {
                ref: generic_storage_relation(
                    ref,
                    self.index.node_outputs,
                    self.index.node_inputs,
                )
                for ref in self.index.logical_tensors
            }
            self._auxiliary[plane] = set()
            self._accesses[plane] = []
            self._components[plane] = []
            self._artifacts[plane] = []
        self._apply_generic_split_relations()
        self._apply_generic_aggregate_relations()

    def _apply_generic_split_relations(self) -> None:
        """Materialize tuple split results in one explicit backing.

        A split/chunk FX node produces a Python container, not a logical
        tensor.  The executable generic instruction therefore writes its
        tensor results into one physical output container.  Target allocators
        with addressable views may replace these relations during
        ``prepare_graph``.
        """
        for node in self.index.model.graph.nodes:
            if (
                storage_op_traits(node.target).effect
                is not StorageEffectKind.SPLIT_VIEW
            ):
                continue
            outputs = self.index.node_outputs.get(node, ())
            if not outputs:
                continue
            for plane in self._planes:
                container = self.add_auxiliary(
                    outputs[0],
                    "tuple_outputs",
                    plane=plane,
                )
                for index, output in enumerate(outputs):
                    self.bind_physical(
                        output,
                        container,
                        plane=plane,
                        kind=StorageRelationKind.COMPONENT,
                        transform=ViewTransform(
                            "tuple_output",
                            parameters=(("result_index", index),),
                        ),
                    )

    def _apply_generic_aggregate_relations(self) -> None:
        """Declare safe direct writes into simple cat/stack backings."""
        aggregate_targets = {
            torch.ops.aten.cat.default,
            torch.ops.aten.stack.default,
        }
        for node in self.index.model.graph.nodes:
            if node.target not in aggregate_targets:
                continue
            tensors = node.args[0] if node.args else ()
            if not isinstance(tensors, (tuple, list)) or not tensors:
                continue
            if not all(isinstance(tensor, Node) for tensor in tensors):
                continue
            dim = node.kwargs.get(
                "dim",
                node.args[1] if len(node.args) > 1 else 0,
            )
            if dim != 0:
                continue
            outputs = self.index.node_outputs.get(node, ())
            if len(outputs) != 1:
                continue

            paths = []
            for tensor in tensors:
                if (
                    tensors.count(tensor) != 1
                    or len(tensor.users) != 1
                    or node not in tensor.users
                ):
                    paths.append(())
                    continue
                path = []
                current = tensor
                while True:
                    current_outputs = self.index.node_outputs.get(
                        current,
                        (),
                    )
                    if len(current_outputs) != 1:
                        path = []
                        break
                    path.append(current_outputs[0])
                    relation = self._relations[self._planes[0]][
                        current_outputs[0]
                    ]
                    if relation.kind not in {
                        StorageRelationKind.ALIAS,
                        StorageRelationKind.VIEW,
                    }:
                        break
                    if (
                        relation.source is None
                        or not self._is_full_coverage_forwarding(
                            current_outputs[0],
                            relation,
                        )
                    ):
                        path = []
                        break
                    source = relation.source.node
                    if len(source.users) != 1 or current not in source.users:
                        path = []
                        break
                    current = source
                if not path or path[-1].node.op in {"placeholder", "get_attr"}:
                    path = []
                paths.append(tuple(path))

            backing = outputs[0]
            for plane in self._planes:
                for slot, path in enumerate(paths):
                    for result in path:
                        self._relations[plane][result] = StorageRelation(
                            result=result,
                            kind=StorageRelationKind.WRITE_INTO,
                            source=backing,
                            transform=ViewTransform(
                                "aggregate_slot",
                                parameters=(("slot", slot),),
                            ),
                        )

    def _is_full_coverage_forwarding(
        self,
        result: ValueRef,
        relation: StorageRelation,
    ) -> bool:
        """Return whether direct writes may cross one storage-only view."""
        if relation.kind is StorageRelationKind.ALIAS:
            return True
        if (
            relation.kind is not StorageRelationKind.VIEW
            or relation.source is None
            or not forwards_aggregate_writes(result.node.target)
        ):
            return False
        result_tensor = self.index.logical_tensors[result]
        source_tensor = self.index.logical_tensors[relation.source]
        if result_tensor.dtype != source_tensor.dtype:
            return False
        try:
            result_numel = math.prod(int(dim) for dim in result_tensor.shape)
            source_numel = math.prod(int(dim) for dim in source_tensor.shape)
        except (TypeError, ValueError):
            return False
        return result_numel == source_numel

    def ref(
        self,
        value: Node | ValueRef,
        path: tuple[int, ...] = (),
    ) -> ValueRef:
        if isinstance(value, ValueRef):
            if value not in self.index.logical_tensors:
                raise KeyError(f"{value.name} is not a logical tensor")
            return value
        return self.index.ref(value, path)

    def relation(
        self,
        value: Node | ValueRef,
        plane: str | StoragePlane,
    ) -> StorageRelation:
        """Return the current relation for one logical tensor and plane."""
        return self._relations[_plane_name(plane)][self.ref(value)]

    def trace_storage_chain(
        self,
        value: Node | ValueRef,
        plane: str | StoragePlane,
        *,
        follow_kinds: Iterable[StorageRelationKind] | None = None,
        follow_copy: Callable[[ValueRef], bool] | None = None,
        follow_write_input: Callable[[ValueRef], bool] | None = None,
    ) -> tuple[ValueRef, ...]:
        """Trace one logical value to the owner of its current backing.

        Alias, view, component, and write-into relations all share their
        source backing. Copies stop the chain unless a target explicitly
        identifies that copy as an in-place physical movement.
        """
        plane_name = _plane_name(plane)
        return _trace_storage_relations(
            self.index,
            self._relations[plane_name],
            self.ref(value),
            follow_kinds=follow_kinds,
            follow_copy=follow_copy,
            follow_write_input=follow_write_input,
        )

    def set_relation(
        self,
        result: Node | ValueRef,
        kind: StorageRelationKind,
        *,
        plane: str,
        source: Node | ValueRef | None = None,
        physical_source: PhysicalTensorId | None = None,
        transform: ViewTransform | None = None,
        component: str | None = None,
    ) -> StorageRelation:
        result_ref = self.ref(result)
        source_ref = None if source is None else self.ref(source)
        relation = StorageRelation(
            result=result_ref,
            kind=kind,
            source=source_ref,
            physical_source=physical_source,
            transform=transform,
            component=component,
        )
        self._relations[_plane_name(plane)][result_ref] = relation
        return relation

    def add_auxiliary(
        self,
        owner: Node | ValueRef,
        name: str,
        *,
        plane: str,
    ) -> PhysicalTensorId:
        physical = PhysicalTensorId(
            plane=_plane_name(plane),
            owner=self.ref(owner),
            component=name,
        )
        self._auxiliary[_plane_name(plane)].add(physical)
        return physical

    def add_access(
        self,
        storage: Node | ValueRef | PhysicalTensorId,
        user: Node,
        *,
        plane: str,
        role: TensorRole,
        kind: str,
    ) -> None:
        """Record a physical read/write hidden from ordinary FX edges."""
        plane_name = _plane_name(plane)
        if user not in self.index.source_positions:
            raise KeyError(f"{user.name} is not in the indexed graph")
        source = storage
        if not isinstance(storage, PhysicalTensorId):
            source = self.ref(storage)
        elif storage.plane != plane_name:
            raise ValueError(
                "A physical access must use the selected storage plane"
            )
        self._accesses[plane_name].append(
            (
                source,
                PhysicalAccess(
                    operation=user,
                    role=role,
                    kind=kind,
                ),
            )
        )

    def bind_physical(
        self,
        result: Node | ValueRef,
        physical: PhysicalTensorId,
        *,
        plane: str,
        kind: StorageRelationKind = StorageRelationKind.VIEW,
        transform: ViewTransform | None = None,
    ) -> StorageRelation:
        plane_name = _plane_name(plane)
        if physical.plane != plane_name:
            raise ValueError(
                "A logical tensor and its physical storage must use the "
                "same storage plane"
            )
        if physical not in self._auxiliary[plane_name]:
            raise ValueError(
                "Physical storage must be declared with add_auxiliary() "
                "before binding a logical tensor"
            )
        return self.set_relation(
            result,
            kind,
            plane=plane_name,
            physical_source=physical,
            transform=transform,
        )

    def add_transfer(
        self,
        value: Node | ValueRef,
        *,
        source_plane: str | StoragePlane,
        destination_plane: str | StoragePlane,
        kind: str = "load",
    ) -> None:
        """Declare an explicit movement between two physical planes."""
        source_name = _plane_name(source_plane)
        destination_name = _plane_name(destination_plane)
        if source_name == destination_name:
            raise ValueError("A storage transfer must cross storage planes")
        unknown = {
            source_name,
            destination_name,
        } - set(self._planes)
        if unknown:
            raise ValueError(
                "Storage transfer references unknown planes: "
                + ", ".join(sorted(unknown))
            )
        declaration = (
            self.ref(value),
            source_name,
            destination_name,
            str(kind),
        )
        if declaration not in self._transfers:
            self._transfers.append(declaration)

    def register_physical_component(
        self,
        logical_tensor: Node | ValueRef | PhysicalTensorId,
        value: torch.Tensor,
        *,
        plane: str | StoragePlane,
        component: str = "primary",
        dtype=None,
        bits: int | None = None,
        backing_size_bytes: int,
        layout: str,
    ) -> None:
        """Register an exact target payload before physical-ID resolution.

        The target owns both the physical layout name and complete backing
        extent. Finalization resolves the logical reference through all
        aliases/components and rejects incompatible declarations that land on
        one backing.
        """
        plane_name = _plane_name(plane)
        if plane_name not in self._components:
            raise ValueError(f"Storage graph has no {plane_name} plane")
        if isinstance(logical_tensor, PhysicalTensorId):
            if logical_tensor.plane != plane_name:
                raise ValueError(
                    f"{logical_tensor.name}: component plane does not "
                    f"match {plane_name}"
                )
            if logical_tensor not in self._auxiliary[plane_name]:
                raise ValueError(
                    f"{logical_tensor.name}: physical component may only "
                    "target a declared auxiliary identity"
                )
            physical_tensor = logical_tensor
            logical_ref = logical_tensor.owner
        else:
            physical_tensor = None
            logical_ref = self.ref(logical_tensor)
        if not isinstance(value, torch.Tensor):
            raise TypeError("Physical component value must be a tensor")
        if not component:
            raise ValueError("Physical component name must not be empty")
        if not layout:
            raise ValueError("Physical component layout must not be empty")
        dtype = value.dtype if dtype is None else dtype
        bits = get_dtype_width(dtype) if bits is None else int(bits)
        if bits <= 0:
            raise ValueError("Physical component width must be positive")
        byte_size = math.ceil(value.numel() * bits / 8)
        backing_size_bytes = int(backing_size_bytes)
        if backing_size_bytes < byte_size:
            raise ValueError(
                f"{logical_ref.name}: {component} needs {byte_size} bytes, "
                f"but its backing declares {backing_size_bytes}"
            )
        self._components[plane_name].append(
            _PhysicalComponentDeclaration(
                logical_tensor=logical_ref,
                physical_tensor=physical_tensor,
                name=component,
                shape=tuple(int(dim) for dim in value.shape),
                dtype=dtype,
                bits=bits,
                byte_size=byte_size,
                value=value,
                backing_size_bytes=backing_size_bytes,
                layout=str(layout),
            )
        )

    def register_serialized_artifact(
        self,
        logical_tensor: Node | ValueRef,
        value: torch.Tensor,
        *,
        plane: str | StoragePlane,
        name: str,
        dtype=None,
        bits: int | None = None,
    ) -> None:
        """Register a target encoding that is not resident in the backing."""
        plane_name = _plane_name(plane)
        if plane_name not in self._artifacts:
            raise ValueError(f"Storage graph has no {plane_name} plane")
        logical_ref = self.ref(logical_tensor)
        if not isinstance(value, torch.Tensor):
            raise TypeError("Serialized artifact value must be a tensor")
        if not name:
            raise ValueError("Serialized artifact name must not be empty")
        dtype = value.dtype if dtype is None else dtype
        bits = get_dtype_width(dtype) if bits is None else int(bits)
        if bits <= 0:
            raise ValueError("Serialized artifact width must be positive")
        self._artifacts[plane_name].append(
            _SerializedArtifactDeclaration(
                logical_tensor=logical_ref,
                name=name,
                shape=tuple(int(dim) for dim in value.shape),
                dtype=dtype,
                bits=bits,
                byte_size=math.ceil(value.numel() * bits / 8),
                value=value,
            )
        )

    def finalize(
        self,
        *,
        execution_order: ExecutionOrder | None = None,
    ) -> "StoragePlan":
        if execution_order is None:
            execution_order = fx_execution_order(self.index.model)
        return StoragePlan(
            self.index,
            execution_order,
            self._relations,
            self._auxiliary,
            self._accesses,
            self._transfers,
            self._components,
            self._artifacts,
        )


def materialize_aggregate_copies(model: GraphModule) -> int:
    """Insert copies for dim-0 aggregate inputs that cannot alias.

    Cat and stack are address-only operations in the serialized generic
    program. Inputs that cannot be produced directly into an aggregate slot
    therefore need an executable copy before storage identities are frozen.
    """
    provisional = StorageGraph(model, (StoragePlane.DRAM,))
    inserted = 0

    for aggregate in tuple(model.graph.nodes):
        if aggregate.target not in {
            torch.ops.aten.cat.default,
            torch.ops.aten.stack.default,
        }:
            continue
        dim = aggregate.kwargs.get(
            "dim",
            aggregate.args[1] if len(aggregate.args) > 1 else 0,
        )
        tensors = aggregate.args[0] if aggregate.args else ()
        outputs = provisional.index.node_outputs.get(aggregate, ())
        if (
            dim != 0
            or len(outputs) != 1
            or not isinstance(tensors, (tuple, list))
            or not all(isinstance(tensor, Node) for tensor in tensors)
        ):
            continue

        backing = outputs[0]
        rewritten = []
        for tensor in tensors:
            tensor_outputs = provisional.index.node_outputs.get(tensor, ())
            relation = (
                provisional.relation(
                    tensor_outputs[0],
                    StoragePlane.DRAM,
                )
                if len(tensor_outputs) == 1
                else None
            )
            direct_write = (
                relation is not None
                and relation.kind is StorageRelationKind.WRITE_INTO
                and relation.source == backing
            )
            if direct_write:
                rewritten.append(tensor)
                continue

            with model.graph.inserting_before(aggregate):
                copy_node = model.graph.call_function(
                    torch.ops.aten.add.Scalar,
                    (tensor, 0),
                )
            copy_node.meta.update(tensor.meta)
            for placement_key in (
                "memory",
                "scratchpad",
                "scratchpad_map",
                "scratchpad_output_segments",
                "target_instruction_nop",
            ):
                copy_node.meta.pop(placement_key, None)
            copy_node.meta["physical_copy"] = True
            if hasattr(tensor, "shape"):
                copy_node.shape = tensor.shape
            if hasattr(tensor, "value"):
                copy_node.value = tensor.value
            rewritten.append(copy_node)
            inserted += 1

        if tuple(rewritten) != tuple(tensors):
            aggregate.update_arg(0, type(tensors)(rewritten))

    if inserted:
        model.graph.lint()
        model.recompile()
    return inserted


class StoragePlan:
    """Immutable logical-to-physical mapping and physical live intervals."""

    def __init__(
        self,
        index: GraphIndex,
        execution_order: ExecutionOrder,
        relations: dict[str, dict[ValueRef, StorageRelation]],
        auxiliary: dict[str, set[PhysicalTensorId]],
        accesses: dict[
            str,
            list[
                tuple[
                    ValueRef | PhysicalTensorId,
                    PhysicalAccess,
                ]
            ],
        ],
        transfers: Iterable[tuple[ValueRef, str, str, str]] = (),
        components: (
            dict[str, list[_PhysicalComponentDeclaration]] | None
        ) = None,
        artifacts: (
            dict[str, list[_SerializedArtifactDeclaration]] | None
        ) = None,
    ):
        validate_execution_order(index.model, execution_order)
        self.index = index
        self.execution_order = execution_order
        self.logical_tensors = dict(index.logical_tensors)
        self.relations = {
            plane: dict(plane_relations)
            for plane, plane_relations in relations.items()
        }
        self._physical_for: dict[str, dict[ValueRef, PhysicalTensorId]] = {}
        self.physical_tensors: dict[
            str, dict[PhysicalTensorId, PhysicalTensor]
        ] = {}
        self.intervals: dict[str, dict[PhysicalTensorId, StorageInterval]] = {}
        self._release_after: dict[
            str, dict[Node, tuple[PhysicalTensorId, ...]]
        ] = {}
        self._accesses: dict[
            str,
            dict[PhysicalTensorId, tuple[PhysicalAccess, ...]],
        ] = {}
        self.serialized_artifacts: dict[
            str,
            dict[tuple[PhysicalTensorId, str], SerializedTensorArtifact],
        ] = {}
        components = {} if components is None else components
        artifacts = {} if artifacts is None else artifacts
        positions = execution_positions(execution_order)

        for plane in self.relations:
            resolved = self._resolve_plane(plane)
            self._physical_for[plane] = resolved
            self.serialized_artifacts[plane] = (
                self._resolve_serialized_artifacts(
                    plane,
                    resolved,
                    artifacts.get(plane, ()),
                )
            )
            representations = self._resolve_representations(
                plane,
                resolved,
                auxiliary.get(plane, set()),
                components.get(plane, ()),
            )
            plane_accesses = self._resolve_accesses(
                plane,
                resolved,
                accesses.get(plane, ()),
            )
            self._accesses[plane] = plane_accesses
            self.physical_tensors[plane] = self._build_physical_tensors(
                plane,
                resolved,
                auxiliary.get(plane, set()),
                plane_accesses,
                representations,
            )
            intervals = self._build_intervals(
                plane,
                resolved,
                plane_accesses,
                positions,
            )
            self.intervals[plane] = intervals
            self._release_after[plane] = self._index_releases(
                intervals,
                positions,
            )
        self.transfers = tuple(
            StorageTransfer(
                logical_tensor=logical,
                source=self._physical_for[source_plane][logical],
                destination=self._physical_for[destination_plane][logical],
                kind=kind,
            )
            for logical, source_plane, destination_plane, kind in transfers
        )
        self.validate()

    @staticmethod
    def _resolve_accesses(
        plane: str,
        resolved: dict[ValueRef, PhysicalTensorId],
        accesses: Iterable[tuple[ValueRef | PhysicalTensorId, PhysicalAccess]],
    ) -> dict[PhysicalTensorId, tuple[PhysicalAccess, ...]]:
        physical_accesses = defaultdict(list)
        for storage, access in accesses:
            physical = (
                storage
                if isinstance(storage, PhysicalTensorId)
                else resolved[storage]
            )
            if physical.plane != plane:
                raise ValueError("A physical access crossed storage planes")
            if access not in physical_accesses[physical]:
                physical_accesses[physical].append(access)
        return {
            physical: tuple(users)
            for physical, users in physical_accesses.items()
        }

    def _resolve_plane(
        self,
        plane: str,
    ) -> dict[ValueRef, PhysicalTensorId]:
        resolved = {}
        active = set()

        def resolve(ref: ValueRef) -> PhysicalTensorId:
            known = resolved.get(ref)
            if known is not None:
                return known
            if ref in active:
                raise ValueError(f"Storage relation cycle involving {ref.name}")
            active.add(ref)
            relation = self.relations[plane][ref]
            shares_storage = relation.kind in {
                StorageRelationKind.ALIAS,
                StorageRelationKind.VIEW,
                StorageRelationKind.WRITE_INTO,
                StorageRelationKind.COMPONENT,
            }
            if relation.physical_source is not None:
                physical = relation.physical_source
            elif shares_storage and relation.source is not None:
                physical = resolve(relation.source)
            else:
                physical = PhysicalTensorId(plane, ref)
            active.remove(ref)
            resolved[ref] = physical
            return physical

        for ref in self.logical_tensors:
            resolve(ref)
        return resolved

    @staticmethod
    def _value_for_ref(ref: ValueRef) -> torch.Tensor | None:
        value = node_value(ref.node)
        try:
            for index in ref.path:
                value = value[index]
        except (IndexError, TypeError):
            return None
        return value if isinstance(value, torch.Tensor) else None

    def _logical_component(
        self,
        ref: ValueRef,
        name: str,
    ) -> PhysicalTensorComponent | None:
        logical = self.logical_tensors[ref]
        value = self._value_for_ref(ref)
        dtype = logical.dtype or getattr(value, "dtype", None)
        try:
            bits = get_dtype_width(dtype)
        except (TypeError, ValueError):
            if value is None:
                return None
            bits = value.element_size() * 8
        shape = (
            tuple(int(dim) for dim in value.shape)
            if value is not None
            else tuple(int(dim) for dim in logical.shape)
        )
        try:
            byte_size = math.ceil(math.prod(shape) * bits / 8)
        except (TypeError, ValueError):
            return None
        return PhysicalTensorComponent(
            name=name,
            logical_tensor=ref,
            shape=shape,
            dtype=dtype,
            bits=bits,
            byte_size=byte_size,
            value=value,
        )

    @staticmethod
    def _declarations_match(
        first: _PhysicalComponentDeclaration,
        second: _PhysicalComponentDeclaration,
    ) -> bool:
        metadata_matches = (
            first.shape == second.shape
            and str(first.dtype) == str(second.dtype)
            and first.bits == second.bits
            and first.byte_size == second.byte_size
        )
        if not metadata_matches:
            return False
        return torch.equal(
            first.value.detach().cpu(),
            second.value.detach().cpu(),
        )

    @staticmethod
    def _artifact_declarations_match(
        first: _SerializedArtifactDeclaration,
        second: _SerializedArtifactDeclaration,
    ) -> bool:
        return (
            first.shape == second.shape
            and str(first.dtype) == str(second.dtype)
            and first.bits == second.bits
            and first.byte_size == second.byte_size
            and torch.equal(
                first.value.detach().cpu(),
                second.value.detach().cpu(),
            )
        )

    def _resolve_serialized_artifacts(
        self,
        plane: str,
        resolved: dict[ValueRef, PhysicalTensorId],
        declarations: Iterable[_SerializedArtifactDeclaration],
    ) -> dict[
        tuple[PhysicalTensorId, str],
        SerializedTensorArtifact,
    ]:
        by_key = defaultdict(list)
        for declaration in declarations:
            physical = resolved[declaration.logical_tensor]
            by_key[(physical, declaration.name)].append(declaration)

        artifacts = {}
        for (physical, name), candidates in by_key.items():
            first = candidates[0]
            if any(
                not self._artifact_declarations_match(first, candidate)
                for candidate in candidates[1:]
            ):
                consumers = ", ".join(
                    candidate.logical_tensor.name for candidate in candidates
                )
                raise ValueError(
                    f"{physical.name}: conflicting serialized artifact "
                    f"{name!r} from {consumers}"
                )
            artifacts[(physical, name)] = SerializedTensorArtifact(
                physical_tensor=physical,
                name=name,
                logical_tensor=first.logical_tensor,
                shape=first.shape,
                dtype=first.dtype,
                bits=first.bits,
                byte_size=first.byte_size,
                value=first.value,
            )
        return artifacts

    def _resolve_representations(
        self,
        plane: str,
        resolved: dict[ValueRef, PhysicalTensorId],
        auxiliary: set[PhysicalTensorId],
        declarations: Iterable[_PhysicalComponentDeclaration],
    ) -> dict[
        PhysicalTensorId,
        tuple[
            tuple[PhysicalTensorComponent, ...],
            int | None,
            str | None,
        ],
    ]:
        """Resolve target payload declarations onto exact physical backings."""
        members = defaultdict(list)
        for logical, physical in resolved.items():
            members[physical].append(logical)
        for physical in auxiliary:
            members.setdefault(physical, [])

        by_physical = defaultdict(list)
        for declaration in declarations:
            physical = declaration.physical_tensor
            if physical is None:
                physical = resolved[declaration.logical_tensor]
                relation = self.relations[plane][declaration.logical_tensor]
                expected_component = (
                    relation.component
                    if relation.kind is StorageRelationKind.COMPONENT
                    else None
                )
            else:
                expected_component = None
                if physical not in auxiliary:
                    raise ValueError(
                        f"{physical.name}: explicit physical component "
                        "does not target an auxiliary identity"
                    )
            if (
                expected_component is not None
                and declaration.name != expected_component
            ):
                raise ValueError(
                    f"{declaration.logical_tensor.name}: component "
                    f"{declaration.name!r} conflicts with storage relation "
                    f"{expected_component!r}"
                )
            by_physical[physical].append(declaration)

        representations = {}
        for physical, physical_members in members.items():
            physical_declarations = by_physical.get(physical, ())
            if physical_declarations:
                backing_sizes = {
                    item.backing_size_bytes for item in physical_declarations
                }
                layouts = {item.layout for item in physical_declarations}
                if len(backing_sizes) != 1 or len(layouts) != 1:
                    details = ", ".join(
                        (
                            f"{item.logical_tensor.name}:{item.name}"
                            f"[{item.backing_size_bytes} bytes,"
                            f" {item.layout}]"
                        )
                        for item in physical_declarations
                    )
                    raise ValueError(
                        f"{physical.name}: conflicting shared-backing "
                        f"declarations: {details}"
                    )

                by_name = defaultdict(list)
                for declaration in physical_declarations:
                    by_name[declaration.name].append(declaration)
                components = []
                for name, candidates in sorted(by_name.items()):
                    first = candidates[0]
                    if any(
                        not self._declarations_match(first, candidate)
                        for candidate in candidates[1:]
                    ):
                        consumers = ", ".join(
                            candidate.logical_tensor.name
                            for candidate in candidates
                        )
                        raise ValueError(
                            f"{physical.name}: conflicting payloads for "
                            f"component {name!r} from {consumers}"
                        )
                    components.append(
                        PhysicalTensorComponent(
                            name=name,
                            logical_tensor=first.logical_tensor,
                            shape=first.shape,
                            dtype=first.dtype,
                            bits=first.bits,
                            byte_size=first.byte_size,
                            value=first.value,
                        )
                    )

                declared_names = {component.name for component in components}
                required_names = {
                    relation.component
                    for ref in physical_members
                    for relation in (self.relations[plane][ref],)
                    if relation.kind is StorageRelationKind.COMPONENT
                    and relation.component is not None
                }
                if not required_names.issubset(declared_names):
                    missing = ", ".join(sorted(required_names - declared_names))
                    raise ValueError(
                        f"{physical.name}: missing physical components: "
                        f"{missing}"
                    )
                has_primary_member = any(
                    self.relations[plane][ref].kind
                    is not StorageRelationKind.COMPONENT
                    for ref in physical_members
                )
                if has_primary_member and "primary" not in declared_names:
                    raise ValueError(
                        f"{physical.name}: target representation has no "
                        "primary component"
                    )
                representations[physical] = (
                    tuple(components),
                    next(iter(backing_sizes)),
                    next(iter(layouts)),
                )
                continue

            named_component_relations = [
                self.relations[plane][ref]
                for ref in physical_members
                if (
                    self.relations[plane][ref].kind
                    is StorageRelationKind.COMPONENT
                    and self.relations[plane][ref].component is not None
                )
            ]
            if named_component_relations:
                names = ", ".join(
                    sorted(
                        {
                            relation.component
                            for relation in named_component_relations
                        }
                    )
                )
                raise ValueError(
                    f"{physical.name}: target storage components require an "
                    f"explicit backing representation: {names}"
                )

            tuple_components = []
            for ref in physical_members:
                relation = self.relations[plane][ref]
                transform = relation.transform
                if (
                    relation.kind is StorageRelationKind.COMPONENT
                    and transform is not None
                    and transform.kind == "tuple_output"
                ):
                    result_index = dict(transform.parameters)["result_index"]
                    component = self._logical_component(
                        ref,
                        f"output[{result_index}]",
                    )
                    if component is not None:
                        tuple_components.append(component)
            if tuple_components:
                representations[physical] = (
                    tuple(tuple_components),
                    sum(component.byte_size for component in tuple_components),
                    "tuple",
                )
                continue

            owner = (
                physical.owner
                if physical.owner in physical_members
                else next(iter(physical_members), None)
            )
            component = (
                None
                if owner is None
                else self._logical_component(owner, "primary")
            )
            representations[physical] = (
                () if component is None else (component,),
                None if component is None else component.byte_size,
                None if component is None else "dense",
            )
        return representations

    def _build_physical_tensors(
        self,
        plane: str,
        resolved: dict[ValueRef, PhysicalTensorId],
        auxiliary: set[PhysicalTensorId],
        accesses: dict[PhysicalTensorId, tuple[PhysicalAccess, ...]],
        representations: dict[
            PhysicalTensorId,
            tuple[
                tuple[PhysicalTensorComponent, ...],
                int | None,
                str | None,
            ],
        ],
    ) -> dict[PhysicalTensorId, PhysicalTensor]:
        members = defaultdict(list)
        physical_relations = defaultdict(list)
        for ref, physical in resolved.items():
            members[physical].append(ref)
            physical_relations[physical].append(self.relations[plane][ref])
        for physical in auxiliary:
            members.setdefault(physical, [])
            physical_relations.setdefault(physical, [])

        def ref_order(ref: ValueRef):
            return self.index.source_positions[ref.node], ref.path

        return {
            physical: PhysicalTensor(
                id=physical,
                logical_tensors=tuple(sorted(physical_members, key=ref_order)),
                relations=tuple(
                    sorted(
                        physical_relations[physical],
                        key=lambda relation: ref_order(relation.result),
                    )
                ),
                components=representations[physical][0],
                backing_size_bytes=representations[physical][1],
                layout=representations[physical][2],
                accesses=accesses.get(physical, ()),
            )
            for physical, physical_members in members.items()
        }

    def _build_intervals(
        self,
        plane: str,
        resolved: dict[ValueRef, PhysicalTensorId],
        accesses: dict[PhysicalTensorId, tuple[PhysicalAccess, ...]],
        positions: dict[Node, int],
    ) -> dict[PhysicalTensorId, StorageInterval]:
        plan_end = len(positions) + 1
        births = defaultdict(list)
        deaths = defaultdict(list)
        persistent = set()

        for ref, physical in resolved.items():
            position = positions[ref.node]
            births[physical].append(position)
            if ref.node.op == "get_attr":
                persistent.add(physical)
            user_positions = [
                positions[user] for user in self.logical_tensors[ref].users
            ]
            deaths[physical].append(max(user_positions, default=position) + 1)
        for physical, physical_accesses in accesses.items():
            user_positions = [
                positions[access.operation] for access in physical_accesses
            ]
            if user_positions:
                births[physical].append(min(user_positions))
                deaths[physical].append(max(user_positions) + 1)

        intervals = {}
        for physical in self.physical_tensors[plane]:
            owner_position = positions[physical.owner.node]
            birth = min(
                owner_position,
                *births.get(physical, (owner_position,)),
            )
            if physical in persistent:
                death = plan_end
            else:
                death = max(deaths.get(physical, (owner_position + 1,)))
            intervals[physical] = StorageInterval(
                physical_tensor=physical,
                birth=birth,
                death=max(birth + 1, death),
            )
        return intervals

    def _index_releases(
        self,
        intervals: dict[PhysicalTensorId, StorageInterval],
        positions: dict[Node, int],
    ) -> dict[Node, tuple[PhysicalTensorId, ...]]:
        by_position = {position: node for node, position in positions.items()}
        releases = defaultdict(list)
        for physical, interval in intervals.items():
            node = by_position.get(interval.death - 1)
            if node is not None:
                releases[node].append(physical)
        return {
            node: tuple(
                sorted(
                    physical_tensors,
                    key=lambda physical: physical.name,
                )
            )
            for node, physical_tensors in releases.items()
        }

    def ref(
        self,
        value: Node | ValueRef,
        path: tuple[int, ...] = (),
    ) -> ValueRef:
        if isinstance(value, ValueRef):
            return value
        return self.index.ref(value, path)

    def relation(
        self,
        value: Node | ValueRef,
        plane: str | StoragePlane,
    ) -> StorageRelation:
        """Return one finalized logical storage relation."""
        return self.relations[_plane_name(plane)][self.ref(value)]

    def trace_storage_chain(
        self,
        value: Node | ValueRef,
        plane: str | StoragePlane,
        *,
        follow_kinds: Iterable[StorageRelationKind] | None = None,
        follow_copy: Callable[[ValueRef], bool] | None = None,
        follow_write_input: Callable[[ValueRef], bool] | None = None,
    ) -> tuple[ValueRef, ...]:
        """Trace immutable relations to the owner of a physical backing."""
        plane_name = _plane_name(plane)
        return _trace_storage_relations(
            self.index,
            self.relations[plane_name],
            self.ref(value),
            follow_kinds=follow_kinds,
            follow_copy=follow_copy,
            follow_write_input=follow_write_input,
        )

    def output_refs(self, node: Node) -> tuple[ValueRef, ...]:
        """Return the exact logical tensor results produced by ``node``."""
        return self.index.node_outputs.get(node, ())

    def has_logical_value(
        self,
        value: Node | ValueRef,
        path: tuple[int, ...] = (),
    ) -> bool:
        """Return whether an exact logical tensor result is indexed."""
        ref = value if isinstance(value, ValueRef) else ValueRef(value, path)
        return ref in self.logical_tensors

    def physical_id(
        self,
        value: Node | ValueRef,
        plane: str,
    ) -> PhysicalTensorId:
        plane_name = _plane_name(plane)
        if plane_name not in self._physical_for:
            raise ValueError(f"Storage plan has no {plane_name} plane")
        return self._physical_for[plane_name][self.ref(value)]

    def physical_tensor(
        self,
        value: Node | ValueRef | PhysicalTensorId,
        plane: str,
    ) -> PhysicalTensor:
        physical = (
            value
            if isinstance(value, PhysicalTensorId)
            else self.physical_id(value, plane)
        )
        return self.physical_tensors[_plane_name(plane)][physical]

    def physical_component(
        self,
        value: Node | ValueRef | PhysicalTensorId,
        plane: str,
        component: str = "primary",
    ) -> PhysicalTensorComponent:
        """Return one finalized component of an exact physical backing."""
        return self.physical_tensor(value, plane).component(component)

    def backing_size_bytes(
        self,
        value: Node | ValueRef | PhysicalTensorId,
        plane: str,
    ) -> int:
        """Return the complete target-owned physical allocation extent."""
        tensor = self.physical_tensor(value, plane)
        if tensor.backing_size_bytes is None:
            raise ValueError(
                f"{tensor.id.name}: physical backing size is unknown"
            )
        return tensor.backing_size_bytes

    def logical_size_bytes(
        self,
        value: Node | ValueRef,
    ) -> int:
        """Return the packed byte size of one logical tensor view."""
        ref = self.ref(value)
        component = self._logical_component(ref, "logical")
        if component is None:
            raise ValueError(f"{ref.name}: logical tensor size is unknown")
        return component.byte_size

    def payload_ref(
        self,
        value: Node | ValueRef | PhysicalTensorId,
        plane: str,
        component: str,
        segment,
    ) -> PhysicalPayloadRef:
        """Bind one finalized plan component to an explicit allocation."""
        physical = self.storage_key(value, plane)
        representation = self.physical_component(
            physical,
            plane,
            component,
        )
        tensor = self.physical_tensor(physical, plane)
        if tensor.backing_size_bytes is None:
            raise ValueError(
                f"{physical.name}: physical backing size is unknown"
            )
        if segment.end - segment.start < tensor.backing_size_bytes:
            raise ValueError(
                f"{physical.name}: placement is smaller than its "
                "physical backing"
            )
        return PhysicalPayloadRef(
            physical_tensor=physical,
            representation=representation,
            segment=segment,
        )

    def serialized_artifact(
        self,
        value: Node | ValueRef | PhysicalTensorId,
        plane: str,
        name: str,
    ) -> SerializedTensorArtifact:
        """Return one finalized nonresident target encoding."""
        physical = self.storage_key(value, plane)
        try:
            return self.serialized_artifacts[_plane_name(plane)][
                (physical, name)
            ]
        except KeyError as error:
            raise KeyError(
                f"{physical.name}: serialized artifact {name!r} "
                "is not declared"
            ) from error

    def serialized_ref(
        self,
        value: Node | ValueRef | PhysicalTensorId,
        plane: str,
        name: str,
    ) -> SerializedTensorRef:
        """Return a finalized reference to a nonresident target encoding."""
        return SerializedTensorRef(
            artifact=self.serialized_artifact(value, plane, name)
        )

    def storage_key(
        self,
        value: Node | ValueRef | PhysicalTensorId,
        plane: str,
    ) -> PhysicalTensorId:
        """Return the validated physical identity for one stored value."""
        plane_name = _plane_name(plane)
        if plane_name not in self.physical_tensors:
            raise ValueError(f"Storage plan has no {plane_name} plane")
        physical = (
            value
            if isinstance(value, PhysicalTensorId)
            else self.physical_id(value, plane_name)
        )
        if physical.plane != plane_name:
            raise ValueError(
                f"{physical.name}: physical storage crossed planes"
            )
        if physical not in self.physical_tensors[plane_name]:
            raise KeyError(
                f"{physical.name} is not declared in the storage plan"
            )
        return physical

    def produced_physical_outputs(
        self,
        node: Node,
        plane: str,
    ) -> tuple[PhysicalTensorId, ...]:
        """Return the distinct physical roots materialized by one FX node."""
        plane_name = _plane_name(plane)
        relations = self.relations[plane_name]
        produced = []
        for output in self.output_refs(node):
            physical = self.physical_id(output, plane_name)
            relation = relations[output]
            if (
                relation.kind
                in {
                    StorageRelationKind.MATERIALIZE,
                    StorageRelationKind.COPY,
                    StorageRelationKind.WRITE_INTO,
                }
                or physical.owner.node is node
            ):
                produced.append(physical)
        return tuple(dict.fromkeys(produced))

    def accesses(
        self,
        value: Node | ValueRef | PhysicalTensorId,
        plane: str,
    ) -> tuple[PhysicalAccess, ...]:
        physical = (
            value
            if isinstance(value, PhysicalTensorId)
            else self.physical_id(value, plane)
        )
        return self._accesses[_plane_name(plane)].get(physical, ())

    def release_after(
        self,
        node: Node,
        plane: str,
    ) -> tuple[PhysicalTensorId, ...]:
        return self._release_after[_plane_name(plane)].get(node, ())

    def validate(self) -> None:
        """Validate logical completeness and plane-local storage identity."""
        logical = set(self.logical_tensors)
        for plane, relations in self.relations.items():
            if set(relations) != logical:
                raise ValueError(
                    f"{plane}: storage relations do not cover every "
                    "logical tensor"
                )
            for ref, physical in self._physical_for[plane].items():
                if physical.plane != plane:
                    raise ValueError(
                        f"{ref.name}: physical storage crossed planes"
                    )
                if physical not in self.physical_tensors[plane]:
                    raise ValueError(
                        f"{ref.name}: physical tensor is not declared"
                    )
            undeclared_accesses = set(self._accesses[plane]) - set(
                self.physical_tensors[plane]
            )
            if undeclared_accesses:
                names = ", ".join(
                    physical.name
                    for physical in sorted(
                        undeclared_accesses,
                        key=lambda value: value.name,
                    )
                )
                raise ValueError(
                    f"{plane}: accesses reference undeclared storage: "
                    f"{names}"
                )
            for tensor in self.physical_tensors[plane].values():
                component_names = {
                    component.name for component in tensor.components
                }
                for relation in tensor.relations:
                    if (
                        relation.kind is StorageRelationKind.COMPONENT
                        and relation.component is not None
                        and relation.component not in component_names
                    ):
                        raise ValueError(
                            f"{tensor.id.name}: relation references missing "
                            f"component {relation.component!r}"
                        )
            for physical, _name in self.serialized_artifacts[plane]:
                if physical not in self.physical_tensors[plane]:
                    raise ValueError(
                        f"{physical.name}: serialized artifact references "
                        "undeclared storage"
                    )
        for transfer in self.transfers:
            if transfer.logical_tensor not in logical:
                raise ValueError(
                    f"{transfer.logical_tensor.name}: transfer references "
                    "an unknown logical tensor"
                )
            for endpoint in (transfer.source, transfer.destination):
                if endpoint not in self.physical_tensors.get(
                    endpoint.plane,
                    {},
                ):
                    raise ValueError(
                        f"{endpoint.name}: transfer endpoint is not declared"
                    )

    def to_dict(self) -> dict:
        """Return a debug representation without serializing FX objects."""
        logical = [
            {
                "name": tensor.ref.name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "inputs": [ref.name for ref in tensor.inputs],
                "users": [user.name for user in tensor.users],
            }
            for tensor in self.logical_tensors.values()
        ]
        physical = {}
        for plane, tensors in self.physical_tensors.items():
            physical[plane] = [
                {
                    "name": tensor.id.name,
                    "backing_size_bytes": tensor.backing_size_bytes,
                    "layout": tensor.layout,
                    "components": [
                        {
                            "name": component.name,
                            "logical_tensor": (component.logical_tensor.name),
                            "shape": list(component.shape),
                            "dtype": str(component.dtype),
                            "bits": component.bits,
                            "byte_size": component.byte_size,
                        }
                        for component in tensor.components
                    ],
                    "serialized_artifacts": [
                        {
                            "name": artifact.name,
                            "logical_tensor": (artifact.logical_tensor.name),
                            "shape": list(artifact.shape),
                            "dtype": str(artifact.dtype),
                            "bits": artifact.bits,
                            "byte_size": artifact.byte_size,
                        }
                        for (
                            (artifact_physical, _artifact_name),
                            artifact,
                        ) in self.serialized_artifacts[plane].items()
                        if artifact_physical == tensor.id
                    ],
                    "logical_tensors": [
                        ref.name for ref in tensor.logical_tensors
                    ],
                    "accesses": [
                        {
                            "operation": access.operation.name,
                            "role": access.role.value,
                            "kind": access.kind,
                        }
                        for access in tensor.accesses
                    ],
                    "relations": [
                        {
                            "result": relation.result.name,
                            "kind": relation.kind.value,
                            "source": (
                                None
                                if relation.source is None
                                else relation.source.name
                            ),
                            "physical_source": (
                                None
                                if relation.physical_source is None
                                else relation.physical_source.name
                            ),
                            "component": relation.component,
                            "transform": (
                                None
                                if relation.transform is None
                                else {
                                    "kind": relation.transform.kind,
                                    "parameters": dict(
                                        relation.transform.parameters
                                    ),
                                }
                            ),
                        }
                        for relation in tensor.relations
                    ],
                    "interval": {
                        "birth": self.intervals[plane][tensor.id].birth,
                        "death": self.intervals[plane][tensor.id].death,
                    },
                }
                for tensor in tensors.values()
            ]
        transfers = [
            {
                "logical_tensor": transfer.logical_tensor.name,
                "source": transfer.source.name,
                "destination": transfer.destination.name,
                "kind": transfer.kind,
            }
            for transfer in self.transfers
        ]
        return {
            "logical_tensors": logical,
            "physical_tensors": physical,
            "transfers": transfers,
        }


__all__ = [
    "GraphIndex",
    "StorageGraph",
    "StoragePlan",
    "materialize_aggregate_copies",
]

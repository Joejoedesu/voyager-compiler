# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Graph-analysis records consumed by Magnet Alpha memory allocation."""

from dataclasses import dataclass
from typing import Mapping, Sequence

from torch.fx import Node

from scheduler.types import OperationClassification
from voyager_compiler.codegen.operands import OperandDescriptor
from voyager_compiler.codegen.storage import ValueRef
from voyager_compiler.codegen.target_backends.magnet_alpha.graph import (
    ShuffleCopyMovement,
)


class AlphaMemoryPlanError(RuntimeError):
    """Raised when an Alpha memory plan is not physically legal."""


@dataclass(frozen=True)
class ResidentWrite:
    backing: Node
    owner: Node
    path: tuple[Node, ...]
    slot_index: int
    channel_offset: int
    slot_channels: int
    logical_backing_channels: int
    resident_channels: int
    concat_channel_index: int | None


@dataclass(frozen=True)
class NmpInputView:
    """One logical channel view into a wider physical NMP row."""

    source_kind: str
    channel_index: int
    split_index: int
    current_channels: int
    total_channels: int
    pitch_gap_words: int
    resident_group_count: int | None = None
    uniform_group_channels: int | None = None

    @property
    def scale(self):
        return self.total_channels / self.current_channels


@dataclass(frozen=True)
class ConcatLayout:
    inputs: tuple[Node, ...]
    shapes: tuple[tuple[int, ...], ...]
    channel_axis: int
    aliases: tuple[Node, ...]


@dataclass(frozen=True)
class ConcatPlan:
    layout: ConcatLayout
    writes: tuple[ResidentWrite, ...]


@dataclass(frozen=True)
class OperationConstraint:
    owner: Node
    implementation: Node
    unit: str
    classification: OperationClassification
    operand_descriptors: tuple[OperandDescriptor, ...]
    implementation_descriptors: tuple[OperandDescriptor, ...]

    @property
    def kind(self):
        return self.classification.kind

    @property
    def operands(self):
        return tuple(
            dict.fromkeys(
                descriptor.tensor for descriptor in self.operand_descriptors
            )
        )

    def operands_for_port(self, name):
        return tuple(
            descriptor.tensor
            for descriptor in self.operand_descriptors
            if descriptor.key == name
        )

    def operand(self, name):
        operands = tuple(dict.fromkeys(self.operands_for_port(name)))
        if len(operands) > 1:
            raise AlphaMemoryPlanError(
                f"Alpha {self.kind} has multiple {name} operands"
            )
        return operands[0] if operands else None

    def implementation_operand(self, name):
        return next(
            (
                descriptor.tensor
                for descriptor in self.implementation_descriptors
                if descriptor.key == name
            ),
            None,
        )


@dataclass(frozen=True)
class MaterializedInputPadding:
    operation: Node
    implementation: Node
    input: Node
    padding: tuple[int, int, int, int]
    physical_shape: tuple[int, ...]
    operand_bits: int
    valid_channels: int
    resident_channels: int
    serialize_channels: bool


@dataclass(frozen=True)
class PhysicalPayloadUse:
    """One operation field awaiting finalized physical placement."""

    operation: Node
    field: str
    logical_tensor: ValueRef
    component: str


@dataclass(frozen=True)
class OperationEgressBoundary:
    """One fully proven terminal host-layout suffix."""

    owners: frozenset[Node]
    source: Node
    dequantize: Node
    padding_slice: Node
    shared_views: tuple[Node, ...]
    host_layouts: tuple[Node, ...]
    output: Node

    @property
    def storage_path(self):
        return (
            self.dequantize,
            self.padding_slice,
            *self.shared_views,
            *self.host_layouts,
        )

    @property
    def boundary_nodes(self):
        return (self.dequantize, *self.host_layouts)


@dataclass(frozen=True)
class ShufflePlan:
    shuffle: Node
    copies: tuple[Node, ...]
    movements: tuple[ShuffleCopyMovement, ...]
    block_channels: int


@dataclass(frozen=True)
class AlphaPreparedMemoryPlan:
    """Frozen output of target graph analysis before physical allocation."""

    persistent_storage: bool
    operation_constraints: tuple[tuple[Node, OperationConstraint], ...]
    eadd_inputs: tuple[tuple[Node, ValueRef], ...]
    materialized_inputs: tuple[tuple[Node, MaterializedInputPadding], ...]
    payload_uses: tuple[PhysicalPayloadUse, ...]
    resident_writes: tuple[tuple[Node, ResidentWrite], ...]
    shuffle_plans: tuple[tuple[Node, ShufflePlan], ...]
    copy_plans: tuple[tuple[Node, tuple[ShufflePlan, ShuffleCopyMovement]], ...]
    cluster_constraints: tuple[tuple[Node, int], ...]
    bank_conflicts: tuple[tuple[Node, frozenset[Node]], ...]

    @property
    def operation_nodes(self) -> frozenset[Node]:
        return frozenset(node for node, _ in self.operation_constraints)

    def constraint_for(self, operation: Node):
        return next(
            (
                constraint
                for owner, constraint in self.operation_constraints
                if owner is operation
            ),
            None,
        )


def freeze_prepared_memory_plan(
    *,
    persistent_storage: bool,
    operation_constraints: Mapping[Node, OperationConstraint],
    eadd_inputs: Mapping[Node, ValueRef],
    materialized_inputs: Mapping[Node, MaterializedInputPadding],
    payload_uses: Sequence[PhysicalPayloadUse],
    resident_writes: Mapping[Node, ResidentWrite],
    shuffle_plans: Mapping[Node, ShufflePlan],
    copy_plans: Mapping[Node, tuple[ShufflePlan, ShuffleCopyMovement]],
    cluster_constraints: Mapping[Node, int],
    bank_conflicts: Mapping[Node, set[Node] | frozenset[Node]],
) -> AlphaPreparedMemoryPlan:
    """Snapshot mutable planning tables without changing their semantics."""
    return AlphaPreparedMemoryPlan(
        persistent_storage=persistent_storage,
        operation_constraints=tuple(operation_constraints.items()),
        eadd_inputs=tuple(eadd_inputs.items()),
        materialized_inputs=tuple(materialized_inputs.items()),
        payload_uses=tuple(payload_uses),
        resident_writes=tuple(resident_writes.items()),
        shuffle_plans=tuple(shuffle_plans.items()),
        copy_plans=tuple(copy_plans.items()),
        cluster_constraints=tuple(cluster_constraints.items()),
        bank_conflicts=tuple(
            (node, frozenset(conflicts))
            for node, conflicts in bank_conflicts.items()
        ),
    )


__all__ = [
    "AlphaMemoryPlanError",
    "AlphaPreparedMemoryPlan",
    "ConcatLayout",
    "ConcatPlan",
    "MaterializedInputPadding",
    "NmpInputView",
    "OperationConstraint",
    "OperationEgressBoundary",
    "PhysicalPayloadUse",
    "ResidentWrite",
    "ShufflePlan",
    "freeze_prepared_memory_plan",
]

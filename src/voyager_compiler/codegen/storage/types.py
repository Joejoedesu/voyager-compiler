# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Logical and physical tensor contracts for memory planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch
from torch.fx import Node

from voyager_compiler.codegen.hardware import TensorRole

if TYPE_CHECKING:
    from voyager_compiler.codegen.memory import Segment


class StoragePlane(str, Enum):
    """Compiler-visible storage planes used by the default mapping path."""

    DRAM = "memory"
    SCRATCHPAD = "scratchpad"


@dataclass(frozen=True)
class ValueRef:
    """One explicit logical tensor result produced by an FX node."""

    node: Node
    path: tuple[int, ...] = ()

    @property
    def name(self) -> str:
        suffix = "".join(f"[{index}]" for index in self.path)
        return f"{self.node.name}{suffix}"


@dataclass(frozen=True)
class PhysicalTensorId:
    """Stable identity of one backing within one storage plane.

    The selected allocator decides whether that backing is resident and
    assigns an address. A plane-scoped identity is therefore not itself an
    allocation record.
    """

    plane: str
    owner: ValueRef
    component: str = "primary"

    @property
    def name(self) -> str:
        suffix = "" if self.component == "primary" else f".{self.component}"
        return f"{self.plane}:{self.owner.name}{suffix}"


@dataclass(frozen=True)
class PhysicalTensorComponent:
    """One exact payload interpretation within a physical backing.

    Components describe physical bytes, not logical FX shapes. A component
    can therefore have a padded shape or a narrower physical dtype than the
    logical tensor that requested it.
    """

    name: str
    logical_tensor: ValueRef
    shape: tuple[int, ...]
    dtype: object
    bits: int
    byte_size: int
    value: torch.Tensor | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Physical component name must not be empty")
        if self.bits <= 0:
            raise ValueError("Physical component width must be positive")
        if self.byte_size < 0:
            raise ValueError("Physical component size must not be negative")
        if self.value is not None:
            if not isinstance(self.value, torch.Tensor):
                raise TypeError("Physical component value must be a tensor")
            if tuple(self.value.shape) != self.shape:
                raise ValueError(
                    "Physical component value does not match its shape"
                )


@dataclass(frozen=True)
class PhysicalPayloadRef:
    """A finalized physical component and its explicit placement.

    The representation is owned by ``StoragePlan``. This reference deliberately
    contains no raw FX node or independently supplied tensor value, preventing
    serialization from rediscovering physical identity through node metadata.
    """

    physical_tensor: PhysicalTensorId
    representation: PhysicalTensorComponent
    segment: "Segment"

    def __post_init__(self) -> None:
        if not all(
            hasattr(self.segment, field)
            for field in ("start", "end", "memory_space")
        ):
            raise TypeError("PhysicalPayloadRef segment must be a Segment")
        if self.segment.end - self.segment.start < (
            self.representation.byte_size
        ):
            raise ValueError(
                f"{self.physical_tensor.name}: component "
                f"{self.representation.name} exceeds its placement"
            )

    @property
    def component(self) -> str:
        return self.representation.name

    @property
    def name(self) -> str:
        """Return a stable file/protobuf name unique to one physical payload."""
        physical = self.physical_tensor.name
        normalized = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in physical
        )
        component = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in self.component
        )
        return f"{normalized}__{component}"


@dataclass(frozen=True)
class SerializedTensorArtifact:
    """One nonresident tensor encoding consumed by post-codegen tooling."""

    physical_tensor: PhysicalTensorId
    name: str
    logical_tensor: ValueRef
    shape: tuple[int, ...]
    dtype: object
    bits: int
    byte_size: int
    value: torch.Tensor = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Serialized artifact name must not be empty")
        if self.bits <= 0 or self.byte_size < 0:
            raise ValueError("Serialized artifact has invalid storage size")
        if not isinstance(self.value, torch.Tensor):
            raise TypeError("Serialized artifact value must be a tensor")
        if tuple(self.value.shape) != self.shape:
            raise ValueError(
                "Serialized artifact value does not match its shape"
            )


@dataclass(frozen=True)
class SerializedTensorRef:
    """A finalized reference to a plan-owned, nonresident tensor artifact."""

    artifact: SerializedTensorArtifact

    @property
    def physical_tensor(self) -> PhysicalTensorId:
        return self.artifact.physical_tensor

    @property
    def name(self) -> str:
        physical = self.physical_tensor.name
        normalized = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in physical
        )
        artifact = "".join(
            character if character.isalnum() or character == "_" else "_"
            for character in self.artifact.name
        )
        return f"{normalized}__artifact__{artifact}"


class StorageRelationKind(str, Enum):
    """How a logical tensor obtains its physical storage."""

    MATERIALIZE = "materialize"
    ALIAS = "alias"
    VIEW = "view"
    COPY = "copy"
    WRITE_INTO = "write_into"
    COMPONENT = "component"


@dataclass(frozen=True)
class ViewTransform:
    """Normalized logical view information retained for address lowering."""

    kind: str
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("View transform kind must be non-empty")
        parameters = tuple(
            (str(name), value) for name, value in self.parameters
        )
        names = tuple(name for name, _value in parameters)
        if len(set(names)) != len(names):
            raise ValueError("View transform parameter names must be unique")
        object.__setattr__(self, "parameters", parameters)

    def parameter(self, name: str, default=None):
        """Return one normalized transform parameter."""
        return next(
            (
                value
                for parameter_name, value in self.parameters
                if parameter_name == name
            ),
            default,
        )


@dataclass(frozen=True)
class StorageRelation:
    """One logical-to-physical storage relation."""

    result: ValueRef
    kind: StorageRelationKind
    source: ValueRef | None = None
    physical_source: PhysicalTensorId | None = None
    transform: ViewTransform | None = None
    component: str | None = None

    def __post_init__(self) -> None:
        if self.source is not None and self.physical_source is not None:
            raise ValueError(
                "A storage relation cannot name logical and physical "
                "sources together"
            )
        shares_storage = self.kind in {
            StorageRelationKind.ALIAS,
            StorageRelationKind.VIEW,
            StorageRelationKind.WRITE_INTO,
            StorageRelationKind.COMPONENT,
        }
        if shares_storage and (
            self.source is None and self.physical_source is None
        ):
            raise ValueError(f"{self.kind.value} requires a storage source")
        if self.kind == StorageRelationKind.COPY and self.source is None:
            raise ValueError("copy requires a logical source")


@dataclass(frozen=True)
class LogicalTensor:
    """One logical graph value, independent of storage decisions."""

    ref: ValueRef
    shape: tuple[Any, ...]
    dtype: object
    inputs: tuple[ValueRef, ...]
    users: tuple[Node, ...]


@dataclass(frozen=True)
class PhysicalTensor:
    """One planned backing shared by aliases, views, or components."""

    id: PhysicalTensorId
    logical_tensors: tuple[ValueRef, ...]
    relations: tuple[StorageRelation, ...]
    components: tuple[PhysicalTensorComponent, ...] = ()
    backing_size_bytes: int | None = None
    layout: str | None = None
    accesses: tuple["PhysicalAccess", ...] = ()

    def __post_init__(self) -> None:
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError(
                f"{self.id.name}: physical component names are not unique"
            )
        if self.backing_size_bytes is not None and self.backing_size_bytes < 0:
            raise ValueError("Physical backing size must not be negative")
        if self.backing_size_bytes is not None and any(
            component.byte_size > self.backing_size_bytes
            for component in self.components
        ):
            raise ValueError(
                f"{self.id.name}: component exceeds physical backing"
            )

    def component(self, name: str = "primary") -> PhysicalTensorComponent:
        matches = tuple(
            component for component in self.components if component.name == name
        )
        if len(matches) != 1:
            raise KeyError(
                f"{self.id.name}: physical component {name!r} is not declared"
            )
        return matches[0]


@dataclass(frozen=True)
class PhysicalAccess:
    """One role-typed read or write hidden from ordinary FX dataflow."""

    operation: Node
    role: TensorRole
    kind: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", TensorRole(self.role))
        if self.kind not in {"read", "write"}:
            raise ValueError("Physical access kind must be read or write")


@dataclass(frozen=True)
class StorageTransfer:
    """One explicit movement of a logical tensor between storage planes."""

    logical_tensor: ValueRef
    source: PhysicalTensorId
    destination: PhysicalTensorId
    kind: str = "load"

    def __post_init__(self) -> None:
        if self.source.plane == self.destination.plane:
            raise ValueError("A storage transfer must cross storage planes")


@dataclass(frozen=True)
class StorageInterval:
    """Half-open execution interval for one physical tensor."""

    physical_tensor: PhysicalTensorId
    birth: int
    death: int


__all__ = [
    "LogicalTensor",
    "PhysicalAccess",
    "PhysicalPayloadRef",
    "PhysicalTensor",
    "PhysicalTensorComponent",
    "PhysicalTensorId",
    "SerializedTensorArtifact",
    "SerializedTensorRef",
    "StorageInterval",
    "StoragePlane",
    "StorageRelation",
    "StorageRelationKind",
    "StorageTransfer",
    "ValueRef",
    "ViewTransform",
]

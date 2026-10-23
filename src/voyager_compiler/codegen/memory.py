# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
import math
import operator
import os
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Optional, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch

from voyager_compiler.codegen.execution_order import execution_positions
from voyager_compiler.codegen.hardware import MemoryLevelConfig, TensorRole
from voyager_compiler.codegen.mapping_utils import (
    TARGET_INSTRUCTION_NOP,
    is_addressing_op,
    is_nop,
)
from voyager_compiler.codegen.operands import (
    operand_descriptors,
    submodule_operand_descriptors,
)
from voyager_compiler.codegen.storage.types import StorageRelationKind
from voyager_compiler.codegen.storage_predicates import (
    StorageEffectKind,
    storage_op_traits,
)
from voyager_compiler.pt2e_utils import dtype_byte_size

if TYPE_CHECKING:
    from voyager_compiler.codegen.storage import PhysicalTensorId, ValueRef

logger = logging.getLogger(__name__)

__all__ = [
    "AllocationRecord",
    "END_TO_END_SCRATCHPAD_MAPPING",
    "MemoryAllocator",
    "MemorySpace",
    "PER_OPERATION_SCRATCHPAD_MAPPING",
    "SCRATCHPAD_MAPPING_MODES",
    "ScratchpadAllocator",
    "Segment",
    "align_size",
    "allocation_record_label",
    "compatible_memory_names",
    "compute_tensor_size",
    "index_allocation_records",
    "is_tensor_split",
    "logical_tensor_segment",
    "live_allocation_records",
    "payload_size_bytes",
    "physical_access_requirements",
    "validate_allocation_lifecycles",
    "validate_execution_liveness",
]


PER_OPERATION_SCRATCHPAD_MAPPING = "per_operation"
END_TO_END_SCRATCHPAD_MAPPING = "end_to_end"
SCRATCHPAD_MAPPING_MODES = (
    PER_OPERATION_SCRATCHPAD_MAPPING,
    END_TO_END_SCRATCHPAD_MAPPING,
)


def is_tensor_split(node) -> bool:
    """Return whether a node produces channel or dimension sub-tensors."""
    return storage_op_traits(node.target).effect is StorageEffectKind.SPLIT_VIEW


class MemorySpace(IntEnum):
    """Logical memory spaces available to the system/hardware."""

    DRAM = 0  # main system memory (e.g., DDR)
    SCRATCHPAD = 1  # accelerator-global SW-managed memory


@dataclass
class Segment:
    start: Union[float, int]
    end: Union[float, int]
    memory_space: Optional[int] = None
    node: Optional[torch.fx.Node] = None
    address: Optional[int] = None
    storage_plane: Optional[MemorySpace] = None
    memory_instance: Optional[str] = None

    def __post_init__(self) -> None:
        s_raw = self.start
        e_raw = self.end

        s = int(s_raw)  # truncate toward zero (matches your original)
        e = math.ceil(e_raw)  # round end up

        if s != s_raw:
            logger.warning(
                "Segment start %r is not an integer. Rounding to %d.", s_raw, s
            )
        if e != e_raw:
            logger.warning(
                "Segment end %r is not an integer. Rounding up to %d.", e_raw, e
            )

        if e < s:
            raise ValueError(f"Segment end ({e}) is less than start ({s}).")

        self.start = s
        self.end = e

        if self.address is not None:
            self.address = int(self.address)
        if self.storage_plane is None and isinstance(
            self.memory_space,
            MemorySpace,
        ):
            self.storage_plane = self.memory_space
        if self.memory_instance is None:
            if self.storage_plane is MemorySpace.DRAM:
                self.memory_instance = "DRAM"
            elif self.storage_plane is MemorySpace.SCRATCHPAD:
                self.memory_instance = "Scratchpad"


@dataclass(frozen=True)
class AllocationRecord:
    """Target-independent ownership of one allocation over its lifetime."""

    allocation_id: int
    physical_tensor: "PhysicalTensorId"
    logical_tensors: tuple["ValueRef", ...]
    role: TensorRole
    memory_name: str
    start: int
    end: int
    memory_space: int
    address: int | None
    birth_position: int
    death_position: int
    primary: bool


def allocation_record_label(record: AllocationRecord) -> str:
    """Return a stable human-readable allocation identifier."""
    suffix = "" if record.primary else " auxiliary"
    return f"{record.physical_tensor.name}{suffix}" f"#{record.allocation_id}"


def index_allocation_records(history) -> dict[object, list]:
    """Index allocation records by exact logical and physical identities."""
    records_by_tensor = {}

    def add(key, record):
        records = records_by_tensor.setdefault(key, [])
        if record not in records:
            records.append(record)

    for record in history:
        add(record.physical_tensor, record)
        for logical in record.logical_tensors:
            add(logical, record)
    return records_by_tensor


def logical_tensor_segment(
    logical: "ValueRef",
    segment_meta_key: str,
):
    """Return the serialized segment for one exact logical tensor."""
    if not logical.path:
        return logical.node.meta.get(segment_meta_key)
    value = logical.node.meta.get(f"{segment_meta_key}_output_segments")
    try:
        for index in logical.path:
            value = value[index]
    except (IndexError, KeyError, TypeError):
        return None
    return value


def live_allocation_records(segment, position, records):
    """Return allocations containing a segment at one execution position."""
    if not isinstance(segment, Segment):
        return ()
    return tuple(
        record
        for record in records
        if (
            record.birth_position <= position < record.death_position
            and segment.memory_space == record.memory_space
            and record.start <= segment.start
            and segment.end <= record.end
        )
    )


def validate_execution_liveness(
    model,
    records_by_tensor,
    *,
    requires_allocation,
    storage_plan,
    storage_plane,
    error_type=RuntimeError,
    segment_meta_key="scratchpad",
    allocation_name=None,
) -> None:
    """Validate that executable operands are live when they are accessed."""
    allocation_name = allocation_name or segment_meta_key
    positions = execution_positions(storage_plan.execution_order)

    def logical_requires_allocation(logical):
        if requires_allocation(logical.node):
            return True
        if not logical.path:
            return False

        # A tuple-producing FX node is only a container, so the node-level
        # predicate can reject it even though each tensor result needs a
        # backing. Decide that case from the exact physical owner. Component
        # outputs that intentionally alias a non-tensor root remain exempt.
        physical = storage_plan.physical_id(logical, storage_plane)
        owner = physical.owner
        if owner.path:
            return True
        return requires_allocation(owner.node)

    def require_live(operation, logical, position, role):
        if not logical_requires_allocation(logical):
            return
        segment = logical_tensor_segment(
            logical,
            segment_meta_key,
        )
        if not isinstance(segment, Segment):
            raise error_type(
                f"{operation.name}: required {role} tensor {logical.name} "
                f"has no {allocation_name} allocation"
            )
        candidates = live_allocation_records(
            segment,
            position,
            records_by_tensor.get(logical, ()),
        )
        if not candidates:
            raise error_type(
                f"{operation.name}: required {role} tensor {logical.name} "
                f"is not live at execution position {position}"
            )
        if len(candidates) > 1:
            labels = ", ".join(
                allocation_record_label(record) for record in candidates
            )
            raise error_type(
                f"{operation.name}: required {role} tensor {logical.name} "
                f"has ambiguous live allocations: {labels}"
            )

    executable = {"call_function", "call_module", "output"}
    for operation in storage_plan.execution_order.nodes:
        if operation.op not in executable:
            continue
        position = positions.get(operation)
        if position is None:
            raise error_type(
                f"{operation.name}: execution position was not recorded"
            )
        for logical in storage_plan.index.node_inputs.get(operation, ()):
            require_live(operation, logical, position, "input")
        if operation.op != "output":
            for logical in storage_plan.output_refs(operation):
                require_live(operation, logical, position, "output")


def validate_allocation_lifecycles(
    history,
    *,
    memory_level: MemoryLevelConfig,
    access_requirements_for,
    error_type=RuntimeError,
    segment_meta_key="scratchpad",
) -> None:
    """Validate target-independent allocation ownership and live intervals."""
    records = tuple(history)
    allocation_ids = [record.allocation_id for record in records]
    if len(set(allocation_ids)) != len(allocation_ids):
        raise error_type("allocation history contains duplicate allocation IDs")

    for record in records:
        label = allocation_record_label(record)
        try:
            memory = memory_level.memory(record.memory_name)
        except ValueError as error:
            raise error_type(
                f"{label}: unknown memory {record.memory_name}"
            ) from error

        expected_space = memory_level.memory_index(record.memory_name)
        if record.memory_space != expected_space:
            raise error_type(
                f"{label}: partition {record.memory_space} does not match "
                f"{record.memory_name} ({expected_space})"
            )
        requirements = access_requirements_for(record.physical_tensor)
        accepted = (
            memory.accepts_accesses(requirements)
            if requirements
            else memory.accepts(record.role)
        )
        if not accepted:
            raise error_type(
                f"{label}: {record.memory_name} does not accept physical "
                f"accesses "
                f"{sorted((role.value, unit) for role, unit in requirements)}"
            )
        if record.birth_position >= record.death_position:
            raise error_type(
                f"{label}: invalid live interval "
                f"[{record.birth_position}, {record.death_position})"
            )
        if record.start < 0 or record.end <= record.start:
            raise error_type(
                f"{label}: invalid byte range "
                f"[{record.start}, {record.end})"
            )
        if len(set(record.logical_tensors)) != len(record.logical_tensors):
            raise error_type(
                f"{label}: allocation contains duplicate logical tensors"
            )
        for logical in record.logical_tensors:
            segment = logical_tensor_segment(
                logical,
                segment_meta_key,
            )
            inside = (
                isinstance(segment, Segment)
                and segment.memory_space == record.memory_space
                and record.start <= segment.start
                and segment.end <= record.end
            )
            if not inside:
                raise error_type(
                    f"{label}: logical tensor {logical.name} is outside "
                    "its backing allocation"
                )


def _fused_sizing_transparent_source(node):
    if (
        node.target == torch.ops.quantized_ops.dequantize.default
        and node.meta.get("fused") is True
        and node.args
    ):
        return node.args[0]
    return None


def _find_user_with_target(node: torch.fx.Node, targets):
    if not isinstance(targets, set):
        if isinstance(targets, (list, tuple)):
            targets = set(targets)
        else:
            targets = {targets}

    for user in node.users:
        if user.target in targets and user.args[0] == node:
            return user

        # Check for users of fused dequantization nodes
        if (
            user.target == torch.ops.quantized_ops.dequantize.default
            and user.meta.get("fused") is True
        ):
            found = _find_user_with_target(user, targets)
            if found is not None:
                return found

        if user.op == "call_module":
            owner = getattr(user.graph, "owning_module", None)
            submodule = (
                user.meta.get("submodule")
                if owner is None
                else owner.get_submodule(str(user.target))
            )
            if not isinstance(submodule, torch.fx.GraphModule):
                raise RuntimeError(f"Cannot inspect submodule {user.target}")
            for inner in submodule.graph.nodes:
                if inner.target not in targets:
                    continue
                primary_input = (
                    inner.args[0]
                    if inner.args and isinstance(inner.args[0], torch.fx.Node)
                    else next(iter(inner.all_input_nodes), None)
                )
                primary_keys = {
                    descriptor.key
                    for descriptor in operand_descriptors(inner)
                    if descriptor.tensor is primary_input
                }
                descriptors = submodule_operand_descriptors(
                    user,
                    inner,
                    owner,
                    extra_transparent_source=(_fused_sizing_transparent_source),
                )
                if any(
                    descriptor.key in primary_keys
                    and descriptor.tensor.meta.get(
                        "source_node",
                        descriptor.tensor,
                    )
                    is node
                    for descriptor in descriptors
                ):
                    return inner

    return None


def align_size(size, width):
    """Round a byte size up to an optional allocation granularity."""
    if width is None:
        return size
    return (size + width - 1) // width * width


def payload_size_bytes(value, operand_bits: int) -> int:
    """Return packed storage bytes for a tensor or element count."""
    if operand_bits <= 0:
        raise ValueError("operand_bits must be positive")
    numel = value.numel() if isinstance(value, torch.Tensor) else int(value)
    if numel < 0:
        raise ValueError("tensor element count must be non-negative")
    return math.ceil(numel * operand_bits / 8)


def compatible_memory_names(
    memory_level: MemoryLevelConfig,
    role: TensorRole,
    access_requirements=(),
) -> tuple[str, ...]:
    """Order memories satisfying every exact physical-tensor access."""
    primary = memory_level.memory_name_for_role(role)
    ordered = (
        primary,
        *(
            memory.name
            for memory in memory_level.instances
            if memory.name != primary
        ),
    )
    requirements = frozenset(access_requirements)
    return tuple(
        name
        for name in ordered
        if (
            memory_level.memory(name).accepts_accesses(requirements)
            if requirements
            else memory_level.memory(name).accepts(role)
        )
    )


def physical_access_requirements(
    storage_plan,
    hardware_config,
    plane,
    value,
    operator_traits=None,
):
    """Resolve exact ``(tensor role, compute unit)`` physical accesses."""
    if hardware_config is None:
        return frozenset()
    physical = storage_plan.storage_key(value, plane)
    tensor = storage_plan.physical_tensor(physical, plane)
    bindings = hardware_config.compute_unit_bindings()

    def execution_unit(operation):
        if operation.op not in {"call_function", "call_module"}:
            return None
        if (
            operation.target == operator.getitem
            or is_addressing_op(operation)
            or is_nop(operation)
            or operation.meta.get(TARGET_INSTRUCTION_NOP, False)
        ):
            return None
        requested = operation.meta.get("compute_unit")
        if requested is None:
            return bindings.get("matrix", hardware_config.pe.name)
        requested = str(requested)
        unit = bindings.get(requested, requested)
        try:
            hardware_config.compute_unit(unit)
        except ValueError as error:
            raise ValueError(
                f"{operation.name}: unknown compute unit {requested!r}"
            ) from error
        return unit

    def input_roles(operation, logical):
        try:
            descriptors = operand_descriptors(
                operation,
                storage_plan.index.model,
                operator_traits=operator_traits,
            )
        except (RuntimeError, TypeError) as error:
            raise ValueError(
                f"{operation.name}: exact operand descriptors are "
                "required for physical placement"
            ) from error
        source = logical.node.meta.get(
            "source_node",
            logical.node,
        )
        roles = frozenset(
            descriptor.role
            for descriptor in descriptors
            if descriptor.tensor.meta.get(
                "source_node",
                descriptor.tensor,
            )
            is source
        )
        if not roles:
            raise ValueError(
                f"{operation.name}: logical input {logical.name} has no "
                "exact operand descriptor"
            )
        return roles

    requirements = set()
    visited_producers = set()
    for logical in tensor.logical_tensors:
        producer_unit = execution_unit(logical.node)
        if (
            producer_unit is not None
            and logical.node not in visited_producers
            and physical
            in storage_plan.produced_physical_outputs(
                logical.node,
                plane,
            )
        ):
            requirements.add((TensorRole.OUTPUT, producer_unit))
            visited_producers.add(logical.node)
        for user in storage_plan.logical_tensors[logical].users:
            unit = execution_unit(user)
            if unit is not None:
                roles = input_roles(user, logical)
                relation = storage_plan.relations[plane][logical]
                if (
                    relation.kind is StorageRelationKind.COMPONENT
                    and relation.source is not None
                ):
                    try:
                        carrier_roles = input_roles(
                            user,
                            relation.source,
                        )
                    except ValueError:
                        carrier_roles = frozenset()
                    if carrier_roles:
                        # A component embedded in a payload travels through
                        # the payload's physical port. Its logical SCALE role
                        # must not force the combined weight/activation word
                        # into a different memory instance.
                        roles = carrier_roles
                requirements.update((role, unit) for role in roles)

    for access in storage_plan.accesses(physical, plane):
        unit = execution_unit(access.operation)
        if unit is None:
            continue
        requirements.add((access.role, unit))
    return frozenset(requirements)


def compute_tensor_size(
    node,
    shape=None,
    is_scratchpad_output=False,
    bank_width=None,
    unroll_dim=None,
    operator_traits=None,
):
    val = node.value
    if isinstance(val, torch.Tensor):
        dtype = node.meta.get("dtype") or val.dtype
        numel = math.prod(shape) if shape is not None else val.numel()
        tensor_size = numel * dtype_byte_size(dtype)

        conv_input_channel_dims = {
            torch.ops.aten.conv2d.default: 1,
            torch.ops.quantized_ops.conv2d.default: -1,
            **(
                {}
                if operator_traits is None
                else operator_traits.conv2d_input_channel_dims
            ),
        }
        conv_user = _find_user_with_target(
            node,
            tuple(conv_input_channel_dims),
        )

        if conv_user is not None:
            dim = conv_input_channel_dims[conv_user.target]
            if val.shape[dim] == 3:
                logger.debug(f"Increase memory for conv2d input {node} by 3x")
                tensor_size *= 3

        # Allocate extra memory for intermediate results like mean and variance.
        # TODO: Should only do this when allocating scratchpad memory for the
        # specific operation. E.g. if a node is consumed by both a softmax and
        # an add node, we shouldn't increase the size for the add path.
        if not is_scratchpad_output:
            if _find_user_with_target(node, torch.ops.aten.softmax.int):
                logger.debug(f"Increase memory for softmax input {node} by 2x")
                tensor_size = tensor_size * 2

            if _find_user_with_target(node, torch.ops.aten.layer_norm.default):
                logger.debug(
                    f"Increase memory for layer_norm input {node} by 2x"
                )
                tensor_size = (tensor_size + numel) * 2

        return align_size(tensor_size, bank_width)

    if isinstance(val, (tuple, list)):
        if shape is not None:
            key = "tiled_output_sizes"
            numel = [math.prod(s) for s in shape]
        else:
            key = "output_sizes"
            numel = [t.numel() for t in val]

        # Sparse outputs need to be aligned with hardware unroll dimension
        if unroll_dim is not None:
            numel = [align_size(s, unroll_dim) for s in numel]

        dtypes = node.meta.get("dtype") or [None for _ in val]

        sizes = [
            align_size(n * dtype_byte_size(dt or t.dtype), bank_width)
            for t, n, dt in zip(val, numel, dtypes)
        ]

        node.meta[key] = tuple(sizes)
        return sum(sizes)

    logger.warning(f"Node {node} has a non-tensor output")
    return None


def _physical_storage_key(plan, plane, value):
    """Resolve one allocator key, allowing only scalar-output raw nodes."""
    if plan is not None:
        return plan.storage_key(value, plane)
    if isinstance(value, torch.fx.Node):
        output = getattr(value, "value", value.meta.get("val"))
        if isinstance(output, (tuple, list)):
            raise RuntimeError(
                f"{value.name}: multi-output allocation requires a bound "
                "storage plan and an exact ValueRef or PhysicalTensorId"
            )
    return value


class MemoryAllocator:
    """
    This class implements a simple first-fit memory manager for allocating
    memory partitions to tensors.

    Attributes:
        total_memory (int): The total amount of memory available for allocation.
        bank_width (int): The alignment requirement for memory allocations.
        memory_space (MemorySpace): The type of memory space to use for allocations.
    """

    def __init__(
        self,
        total_memory=None,
        bank_width=None,
        memory_space=None,
        *,
        storage_plane=None,
        memory_instance=None,
    ):
        self.total_memory = (
            (1 << 63) - 1 if total_memory is None else total_memory
        )
        self.bank_width = bank_width
        self.memory_space = (
            MemorySpace.DRAM if memory_space is None else memory_space
        )
        self.storage_plane = (
            (MemorySpace.DRAM if memory_space is None else memory_space)
            if storage_plane is None
            and (memory_space is None or isinstance(memory_space, MemorySpace))
            else storage_plane
        )
        self.memory_instance = memory_instance
        if (
            self.memory_instance is None
            and self.storage_plane is MemorySpace.DRAM
        ):
            self.memory_instance = "DRAM"

        self.segments = [
            Segment(
                0,
                self.total_memory,
                self.memory_space,
                storage_plane=self.storage_plane,
                memory_instance=self.memory_instance,
            )
        ]
        self.memory_map = {}
        self.snapshots = []
        self._storage_plan = None
        self._storage_plane = None
        self._operator_traits = None

    def bind_operator_traits(self, operator_traits) -> None:
        """Bind traits from the selected target for storage sizing."""
        self._operator_traits = operator_traits

    def bind_storage_plan(self, plan, plane) -> None:
        """Use one analyzed physical identity for allocation and release."""
        plane = getattr(plane, "value", plane)
        if self._storage_plan is not None:
            if self._storage_plan is plan and self._storage_plane == plane:
                return
            raise RuntimeError("memory allocator storage plan is already bound")
        if self.memory_map:
            raise RuntimeError(
                "storage plan must be bound before memory allocation"
            )
        self._storage_plan = plan
        self._storage_plane = plane

    def _storage_key(self, value):
        return _physical_storage_key(
            self._storage_plan,
            self._storage_plane,
            value,
        )

    @staticmethod
    def _owner_node(key, fallback):
        owner = getattr(key, "owner", None)
        return getattr(owner, "node", fallback)

    def allocate_memory(
        self,
        node,
        role=None,
        *,
        physical_tensor=None,
        tensor_size=None,
    ):
        """Allocate ``node`` using an optional explicit physical identity.

        ``node`` remains the logical/execution owner used for sizing.  The
        allocator map is keyed only by ``physical_tensor`` once storage
        analysis has selected a multi-output result.
        """
        key = self._storage_key(
            node if physical_tensor is None else physical_tensor
        )
        existing = self.memory_map.get(key)
        if existing is not None:
            return Segment(
                existing.start,
                existing.end,
                existing.memory_space,
                address=existing.address,
                storage_plane=existing.storage_plane,
                memory_instance=existing.memory_instance,
            )

        owner = (
            node if physical_tensor is not None else self._owner_node(key, node)
        )
        if not hasattr(owner, "value"):
            logger.warning(f"Node {node} does not have a value attribute")
            return None

        if tensor_size is None:
            tensor_size = compute_tensor_size(
                owner,
                bank_width=self.bank_width,
                operator_traits=self._operator_traits,
            )

        if tensor_size is None:
            return None

        tensor_size = align_size(tensor_size, self.bank_width)

        for index, segment in enumerate(self.segments):
            segment_size = segment.end - segment.start
            if segment.node is None and segment_size >= tensor_size:
                if segment_size > tensor_size:
                    new_segment = Segment(
                        start=segment.start + tensor_size,
                        end=segment.end,
                        memory_space=self.memory_space,
                        storage_plane=self.storage_plane,
                        memory_instance=self.memory_instance,
                    )
                    segment.end = segment.start + tensor_size
                    self.segments.insert(index + 1, new_segment)

                self.memory_map[key] = segment
                segment.node = key

                return Segment(
                    segment.start,
                    segment.end,
                    self.memory_space,
                    storage_plane=self.storage_plane,
                    memory_instance=self.memory_instance,
                )

        raise RuntimeError(f"Memory allocation failed for tensor {owner.name}")

    def free_memory(self, node):
        key = self._storage_key(node)
        if key in self.memory_map:
            segment = self.memory_map[key]
            segment.node = None
            self.merge_segments()
            del self.memory_map[key]

    def allocation_segment(self, value):
        """Return the live segment for one exact physical identity."""
        segment = self.memory_map.get(self._storage_key(value))
        if segment is None:
            return None
        return Segment(
            segment.start,
            segment.end,
            segment.memory_space,
            address=segment.address,
            storage_plane=segment.storage_plane,
            memory_instance=segment.memory_instance,
        )

    def create_view(self, segment, byte_offset, size):
        """Return a byte-addressed view into an existing allocation."""
        if byte_offset < 0:
            raise ValueError("View byte offset cannot be negative")
        start = segment.start + byte_offset
        if start + size > segment.end:
            raise ValueError("View exceeds its backing allocation")
        return Segment(
            start,
            start + size,
            segment.memory_space,
            address=(
                segment.address + byte_offset
                if segment.address is not None
                else None
            ),
            storage_plane=segment.storage_plane,
            memory_instance=segment.memory_instance,
        )

    def merge_segments(self):
        i = 0
        while i < len(self.segments) - 1:
            current_partition = self.segments[i]
            next_partition = self.segments[i + 1]
            if current_partition.node is None and next_partition.node is None:
                current_partition.end = next_partition.end
                self.segments.pop(i + 1)
            else:
                i += 1

    def snapshot(self):
        partitions = [
            Segment(
                start=segment.start,
                end=segment.end,
                node=segment.node.name if segment.node else None,
            )
            for segment in self.segments[:-1]
        ]
        self.snapshots.append(partitions)

    def dump_snapshots(
        self, filename="dump_snapshot.png", colormap_name="tab20"
    ):
        """
        Plots memory usage over time from a list of memory snapshots.
        Tensors (partitions with nodes) cycle through colors from the colormap.
        Free partitions are shown in gray.
        """
        fig, ax = plt.subplots(figsize=(10, 10))

        cmap = cm.get_cmap(colormap_name)
        color_cycle = [cmap(i) for i in range(cmap.N)]
        free_color = (0.85, 0.85, 0.85)

        id_to_color = {}
        color_index = 0

        for t, snapshot in enumerate(self.snapshots):
            for segment in snapshot:
                if segment.node is None:
                    color = free_color
                else:
                    if segment.node not in id_to_color:
                        id_to_color[segment.node] = color_cycle[
                            color_index % len(color_cycle)
                        ]
                        color_index += 1
                    color = id_to_color[segment.node]

                ax.bar(
                    x=t,
                    height=segment.end - segment.start,
                    width=1.0,
                    bottom=segment.start,
                    color=color,
                    linewidth=0,
                )

        def format_bytes(x, _):
            if x >= 1 << 30:
                return f"{x / (1 << 30):.0f}GB"
            elif x >= 1 << 20:
                return f"{x / (1 << 20):.0f}MB"
            elif x >= 1 << 10:
                return f"{x / (1 << 10):.0f}KB"
            else:
                return f"{x:.0f}B"

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_bytes))

        max_bytes = max(p.end for snapshot in self.snapshots for p in snapshot)
        max_mb = max_bytes / (1 << 20)

        # Auto interval using base-10 logic
        locator = mticker.MaxNLocator(
            nbins="auto", steps=[1, 2, 5, 10], integer=True
        )
        tick_vals_mb = locator.tick_values(0, max_mb)
        tick_vals_bytes = [int(mb * (1 << 20)) for mb in tick_vals_mb]

        ax.set_yticks(tick_vals_bytes)

        ax.set_title("Active Memory Timeline")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0)


class ScratchpadAllocator:
    """Allocate model-wide tensors in accelerator-global scratchpad.

    Each global-memory instance retains its own first-fit arena. Tensor roles
    select the instance, and ``Tensor.scratchpad.partition`` matches the
    instance order in ``MemoryLevelConfig``. This allocator never assigns
    DRAM addresses; callers must serialize its segments as scratchpad storage.
    """

    def __init__(
        self,
        memory_level: MemoryLevelConfig,
        *,
        hardware_config=None,
    ):
        self.memory_level = memory_level
        self._hardware_config = hardware_config
        self.total_memory = sum(
            instance.capacity_bytes for instance in memory_level.instances
        )
        self._allocators = {
            instance.name: MemoryAllocator(
                total_memory=instance.capacity_bytes,
                bank_width=instance.allocation_alignment_bytes,
                memory_space=index,
                storage_plane=MemorySpace.SCRATCHPAD,
                memory_instance=instance.name,
            )
            for index, instance in enumerate(memory_level.instances)
        }
        self._allocation_arenas = {}
        self._storage_plan = None
        self._storage_plane = None
        self._access_requirements_for = None
        self._operator_traits = None

    def bind_operator_traits(self, operator_traits) -> None:
        """Bind selected-target sizing policy to each physical arena."""
        self._operator_traits = operator_traits
        for allocator in self._allocators.values():
            allocator.bind_operator_traits(operator_traits)

    def bind_storage_plan(self, plan, plane) -> None:
        """Bind every physical instance to the shared storage analysis."""
        plane = getattr(plane, "value", plane)
        if self._storage_plan is not None:
            if self._storage_plan is plan and self._storage_plane == plane:
                return
            raise RuntimeError(
                "scratchpad allocator storage plan is already bound"
            )
        if self._allocation_arenas:
            raise RuntimeError(
                "storage plan must be bound before scratchpad allocation"
            )
        self._storage_plan = plan
        self._storage_plane = plane
        if (
            self._access_requirements_for is None
            and self._hardware_config is not None
        ):
            self._access_requirements_for = lambda value: (
                physical_access_requirements(
                    plan,
                    self._hardware_config,
                    plane,
                    value,
                    self._operator_traits,
                )
            )
        for allocator in self._allocators.values():
            allocator.bind_storage_plan(plan, plane)

    def bind_access_requirements(self, access_requirements_for) -> None:
        """Bind physical-tensor access requirements before allocation."""
        if not callable(access_requirements_for):
            raise TypeError("access_requirements_for must be callable")
        if self._access_requirements_for is not None:
            if self._access_requirements_for is access_requirements_for:
                return
            raise RuntimeError(
                "scratchpad access-requirement resolver is already bound"
            )
        if self._allocation_arenas:
            raise RuntimeError(
                "access requirements must be bound before scratchpad "
                "allocation"
            )
        self._access_requirements_for = access_requirements_for

    def _storage_key(self, value):
        return _physical_storage_key(
            self._storage_plan,
            self._storage_plane,
            value,
        )

    def allocate_memory(
        self,
        node,
        role: TensorRole,
        *,
        physical_tensor=None,
        tensor_size=None,
        access_requirements=None,
    ):
        value = node if physical_tensor is None else physical_tensor
        key = self._storage_key(value)
        if access_requirements is None:
            if (
                self._access_requirements_for is None
                and self._storage_plan is not None
            ):
                raise RuntimeError(
                    "planned scratchpad allocation requires a bound "
                    "access-requirement resolver or HardwareConfig"
                )
            access_requirements = (
                ()
                if self._access_requirements_for is None
                else self._access_requirements_for(key)
            )
        access_requirements = frozenset(access_requirements)
        existing_allocator = self._allocation_arenas.get(key)
        if existing_allocator is not None:
            memory = next(
                memory
                for memory in self.memory_level.instances
                if self._allocators[memory.name] is existing_allocator
            )
            accepted = (
                memory.accepts_accesses(access_requirements)
                if access_requirements
                else memory.accepts(role)
            )
            if not accepted:
                raise ValueError(
                    f"{memory.name} cannot serve physical accesses "
                    f"{sorted((item.value, unit) for item, unit in access_requirements)}"
                )
            return existing_allocator.allocate_memory(
                node,
                role,
                physical_tensor=key,
                tensor_size=tensor_size,
            )
        candidates = compatible_memory_names(
            self.memory_level,
            role,
            access_requirements,
        )
        if not candidates:
            raise ValueError(
                f"{self.memory_level.name}: no memory accepts physical "
                f"accesses "
                f"{sorted((item.value, unit) for item, unit in access_requirements)}"
            )

        last_error = None
        for memory_name in candidates:
            allocator = self._allocators[memory_name]
            try:
                segment = allocator.allocate_memory(
                    node,
                    role,
                    physical_tensor=key,
                    tensor_size=tensor_size,
                )
            except RuntimeError as error:
                last_error = error
                continue
            if segment is not None:
                self._allocation_arenas[key] = allocator
            return segment
        raise last_error

    def free_memory(self, node):
        key = self._storage_key(node)
        allocator = self._allocation_arenas.pop(key, None)
        if allocator is not None:
            allocator.free_memory(key)

    def allocation_segment(self, value):
        """Return the live segment for one exact physical identity."""
        key = self._storage_key(value)
        allocator = self._allocation_arenas.get(key)
        if allocator is None:
            return None
        return allocator.allocation_segment(key)

    def create_view(self, segment, byte_offset, size):
        """Return a view using the selected physical-instance allocator."""
        partition = segment.memory_space
        if partition is None:
            raise ValueError("Scratchpad view requires a memory partition")
        memory = self.memory_level.instances[partition]
        return self._allocators[memory.name].create_view(
            segment,
            byte_offset,
            size,
        )

    def snapshot(self):
        for allocator in self._allocators.values():
            allocator.snapshot()

    def dump_snapshots(self, filename="dump_snapshot.png"):
        if len(self._allocators) == 1:
            next(iter(self._allocators.values())).dump_snapshots(filename)
            return

        root, extension = os.path.splitext(filename)
        for name, allocator in self._allocators.items():
            allocator.dump_snapshots(f"{root}_{name}{extension}")

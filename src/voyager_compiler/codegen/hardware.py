# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared typed hardware model and semantic hardware-pipeline matching."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from math import ceil, lcm

import torch

from voyager_compiler.codegen.targets import DEFAULT_COMPILATION_TARGET

COMMON_IMAGE_INPUT_CHANNELS = frozenset({1, 3})
DEFAULT_DRAM_BANDWIDTH_GBPS = 200.0
DEFAULT_ACCELERATOR_FREQUENCY_GHZ = 1.0


def get_op_overload(op_name: str) -> list[torch._ops.OpOverload]:
    """Return every overload resolved by one symbolic operation name."""
    all_overloads = []
    operator_libraries = (
        torch.ops.aten,
        torch.ops.quantized_ops,
    )
    for library in operator_libraries:
        # Also check the in-place spelling (for example, ``add_`` for
        # ``add``).
        for name in (op_name, f"{op_name}_"):
            if (packet := getattr(library, name, None)) is None:
                continue
            all_overloads.extend(
                getattr(packet, overload) for overload in packet.overloads()
            )
    return all_overloads


class OpMatcher:
    """One hardware-pipeline stage and its optional node predicate."""

    targets: tuple[torch._ops.OpOverload, ...]
    predicate: Callable[[torch.fx.Node], bool] | None = None

    def __init__(self, *ops, predicate=None):
        self.predicate = predicate
        targets = []
        for op in ops:
            if isinstance(op, torch._ops.OpOverload):
                targets.append(op)
            else:
                targets.extend(get_op_overload(op))
        self.targets = tuple(targets)

    def matches(self, node: torch.fx.Node) -> bool:
        if node.target not in self.targets:
            return False
        return self.predicate(node) if self.predicate else True


def get_dtype_width(dtype) -> int:
    value = str(dtype).split(".")[-1]
    bit_search = re.search(r"[^\d](\d+)(_.*)?$", value)
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    return int(bit_search.groups()[0])


def get_node_dtype_width(node, preferred_dtype=None) -> int:
    """Return a node's scalar width, honoring an explicit physical dtype."""
    dtype = preferred_dtype
    if dtype is None:
        dtype = node.meta.get("dtype")
    if dtype is None:
        value = getattr(node, "value", None)
        if isinstance(value, (tuple, list)):
            value = next(
                (
                    item
                    for item in reversed(value)
                    if getattr(item, "dtype", None) is not None
                ),
                None,
            )
        dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise ValueError(f"Node {node.name} has no dtype information")
    return get_dtype_width(dtype)


class TensorRole(str, Enum):
    ACTIVATION = "activation"
    OUTPUT = "output"
    WEIGHT = "weight"
    SCALE = "scale"
    BIAS = "bias"
    PARTIAL_SUM = "partial_sum"


MemoryAccess = tuple[TensorRole, str | None]


@dataclass(frozen=True)
class HierarchyBandwidth:
    """Directional bandwidth between two named hardware resources.

    Endpoints may be memory hierarchy names or compute-unit names. Routes
    without a ``shared_group`` are independent. Routes with the same group
    are treated as mutually exclusive users of one physical interface.
    """

    name: str
    source: str
    destination: str
    bandwidth_bits_per_cycle: float
    roles: frozenset[TensorRole]
    shared_group: str | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("bandwidth route name must not be empty")
        if not self.source or not self.destination:
            raise ValueError("bandwidth route endpoints must not be empty")
        if self.source == self.destination:
            raise ValueError("bandwidth route endpoints must be different")
        if self.bandwidth_bits_per_cycle <= 0:
            raise ValueError(
                f"{self.name}: bandwidth_bits_per_cycle must be positive"
            )
        roles = frozenset(TensorRole(role) for role in self.roles)
        if not roles:
            raise ValueError(f"{self.name}: roles must not be empty")
        if self.shared_group == "":
            raise ValueError(f"{self.name}: shared_group must not be empty")
        object.__setattr__(self, "roles", roles)

    def cycles_for(self, transferred_bits: int) -> float:
        if transferred_bits < 0:
            raise ValueError("transferred bits must be non-negative")
        return transferred_bits / self.bandwidth_bits_per_cycle


@dataclass(frozen=True)
class DTypeMode:
    operand_bits: int
    banks_per_value: int = 1
    output_channel_alignment: int | None = None

    def __post_init__(self):
        if self.operand_bits <= 0:
            raise ValueError("operand_bits must be positive")
        if self.banks_per_value <= 0:
            raise ValueError("banks_per_value must be positive")
        if (
            self.output_channel_alignment is not None
            and self.output_channel_alignment <= 0
        ):
            raise ValueError("output_channel_alignment must be positive")


@dataclass(frozen=True)
class UnrollDimension:
    """One physical parallel dimension of a compute unit."""

    name: str
    extent: int
    dtype_extents: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        if not self.name:
            raise ValueError("unroll dimension name must not be empty")
        if self.extent <= 0:
            raise ValueError(f"{self.name}: extent must be positive")
        dtype_extents = tuple(
            (int(bits), int(extent)) for bits, extent in self.dtype_extents
        )
        bits = [item[0] for item in dtype_extents]
        if len(set(bits)) != len(bits):
            raise ValueError(f"{self.name}: dtype extent widths must be unique")
        if any(bits <= 0 or extent <= 0 for bits, extent in dtype_extents):
            raise ValueError(f"{self.name}: dtype extents must be positive")
        object.__setattr__(self, "dtype_extents", dtype_extents)

    def extent_for(self, input_bits: int | None = None) -> int:
        if input_bits is not None:
            for bits, extent in self.dtype_extents:
                if bits == input_bits:
                    return extent
        return self.extent


@dataclass(frozen=True)
class ComputeUnit:
    """A named compute resource with physical unroll dimensions."""

    name: str
    unroll_dimensions: tuple[UnrollDimension, ...]
    accepted_input_bits: frozenset[int] = frozenset()
    accepted_operations: frozenset[str] = frozenset()
    input_word_bits: int | None = None
    scalar_bits: int | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("compute unit name must not be empty")
        dimensions = tuple(self.unroll_dimensions)
        if not dimensions:
            raise ValueError(f"{self.name}: unroll dimensions are required")
        names = [dimension.name for dimension in dimensions]
        if len(set(names)) != len(names):
            raise ValueError(
                f"{self.name}: unroll dimension names must be unique"
            )
        accepted_input_bits = frozenset(
            int(bits) for bits in self.accepted_input_bits
        )
        if any(bits <= 0 for bits in accepted_input_bits):
            raise ValueError(
                f"{self.name}: accepted input widths must be positive"
            )
        accepted_operations = frozenset(
            str(operation) for operation in self.accepted_operations
        )
        if any(not operation for operation in accepted_operations):
            raise ValueError(
                f"{self.name}: accepted operation names must not be empty"
            )
        if self.input_word_bits is not None and self.input_word_bits <= 0:
            raise ValueError("input_word_bits must be positive")
        if self.scalar_bits is not None and self.scalar_bits <= 0:
            raise ValueError("scalar_bits must be positive")
        object.__setattr__(self, "unroll_dimensions", dimensions)
        object.__setattr__(
            self,
            "accepted_input_bits",
            accepted_input_bits,
        )
        object.__setattr__(
            self,
            "accepted_operations",
            accepted_operations,
        )

    def unroll_dimension(self, name: str) -> UnrollDimension:
        for dimension in self.unroll_dimensions:
            if dimension.name == name:
                return dimension
        raise ValueError(f"{self.name}: unknown unroll dimension {name}")

    def unroll_extent(
        self,
        name: str,
        input_bits: int | None = None,
    ) -> int:
        if (
            input_bits is not None
            and self.accepted_input_bits
            and input_bits not in self.accepted_input_bits
        ):
            raise ValueError(f"{self.name}: unsupported {input_bits}-bit input")
        return self.unroll_dimension(name).extent_for(input_bits)

    def supports_operation(self, operation: str) -> bool:
        return (
            not self.accepted_operations
            or operation in self.accepted_operations
        )

    @property
    def input_lanes(self) -> int:
        """Compatibility view of the default matrix IC dimension."""
        return self.unroll_extent("IC")

    @property
    def output_lanes(self) -> int:
        """Compatibility view of the default matrix OC dimension."""
        return self.unroll_extent("OC")


def PEConfig(
    input_lanes: int,
    output_lanes: int,
    input_word_bits: int | None = None,
    scalar_bits: int | None = None,
) -> ComputeUnit:
    """Construct the legacy unnamed matrix compute resource."""
    return ComputeUnit(
        name="matrix",
        unroll_dimensions=(
            UnrollDimension("IC", input_lanes),
            UnrollDimension("OC", output_lanes),
        ),
        accepted_operations=frozenset({"gemm"}),
        input_word_bits=input_word_bits,
        scalar_bits=scalar_bits,
    )


@dataclass(frozen=True)
class Block:
    """One ordered functional stage within a hardware datapath."""

    name: str
    ops: tuple[str | torch._ops.OpOverload, ...] = ()
    predicate: Callable | None = None
    source: str = ""

    def __post_init__(self):
        if not self.name:
            raise ValueError("datapath block name must not be empty")
        if not isinstance(self.ops, tuple):
            raise TypeError(
                f"{self.name}: ops must be an ordered tuple, got "
                f"{type(self.ops).__name__}"
            )
        for op in self.ops:
            if isinstance(op, str):
                if not op:
                    raise ValueError(
                        f"{self.name}: operation names must not be empty"
                    )
            elif not isinstance(op, torch._ops.OpOverload):
                raise TypeError(
                    f"{self.name}: operation targets must be symbolic names "
                    "or exact OpOverload objects"
                )


@dataclass(frozen=True)
class Unit:
    """One node in a datapath graph, with internally ordered blocks."""

    name: str
    blocks: tuple[Block, ...] = ()
    issue: Block | None = None
    compute_unit: ComputeUnit | None = None


@dataclass(frozen=True)
class Link:
    """A directed connection between two named datapath units."""

    source: str
    destination: str


@dataclass(frozen=True)
class Datapath:
    """Immutable compute-unit topology used to derive fusion patterns."""

    units: tuple[Unit, ...]
    links: tuple[Link, ...] = ()
    sinks: tuple[str, ...] = ()

    def unit(self, name: str) -> Unit:
        for candidate in self.units:
            if candidate.name == name:
                return candidate
        raise KeyError(f"{name}: no such unit in this datapath")

    @property
    def compute_units(self) -> tuple[ComputeUnit, ...]:
        return tuple(
            unit.compute_unit
            for unit in self.units
            if unit.compute_unit is not None
        )


def _walks(datapath: Datapath) -> tuple[tuple[Unit, ...], ...]:
    """Return every path from an issuing unit to a sink."""
    outgoing = {}
    for link in datapath.links:
        outgoing.setdefault(link.source, []).append(link.destination)

    sinks = set(datapath.sinks)
    found = []

    def extend(path: tuple[Unit, ...], seen: frozenset[str]) -> None:
        last = path[-1]
        successors = outgoing.get(last.name, ())
        if last.name in sinks or not successors:
            found.append(path)
            return
        for name in successors:
            if name in seen:
                continue
            extend(path + (datapath.unit(name),), seen | {name})

    for unit in datapath.units:
        if unit.issue is not None:
            extend((unit,), frozenset({unit.name}))
    return tuple(found)


def _blocks_of(path: tuple[Unit, ...]) -> tuple[Block, ...]:
    """Return the blocks along one walk, with its issue block first."""
    blocks = []
    for position, unit in enumerate(path):
        if position == 0 and unit.issue is not None:
            blocks.append(unit.issue)
        blocks.extend(unit.blocks)
    return tuple(blocks)


def _stage(block: Block) -> OpMatcher:
    return OpMatcher(*block.ops, predicate=block.predicate)


def derive_patterns(
    datapath: Datapath | None,
) -> tuple[tuple[OpMatcher, ...], ...]:
    """Return the fusion patterns implied by a hardware datapath."""
    if datapath is None:
        return ()
    patterns = []
    for path in _walks(datapath):
        stages = tuple(_stage(block) for block in _blocks_of(path) if block.ops)
        if len(stages) > 1:
            patterns.append(stages)
    return tuple(patterns)


@dataclass(frozen=True)
class MemoryInstance:
    name: str
    bank_count: int
    depth_per_bank: int
    bits_per_bank_entry: int
    accepted_accesses: frozenset[MemoryAccess]
    access_cost: float
    logical_capacity: int | None = None
    alignment_bytes: int | None = None
    access_granularity_bits: int | None = None
    read_bandwidth: float | None = None
    write_bandwidth: float | None = None
    double_buffered: bool = False

    def __post_init__(self):
        if not self.name:
            raise ValueError("memory instance name must not be empty")
        if self.bank_count <= 0:
            raise ValueError(f"{self.name}: bank_count must be positive")
        if self.depth_per_bank <= 0:
            raise ValueError(f"{self.name}: depth_per_bank must be positive")
        if self.bits_per_bank_entry <= 0:
            raise ValueError(
                f"{self.name}: bits_per_bank_entry must be positive"
            )
        accesses = frozenset(
            (TensorRole(role), unit if unit is None else str(unit))
            for role, unit in self.accepted_accesses
        )
        if not accesses:
            raise ValueError(
                f"{self.name}: accepted_accesses must not be empty"
            )
        if any(unit == "" for _role, unit in accesses):
            raise ValueError(
                f"{self.name}: compute unit names must not be empty"
            )
        object.__setattr__(self, "accepted_accesses", accesses)
        if self.logical_capacity is not None and self.logical_capacity <= 0:
            raise ValueError(f"{self.name}: logical_capacity must be positive")
        if self.alignment_bytes is not None and self.alignment_bytes <= 0:
            raise ValueError(f"{self.name}: alignment_bytes must be positive")
        if (
            self.access_granularity_bits is not None
            and self.access_granularity_bits <= 0
        ):
            raise ValueError(
                f"{self.name}: access_granularity_bits must be positive"
            )
        if self.read_bandwidth is not None and self.read_bandwidth < 0:
            raise ValueError(
                f"{self.name}: read_bandwidth must be non-negative"
            )
        if self.write_bandwidth is not None and self.write_bandwidth < 0:
            raise ValueError(
                f"{self.name}: write_bandwidth must be non-negative"
            )
        if self.double_buffered and self.physical_capacity_bits < 2:
            raise ValueError(
                f"{self.name}: double buffering requires two storage units"
            )
        if (
            self.double_buffered
            and self.logical_capacity is not None
            and self.logical_capacity < 2
        ):
            raise ValueError(
                f"{self.name}: double buffering requires two logical units"
            )

    @property
    def physical_capacity_bits(self) -> int:
        return self.bank_count * self.depth_per_bank * self.bits_per_bank_entry

    @property
    def physical_capacity_bytes(self) -> int:
        return self.physical_capacity_bits // 8

    @property
    def capacity_bits(self) -> int:
        """Return storage available to one resident tile."""
        divisor = 2 if self.double_buffered else 1
        return self.physical_capacity_bits // divisor

    @property
    def capacity_bytes(self) -> int:
        return self.capacity_bits // 8

    @property
    def bank_capacity_bytes(self) -> int:
        """Return one bank's storage available to a resident tile."""
        divisor = 2 if self.double_buffered else 1
        return self.depth_per_bank * self.bits_per_bank_entry // 8 // divisor

    @property
    def usable_logical_capacity(self) -> int | None:
        """Return logical storage available to one resident tile."""
        if self.logical_capacity is None:
            return None
        divisor = 2 if self.double_buffered else 1
        return self.logical_capacity // divisor

    @property
    def allocation_alignment_bytes(self) -> int:
        """Return the minimum legal alignment for tensor base addresses."""
        entry_bytes = ceil(self.bits_per_bank_entry / 8)
        access_bytes = ceil((self.access_granularity_bits or 8) / 8)
        return lcm(
            self.alignment_bytes or 1,
            entry_bytes,
            access_bytes,
        )

    @property
    def accepted_roles(self) -> frozenset[TensorRole]:
        """Return the tensor-role projection of accepted accesses."""
        return frozenset(role for role, _unit in self.accepted_accesses)

    def accepts(
        self,
        role: TensorRole,
        compute_units: tuple[str, ...] | frozenset[str] = (),
    ) -> bool:
        """Return whether every accessing compute unit can use this role."""
        role = TensorRole(role)
        accesses = self.accepted_accesses
        if (role, None) in accesses:
            return True
        if not compute_units:
            return role in self.accepted_roles
        return all((role, unit) in accesses for unit in compute_units)

    def accepts_accesses(
        self,
        requirements,
    ) -> bool:
        """Return whether every exact ``(role, unit)`` access is legal."""
        return all(
            self.accepts(role, () if unit is None else (unit,))
            for role, unit in requirements
        )


@dataclass(frozen=True)
class MemoryLevelConfig:
    name: str
    instances: tuple[MemoryInstance, ...]
    input_memory: str
    output_memory: str
    weight_memory: str

    def __post_init__(self):
        if not self.instances:
            raise ValueError(f"{self.name}: instances must not be empty")
        names = [memory.name for memory in self.instances]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: memory names must be unique")

        bindings = {
            TensorRole.ACTIVATION: self.input_memory,
            TensorRole.OUTPUT: self.output_memory,
            TensorRole.WEIGHT: self.weight_memory,
        }
        by_name = {memory.name: memory for memory in self.instances}
        for role, memory_name in bindings.items():
            if memory_name not in by_name:
                raise ValueError(f"{self.name}: {memory_name} does not exist")
            memory = by_name[memory_name]
            accepted_roles = memory.accepted_roles
            role_is_output = role == TensorRole.OUTPUT
            accepts_output = (
                TensorRole.OUTPUT in accepted_roles
                or TensorRole.PARTIAL_SUM in accepted_roles
            )
            if role not in accepted_roles and not (
                role_is_output and accepts_output
            ):
                raise ValueError(
                    f"{self.name}: {memory_name} cannot store {role.value}"
                )

    def memory_index(self, name: str) -> int:
        for index, memory in enumerate(self.instances):
            if memory.name == name:
                return index
        raise ValueError(f"{self.name}: unknown memory {name}")

    def memory(self, name: str) -> MemoryInstance:
        return self.instances[self.memory_index(name)]

    def memory_name_for_role(self, role: TensorRole) -> str:
        if role == TensorRole.ACTIVATION:
            return self.input_memory
        if role in (TensorRole.OUTPUT, TensorRole.PARTIAL_SUM):
            return self.output_memory
        if role in (TensorRole.WEIGHT, TensorRole.BIAS):
            return self.weight_memory
        candidates = [
            memory.name
            for memory in self.instances
            if role in memory.accepted_roles
        ]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(f"{self.name}: no memory accepts {role.value}")
        raise ValueError(
            f"{self.name}: {role.value} has ambiguous memories " f"{candidates}"
        )

    def capacity_bytes_for_role(self, role: TensorRole) -> int:
        return self.memory(self.memory_name_for_role(role)).capacity_bytes

    def fits_role_bytes(self, role_bytes: dict[TensorRole, int]) -> bool:
        """Check capacities while preserving shared physical instances."""
        bytes_by_memory = {memory.name: 0 for memory in self.instances}
        for role, size in role_bytes.items():
            if size < 0:
                raise ValueError("role byte usage must be non-negative")
            memory_name = self.memory_name_for_role(role)
            bytes_by_memory[memory_name] += size
        return all(
            bytes_by_memory[memory.name] <= memory.capacity_bytes
            for memory in self.instances
        )

    def fits_segments(self, segments) -> bool:
        """Check banked placements against their physical instances."""
        for segment in segments:
            partition = segment.memory_space
            if partition is None or not 0 <= partition < len(self.instances):
                return False
            if segment.start < 0:
                return False
            if segment.end > self.instances[partition].capacity_bytes:
                return False
        return True

    @property
    def operand_partitions(self) -> list[int]:
        return [
            self.memory_index(self.input_memory),
            self.memory_index(self.output_memory),
            self.memory_index(self.weight_memory),
        ]


@dataclass(frozen=True)
class CollectorConfig:
    name: str
    depth_words: int
    lanes_per_word: int
    accumulator_bits: int

    def __post_init__(self):
        if self.depth_words <= 0:
            raise ValueError(f"{self.name}: depth_words must be positive")
        if self.lanes_per_word <= 0:
            raise ValueError(f"{self.name}: lanes_per_word must be positive")
        if self.accumulator_bits <= 0:
            raise ValueError(f"{self.name}: accumulator_bits must be positive")


@dataclass(frozen=True)
class HardwareConfig:
    pe: ComputeUnit
    local_memory: MemoryLevelConfig
    global_memory: MemoryLevelConfig
    output_collector: CollectorConfig
    external_memory: MemoryLevelConfig | None = None
    dtype_modes: tuple[DTypeMode, ...] = ()
    compute_units: tuple[ComputeUnit, ...] = ()
    datapath: Datapath | None = None
    hierarchy_bandwidths: tuple[HierarchyBandwidth, ...] = ()
    target: str = DEFAULT_COMPILATION_TARGET
    mapping_num_banks: int | None = None
    mapping_bank_width: int | None = None
    technology_nm: int | None = None
    pad_depthwise_channels: bool = True
    partial_matrix_input_channels: frozenset[int] = frozenset()

    def __post_init__(self):
        partial_matrix_input_channels = frozenset(
            self.partial_matrix_input_channels
        )
        if any(channels <= 0 for channels in partial_matrix_input_channels):
            raise ValueError("partial_matrix_input_channels must be positive")
        object.__setattr__(
            self,
            "partial_matrix_input_channels",
            partial_matrix_input_channels,
        )
        compute_units = tuple(self.compute_units) or (self.pe,)
        names = [unit.name for unit in compute_units]
        if len(set(names)) != len(names):
            raise ValueError("compute unit names must be unique")
        matching = [unit for unit in compute_units if unit.name == self.pe.name]
        if matching != [self.pe]:
            raise ValueError(
                "pe must be the matching instance in compute_units"
            )
        object.__setattr__(self, "compute_units", compute_units)
        if self.datapath is not None:
            compute_units_by_name = {unit.name: unit for unit in compute_units}
            for unit in self.datapath.units:
                if unit.issue is None:
                    continue
                if unit.compute_unit is None:
                    raise ValueError(
                        f"datapath issuing unit {unit.name} has no "
                        "compute_unit"
                    )
                declared = compute_units_by_name.get(unit.compute_unit.name)
                if declared is None:
                    raise ValueError(
                        f"datapath issuing unit {unit.name} references "
                        f"unknown compute unit {unit.compute_unit.name}"
                    )
                if declared != unit.compute_unit:
                    raise ValueError(
                        f"datapath issuing unit {unit.name} disagrees with "
                        f"compute unit {unit.compute_unit.name}"
                    )
        compute_unit_names = frozenset(names)
        bindings = self.compute_unit_bindings()
        if any(not role for role in bindings):
            raise ValueError("compute unit binding roles must not be empty")
        unknown_bindings = {
            unit for unit in bindings.values() if unit not in compute_unit_names
        }
        if unknown_bindings:
            raise ValueError(
                "compute unit bindings reference unknown units "
                f"{sorted(unknown_bindings)}"
            )
        for level in self.memory_levels:
            for memory in level.instances:
                unknown = {
                    unit
                    for _role, unit in memory.accepted_accesses
                    if unit is not None and unit not in compute_unit_names
                }
                if unknown:
                    raise ValueError(
                        f"{memory.name}: unknown compute units "
                        f"{sorted(unknown)}"
                    )
        hierarchy_bandwidths = tuple(self.hierarchy_bandwidths)
        route_names = [route.name for route in hierarchy_bandwidths]
        if len(set(route_names)) != len(route_names):
            raise ValueError("bandwidth route names must be unique")
        resource_names = {
            self.local_memory.name,
            self.global_memory.name,
            *(unit.name for unit in compute_units),
        }
        if self.external_memory is not None:
            resource_names.add(self.external_memory.name)
        routed_roles = set()
        for route in hierarchy_bandwidths:
            unknown = {route.source, route.destination} - resource_names
            if unknown:
                raise ValueError(
                    f"{route.name}: unknown resources {sorted(unknown)}"
                )
            for role in route.roles:
                key = (route.source, route.destination, role)
                if key in routed_roles:
                    raise ValueError(
                        "bandwidth routes must uniquely bind each source, "
                        "destination, and role"
                    )
                routed_roles.add(key)
        object.__setattr__(
            self,
            "hierarchy_bandwidths",
            hierarchy_bandwidths,
        )
        widths = [mode.operand_bits for mode in self.dtype_modes]
        if len(set(widths)) != len(widths):
            raise ValueError("dtype mode widths must be unique")
        for mode in self.dtype_modes:
            output_lanes = self.output_lanes(mode.operand_bits)
            alignment = self.matrix_output_alignment(mode.operand_bits)
            if output_lanes % alignment:
                raise ValueError(
                    "output_channel_alignment must divide the physical "
                    f"output lanes for {mode.operand_bits}-bit operands"
                )
        if self.output_collector.lanes_per_word != self.pe.output_lanes:
            raise ValueError(
                "output collector lanes must equal PE output lanes"
            )
        if self.mapping_num_banks is not None and self.mapping_num_banks <= 0:
            raise ValueError("mapping_num_banks must be positive")
        if self.mapping_bank_width is not None and self.mapping_bank_width <= 0:
            raise ValueError("mapping_bank_width must be positive")
        if self.technology_nm is not None and self.technology_nm <= 0:
            raise ValueError("technology_nm must be positive")

    def dtype_mode(self, operand_bits: int) -> DTypeMode:
        for mode in self.dtype_modes:
            if mode.operand_bits == operand_bits:
                return mode
        return DTypeMode(operand_bits=operand_bits)

    @property
    def memory_levels(self) -> tuple[MemoryLevelConfig, ...]:
        """Return configured memories from compute-local to outermost."""
        levels = (self.local_memory, self.global_memory)
        if self.external_memory is not None:
            levels += (self.external_memory,)
        return levels

    def memory_is_double_buffered(
        self,
        level: MemoryLevelConfig,
        role: TensorRole,
    ) -> bool:
        """Return whether a role's memory uses ping-pong storage."""
        name = level.memory_name_for_role(role)
        return level.memory(name).double_buffered

    def input_lanes(self, operand_bits: int) -> int:
        self.dtype_mode(operand_bits)
        return self.pe.unroll_extent("IC", operand_bits)

    def output_lanes(self, operand_bits: int | None = None) -> int:
        return self.pe.unroll_extent("OC", operand_bits)

    def matrix_output_alignment(self, operand_bits: int) -> int:
        """Return the software-resident output-channel alignment."""
        mode = self.dtype_mode(operand_bits)
        return (
            mode.output_channel_alignment
            if mode.output_channel_alignment is not None
            else self.output_lanes(operand_bits)
        )

    def unroll_dims_for(self, operand_bits: int) -> tuple[int, int]:
        return self.input_lanes(operand_bits), self.output_lanes(operand_bits)

    def supports_partial_matrix_input(self, channels: int) -> bool:
        """Return whether matrix input lanes may be masked for this extent."""
        return channels in self.partial_matrix_input_channels

    def matrix_input_resident_channels(
        self,
        channels: int,
        operand_bits: int,
    ) -> int:
        """Return the physically resident channels for a matrix input."""
        self.dtype_mode(operand_bits)
        return channels

    def matrix_weight_physical_channels(
        self,
        resident_channels: int,
        operand_bits: int,
    ) -> int:
        """Return the physical input-channel stride of matrix weights."""
        self.dtype_mode(operand_bits)
        return resident_channels

    def supports_partial_matrix_output(
        self,
        channels: int,
        operand_bits: int,
    ) -> bool:
        """Return whether one compact resident K vector may mask lanes."""
        output_lanes = self.output_lanes(operand_bits)
        alignment = self.matrix_output_alignment(operand_bits)
        return 0 < channels < output_lanes and channels % alignment == 0

    def matrix_output_resident_channels(
        self,
        channels: int,
        operand_bits: int,
    ) -> int:
        """Return the legal resident K extent after software padding."""
        if channels <= 0:
            raise ValueError("matrix output channels must be positive")
        output_lanes = self.output_lanes(operand_bits)
        alignment = self.matrix_output_alignment(operand_bits)
        aligned = ceil(channels / alignment) * alignment
        if aligned % output_lanes == 0 or self.supports_partial_matrix_output(
            aligned,
            operand_bits,
        ):
            return aligned
        return ceil(aligned / output_lanes) * output_lanes

    @property
    def unroll_dims(self) -> tuple[int, int]:
        return self.pe.input_lanes, self.pe.output_lanes

    def compute_unit(self, name: str) -> ComputeUnit:
        for unit in self.compute_units:
            if unit.name == name:
                return unit
        raise ValueError(f"unknown compute unit {name}")

    def compute_unit_bindings(self) -> dict[str, str]:
        """Bind generic execution roles to configured compute units."""
        return {"matrix": self.pe.name}

    def hierarchy_bandwidth(
        self,
        source: str,
        destination: str,
        role: TensorRole,
    ) -> HierarchyBandwidth:
        """Resolve one unambiguous directional bandwidth route."""
        role = TensorRole(role)
        for route in self.hierarchy_bandwidths:
            if (
                route.source == source
                and route.destination == destination
                and role in route.roles
            ):
                return route
        raise ValueError(
            f"No bandwidth route from {source} to {destination} for "
            f"{role.value}"
        )

    def estimate_transfer_cycles(
        self,
        transfers: tuple[
            tuple[str, str, TensorRole | str, int],
            ...,
        ],
    ) -> float:
        """Estimate cycles with overlap across independent interfaces.

        Each transfer is ``(source, destination, role, bits)``. Transfers on
        one route, or on routes in the same shared group, serialize. Distinct
        groups are assumed to overlap, so the slowest group sets the time.
        """
        cycles_by_group = {}
        for source, destination, role, transferred_bits in transfers:
            route = self.hierarchy_bandwidth(
                source,
                destination,
                role,
            )
            group = route.shared_group or f"route:{route.name}"
            cycles_by_group[group] = cycles_by_group.get(group, 0.0) + (
                route.cycles_for(transferred_bits)
            )
        return max(cycles_by_group.values(), default=0.0)

    def estimate_runtime_cycles(
        self,
        compute_cycles: float | int,
        transfers: tuple[
            tuple[str, str, TensorRole | str, int],
            ...,
        ],
    ) -> float:
        """Apply the configured bandwidth roofline to compute cycles."""
        transfer_cycles = (
            self.estimate_transfer_cycles(transfers)
            if self.hierarchy_bandwidths
            else 0.0
        )
        return max(
            float(compute_cycles),
            float(ceil(transfer_cycles)),
        )

    @property
    def global_capacity_bytes(self) -> int:
        return sum(
            memory.capacity_bytes for memory in self.global_memory.instances
        )

    @property
    def global_tile_capacity_bytes(self) -> int:
        """Return the legacy scalar budget only for one shared memory."""
        bound_names = {
            self.global_memory.input_memory,
            self.global_memory.output_memory,
            self.global_memory.weight_memory,
        }
        if len(bound_names) != 1:
            raise ValueError(
                "The global hierarchy has separate operand memories; use "
                "global_memory role-aware capacity checks"
            )
        return self.global_memory.memory(bound_names.pop()).capacity_bytes

    @property
    def local_input_depth(self) -> int:
        index = self.local_memory.memory_index(self.local_memory.input_memory)
        return self.local_memory.instances[index].depth_per_bank

    @property
    def global_num_banks(self) -> int | None:
        return self.mapping_num_banks

    @property
    def memory_alignment_bytes(self) -> int | None:
        return self.mapping_bank_width

    @classmethod
    def _legacy_compute_units(
        cls,
        matrix: ComputeUnit,
    ) -> tuple[ComputeUnit, ...]:
        """Return compute resources required by a legacy matrix profile."""
        return (matrix,)

    @classmethod
    def from_legacy(
        cls,
        *,
        unroll_dims,
        cache_size: int,
        input_buffer_size: int,
        weight_buffer_size: int,
        accum_buffer_size: int,
        num_banks: int | None = None,
        bank_width: int | None = None,
        double_buffered_accum_buffer: bool = False,
        double_buffered_l2: bool = False,
        dram_size: int | None = None,
        dram_bandwidth: float = DEFAULT_DRAM_BANDWIDTH_GBPS,
        frequency: float = DEFAULT_ACCELERATOR_FREQUENCY_GHZ,
        target: str = DEFAULT_COMPILATION_TARGET,
    ):
        """Convert the scalar CLI hardware options into a typed profile."""
        if isinstance(unroll_dims, int):
            unroll_dims = (unroll_dims, unroll_dims)
        if unroll_dims is None or len(unroll_dims) != 2:
            raise ValueError("unroll_dims must contain input and output lanes")
        if cache_size is None or cache_size <= 0:
            raise ValueError("cache_size must be positive")

        input_lanes, output_lanes = unroll_dims
        global_banks = num_banks or 1
        entry_bytes = bank_width or 1
        global_word_bytes = global_banks * entry_bytes
        if cache_size % global_word_bytes != 0:
            raise ValueError(
                "cache_size must be divisible by num_banks * bank_width"
            )
        local_memory = MemoryLevelConfig(
            name="pe_local",
            instances=(
                MemoryInstance(
                    "input_buffer",
                    1,
                    input_buffer_size,
                    input_lanes,
                    frozenset({(TensorRole.ACTIVATION, None)}),
                    10,
                    logical_capacity=input_buffer_size * input_lanes,
                ),
                MemoryInstance(
                    "accum_buffer",
                    1,
                    accum_buffer_size,
                    output_lanes,
                    frozenset(
                        {
                            (TensorRole.OUTPUT, None),
                            (TensorRole.PARTIAL_SUM, None),
                        }
                    ),
                    10,
                    logical_capacity=accum_buffer_size * output_lanes,
                    double_buffered=double_buffered_accum_buffer,
                ),
                MemoryInstance(
                    "weight_buffer",
                    1,
                    weight_buffer_size,
                    output_lanes,
                    frozenset({(TensorRole.WEIGHT, None)}),
                    10,
                    logical_capacity=weight_buffer_size * output_lanes,
                ),
            ),
            input_memory="input_buffer",
            output_memory="accum_buffer",
            weight_memory="weight_buffer",
        )
        shared_accesses = frozenset(
            {
                (TensorRole.ACTIVATION, None),
                (TensorRole.OUTPUT, None),
                (TensorRole.WEIGHT, None),
                (TensorRole.SCALE, None),
                (TensorRole.BIAS, None),
            }
        )
        global_memory = MemoryLevelConfig(
            name="scratchpad",
            instances=(
                MemoryInstance(
                    "shared_cache",
                    global_banks,
                    cache_size // global_word_bytes,
                    entry_bytes * 8,
                    shared_accesses,
                    100,
                    logical_capacity=cache_size,
                    alignment_bytes=bank_width,
                    double_buffered=double_buffered_l2,
                ),
            ),
            input_memory="shared_cache",
            output_memory="shared_cache",
            weight_memory="shared_cache",
        )
        matrix = PEConfig(
            input_lanes=input_lanes,
            output_lanes=output_lanes,
        )
        external_memory = None
        hierarchy_bandwidths = ()
        if dram_size is not None:
            if dram_size <= 0:
                raise ValueError("dram_size must be positive")
            if dram_bandwidth <= 0:
                raise ValueError("dram_bandwidth must be positive")
            if frequency <= 0:
                raise ValueError("frequency must be positive")
            external_memory = MemoryLevelConfig(
                name="dram",
                instances=(
                    MemoryInstance(
                        "dram",
                        1,
                        dram_size,
                        8,
                        shared_accesses,
                        1000,
                    ),
                ),
                input_memory="dram",
                output_memory="dram",
                weight_memory="dram",
            )
            bandwidth_bits_per_cycle = dram_bandwidth * 8 / frequency
            hierarchy_bandwidths = (
                HierarchyBandwidth(
                    name="dram_to_scratchpad",
                    source=external_memory.name,
                    destination=global_memory.name,
                    bandwidth_bits_per_cycle=bandwidth_bits_per_cycle,
                    roles=frozenset(
                        {
                            TensorRole.ACTIVATION,
                            TensorRole.WEIGHT,
                        }
                    ),
                    shared_group="dram",
                ),
                HierarchyBandwidth(
                    name="scratchpad_to_dram",
                    source=global_memory.name,
                    destination=external_memory.name,
                    bandwidth_bits_per_cycle=bandwidth_bits_per_cycle,
                    roles=frozenset({TensorRole.OUTPUT}),
                    shared_group="dram",
                ),
            )
        return cls(
            pe=matrix,
            compute_units=cls._legacy_compute_units(matrix),
            local_memory=local_memory,
            global_memory=global_memory,
            external_memory=external_memory,
            hierarchy_bandwidths=hierarchy_bandwidths,
            output_collector=CollectorConfig(
                name="accum_buffer",
                depth_words=accum_buffer_size,
                lanes_per_word=output_lanes,
                accumulator_bits=32,
            ),
            target=target,
            mapping_num_banks=num_banks,
            mapping_bank_width=bank_width,
        )


def resolve_unroll_dims(
    input_node,
    unroll_dims=None,
    hardware_config: HardwareConfig | None = None,
) -> tuple[int, int]:
    """Resolve the effective IC0/OC0 pair for one matrix operation."""
    if hardware_config is not None:
        if unroll_dims is not None:
            raise ValueError(
                "hardware_config cannot be combined with fixed unroll_dims"
            )
        operand_bits = get_node_dtype_width(input_node)
        return hardware_config.unroll_dims_for(operand_bits)

    if isinstance(unroll_dims, int):
        return unroll_dims, unroll_dims
    if unroll_dims is None or len(unroll_dims) != 2:
        raise ValueError("unroll_dims must contain input and output lanes")
    return tuple(unroll_dims)


__all__ = [
    "Block",
    "COMMON_IMAGE_INPUT_CHANNELS",
    "CollectorConfig",
    "ComputeUnit",
    "DEFAULT_ACCELERATOR_FREQUENCY_GHZ",
    "DEFAULT_DRAM_BANDWIDTH_GBPS",
    "DTypeMode",
    "Datapath",
    "derive_patterns",
    "get_dtype_width",
    "get_node_dtype_width",
    "HardwareConfig",
    "HierarchyBandwidth",
    "MemoryAccess",
    "MemoryInstance",
    "MemoryLevelConfig",
    "OpMatcher",
    "PEConfig",
    "Link",
    "resolve_unroll_dims",
    "get_op_overload",
    "TensorRole",
    "Unit",
    "UnrollDimension",
]

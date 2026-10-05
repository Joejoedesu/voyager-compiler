"""Typed hardware IR and the named Voyager design.

The graph (compute units, memory levels/instances, and connections) owns the
hardware specifications. The historical flat properties are read-only views
for the Voyager backend, not a second configuration. Physical units are
explicit: clock GHz, link GB/s or bytes/elements per cycle, latency ns, energy
pJ/bit. Historical DRAM capacity in 'GB' means GiB, as before.
"""

import math
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Optional, Tuple


class StorageTarget(str, Enum):
    ACTIVATION = "activation"
    WEIGHT = "weight"
    PSUM = "psum"
    SCALE = "scale"
    INDEX = "index"


class CapacityUnit(str, Enum):
    BYTES = "bytes"
    ELEMENTS = "elements"
    ROWS = "rows"


class BandwidthUnit(str, Enum):
    GB_PER_SECOND = "GB/s"
    BYTES_PER_CYCLE = "bytes/cycle"
    ELEMENTS_PER_CYCLE = "elements/cycle"


@dataclass(frozen=True)
class DataType:
    """A logical format; names follow Voyager's graph dtype metadata."""

    name: str
    bits: int

    def __post_init__(self):
        if not self.name or not isinstance(self.bits, int) or self.bits <= 0:
            raise ValueError("A dtype needs a name and a positive bit width")


class AccessMode(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"
    INTERNAL = "internal"


@dataclass(frozen=True)
class ParameterProvenance:
    """Evidence for a field, addressed from the root hardware graph.

    Paths use dataclass fields and tuple member names (or numeric indices).
    Values remain in their owning fields; evidence never duplicates them.
    """

    parameter: str
    classification: str
    source: Optional[str] = None
    notes: str = ""

    def __post_init__(self):
        if not self.parameter or self.classification not in (
            "documented",
            "measured",
            "assumed",
            "derived",
        ):
            raise ValueError("Provenance needs a field path and classification")
        if (
            self.classification in ("documented", "measured")
            and not self.source
        ):
            raise ValueError("Documented/measured parameters need a source")


@dataclass(frozen=True)
class OperationTiming:
    """Optional fixed hardware quantities, in cycles and scalar operations/cycle.

    None means unknown, not zero. These are facts for a scheduler/cost adapter,
    not a model of same-mode pipelining or a shape-dependent cost formula.
    """

    startup_cycles: Optional[float] = None
    latency_cycles: Optional[float] = None
    issue_interval_cycles: Optional[float] = None
    occupancy_cycles: Optional[float] = None
    operations_per_cycle: Optional[float] = None

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Timing values must be finite and nonnegative")
        if self.operations_per_cycle == 0 or self.issue_interval_cycles == 0:
            raise ValueError("Throughput and issue interval must be positive")


@dataclass(frozen=True)
class OperandCapability:
    """One operand's exact dtype, access direction and eligible memories.

    Empty memories means unspecified; INTERNAL describes compute-internal
    values such as an inter-PE result and cannot name external stores.
    """

    name: str
    dtype: DataType
    access: AccessMode
    memories: Tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "memories", tuple(self.memories))
        if not self.name or not isinstance(self.dtype, DataType):
            raise ValueError("An operand needs a name and DataType")
        if not isinstance(self.access, AccessMode):
            raise TypeError("Operand access must be an AccessMode")
        if len(set(self.memories)) != len(self.memories) or any(
            not m for m in self.memories
        ):
            raise ValueError("Operand memory names must be unique and nonempty")
        if self.access == AccessMode.INTERNAL and self.memories:
            raise ValueError("An internal operand cannot name external memory")


@dataclass(frozen=True)
class OperationCapability:
    """An exact operation signature; alternatives are separate entries."""

    operation: str
    operands: Tuple[OperandCapability, ...]
    timing: Optional[OperationTiming] = None

    def __post_init__(self):
        object.__setattr__(self, "operands", tuple(self.operands))
        if not self.operation or not self.operands:
            raise ValueError(
                "An operation capability needs a name and operands"
            )
        if any(not isinstance(o, OperandCapability) for o in self.operands):
            raise TypeError("Operands must be OperandCapability instances")
        if len({o.name for o in self.operands}) != len(self.operands):
            raise ValueError("Operand names must be unique within a signature")
        if self.timing is not None and not isinstance(
            self.timing, OperationTiming
        ):
            raise TypeError("Operation timing must be OperationTiming")

    def matches(self, dtypes, memories=None):
        """Match complete dtype signatures and any supplied placement choices."""
        memories = memories or {}
        return (
            set(dtypes) == {o.name for o in self.operands}
            and not memories.keys() - dtypes.keys()
            and all(
                dtypes[o.name] == o.dtype
                and (
                    o.name not in memories
                    or (
                        o.access != AccessMode.INTERNAL
                        and (not o.memories or memories[o.name] in o.memories)
                    )
                )
                for o in self.operands
            )
        )


@dataclass(frozen=True)
class UnrollingRef:
    unit: str
    dimension: str


@dataclass(frozen=True)
class SpatialUnrolling:
    dimension: str
    factor: Optional[int] = None
    fallback: Optional[UnrollingRef] = None

    def __post_init__(self):
        if not self.dimension or (
            self.factor is not None
            and (not isinstance(self.factor, int) or self.factor <= 0)
        ):
            raise ValueError("Spatial unrolling requires a positive factor")
        if self.factor is None and self.fallback is None:
            raise ValueError("Spatial unrolling needs a factor or reference")


@dataclass(frozen=True)
class ComputeMode:
    """Capabilities of one mode of a physical compute unit."""

    name: str
    operations: Tuple[OperationCapability, ...]
    spatial_unrolling: Tuple[SpatialUnrolling, ...] = ()
    timing: Optional[OperationTiming] = None

    def __post_init__(self):
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(
            self, "spatial_unrolling", tuple(self.spatial_unrolling)
        )
        if not self.name or not self.operations:
            raise ValueError("A compute mode needs a name and capabilities")
        if any(not isinstance(o, OperationCapability) for o in self.operations):
            raise TypeError(
                "Mode operations must be OperationCapability instances"
            )
        dimensions = [d.dimension for d in self.spatial_unrolling]
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("Mode unrolling dimensions must be unique")
        if self.timing is not None and not isinstance(
            self.timing, OperationTiming
        ):
            raise TypeError("Mode timing must be OperationTiming")


@dataclass(frozen=True)
class ComputationUnit:
    name: str
    supported_dtypes: Tuple[DataType, ...] = ()
    supported_operations: frozenset[str] = frozenset()
    spatial_unrolling: Tuple[SpatialUnrolling, ...] = ()
    launch_overhead_cycles: float = 0
    row_overhead_cycles: float = 0
    modes: Tuple[ComputeMode, ...] = ()
    mode_concurrency: str = "exclusive"

    def __post_init__(self):
        object.__setattr__(self, "modes", tuple(self.modes))
        if self.mode_concurrency != "exclusive":
            raise ValueError(
                "Independent execution requires separate compute units"
            )
        if any(not isinstance(m, ComputeMode) for m in self.modes):
            raise TypeError("Modes must be ComputeMode instances")
        if len({m.name for m in self.modes}) != len(self.modes):
            raise ValueError("Mode names must be unique within a unit")
        if self.modes and (self.supported_dtypes or self.supported_operations):
            raise ValueError(
                "Use mode capabilities or legacy capability sets, not both"
            )
        object.__setattr__(
            self, "supported_dtypes", tuple(self.supported_dtypes)
        )
        object.__setattr__(
            self, "supported_operations", frozenset(self.supported_operations)
        )
        object.__setattr__(
            self, "spatial_unrolling", tuple(self.spatial_unrolling)
        )
        dimensions = [d.dimension for d in self.spatial_unrolling]
        if any(not isinstance(d, DataType) for d in self.supported_dtypes):
            raise TypeError("Supported dtypes must be DataType instances")
        if len({d.name for d in self.supported_dtypes}) != len(
            self.supported_dtypes
        ):
            raise ValueError("Supported dtype names must be unique")
        if any(
            not isinstance(op, str) or not op
            for op in self.supported_operations
        ):
            raise ValueError("Supported operations must be nonempty names")
        if len(set(dimensions)) != len(dimensions):
            raise ValueError(f"Duplicate unrolling dimension in {self.name}")
        if any(
            not math.isfinite(v) or v < 0
            for v in (self.launch_overhead_cycles, self.row_overhead_cycles)
        ):
            raise ValueError("Compute timing must be nonnegative")

    def mode(self, name):
        for mode in self.modes:
            if mode.name == name:
                return mode
        raise ValueError(f"Unknown mode {name!r} on {self.name}")

    @property
    def operations(self):
        if self.modes:
            return frozenset(
                o.operation for m in self.modes for o in m.operations
            )
        return self.supported_operations

    def unrolling(self, dimension, mode=None):
        if mode is not None:
            for dim in self.mode(mode).spatial_unrolling:
                if dim.dimension == dimension:
                    return dim
        return next(
            d for d in self.spatial_unrolling if d.dimension == dimension
        )

    def supports(self, operation: str, dtype: Optional[DataType] = None):
        if self.modes:
            return any(
                o.operation == operation
                and (dtype is None or any(p.dtype == dtype for p in o.operands))
                for m in self.modes
                for o in m.operations
            )
        return operation in self.supported_operations and (
            dtype is None or dtype in self.supported_dtypes
        )

    def supports_signature(self, mode, operation, dtypes, memories=None):
        return any(
            o.operation == operation and o.matches(dtypes, memories)
            for o in self.mode(mode).operations
        )

    def operation_timing(self, mode, capability):
        selected = self.mode(mode)
        if capability not in selected.operations:
            raise ValueError("Capability does not belong to the selected mode")
        return (
            capability.timing
            if capability.timing is not None
            else selected.timing
        )


@dataclass(frozen=True)
class FusionStage:
    unit: str
    operations: Tuple[str, ...]
    predicate: Optional[str] = None

    def __post_init__(self):
        object.__setattr__(self, "operations", tuple(self.operations))


@dataclass(frozen=True)
class ISAPipeline:
    """An explicit single-ISA-call fusion contract, not a physical route.

    The name identifies a capability, not an encoded opcode. Ordered stage
    alternatives retain the backend matcher's existing optional-stage rules.
    Connectivity alone must never manufacture one of these contracts.
    """

    name: str
    stages: Tuple[FusionStage, ...]

    def __post_init__(self):
        object.__setattr__(self, "stages", tuple(self.stages))
        if not self.name or not self.stages:
            raise ValueError("An ISA pipeline needs a name and stages")


@dataclass(frozen=True)
class MemorySize:
    value: Optional[float]
    unit: CapacityUnit

    def __post_init__(self):
        if not isinstance(self.unit, CapacityUnit):
            raise TypeError("Memory size needs a CapacityUnit")
        if self.value is not None and (
            not math.isfinite(self.value) or self.value <= 0
        ):
            raise ValueError("Memory capacity must be positive or unspecified")


@dataclass(frozen=True)
class MemoryPort:
    """Physical ports shared by incident connections, not scheduler resources.

    READ_WRITE ports share their count between reads and writes. Distinct read
    and write port declarations describe independent directions. Scope locates
    the copies of these ports within the memory geometry.
    """

    name: str
    access: AccessMode
    count: int = 1
    scope: str = "memory"

    def __post_init__(self):
        if not self.name or not isinstance(self.access, AccessMode):
            raise ValueError("A memory port needs a name and access direction")
        if self.access == AccessMode.INTERNAL:
            raise ValueError("Memory ports cannot have internal access")
        if not isinstance(self.count, int) or self.count <= 0:
            raise ValueError("Port count must be positive")
        if self.scope not in ("memory", "partition", "bank", "partition_bank"):
            raise ValueError("Unknown memory port scope")

    def permits(self, access):
        return self.access == AccessMode.READ_WRITE or self.access == access


@dataclass(frozen=True)
class MemoryInstance:
    """One store, optionally replicated along compute dimensions.

    Size is per replica. Banks partition that size; buffering consumes that
    capacity rather than multiplying it. A target set describes shared versus
    dedicated storage. A scale row is deliberately distinct from an element.
    """

    name: str
    size: MemorySize
    targets: frozenset[StorageTarget]
    replication: Tuple[UnrollingRef, ...] = ()
    banks: Optional[int] = None
    word_bytes: Optional[int] = None
    buffering: int = 1
    reserved_bytes: int = 0
    access_cost: float = 0
    static_cost: float = 0
    energy_pj_per_bit: float = 0
    bank_switch_cycles: float = 0
    partitions: int = 1
    row_bytes: Optional[int] = None
    allocation_alignment_bytes: int = 1
    partition_start_alignment: int = 1
    single_bank_allocation: bool = False
    ports: Tuple[MemoryPort, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "ports", tuple(self.ports))
        if any(not isinstance(p, MemoryPort) for p in self.ports):
            raise TypeError("Ports must be MemoryPort instances")
        if len({p.name for p in self.ports}) != len(self.ports):
            raise ValueError("Port names must be unique within a memory")
        for value in (
            self.partitions,
            self.allocation_alignment_bytes,
            self.partition_start_alignment,
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(
                    "Partition counts and placement alignments must be positive integers"
                )
        if self.partition_start_alignment > self.partitions:
            raise ValueError("Partition alignment exceeds the partition count")
        if not isinstance(self.single_bank_allocation, bool):
            raise TypeError("Single-bank allocation must be boolean")
        if (
            self.single_bank_allocation
            or any(p.scope in ("bank", "partition_bank") for p in self.ports)
        ) and self.banks is None:
            raise ValueError("Bank allocation/port rules require a bank count")
        object.__setattr__(self, "targets", frozenset(self.targets))
        object.__setattr__(self, "replication", tuple(self.replication))
        if not self.targets or any(
            not isinstance(t, StorageTarget) for t in self.targets
        ):
            raise ValueError(
                "Memory targets must be a nonempty StorageTarget set"
            )
        for value in (self.banks, self.word_bytes, self.buffering):
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(
                    "Bank count, word width and buffering must be positive integers"
                )
        if any(
            not math.isfinite(v) or v < 0
            for v in (
                self.reserved_bytes,
                self.access_cost,
                self.static_cost,
                self.energy_pj_per_bit,
                self.bank_switch_cycles,
            )
        ):
            raise ValueError(
                "Memory costs and reservations must be nonnegative"
            )
        if self.banks is not None and self.size.value is None:
            raise ValueError("Banking requires a specified memory capacity")
        if self.banks is not None and self.bank_size <= 0:
            raise ValueError("Each memory bank must have positive capacity")
        if self.partitions > 1 and self.size.value is not None:
            if self.size.value % (self.partitions * (self.banks or 1)):
                raise ValueError(
                    "Partition/bank geometry must divide capacity evenly"
                )
        if self.row_bytes is not None:
            if not isinstance(self.row_bytes, int) or self.row_bytes <= 0:
                raise ValueError("Row size must be a positive byte count")
            if self.size.unit != CapacityUnit.BYTES:
                raise ValueError("Row byte geometry requires capacity in bytes")
            cell_size = self.partition_bank_size
            if cell_size is not None and cell_size % self.row_bytes:
                raise ValueError("Rows must divide each partition/bank evenly")
        if self.reserved_bytes:
            if self.size.unit != CapacityUnit.BYTES or self.size.value is None:
                raise ValueError(
                    "A reservation requires a byte-addressed capacity"
                )
            if self.reserved_bytes >= self.size.value:
                raise ValueError("A reservation must leave usable memory")
            if self.bank_size and self.reserved_bytes % self.bank_size:
                raise ValueError(
                    "A reservation must be aligned to a whole bank"
                )

    @property
    def bank_size(self):
        # A logical bank spans all partitions, preserving Voyager's flat view.
        return None if self.banks is None else self.size.value // self.banks

    @property
    def partition_size(self):
        return (
            None
            if self.size.value is None
            else self.size.value / self.partitions
        )

    @property
    def partition_bank_size(self):
        size = self.partition_size
        return None if size is None else size / (self.banks or 1)

    def port(self, name):
        for port in self.ports:
            if port.name == name:
                return port
        raise ValueError(f"Unknown port {name!r} on {self.name}")

    @property
    def usable_size(self):
        if self.size.value is None:
            return None
        return self.size.value - self.reserved_bytes


@dataclass(frozen=True)
class MemoryLevel:
    name: str
    instances: Tuple[MemoryInstance, ...]
    spatial_scope: Tuple[UnrollingRef, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "instances", tuple(self.instances))
        object.__setattr__(self, "spatial_scope", tuple(self.spatial_scope))
        if not self.name or not self.instances:
            raise ValueError(
                "A memory level must contain at least one instance"
            )


@dataclass(frozen=True)
class MemoryHierarchy:
    """Levels ordered from closest to compute to outermost storage."""

    levels: Tuple[MemoryLevel, ...]

    def __post_init__(self):
        object.__setattr__(self, "levels", tuple(self.levels))
        if not self.levels or len({l.name for l in self.levels}) != len(
            self.levels
        ):
            raise ValueError(
                "Memory levels must be nonempty and uniquely named"
            )

    @property
    def instances(self):
        return tuple(m for level in self.levels for m in level.instances)


@dataclass(frozen=True)
class Bandwidth:
    """A link rate, optionally derived from a memory port or compute lanes.

    Port-derived rates use word_bytes when set, otherwise the specified lane
    rate. This preserves the historical dtype-dependent SRAM fallback without
    copying the memory port width into every incident edge.
    """

    value: Optional[float]
    unit: BandwidthUnit
    memory_port: Optional[str] = None
    unrolling: Tuple[UnrollingRef, ...] = ()
    reduction: str = "product"

    def __post_init__(self):
        object.__setattr__(self, "unrolling", tuple(self.unrolling))
        if not isinstance(self.unit, BandwidthUnit):
            raise TypeError("Bandwidth needs a BandwidthUnit")
        if self.value is not None and (
            not math.isfinite(self.value) or self.value <= 0
        ):
            raise ValueError("Bandwidth must be positive or unspecified")
        if self.value is not None and self.unrolling:
            raise ValueError(
                "Bandwidth cannot have both a constant rate and lane references"
            )
        if self.reduction not in ("product", "min"):
            raise ValueError("Unknown bandwidth unrolling reduction")

    def bytes_per_cycle(self, config, element_bits=None):
        if self.memory_port is not None:
            word = config.memory_instance(self.memory_port).word_bytes
            if word is not None:
                return word
        rate = self.value
        if self.unrolling:
            widths = [config.unroll(ref) for ref in self.unrolling]
            rate = min(widths) if self.reduction == "min" else math.prod(widths)
        if rate is None:
            return None
        if self.unit == BandwidthUnit.GB_PER_SECOND:
            return rate / config.frequency
        if self.unit == BandwidthUnit.ELEMENTS_PER_CYCLE:
            if element_bits is None or element_bits <= 0:
                raise ValueError(
                    "Element-rate bandwidth needs a positive element bit width"
                )
            return rate * element_bits / 8
        return rate


@dataclass(frozen=True)
class TransferGeometry:
    """Maximum rectangle encoded by one transfer command, in physical bytes.

    Tensor extents/dependencies belong to the program. These are interface
    capabilities, not a selected software tile or an instruction burst policy.
    """

    max_rows: int
    max_row_bytes: int

    def __post_init__(self):
        for value in (self.max_rows, self.max_row_bytes):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(
                    "Transfer dimensions must be positive integers"
                )

    def command_count(self, rows, row_bytes):
        if rows < 0 or row_bytes < 0:
            raise ValueError("Transfer extent cannot be negative")
        return math.ceil(rows / self.max_rows) * math.ceil(
            row_bytes / self.max_row_bytes
        )


@dataclass(frozen=True)
class Connection:
    name: str
    source: str
    target: str
    bandwidth: Optional[Bandwidth] = None
    bidirectional: bool = True
    latency_ns: Optional[float] = 0
    buffer_depth: int = 0
    source_port: Optional[str] = None
    target_port: Optional[str] = None
    shared_with: Optional[str] = None
    startup_ns: float = 0
    # Endpoint throughput can be lower than its share of the physical bus.
    # This limit does not consume bandwidth on behalf of other connections.
    service_bandwidth: Optional[Bandwidth] = None
    transfer_geometry: Optional[TransferGeometry] = None
    # Commands on the same controller serialize endpoint service.
    service_resource: Optional[str] = None

    def __post_init__(self):
        if self.transfer_geometry is not None and not isinstance(
            self.transfer_geometry, TransferGeometry
        ):
            raise TypeError("Transfer geometry must be TransferGeometry")
        if self.service_resource is not None and not self.service_resource:
            raise ValueError("Transfer service resource cannot be empty")
        if (self.bandwidth is None) == (self.shared_with is None):
            raise ValueError(
                "A connection needs its own bandwidth or a shared connection reference"
            )
        if self.bandwidth is not None and not isinstance(
            self.bandwidth, Bandwidth
        ):
            raise TypeError("Connection bandwidth must be Bandwidth")
        if self.service_bandwidth is not None and not isinstance(
            self.service_bandwidth, Bandwidth
        ):
            raise TypeError("Connection service bandwidth must be Bandwidth")
        if not math.isfinite(self.startup_ns) or self.startup_ns < 0:
            raise ValueError("Transfer startup must be finite and nonnegative")
        if self.latency_ns is not None and (
            not math.isfinite(self.latency_ns) or self.latency_ns < 0
        ):
            raise ValueError("Link latency must be nonnegative")
        if not isinstance(self.buffer_depth, int) or self.buffer_depth < 0:
            raise ValueError("Link buffer depth must be nonnegative")


@dataclass(frozen=True, init=False)
class AcceleratorConfig:
    name: str
    computation_units: Tuple[ComputationUnit, ...]
    memory: MemoryHierarchy
    connections: Tuple[Connection, ...]
    frequency: float
    backend: Optional[str]
    isa_pipelines: Tuple[ISAPipeline, ...]
    provenance: Tuple[ParameterProvenance, ...]

    def __init__(
        self,
        *,
        name=None,
        computation_units=None,
        memory=None,
        connections=None,
        frequency=None,
        backend=None,
        isa_pipelines=None,
        provenance=None,
        **legacy,
    ):
        graph = (computation_units, memory, connections)
        if all(part is None for part in graph):
            # Backward-compatible construction; only the named profile owns
            # the flat defaults. No legacy knob is retained in this object.
            if frequency is not None:
                legacy["frequency"] = frequency
            profile = voyager_config(**legacy)
            for field in self.__dataclass_fields__:
                object.__setattr__(self, field, getattr(profile, field))
            if name is not None:
                object.__setattr__(self, "name", name)
            if backend is not None:
                object.__setattr__(self, "backend", backend)
            if isa_pipelines is not None:
                object.__setattr__(self, "isa_pipelines", tuple(isa_pipelines))
            if provenance is not None:
                object.__setattr__(self, "provenance", tuple(provenance))
        else:
            if any(part is None for part in graph) or legacy:
                raise TypeError(
                    "Supply the complete typed graph, or Voyager's flat options, not both"
                )
            if not name or frequency is None:
                raise ValueError(
                    "A typed hardware graph needs a name and frequency"
                )
            for key, value in dict(
                name=name,
                computation_units=tuple(computation_units),
                memory=memory,
                connections=tuple(connections),
                frequency=frequency,
                backend=backend,
                isa_pipelines=tuple(isa_pipelines or ()),
                provenance=tuple(provenance or ()),
            ).items():
                object.__setattr__(self, key, value)
        self._validate()

    def _validate(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Hardware must have a nonempty name")
        if not math.isfinite(self.frequency) or self.frequency <= 0:
            raise ValueError("Clock frequency must be finite and positive")
        components = (*self.computation_units, *self.memory.instances)
        names = [c.name for c in components]
        if any(not n for n in names) or len(names) != len(set(names)):
            raise ValueError("Compute and memory instance names must be unique")
        if len({c.name for c in self.connections}) != len(self.connections):
            raise ValueError("Connection names must be unique")
        for edge in self.connections:
            if (
                edge.service_resource is not None
                and edge.service_resource not in names
            ):
                raise ValueError(
                    f"Unknown transfer service resource {edge.service_resource!r}"
                )
            if edge.source not in names or edge.target not in names:
                raise ValueError(f"Unknown endpoint on connection {edge.name}")
            if edge.source == edge.target:
                raise ValueError("A connection needs distinct endpoints")
            budget = self.bandwidth_connection(edge.name)
            if edge.shared_with is not None:
                memory_names = {m.name for m in self.memory.instances}
                if not ({edge.source, edge.target} & memory_names) or not (
                    {budget.source, budget.target} & memory_names
                ):
                    raise ValueError(
                        "Only memory/transfer connections can share bandwidth"
                    )
            for endpoint, port_name, access in (
                (edge.source, edge.source_port, AccessMode.READ),
                (edge.target, edge.target_port, AccessMode.WRITE),
            ):
                if port_name is not None:
                    port = self.memory_instance(endpoint).port(port_name)
                    required = (
                        AccessMode.READ_WRITE if edge.bidirectional else access
                    )
                    if not port.permits(required):
                        raise ValueError(
                            f"Port direction disagrees with connection {edge.name}"
                        )
            for rate in (budget.bandwidth, edge.service_bandwidth):
                if rate is not None:
                    if rate.memory_port is not None:
                        self.memory_instance(rate.memory_port)
                    for ref in rate.unrolling:
                        self.unroll(ref)
        for unit in self.computation_units:
            for dim in unit.spatial_unrolling:
                self.unroll(UnrollingRef(unit.name, dim.dimension))
            for mode in unit.modes:
                for dim in mode.spatial_unrolling:
                    if dim.factor is None:
                        self.unroll(dim.fallback)
                for capability in mode.operations:
                    for operand in capability.operands:
                        for memory in operand.memories:
                            self.memory_instance(memory)
                            if operand.access in (
                                AccessMode.READ,
                                AccessMode.READ_WRITE,
                            ) and not self.connected(memory, unit.name):
                                raise ValueError(
                                    f"No read connection for {unit.name}/{operand.name} from {memory}"
                                )
                            if operand.access in (
                                AccessMode.WRITE,
                                AccessMode.READ_WRITE,
                            ) and not self.connected(unit.name, memory):
                                raise ValueError(
                                    f"No write connection for {unit.name}/{operand.name} to {memory}"
                                )
        if len({p.name for p in self.isa_pipelines}) != len(self.isa_pipelines):
            raise ValueError("ISA pipeline names must be unique")
        for pipeline in self.isa_pipelines:
            for stage in pipeline.stages:
                supported = self.compute_unit(stage.unit).operations
                if not set(stage.operations) <= supported:
                    raise ValueError(
                        f"ISA pipeline {pipeline.name} requires unsupported operations on {stage.unit}"
                    )
        for level in self.memory.levels:
            for ref in level.spatial_scope:
                self.unroll(ref)
            for mem in level.instances:
                for ref in mem.replication:
                    self.unroll(ref)
        if any(not isinstance(p, ParameterProvenance) for p in self.provenance):
            raise TypeError(
                "Provenance must contain ParameterProvenance records"
            )
        if len({p.parameter for p in self.provenance}) != len(self.provenance):
            raise ValueError("A parameter must have a single provenance record")
        for record in self.provenance:
            self.parameter_value(record.parameter)

    def parameter_value(self, path):
        """Resolve evidence paths without evaluating code or derived properties."""
        value = self
        for part in path.split("."):
            if is_dataclass(value) and part in {f.name for f in fields(value)}:
                value = getattr(value, part)
            elif isinstance(value, tuple):
                if part.isdecimal() and int(part) < len(value):
                    value = value[int(part)]
                else:
                    matches = [
                        v for v in value if getattr(v, "name", None) == part
                    ]
                    if len(matches) != 1:
                        raise ValueError(
                            f"Unknown or ambiguous provenance path: {path}"
                        )
                    value = matches[0]
            else:
                raise ValueError(f"Unknown provenance path: {path}")
        return value

    def connected(self, source, target):
        """Direct access only; copies through other memories are separate operations."""
        return any(
            (edge.source == source and edge.target == target)
            or (
                edge.bidirectional
                and edge.source == target
                and edge.target == source
            )
            for edge in self.connections
        )

    def bandwidth_connection(self, name):
        """Canonical physical bandwidth budget, shared across all referring links.

        Both directions count against one budget. Independent directions use
        separate directed connections. No reservation/scheduling state lives here.
        """
        seen = set()
        edge = self.connection(name)
        while edge.shared_with is not None:
            if edge.name in seen:
                raise ValueError("Cyclic shared-bandwidth reference")
            seen.add(edge.name)
            edge = self.connection(edge.shared_with)
        return edge

    def connection_bytes_per_cycle(self, name, element_bits=None):
        return self.bandwidth_connection(name).bandwidth.bytes_per_cycle(
            self, element_bits
        )

    def compute_unit(self, name):
        for unit in self.computation_units:
            if unit.name == name:
                return unit
        raise ValueError(f"Unknown compute unit {name!r} in {self.name}")

    def memory_instance(self, name):
        for mem in self.memory.instances:
            if mem.name == name:
                return mem
        raise ValueError(f"Unknown memory instance {name!r} in {self.name}")

    def connection(self, name):
        for edge in self.connections:
            if edge.name == name:
                return edge
        raise ValueError(f"Unknown connection {name!r} in {self.name}")

    def unroll(self, ref, seen=frozenset()):
        if ref in seen:
            raise ValueError("Cyclic spatial-unrolling reference")
        try:
            dim = self.compute_unit(ref.unit).unrolling(ref.dimension)
        except StopIteration:
            raise ValueError(f"Unknown unrolling dimension {ref}") from None
        if dim.factor is not None:
            return dim.factor
        return self.unroll(dim.fallback, seen | {ref})

    def capacity(self, memory):
        if memory.usable_size is None:
            return None
        return memory.usable_size * math.prod(
            self.unroll(r) for r in memory.replication
        )

    def require_backend(self, backend):
        if self.backend != backend:
            raise NotImplementedError(
                f"Hardware {self.name!r} needs a {self.backend!r} lowering; "
                f"this compiler path implements {backend!r}"
            )

    # Voyager compatibility views. Values live exclusively in the graph.
    @property
    def pe_array_size(self):
        dims = self.compute_unit("matrix").spatial_unrolling
        return (
            None
            if not dims
            else (
                self.unroll(UnrollingRef("matrix", "IC")),
                self.unroll(UnrollingRef("matrix", "OC")),
            )
        )

    @property
    def vector_unit_width(self):
        return self.compute_unit("vector").unrolling("lanes").factor

    @property
    def matrix_vector_unit_width(self):
        return self.compute_unit("matrix_vector").unrolling("lanes").factor

    @property
    def accumulator_width(self):
        return self.compute_unit("vector").unrolling("pool_channels").factor

    @property
    def vector_lanes(self):
        return self.unroll(UnrollingRef("vector", "lanes"))

    @property
    def matrix_vector_lanes(self):
        return self.unroll(UnrollingRef("matrix_vector", "lanes"))

    @property
    def accumulator_lanes(self):
        return self.unroll(UnrollingRef("vector", "pool_channels"))

    @property
    def input_buffer_size(self):
        return self.memory_instance("input_buffer").size.value

    @property
    def weight_buffer_size(self):
        return self.memory_instance("weight_buffer").size.value

    @property
    def accum_buffer_size(self):
        return self.memory_instance("accum_buffer").size.value

    @property
    def double_buffered_accum_buffer(self):
        return self.memory_instance("accum_buffer").buffering == 2

    @property
    def scratchpad_size(self):
        return self.memory_instance("scratchpad").size.value

    @property
    def scratchpad_offset(self):
        return self.memory_instance("scratchpad").reserved_bytes

    @property
    def num_banks(self):
        return self.memory_instance("scratchpad").banks

    @property
    def bank_width(self):
        return self.memory_instance("scratchpad").word_bytes

    @property
    def double_buffered_l2(self):
        return self.num_slots == 2

    @property
    def num_slots(self):
        return self.memory_instance("scratchpad").buffering

    @property
    def bank_size(self):
        return self.memory_instance("scratchpad").bank_size

    @property
    def usable_scratchpad_size(self):
        return self.memory_instance("scratchpad").usable_size

    @property
    def usable_banks(self):
        if self.num_banks is None:
            return None
        return self.num_banks - self.scratchpad_offset // self.bank_size

    @property
    def dram_size(self):
        size = self.memory_instance("dram").size.value
        return None if size is None else size / 1024**3

    @property
    def bytes_per_cycle(self):
        return self.connection("dram_sram").bandwidth.bytes_per_cycle(self)

    @property
    def dram_bandwidth(self):
        rate_spec = self.connection("dram_sram").bandwidth
        if rate_spec.unit == BandwidthUnit.GB_PER_SECOND:
            return rate_spec.value
        rate = self.bytes_per_cycle
        return None if rate is None else rate * self.frequency

    @property
    def dram_access_latency(self):
        return self.connection("dram_sram").latency_ns

    @property
    def access_latency_cycles(self):
        return self.dram_access_latency * self.frequency

    @property
    def dram_energy_per_bit(self):
        return self.memory_instance("dram").energy_pj_per_bit

    @property
    def dram_energy_per_byte(self):
        return self.dram_energy_per_bit * 8 * 1e-12

    @property
    def bank_switch_cycles(self):
        return self.memory_instance("scratchpad").bank_switch_cycles

    @property
    def kernel_launch_overhead(self):
        return self.compute_unit("vector").launch_overhead_cycles

    @property
    def output_slack(self):
        return self.connection("matrix_vector_stream").buffer_depth

    @property
    def spmm_row_cycles(self):
        return self.compute_unit("sparse_matrix").row_overhead_cycles

    @property
    def spmm_scale_rows(self):
        return self.memory_instance("spmm_scales").size.value

    def sram_bandwidth_bits(self, element_bits):
        bits = (
            self.connection("sram_input").bandwidth.bytes_per_cycle(
                self, element_bits
            )
            * 8
        )
        if bits != int(bits):
            raise NotImplementedError(
                "Voyager's SRAM port must transfer a whole number of bits per cycle"
            )
        return int(bits)

    def compute_bandwidth(self, unit="vector"):
        return self.connection(f"sram_{unit}").bandwidth.bytes_per_cycle(self)

    @classmethod
    def from_args(cls, args):
        """Existing command-line options instantiate the Voyager profile."""
        names = (
            "pe_array_size",
            "vector_unit_width",
            "matrix_vector_unit_width",
            "accumulator_width",
            "frequency",
            "input_buffer_size",
            "weight_buffer_size",
            "accum_buffer_size",
            "double_buffered_accum_buffer",
            "scratchpad_size",
            "scratchpad_offset",
            "num_banks",
            "bank_width",
            "double_buffered_l2",
            "dram_size",
            "dram_bandwidth",
            "dram_access_latency",
            "dram_energy_per_bit",
        )
        return voyager_config(**{name: getattr(args, name) for name in names})


def voyager_config(
    *,
    pe_array_size=(32, 32),
    vector_unit_width=None,
    matrix_vector_unit_width=None,
    accumulator_width=None,
    frequency=1.0,
    input_buffer_size=1024,
    weight_buffer_size=1024,
    accum_buffer_size=1024,
    double_buffered_accum_buffer=False,
    scratchpad_size=None,
    scratchpad_offset=0,
    num_banks=None,
    bank_width=None,
    double_buffered_l2=True,
    dram_size=16.0,
    dram_bandwidth=64.0,
    dram_access_latency=100.0,
    dram_energy_per_bit=6.25,
):
    """Instantiate Voyager, the default profile. All Voyager defaults live here.

    Other accelerators construct AcceleratorConfig from their own typed graph;
    they do not inherit Voyager's topology or its measured timing constants.
    """
    ic, oc = UnrollingRef("matrix", "IC"), UnrollingRef("matrix", "OC")
    vl = UnrollingRef("vector", "lanes")
    # Null hardware is retained for transform()'s no-padding mode. Its
    # otherwise unused widths have a concrete fallback, without dangling refs.
    active = pe_array_size is not None

    def dim(name, factor, fallback):
        return SpatialUnrolling(
            name,
            factor if factor is not None or active else 1,
            fallback if active else None,
        )

    dtypes = tuple(DataType(f"int{bits}", bits) for bits in range(1, 33)) + (
        DataType("bfloat16", 16),
        DataType("float32", 32),
        DataType("fp8_e4m3", 8),
        DataType("fp8_e5m2", 8),
        DataType("fp8_e5m3", 8),
        DataType("posit8_1", 8),
        DataType("lut4_to_int6", 4),
    )
    matrix_ops = frozenset(
        ("conv2d", "linear", "matmul", "conv2d_mx", "linear_mx", "matmul_mx")
    )
    vector_ops = frozenset(
        (
            "dequantize",
            "add",
            "sub",
            "mul",
            "div",
            "exp",
            "abs",
            "relu",
            "gelu",
            "sigmoid",
            "silu",
            "tanh",
            "hardtanh",
            "layer_norm",
            "softmax",
            "quantize",
            "quantize_mx",
            "quantize_mx_outlier",
            "quantize_affine",
            "max_pool2d",
            "avg_pool2d",
            "adaptive_avg_pool2d",
            "linear",
            "matmul",
        )
    )
    units = (
        ComputationUnit(
            "matrix",
            dtypes,
            matrix_ops,
            (
                (
                    SpatialUnrolling("IC", pe_array_size[0]),
                    SpatialUnrolling("OC", pe_array_size[1]),
                )
                if active
                else ()
            ),
        ),
        ComputationUnit(
            "vector",
            dtypes,
            vector_ops,
            (
                dim("lanes", vector_unit_width, oc),
                dim("pool_channels", accumulator_width, vl),
            ),
            # Sphinx measurement: issue and fill/drain per normalization pass.
            launch_overhead_cycles=96,
        ),
        ComputationUnit(
            "matrix_vector",
            dtypes,
            frozenset(("linear", "matmul", "linear_mx", "matmul_mx")),
            (dim("lanes", matrix_vector_unit_width, oc),),
        ),
        ComputationUnit(
            "sparse_matrix",
            dtypes,
            frozenset(("spmm",)),
            (
                (
                    SpatialUnrolling("IC", fallback=ic),
                    SpatialUnrolling("OC", fallback=oc),
                )
                if active
                else ()
            ),
            # Sparse accumulator ring drain/restart (measured 7.3–7.5 cycles).
            row_overhead_cycles=8,
        ),
    )
    activation, psum, weight = (
        StorageTarget.ACTIVATION,
        StorageTarget.PSUM,
        StorageTarget.WEIGHT,
    )
    shared = frozenset(StorageTarget)
    elements, byte, rows = (
        CapacityUnit.ELEMENTS,
        CapacityUnit.BYTES,
        CapacityUnit.ROWS,
    )
    scope = (ic, oc) if active else ()
    levels = (
        MemoryLevel(
            "PE",
            tuple(
                MemoryInstance(
                    f"pe_{role.value}",
                    MemorySize(1, elements),
                    frozenset((role,)),
                    access_cost=1,
                )
                for role in (activation, psum, weight)
            ),
            scope,
        ),
        MemoryLevel(
            "L1",
            (
                MemoryInstance(
                    "input_buffer",
                    MemorySize(input_buffer_size, elements),
                    frozenset((activation,)),
                    (ic,) if active else (),
                    access_cost=10,
                ),
                MemoryInstance(
                    "accum_buffer",
                    MemorySize(accum_buffer_size, elements),
                    frozenset((psum,)),
                    (oc,) if active else (),
                    buffering=2 if double_buffered_accum_buffer else 1,
                    access_cost=10,
                ),
                MemoryInstance(
                    "weight_buffer",
                    MemorySize(weight_buffer_size, elements),
                    frozenset((weight,)),
                    (oc,) if active else (),
                    access_cost=10,
                ),
                MemoryInstance(
                    "spmm_scales",
                    MemorySize(32, rows),  # SpMMUnit.h's DoubleBuffer<32>
                    frozenset((StorageTarget.SCALE,)),
                    buffering=2,
                ),
            ),
        ),
        MemoryLevel(
            "L2",
            (
                MemoryInstance(
                    "scratchpad",
                    MemorySize(scratchpad_size, byte),
                    shared,
                    banks=num_banks,
                    word_bytes=bank_width,
                    buffering=2 if double_buffered_l2 else 1,
                    reserved_bytes=scratchpad_offset,
                    access_cost=100,
                    bank_switch_cycles=8,  # ordered read-master round trip
                ),
            ),
        ),
        MemoryLevel(
            "DRAM",
            (
                MemoryInstance(
                    "dram",
                    MemorySize(
                        None if dram_size is None else dram_size * 1024**3, byte
                    ),
                    shared,
                    access_cost=1000,
                    energy_pj_per_bit=dram_energy_per_bit,
                ),
            ),
        ),
    )
    sram_rate = Bandwidth(
        None if active else 1,
        BandwidthUnit.ELEMENTS_PER_CYCLE,
        memory_port="scratchpad",
        unrolling=scope,
        reduction="min",
    )
    external_rate = Bandwidth(dram_bandwidth, BandwidthUnit.GB_PER_SECOND)
    register_rate = Bandwidth(1, BandwidthUnit.ELEMENTS_PER_CYCLE)
    edges = [
        Connection(
            "dram_sram",
            "dram",
            "scratchpad",
            external_rate,
            latency_ns=dram_access_latency,
        ),
        Connection("sram_vector", "scratchpad", "vector", external_rate),
        Connection(
            "sram_matrix_vector", "scratchpad", "matrix_vector", external_rate
        ),
        Connection("sram_sparse", "scratchpad", "sparse_matrix", sram_rate),
        Connection(
            "matrix_vector_stream",
            "matrix",
            "vector",
            Bandwidth(
                None,
                BandwidthUnit.ELEMENTS_PER_CYCLE,
                unrolling=(oc,) if active else (vl,),
            ),
            bidirectional=False,
            # Matrix FIFO, vector input FIFO and intermediate pipeline stages;
            # the existing model uses 24 vectors for the measured 20–40 range.
            buffer_depth=24,
        ),
        Connection("sram_scales", "scratchpad", "spmm_scales", sram_rate),
        Connection(
            "scales_sparse", "spmm_scales", "sparse_matrix", register_rate
        ),
    ]
    for key, memory, role in (
        ("input", "input_buffer", activation),
        ("accum", "accum_buffer", psum),
        ("weight", "weight_buffer", weight),
    ):
        reg = f"pe_{role.value}"
        edges.extend(
            (
                Connection(f"sram_{key}", "scratchpad", memory, sram_rate),
                Connection(f"{key}_pe", memory, reg, register_rate),
                Connection(f"{key}_matrix", reg, "matrix", register_rate),
            )
        )
    quant = (
        "quantize",
        "quantize_mx",
        "quantize_mx_outlier",
        "quantize_affine",
    )
    gemm_quant = tuple(op for op in quant if op != "quantize_mx_outlier")
    anchor = FusionStage(
        "matrix",
        ("conv2d", "linear", "matmul", "conv2d_mx", "linear_mx", "matmul_mx"),
        "matrix_tail",
    )
    dequant = FusionStage("vector", ("dequantize",))
    pipelines = (
        (
            anchor,
            dequant,
            FusionStage(
                "vector", ("add", "sub", "mul", "div"), "constant_divisor"
            ),
            FusionStage("vector", ("exp", "abs", "relu")),
            FusionStage("vector", ("add", "mul", "div"), "constant_divisor"),
            FusionStage("vector", gemm_quant),
        ),
        (
            anchor,
            dequant,
            FusionStage(
                "vector", ("gelu", "sigmoid", "silu", "tanh", "hardtanh")
            ),
            FusionStage("vector", gemm_quant),
        ),
        (
            FusionStage("vector", ("layer_norm", "softmax")),
            FusionStage("vector", quant),
        ),
    )
    return AcceleratorConfig(
        name="voyager",
        backend="voyager",
        frequency=frequency,
        computation_units=units,
        memory=MemoryHierarchy(levels),
        connections=tuple(edges),
        isa_pipelines=tuple(
            ISAPipeline(name, stages)
            for name, stages in zip(
                (
                    "matrix_elementwise",
                    "matrix_activation",
                    "reduction_quantize",
                ),
                pipelines,
            )
        ),
    )


# Compatibility exports for the CLI. These are views of the named instance,
# so changing a profile default changes the CLI and library together.
VOYAGER = voyager_config()
DEFAULT_PE_ARRAY_SIZE = VOYAGER.pe_array_size
DEFAULT_FREQUENCY_GHZ = VOYAGER.frequency
DEFAULT_INPUT_BUFFER_SIZE = VOYAGER.input_buffer_size
DEFAULT_WEIGHT_BUFFER_SIZE = VOYAGER.weight_buffer_size
DEFAULT_ACCUM_BUFFER_SIZE = VOYAGER.accum_buffer_size
DEFAULT_SCRATCHPAD_OFFSET = VOYAGER.scratchpad_offset
DEFAULT_DOUBLE_BUFFERED_L2 = VOYAGER.double_buffered_l2
DEFAULT_DRAM_SIZE_GB = VOYAGER.dram_size
DEFAULT_DRAM_BANDWIDTH_GBS = VOYAGER.dram_bandwidth
DEFAULT_DRAM_ACCESS_LATENCY_NS = VOYAGER.dram_access_latency
DEFAULT_DRAM_ENERGY_PJ_PER_BIT = VOYAGER.dram_energy_per_bit

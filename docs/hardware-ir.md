# Hardware IR

`AcceleratorConfig` owns an immutable hardware graph. `voyager_config()` creates
the named `voyager` instance; `VOYAGER` is that profile with its default values.
Voyager's constants belong to this factory, not to the generic IR types or to
the tiling and allocation algorithms. A future Gemmini or Trainium profile can
construct these types for its physical topology, compute modes, operand
capabilities, memory geometry, ports, and shared transfer bandwidth. Target
adapters still own scheduling and allocation decisions.

The existing keyword constructor and command-line arguments remain supported:

```python
from voyager_compiler.hardware_config import AcceleratorConfig, voyager_config

hardware = voyager_config(
    pe_array_size=(64, 64),
    scratchpad_size=2 * 1024 * 1024,
    num_banks=16,
    bank_width=32,
)
# Equivalent compatibility entry point:
hardware = AcceleratorConfig(
    pe_array_size=(64, 64),
    scratchpad_size=2 * 1024 * 1024,
    num_banks=16,
    bank_width=32,
)
```

## Structure and units

| IR type | Meaning |
| --- | --- |
| `ComputationUnit` | One physical engine with mutually exclusive modes; the legacy dtype/operation sets remain available for Voyager. |
| `ComputeMode` | Exact operation capabilities, optional mode-specific spatial unrolling, and fixed timing facts. |
| `OperationCapability`, `OperandCapability` | An operation signature with named operand dtypes, access directions, and eligible memory instances. |
| `OperationTiming` | Optional startup, completion latency, issue interval, occupancy (cycles), and scalar operations/cycle. Unknown values stay `None`. |
| `SpatialUnrolling` | A named loop dimension and positive factor. An optional `UnrollingRef` inherits another unit's dimension, avoiding duplicated widths. |
| `MemoryHierarchy` | Ordered levels, nearest to compute first. Each `MemoryLevel` contains one or more `MemoryInstance`s. |
| `MemoryInstance` | Size, replication, partitions/banks/rows, fixed placement restrictions, physical ports, buffering, reservations, and costs. |
| `MemoryPort` | Read, write, or shared read/write ports, counted per memory, partition, logical bank, or partition/bank pair. |
| `MemorySize` | Explicit bytes, elements, or rows. L1 sizes are elements per replicated lane; DRAM and scratchpad sizes are bytes. |
| `Connection` | Directed/bidirectional access, endpoint memory ports, bandwidth (own or shared), link latency, transfer startup, and FIFO depth. |
| `Bandwidth` | GB/s, bytes/cycle, or elements/cycle. A rate may reference a memory port or spatial dimensions. Element rates require the operand's bit width. |
| `FusionStage` | Compute unit and operations that may occupy a stage, plus an optional backend legality predicate. |
| `ISAPipeline` | Explicit ordered fusion contract exposed by one target ISA call, separate from the physical connection graph. |
| `ParameterProvenance` | Root-relative field path, classification, source, and notes; the parameter value remains in its owning field. |

Storage targets are `ACTIVATION`, `WEIGHT`, `PSUM`, `SCALE`, and `INDEX`.
For example, `{WEIGHT, ACTIVATION}` describes a shared store, while `{PSUM}`
describes a dedicated partial-sum store. Names are globally unique across
compute units and memory instances, and every connection endpoint is validated.
Unrolling references must exist and cannot form cycles.

Clock frequency is GHz, link latency is ns, and energy is pJ/bit. The legacy
`dram_size` option retains its historical GiB conversion (`size * 1024**3`),
despite its old “GB” spelling. A size of `None` means unspecified, preserving the
old config's ability to describe partially specified hardware before compiling.

## The Voyager instance

The profile describes the matrix, vector, matrix-vector, and sparse-matrix
engines. Its hierarchy is PE registers → dedicated L1 stores → shared banked
L2 scratchpad → DRAM. L1 also contains the sparse engine's scale-row store.
Links describe the direct matrix-to-vector stream, compute memory ports, and
memory-to-memory transfers.

The old hardcoded timing values are now instance data: vector launch overhead
96 cycles, bank switching 8 cycles, sparse row overhead 8 cycles, matrix/vector
FIFO depth 24 vectors, and sparse scale capacity 32 rows. The hierarchy's
relative access costs also live on its memory instances.

The profile represents the existing compiler's Voyager design family. Its
dtype declarations describe logical formats; a concrete hardware variant can
narrow them. They are capability metadata and support queries, not a new
instruction selector or a guarantee that every operator/dtype combination has
a lowering. Explicit ISA pipelines are validated against their units' operation
capabilities; inconsistent declarations fail instead of silently changing the
declared pipeline.

## Changing an instance

Use `dataclasses.replace` on the typed graph. For example, to change vector
width and launch overhead while retaining the rest of the design:

```python
from dataclasses import replace
from voyager_compiler.hardware_config import SpatialUnrolling

vector = hardware.compute_unit("vector")
vector = replace(
    vector,
    spatial_unrolling=(
        SpatialUnrolling("lanes", 32),
        vector.unrolling("pool_channels"),
    ),
    launch_overhead_cycles=64,
)
hardware = replace(
    hardware,
    name="voyager-vector32",
    computation_units=tuple(
        vector if unit.name == vector.name else unit
        for unit in hardware.computation_units
    ),
)
assert hardware.vector_lanes == 32
assert hardware.accumulator_lanes == 32  # inherited through UnrollingRef
```

The old flat properties (`pe_array_size`, `vector_lanes`, `bank_size`, and the
rest) resolve into this graph. They are not stored alongside it. Consequently,
padding, allocation, bufferization, and cost estimation see the same values.
Do not mix flat options with a complete typed graph, or use
`replace(hardware, pe_array_size=...)`: change the graph or instantiate a new
Voyager design through `voyager_config(...)`.

## Compiler boundary

`voyager_adapter.py` adapts the selected instance to the existing Voyager
backend. Interstellar capacities, partitions, parallelism, bank sizes, and
access costs come from the memory hierarchy. Tiling reads launch/bank/sparse
timing and connection bandwidth from the same instance; reporting uses those
same models and records the hardware graph in its Architecture sheet.

The CI driver and benchmark frontend obtain fusion patterns from the instance's
`isa_pipelines`, instead of maintaining separate copies of Voyager's vector
pipeline. Existing explicit `transform(..., patterns=vector_pipeline)` and
`fuse_operator(..., vector_pipeline)` callers retain their supplied policy.

One fused operator corresponds to one target ISA call. A compute-to-compute
connection alone does not establish that this ISA call exists. The current
profile therefore preserves the three explicit pipeline templates exactly.
A future topology-based pipeline discovery pass may propose dataflow routes,
but must check ISA legality before authorizing fusion; it must not turn every
connected path into a fused instruction. Clearing `isa_pipelines` disables
profile-derived fusion even if all physical connections remain present.

Describing another accelerator does not automatically implement its lowering.
The current backend requires the Voyager four-level arrangement, a unified
scratchpad and DRAM, one or two scratch/accumulator slots, and equal matrix L1
port bandwidths. Unsupported backends and hierarchy forms fail explicitly.
Adding Gemmini or Trainium means adding a named instance and the relevant
backend adapter/kernel selection, while reusing and extending the generic IR
types where the target requires additional constraints.

Algorithm choices remain compiler policy: search timeouts, search tolerance,
loop ordering, and minimum CSR search candidates are not hardware capacities.

## Validation

```sh
../ml-env/bin/python voyager-base/test/test_hardware_config.py -v
../ml-env/bin/python voyager-base/test/run_ci.py /tmp/voyager-ci \
  --baseline reference-point/2026-09-27_01-41-40 \
  --suite voyager-base/test/regression_suite.txt
```

Run these commands from `AGEN-voyager`. The focused hardware tests cover
graph references and validation, unit/capacity conversion, non-Voyager hierarchy
construction, ISA fusion legality, and rejection of unsupported topologies. The
marked CI cases compare emitted `model.txt` artifacts against the fixed baseline.
Its numerical warnings are reported separately and are not CI exit failures.

## Physical modes and operation signatures

A `ComputationUnit` is one physical unit. Its `mode_concurrency` is always
`"exclusive"`: different modes of that unit are not independently schedulable
engines. Use two unit instances to describe independent execution, even when
both support the same operation. The IR has no same-mode pipeline simulation,
queue model, mode-switch state machine, or general resource reservation system.

For new targets, use `modes` instead of the legacy `supported_dtypes` and
`supported_operations` sets. Supplying both is rejected. Every
`OperationCapability` names an operation and a complete tuple of operands.
Each operand has an exact dtype and a read/write/read-write/internal role;
its memory tuple lists the eligible stores (empty means unspecified).
Internal values, such as an inter-PE result, cannot name external memory.
Alternative signatures are separate capabilities, avoiding an accidental
Cartesian product of supported input and accumulator types.

```python
from voyager_compiler.hardware_config import (
    AccessMode, ComputationUnit, ComputeMode, DataType,
    OperandCapability, OperationCapability,
)

# Illustrative capabilities, not a claim about a particular generated device.
def mac_mode(name, input_dtype, accumulation_dtype):
    return ComputeMode(name, (OperationCapability("matmul", (
        OperandCapability("lhs", input_dtype, AccessMode.READ, ("scratchpad",)),
        OperandCapability("rhs", input_dtype, AccessMode.READ, ("scratchpad",)),
        OperandCapability("accumulator", accumulation_dtype, AccessMode.INTERNAL),
        OperandCapability("result", accumulation_dtype, AccessMode.WRITE, ("accumulator",)),
    )),))

matrix = ComputationUnit("matrix0", modes=(
    mac_mode("int8", DataType("int8", 8), DataType("int32", 32)),
    mac_mode("bf16", DataType("bf16", 16), DataType("fp32", 32)),
))
```

The enclosing hardware graph validates each named store and the direct
connection needed for its access direction. A copy through another memory is
an operation, not implicit reachability. `supports_signature(mode, operation,
dtypes, memories)` checks the exact signature and supplied placement choices.
The older `supports(operation, dtype)` remains a coarse capability query;
it does not establish a complete signature's legality.

A mode may override spatial unrolling dimensions, queried with
`unit.unrolling(dimension, mode=...)`; unspecified dimensions use the physical
unit's base values. Operation timing overrides mode timing as a complete record.
`operation_timing(mode, capability)` resolves that choice. Missing timing is
unknown, not zero. These optional fixed hardware facts do not prescribe
shape-dependent cost formulas or scheduling policy.

## Memory geometry and physical access

`size` is the total capacity of one replica. `partitions` divides it into
independent physical partitions. `banks` counts logical banks spanning all
partitions; equivalently, each partition has that many bank slices. This keeps
Voyager's existing `bank_size` interpretation:

```text
partition_size      = size / partitions
bank_size           = size / banks
partition_bank_size = size / partitions / banks
```

The quantities retain `MemorySize.unit`; they are bytes only for a byte-sized
memory. A row width in `row_bytes` requires byte capacity. Partitioned geometry
and rows must divide capacity evenly. `word_bytes` is an access width and need
not be the physical row size.

The fixed placement facts are `allocation_alignment_bytes`,
`partition_start_alignment`, and `single_bank_allocation`. The target allocator
interprets these facts when placing tensors. The IR does not choose partitions,
banks, offsets, or a mapping from tensor dimensions to memory dimensions.
Shape-dependent start rules and tile legality stay in the target interface.

A `MemoryPort` specifies `READ`, `WRITE`, or `READ_WRITE`, a physical count,
and a scope (`memory`, `partition`, `bank`, or `partition_bank`). A read/write
port shares its count across both directions; independent read and write ports
are separate declarations. Scoped ports describe independent copies in the
memory geometry. Which tensor access selects a particular bank remains an
allocator/scheduler concern.

Connections bind endpoint ports with `source_port` and `target_port`. Forward
transfers read the source memory and write the destination memory; bidirectional
connections require read/write ports. References and directions are validated.

## Shared memory and transfer bandwidth

There is no separate shared-resource registry. Connections describe the paths,
memory ports describe the physical access points, and `shared_with` identifies
one bandwidth budget used by multiple memory/transfer paths:

```python
from voyager_compiler.hardware_config import Bandwidth, BandwidthUnit, Connection

# Illustrative bandwidth; "read" is a declared port on the memory "sram".
vector_link = Connection(
    "sram_vector", "sram", "vector",
    Bandwidth(32, BandwidthUnit.BYTES_PER_CYCLE),
    bidirectional=False, source_port="read",
)
scalar_link = Connection(
    "sram_scalar", "sram", "scalar",
    shared_with="sram_vector", bidirectional=False, source_port="read",
)
```

An alias cannot also supply its own bandwidth. `bandwidth_connection(name)`
resolves the canonical connection; all aliases consume that single aggregate
budget. `connection_bytes_per_cycle(name, element_bits=...)` resolves its rate.
Unknown references and cycles fail validation. Shared budgets are restricted
to memory/transfer connections. Both directions consume one budget; independent
directional budgets use separate directed connections.

Referring to the same endpoint port expresses shared port availability;
`shared_with` expresses shared bandwidth. A scheduler must honor both. Equal
numeric bandwidth values alone do not imply that two paths share hardware.
The existing `Bandwidth.memory_port` field remains a memory word-width
reference for legacy rate calculation; it is not an endpoint port selector.
`startup_ns` is per-transfer setup, separate from the existing `latency_ns`.
Neither field describes transaction sizes or transfer-layout capabilities.

## Parameter evidence and adapter boundary

`AcceleratorConfig.provenance` holds `ParameterProvenance` records. A path such
as `connections.sram_vector.bandwidth.value` resolves through dataclass fields
and named tuple members; numeric indices address unnamed members, for example
`computation_units.matrix0.modes.int8.operations.0.timing.latency_cycles`.
Unknown paths and duplicate records fail validation. Classifications are
`documented`, `measured`, `assumed`, and `derived`; documented/measured records
require a source. Notes can record measurement conditions or derivations.
Units remain explicit in the owning field/type. Evidence is optional; its
absence must not be interpreted as proof of a documented hardware value.

Voyager's default profile retains its existing parameters and behavior. Its
current Interstellar/allocator adapter rejects explicit compute modes, extended
memory geometry/placement/ports, shared connection budgets, and extra transfer
startup instead of ignoring them. Provenance is metadata and is allowed.
Target adapters must consume these declarations before claiming support.

The workspace's `formal-solver` models one Trainium2 NeuronCore-v3 and has no
Gemmini profile. The new types can describe its physical engines, memory
geometry, access paths, and shared transfer limits. Its tensor layouts,
partition-aware allocation, PSUM bank choices, and scheduling remain separate
work. Likewise, Gemmini's selected dataflows and instruction-specific shape
rules belong in its adapter. Neither backend is implemented by this IR change.

Intentionally deferred from the brainstorm: (6) tile/shape constraints,
(7) dataflow and transformation capabilities, (9) tensor-to-memory mappings,
(12) a separate named-resource registry, (13) resource reservation contracts,
and (14) transfer capability rules. SoC/ROB modeling and same-mode pipelining
are also outside this implementation.

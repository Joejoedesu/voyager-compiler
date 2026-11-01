# Extension roles and handoffs

A role identifies a decision and its output, not a required file or a compiler
phase. Start with the row that owns the requested change. One person can fill
several roles; one small target can keep several concerns in the same module.
The [compilation guide](compilation.md) owns shared invariants, API semantics,
validation scope and the ordered coverage/performance objectives. This document
routes work without redefining those contracts. Coverage summaries and performance
validation are selected for the workload/target task. They do not require a new
coverage subsystem, a universal execution checker or fresh measurements for every
registration. Missing scheduler capabilities are scoped framework work only when
the requested implementation needs them; an old gap is not a standing task.

The [extension catalog](extensions.md) maps public bindings to signatures,
consumers and decision owners. Roles are concerns; they are not extra callbacks.

## Focused role readers

Each reader includes problem-driven implementation decisions, boundaries and a
checkable handoff. Start from specifications and shared contracts when no target
implementation exists; an existing target guide is optional input.

Group work by the public binding that consumes it. Read the applicable reader;
there is no requirement to load every role or create one source file per callback.
Paired encoding/emission hooks share a frozen-artifact boundary, while runtime is
separate. Recipe matching, costing and buffer construction have separate readers
within `ImplementationSpec`. These are documentation groupings, not new APIs.

### Registered extension points

| Extension binding | Role reader | Decision owned |
| --- | --- | --- |
| `WorkloadSpec` | [Workload integration](roles/workload.md) | Model/assets, inputs/state, export/reference and calibration sample acquisition |
| `TargetSpec.name`, `TargetSpec.hardware` | [Target registration and hardware](roles/target.md) | Registry assembly, physical capabilities and configuration |
| `TargetSpec.operators` + schemas/implementations | [Operation semantics](roles/semantics.md) | Computation, signatures and effects; registered classification alone is not a complete definition |
| `TargetSpec.input_preprocessing` | [Input preprocessing](roles/input-preprocessing.md) | Optional graph/input representation before quantization; matching calibration and explicit input boundary |
| `TargetSpec.quantization` | [Quantization](roles/quantization.md) | Formats, scales/domains, calibration and conversion |
| `TargetSpec.legalize` | [Graph legalization](roles/legalization.md) | Supported-form rewrites before shared normalization |
| `TargetSpec.normalization` | [Normalization policy](roles/normalization.md) | Shared layout, alignment, constant preparation and fusion controls |
| `TargetSpec.lower` | [Custom-operation lowering](roles/lowering.md) | Normalized tensor graph to target operations, before search |
| `TargetSpec.implementations`, `ImplementationSpec.matches`, `ImplementationSpec.candidates`, `ImplementationSpec.validate` (plus recipe identity, description and shape-cache policy) | [Implementation alternatives](roles/implementations.md) | Discoverable paired recipes, legality and alternative parameters |
| `TargetSpec.model`, `ImplementationSpec.model`, `ImplementationSpec.estimate` | [Performance models](roles/performance.md) | Fit/cost estimates used to compare implementations and mappings |
| `ImplementationSpec.build` | [Bufferized kernel construction](roles/bufferization.md) | Explicit buffers, copies, loops, compute and dependencies for a candidate |
| `TargetSpec.refine` | [Buffer and storage refinement](roles/refinement.md) | Post-build target declarations and permitted refinements before allocation |
| `TargetSpec.encode`, `TargetSpec.emit` | [Encoding and emission](roles/emission.md) | Save common/target IR, then generate target files from that bundle |
| `TargetSpec.runtime` | [Target runtime](roles/runtime.md) | Endpoint submission, outputs and state |
| `TargetSpec.characterize` | [Performance characterization](roles/characterization.md) | Attributable measurements for offline fitting/validation |

### Shared framework and coordination

These readers describe work on common machinery; they are not extra target hooks.

| Owner | Role reader | Boundary |
| --- | --- | --- |
| `compiler/search/` | [Shared search engines](roles/search.md) | Existing engine enumeration, cache and selection; no independent engine-plugin registry |
| `compiler/analysis.py`, `compiler/allocate.py`, `compiler/banked_allocation.py` | [Shared storage analysis and allocation](roles/allocation.md) | Authoritative placement, banks and lifetimes; targets declare requirements rather than override allocation |
| Cross-interface validation and handoff | [Integration and evidence](roles/integration.md) | Coordinate scoped changes and evidence; no new compiler stage or mandatory adoption campaign |

Shared-contract changes edit the canonical interface and its consumer when a
required capability cannot be represented. Preserve one selected plan and one
final allocation owner. Existing reports and unchecked task steps do not imply
that implementation is missing.

## Scheduling and performance handoff

The search engine enumerates and selects; the model evaluates; the builder
realizes movement/control; the emitter and runtime submit instructions. Review
these together when a change affects execution assumptions. The detailed
[scheduling relationship](compilation.md#scheduling-and-the-general-performance-model)
and [execution-model debugging guidance](compilation.md#execution-model-and-converter-consistency)
explain the handoff. Select validation methods for the target and affected
assumptions; a universal checker is not part of the role contract.

For example, a hardware-loop alternative needs matching semantics, explicit
capacity/ownership declarations, paired cost assumptions and an emitter that
consumes the selected mode. Moving operand loads or output stores can change
reuse, accumulator residency or submission visibility. Output checks alone do
not establish a performance claim. Compare legal alternatives under the same
case boundary when fitting a model; ordinary registration can use analytical costs.

## Routing common changes

| Requested change | Start here | Shared changes only when needed |
| --- | --- | --- |
| Choose an input representation before quantization | [Input preprocessing](roles/input-preprocessing.md) | Missing graph transformation or runtime preparation contract |
| Add a workload | Workload integration | New frontend/state semantics unsupported by `WorkloadSpec` |
| Support another ISA form for an existing op | Implementation alternatives, paired model, builder and emitter | New effects, native completion or candidate-result capability |
| Compare algorithms for one fixed region | [Algorithm comparison](compilation/algorithm-search.md), implementation alternatives and paired builders/models | New observable semantics or overlapping-region selection; sequential component composition is already shared |
| Improve poor schedule selection | Model/realization evidence, then search | Missing representable candidates or search-engine protocol |
| Reuse existing bufferized kernels on a target | Target model and existing builder hooks | A genuinely different algorithmic control structure |
| Fix lifetime, packed scale or bank conflicts | Analysis/allocation and target layout declarations | Extend canonical buffer/allocation records only if necessary |
| Add measured performance tuning | Performance characterization and `performance/` | New evidence type or fitting behavior with a concrete use case |

## Relationship to historical role guides

Historical role names describe concerns, not current import paths or instructions
to recreate modules. Use the canonical owners above. Shared search supports recipe
selection and three mapping/tile engines; it does not expose every historical
scheduler-provider/result API. See the [scheduler contract](compilation.md#scheduler-interface-and-limits).
The [adoption procedures](adoption.md) cover target registration/extension and
workload × target binding/refinement. They do not require status trackers.

Search reuse is split between target/recipe constraints (`MatrixSpace`, model and
candidate callbacks), shared enumeration (`search_matrix` and other engines), and
execution construction (bufferization). See [adapter ownership](compilation/performance-models.md#target-owned-matrix-search-adapters).
Workload input rewrites retain their [explicit boundary](compilation/rewrites.md#input-boundary-rewrites).
For a performance task, [diagnose material utilization gaps](compilation/performance-models.md#required-diagnosis-when-utilization-or-selected-performance-is-poor)
before declaring the implementation adequate; findings go to the responsible role.

Reuse means calling shared implementations, not merely placing a private compiler
behind a hook. During a target change, check that selected mappings survive into
emission, geometry/footprints have one owner, and numerical preparation and replay
mechanics reuse applicable shared code. See [search reuse](compilation/performance-models.md#connect-search-to-executable-code),
[quantization](roles/quantization.md) and [emission](roles/emission.md).

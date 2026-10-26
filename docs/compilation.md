# Voyager compiler architecture

This is the implementation tree for `Joejoedesu/voyager-compiler`, branch
`AGEN-compiler`, based on compiler-2 commit
`cef1e7e42b31e1cfcc01a1e4e02cedab11e3e9ef` (2026-10-20). The [detailed walkthrough](../../reports/voyager-compiler-walkthrough.html)
combines the flow, extension ownership, adoption guidance and dated evidence.
The original acceptance snapshot is in `../reports/voyager-implementation-2026-10-23/`;
cleanup/review evidence is in `../reports/voyager-documentation-2026-10-24/`.
Start with [documentation navigation](README.md), then read the relevant
sections of this reference. [Target and workload adoption](adoption.md) gives the development
workflow; [roles and handoffs](roles.md) route a task to its focused reader and
source owner. Loading every role or this entire reference is not required.

The shared stages and registered extensions below are implemented. Imports use
one canonical hierarchy: `targets/`, `hardware.py`, `extensions.py`, `ir/` and
`compiler/`. The forwarding-only `hardware_config.py`, `backend/`,
`compiler/search/runtime_model.py` and `codegen/transform/{tiling,bufferize}/`
paths have been removed. GPU target lowering and its training-example switch
have also been removed. CUDA tensor execution for shared numerical algorithms
is independent of target registration and remains available. Smaller graph
utilities, protobuf definitions, reporting and quantization algorithms retain
their existing substantive modules.

| Removed import | Canonical owner |
| --- | --- |
| `hardware_config` | `hardware` |
| `backend.<target>` | `targets/<target>/` and its registered implementations |
| `Backend` / `config.backend()` | `TargetSpec.implementations`; the planner selects registered recipes |
| `codegen.transform.tiling` | `compiler.search` (`interstellar`, `tile_shapes`, `attention_tiles`) |
| `codegen.transform.bufferize` | `compiler.bufferize`; allocation is `compiler.allocate` |
| `compiler.search.runtime_model` | `extensions.OpModel` / `RuntimeModel`; records in `ir.plan` |

These internal import paths are intentionally no longer compatibility APIs.
The documented public `transform()` / `compile()` entry points remain available.

## Guide scope and implementation objectives

This guide describes the shared contract and implementation structure. It is not
a task-status tracker; inspect current bindings before treating a stated gap as work.
Target READMEs specialize it without weakening shared invariants; skills route
work to these owners. The unified walkthrough separates current implementation, dated evidence and remaining gaps;
donor guides remain historical evidence. Proposed and donor APIs, defaults and
performance results do not override this checkout's interfaces or establish
current validation. Source
and dated run artifacts establish what is implemented; if they conflict with a
required contract, report and resolve the discrepancy rather than silently
changing the requirement or treating the design as completed work.

Target adoption has two ordered objectives, with numerical/state/effect correctness
as a prerequisite:

1. **Maximize correctly supported workload coverage.** During bring-up, identify
   unsupported operations, shape/dtype restrictions and expensive host execution.
   Use graph inspection, compiler diagnostics or a small workload/target summary
   to guide the next mappings. Distinguish declared support, emitted mapping and
   endpoint-validated execution. A coverage percentage is optional; when useful,
   define its semantic-work denominator and keep it fixed across the comparison.
   Unsupported work stays in that denominator, and introduced copies or fused
   call counts must not inflate the numerator.
2. **Improve useful hardware utilization for that mapped portion.** Compare
   candidates under the same workload semantics, mapped work, precision and timing
   scope. Use target cost estimates during search. When assessing actual
   performance, use applicable endpoint evidence for latency/throughput and useful
   work per elapsed cycle. Padding and array-busy time are separate diagnostics;
   include host/device boundaries when claiming end-to-end latency.

Coverage is a priority, not a 100% prerequisite for tuning: improve currently
mapped regions while coverage grows. Account for compile-time eliminated work
separately from unsupported work; extra padded activity does not improve useful
utilization.

Coverage diagnostics are a target-adoption aid. There is no required common
coverage IR, persistent source-op ledger or global coverage optimizer. Add source
correspondence only when a concrete comparison needs it. The absence of a uniform
percentage report does not make an existing mapping or the shared framework
incomplete. Missing operation support remains a concrete workload/target task.

Evidence has a declared scope. Lowered native-call, view or copy counts do not
establish source-op coverage. A pinned artifact/status/numerical-summary gate does
not establish device accuracy or utilization. Target-owned guides define their
comparison boundary and link dated results.

The framework provides candidate cost models, measurement adapters and fitting.
Characterization and execution-model debugging are selected per target and task;
they are not mandatory stages of compilation or target registration. Analytical
models can guide search without a newly fitted profile. Reuse donor measurements
when their hardware, execution assumptions and observation scope still apply,
recording provenance and any limits. Collect new measurements when changed
behavior or a requested performance claim needs new evidence. Track supported
operations, correctness and performance evidence separately.

## The compilation flow

```text
WorkloadSpec + options                        TargetSpec + hardware
          │                                          │
prepare/export → quantization (default or target override) → target legalize → common normalize
          → target custom-operation lowering (optional)
          │                     FX + numerical/value contracts
          ▼
shared search: registered implementations × their candidate parameters
          ├─ Interstellar: hierarchical GEMM/conv mappings
          ├─ tile shapes: vector, reduction, GEMV, pooling
          └─ attention tiles: coupled query/key blocks
          │                     provisional CompilationPlan
          ▼
bufferize selected recipe → refine layouts/residency/native calls
          │                     executable BufferProgram (FX)
          ▼
allocate actual buffers → finalize invocation effects → validate → freeze
          │                     final AllocationPlan + choices + identities
          ▼
encode common IR → target emit → save checksummed bundle → runtime/replay

capacity rejection ── bounded replan (default at most two retries) ──► search
accelerator measurement ── fit + held-out validation ──► performance profile
```

`pipeline.compile_workload()` owns this order. `compile_bufferized()` starts
from an already prepared/normalized graph. The public `transform()` and
`compile()` compatibility interfaces still work; `compile()` returns the encoded
protobuf model, while the new pipeline returns a `CompilationResult` containing
the executable graph, frozen plan, bundle, inputs and reference.

| Responsibility | Source owner | Output or hook |
| --- | --- | --- |
| Construct model, check assets, inputs and export | `compiler/prepare.py` | `PreparedWorkload`; optional workload frontend/preprocessing/reference |
| Quantization and calibration | `compiler/quantize.py`, `quantization/` | Shared PT2E or `QuantizationPolicy.run` target override with resolved configuration; [hook order and controls](extensions.md#quantization) |
| Target legality and common rewrites | `TargetSpec.legalize`, `compiler/normalize.py`, `TargetSpec.lower` | Legalize → shared layouts/rewrites → custom operations; [stage boundaries](extensions.md#graph-legalization) |
| Alias/effect analysis | `compiler/analysis.py` | Derived `ValueRelations`, graph identity |
| Implementation and tile search | `compiler/search/` | Candidates, rejection reasons, selected `ImplementationChoice` |
| Loops, buffers, movement, kernel bodies | `compiler/bufferize/` | Executable FX using shared allocation/copy/view/insert/control ops |
| Layout/residency refinement | `TargetSpec.refine` | Same buffer graph; no private allocator |
| Placement and invocation effects | `compiler/allocate.py`, `compiler/banked_allocation.py`, `compiler/bufferize/invocation.py` | Concrete arenas, banks, slots, native reservations and read/write effects |
| Frozen encoding and target projection | `compiler/emit.py`, `TargetSpec.encode`, `TargetSpec.emit` | Generic model plus selected target instruction/runtime |
| Bundle and replay | `artifacts.py`, `runtime.py` | Checksums/provenance; host or registered target execution |
| Offline performance calibration | `performance/` | Measurements, fitted profile and held-out errors |

## Static tensor preparation and the storage limit

`codegen/transform/rewrites.py:fold_constant_generators` prepares immutable
parameter/weight/scale subgraphs during common normalization. It runs before
layout selection and again after layout selection, before operator fusion and
bufferization. GEMM weight transposes and broadcast/scale relayouts all enter this same decision.
`data_layout.py` requests ordinary reshape/permute/transpose operations instead
of eagerly creating extra parameter buffers. A private scale can be prepared
once and its original retired; if another consumer still needs the original,
the shared budget can keep the relayout as a view or runtime operation. For
example, Conv2d plus a `[4,1,1]` scale that is also returned keeps 32 bytes
of weights/scales, instead of adding a 16-byte NHWC scale copy. Shape-only
relayouts need no data movement when bufferization can represent them as views.

For each connected static component, identify its externally consumed outputs
and the original attributes that become entirely unused. Accept preparation
only when **prepared output bytes ≤ released original bytes**. Every retained
original and every escaping output counts. Multiple reads of the same attribute
count once; different attributes count separately, even if their eager tensors
share storage, because the emitter stores them separately. Intermediate results
are discarded. A slice is stored compactly rather than keeping its old backing.
For example, `W * per_output_scale -> transpose` may replace W; if another
consumer still needs W, storing both representations is rejected unless the
other released inputs actually cover the extra bytes.

There is also a per-operation bound: the result cannot exceed the largest
input's encoded bytes. The accounting uses the logical quantized dtype, rounds
sub-byte payloads up, and includes an already declared `physical_shape`. It is
an accounting rule for this preparation stage, not compiler peak RAM or a
promise about later target packing, bank alignment or hardware-required padding.
Those remain explicit target/allocation decisions. No new precision conversion
is introduced. Operations already present in the graph retain their numerical
semantics and order; this does not reassociate `matmul(X, W) * scale` into
`matmul(X, W * scale)`.

| Eligible family | Examples and limits |
| --- | --- |
| Shape and layout | reshape/view/flatten/squeeze/unsqueeze, transpose/permute; prepare only if the final representations fit the budget |
| Subsetting | slice/narrow/select, static index_select/gather; all tensor indices must be static |
| Elementwise arithmetic | add/sub/mul/div, neg/abs/clamp, min/max, reciprocal/sqrt/rsqrt/pow/exp/log/sin/cos; reject expanding broadcast products |
| Existing dtype conversions | casts to the same or narrower element width, preserving the graph's specified conversion |
| Reductions | sum/mean/amin/amax; account for result dtype and size |

Materialized repeat/tile/expand, enlarging padding, widening casts and
full-tensor dequantization are not prepared as parameter transformations. A
finite ATen allowlist governs those transformations. Neither preparation path
accepts random, unrecognized custom or stateful functions; deterministic
generator setup also retains its explicit constant-quantization operation.
Mutable values are excluded, including writes through schema aliases, `out=`
arguments and opaque calls. This includes fresh computed or generated tensors:
`p = W * scale; p.add_(...)` must still recreate p on every invocation even if
W itself is immutable. Parameters must remain fixed for the compiled artifact's lifetime;
changing weights or scales requires preparation again. Unaccepted operations
remain in the normal graph for layout handling, fusion and bufferization.

The pre-existing deterministic input-free generator mechanism is separate:
`arange`, zeros/ones/full/eye/scalar_tensor and shape-only `full_like`, together
with their deterministic derived mask/position-table computation, create
constants without a stored representation to replace. This existing creation
path retains broadcasts and matrix products used to build those tables. Its
final prepared bytes are explicitly reported as `generator_bytes`; they are
**not** claimed as storage-neutral parameter transformations. Random/empty
generators are not prepared. Deferred
values retain a producer recipe, while real prepared parameters release their
unused original attributes. `model.meta["constant_folding"]` records phases,
accepted/rejected components and byte budgets; it is a local diagnostic, not a
new IR or public extension protocol.

Target owners specify needed layouts through existing normalization policy;
shared folding owns static eligibility and byte accounting. If a retained
original prevents prepacking, target builders/model/emitter must price and
realize the remaining runtime transpose. Layout heuristics must charge retained
or mutable representations; shared preparation decides the final byte budget,
and the kernel cost model prices the operations that remain. Removing a graph
operation alone does not establish a measured speedup.

## Related optimizations across the pipeline

The [shared transformation catalog](compilation/rewrites.md) records available
rewrite families, where they run, how they are enabled, and where target code
must supply an implementation. Start there before adding a pass. It includes
logical shape/padding rewrites, layouts, static preparation, quantization folding,
fusion, cache updates and buffer-level movement/residency.

[Padding details](compilation/padding.md) distinguish logical values, local tile
filling and physical storage. Absorption into a producer/consumer is a placement
choice, not a separate padding type. Physical allocation is a related compiler
stage, not automatically a tensor-graph rewrite.

## IR ownership and why these records differ

### Tensor graph, execution graph and compilation decisions

Use these explicit names when discussing a stage:

| Name in this guide | Concrete object | What it says |
| --- | --- | --- |
| Tensor-operation graph (before bufferization) | FX `GraphModule`, initially `PreparedWorkload.graph` | What tensor operations compute, including shapes, dtypes, quantization, aliases and writes. Say exported, quantized, or normalized to identify its stage. |
| Bufferized execution graph | `BufferProgram.graph`, also an FX `GraphModule` | How the chosen implementation executes: allocations, buffer views, transfers, loops, compute, dependencies and native invocations. |
| Implementation and allocation decisions | `CompilationPlan`, attached as `BufferProgram.plan` | Which recipe/parameters were selected per site, alternatives/rejections, tile/placement estimates, and later final addresses/bank placement. |

Earlier descriptions called the first object the **semantic graph** and the
third the **selected plan**. Neither is a separate class with that name.
The tensor graph is not always hardware-independent: target legalization and
normalization can introduce target arithmetic, layouts and padded shapes before
bufferization. Both graph forms use FX, and passes rewrite them in place; these
names describe compilation stages, not three parallel graph infrastructures.

For `C = matmul(A, B)`, the tensor graph describes the arithmetic and shapes.
The decisions record a recipe such as explicit commands or a hardware-loop
macro, its parameters and cost. The execution graph contains the corresponding
copies, compute and waits (or the declared macro), and allocation later adds
actual buffer locations to `CompilationPlan`. **The plan is a decision record,
not an executable command schedule**; the execution graph owns operation order
and dependencies. Before allocation it is provisional; after validation/freeze,
the graph and decisions together define what emitters must preserve.

| Record | Meaning | Why it is separate |
| --- | --- | --- |
| Physical hardware graph (`hardware.py`) | Memories, capacities, bank geometry, engines, interconnect, timing facts/provenance | Workload choices and fitted correction factors must not change the identity of the machine |
| `PreparedWorkload`, numerical contracts (`ir/graph.py`) | Exported computation, example/state inputs, formats and scale-domain relations | Semantics precede implementation and storage selection |
| `ValueRelations` (`ir/graph.py`) | Derived aliases and schema writes, tied to graph identity | Analysis has no authority to allocate addresses; it is not another storage graph |
| `ImplementationChoice`, `TileResult` (`ir/plan.py`) | Selected recipe, parameters, mapping, placement estimate and cost | An estimate remains provisional until the expanded buffers fit |
| `BufferProgram`, `TargetInvoke` (`ir/buffer.py`) | Actual executable operations; native effects and ownership | ISA projections must follow this selected program |
| `AllocationPlan`, `BufferAllocation` (`ir/plan.py`) | Final physical memory, address, bank cells, slots or native reservation for each buffer | Values can share storage through views or reuse at disjoint times; neither fact changes value semantics |
| `BankLayout`, `BufferComponent` (`ir/plan.py`) | Selected bank constraints and fields packed into one physical backing | A VSQ scale component is a distinct logical tensor, not a tensor view of its integer payload |
| `ExecutionOptions`, performance profile | Software buffering policy and empirical cost corrections | Neither belongs in physical capacity or architectural identity |

The final `CompilationPlan` includes allocation. There is no parallel
`value_storage`/`storage_plan` hierarchy. Ordinary tensor views resolve to buffer
roots. Packed components retain their own logical identity, share the backing's
placement, and combine lifetimes; the declared backing reserves the physical
bytes. `physical_shape` can reserve padded storage without changing the logical
shape. Target encoding defines how the components occupy fields of that backing.

The allocator checks byte intervals for ordinary arenas and bank/entry cells for
clustered arenas. Two cluster modes can have different virtual offsets while
occupying the same physical cells; those cells determine overlap. Bank-port
conflicts can be stronger than byte overlap. Async operands stay live through
completion waits, including the final pipeline drain. Whole-kernel output
buffers exist before invocation so dying inputs cannot overwrite their outputs.

Freezing protects selected plans, buffer bindings and plan-relevant graph
metadata. Emitters may encode these choices but cannot retile or allocate new
scratch. Content checksums protect saved files. The in-memory graph digest is
not a checksum of every tensor's mutable `.data`; do not claim it detects
arbitrary weight mutation.

## Search and reusable bufferized kernels

`MatrixSpace` and `GridSpace` are retained internal candidate descriptors, not
separate IR pipelines. `MatrixSpace` invokes Interstellar's hierarchical loop
mapping search. `GridSpace` invokes dimension-wise tile enumeration in
`tile_shapes.py`. `AttentionSpace` searches coupled query/key sizes. Their
results are `TileResult` records consumed by the same builders and allocator.

An `ImplementationSpec` binds `matches`, `candidates`, `build`, and optional
`model`, `estimate` and `validate` hooks. Overlapping recipes are compared; a
first matching rule is not alternative search. Every competing recipe requires
a finite cost. The shared planner isolates candidate metadata, records feasible
alternatives/rejections, and chooses within the runtime tolerance using traffic
and latency. The current candidate budget is 64 per site. Capacity failure may
exclude the selected recipe/parameter pair and retry from the prepared graph.
This is bounded local replanning, not a global optimality guarantee.

Use the existing bufferized template when only layout, tile shape, placement or
instruction realization changes. GEMM/conv, pointwise/reduction/pool, attention,
CSR/outlier and whole-tensor templates share movement/control primitives.
`build_native()` is for a macro with synchronous completion at its IR boundary
that owns its private allocation and implicit movement. A new algorithmic control structure may need a new
builder, while retaining the same shared ops and `BufferProgram` contract.

`TargetInvoke.allocation_owner` and `scheduling_owner` are independent. Both a
software-expanded recipe and a hardware macro may reserve private memory. Hardware
scheduling may also consume compiler-selected addresses. Native private arenas
cannot overlap live compiler allocations; declare ownership and demand before
final allocation.

### Asynchronous buffer operations and native completion

Bufferization makes storage and execution explicit; it does not by itself mean
every operation is asynchronous. The shared path already supports asynchronous
DMA and compute, while `TargetInvoke` currently exposes synchronous completion:

| Execution mechanism | Completion and buffer lifetime |
| --- | --- |
| `async_copy` and `async_wait` | The transfer signals a semaphore; its source/destination remain live through the consuming wait. |
| `commit(..., dependencies, post)` | A compute region waits for input/slot dependencies and signals `post` on completion. The async pipeline overlaps tiles and drains the last outstanding work. |
| Standalone `TargetInvoke` | Its reads/writes are treated as complete at the invocation boundary; no native launch/completion event binding exists yet. `completion != "synchronous"` is rejected. |

Synchronous at the IR boundary does not require serial hardware internals or a
blocking CPU instruction. The backend must preserve completed-before-use/reuse
semantics through ordering or waits. Existing host replay runs these operations
synchronously and checks semaphore dependencies; it does not measure overlap
or prove device completion behavior.

An asynchronous native extension would need to connect launch completion to
consumer waits and keep input/output buffers and private arenas live until that
completion. Reuse the shared semaphore/lifetime mechanisms where possible;
the missing contract is the native binding, not a second async framework.
Wrapping a nonblocking launch in `commit` is insufficient if it signals `post`
on submission rather than actual completion. See
[shared async operations](../src/voyager_compiler/compiler/bufferize/ops.py),
[lifetime tracking](../src/voyager_compiler/compiler/allocate.py) and
[native invocation contract](../src/voyager_compiler/ir/buffer.py).

### Padding ownership

Model/input padding, target shape legalization, tile/halo padding and extra
physical storage are different decisions. Use the
[padding guide](compilation/padding.md) to choose the owner for a workload–target pair,
the required fill/mask/slice behavior and the validation boundary. Physical
allocation alone never creates logically padded values.

## Scheduling and the general performance model

“Schedule” refers to several decisions. They share a selected plan, but have
different owners:

| Decision | Current mechanism | Performance-model relationship |
| --- | --- | --- |
| Which implementation of a semantic op? | `planner.select_implementation()` compares registered recipes and their parameters | Recipe `estimate` or the paired model's selected tile cost scores the candidate; it does not choose a different recipe during emission |
| Which software tile, traversal and placement proposal? | Interstellar, dimension-wise or attention search through `Tiler` | `RuntimeModel.prepare(node)` produces an `OpModel`; `space` defines candidates, `fit` proposes legal placement, `cost` estimates that mapping |
| How do buffers, copies and compute execute? | Recipe `build`, shared bufferized templates, then target `refine` | The expanded program must realize the traffic, reuse, dependencies and scratch that were estimated; actual allocation validates the footprint |
| Which primitive instructions or native macro implement it? | Selected recipe plus `TargetSpec.emit`, then runtime submission | Instruction expansion, queue visibility and implicit native traffic must agree with the priced execution policy |
| How do model coefficients improve? | Offline `performance.characterize()` and `fit_profile()` | Measurements calibrate cost for future searches; fitting is not a runtime scheduler and does not repair a different emitted execution |

The general layer is a **protocol**, not one hardware-independent latency
formula. Targets have different engines, memory systems and command behavior.
Their models implement the same fit/cost contracts while
using target-owned formulas. A native macro can have an analytical or measured
cost without pretending that Voyager schedules its private micro-operations.

In the current implementation, a builder may ask `Tiler` to search mappings
while constructing a recipe candidate. The planner then compares its returned
`TileResult.cost` (or an explicit `estimate`) and records the selected
`ImplementationChoice`. Allocation is later and authoritative. This is not a
single global scheduling optimizer: graph fusion/normalization, local recipe
search and target-local factor selection have separate scopes. The default
local runtime tolerance is 2%; traffic/energy tie behavior is retained from the
base. The coverage-first objective above is the adoption objective; the current
planner has no source-op coverage score or global coverage optimizer.

A new scheduling engine and a new hardware cost model are separate extensions.
An existing engine can enumerate candidates for a new target model. An ISA
alternative with different execution ownership needs its own paired recipe.
The following section describes the current shared interface and its limits.

<a id="where-the-nv-scheduler-interface-went"></a>
## Scheduler interface and limits

| Concern | Current owner | Boundary |
| --- | --- | --- |
| Recipe dispatch | `compiler/search/planner.py:select_implementation` | Compares `ImplementationSpec` candidates during bufferization |
| Candidate enumeration | Recipe `candidates`, paired `model` / `build` / `estimate` | Recipe registration is public; no separate scheduler-engine plugin registry |
| Operation facts | `RuntimeModel.prepare(node)` → target `OpModel` | FX facts; no general immutable FX-free workload snapshot protocol |
| Shared engines | `compiler/search/{interstellar,tile_shapes,attention_tiles}.py` | Three existing engines; another engine needs an explicit adapter/contract |
| Selected result | `ir/plan.py:TileResult` wrapped in `ImplementationChoice` | Tiling, placement, cost and aggregate access counts; no complete generic typed access-event/energy/named-factor schema |

Public extension contracts live in `extensions.py`; search state lives in
`compiler/search/tiler.py`. Physical placement remains in the shared allocator,
after bufferized expansion. Target-local metadata does not create a generic API.

Extend these canonical owners only when the requested execution/search capability
needs it. Preserve selected decisions and one authoritative allocation plan;
do not add a parallel end-to-end pipeline. This section records interface limits,
not a mandatory backlog for every target-adoption task.

## Execution model and converter consistency

This is a target-specific implementation and debugging direction. The shared
framework does not require a universal execution-conformance checker, a new event
IR or a prescribed trace format. Choose code review, targeted command checks,
traces or measurements according to the target and the question being investigated.

The performance model must describe the execution produced by the complete
builder → converter/emitter → runtime submission path. Numerical equivalence
and frozen IR do not prove that relationship. An emitter can leave the frozen
graph untouched while generating different DMA placement, reuse or synchronization.
The same opcode counts and total bytes can also have different latency when
commands enter different submission windows.

| Assumption used by search | Target-specific review question |
| --- | --- |
| Operand stays resident across iterations | Loads occur at the declared loop scope; buffer generations and physical regions remain valid until last use |
| Accumulator stays on chip across K | Initialize once, accumulate at the selected precision, then store the completed result; spills/reloads must be explicit and priced |
| Transfer volume and command overhead | Match padded bytes, DMA rectangles/strides, ISA expansion and configuration/preload commands; account for native macro fetches |
| DMA overlaps compute | The next load is submitted early enough, has a free destination slot, and is not blocked by a barrier, bank conflict, finite queue or shared interface |
| Hardware-loop contexts overlap | Preserve launch/configuration order and context lifetime; price repeated implicit fetches and commands/drains that prevent admission |
| Buffers are reusable | Read/write footprints, slot generations and completion dependencies agree with final addresses and runtime waits/fences |

A model may assume weights loaded outside a reused loop, the next
tile's operand load visible during current compute, and output store only after the reduction.
Reloading retained weights, placing that operand load behind a long execute burst, or
spilling partial results changes the execution being modeled. Out-of-order
hardware cannot issue a load that has not been submitted. Extra fences can
serialize otherwise numerically equivalent streams.

Share geometry, traversal, buffering and submission-policy interpretation between
paired model and realization where practical. Preserve the selected policy in
plan/provenance. ISA expansion and dependency-preserving reordering are permitted
when their resource and submission behavior is modeled. A changed execution policy
is a modeled choice, not an invisible converter optimization. Native macros keep
their private traffic/control contracts without requiring software-loop expansion.

When investigating a model/realization mismatch or assessing a performance
change, use the relevant checks below. They are diagnostic options, not a
mandatory checklist for every edit or target:

1. Force representative legal candidates, including multi-tile/reduction cases,
   reuse, available buffer depths and competing explicit/native implementations.
2. Compare planned movement with the **actual emitted and submitted stream**:
   windows, padded bytes, command counts, loop scope, buffer generations,
   dependency order, waits and submission bursts. Include runtime-inserted work.
   Check the required partial order and reuse/admission conditions; a harmless
   legal reorder need not match a byte-identical stream.
3. Check actual admission/overlap/stalls through endpoint traces or cycle
   measurements, then assess candidate ranking on held-out cases. Static command
   checks establish structural agreement, not measured utilization.
4. Record hardware, recipe/model, converter/runtime, submission-policy and profile
   identities with command artifacts and timing scope. Recheck affected evidence
   after a policy change. Fix structural mismatches before fitting coefficients.

Use existing plans, buffer graphs, command artifacts and validation reports.
Diagnostic event summaries are not a second executable IR or another allocator.

Target guides own execution mechanisms, model assumptions and validation evidence.
Consult the selected target's guide when evaluating a concrete realization;
shared contracts do not prescribe a universal performance checker.

## Extension policy and minimal examples

Use the [extension catalog](extensions.md) for exact hook signatures, stage
consumers, recipe callbacks and ownership boundaries. Target file layouts should
make those bindings visible; helpers are not independent extension points.

A workload normally needs one file. Assets, preprocessing, calibration, state
handling, custom reference and cases are optional fields, not mandatory files.
The default frontend exports a module and obtains its host reference.

```python
import torch
from voyager_compiler.extensions import WorkloadSpec, registry
from voyager_compiler.pipeline import compile_workload

class Toy(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 1)

workload = WorkloadSpec(
    "toy", lambda options: Toy(),
    lambda model, options: ((torch.ones(4, 16),), {}),
)
result = compile_workload(workload, registry.target("voyager"), "fresh-output")
```

For a target, start with `targets/<name>/target.py`, a physical hardware factory,
a `RuntimeModel`, and registered implementation recipes. Colocate a small
family in `implementations.py`; use an `implementations/` package when it has
multiple substantial families. `describe_target(registry.target(name))` lists
recipes and owned hooks. Installed targets/workloads can use the
`voyager.targets`/`voyager.workloads` entry-point groups. Package discovery imports
`targets/<name>/target.py`; that module registers its `TargetSpec` (typically as
`TARGET`). An installed entry point exports the spec object, not a factory that
the loader will call. Use unique names and register once. A direct spec argument
can be used for scoped experiments without changing the global registry. Registration never
replaces the compiler pipeline or adds central target-name branches.

A new lowering of an existing operation edits the target's recipe/model/builder,
its encoder and evidence. Legality must reject unsupported shapes and semantics.
Declare scratch, native effects, physical layout and completion before freeze.
A genuinely new alias/effect/control capability requires a scoped shared
contract change and a regression proving its behavior. [Roles and handoffs](roles.md)
map these responsibilities to current files. B1–B12 in the design are concerns,
not twelve required files or an instruction to edit every compiler stage.

Generic STE, fake quantization and optional training capture remain available.
Workload/target numerical policy chooses the applicable lifecycle; deployment
support does not imply a target-specific trainer or optimizer exists.

## Implemented target scope

Target-owned [guides](../src/voyager_compiler/targets/README.md) describe bound
extension points, supported forms, numerical/ABI constraints and validation.
Inspect callable registration and relevant evidence before treating any old
scope description as a missing feature. This guide owns shared contracts only.

## Performance characterization and model fitting

This optional target-development workflow is already implemented. Using it or
collecting fresh measurements is not a prerequisite for ordinary compilation,
target registration or shared-framework completion. A target can use an
analytical model and can reuse applicable donor evidence with explicit provenance.
New timings are useful for model fitting, performance debugging or substantiating
claims about changed behavior; estimates remain labeled as estimates.

Target registration can provide `characterize(result, case, config)`, or callers
can supply a measurement adapter to `performance.characterize()`. Each case
compiles normally and the adapter returns a `Measurement` attributed to its
case, implementation family, target, hardware identity, clock and evidence
source. Device, RTL and cycle-accurate simulator timings are accepted; host
replay wall time is rejected as accelerator performance evidence.

`CharacterizationCase(..., held_out=True)` reserves evaluation cases. The fitter
rejects overlapping case identities and requires every fitted family in the
holdout. It fits a nonnegative affine correction and saves measurements,
`profile.json` and `held-out.json`. Pass the resulting profile to
`compile_workload(..., performance_profile=profile)` to use it in shared search.
A recipe-specific profile supersedes an OpModel-family correction instead of
applying both. A profile for another target/hardware identity is rejected.
`Measurement.family` must match the applied correction key: the recipe name for
a recipe-specific correction or the `OpModel` class name for a model-family
correction. Align the measured case and estimated cost boundary; a whole-network
time is not automatically an operation-family training point. The current fitter
reports held-out timing errors, not ranking or selection regret; compare measured
alternatives separately when validating selection.

When characterizing a target, choose cases spanning shapes, memory pressure and
competing ISA families.
Force a recipe by deriving an unregistered test target that retains the chosen
recipe and noncompeting recipes needed elsewhere in the graph. For an isolated
one-operation case, only the chosen recipe may suffice. Pass that target directly
to compilation/characterization; do not re-register its existing name. Confirm
the selected identity so the incumbent winner cannot hide alternatives. Fitting
is infrastructure here, not a claim of tuned hardware performance. The unit test
uses explicitly synthetic timing fixtures; no measured profile ships by default.
That is the evidence state of this integration, not a missing compiler mechanism
or a requirement to collect a new dataset for every target.

## Reference gates and reproducible commands

Run affected shared contracts with the existing environment:

```bash
PYTHONPATH=src:test /home/zhouhua/Research/ML/ml-env/bin/python \
  -m unittest discover -s tests -v
```

Use the selected [target guide](../src/voyager_compiler/targets/README.md) for
pinned workload cases, environment inputs and endpoint-specific commands.
`validation/reference-baselines.json` identifies pinned comparators where
applicable. Preserve required gates and approved comparator replacements; do not
silently replace a failed baseline or reinterpret a retained warning as accuracy.

Use fresh result directories. Record relevant source/configuration identities and
comparison boundaries; do not edit source during a running gate. Reuse evidence
only when affected behavior and configuration are unchanged. Host replay,
artifact compatibility, endpoint correctness and measured performance remain
separate claims. The default Voyager target's [guide](../src/voyager_compiler/targets/voyager/README.md)
contains its pinned 11-case development suite.

## Source provenance

This tree is based on compiler-2 `cef1e7e42b31e1cfcc01a1e4e02cedab11e3e9ef`.
Target guides and dated evidence own donor/source identities and run details.
Preserve donor checkouts, dirty work and applicable source notices.

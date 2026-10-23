<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Voyager compiler architecture

This document describes the shared compiler stages, representations, and
target-backend extension contract implemented by Voyager. It intentionally
does not document one target's operators, arithmetic, memories, schedules, or
external runtime. Those details belong under
[`docs/targets/<target>/`](targets/README.md).

Verify symbols with `rg` before editing. This is an ownership and data-flow
map, not a promise that internal line numbers or helper names remain stable.

## End-to-end flow

```text
PyTorch module
  -> export to an executable FX GraphModule
  -> PT2E annotation and observer/fake-quant insertion
  -> calibration or model-owned QAT
  -> explicit executable reference quantized operators and values
  -> selected-target graph and quantization rewrites
  -> full shape propagation
  -> shared padding, tiling, layout, and graph-grouping passes
  -> selected-target memory rewrites and final operation classification
  -> execution order, StoragePlan, and LocalScheduleResult
  -> physical allocation
  -> canonical compiler artifacts
  -> optional target-owned collateral conversion
  -> optional target-owned endpoint glue
  -> target operation execution at the frozen injection endpoint
  -> external system workload runtime and accuracy, outside Voyager adoption
```

The public orchestration entry points are in
`src/voyager_compiler/__init__.py` and
`src/voyager_compiler/quantize_pt2e.py`:

- `prepare_pt2e()` and `prepare_qat_pt2e()` export when necessary, run shared
  PT2E preparation, and invoke selected-target preparation.
- `finalize_pt2e_calibration()` finalizes target-owned prepared state and
  shared observer qparams.
- `convert_pt2e()` removes preparation-only target state and converts fake
  quantizers into executable reference operators and explicit values.
- `transform()` runs target graph recognition and quantization lowering before
  the shared graph passes and required composite rewrites.
- `compile()` selects the scheduling and memory-planning path, then emits
  canonical artifacts.

Every stage receives the same typed `HardwareConfig`, when one exists. There
is no second target selector hidden in a later pass.

Target adoption establishes that profile during R from the authoritative
target default configuration. The R record ties the external configuration,
permitted overrides, derived-config diff, and typed `HardwareConfig` to the
same machine. W and X reuse that identity unless a new R claim changes it.

### Passes, terminal lowering, and code generation

These terms describe distinct abstraction changes:

- A graph **pass** consumes FX and returns FX. It may canonicalize,
  decompose, legalize, pad, tile, change layout, or fuse while operator and
  tensor semantics remain visible. Shared passes live in `codegen/passes/`.
- **Terminal lowering** changes representation. The optional
  `codegen/lowering/` path creates explicit loops, buffers, loads, stores, and
  copies. Its output is not the operator-level FX graph consumed by ordinary
  semantic passes.
- **Code generation** projects an already transformed, scheduled, and mapped
  representation into compiler artifacts. It may encode prior decisions but
  must not choose a target, schedule, quantization policy, or placement.

`codegen/passes/lowering.py` has a historical name. It owns graph-to-graph
decomposition and legalization, not the terminal bufferized lowering.

The normal scheduled path and the bufferized terminal path are separate
consumers. Support in one does not imply support in the other. The current
bufferized path cannot preserve a `LocalScheduleResult`, so it fails closed
when a selected target or scheduler requires that contract.

## Shared ownership

| Responsibility | Source of truth |
| --- | --- |
| Public pipeline and stage order | `src/voyager_compiler/__init__.py` |
| Quantization schemes and specs | `src/voyager_compiler/quantizer/quantizer.py` |
| Observer state, fake quantization, and STE | `src/voyager_compiler/fake_quantize.py` |
| PT2E preparation and conversion | `src/voyager_compiler/quantize_pt2e.py` |
| Prepared quantization regions | `src/voyager_compiler/quantization_regions.py` |
| Prepared and physical quantization contracts | `src/voyager_compiler/quantization_contracts.py` |
| Generic executable reference math | `src/voyager_compiler/decomposed.py` |
| Hardware schema and validation | `src/voyager_compiler/codegen/hardware.py` |
| Target identity and model provenance | `codegen/targets.py` |
| Backend specification and dispatch | `codegen/target_backends/spec.py`, `codegen/target_backends/__init__.py` |
| One target implementation | `codegen/target_backends/<target>/` |
| Shared graph transformations | `src/voyager_compiler/codegen/passes/` |
| Scheduler-neutral types and dispatch | `src/scheduler/types.py`, `src/scheduler/dispatch.py` |
| Compiler-to-scheduler projection | `src/scheduler/voyager.py` |
| Scheduler engines and providers | `src/scheduler/<engine>/` or a target-owned scheduler package |
| Logical and physical storage identity | `src/voyager_compiler/codegen/storage/` |
| Generic mapping and allocation | `codegen/mapping.py`, `codegen/memory.py` |
| Canonical protobuf schema and emission | `codegen/param.proto`, `codegen/mapping.py` |
| Versioned supplemental artifacts | `codegen/artifacts.py` |
| External collateral projection and optional endpoint glue | `conversion/<target>/` |
| Workload and model policy | `examples/<workload>/`, `test/utils/models/` |

The collateral-conversion responsibility owns this target external boundary.
A work order names the required files explicitly; ownership does not make
endpoint glue mandatory.

`pt2e_utils.py` retains its historical filename but owns target-neutral
PyTorch export and FX helpers. It must remain independent of the PT2E
orchestrator: `quantize_pt2e.py` may import it, never the reverse.

## Target selection and provenance

`HardwareConfig.target` is the public source of compilation-target identity.
Omitting a hardware profile selects the generic path. `codegen/targets.py`
resolves that identity and stores it in `GraphModule.meta`; every later public
stage rejects a different target before mutating the graph.

Target identity has two explicit inventories today:

1. `COMPILATION_TARGETS` declares accepted public target names.
2. `_TARGET_BACKENDS` binds every non-generic name to one immutable
   `TargetBackendSpec`.

The registry validates that the two inventories agree and rejects duplicate
registrations. Registration is currently static Python composition, not
runtime plugin discovery: adding a target requires adding its identity and
importing its `BACKEND_SPEC` into the central registry.

The generic target has no `TargetBackendSpec`. Dispatch returns neutral
defaults and continues through shared behavior. Its source-owned hardware
pipeline lives in
`codegen/target_backends/generic/hardware_pipeline.py` and is returned by
`get_target_hardware_pipeline`; it is not a concrete backend registration.

## Target-backend extension contract

`TargetBackendSpec` combines executable composition with declarative target
facts. Shared orchestration invokes its callbacks and consumes required
execution patterns at fixed lifecycle boundaries. Declarative fields such as
`hardware_pipeline` and `execution_fusion_patterns` are exposed through
accessors; their owning caller decides whether and where to consume them.

### Backend-supplied degrees of freedom

| Binding | Target supplies | Shared compiler retains |
| --- | --- | --- |
| Identity and profile | Target name and a `HardwareConfig` subtype | Resolution, provenance binding, and type validation |
| Operator traits | Immutable operator families, operand roles, padding components, layout and prepared-domain rules | Generic padding, tiling, layout, storage, and mapping algorithms |
| Graph rewrite | Idempotent target pattern recognition and explicit target operators | Transform ordering and shape propagation |
| Quantization lifecycle | Preparation, finalization, cleanup, reference-to-target lowering, and optional recalibration/QAT hooks | PT2E observer construction, reference conversion, and public lifecycle |
| Hardware pipeline | Ordered PyTorch-semantic stages and predicates, optionally derived from a target datapath | Shared matching and datapath projection in `codegen/hardware.py`, plus sequential fusion mechanics |
| Composite graph rewrite | Required target execution patterns, optional pre-fusion graph preparation, and Q/DQ grouping legality | Generic fusion implementation and pass ordering |
| Scheduler binding | Immutable `SchedulerProviderSet`, optional operation classifier, and optional GEMM candidate-legality predicate | Neutral workload creation, engine selection, caching, and result binding |
| Memory rewrite | Explicit target movers and final-FX classification before planning | Execution order, `StorageGraph`, and `StoragePlan` finalization |
| Allocation | A target allocator plus explicit model-wide and operation-scoped validation callbacks | Allocation invocation, storage-plan binding, and artifact orchestration |
| Mapping policy | Capability flags such as aggregate-copy ownership and bufferized support | Generic fallback behavior and fail-closed validation |
| Serialization boundary | Optional artifact declaration, metadata preparation, and validation | Canonical protobuf and supplemental artifact emission |
| Executable helper | Optional target implementation for a shared execution entry such as tiled convolution | Entry validation and shared call contract |

The required callable fields are `apply_graph_rewrites`,
`apply_quantization_rewrites`, `apply_memory_rewrites`,
`prepare_quantization`, `finalize_quantization_calibration`, and
`create_scratchpad_allocator`. The specification also requires the target
identity, hardware type, operator traits, scheduler provider set, and the
`allow_qdq_fusion`, `supports_bufferized`, and
`uses_generic_aggregate_copies` capabilities.

The current optional callbacks are `classify_scheduler_operation`,
`prepare_execution_fusion`,
`scratchpad_validate_end_to_end_operations`,
`scratchpad_has_operation_scope`,
`scratchpad_allocate_operation_scope`,
`scratchpad_validate_operation_scopes`,
`pad_matrix_output_operands`, `prepare_quantization_recalibration`,
`prepare_quantization_dynamic_qat`, `prepare_quantization_qat`,
`refresh_quantization_qat`, `cleanup_quantization_preparation`,
`run_tiled_conv2d`,
`prepare_serialized_artifacts`,
`prepare_serialization_metadata`, and
`validate_serialization_metadata`.

Two declarative fields are easy to conflate:

| Field | Declares | Consumed by |
| --- | --- | --- |
| `hardware_pipeline` | ordered `OpMatcher` stages over ordinary PyTorch semantics the hardware can chain, authored directly or derived from `HardwareConfig.datapath` | `transform`, only when a caller supplies it as patterns with `fuse_operator=True` |
| `execution_fusion_patterns` | stricter required sequences for target-lowered operations | `transform`, even when optional operator fusion is disabled |

Registration alone runs no matcher. A target should derive required linear
execution patterns from its hardware pipeline when the two views coincide.
Final helper topology, qparams, shapes, and ABI fields are validated at the
target lowering/serialization boundary and independently by the converter;
they are not another shared backend-spec inventory.

`codegen/hardware.py` owns the target-neutral `OpMatcher`, operation
name resolution, datapath traversal and validation, and matcher-pattern
derivation. Targets own only their hardware topology, predicates, and declared
matcher inventory; no target specializes the shared functions, so they stay out
of `target_backends`.

Required hooks make one backend usable through the ordinary path. Optional
hooks add only behavior with a real consumer. An absent optional hook means
the target does not customize that boundary; it is not permission for generic
code to infer target behavior from names.

`TargetOperatorTraits` is the declarative half of the interface. Shared
passes consume those facts directly, so a backend can extend recognized
operators, operands, layouts, padding, and prepared-domain transfer without
forking the shared algorithms. `conv2d_tiling_epilogue_operators` identifies
epilogues across which output-channel tiling remains one legal graph rewrite;
`quantization_output_epilogue_operators` identifies activations across which
the selected backend may relocate an output observer;
`execution_fusion_predecessor_inputs` identifies the composite input positions
whose producers may be absorbed into execution fusion. A callback is
appropriate when the target must change FX, construct a target-owned
allocator, or validate target state.

### Invocation order

```text
resolve_compilation_target(HardwareConfig.target)
  -> validate profile type and bind GraphModule provenance

prepare_pt2e / prepare_qat_pt2e
  -> shared PT2E preparation
  -> prepare_quantization

finalize_pt2e_calibration
  -> finalize_quantization_calibration
  -> finalize shared observer qparams

convert_pt2e
  -> finalize calibration
  -> cleanup_quantization_preparation
  -> shared explicit reference conversion

transform
  -> apply_graph_rewrites
  -> apply_quantization_rewrites
  -> propagate complete tensor shapes
  -> shared decomposition, padding, tiling, and layout
  -> prepare_execution_fusion when the selected target provides it
  -> required target composite rewrites from execution_fusion_patterns
  -> optional caller-supplied grouping patterns when fuse_operator is true

run_memory_mapping
  -> apply_memory_rewrites
  -> bind final operation classification
  -> choose execution order and build StorageGraph
  -> let the selected allocator prepare target storage relations
  -> declare canonical nonresident target artifacts
  -> finalize StoragePlan
  -> construct neutral scheduler workloads and schedules
  -> create or invoke the selected allocator
  -> prepare target serialization metadata

gen_code
  -> validate target serialization metadata
  -> emit canonical compiler artifacts

optional conversion/<target>
  -> project canonical artifacts into an external schema

optional endpoint glue inside conversion/<target>
  -> when the endpoint requires integration, consume canonical artifacts or
     converter output
  -> package, stage/link, launch, complete, read back, and compare

target operation execution
  -> inject the accepted artifact at the frozen target endpoint

external system runtime, outside Voyager adoption
  -> may orchestrate accelerator regions with assumed external CPU work
  -> does not create Voyager CPU lowering, launch, or validation work
```

The fixed ordering is part of the contract. A graph rewrite cannot allocate
memory, a scheduler provider cannot mutate FX, and serialization cannot repair
missing qparams, schedules, storage identities, or addresses.

### Representation rules for target extensions

The final target FX graph remains the semantic instruction carrier. A target
may introduce custom operators, explicit physical movers, or composite
execution
regions, but it does not get a second shared execution IR. A target operator
needed during export or reference execution requires executable and FakeTensor
behavior in the target package.

Target values required downstream must remain explicit FX operands or enter a
versioned shared artifact. Metadata is appropriate for transient analysis, not
for numerical state that disappears before its consumer.

`ComputeUnit.accepted_operations` is a coarse resource filter, not a complete
support declaration. `supported` means a scoped target realization matches
semantics under the stage tolerance. Record its applicable constraints and a
named adapter/backend entrypoint, but check entrypoint acceptance, conversion,
execution, and comparison separately. A failed check does not erase support.

R/W/X and injection-point control are owned by the
[adoption model](backend-adoption/adoption-levels.md). In brief, R registers a
target plus a concrete injection point, W binds a workload through a partial-
or full-control point under the fixed one-LSB semantic rule, and X binds that
workload bit-accurately through a full-control point. Full control is a
performance property: Voyager's scheduler and mapper determine each emitted
call's performance-relevant mapping. Tiling, allocation, schedule, placement,
and addresses may be evidence for a concrete target, but are not a universal
field checklist.

`unsupported` is the accelerator-module umbrella for operations assumed to
execute on an external CPU. Voyager does not implement or prove CPU lowering,
execution, transfers, or launch behavior for those operations. They remain in
semantic workload accounting but contribute zero accelerator coverage.
Scheduling or mapping failure records an accelerator-module gap; it does not
create CPU implementation work.

Per-operation compilation may retain an explicitly classified unsupported
call as an external boundary and serialize it with `op: "unsupported"`.
End-to-end scratchpad compilation cannot contain such a boundary and rejects
the graph after target classification, before executable allocation finishes.

A semantic region spanning several physical units is either explicit target FX
operations or a region declared matchable by the target's `hardware_pipeline`.
That pipeline is a capability declaration, not a graph representation: when
matching is enabled, the existing grouping rewrite records a successful match
as `fused_op`, and target-owned matching still owns the DAG topology and
predicates a linear matcher cannot express. The adapter or native stack may
lower a bound region into one or more accelerator calls.

Do not use a converter's capture signature as the hardware pipeline, add
another shared execution representation, or infer one physical instruction
from graph grouping.

When the current interface is insufficient, add a shared hook or trait only
for a backend-neutral fact with a real consumer. Do not add target-named
branches, target address fields, or external-runtime taxonomy to shared code.

## Quantization representation

Voyager preserves two conceptual quantized levels:

```text
reference quantized FX
  explicit payloads and qparams, executable with PyTorch

selected-target FX
  shared operators retained where valid, plus only required target operators,
  operands, and execution regions
```

`QuantizationSpec` describes requested edge semantics. During PTQ or QAT,
fake-quant modules own mutable observer state and return reconstructed floating
values so ordinary PyTorch operators remain executable. `convert_pt2e()`
replaces them with stateless explicit operators and qparams. Target lowering
then recognizes and rewrites only the selected target's supported forms.

For PTQ, that converted shared PT2E/Q-DQ graph is the default implementation,
not scaffolding that every backend replaces. A target reuses the shared
observers, qparam finalization, Q/DQ operators, quantization contracts, and
decomposed executable math whenever they express its required semantics.
During W, a target-specific semantic change is justified only by a recorded
comparison showing that this shared executable path exceeds one output-code
LSB. A target-only encoded component, ABI form, intrinsic operand, or physical
execution region may justify representation or legality projection without
changing the arithmetic. Exact fixed-point rules and hardware behavior belong
to X. An operation whose shared W form is already correct remains semantically
unchanged. Do not copy shared PT2E or Q/DQ logic into a backend merely to make
it target-owned.

Encoded or hierarchical values may contain a payload plus several scales,
mantissas, or codebook components. Reshape, slice, split, concat, add, and
fan-out require a domain-transfer rule or an explicit requantization. Do not
reconstruct required components from a lossy float or a later graph walk.

Prepared-domain rules may use `PreparedDomainRelation.INPUT_JOIN` to require
the direct input edges of a join to share one quantization domain without
forcing a multi-user producer's other edges into that domain.
`QUANTIZATION_INPUT_REGION` records that input-only region on the join; it is
distinct from the operation/output `QUANTIZATION_REGION`. A target lowering
must consume the explicit direct-edge Q/DQ result rather than infer an
upstream producer scale.

The shared QAT API owns PT2E preparation and fake-quant execution. A
model-owned training driver owns the optimizer, EMA, phase schedule, observer
refresh, checkpoint selection, and workload metric. Optional target QAT hooks
only provide target-exact fake-forward behavior and refresh target-derived
operands; they do not make training policy part of the compiler.

Target-specific arithmetic, formats, calibration policy, and QAT recipes
belong in the selected target and workload documentation.

## Scheduler subsystem

Scheduler engines are sibling packages, not compiler graph passes. The shared
boundary is:

```text
final classified FX operation
  -> typed operand descriptors
  -> OperationWorkload or GemmWorkload
  -> scheduler/dispatch.py with the selected SchedulerProviderSet
  -> LocalScheduleResult
  -> result bound to the same final FX operation
```

`scheduler/types.py` owns immutable workload, provider, mapping, and result
types. `scheduler/voyager.py` is the compiler integration adapter and the only
scheduler module that projects a classified FX operation into a neutral
workload. `scheduler/dispatch.py` chooses an engine inside the provider set;
it does not receive a target ID or implement target policy.

A target binds providers that understand its frozen operation classification
or physical GEMM projection. Providers choose loop schedules and estimates;
they do not change the graph, allocate tensors, or redefine hardware. A
backend may support several engines behind one provider set. The optional
`SchedulerProviderSet.gemm_tiling_candidate_fits` predicate rejects a candidate
that cannot be represented by the target endpoint; it is a legality filter,
not a second cost model or a graph-rewrite hook.

`HardwareConfig` remains the resource source of truth. Compute units,
memories, unroll dimensions, accepted accesses, capacity, buffering, and
directional bandwidth are typed shared structures. The shared methods
`matrix_input_resident_channels()` and `matrix_weight_physical_channels()`
provide target-overridable physical channel extents to tiling and allocation;
their defaults preserve the logical channel counts. Target subclasses may add
validated target facts, but shared scheduler result types do not grow one
field per target.

`MappingPoint` is the canonical loop mapping. Provider-owned named factors
may project that mapping into an exact external interface, but they remain
immutable facts attached to `LocalScheduleResult`, not a second schedule
derived during conversion.

## Storage, mapping, and allocation

The representation flow is:

```text
final FX values and operations
  -> StorageGraph relations
  -> StoragePlan aliases, views, copies, components, and liveness
  -> allocator banks, placements, and addresses
  -> canonical serialized memory references
```

`codegen/storage/` is the source of truth between FX and allocation:

- `ValueRef(node, path)` identifies each logical tensor output.
- `LogicalTensor` records shape, dtype, inputs, and users without deciding
  storage.
- `PhysicalTensorId` identifies a backing within one storage plane, not an
  address.
- `StorageRelation` distinguishes materialization, alias, view, copy,
  direct-write, and packed-component behavior.
- `StoragePlan` resolves relations and derives physical live intervals.
- `PhysicalTensorComponent` and `PhysicalPayloadRef` identify exact resident
  bytes and their realized allocation.
- `SerializedTensorArtifact` identifies a canonical, intentionally
  nonresident value without pretending it has a runtime address.

The target memory-rewrite hook runs before execution order, scheduling,
storage finalization, and placement. It may insert explicit executable movers
and bind final execution facts. After `StorageGraph` construction, an optional
allocator graph-preparation step may add target storage relations before the
shared plan is finalized. The selected allocator then consumes the same plan
as generic mapping. It must not maintain a second alias or liveness graph.

The shared `ScratchpadAllocator` protocol performs operator-trait, storage-plan,
and access-requirement binding; allocation and release; segment lookup and
views; and snapshots. Mapping also conditionally consumes target extensions
for graph preparation, operation constraints, target tensor sizing, split
views, and final plan validation. A target allocator may implement those
extensions, but they consume the shared graph and `StoragePlan`; they do not
replace either representation.

Operation-scoped allocation is an optional, explicitly registered backend
extension used by per-operation compilation. The backend specification must
provide its has, allocate, and validate callbacks as an all-or-none group.
They operate on the selected allocator and the shared storage plan; shared
mapping does not discover allocator methods by name. These callbacks change
allocation scope, not layout, alias, liveness, or address ownership.

DRAM and accelerator scratchpad are distinct storage planes even when they
contain the same logical value. Equal numerical addresses after reuse do not
imply equal physical identity. Allocation and validation use exact
`PhysicalTensorId` and `ValueRef` keys; a single-output node view is only a
compatibility convenience.

The finalized plan is attached to `GraphModule.meta["storage_plan"]` for its
immediate consumers and emitted as versioned `storage_plan.json` for
inspection. `model.txt` contains realized executable memory references.

## Serialization and external conversion

Canonical compiler emission records decisions already represented by the
graph, hardware profile, schedule result, storage plan, and allocator. The
main artifacts are:

- `model.txt` for operations, operands, qparams, schedules, and addresses;
- tensor payload files;
- `hardware_config.json` for the resolved typed hardware profile;
- `storage_plan.json` for logical and physical storage relations; and
- `performance_estimates.json` for backend-neutral local estimates; and
- `supplemental_artifacts.json` for versioned, namespaced payloads that a
  target derives from finalized compiler facts for later consumers.

The shared supplemental-artifact registry owns deterministic namespacing,
replacement, removal, and one schema-versioned manifest. Each target owns its
nested payload schema and meaning. Registering a payload does not promote its
fields into shared scheduling, storage, or mapping semantics.

These existing representations are intentionally extensible. `StoragePlan`
owns resident logical-to-physical relations, identity, size, layout, and
liveness; the selected allocator owns placement and addresses. A
`PhysicalPayloadRef` names a realized resident payload and its placement, while
`SerializedTensorArtifact` declares a canonical file-backed nonresident value.
A target may add richer, versioned facts under its namespace in
`supplemental_artifacts.json`. `model.txt` remains a typed compiler schema:
typed `OpOverload.kwargs` may extend compiler facts, but it must not become an
arbitrary bag of runtime metadata. A converter or endpoint adapter fails closed
when required dtype, layout, qparams, or addressing facts are absent.

Target serialization hooks may declare canonical nonresident values, prepare
target metadata from completed mapping, register target supplemental evidence,
and fail closed on missing facts. They must not project an external runtime's
execution kind, descriptor, composite routing, lane encoding, or launch policy
into compiler semantics.

`conversion/<target>/` is a later consumer. It may deterministically project
canonical artifacts into one external schema, but converter acceptance is a
separate check from semantic support. In the base case, only collaterals are
needed. Conversion is unnecessary when the authoritative endpoint accepts
canonical Voyager artifacts directly; otherwise it generates only the required
collateral and no new runtime code.

An integrated endpoint may additionally require packaging, entrypoint
generation, staging or linking, launch, completion detection, or readback glue
inside `conversion/<target>/**`. The collateral-conversion responsibility owns
this single source boundary. The normative
[injection contract](backend-adoption/contracts/execution.md#injection-contract)
states which artifacts and glue the endpoint requires. Initial adoption still
requires one `model.txt` operation or composite to translate, execute, and
complete its output comparison; it does not require a model-wide
scratchpad-only path or external hybrid runtime. Voyager adoption never builds
the CPU side of that runtime. Compilation and conversion alone prove none of
target execution, numerical equivalence, or workload accuracy.

Execution lowering records `accelerator`, `unsupported`, or `unverified`
assignment before optional scheduling. Freeze final accelerator assignment
only after final FX/`model.txt` translates into legal executable backend calls.
Unsupported operations are assumed external CPU work without requiring a
Voyager CPU executor. Later scheduling, mapping, conversion, or runtime failure
remains an accelerator-module gap, does not change assignment, and creates no
CPU-lowering obligation.

## Dependency rules

- Model policy may call generic compiler APIs; generic compiler modules do not
  import model helpers.
- Shared passes call target dispatch and consume traits; they do not import a
  concrete target package.
- Target compiler packages may import shared compiler contracts and their
  target-owned scheduler providers.
- Scheduler-neutral types remain usable without importing Voyager. Scheduler
  providers do not mutate FX or allocate storage.
- Mapping may consume schedules and produce allocations. Serialization may
  consume both but derives neither.
- External conversion consumes canonical compiler artifacts and does not
  become compiler execution lowering. It encodes finalized facts and never
  invents or repairs computation or qparams.
- Optional endpoint adaptation under `conversion/<target>/**` consumes
  canonical artifacts or converter output required by the frozen injection
  contract. Endpoint glue does not invent layouts, qparams, schedules, or
  hardware capacity.
- A target representation needs executable semantics, FakeTensor behavior
  where export requires it, and an end-to-end target compilation lane.

## Target documentation boundary

Each target documents its implementation and evidence under
`docs/targets/<target>/`:

- `README.md` routes its hardware profile, operators, lowering, scheduling,
  memory, serialization, conversion, and workload-specific documentation;
- `operations.yaml` indexes named support and authoritative implementation or
  adapter locators when maintained;
- `ownership.md` records legacy files that combine responsibilities; and
- `regression.md` defines concrete end-to-end validation lanes.

The shared architecture may name an extension point but must not accumulate
one target's operator list, numeric format, bank geometry, scheduler formula,
workload recipe, converter schema, or runtime command.

## Adoption-facing invariants

The adoption lifecycle and its rules are owned by
[backend-adoption/adoption-levels.md](backend-adoption/adoption-levels.md) and
are not restated here. What this document fixes are the *representation*
boundaries that adoption must not violate:

- Final target FX and `model.txt` are the semantic source of truth for ordinary
  operations, composites, and explicit multi-call graph rewrites. Do not add a
  neutral call-plan or tile-plan IR merely for adoption coordination.
- Candidate and final assignment are evidence-lifecycle states, not persistent
  compiler representations. Final classification remains bound in the normal
  target lowering path.
- Graph tiling represents only the split needed by a concrete operation.
  Halos, tails, or partial sums are operator-specific semantics, not a missing
  universal schema. `LocalScheduleResult` remains local scheduling inside one
  already-formed call.
- Target registration is static and requires editing the target inventory and
  backend registry. This is explicit and validated, but not dynamic discovery.
- The target allocator has a shared base plus several conditionally consumed
  extension methods rather than one fully typed immutable protocol. Keep new
  allocator behavior scoped to an actual mapping consumer.
- `codegen/mapping.py` remains large and contains both mapping orchestration
  and canonical serialization entry points. Do not grow it with target policy;
  extract only around a shared responsibility with target-neutral tests.
- The bufferized terminal path cannot preserve the ordinary scheduled target
  contract and therefore remains a separately validated consumer.

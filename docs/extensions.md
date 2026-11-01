# Extension points and decision ownership

Use this catalog to answer **where a change enters the compiler**. The
[role readers](roles.md) describe who owns the work; the
[compilation reference](compilation.md) describes stage invariants. A Python file
is not an extension point: the public binding and its consumer are the contract.
Target guides map their actual files to these bindings.

## Workload boundary

`WorkloadSpec` lives in `extensions.py`; `compiler/prepare.py` consumes it.

| Binding | Signature/result | Owns |
| --- | --- | --- |
| `name` | Stable workload identity | Registry and provenance |
| `build` | `(options) → model` | Construct/load the workload |
| `example_inputs` | `(model, options) → (args, kwargs)` | Reproducible inputs and state |
| `assets` | `(path, expected_checksum)` entries | Asset identity checked before construction |
| `prepare` (optional) | `(model, args, kwargs, options) → PreparedWorkload` | Custom frontend; replaces default preprocessing/reference/export, not the compiler |
| `preprocess` (optional) | `(args, kwargs, options) → (args, kwargs)` | Default frontend's input preparation |
| `reference` (optional) | `(model, args, kwargs) → outputs` | Reference semantics; default is the model call |
| `calibration` (optional) | `(prepared, options) → calibration iterable` | Supplies samples when the caller has not supplied them; consumed by `compile_workload` |
| `cases` | Workload-owned case metadata | Not an automatically executed compiler callback |

Use `prepare` for a custom frontend. For an optional input representation or
operation rewrite on the exported graph, use the separate
[input preprocessing boundary](#input-preprocessing). Existing custom frontends
remain valid; do not apply the same rewrite in both places.

Pair-specific policy can stay with the workload when inseparable from its
reference and inputs. Do not move target-independent model preparation into a
target emitter.

## Target boundaries

`TargetSpec` is assembly. Its callbacks execute at the following boundaries.
Encoding is mandatory; its target callback is optional. Emission is independently
runnable after the bundle is saved; see [encoding and emission](roles/emission.md)
for the saved-file contract and command.

| Binding | Signature/result | Shared consumer and boundary |
| --- | --- | --- |
| `name`, `hardware` | `hardware() → AcceleratorConfig` | Registry/pipeline resolve identity and physical capabilities; configuration is not a cost correction |
| `operators` | `(operator, OperatorSemantics)` entries | Registry/classifier consume compute traits; Torch schemas own aliases/writes. This is not a full layout/padding/operand-role trait inventory |
| `input_preprocessing` | `(prepared, config, options) → PreparedWorkload` | Optional exported-graph/input rewrite before quantization; caller option overrides the target default |
| `quantization` | Optional `QuantizationPolicy` | `compiler/quantize.py` dispatches shared PT2E or a target override, described below |
| `legalize` | `(prepared, config, options)`, in-place | Pipeline before common normalization; explicit graph rewrites and supported-form legalization |
| `normalization` | Normalization option pairs | Shared normalization consumes policy/patterns; this is data, not another pass manager |
| `lower` | `(prepared, config, options)`, in-place | After shared normalization, before search: lower supported normalized tensor operations into custom target operations; preserve qparams/layout and reference semantics |
| `model` | `(config, accumulate_fp32) → RuntimeModel` | Shared `Tiler` uses operation fit/cost models; whole-operation recipes may instead provide `estimate` |
| `implementations` | `ImplementationSpec` entries | Shared planner compares legal realizations and bufferizes the winner; details below |
| `refine` | `(program, config)`, in-place | After bufferization, before final allocation. Own target execution/storage declarations and permitted refinement; do not allocate final addresses |
| `encode` | `(ir: EncodedIR, program, config) → EncodedIR` | Project frozen decisions into common `model`, target-owned `target_data` and optional array `payloads`; shared code saves and checksums the bundle |
| `emit` | `(bundle: SavedBundle, output_directory)` | Read saved IR/records/payloads to generate target files; no live graph, tiling, ISA selection or new allocation |
| `runtime` | `(result, args, kwargs) → outputs` | `runtime.run`; optional endpoint execution. Absence means host replay, not an unimplemented compiler |
| `characterize` | `(result, case, config) → Measurement` | `performance.characterize`; optional attributable timing, not a mandatory compilation stage |

Callback execution follows the [pipeline order](compilation.md#the-compilation-flow);
registry declarations such as operators/hardware are not sequential passes. A large
callback may have private helpers; those helpers do not create new lifecycle boundaries.
A missing optional callback means the target does not specialize that boundary.

## Operation semantics

`TargetSpec.operators` registers operation classifications; schemas and executable
implementations define computation, signatures, aliasing and writes. Classification
is not a complete arithmetic/effect definition or a sequential compiler pass.
The [operation-semantics role](roles/semantics.md) owns these definitions;
quantization and lowering consume them.

<a id="quantization-and-custom-operation-lowering"></a>

## Input preprocessing

`TargetSpec.input_preprocessing(prepared, config, options) -> PreparedWorkload`
selects an optional input/graph rewrite to improve mapping and hardware utilization
before quantization. The caller's `options["input_preprocessing"]` overrides the
target default; `None` disables it. Compare alternatives through separate compilations.

Calibration samples are acquired before this hook and executed afterward. Return
an executable graph, matching examples and serializable
`provenance["input_preprocessing"]`. If inputs change, replace
`options["calibration"]` consistently without mutating caller samples; preserve
replayability for multi-pass calibration. The driver retains asset provenance and
refreshes the working reference. `result.original_reference` preserves the original
frontend outputs; a custom frontend must supply that reference.

Constant preparation remains separate. Runtime expansion is constrained by live
memory and execution cost. Host preparation requires the agreed input/timing
boundary: this hook does not install a runtime wrapper, and `result.args` contains
only the prepared example. Keep other dynamic preparation executable in the graph.

See [implementation guidance](roles/input-preprocessing.md) and
[utilization diagnosis](compilation/performance-models.md#required-diagnosis-when-utilization-or-selected-performance-is-poor).

## Quantization

`TargetSpec.quantization` binds the [quantization role](roles/quantization.md):
formats, scale selection, calibration and conversion. This is separate from
operation definitions, shared layout selection and custom-operation lowering.

`QuantizationPolicy` configures the default PT2E flow or overrides **only the
quantization stage**. Both return a `PreparedWorkload` with an executable graph
and explicit numerical representation. The shared driver refreshes its reference;
normalization, search, allocation and emission remain shared. Without a caller
quantizer, policy quantizer or stage override, quantization leaves the graph
unchanged; registering a target does not automatically select a numerical format.
The refreshed reference is the converted graph's output. Preserve a separate
pre-quantization reference when evaluating quantization accuracy; equality to the
converted graph only checks faithful execution of that numerical representation.

`run(prepared, config, options)` receives the same resolved hardware configuration
as later compilation. No additional context IR or hardware factory is involved.
The workload owns sample preprocessing and saved-profile assets; the target owns
format conversion and any hardware-aware recalibration. A binding may populate
its target recipe in the invocation's mutable options during `prepare`; these
options are a per-call copy, and executable recipe objects are not provenance.
Return only serializable format/profile evidence in `prepared.provenance`.

An override must record `provenance["quantization"]`. The shared converter is not
called after it. Explicit `options["quantizer"] = None` bypasses either flow;
an explicit non-None quantizer must be honored by the override. For an already
quantized graph, disable preparation explicitly (or let a target policy recognize
its own input representation).

| Policy field | Consumer and contract |
| --- | --- |
| `run(prepared, config, options)` | Optional stage override returning `PreparedWorkload`; replaces the default quantizer/prepare/calibrate/finalize/convert sequence, not later compiler stages |
| `quantizer(prepared, options)` | Default quantizer factory; caller `options["quantizer"]` wins, including explicit `None` to disable preparation |
| `prepare(graph, options)` | Optional in-place observer/domain work after shared `prepare_pt2e`, before calibration |
| `finalize(graph, options)` | Optional in-place scale finalization/cleanup after calibration, before shared `convert_pt2e` |
| `bias` | Default conversion accumulator/bias dtype; explicit `options["bias"]` wins |
| `example_calibration` | Opt into the supplied example when no calibration iterable is provided; an empty iterable rejects |
| `integer_rounding` | Opt into direct FP32 nearest-even integer conversion; the default preserves existing lookup-table behavior |

The shared quantizer retains per-module name/type, operator and occurrence-order
configuration. An unsupported target format must reject, not silently replace the
user's specification. `quantization/regions.py` adapts declarative domain-transfer
rules from `ir/graph.py` to PT2E shared observers; existing explicit annotations
remain boundaries. `quantization/calibration.py` contains optional static-observer
helpers. The training observer's delayed updates remain unchanged.

Quantization conversion produces explicit Q/DQ boundaries or target operations
and compound values with declared numerical formats. Value-preserving byte packing
needs no recalibration. Changing scale precision, shared-scale groups or arithmetic
can change values; model that change explicitly and recalibrate only when the
recipe calls for it. A second calibration pass does not establish bit accuracy:
the reference arithmetic and emitted implementation must agree. Repeated passes
need a replayable sample source; a saved deployment profile may bypass fitting.
No common ordered-pass DSL or format-handler registry is required. Domain sharing chooses
which values use one observer; Q/DQ folding is a later graph rewrite and must not
erase a required rounding boundary. Static constant preparation folds immutable
transforms under the storage budget. These are separate mechanisms.

## Graph legalization

`TargetSpec.legalize(prepared, config, options)` runs before shared normalization.
Use it for supported-form rewrites that must precede layout and fusion. Preserve
logical outputs and the prepared numerical representation. See the
[legalization role](roles/legalization.md).

## Normalization policy

`TargetSpec.normalization` supplies option pairs, not a second pass manager. Shared
normalization chooses tensor/GEMM layout, prepares constants and applies legal
fusion. Broad fusion patterns live in the target package and are bound here;
frontends and tests consume the same table. Do not derive fusion policy from
hardware topology or ISA-pipeline declarations. Instruction-level fusion inside
builders remains separate. A concrete hardware-dependent legality condition
belongs to its rule or implementation candidate; there is no general
configuration-aware pattern resolver. `pad_to_hardware=False` lets a target with
native partial tiles skip global alignment padding without falsifying physical
hardware dimensions. See the [normalization role](roles/normalization.md) and
[shared transformation catalog](compilation/rewrites.md).

## Custom-operation lowering

`TargetSpec.lower(prepared, config, options)` consumes the normalized graph and
its chosen scales/layouts, before search. It maps tensor operations to supported
target operations; it does not recalibrate, choose new tilings or implement
another layout optimizer. This graph hook is distinct from
`ImplementationSpec.build`, which constructs bufferized execution for a candidate.
See the [custom-operation lowering role](roles/lowering.md).

## Implementation, scheduling and buffer lowering

An `ImplementationSpec` keeps each realizable alternative's contracts together.

| Binding | Signature/result | Owns |
| --- | --- | --- |
| `name` | Stable unique recipe identity within the target | Selection records and optional recipe-specific profile keys |
| `description` | Human-readable recipe description | Discovery, not legality |
| `cache_by_shape` | Defaults to `False` | Opt into structural/shape/dtype built-graph sharing only when candidate behavior is fully represented by that cache signature |
| `matches` | `(node, anchor) → bool` | Which semantic operation this recipe can implement |
| `candidates` | `(node, context) → parameter dictionaries` | Enumerate legal tile/algorithm/ISA alternatives; no emission or address assignment |
| `model` (optional) | `(config, accumulate_fp32, parameters) → RuntimeModel` | Override the target model for this candidate |
| `build` | `(node, context, parameters) → FX GraphModule` | Realize the candidate in buffers, copies, loops and native invocations |
| `estimate` (optional) | `(node, graph, context, parameters) → Cost` | Whole-operation cost; otherwise use the builder's selected `TileResult.cost` |
| `validate` (optional) | `(node, graph, context, parameters)` | Post-build candidate check; reject unsupported candidates with `CandidateRejected` |

`TargetSpec.model` is required and is instantiated before recipe evaluation.
`ImplementationSpec.model` overrides it for a trial; `estimate` supplies a
whole-region cost. A target using only self-contained native recipes may have a
minimal `RuntimeModel.prepare` returning `None`, provided its recipes implement
and cost all supported forms. `None` means no local tiling model, not proof of
operation support or permission to invent a zero cost.

Keep `cache_by_shape=False` for new recipes unless matching, parameters, legality
and generated code are determined by the existing structural/shape/dtype key.
Tensor-content-dependent specialization is not covered merely because shapes
match. This is distinct from an `OpModel.key` used by mapping search.

The shared planner selects; the candidate callback proposes; the model estimates;
the builder constructs. A builder may request shared tile search through its
context, but emission cannot repeat that search. Competing implementations need
finite comparable costs. The current shared engines are hierarchical matrix,
dimension-wise grid and attention search; recipe registration does not provide
an independent engine-plugin registry or a complete typed access-event/energy API.

For a fixed region, `build_sequential_region` composes component recipes and
`estimate_sequential_region` prices their sequential execution. The planner retains
its `region_choices` metadata in `ImplementationChoice.components`; the parent
cost already includes those choices. `NoLegalMapping` is an expected
`CandidateRejected`, while unexpected implementation errors propagate. See
[algorithm comparison](compilation/algorithm-search.md) for the boundary, cost
rules and opt-in attention example. This does not search overlapping graph covers.

## Execution and comparison are explicit

Pass the target to `runtime.run` to use its registered runtime hook. The helper
does not infer a target from the compiled result; omitting it runs host replay.
Comparison is also opt-in through `reference`, and tolerances must match the
chosen numerical contract. Its returned `seconds` is host elapsed time around the
call, including a target hook if used; it is not automatically accelerator cycles.
Use characterization for attributable accelerator timing.

```python
from voyager_compiler.runtime import run
outputs, execution = run(
    result, target=target, reference=expected_outputs,
    rtol=chosen_rtol, atol=chosen_atol,
)
```

`expected_outputs` must match the chosen ABI/state and numerical boundary. The
converted graph reference and a pre-quantization accuracy reference answer
different questions. For stateful workloads, keep independent initial state for
reference and target runs; preparation/reference evaluation may mutate inputs.

## Storage is declared; allocation stays shared

Targets express memory requirements through builders/refinement and canonical
`BufferRequirement`, `BufferComponent`, `BankLayout`, `physical_shape`, memory
bindings and `TargetInvoke` ownership. The shared allocator validates capacities,
physical banks, conflicts and lifetimes, then fixes addresses. There is deliberately
no target hook replacing the final allocator. A missing representable physical
constraint requires a shared-contract change, not a hidden target allocator.

A native call may own private storage while the compiler owns its command order.
Declare this before freeze. No general stage may infer ownership from a target's
name. See [storage and IR decisions](compilation.md#ir-ownership-and-why-these-records-differ).

## File organization should expose these responsibilities

A target's registration module should let a reader trace each bound callback to
its owner. The useful separation is:

```text
registration      connects public bindings; no pass implementation
hardware          physical facts and capabilities
semantics         operation meanings, signatures and legality
quantization      default policy or target format/calibration override
scheduling        candidates, capacity estimates and costs
graph lowering    normalized graph to target operations (TargetSpec.lower)
buffer building   candidate buffers/control (ImplementationSpec.build)
refinement        post-build target declarations (TargetSpec.refine)
emission          frozen-program conversion; private command/bit encoders
runtime           endpoint submission and output/state
characterization  attributable measurement for offline fitting
```

These are responsibilities, not mandatory files. Combine small related
owners; use a package when one responsibility has substantial internals. An ISA
bit-packing helper belongs inside emission unless it has another real consumer.
A replay-image writer is collateral preparation, not a bound runtime hook merely
because its filename says `runtime.py`. Do not add forwarding files to preserve
accidental internal import paths after a refactor.

Keep dependency direction visible: quantization preparation, scheduling and emission
can read semantic facts; lowering uses the candidate's resource contract; emission
must not import the search owner to reconstruct decisions. Read selected decisions
from the frozen plan. A private expansion can unpack instructions for a selected
tile but must not return a newly selected tile or schedule.

## Discover actual bindings

```python
from voyager_compiler.extensions import describe_target, registry
info = describe_target(registry.target("voyager"))
print(info["bindings"])
print(info["implementations"][0]["bindings"])
```

The description reports bound callable paths and explicit `None` for absent
optional hooks. It is navigation, not proof of operation support or endpoint
execution. Follow the selected [target guide](../src/voyager_compiler/targets/README.md)
for supported forms and evidence. Historical extension comparisons belong in dated
reports; they must not become hidden requirements for every target.

### Fine-grained reuse of matrix search

`MatrixSpace` owns target/recipe architecture and schedule constraints, with an
optional complete-mapping `mapping_filter`. `search_matrix(op, layer, ...)` reuses
Interstellar and returns `TileResult` for a custom builder. The standard FX/grid
adapter is a separate consumer with its own reduction-order limitation. See
[construction and limits](compilation/performance-models.md#target-owned-matrix-search-adapters).
Neither a new target-name switch nor a scheduler registry is needed for this route.

Optional exported-graph/input transformations use `TargetSpec.input_preprocessing`;
custom frontend construction remains with `WorkloadSpec.prepare`/`preprocess`; see [input rewrites](compilation/rewrites.md#input-boundary-rewrites).

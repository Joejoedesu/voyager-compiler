<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# R/W/X target-adoption guide template

Copy this file into a new experiment goal as `guide.md` or `guide.txt`.
Replace every `{{PLACEHOLDER}}` before starting. Use `not permitted` or
`not applicable` when that is the intended value.

This is a practical run guide. It tells an agent where to work, how Voyager
output enters the target stack, which operations to attempt, and how to
progress from operation registration to semantic workload binding and then
exact target realization.

## 1. Required inputs

### Repositories, workspace, and environment

| Input | Value |
| --- | --- |
| Goal directory | `{{GOAL_DIRECTORY}}` |
| Base Voyager repository | `{{BASE_VOYAGER_REPO}}` |
| Base Voyager revision | `{{BASE_VOYAGER_REVISION}}` |
| New writable Voyager workspace | `{{NEW_VOYAGER_WORKSPACE}}` |
| Target hardware repository | `{{TARGET_HARDWARE_REPO}}` |
| Target hardware revision | `{{TARGET_HARDWARE_REVISION}}` |
| New writable target workspace | `{{NEW_TARGET_WORKSPACE}}` |
| Separate conda environment permitted | `{{YES_OR_NO}}` |
| Existing environment, when a new one is not permitted | `{{ENV_PATH}}` |
| New environment path/name, when permitted | `{{NEW_ENV_PATH_OR_NAME}}` |
| Dependency-install policy | `{{DEPENDENCY_POLICY}}` |

The base repositories are foundations, not workspaces. Create the two new
writable workspaces above. Do not modify, clean, or build inside either base
repository.

If a separate conda environment is permitted, create it only at the named
location and record its package state. If it is not permitted, use the named
existing environment without changing it unless the guide explicitly allows
that change.

### Target configuration

| Input | Value |
| --- | --- |
| Target name | `{{TARGET_NAME}}` |
| Target default configuration | `{{TARGET_DEFAULT_CONFIG}}` |
| Derived run configuration | `{{DERIVED_RUN_CONFIG}}` |
| Permitted configuration overrides | `{{PERMITTED_OVERRIDES}}` |
| Prohibited hardware changes | `{{PROHIBITED_CHANGES}}` |

Construct the run configuration from the target's default configuration.
Copy or resolve that configuration into the new target workspace, then make
only the overrides listed above. Record a machine-readable or textual diff.
Do not silently enlarge memories, arrays, queues, or other hardware resources
to make an operation fit.

### Workload and injection boundary

| Input | Value |
| --- | --- |
| Workload/model | `{{WORKLOAD}}` |
| Model weights and revision/hash | `{{WEIGHTS_IDENTITY}}` |
| Example/calibration inputs | `{{INPUT_IDENTITY}}` |
| Requested R operation kinds | `{{R_OPERATION_KIND_LIST}}` |
| R semantic reference and tolerance | `{{R_REFERENCE_AND_TOLERANCE}}` |
| Voyager output artifact | `{{MODEL_TXT_OR_OTHER_ARTIFACT}}` |
| Target adapter or converter | `{{ADAPTER_ENTRY_POINT}}` |
| Target execution command/API | `{{TARGET_EXECUTION_ENTRY_POINT}}` |
| Injection point | `{{EXACT_INJECTION_POINT}}` |
| Injection control class and evidence | `{{PARTIAL_OR_FULL_CONTROL_AND_EVIDENCE}}` |
| Target completion/readback point | `{{COMPLETION_AND_READBACK_POINT}}` |
| W semantic reference and output-code interpretation | `{{REFERENCE_AND_TOLERANCE}}` (`rtol=0`, `atol=1` output-code LSB is fixed) |
| Candidate quantization schemes | `{{QUANTIZATION_SCHEME_IDS}}` |
| W performance-estimate inputs | `{{COSTS_CALL_LATENCY_AND_PROVENANCE}}` |
| X packaged-stream provenance to the same W reference | `{{PACKAGED_STREAM_PROVENANCE_OR_NOT_APPLICABLE}}` |
| X full-control scope | `{{X_SCOPE_OR_NOT_APPLICABLE}}` |

Describe the injection point concretely. Name the Voyager file or in-memory
object consumed, the function or command that consumes it, the converted
collateral it creates, and the first target component that reads that
collateral. Classify the point as partial-control or full-control using the
[performance-based definition](adoption-levels.md#injection-points-and-control),
and record the target-specific evidence. Do not say only “run the simulator”
or infer full control from the presence of one schedule or allocation field.

A **target adapter** is the general boundary between Voyager and the target
stack. It may include a native API, converter, packaging, launch, or readback
components. A **converter** is the specific adapter component that projects
canonical Voyager results into collateral accepted by the injection point. A
workload adapter is different: it owns model, data, PTQ, and comparison policy.

## 2. Rules for the whole run

1. Record the source revisions, dirty state, environment, target configuration,
   commands, and output locations before doing implementation work.
2. Read the new Voyager workspace's `AGENTS.md` and `CLAUDE.md`.
3. Use final target FX and canonical `model.txt` as Voyager's semantic
   carrier. Generate them through the ordinary compiler path; do not replace
   them with hand-authored target collateral.
4. Follow one bounded work order per owned output and role. This guide supplies
   campaign parameters, stage tolerances, and workflow order; it does not
   broaden a work order's writable paths or ownership.
5. Use existing executable shared PyTorch and Voyager semantics as the W
   baseline before adding target code. General semantics belong in shared
   lowering. Target-specific
   recognition, composites, and multi-call rewrites belong in the target
   backend. Canonical serialization emits finalized facts; the target converter
   projects those facts into target packing, descriptors, and the external ABI.
   R semantic lowering may construct computation or qparams needed to meet its
   declared tolerance. W target-specific semantic lowering may do so only after
   a greater-than-one-LSB failure is attributed to shared arithmetic after
   adapter and endpoint projection are otherwise correct. Representation or
   legality projection may be added without changing that arithmetic. Fix
   adapter, endpoint, and runtime failures at their own boundaries. The
   converter may not repair semantics. In X, target-exact computation and
   qparams must already be core compiler facts before conversion.
6. Keep the derived target configuration based on the default configuration.
   Every deviation must be intentional and reported.
7. Accelerator-unsupported operations may be assumed to run on an external
   CPU. Mark them `unsupported`; do not implement CPU lowering, runtime, or
   execution code in Voyager for this task. An accelerator submission may
   contain only work the accelerator implements. Unsupported operations may
   remain in the full model or non-executable boundary metadata, but must never
   be emitted as executable accelerator work. This assumption does not prove
   CPU or connected end-to-end execution.
8. Keep these facts separate:
   - the operation has a semantic match under the stage tolerance;
   - Voyager recognized the operation;
   - Voyager serialized the operation;
   - the target adapter accepted it;
   - the target executed it; and
   - its result passed the applicable comparison.
9. Preserve failed experiments and explain what boundary failed. A conversion,
   execution, or comparison failure does not erase an established semantic
   match. Continue the compiler/adapter implementation, or identify an
   independently evidenced hardware bug. Do not report an operation as mapped
   merely because a parser recognized its name.
10. Record semantic support as `supported`, `unsupported`, or `unverified`.
    Record each translation, execution, and comparison check separately as
    `pass`, `fail`, `blocked`, or `unverified`.

## 3. R: establish the target injection path

### R goal

Establish a reproducible path from Voyager output into the target hardware
stack through the named partial- or full-control injection point and attempt
every operation kind in `{{R_OPERATION_KIND_LIST}}`.
R establishes the default-derived hardware profile, target capability by
operation kind under `{{R_REFERENCE_AND_TOLERANCE}}`, and the injection
foundation. It does not require a full workload mapping.

### R1. Build the target configuration

- [ ] Locate the target's default configuration and record its provenance.
- [ ] Copy or derive `{{DERIVED_RUN_CONFIG}}` from that default.
- [ ] Apply only `{{PERMITTED_OVERRIDES}}`.
- [ ] Save the default-to-derived configuration diff.
- [ ] Construct or validate Voyager's typed `HardwareConfig` from the same
      default-derived facts under the `hardware-profile` role.
- [ ] Prove the external target configuration and Voyager `HardwareConfig`
      describe the same machine and record the provenance.
- [ ] Record hardware capacities and supported data/quantization formats
      relevant to the requested operations.
- [ ] When available, run one stock target example as an independent endpoint
      and configuration sanity check.

The stock example is useful provenance, not an additional R gate. If an
available stock example fails, diagnose that foundation before changing
Voyager.

### R2. Freeze the injection point

Fill this table from an observed successful or attempted run.

| Boundary | Exact file, object, command, or API | What is consumed |
| --- | --- | --- |
| Voyager output | `{{VOYAGER_OUTPUT}}` | `{{OPS_FIELDS_AND_METADATA}}` |
| Adapter input | `{{ADAPTER_INPUT}}` | `{{CONSUMED_VOYAGER_CONTENT}}` |
| Adapter output | `{{TARGET_COLLATERAL}}` | `{{EMITTED_CONTENT}}` |
| Target load/injection | `{{TARGET_LOAD_POINT}}` | `{{LOADED_CONTENT}}` |
| Completion/readback | `{{READBACK_POINT}}` | `{{RESULT_CONTENT}}` |

Report fields that are actually consumed. A field present in `model.txt` but
ignored by the adapter or target is not part of the effective injection
contract. Enumerate any hidden native decisions that materially affect mapping
cost, then record the control classification and its evidence.

### R3. Push each requested operation kind

For every requested operation kind:

1. Construct a small, natural representative of the requested logical PyTorch
   operation. Do not choose a convenient fixed microkernel shape and generalize
   its result to other metadata.
2. Export and lower it through the normal Voyager path.
3. Serialize canonical `model.txt` or the named Voyager output.
4. Pass that output through the named adapter and injection point.
5. Execute the target endpoint and read back the complete produced result.
6. Compare the complete result under `{{R_REFERENCE_AND_TOLERANCE}}`.
7. Record the tested signature and restrictions discovered from the attempt.

Use a direct representation of the logical operation first. Do not force all
operations into the shape of a previously successful microkernel.

| Operation kind | Tested metadata | Semantic support | Serialized check | Adapter check | Execution check | Comparison check | Restrictions/gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `{{OP_KIND}}` | `{{SHAPE_DTYPE_ATTRIBUTES}}` | `{{SUPPORTED_UNSUPPORTED_UNVERIFIED}}` | `{{RESULT}}` | `{{RESULT}}` | `{{RESULT}}` | `{{RESULT}}` | `{{RESULT}}` |

Semantic `supported` means the logical operation has a target realization
that matches the R reference under the declared R tolerance. It does not mean
all later checks passed. A supported form may have a failed converter,
execution, or numerical check; keep support and the failed check separate.
`unsupported` requires target capability evidence. Unknowns are
`unverified`.

### R completion

R is complete when the derived default-based configuration and injection path
are reproducible, every requested operation kind has an honest result, and at
least one Voyager-generated operation has executed and passed comparison
through the full injection path. Scope every support result to its tested
signature, R tolerance, target, and injection point. Keep failed checks,
unsupported forms, and unverified forms in the final report.

## 4. W: bind the workload semantically

### W correctness rule

W requires semantic agreement with the named PyTorch/reference behavior in
output-code space with `rtol=0`, `atol=1` LSB. W does **not** require
bit-identical output. Different rounding, quantized representations, or
approximate target arithmetic are permitted only when they meet this fixed
semantic comparison. `{{REFERENCE_AND_TOLERANCE}}` must name the reference and
how its outputs are represented as comparable codes; it may not relax the
one-LSB rule.

Begin with an executable shared semantic operator such as `conv2d_mx` when it
already expresses the workload operation. Add only the representation or
legality projection needed to reach the registered injection point, and use the
existing target-native software stack there. Keep the shared semantics when
they pass the W comparison. Add target-specific semantic lowering only after a
recorded greater-than-one-LSB failure is attributed to shared arithmetic and
the adapter and endpoint projection are otherwise correct. Classify and fix
representation, adapter, endpoint, and runtime failures separately.

Do not make R's exact collateral, tensor shape, or comparison rule a W gate.
The W adapter must still emit a structurally legal endpoint ABI, but W does not
require proof of bit-identical conversion arithmetic. Do not silently add
target-internal instruction metadata, local schedule factors, scratchpad
placement, or address-generator controls to a W-only claim. The ordinary
compiler may still generate graph-visible/global addresses and other canonical
facts. Record whether hidden post-emission mapping decisions make the point
partial-control. W may complete through either point; exact output through a
partial-control point does not become X. A complete same-scope X proof uses a
full-control point and subsumes W; do not label it `W + X`.

W may still choose semantic graph splits, grouping, fusion, and serialization
visible at the injection point. These are workload mapping choices, not claims
that Voyager controls the target's physical loop schedule or addresses.

### W0. Inventory the workload

Export the exact workload and enumerate:

1. every distinct logical operation kind; and
2. every distinct metadata signature used by each kind.

A metadata signature should include every behaviorally or target-relevant
field, for example batch, input/output H and W, channels, kernel dimensions,
stride, padding, dilation, groups, dtype, layout, and quantization scheme.

| Signature ID | Kind | Occurrences | Input N/C/H/W | Output N/K/H/W | Kernel R/S | Stride | Padding | Dilation | Groups | Dtype/layout/qscheme |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | ---: | --- |
| `{{ID}}` | `{{OP_KIND}}` | `{{COUNT}}` | `{{VALUES}}` | `{{VALUES}}` | `{{VALUES}}` | `{{VALUES}}` | `{{VALUES}}` | `{{VALUES}}` | `{{VALUE}}` | `{{VALUES}}` |

### W1. Map one isolated operation of every kind

Before serializing the full workload, map each workload operation kind by
itself:

1. Start with its natural PyTorch semantics and executable shared operator.
2. Add only required representation or legality projection while preserving
   that shared arithmetic.
3. Try each relevant declared qscheme when it can change target realizability.
4. Generate Voyager output and pass it through the same R injection point,
   reusing its target-native software stack.
5. Execute and compare the shared baseline under the W semantic rule.
6. Attribute any failure before changing semantics. Permit target-specific
   semantic lowering only for a greater-than-one-LSB failure owned by shared
   arithmetic after adapter and endpoint projection are otherwise correct.
   Keep representation, adapter, endpoint, and runtime failures at their own
   boundaries.
7. Keep the shared semantics when they pass.
8. Record semantic support separately from serialization, adapter, execution,
   and comparison checks.

When R already established authoritative hardware non-support for the same
logical form, carry it into the inventory instead of implementing another
lowering. Re-evaluate it only when W has materially different metadata or the
work order explicitly asks to challenge that capability result.

For lowering changes, use this ownership order:

1. reuse an existing shared PyTorch/Voyager lowering;
2. generalize shared lowering only when the rule is target-neutral;
3. add target-local representation or legality projection only for a real
   target constraint, without changing shared arithmetic;
4. add target-specific semantic lowering only for a recorded greater-than-one-
   LSB failure attributed to shared arithmetic after adapter and endpoint
   projection are otherwise correct; and
5. do not add target-name branches to shared lowering.

Record the hard gate before continuing to W2:

| Kind | Natural representative | Shared lowering reused or new owner | Semantic support | Serialization check | Adapter check | Execution check | W comparison |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `{{OP_KIND}}` | `{{SEMANTICS_AND_METADATA}}` | `{{PATH_OR_OWNER}}` | `{{SUPPORTED_UNSUPPORTED_UNVERIFIED}}` | `{{RESULT}}` | `{{RESULT}}` | `{{RESULT}}` | `{{RESULT}}` |

In W, `supported` means the target realization semantically matches the
PyTorch operation under `{{REFERENCE_AND_TOLERANCE}}`. Later checks may still
fail. A failed check is implementation or hardware-debug work, not permission
to erase semantic support or silently route the form elsewhere.

The first optimization objective is to map as many operation **kinds** as the
accelerator supports. Do not start fusion or call-count optimization while
basic supported kinds remain unmapped.

### W2. Map the workload's real metadata variants

For every kind that W1 proved supported, attempt each distinct metadata
signature observed in the workload, including
its qscheme and qparams. A different declared qscheme may make a previously
unmappable kind or signature target-realizable; treat that as a new direct
attempt, not as post-coverage tuning.

If serialization, adapter, execution, or comparison failed in W1, continue at
that failed boundary before attempting dependent signatures. Do not erase the
semantic support result or silently skip the kind.

Try the logical operation directly first. If a real target constraint prevents
direct execution, use a semantic-preserving split or decomposition and record:

- the exact constraint;
- why the decomposition preserves PyTorch semantics;
- accelerator calls per original occurrence;
- total projected calls and collateral size for the workload; and
- the comparison result.

Do not rewrite every workload operation into one fixed R kernel merely because
that kernel already executed. Reject decompositions whose projected call count
or collateral growth is unreasonable for the workload, and search for a more
natural target form.

| Signature ID | Shared lowering reused or new owner | Direct execution | Decomposition, if needed | Calls/occurrence | Projected workload calls/size | Semantic comparison |
| --- | --- | --- | --- | ---: | --- | --- |
| `{{ID}}` | `{{PATH_OR_OWNER}}` | `{{RESULT}}` | `{{NONE_OR_DESCRIPTION}}` | `{{COUNT}}` | `{{CALLS_AND_SIZE}}` | `{{RESULT}}` |

The second optimization objective is to map as many supported workload
metadata variants and occurrences as possible.

### W3. Serialize the workload

After W1 and W2:

- reuse shared semantic lowering first: match against the operators already in
  `src/voyager_compiler/decomposed.py` and add a target operator only for a
  named gap;
- place target-specific recognition, composites, and explicit multi-call
  rewrites in the target final-FX backend;
- keep target-specific ABI projection, packing, and descriptors in the target
  adapter or converter;
- retain ordinary unsupported workload operations in final FX/`model.txt`
  with separate unsupported assignment evidence;
- serialize final target FX and canonical `model.txt`;
- inject the serialized supported regions through the established boundary;
  and
- compare every accelerator-owned output under the W semantic rule.

The full `model.txt` may contain both accelerator-supported and unsupported
workload operations. Its size or operation count must be explained by actual
workload lowering and any recorded decomposition, not by accidental expansion.
Executable accelerator collateral contains only work the accelerator
implements. Non-executable boundary records may remain when the target
interface requires them. At boundaries from unsupported operations, use frozen
reference-provided tensors under the external-CPU assumption. Do not claim that
this proves CPU execution or connected end-to-end execution.

### W4. Performance-guided serialization

When performance refinement is requested, iteratively improve the form after
the initial direct mapping and serialized baseline. This is optional refinement,
not part of W's semantic definition. Use a coarse, provenance-qualified estimate;
do not claim exact runtime cycles. At a partial-control point, keep hidden
native mapping cost unresolved or use a clearly labeled endpoint measurement;
do not attribute it to Voyager's mapping. Freeze the model, inputs, target,
endpoint and control class, semantic reference, fixed one-LSB rule, and
coverage denominator. Candidate
quantization scheme, calibration/qparams, grouping, and decomposition choices
may vary, but each candidate must record them explicitly. Qscheme changes may
increase coverage, reduce calls, or avoid conversion, so do not restrict them
to equal-coverage tuning. Relevant experiments include:

- target- and adapter-supported quantization schemes and precisions;
- legal fusion of adjacent supported nodes in target final-FX lowering;
- grouping more supported work into one accelerator or ISA call through
  target final-FX lowering;
- selection among layout, packing, or serialization forms already accepted by
  the injection point;
- semantic graph-level tiling or multi-call decomposition when required by
  real capacity limits; and
- reuse or residency opportunities explicitly exposed by the target interface,
  with ownership and consumption claims no stronger than the injection-point
  evidence supports.

Do not put optimization policy in the canonical serializer. Change the owning
semantic or target lowering, regenerate canonical output, and let the converter
project it. New physical schedule/address work may help establish a
full-control point, but no single field or work item earns X.

Estimate:

```text
estimated_total_time =
    estimated_accelerator_time
  + observed_or_estimated_calls * per_call_latency
  + estimated_conversion_or_materialization_time
  + other_qualified_overhead
```

Optimize in this order:

1. maximize semantically supported and target-mappable operation-kind,
   metadata-signature, and occurrence coverage; and
2. at equal coverage, minimize estimated total time.

Each call contributes latency, so total call count is already represented in
estimated total time. Report calls and work per call to explain the estimate;
do not create a separate call-count objective that can override total time.

Define the call layer being counted. A graph node, helper invocation, engine
submission, and dynamic ISA call are not automatically the same thing.
Graph fusion counts only when the target trace or counter shows the expected
grouping at the selected call layer.

Accept an experiment only when:

- the W semantic comparison still passes;
- it increases fixed-denominator coverage; or
- at equal coverage, it reduces estimated total time under the same estimation
  method.

| Experiment | Qscheme/qparams and lowering change | Kind coverage | Signature coverage | Occurrence coverage | Work/call | Calls | Estimated total time | Semantic check | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `{{ID}}` | `{{CHANGE}}` | `{{VALUE}}` | `{{VALUE}}` | `{{VALUE}}` | `{{VALUE}}` | `{{VALUE}}` | `{{VALUE_AND_UNIT}}` | `{{PASS_FAIL}}` | `{{KEEP_REJECT}}` |

### W completion

W is complete when:

- every workload operation kind was attempted in isolation;
- every distinct real metadata signature of each supported kind was attempted
  through the injection point;
- supported workload regions were serialized, target-executed, and compared;
- unsupported operations are identified without new Voyager CPU code;
- fixed-denominator checked accelerator coverage is reported;
- call grouping, call count, and estimated total time are reported when
  performance refinement was requested; and
- non-bit-accurate results are labeled with W's fixed one-LSB rule; and
- the injection point is classified as partial-control or full-control, with
  performance claims bounded accordingly.

## 5. X: establish a bit-accurate full-control workload binding

### X goal

Bind the same frozen workload through a full-control injection point and match
the same workload-semantic reference already frozen for W with zero tolerance.
X carries all W mapping, final-assignment, execution, coverage, and
semantic-comparison obligations. When the reference is packaged for the
endpoint, retain the provenance showing that the package is a direct
serialization or semantics-preserving projection of that reference.

Use separate X work orders with `x_step: bit_accurate_core_lowering` and
`x_step: inner_control` when their outputs have different owners. Their order
follows real dependencies. Neither local result completes X: the same-scope W
proof, full-control classification and endpoint consumption, exact output, and
workload-reference provenance must all pass.

### X1. Freeze the full-control injection contract

First inventory every variable decision that materially affects the mapping
cost of an emitted call. Identify its Voyager scheduler or mapper owner, its
canonical representation, and the endpoint field or behavior that consumes it.
Also identify any decision left to a runtime or helper. If such a hidden choice
materially affects mapping cost, the point is partial-control and cannot support
X through this interface.

Tiling, tensor allocation, scheduling, storage placement, addresses, and
instructions are common evidence, not universal criteria. Fixed
microarchitectural behavior may remain fixed when it is not a hidden software
mapping choice and its cost is reliably modeled.

For every input, weight, bias, intermediate, and output crossing the point,
freeze:

- logical and physical dtype;
- scale, zero point, and quantization granularity;
- accumulator width and arithmetic order;
- rounding, overflow, and saturation behavior;
- tensor layout, blocking, padding, and packing;
- byte order and binary or descriptor encoding; and
- applicable allocation, address, transfer, completion, and mapping ownership.

| Decision or boundary | Voyager representation | Required target representation | Consumer/evidence |
| --- | --- | --- | --- |
| `{{VALUE_OR_DECISION}}` | `{{VOYAGER_FORM}}` | `{{TARGET_FORM}}` | `{{CONSUMER_AND_EVIDENCE}}` |

### X2. Preserve the frozen W reference and implement bit-accurate core lowering

- Keep canonical semantics and quantization facts in shared Voyager IR.
- Put only target-neutral, reusable exact arithmetic in the narrow shared
  owner.
- Put target-specific fixed-point rules, rounding, saturation, physical qparams,
  and hardware arithmetic in target-owned core lowering.
- Ensure every computation and qparam needed by the endpoint is present in
  final target FX, `model.txt`, or referenced canonical compiler artifacts
  before conversion starts.
- Let the converter encode and project only finalized facts. It must not invent,
  repair, or numerically reconstruct computation, qparams, schedules, mappings,
  or allocations.
- Compare the scoped target output against the frozen workload-semantic
  reference with zero tolerance.
- If the compiler or converter packages an expected stream, prove reproducibly
  that it is a direct serialization or semantics-preserving projection of the
  same reference.

An exact comparison only to an expected stream of unknown or endpoint-generated
provenance is self-consistency evidence. W's one-LSB boundary remains an
intermediate checkpoint, not a separately maintained result after X passes.

### X3. Establish and prove full control

Implement every performance-relevant mapping decision identified in X1 using
its normal owner. Local-loop decisions belong to a scheduler. `StoragePlan`
owns storage identities and lifetimes; mapping and allocation own placement and
addresses. The target converter projects finalized facts into the external
interface; it must not invent or repair them.

Do not create decorative fields. Prove the adapter and endpoint consume the
artifact carrying Voyager's decisions and do not replace them with hidden
native choices. A legal control variation that changes the expected trace,
address, call, counter, or modeled cost while preserving exact output is strong
causality evidence when the claim depends on that field; it is not a universal
field-by-field prerequisite.

### X4. Validate the workload binding

- [ ] Every same-scope W mapping, final-assignment, execution, coverage, and
      semantic-comparison obligation passes.
- [ ] The injection point has recorded full-control evidence.
- [ ] Voyager owns every variable decision that materially affects per-call
      mapping performance, and the endpoint consumes its representation.
- [ ] Canonical compiler facts and conversion preserve the frozen
      workload-semantic reference.
- [ ] Complete target output is bit-accurate under the frozen X contract.
- [ ] Any packaged expected stream has direct provenance from that reference.
- [ ] Any performance model is validated separately from the X stage claim.

| X decision | Owner | Emitted artifact/field | Consumption evidence | Exactness result |
| --- | --- | --- | --- | --- |
| `{{DECISION}}` | `{{OWNER}}` | `{{FIELD}}` | `{{TRACE_OR_TEST}}` | `{{RESULT}}` |

### X completion

X is complete when the same frozen workload passes all W obligations through
the full-control point and scoped target output matches the frozen
workload-semantic reference exactly, with provenance retained for any packaged
expected stream. This complete result subsumes W and R for the same tuple;
report and maintain X rather than a second W-only result.

A successful exact operation or region test is an X-oriented subcheck, not a
completed workload X binding. If unsupported work runs on a host, the X
performance scope covers the accelerator portion unless connected execution
and host/transition accounting are also retained.

## 6. Required final report

Produce a concise final report containing:

1. repository revisions, workspace paths, environment, and derived target
   configuration diff;
2. the exact Voyager-to-target injection point, its partial- or full-control
   classification and evidence, and which Voyager output fields were consumed;
3. the R operation-kind result table;
4. the W workload inventory, real-metadata result table, final `model.txt`,
   semantic comparison, and, when requested, performance experiment table;
5. the X full-control, conversion, exact-execution, and workload-reference
   provenance results when X is claimed;
6. successful operations, unsupported operations, and unverified gaps;
7. emitted, estimated, bounded, observed, or unknown accelerator/ISA call
   evidence with its named layer and provenance;
8. commands and paths for accepted collateral, logs, traces, and outputs; and
9. agent run status: completed, still running, blocked, or stopped, including
   the current activity and next unresolved boundary.

Operational run status and handoff outcome are different fields. A
`still running` snapshot is a `partial` handoff; a completed run is `complete`;
and a blocked run is `blocked`. A stopped run is `complete`, `partial`, or
`blocked` according to what finished and why it stopped.

Report partial progress honestly. A valid result may contain unsupported or
unverified operations; it must not hide them by changing the workload,
reference, target configuration, or injection boundary.

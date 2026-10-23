<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Backend-adoption model

Voyager adoption has two independent views:

1. the lifecycle of a target adoption: registration (`R`), workload binding
   (`W`), and bit-accurate full-control workload binding (`X`);
   and
2. the compilation path and the owner of each abstraction.

Do not turn lifecycle or evidence records into another compiler IR.

The stages follow a working order. For the same
frozen claim scope, `X => W => R`: completing W subsumes R, and completing X
subsumes W and R. Report the strongest completed stage as the achieved stage.
A work-order mode names the bounded action currently advancing that stage; it
is not an additive completed-state label. Incomplete X work does not advance the
achieved stage; it remains W only when W was already complete, and never creates
a `W + X` state.

Working order does not require three separately retained endpoint campaigns.
W is an intermediate semantic mapping and acceptance boundary on the way to X.
A complete X proof for the same frozen scope carries the W mapping, coverage,
execution, and comparison obligations and strengthens the numerical comparison
from one LSB to zero tolerance. Once X passes, retain and report the X claim;
there is no separate W result to maintain for that scope.

## Injection points and control

An **injection point** is where Voyager hands emitted work to a target path and
observes the execution result. Its frozen contract names
the artifact or API consumed by the target path, the adapter and native stack,
the target configuration, the input and output boundary, launch and readback,
and the semantic reference and comparison procedure. The reference may belong
to the validation harness rather than the target itself; it is comparison
collateral, never an accelerator input. A converter may package the reference
for the endpoint without creating a second semantic reference. That does not
change the control classification.

Classify each concrete injection point by whether Voyager controls the
performance-relevant mapping:

- At a **partial-control injection point**, a native runtime, helper, or other
  layer makes hidden performance-relevant mapping decisions after Voyager
  emits a call. Voyager can prove semantic coverage through this point, but it
  cannot confidently attribute or optimize the call's mapping performance.
- At a **full-control injection point**, the decisions made by Voyager's
  scheduler and mapper determine the performance-relevant mapping of each
  emitted target call or ISA execution. The emitted artifact and frozen target
  configuration therefore expose enough information to estimate that call's
  mapping-sensitive cost reliably.

This classification is performance-based, not a universal field checklist.
Explicit tiling, tensor allocation, loop scheduling, placement, and addresses
are common evidence of full control when those choices affect the target, but
no one field is a mandatory criterion for every interface. Conversely, a
fixed microarchitectural mechanism is not a disqualifying black box when it
contains no hidden software mapping choice and its cost is reliably modeled.
The classification is scoped to the named injection point and target
configuration; the same target may expose both partial- and full-control
points.

Full control makes reliable mapping-sensitive estimation possible; it does
not by itself validate a particular cycle, latency, or energy model. R and W
may use either class of injection point. X requires a full-control point.

## One semantic compilation path

```text
model/export and PTQ policy
  -> executable reference Q/DQ graph
  -> final target FX
  -> model.txt
  -> target adapter or native software stack
  -> accelerator execution
```

Final FX and `model.txt` are the semantic source of truth. A fused operation
in `model.txt` may be translated by the native backend stack into one or more
accelerator calls. Binding proves the accelerator assignments execute and
produce the correct result. Accelerator-unsupported operations are assumed to
execute on an external CPU and are excluded from accelerator coverage. Do not
implement their CPU path or introduce a separate call-plan representation.

When Voyager explicitly rewrites one operation into several accelerator
calls, those calls and their result-combination operations belong in final FX
and `model.txt`. When the native stack decomposes one serialized operation,
that decomposition stays in target collateral or the runtime and is part of a
partial-control point when it materially affects mapping cost. X requires a
full-control point where Voyager determines the relevant decomposition.

## Compilation ownership

| Abstraction | Owner |
| --- | --- |
| Model, export, inputs, and calibration policy | Workload adapter |
| Quantization arithmetic and explicit Q/DQ values | PT2E quantization and executable operators |
| Supported operation, quantization, and composite forms | Target registration |
| Composite grouping and explicit multi-call graph rewrites | Target lowering |
| Final semantic operation sequence | Final target FX |
| Canonical compiler serialization | `model.txt` and existing compiler artifacts |
| ABI or collateral projection | Target adapter under `conversion/<target>/` |
| `R` semantics under its declared tolerance | Shared or target final-FX lowering |
| `W` semantic baseline | Executable shared final-FX operators |
| `W` target semantic correction after a recorded greater-than-one-LSB failure attributed to shared arithmetic with correct adapter and endpoint projection | Target final-FX lowering |
| `W` representation or legality projection that preserves shared arithmetic | Target final-FX lowering or target adapter, as appropriate |
| `X` bit-accurate fixed-point arithmetic, physical qparams, and hardware semantics | Target-owned core lowering before conversion |
| Decisions hidden behind a partial-control injection point | Native backend stack |
| Performance-relevant scheduling, mapping, allocation, placement, addressing, instructions, or descriptors exposed by a full-control injection point | Voyager scheduler, mapper, and target lowering |

Serialization records existing compiler decisions. A converter may validate,
rename, pack, or project those decisions into a frozen target interface. It
must not repair missing graph semantics, qparams, schedules, placement, or
addresses.

## Semantic support and check results

`supported` means the logical operation has a target realization that matches
its reference semantics under the selected stage tolerance. Keep that semantic
status separate from recognition, serialization, conversion, execution, and
comparison check results.

A supported form may still have a failed compiler, converter, endpoint, or
comparison check. Continue implementation at the failed boundary, or identify
an independently evidenced hardware bug. A later failure does not erase
semantic support or a previously proven execution assignment.

The fidelity required to fully declare a stage result is:

- `R`: the registered R reference and tolerance;
- `W`: the workload binding's declared semantic reference compared in
  output-code space with `rtol=0` and `atol=1` LSB; and
- `X`: bit-accurate target behavior with no numerical tolerance through a
  full-control point, against the frozen workload-semantic reference at the
  declared comparison boundary.

`unsupported` requires target capability or authoritative routing evidence.
Unknown support stays `unverified`. Check results use `pass`, `fail`,
`blocked`, or `unverified`; these are not support states.

## Invariants across phases

- The stages follow the working order `R -> W -> X`; within one
  frozen claim scope, a verified later stage carries every earlier stage.
- Numerical correctness is mandatory under the stage's declared fidelity.
- Target capability, candidate assignment, final assignment, conversion, and
  executed evidence are distinct claims.
- Operations assumed to execute on CPU are accelerator-module `unsupported`;
  Voyager does not implement or validate their CPU path.
- Every claimed decision names whether Voyager or the native endpoint owns it.
- Every claim names its concrete injection point and whether that point is
  partial-control or full-control, with evidence for the classification.
- Only work passing the declared W qualification counts as checked accelerator
  coverage; report target execution separately.
- Performance comparisons use the same workload and estimation boundary.

## R: register a target

Registration is scoped to a `(target, injection point)` pair and precedes
workload-specific changes. The point may be partial-control or full-control.
R answers whether at least one declared PyTorch semantic operation or
composite can propagate through Voyager, execute on the target through that
point, and compare completely under its declared R oracle and tolerance.

R also owns construction of the Voyager hardware profile from the target's
authoritative default configuration. Record the target source revision,
default-to-derived configuration diff, permitted overrides, and the provenance
that ties the typed `HardwareConfig` to the machine used by the endpoint.

Record:

- supported operation forms and their shape, dtype, layout, and capacity
  restrictions;
- quantization schemes, Q/DQ boundaries, rounding, saturation, and qparam
  encodings;
- supported composite patterns or native high-level entrypoints;
- the accepted compiler artifact or target collateral boundary;
- the injection point's partial- or full-control classification and supporting
  evidence;
- which decisions are Voyager-owned and endpoint-owned;
- available correctness, cycle, traffic, and utilization observations; and
- the R semantic oracle and tolerance used for each support statement.

### Match semantics against the shared operator library first

Semantic support means the *logical* operation has a target realization under
the R tolerance. It does not require a target-specific operator. Before writing
one, check whether an existing shared lowering already expresses the semantics
you are matching: `src/voyager_compiler/decomposed.py` registers `conv2d`,
`conv2d_mx`, `linear`, `linear_mx`, `matmul`, `matmul_mx`, `max_pool2d`,
`adaptive_avg_pool2d`, `layer_norm`, and the `quantize`/`dequantize` family
including the `_mx` and `_vsq` forms.

Prefer, in order:

1. an existing shared operator that already matches the semantics, configured
   through traits and hardware facts;
2. an existing shared operator plus a declared trait or capability flag that
   narrows it to what the target accepts; then
3. a new target operator, only for a semantic or representation gap you can
   name.

A custom lowering written to prove semantic equivalence that a shared operator
already provides is duplicated arithmetic. It has to be recalibrated, revalidated,
and kept bit-consistent with the shared form forever, and it hides the real
registration answer, which is whether the target executes the shared semantics.
Record the shared operator you matched against in the operation inventory so a
later reader can tell reuse from reimplementation.

Close registration by freezing the injection contract and executing one
Voyager-generated operation or composite through that endpoint with complete
output comparison. A stock target operation is a useful independent endpoint
sanity check when available, not a second requirement for R.

Passing `R` proves the recorded capability inventory and one operation anchor
for the frozen `(target, injection point)`.
It does not prove workload coverage, connected execution, full-model
performance, or full control unless that concrete R injection contract
separately proves the full-control classification.

## W: bind and refine a workload-target pair

Binding is scoped to one exact `(workload, target, injection point)` tuple. The
point may be partial-control or full-control. Freeze the model, weights,
inputs, target configuration, semantic reference and output-code
interpretation. W comparison is fixed at `rtol=0`, `atol=1` output-code LSB.
Also freeze the coverage denominator and, when performance refinement is in
scope, its estimation boundary. Candidate
quantization schemes, calibration, and qparams may vary when declared for each
candidate.

Begin with existing executable shared PT2E Q/DQ semantics, including shared
operators such as `conv2d_mx` when they already express the workload operation.
Use the target adapter and the software stack available at the injection point
to execute that shared form. Retain the shared semantics when they meet W's
fixed one-LSB comparison. Add target-specific **semantic** lowering only when a
recorded greater-than-one-LSB failure is attributed to shared arithmetic after
adapter and endpoint projection are otherwise correct. Fix adapter, endpoint,
and runtime defects at their own boundaries rather than hiding them in semantic
lowering.

A complete W result proves a coverage-proven mapping through the named
point. It does not, by itself, prove that Voyager can reliably estimate or
optimize that mapping's performance when the point is partial-control.

Keep that numerical gate separate from representation and legality. ABI
projection, packing, or a semantic-preserving graph decomposition may still be
needed to make the shared semantics executable at the target. Those changes do
not by themselves justify new target arithmetic. At a partial-control point, W
deliberately permits the existing target software stack to retain target-
internal instruction choice, local scheduling, tiling, scratchpad behavior,
and local address generation. W may also reuse existing software at a full-
control point, but only when it deterministically consumes Voyager's mapping
rather than replacing it with a hidden choice.

Use this loop:

1. inventory every workload operation kind and distinct metadata signature;
2. select an executable shared operator for one natural isolated instance of
   every kind and each relevant declared qscheme;
3. add only the representation or legality projection needed to reach the
   injection point, serialize that shared baseline, and let the adapter or
   native stack translate it;
4. execute and compare the shared baseline under W's fixed one-LSB rule;
5. if comparison exceeds one LSB, attribute the failure before changing
   semantics: only shared-arithmetic ownership permits target-specific semantic
   lowering; adapter, endpoint, and runtime failures remain at their own
   boundaries;
6. only after a direct legality failure, consider a semantic-preserving
   decomposition and reject unreasonable projected call or collateral growth;
7. repeat the shared-baseline sequence for every real workload metadata
   signature of each semantically supported kind;
8. create candidate assignments and legalize executable Q/DQ regions;
9. serialize the supported workload forms to final FX and `model.txt`;
10. after the direct baseline, apply any composite, grouping, or graph-visible
   multi-call rewrite required for semantic translation; when performance
   refinement is in scope, also revisit qschemes and forms for coverage or
   cost, then reserialize;
11. prove that the target adapter or native stack can translate every final
   accelerator assignment;
12. freeze final execution assignments;
13. execute and compare every final assignment under W's fixed one-LSB rule;
14. report checked accelerator coverage against the denominator frozen when
    this binding started; and
15. when performance refinement is in scope, estimate calls, call overhead,
    conversion/materialization cost, and total time, then refine the largest
    qualified gap.

Step 14 is not optional. Coverage is the semantic accounting for how much of
the workload executed on the accelerator and passed, so it belongs to W
completion. The cost estimates in step 15 are the optional refinement.

Do not treat the one R execution anchor as a universal workload kernel. Direct
logical execution is the first attempt for every kind and real metadata
signature.

`candidate_assignment` means an operation may become executable directly,
inside a composite, or after multi-call tiling. It is exploratory and does not
count as executed coverage.

`final_execution_assignment` means the final FX/`model.txt` form has a proven
backend translation. A later conversion, mapping, or runtime failure is a gap
at that accelerator boundary and does not change assignment. Classify declared
external CPU work as `unsupported` from capability or routing evidence before
lowering, without implementing a CPU path in Voyager. Semantic support,
assignment, and each later check remain separate.

### Composite regions

For composite regions, `W` may preserve several source operations as one
high-level graph node only when every member is accelerator-implemented or
correctly absorbed at its boundary. Keep unsupported operations as separate
assignments outside the composite; their assumed CPU execution needs no
Voyager proof. Require legal backend calls and comparison of the complete output
at the accelerator-owned boundary. Record observed or frozen one-to-one call
evidence when available; otherwise report an emitted or estimated count, a
justified bound, or `unknown` with provenance. A composite graph node does not
imply hardware fusion.

Final FX and `model.txt` operations are the semantic input. Target collateral
or runtime traces report the backend calls produced from them; that evidence
is not another compiler representation.

Do not require atomic target operators, internal engine routing, accelerator
residency, scratchpad placement, or instruction-level metadata merely to
complete `W`. Record whether hidden native decisions make the selected point
partial-control. X is unavailable through that point unless a full-control
injection point is exposed and the workload is rebound and revalidated there.

For example, `DQ -> Conv2d -> ReLU -> Q` may remain one fused operation in
`model.txt` while the native backend emits one or several accelerator calls.
Every semantic member must be implemented by the target path; Q/DQ may be
absorbed into boundary encoding only when their numerical behavior is
preserved. Record emitted calls as translation evidence. Claim an actual
executed call count only from runtime observation or a frozen endpoint
guarantee that emitted calls map one-to-one to executed calls.

### Multi-call tiling

A logical operation may be lowered into multiple accelerator instruction
calls. The binding records the generated calls and how their results form the
original operation's output. Every call must be legal and executable, and the
combined output must match the untiled operation.

When Voyager chooses the split, this record is the final FX and `model.txt`,
not a separate tile-plan IR. Record operator-specific details only when the
selected split needs them. Inner loops, scratchpad placement, banks, and
address generation may remain in the native stack for W. If any hidden native
choice materially determines the mapping performance of an emitted call, the
resulting point is partial-control and cannot support X.

For example, an output-channel split may need only several Conv calls and a
concatenation. Spatial halos or reduction accumulation are required only for a
split whose semantics need them; they are not universal adoption fields.

### Execution and performance

A region execution may use captured inputs while bring-up is incomplete, but
that proves only the region. Connected execution requires inputs produced by
the workload path under test.

Cost estimation is optional after the W semantic and execution boundary
passes; it is not part of the definition of W completion. Reported coverage
is, and is required regardless. At a partial-control point, keep hidden native
mapping cost explicit as unknown or use a clearly labeled endpoint measurement;
do not attribute that cost to Voyager's emitted mapping. When requested, use a
coarse estimate rather than claiming exact runtime. First maximize
fixed-denominator checked accelerator coverage under the W qualification. At
equal coverage, minimize estimated total time. Include per-call latency, so
reducing calls already improves the estimate when other costs remain equal.
Record call count, work per call, accelerator cost, and
conversion/materialization cost as explanatory components. Different declared
qschemes may compete when the model, inputs, target, endpoint, semantic
reference, fixed one-LSB rule, and coverage denominator remain fixed.

## X: bit-accurate full-control workload binding

`X` is scoped to the same exact `(workload, target, injection point)` tuple as
W, and that injection point must be full-control. X carries the complete W
mapping, coverage, execution, and comparison obligations and strengthens the
numerical comparison to zero-tolerance, bit-accurate target execution. A
separately retained W campaign is unnecessary when the complete X proof covers
the same frozen scope: passing X automatically passes and subsumes W and R.

Full control means Voyager's scheduler and mapper determine the
performance-relevant mapping of every emitted accelerator call or ISA
execution in the claimed workload scope. The endpoint must consume the
resulting artifact. If a runtime or helper still makes a hidden
performance-relevant mapping decision after emission, that injection point is
partial-control and the binding cannot claim X through it, even if its output
is bit-exact.

Target fixed-point arithmetic, rounding, saturation, physical qparams, and
every numerical fact needed by the endpoint must be explicit in final target
FX, `model.txt`, or referenced canonical artifacts before conversion. The
converter encodes finalized facts; it never invents or repairs computation,
qparams, schedules, mappings, or allocation decisions.

W and X use one frozen workload-semantic reference at the same declared
comparison boundary. The X proof is:

```text
target actual output == frozen workload-semantic reference output
```

A compiler- or converter-packaged expected stream is sufficient when retained
provenance shows that it is a direct serialization or semantics-preserving
projection of that frozen Voyager reference. No second numerical "bridge"
comparison is required in that case. An expected stream with unknown or
endpoint-self-generated provenance proves only self-consistency.

Tiling, tensor allocation, local loop order, placement, banking, transfers,
addresses, instructions, descriptors, and engine routing are common
target-specific evidence that Voyager controls the mapping. They are not a
universal checklist: record the variable decisions that materially affect
performance for the concrete interface, identify their Voyager owner, and
prove the endpoint consumes them. Emitting an ignored field is not evidence of
control. A legal variation is stronger field-causality or schedule-exact
evidence, but is required only when the claim depends on that causality.

Do not infer that X has started from an exact W mismatch count, a prospective
lower-level interface, a full-control-capable ABI, or a successful exact
operation test. X begins only after the full-control injection point, frozen
workload and comparison scope, workload-semantic reference and its provenance,
and scoped lowering or mapping work are identified. Operation-local exact or
full-control checks are X-oriented subchecks; they do not constitute a
completed workload X binding.

X provides the transparency and compiler control needed for reliable per-call,
mapping-sensitive performance estimation. It does not automatically validate
a particular latency, cycle, traffic, or energy model; validate those models
against target evidence separately. When unsupported operations execute on a
host, an X performance claim covers the accelerator portion unless connected
whole-workload execution and host/transition accounting are also retained.

The implication is valid only for the same frozen scope:

```text
X(workload, target, I_full) => W(workload, target, I_full)
                            => R(target, I_full)
```

A target registered through a partial-control point may later expose a
different full-control point, but that new tuple must be registered, rebound,
and revalidated. A full-control point is X-capable, not proof that X has been
achieved.

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Backend-adoption evidence contract

Use this contract to gate handoffs and record target evidence. The canonical
lifecycle and compiler path are defined by the
[adoption model](../adoption-levels.md). Performance claims additionally obey
the [performance contract](performance.md).

## State vocabularies

Three enums exist. They are not interchangeable, and each has its own schema
home:

| Vocabulary | Values | Applies to |
| --- | --- | --- |
| Semantic support | `supported`, `unsupported`, `unverified` | one operation or capability |
| Boundary check | `pass`, `fail`, `blocked`, `unverified`, `not_in_scope` | one named check that ran |
| Gap `evidence_state` | `unsupported`, `approximated`, `fail`, `blocked`, `unverified` | one recorded gap |

Use `pass` or `fail` only for a check that actually ran. Use `blocked` only
when a concrete dependency stopped a required check from running at all, and
`unverified` when evidence is merely insufficient. `not_in_scope` marks an
explicitly excluded boundary. `approximated` records a measured deviation; it
belongs to a gap record and is never a check result.

`unsupported` appears in two vocabularies with different meanings: absence of
target capability for a form, and the classification of a recorded gap. It is
never a check result.

These are evidence states, not compiler objects.

## Semantic support states

The semantic-support enum states capability under the named stage fidelity and
nothing else. R uses its declared numerical
tolerance, W uses `rtol=0`, `atol=1` output-code LSB, and X requires
bit-accurate output with zero tolerance. Support does not mean that
serialization, conversion, execution, or comparison passed.

Record those later boundaries with the boundary-check enum. A semantically
supported form with a failed check remains supported while the failed compiler,
adapter, endpoint, or hardware boundary is investigated. Classify a hardware
bug only with independent target-authoritative evidence; otherwise retain the
actual failed or unverified boundary.

## Assignment states

Keep two assignment moments:

- `candidate_assignment`: an occurrence may execute on the accelerator
  directly, inside a composite, or after multi-call tiling. It is exploratory
  and contributes no executed coverage.
- `final_execution_assignment`: the final target FX/`model.txt` form has a
  proven translation to the selected backend. Freeze it before execution.

Use simple owners for either record: `accelerator`, `unsupported`, or
`unverified`. Use `unsupported` when accelerator non-support or declared
external routing is evidenced. It is the accelerator-module umbrella for work
then assumed to execute on an external CPU. That CPU assumption requires no
Voyager lowering, executor, transfer, launch, or execution evidence.
Unsupported work contributes zero accelerator coverage. Use `unverified` when
accelerator support or routing is not yet known.

Normalize any declared legacy artifact or log route such as `cpu` or `host` to
`unsupported` in accelerator assignment records. That route is classification
evidence; it never creates a CPU implementation task for Voyager.

Coverage exclusion and accelerator execution are independent facts. Credit an
`accelerator` assignment only when target trace or a frozen endpoint guarantee
shows that the claimed accelerator work executed there. This positive target
proof neither implements nor validates the CPU path for unsupported work.

Later scheduling, conversion, mapping, or runtime failure does not change an
assignment, reclassify it as `unsupported`, or authorize a Voyager CPU path.
Record the accelerator-module gap instead.

## Semantic source and execution proof

Final FX and `model.txt` are the semantic record; see the
[adoption model](../adoption-levels.md) for why. Do not repeat their
membership, order, tensor dependencies, or result construction in an evidence
artifact. Record only what a check observed.

To credit a composite accelerator assignment, require:

- every semantic member credited to the accelerator is implemented by the
  target path or correctly absorbed into target boundary arithmetic or
  encoding;
- complete output comparison passes at the accelerator-owned boundary; and
- call evidence is qualified as observed or frozen one-to-one when available;
  otherwise record emitted or estimated calls, a justified bound, or `unknown`
  with provenance.

For multi-call tiling, execute every generated call and compare the combined
result with the untiled operation. Require operator-specific split evidence
only when needed by that split.

A converter reports emitted calls as translation evidence. Executed calls
require runtime observation or a frozen guarantee that emitted calls execute
one-to-one.

## Worked example

A `Conv2d` was lowered, converted, and run on the endpoint. Output differed
from the reference by more than one output-code LSB. Record:

```yaml
semantic_support: supported      # established by prior evidence ID
checks:
  serialization: pass
  adapter:       pass
  execution:     pass            # it ran; "ran" is not "was correct"
  comparison:    fail            # the numerical mismatch lives here
assignment:
  final_execution_assignment: accelerator   # unchanged by the failure
```

Three mistakes this prevents:

- **Do not** set `semantic_support: unsupported`. Support is a capability
  claim at a tolerance; it is not erased by a downstream check.
- **Do not** reassign to `unsupported` and route the operation to CPU. A
  comparison failure is an accelerator-module gap to fix, not a routing change.
- **Do not** record `blocked`. Nothing prevented the check from running — it
  ran and it failed. `blocked` is only for a concrete dependency that stopped
  the check from executing at all.

The next action is a gap record naming the comparison boundary, its owner, and
the unblock condition — not a status downgrade.

A greater-than-one-LSB result justifies target-specific W semantic lowering
only when the recorded failure belongs to the executable shared semantic
baseline and the adapter and endpoint projection are otherwise correct. A
converter, endpoint, or runtime defect must be fixed at its own boundary rather
than hidden in semantic lowering.

## Required claim boundaries

Record only applicable boundaries:

### Registration (`R`)

- default-derived target configuration, matching Voyager `HardwareConfig`,
  configuration diff, and provenance;
- target operation, composite, and quantization capability inventory;
- semantic support results under the frozen R tolerance;
- frozen injection contract, including partial- or full-control classification
  and target-specific evidence;
- one Voyager-generated operation or composite translated, executed through
  the named point, and completely compared against the registered R reference.

A separate stock-native operation is useful endpoint/configuration provenance
when available, but it is not another R gate and is never the comparison
oracle. This keeps same-scope `W => R` valid without an unrelated prerequisite.

Registration breadth and the one-operation anchor are separate facts.

### Workload binding (`W`)

- exact workload, inputs, PTQ/calibration state, target configuration, and
  semantic reference, output-code interpretation, and concrete partial- or
  full-control injection point;
- shared executable semantic operator identity, such as `conv2d_mx`, and its
  comparison at the frozen W boundary;
- any representation or legality projection needed to execute that shared
  arithmetic, with the target constraint that required it; and
- for target-specific semantic lowering, evidence that the executable shared
  baseline exceeded `rtol=0`, `atol=1` output-code LSB;
- one direct isolated attempt per operation kind and one direct attempt per
  distinct real metadata signature of every semantically supported kind;
- candidate assignment, including the selected qscheme/calibration/qparams;
- executable Q/DQ and final target FX/`model.txt`;
- final assignment after backend translation;
- target execution for accelerator assignments, complete comparison, and
  exclusion of unsupported work from accelerator coverage;
- accelerator-call evidence: an observed count or frozen one-to-one guarantee
  when available, otherwise an emitted or estimated count, justified bound, or
  `unknown` with provenance; and
- fixed-denominator checked accelerator coverage, always; and calls, per-call
  latency, conversion/materialization cost, and estimated total time when
  refinement is in scope; and
- a published coverage denominator that separates mapped, executed,
  within-tolerance, and unassigned work, naming the authoritative counter. A
  single percentage is not a coverage claim: report the counts and say which
  component produced them, since a target validation pass and a serialized
  node count can legitimately disagree.

### Bit-accurate full-control workload binding (`X`)

- all W workload mapping, final assignment, coverage, execution, and comparison
  evidence for the same frozen `(workload, target, injection point)` tuple;
- evidence that the injection point is full-control: enumerate the variable
  decisions that materially affect the cost of each emitted call, identify
  their Voyager scheduler or mapper owner, and show how the endpoint consumes
  their canonical representation;
- bit-accurate target arithmetic, physical qparams, canonical artifacts, and
  complete zero-tolerance target output comparison;
- frozen workload-semantic reference identity, comparison boundary, and
  provenance through any compiler/converter packaging used by the endpoint;
  and
- validation of any latency, cycle, traffic, or energy model claimed in
  addition to the stage result.

Tiling, tensor allocation, loop order, scratchpad placement, transfers,
addresses, instructions, and descriptors are evidence only when they are
performance-relevant choices for the concrete point. Emitting a field that
execution may ignore is not control evidence. A legal variation is stronger
evidence when a claim depends on field causality, but is not a universal X
prerequisite. Bit-exact output through a partial-control point remains at most
W, and an operation-local exact check does not complete a workload X binding.

## Injection-contract evidence

Use the injection-point definition in the
[adoption model](../adoption-levels.md#injection-points-and-control) and the
recording requirements in the [execution contract](execution.md#injection-contract).
Record the frozen instance's path, source revision, and content hash.

Contract `pass` means observed endpoint evidence demonstrated the declared
artifact, entrypoint, completion, readback, and comparison boundary. A
stock-target probe may supply independent provenance, but only a
Voyager-generated execution can close R. Contract evidence alone does not
prove Voyager output was accepted; translation and execution remain separate
checks.

## Connected execution

A target operation using captured or reference input may prove operation
execution. It does not prove a connected workload. A connected claim requires
each non-model input to identify the declared workload input or an earlier
executed producer.

## Test surface

Every adopted backend converged on the same shape, so it is now the rule:

- one public runner per target under `test/`, invoked by the target's
  `regression.md`;
- converter-owned tests beside the converter in `conversion/<backend>/`; and
- no target dependency under `test/unit_tests/`, which stays target-neutral.

Do not build a monolithic probe or campaign controller in the source tree. A
diagnostic that is not a durable lane belongs in scratch space and is not
acceptance evidence.

## Validation ladder

1. Run syntax, schema, formatting, and scope checks.
2. Run target-neutral unit tests for shared contracts.
3. For `R`, compile one Voyager operation or composite, translate it through
   the frozen target boundary, execute it, and compare every output against the
   registered R reference under its declared tolerance. When retained, a
   stock-native run is a separate endpoint sanity check, not this comparison's
   oracle or an additional R gate.
4. For W, preserve the exact export, reference, output-code interpretation,
   fixed one-LSB rule, target, and coverage
   denominator. Try every operation kind directly, then every real metadata
   signature directly, before any required decomposition. Record each
   candidate's qscheme/calibration/qparams, serialize final FX, prove backend
   translation, freeze final assignment, execute the selected target lane, and
   exclude unsupported work from accelerator coverage.
5. Keep unsupported operations outside accelerator composites. Account for
   every composite member and report the native calls. For multi-call tiling,
   execute every generated call and compare the combined output.
6. Run connected execution only when that stronger boundary is in scope.
7. Apply the performance contract for coverage, call-count, cycle, utilization,
   or mapping-quality claims.
8. For X, require the complete same-scope W proof, a full-control injection
   point, successful endpoint execution of the artifact carrying Voyager's
   performance-relevant mapping, and zero-tolerance target output against the
   frozen workload-semantic reference. If conversion packages the reference,
   retain its direct provenance; do not invent a second oracle comparison. A
   complete same-scope X result subsumes W, so do not require a separate W-only
   regression.
9. Run full-workload accuracy only with a real connected runtime and the
   target-specific accuracy contract.

Compilation or conversion alone does not prove target execution, numerical
equivalence, connected execution, reliable mapping-sensitive performance, or
X.

## Blockers and gaps

Every non-pass in-scope gap records the precise boundary, current owner,
evidence, next action, and observable unblock condition.

- Internal compiler, target adapter, conversion, or endpoint integration gaps
  require continuation or a successor work order.
- A hardware-capability gap may terminate only the affected operation form.
- A numerical mismatch is `fail`, not `blocked`.
- A failed check does not erase semantic support. Continue at that boundary or
  record independently evidenced hardware-bug ownership.
- External stack or resource blockers require the exact attempted command and
  exhausted in-scope alternatives.
- Unknown ownership or behavior is `unverified` and non-terminal.

Completing a bounded child work order never terminates the parent adoption.

## Record results

Use the target `operations.yaml` for the current `R`/`W`/`X` stage gate and its
compact regression status. Only `status: pass` makes that stage achieved; a
failed, blocked, or unverified later gate does not erase an earlier completed
stage. Each workload check retains one hash for its
Voyager-emitted collateral and reports conversion as a named `passed/total`
count. Add a status check only for a distinct retained Voyager package, not
every converter or runtime sanity run. Converted files and runtime results are
not hash-gated. Keep commands, numerical statistics, limitations, and
maintenance lanes in the target regression document, with related gap IDs in
the operation inventory.

For W, retain each checked operation/occurrence and its qualification, target
execution separately, estimated calls, fixed-denominator coverage, per-call
latency, conversion/materialization cost, and estimated total time. State the
estimation method and provenance. A
measured runtime result is a separate stronger observation, not the definition
of W performance refinement.

For the same frozen scope and comparison boundary, a passing X exact comparison
also satisfies W's one-LSB numerical requirement. Record a fully qualified X
as the strongest stage and do not require or report a duplicate W-only
execution.

Do not collapse capability, candidate assignment, final assignment, converted
operations, executed coverage, or performance into one backend percentage.
The regression gate updates shared status only when it owns that path;
otherwise it returns evidence to integration.

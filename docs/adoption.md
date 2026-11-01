# Target and workload adoption

Two procedures cover adoption work. Start at the relevant step; neither procedure
is a mandatory campaign for every change.

| Procedure | Use when | Ordered path |
| --- | --- | --- |
| [Target registration and extension](adoption/registration.md) | A binding or capability is missing or changing | Inspect current capabilities → identify the gap → pair alternatives → bind hooks → check affected behavior |
| [Workload × target binding and refinement](adoption/workload-binding.md) | A concrete pair needs coverage or better mapping | Reuse preparation → inspect real signatures → address gaps → connect regions → validate and refine |

For a new target/workload pair, the high-level route is **target registration →
workload × target binding → coverage and performance refinement**. Binding can
reveal a missing target capability and return to registration; registration does
not require completing every possible operation before trying a workload.

These are dependency-aware guides: reuse completed work, skip inapplicable steps,
and iterate when a later check identifies an earlier problem. A small emitter fix
can start with the [emission role](roles/emission.md); it does not need a new
registration campaign. The [compilation guide](compilation.md#guide-scope-and-implementation-objectives)
owns correctness, coverage and utilization objectives. Coverage can grow while
performance improves on already mapped regions; 100% coverage is not a tuning gate.

## Start from current state

If no target implementation exists, use the [fresh-target route](adoption/registration.md#when-no-target-implementation-exists). Authoritative hardware/ISA/API facts and shared
contracts are the starting inputs; a target guide or donor is not required.

Before implementing a procedure step:

1. Inspect the selected checkout's registration and callable implementation,
   relevant tests and existing artifacts. Search for the behavior, not just the
   name used in an old report.
2. Compare that behavior with the requested scope: operation/signature, numerical
   contract, hardware, recipe, endpoint and runtime where applicable. Reuse code
   and evidence that still apply. Record reused evidence briefly in the handoff;
   do not create a persistent checklist merely to record skipped steps.
3. Implement only missing or changed behavior. A stale TODO, report or empty
   tracker is not evidence of a missing feature. If documentation conflicts with
   code, identify whether it is stale wording, a real contract violation or a new
   requirement before choosing the fix.
4. Validate affected claims. A changed recipe/runtime invalidates evidence that
   depended on it; unrelated evidence need not be recreated. Reuse is scoped by
   relevant behavior and configuration, not merely by matching dates or labels.

No status file can prove this discipline was followed. Current-source inspection
and attributable checks reduce stale-document risk; they do not eliminate it.

## Evidence follows the claim

Describe the actual result instead of assigning a stage-completion label. The optional
[adoption work record](adoption/harness.md) keeps current opportunities and checks
for a continuing task; it is not a support ledger or evidence of completion.

| Claim | Appropriate evidence and limits |
| --- | --- |
| Compiler binding exists | Current registration, discoverable recipes and relevant compilation checks; endpoint execution may still be unverified |
| Workload is mapped | Final mapping/support account, including unsupported or unknown forms; isolated kernels do not establish connected whole-model behavior |
| Numerical behavior is correct | Named reference, formats/state/tolerance and comparison boundary; distinguish host replay from actual endpoint outputs |
| Mapping decisions survive submission | Target-specific review/checks of the converter and runtime or traces for the decisions in question; exact output alone does not prove reuse or overlap |
| Performance improves | Fixed semantic/mapped work and timing scope, with estimates distinguished from applicable endpoint measurements |

Compiler-only registration can finish its requested scope without device access.
Claiming execution requires execution evidence. Exact numerical correctness,
decision ownership and measured performance are independent properties. None
implies the others. Exact arithmetic and execution-control work belong to the
[operation semantics](roles/semantics.md), [emission](roles/emission.md) and
[runtime](roles/runtime.md) roles when
needed; they are not a third adoption procedure or a prerequisite for tuning.
If the requested result combines exact behavior and full mapping control,
validate both; an operation-local check remains operation-local.

Use the selected numerical contract. A tolerance of one output-code LSB
applies only when explicitly chosen; it is not a universal numerical rule.
Exact behavior means exact target behavior at the declared boundary, including
code/bit comparison where relevant. Existing pinned contracts remain unchanged.

## Freeze the execution endpoint when relevant

The endpoint is where emitted work
enters the target stack and results can be observed. For an endpoint-dependent
claim, identify the artifact/API, converter/runtime revisions, hardware,
launch/readback path, input/output/state boundary and reference provenance.
References are validation collateral, never substitute accelerator results.

A **partial-control endpoint** hides performance-relevant choices, such as a
library privately choosing tiles. A **full-control endpoint** consumes Voyager's
choices that materially determine the mapping. Fixed hardware loops do not
automatically imply partial control, and generated code or explicit commands alone
do not prove full control. Reuse an existing endpoint description when applicable.

Either kind can support workload coverage and empirical performance modeling.
Full compiler control is needed only for a claim that promises that control.
Review commands, converter code, runtime behavior or traces according to the
target and affected assumption; no universal checker or fixed tracing format is
required. Stable native-owned behavior may be modeled, with its identity and
limits stated. Fitting must not conceal a known implementation mismatch.

## Task input and handoff

For work spanning experiments or resumptions, use the compiler-owned
[adoption harness](adoption/harness.md). Keep the best validated candidate,
open opportunities and pending checks together in task artifacts. Reconcile
that record against current source and receipts on resume.

Load the relevant role and linked contract sections, not every reader. A focused
single-agent task and a delegated task need the same concrete context:

```text
objective and scope: requested change; target/workload/endpoint as relevant
existing behavior: source bindings, applicable evidence, actual missing behavior
ownership: concerns/files to change, dependencies and selected contract links
acceptance: affected checks, comparison boundary and required gates
return: changed/reused work, reason, evidence, limitations and next owner if needed
```

Keep detailed logs in artifacts. Tightly coupled changes share the applicable
execution assumptions and an integration owner. Role separation permits scoped
delegation; it does not require a team or authorize additional actions. See
[integration](roles/integration.md).

## Target specialization

When present, load the selected [target guide](../src/voyager_compiler/targets/README.md)
for actual bindings and endpoint restrictions. Otherwise establish them from the
source specifications and record them as the implementation develops. Shared procedures
supply ordered links and ownership boundaries; they do not claim that a target
lacks an implementation. Keep support status and dated run results in the target
owner and evidence, not in this procedure.

The old R/W/X labels are retired from task routing. Exact arithmetic and execution
ownership remain relevant role concerns, not a third stage to complete. Historical
reports preserve their original terminology without prescribing the current flow.

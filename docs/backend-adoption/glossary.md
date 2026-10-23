<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Adoption vocabulary and worked examples

Read this once before your first adoption task. It summarizes the terms used
across the adoption documents and shows what each one looks like in practice.

It is not normative. Every rule lives in the
[adoption model](adoption-levels.md); every contract detail lives under
[contracts](contracts/README.md). When this file and those disagree, they win.

## Injection point and control

An **injection point** is where Voyager hands emitted work to a target path and
observes the execution result.

A **partial-control** point hides a performance-relevant mapping decision in a
native helper, runtime, or other post-emission layer. It can prove correctness
and accelerator coverage, but it cannot show that Voyager selected a
predictable, performance-oriented mapping. A **full-control** point exposes the
performance-relevant mapping to Voyager's scheduler and mapper, making reliable
per-call mapping-sensitive estimation possible. Tiling and allocation are
common evidence, not universal criteria. See the
[normative definition](adoption-levels.md#injection-points-and-control).

## R, W, X in one paragraph each

**R — register a target and injection point.** Write down what a target and its existing software
stack can already execute, derive Voyager's `HardwareConfig` from the target's
own default configuration, then prove the pipeline is connected by running one
operation end to end. R is scoped to the target and named partial- or
full-control injection point: nothing here mentions a workload.
*Done looks like:* an operation inventory, a frozen injection contract, and one
Voyager-generated operation executed on that endpoint and compared against the
registered R reference under its declared tolerance. A separate stock-target
probe is useful endpoint provenance when available, but is not another R gate.

**W — bind a workload to a target through an injection point.** Take one exact
`(workload, target, injection point)` tuple
and make as much of that workload as possible run on the accelerator, under a
fixed semantic comparison of `rtol=0`, `atol=1` output-code LSB. Start with
existing executable shared semantics, such as `conv2d_mx`; add target-specific
semantic lowering only after correct adapter and endpoint projection attributes
a greater-than-one-LSB failure to the shared arithmetic. You try every operation
kind directly first, then every real shape/dtype signature, and only then reach
for composites or semantic graph decomposition. W may reuse the target's
existing software stack at the injection point and normally does not require
detailed physical tiling, allocation, scheduling, or address generation.
Representation or legality work may be needed to inject unchanged shared
semantics; that alone is not a reason to redefine their arithmetic.
Because W permits either a partial- or full-control point, W alone proves a
coverage-proven mapping, not reliable performance-oriented mapping.
*Done looks like:* final FX and `model.txt` for the workload, a frozen list of
which operations execute on the accelerator, execution evidence, and a coverage
result. Estimated-time refinement is recorded when it is in scope.

**X — bind a workload bit-accurately through a full-control point.** W accepts
one output-code LSB; X accepts no numerical difference. X also requires that
Voyager's scheduler and mapper determine the performance-relevant mapping of
each emitted call. Target actual output is compared exactly with the frozen
workload-semantic reference at the declared boundary. That reference may be
packaged as validation collateral, but it is never an accelerator input; when
the package is a direct serialization of Voyager's reference, its retained
provenance is sufficient and there is no second reference to compare. *Done
looks like:* all W mapping and coverage obligations, exact target output,
reference provenance, and endpoint execution of the full-control artifact.
Exact output at a partial-control point is still at most W.

The stages follow a working order: for one frozen claim scope,
`X => W => R`. Report the strongest completed stage. A work order still names
the one stage it advances. Unfinished X work does not advance the achieved
stage; it remains W only when W was already complete. There is no additive
`W + X` achieved state. W is an intermediate milestone toward X, not a
parallel obligation afterward. A complete X proof over the same frozen scope
also satisfies W, so no separate W-only result is required or reported once X
passes.

## Terms you will hit immediately

| Term | Means | Concrete example |
| --- | --- | --- |
| `supported` | The logical operation has a target realization matching reference semantics **at the stage's fidelity**. A capability claim, not a test result. | `Conv2d` is `supported` because shared `conv2d_mx`, with required target projection, executed within W's fixed one-LSB rule. |
| `unsupported` | The accelerator cannot run it, so it is assumed to run on an external CPU. Voyager writes **no** CPU path for it. | An unusual `topk` the accelerator lacks: mark `unsupported`, exclude from coverage, move on. |
| `unverified` | Nobody has checked yet. Non-terminal. | A workload kind you have not reached in the inventory sweep. |
| `pass` / `fail` / `blocked` | Results of a **named check that ran**. Distinct from support. | Serialization `pass`, endpoint comparison `fail`. The operation is still `supported`. |
| `candidate_assignment` | "This *might* run on the accelerator." Exploratory. Counts as **zero** coverage. | You think 40 Conv2d occurrences can be accelerated, before proving translation. |
| `final_execution_assignment` | The serialized form has a **proven** backend translation. Frozen. | Those 40 Conv2d now emit legal backend calls, so the assignment freezes. |
| Composite (`fused_op`) | Several source operations kept as one graph node. `fused_op.op_list` **retains** every member. | `DQ -> Conv2d -> ReLU -> Q` stays one node in `model.txt`, still listing its members. |
| Custom lowering | A matched pattern **replaced by** one new target operation. The originals are erased, so there are no members. | A reshape/transpose/cat spine becomes a single `magnet_alpha::channel_shuffle` operation. |
| Multi-call tiling | Voyager splits one logical operation into several accelerator calls. | One large Conv2d becomes 4 output-channel Conv calls plus a concat, all visible in final FX. |
| `checked_accelerator_coverage` | Fraction of the frozen semantic work set that passed the required W checks. | 80 of 100 weighted work units passed, so coverage is `0.80`. |
| `estimated_calls` | Accelerator calls expected at the frozen call boundary. | 9 calls for the candidate versus 12 for the baseline. |
| `estimated_unresolved_time` | The part of the time model you **cannot** account for. Keep it rather than inventing a number. | You cannot model host sync cost, so it stays here instead of being guessed. |
| `bit_accurate_core_lowering` | An X work order: make compiler output bit-exact **before** conversion. | Fixed-point rounding moves out of the converter and into target lowering. |
| Injection contract | The frozen description of exactly what artifact the endpoint accepts and what it consumes. | "Endpoint takes converted collateral at this path, this revision, this hash." |
| Partial-control injection point | A point with a hidden post-emission performance-relevant mapping decision. | A helper receives one Conv2d call, then privately chooses tiles, buffering, and transfers. |
| Full-control injection point | A point where Voyager's scheduler/mapper determines the performance-relevant mapping of each emitted call. | The endpoint consumes Voyager-selected schedules, placement, and addresses needed by its ISA. |

## Three distinctions that cause most mistakes

**Two different things get called "fusion."** A *composite* groups operations
and keeps them: it serializes as a `fused_op` whose `op_list` still names every
member, so rules about "every member" apply to it. A *custom lowering* matches
a pattern and replaces it with one new target operation, erasing the originals;
the result is an ordinary operation with no members. Before applying any
composite rule, check which one you have — the presence of `op_list` is the
test.


**Support is not a check result.** An operation stays `supported` even when its
conversion or endpoint comparison fails. Record the failed boundary and keep
working there; do not downgrade support and do not reassign the operation.

**A composite node is not hardware fusion.** Grouping operations in the graph
proves graph structure only. It does not prove one instruction, one engine, or
any speedup. A composite may still become several accelerator calls.

That last sentence is the *neutral* rule, and a target contract may be
stricter. Magnet Alpha, for instance, requires a composite to be exactly one
Alpha call with every member Alpha-supported. Always read the selected
target's contract before assuming a composite may span calls.

## Worked example: one Conv2d through all three phases

Target `magnet-alpha`, workload `yolo5-face-nv`.

**At R** you record that the target supports `Conv2d` with given shape and dtype
limits and derive `HardwareConfig` from the target default. You may run one
stock Conv2d as an independent endpoint sanity check. You then push one
Voyager-generated Conv2d through that endpoint and compare its complete output
against the registered R reference at the declared R tolerance. You do not
mention YOLO anywhere.

**At W** you inventory the workload and find 40 Conv2d occurrences across 7
distinct shape signatures. You use shared `conv2d_mx` as the executable
semantic baseline, then try all 7 real signatures directly through the target's
existing software stack. Six translate; one exceeds a capacity limit, so you
split it into 4 calls plus a concat and record that semantic-preserving legality
projection in final FX. Because the shared arithmetic remains within one LSB,
you add no target-specific semantic lowering. You freeze the final assignment,
execute, and compare with `rtol=0`, `atol=1` output-code LSB. Coverage is
reported against a denominator frozen **before** you started refining;
estimated total time is added only when performance refinement is in scope.

**At X** you bind the workload through a full-control point where Voyager's
scheduler and mapper select every variable decision that materially affects
the cost of each emitted call. For this interface, scratchpad placement is one
such decision, so you prove the endpoint consumes it. You move exact rounding
and saturation into target core lowering until target output matches the
frozen workload-semantic reference bit-for-bit. If the converter packages that
reference for the endpoint, retain its direct provenance from Voyager's
semantic execution. The achieved stage is now X; its exact regression subsumes
the former W comparison.

## Where to go next

- Rules and phase definitions: [adoption model](adoption-levels.md)
- Planning and work orders: [coordinator contract](contracts/coordinator.md)
- Doing one bounded task: [execution contract](contracts/execution.md)
- Judging evidence: [evidence contract](contracts/evidence.md)
- Coverage and time claims: [performance contract](contracts/performance.md)
- Which files you may write: [role registry](roles/README.md)

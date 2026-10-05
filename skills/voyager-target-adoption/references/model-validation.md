# Reliable analytical models for target adoption

Use this reference for new cost/resource models and for diagnosing a search that
underperforms a simpler scheduler. Match effort to the affected capabilities;
these are diagnostic questions, not a requirement to simulate every possible case.

## Describe the execution being searched

For a candidate software tile, identify:

- Input, weight, partial-sum and final-output representations, including padding,
  wide temporaries, live generations and aliasing. Round storage for real bank,
  row and allocation rules. Sum simultaneously live demands per physical store.
- Expanded compute work: array microtiles, useful/padded work, reuse-dependent
  preloads, setup/drain, issue interval and sustained throughput. One software
  compute operation need not mean one hardware command.
- Transfers: physical bytes, rectangle/stride geometry, expanded command counts,
  recurrence across the loop nest, and when operands become ready or reusable.
- Shared resources and concurrency: bus bandwidth, endpoint/controller service,
  ports/banks and dispatch path. Two equal bandwidth numbers do not imply either
  independence or sharing; resource identities and dependencies decide.

A useful estimate distinguishes stream-fill latency, per-command overhead,
payload bandwidth and endpoint throughput. Whether terms add or overlap follows
the actual pipeline. Do not charge a pipelined setup cost serially per microtile,
or charge every command a full stream-fill latency without evidence. Conversely,
`bytes / bandwidth` alone misses fragmented transfers and command-limited work.

Capacity is necessary but not sufficient: a byte-fitting candidate can violate
bank placement or port availability. Reuse the target's sizing/alignment helpers
in search and placement, and retain final allocation checks. For resident regions,
account for retained tensors and weights over their lifetimes; remove DMA only
for edges that actually stay on chip. Logical output bytes do not substitute for
physical accumulator bytes.

## Explain overlap and finite queues concretely

Separate an async operation in the shared program from commands expanded by the
converter, commands submitted through the interface, and operations executing in
hardware. Out-of-order execution cannot issue a future load that has not entered
the hardware yet.

For example, suppose compute A expands into 100 execute commands followed in the
submitted stream by load B. A finite execute queue may block admission of that
load until much of A has completed. Reordering independent load B earlier may
recover overlap, subject to free destination storage, physical aliases and
semaphore dependencies. Smaller software tiles can shorten bursts but also
increase setup and DMA commands or reduce reuse. A larger queue does not remove
those dependencies or guarantee latency hiding.

Keep three quantities distinct: hardware queue depths; the converter's chosen
execute/load/store burst policy; and the instruction counts induced by a tile.
Only add an analytical admission/overlap term if it explains measured behavior
and corresponds to the converter's policy. Avoid an arbitrary penalty proportional
to queue size. First check whether the bufferizer exposes the next tile's transfer
and whether the converter legally moves it into a useful submission window.

## Diagnose ranking before optimizing the search

Use a small, varied candidate set, chosen from ordinary search candidates and
available competing schedules. Include the relevant contrasts: greater reuse,
smaller DMA rectangles, ragged shapes, different buffer depths/bank placement or
submission policy. Keep candidate geometry separate from its loop traversal.

| Question | Diagnosis and next action |
| --- | --- |
| Can the good schedule be represented? | Examine divisibility, batch decomposition, dataflow and builder-supported loop forms. If absent, scoring changes cannot recover it. |
| Is it rejected as illegal? | Show physical footprints and the precise hardware, lowering or policy restriction. Check for inherited Voyager assumptions. |
| Does it receive the wrong score? | Compare compute, traffic, command count, reuse, overlap and exposed startup/drain. Fix the missing resource or interpretation. |
| Does it rank well but remain unselected? | Inspect objective, tolerance, tie-breaking, enumeration, caches, deadlines and fallback paths. |
| Is the chosen plan poorly realized? | Compare its modeled sequence/buffering with emitted command order, addresses, synchronization and reuse. |

A simple scheduler is an empirical comparator, not proof that Interstellar cannot
find the schedule. Check that both use equivalent execution forms: a hardware
loop engine and a stream of explicit instructions may have different submission
costs even for the same software tile.

Record candidate mapping/order, buffer plan, legality/rejection reason, predicted
cycles, measured cycles and utilization. Within measured candidates, report
selection regret when useful: measured cycles of the model-selected candidate
divided by the minimum measured cycles minus one. This is not proof of a global
optimum. Ranking and selection quality matter even when absolute cycle error is
small. Explicitly report candidates that cannot be compared under the same
execution contract.

## Calibrate mechanisms, not desired tiles

Use narrow experiments where uncertainty matters: load-only/store-only bandwidth,
short versus long rectangles, compute-only partial rows, fresh versus retained
weights, simultaneous DMA and compute, and different bank/submission choices.
Change one mechanism at a time when diagnosing a mismatch. Separate correctness
failures from timing/model errors.

Attach measurements to hardware parameters or target-owned model assumptions,
including their operating conditions. Do not adjust a coefficient or impose a
capacity cap solely to make a known winning tile win. Test the corrected model
on shapes or candidates not used for calibration. Hardware-specific empirical
throughput and explicit conservative heuristics are acceptable; label them and
show sensitivity where it affects selection. No fixed error threshold or near-99%
utilization requirement applies universally; honor the user's agreed acceptance
criteria and investigate large avoidable gaps.

In the Gemmini adoption, useful mechanisms included INT8 operands versus INT32
accumulators, physically banked storage, command geometry, sustained array row
service with retained weights, and submission visibility of asynchronous DMA.
Bank separation and the 32/4/4 execute/load/store burst default were compiler
choices, not universal accelerator facts. The conservative pointwise workspace
allowance was a policy bound, not exact temporary-lifetime analysis. Do not copy
these values into another target's hardware model.

## Keep comparisons and claims reproducible

Use identical problem semantics, precision/rounding, padding, hardware instance,
queue depths, memory model and timing scope for local comparisons. Distinguish a
paper's reported speedup from a locally reproduced baseline. Define utilization:
useful MACs divided by peak MACs/cycle and measured cycles, for example, differs
from padded work or array-active occupancy. State whether one MAC counts as one
or two operations and keep the numerator/peak convention consistent.

Authenticate reused simulator results against commands, memory manifests, input
and output hashes, simulator binary/build, and relevant configuration. A hardware
profile name or command count alone is insufficient. Fresh compilation of an
identical stream plus authenticated old simulation is valid reused evidence;
label it rather than claiming a fresh simulation. Validate each actual output
feeding the next network segment and final numerical reference.

For segmented replay, the sum of independent accelerator cycles excludes host
staging and other inter-segment costs unless explicitly measured. Do not call it
end-to-end application latency. Retain failed/unsupported/unmeasured cases in the
coverage account. Save concise evidence in the selected repository's existing
report/results conventions so future work can retrieve it without rerunning.

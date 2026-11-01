# Reliable analytical models for target adoption

Use this reference for new cost/resource models and for diagnosing a search that
underperforms a simpler scheduler. Match effort to the affected capabilities;
these are diagnostic questions, not a requirement to simulate every possible case.
The framework does not require a universal execution-conformance checker or
fresh measurements at every registration. Select inspection, targeted checks,
traces or measurements for the target and requested claim. Analytical models and
applicable donor measurements can guide search before further characterization.

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

Use this example only when the endpoint exposes comparable command-admission
behavior. It is not a requirement that every target have these queues.

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
Ground an analytical admission/overlap term in the converter's policy and
available architecture or measurement evidence. Avoid an arbitrary penalty proportional
to queue size. First check whether the bufferizer exposes the next tile's transfer
and whether the converter legally moves it into a useful submission window.

## Check model, converter and runtime agreement

Review the actual emitted and submitted artifacts for representative legal
candidates. A frozen plan prevents mutation, but cannot prove an emitter follows
it. Equal outputs, opcode counts or total bytes do not establish equivalent reuse
or overlap.

| Modeled property | Check in realization |
| --- | --- |
| Resident operands | Loads occur at the selected loop scope and buffer generations survive all uses |
| On-chip reduction | Initialization precedes accumulation; final output is stored after the last K step; spills are explicit and priced |
| Traffic and expansion | Physical padding, strides/rectangles, commands, configuration/preload overhead and implicit native fetches agree |
| Overlapped next-tile load | The submitted load is visible early enough, has free storage and satisfies dependencies/ports/queues |
| Hardware-loop ownership | Context reservations, configuration order and implicit movement match the macro model; do not fabricate software-owned inner control |
| Runtime policy | Include wrapper commands, drains and fences in the priced boundary; preserve the selected submission policy |

Check the required partial order and reuse/admission conditions rather than
requiring identical ordering for independent commands. Include multi-tile and
split-reduction cases, buffer-depth contrasts, retained operands and competing
ISA forms where supported. Endpoint traces or suitable timing observations are needed to validate actual
admission/overlap; static order checks establish only structural agreement.

Use existing graphs/plans/command artifacts for diagnostic summaries, not a
second executable IR. Attribute evidence to hardware, recipe/model, converter,
runtime, submission policy and calibration profile identities where relevant.
Record what the chosen checks establish. Their form varies by target; a shared
conformance checker is not a required framework component or completion gate.

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

For a performance investigation, record candidate mapping/order, buffer plan,
legality/rejection reason and predicted latency; add measured latency/utilization
when those observations are available. Within measured candidates, report
selection regret when useful: measured latency of the model-selected candidate
divided by the minimum measured latency minus one, using the same unit and scope. This is not proof of a global
optimum. Ranking and selection quality matter even when absolute timing error is
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

Target-specific queue policies, bank separation choices and conservative workspace
allowances belong in that target's guide and model. They are not universal hardware
facts. Distinguish a policy bound from exact temporary-lifetime analysis, and do
not copy a historical winning setting into another target's physical model.

## Keep comparisons and claims reproducible

For equivalent-execution comparisons, use identical problem semantics,
precision/rounding, padding, hardware instance, memory model and timing scope.
For different quantization assignments, keep model/data/accuracy limits fixed and
compare their full executable costs with changed arithmetic explicit. Distinguish a
paper's reported speedup from a locally reproduced baseline. Define utilization:
useful MACs divided by peak MACs per time unit and measured elapsed time differs
from padded work or array-active occupancy. Match units: cycles with MACs/cycle,
or milliseconds with MACs/ms. Retain the endpoint observation and measurement
method; converting device time to cycles requires a justified clock. State whether one MAC counts as one
or two operations and keep the numerator/peak convention consistent.

Prior measurements can inform a model when the relevant hardware mechanisms,
execution assumptions and observation scope still apply. State that provenance
and any extrapolation; new measurements are needed when the claim exceeds that
evidence. For a claim that an exact previous simulator execution validates the
current emitted program, authenticate reused results against commands, memory manifests, input
and output hashes, simulator binary/build, and relevant configuration. A hardware
profile name or command count alone is insufficient. Fresh compilation of an
identical stream plus authenticated old simulation is valid reused evidence;
label it rather than claiming a fresh simulation. Validate each actual output
feeding the next network segment and final numerical reference.

For segmented replay, the sum of independent accelerator execution times excludes host
staging and other inter-segment costs unless explicitly measured. Do not call it
end-to-end application latency. Retain failed/unsupported/unmeasured cases in the
coverage account. Save concise evidence in the selected repository's existing
report/results conventions so future work can retrieve it without rerunning.

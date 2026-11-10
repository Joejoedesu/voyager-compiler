# Adoption work record

Use this small, compiler-owned harness for a task that spans experiments or
handoffs. It keeps optimization opportunities separate from pending validation.
It is coordination support, not a compiler extension, scheduler or evaluation
service. Current code and applicable evidence remain authoritative.

## Choose the task route

| Route | Main question | Useful opportunities and checks |
| --- | --- | --- |
| [Target registration](registration.md) | What execution can this hardware/API express through the shared compiler? | Inspect ISA/API alternatives, arithmetic, layouts, live resources and execution ownership. Pair each useful recipe with legality, model and builder. Check representative signatures and conversion/runtime assumptions. |
| [Performant workload binding](workload-binding.md) | Which supported assignment executes this workload fastest within its accuracy and coverage constraints? | Compare numerical policies with their executable lowerings, input representations, algorithms and fusion. Investigate missing coverage and dominant costs, then refine tiles and ranking. Check connected outputs and complete execution cost. |
| [Hardware–compiler co-design and DSE](hardware-codesign.md) | Which implemented hardware and mapping combination satisfies the task and best serves its objective? | Track candidate hardware/compiler identities, required support, connected validation and comparisons among feasible designs. |

Binding can reveal a missing registration capability. Follow that dependency and
return; these routes are not completion gates. Load the relevant role/contract
for the current decision instead of the entire guide set.

## Keep one task-local record

[tools/adoption.py](../../tools/adoption.py) creates, checks and summarizes a JSON
record in the task's output directory. It uses only Python's standard library:

```sh
python tools/adoption.py init results/adoption/work.json \
  --route binding --target TARGET --workload WORKLOAD \
  --objective 'Minimize execution time at the required coverage' \
  --constraints 'Fixed model, inputs, accuracy limits and timing boundary: CONTRACT'
python tools/adoption.py show results/adoption/work.json
```

Edit the record with normal file tools; designate one writer. Keep these fields
short and link detailed artifacts:

- `best`: the best validated artifact identity, scoped `claim` and `evidence`.
  Leave null until one exists. Optional `metric` contains `name`, `value`, `unit`,
  `kind` (`measured` or `estimated`) and `boundary`.
- `opportunities`: entries with `id`, `question`, `state`, and `next` action while
  open/active. Use `reason` when resolved/deferred; link `evidence` where available.
- `checks`: entries with `id`, `question`, `artifact`, `state` and `evidence` for
  passed/failed checks. Optional `job` and `owner` locate work already in progress.

For example, an opportunity asks whether changing scale grouping permits a wider
matrix operation; its next action compares accuracy and full execution cost.
A check asks whether a particular compiled artifact matches its numerical
reference. Passing that check does not resolve the optimization question.
`check` validates record structure, not numerical truth or performance claims.

## Required work and feasible designs

The current tool accepts `registration` and `binding` routes. For a co-design
task with workloads, use `binding` and describe the hardware-design scope in
`objective` and `constraints`; no additional tool route is required.

Keep required unfinished implementation distinct from optional optimization in
the entry's `question` and `next` action. Pair each required capability with a
pending acceptance check. A missing controller or workload lowering is required
work when the task depends on it; recording it as an opportunity does not make
it deferrable. Revise a candidate if necessary while preserving the task's scope.

State in `best.claim` whether the artifact is a partial development prototype or
a feasible design satisfying all hard task requirements, and name its hardware
and compiler identities. A prototype may be retained as the best validated
progress so far; it is not eligible as the final design until required support
and validation are complete. Final selection also requires the task's DSE
evidence. These are record conventions using existing fields, not additional
checks enforced by `tools/adoption.py`.

## Use it to choose the next experiment

On resume, inspect current source and the record, reconcile running jobs, and
select the next experiment likely to change the implementation decision. For
equivalent lowerings, preserve the integer contract. For different quantization
assignments, preserve model/data/accuracy limits and disclose changed arithmetic.
Compare the same required coverage and timing boundary in both cases.

Record time in the endpoint's observable unit: cycles, microseconds or milliseconds,
with measurement method, repetitions/statistic and included work in the evidence.
Keep estimates separate from measurements. Convert units only with a justified
relationship; an unknown device clock is not a reason to invent cycle counts.
Keep excluded initial host preparation visible and charge internal movement.

Freeze the source/artifact snapshot used by a running check. A separate candidate
can continue developing; attach its checks to its own identity. Promote `best`
explicitly after reviewing evidence, and retain unresolved opportunities through
intermediate handoffs. Continue required implementation and validation after
such a handoff. At completion, summarize results and optional remaining questions in the task
report; do not turn this record into a permanent target-support tracker.

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
handoffs. At completion, summarize results and remaining questions in the task
report; do not turn this record into a permanent target-support tracker.

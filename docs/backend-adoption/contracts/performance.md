<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Backend-adoption performance contract

Performance work after W is optional refinement, not part of W's semantic
completion rule. When requested, it is an explicitly qualified estimate useful
for ordering refinements such as increasing accelerator coverage, reducing
launches, avoiding conversions, and selecting a supported quantization scheme.
At a partial-control injection point, hidden native mapping decisions remain a
black-box cost or measurement; the W stage label does not make the estimate a
reliable prediction of Voyager's mapping. At a full-control point, Voyager has
the transparency needed for mapping-sensitive per-call estimation, but the
particular model still needs validation. Neither case is automatically an exact
runtime-performance claim.

Use this contract only after semantic accounting and target translation are
defined. Reference final FX and `model.txt` by artifact identity; do not
restate their graph semantics here. W candidates use the fixed `rtol=0`,
`atol=1` output-code-LSB rule when deciding whether a numerical check passes.

Semantic support and check results are separate facts (see the
[glossary](../glossary.md)). What matters here is the consequence for
coverage: an item scores **zero** checked accelerator coverage until its
required check passes, even while it correctly stays `supported`.

## Freeze the comparison

Before comparing alternatives, freeze:

- model identity and workload inputs;
- target hardware and hardware configuration;
- target injection endpoint, partial- or full-control classification, and the
  comparison boundary;
- reference implementation and fixed W output-code interpretation;
- one candidate-independent semantic work set `S` with weights `w_i`;
- denominator `D = sum(w_i)`;
- the accelerator-call boundary; and
- the estimation model, units, and provenance used for every cost component.

Use uniform weights for dynamic operation occurrences unless a different
denominator better reflects the research question. Baseline-cost weights are
valid when frozen before refinement. Never change `S`, `w_i`, or the boundary
because grouping, tiling, quantization, or accelerator assignment changed.
This freeze is scoped to one comparison. A different compilation configuration
legitimately produces a different work set and denominator; that starts a new
comparison rather than invalidating either one.
Generated calls and conversion operations are costs; they are not new semantic
work items in `S`.

Keep accelerator-unsupported workload items **in** `S`. They stay in the
denominator and contribute zero to the numerator — that is what makes coverage
honest. Dropping them from `S` inflates coverage and is the most common way
this metric gets faked.

Record the compiler and target-software revisions for every candidate. A
candidate may intentionally use a different supported quantization scheme,
calibration procedure or state, or set of finalized qparams. These are valid
performance choices because they can change accelerator coverage, call count,
and required conversions. For each candidate, record:

- qscheme and dtype choices;
- calibration inputs, procedure, and state identity;
- finalized qparam artifact identity;
- inserted requantization, packing, layout, or materialization operations; and
- the semantic and downstream check results under the frozen reference and
  one-LSB rule.

Changing the model, workload inputs, target, endpoint, reference, output-code
interpretation, or work-set denominator starts a different comparison.
Changing a declared candidate quantization choice does not, provided all
candidates are judged against those frozen terms.

## Estimate each candidate

For each candidate `x`, report:

```text
checked_accelerator_coverage(x) =
  sum(w_i for work item i that passes the required W checks) / D

estimated_calls(x) = calls expected at the frozen accelerator-call boundary

estimated_total_time(x) =
    estimated_accelerator_time
  + estimated_calls * launch_latency_per_call
  + estimated_transfer_or_materialization_time
  + estimated_host_time
  + estimated_synchronization_time
  + estimated_unresolved_time
```

The required W checks and fixed one-LSB fidelity come from the work order.
State whether coverage is based on complete target execution, an accepted
injection-point check, or another declared W qualification. Report semantic
support separately when an item fails conversion or a later check.

Derive `estimated_calls` from emitted collateral only when the target contract
guarantees that the relevant records launch one-to-one. Otherwise report the
known call count, a justified bound, or `unknown`. Keep emitted descriptor or
helper counts separate when they are not accelerator calls.

Every accelerator call introduces launch latency. Call count therefore enters
`estimated_total_time`; it is a diagnostic and model input, not a separate
higher-priority performance objective. Fewer calls can reduce time, but a
multi-call implementation can still win through greater coverage or less
accelerator, transfer, or conversion cost.

Include expected runtime requantization, layout changes, packing or unpacking,
copies, and tensor materialization in the appropriate component. Report the
offline collateral-converter wall time separately unless the declared metric
boundary intentionally includes deployment preparation.

Give every component a source: target documentation, an analytical model, a
trace-derived constant, or a stated assumption. Use ranges when the available
evidence does not justify a point estimate. At a partial-control point, do not
invent native schedules, traffic, or memory events to close the model; retain
their cost as `estimated_unresolved_time` or use a clearly labeled endpoint
measurement. Do not invent external-CPU time either.

## Compare candidates

For baseline `b` and candidate `c`, report:

```text
coverage_gain =
  checked_accelerator_coverage(c) - checked_accelerator_coverage(b)

estimated_call_reduction = estimated_calls(b) - estimated_calls(c)
estimated_time_reduction = estimated_total_time(b) - estimated_total_time(c)
estimated_speedup = estimated_total_time(b) / estimated_total_time(c)
```

Default ordering:

1. maximize checked accelerator coverage over the fixed semantic work set;
2. at equal coverage, minimize estimated total time.

This ordering already accounts for launch count through estimated total time.
Preserve the complete metric vector and candidate quantization state when a
coverage gain costs more estimated time or when a time gain changes calls or
conversion work.

Example: if `D = 100`, a baseline covers 60 work units in 12 estimated calls
and 1000 estimated cycles, while a candidate covers 80 in 9 calls and 900
cycles, report coverage gain `0.20`, estimated call reduction `3`, estimated
time reduction `100` cycles, and estimated speedup `1.11x`. If the candidate
instead takes 1100 estimated cycles, it still wins the first ordered objective;
retain the coverage/time tradeoff rather than collapsing it into one score.

## Validate with measured runtime when available

Measured runtime can validate or calibrate an estimate, but it is a separate
and stronger observation. A black-box measurement at a partial-control point
does not prove that Voyager predicted or controlled the mapping. For a measured
claim:

- use the same model, inputs, target, endpoint, reference, tolerance, work set,
  and observation boundary;
- report the timing source, units, repetitions, aggregation, and actual calls;
- report measured total time separately from estimated total time; and
- report estimation error instead of silently relabeling an estimate as a
  measurement.

Use `estimated_speedup` for modeled results and `measured_speedup` only for
matched runtime observations. Decompose measured time only into non-overlapping
components supported by endpoint counters or traces. Put missing observations
in an unexplained residual.

## Diagnose and refine

Report the largest current cost component, its evidence source, and the next
bounded refinement. Useful categories are:

- unchecked or accelerator-unsupported workload coverage;
- call or launch overhead;
- accelerator execution cost;
- transfer, conversion, or materialization cost;
- host or synchronization cost; and
- unresolved estimation or measurement gaps.

Do not infer an inner schedule or memory cause from an aggregate estimate or
total-time measurement alone.

## Optional utilization

Claim utilization only when all terms are qualified:

```text
utilization = useful accelerator work
              / (qualified peak rate * accelerator-active interval)
```

Record precision, sparsity assumptions, peak provenance, interval, and whether
the result is measured or modeled. Utilization diagnoses performance; it does
not replace checked accelerator coverage or estimated total time.

## Ownership limit

A W estimate never claims the target consumed an incidental compiler fact.
Attribute hidden native-selected behavior to the native stack and classify the
point as partial-control when that behavior materially affects mapping cost. A
full-control point requires the endpoint to consume Voyager's
performance-relevant scheduling and mapping decisions. That control is
necessary but not sufficient for X: the same workload must also meet X's W,
exactness, and workload-reference provenance obligations. See
[adoption model, X](../adoption-levels.md#x-bit-accurate-full-control-workload-binding).

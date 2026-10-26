# Performance models

Use for: Fit/cost estimates, candidate ranking, utilization assumptions and model corrections.

## Binding and ownership

`TargetSpec.model / ImplementationSpec.model / ImplementationSpec.estimate`. Estimate fit/cost for candidate selection. Measurement acquisition belongs to performance characterization.

## Load for this task

Same mapped semantic work and numerical scope, hardware, recipe/policy, builder/emitter/runtime behavior, estimates and any attributable observations.

Read the relevant [shared contract](../compilation.md#execution-model-and-converter-consistency) and the selected
target's bindings. Start with these source owners; follow imports as needed:

- [RuntimeModel and OpModel](../../src/voyager_compiler/extensions.py)
- [Measurements, fitting and holdouts](../../src/voyager_compiler/performance/)
- [Target models and execution](../../src/voyager_compiler/targets/)

## Starting a model for a new target

No existing target model is required. Start from hardware/ISA facts and the builder's
execution structure. Choose shared matrix, grid or attention search when it represents
the needed choices; use recipe candidates and a whole-region estimate otherwise.
Interstellar enumerates mappings but calls the target's fit/latency model.

Start implementing at [Create and register the model](../compilation/performance-models.md#create-and-register-the-model)
only when implementing that part. A default target model is an optional example,
not a prerequisite or a universal formula.

## Implementation decisions

| Problem | Next action |
| --- | --- |
| No model exists | Use the linked construction guide: choose a representable search space, define placement/fit and price actual compute/movement/control. |
| Reusing a model is attractive | Check its capacities, layouts, bandwidths, startup, partial sums and overlap against this builder/runtime. Reuse verified components, not a target label or fitted coefficient without provenance. |
| Predictions look plausible but choices are slow | Distinguish missing/pruned alternatives from wrong cost ranking and wrong execution. Compare forced candidates before fitting; a constant correction cannot recover an absent candidate. |

## Do

1. Separate unavailable candidates, legality failures, scoring errors, selection and realization. Compare representative forced candidates, including known good schedules.

2. Trace the relevant execution assumption through builder, emission and submission. A repeated load or early store changes modeled reuse even if host outputs match.

3. Price target-specific transfers, placement, scratch and overlap. Keep physical facts separate from fitted coefficients. Fix structural mismatches before fitting.

4. Use analytical models for ordinary search. Reuse applicable donor measurements; when tuning needs new evidence use the existing characterize/fitting path and held-out cases, assessing ranking as well as error.

## Do not

- Fit coefficients to conceal wrong transfer placement, missing computation or a known realization defect.
- Present synthetic/host timing or padded busy activity as measured useful device performance.
- Require fresh characterization for unrelated registration work.

Measurements are supplied by the [characterization role](characterization.md);
model fitting consumes that evidence.

## Handoff and validation

Return changed assumptions/formulas, recipe/policy identities, estimated versus measured results, evidence applicability and unresolved cost. Handoff movement changes with the builder/emitter contract.

Choose code review, targeted commands, traces or timings for the actual claim. Do not equate static instruction-site totals, host runtime or synthetic samples with device performance.

[Role index](../roles.md) · [Adoption and task handoffs](../adoption.md)

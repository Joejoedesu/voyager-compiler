# Implementation alternatives

Use for: Another ISA form, lowering recipe, unsupported signature or missing alternative.

## Binding and ownership

`TargetSpec.implementations / ImplementationSpec.matches / ImplementationSpec.candidates / ImplementationSpec.validate`. Assemble paired recipes and expose alternatives. Model and builder readers detail their own callbacks.

## Load for this task

Semantic operation and restrictions, hardware/endpoint, existing recipes and the requested alternative.

Read the relevant [shared contract](../compilation.md#search-and-reusable-bufferized-kernels) and the selected
target's bindings. Start with these source owners; follow imports as needed:

- [ImplementationSpec](../../src/voyager_compiler/extensions.py)
- [Candidate selection](../../src/voyager_compiler/compiler/search/planner.py)
- [Paired target models/builders](../../src/voyager_compiler/targets/)

## Implementation decisions

| Problem | Next action |
| --- | --- |
| ISA/API has several equivalent forms | For each, identify legal shapes/dtypes, layouts, tails, accumulator behavior, memory/control ownership and expected movement. Register overlapping recipes with paired builders/costs. |
| A shape or dtype changes the fastest form | Expose the meaningful parameters, include boundary/resource-limited cases, and keep the alternative reachable. Check pruning/budgets before blaming the cost model. |
| Another algorithm saves memory by changing computation | Keep the same observable region; pair a new builder with its full cost, including recomputation. Reuse inner tile engines; arbitrary overlapping-region search needs framework work. |

## Do

1. Decide whether an existing bufferized template plus target primitives expresses the form. Add a new builder for changed algorithmic control; shared IR changes need a missing semantic capability.

2. Pair matches/candidates, model/estimate, build and validation for each recipe. Alternatives need finite comparable costs and independent candidate metadata.

3. Expose legal modes to discovery/search. Equivalent instruction families may differ in resource/control ownership; each needs a paired model and realization.

4. Carry selected identity/parameters through emission. Distinguish unsupported semantics, rejected legality and a candidate losing on cost. Do not silently fall back to a donor path.

## Do not

- Hide an ISA choice in first-match dispatch or emission; expose competing forms with paired model/builders.
- Create a new bufferized path for an API spelling when an existing template already expresses its control structure.

Keep new recipes' `cache_by_shape=False` unless the structural/shape/dtype key
covers all candidate decisions; see the [recipe contract](../extensions.md#implementation-scheduling-and-buffer-lowering).

## Handoff and validation

Return the recipe, restrictions, alternative/rejection evidence, paired model/builder/emitter changes and comparison cases. Handoff actual allocation failures to storage and scoring issues to performance.

Force each relevant alternative, compare semantics and emitted mode, then check normal selection. Use the affected target regression; candidate enumeration alone does not prove realization.

[Role index](../roles.md) · [Adoption and task handoffs](../adoption.md)

# Shared search engines

Use for: Candidate enumeration, cache behavior or mapping selection that the existing engine cannot express.

## Binding and ownership

`Framework: compiler/search/`. This is framework development, not a registered scheduler-engine plugin. Target alternatives use ImplementationSpec.
For a new target, first follow the [model bootstrap and reuse guide](performance.md#starting-a-model-for-a-new-target). Reusing a search engine does not require inheriting another target's cost formulas.

## Load for this task

Candidate space, constraints, cost units/objective and a reproducible selection problem with known alternatives.

Read the relevant [shared contract](../compilation.md#scheduling-and-the-general-performance-model) and the selected
target's bindings. Start with these source owners; follow imports as needed:

- [Hierarchical matrix search](../../src/voyager_compiler/compiler/search/interstellar.py)
- [Dimension-wise tiles](../../src/voyager_compiler/compiler/search/tile_shapes.py)
- [Attention tiles](../../src/voyager_compiler/compiler/search/attention_tiles.py)
- [Shared entry](../../src/voyager_compiler/compiler/search/tiler.py)

## Implementation decisions

| Problem | Next action |
| --- | --- |
| Good implementation never appears | Check recipe matching, legality, candidate pruning, budget and constraints. A correct cost cannot choose a candidate that is absent. |
| Candidates exist but selection is poor | Check fit/cost units, objective/tolerance and profile/cache identity. Route bad execution assumptions to model/builder owners before changing enumeration. |
| Algorithm needs coupled choices the current space lacks | Try recipe parameters and nested existing engines first. Extend the shared result/engine contract only for the concrete missing decisions and their consumers. |

## Do

1. Check representability and legality before changing search heuristics. A poor model or unregistered recipe is not necessarily an engine failure.

2. Preserve named factors, placements, scratch and TileResult cost. Keep cache keys sensitive to relevant hardware, policy and profile changes.

3. Keep deterministic selection and bounded capacity retries. Search proposes mappings; final expanded allocation validates storage.

4. Use the existing shared search contract. Independent scheduler-engine registration and richer event/result contracts need explicit shared interfaces; a target option is not such an interface.

## Do not

- Generate target instructions or final addresses, or smuggle target-specific dispatch into a shared engine.
- Change hardware capacities or bypass legality to make a candidate win; send model/builder mismatches to their owners.

## Handoff and validation

Return the candidate-space/selection change, cache/constraint reasoning and evidence against forced alternatives. Coordinate cost interpretation with performance and realization with builders.

Check the meaningful changed space or constraint and deterministic choice. Validate a known feasible/good candidate remains reachable; avoid tests that merely mirror enumeration code.

[Role index](../roles.md) · [Adoption and task handoffs](../adoption.md)

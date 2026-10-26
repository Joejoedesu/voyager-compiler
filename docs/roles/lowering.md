# Custom-operation lowering

Use for: Map normalized tensor operations into target operations before search.

## Extension and inputs

Binding: `TargetSpec.lower`.

Normalized graph, selected layouts, qparams and supported target operation definitions.

Read the [extension catalog](../extensions.md) and the relevant
[compiler contract](../compilation.md#the-compilation-flow).
Source: [pipeline.py](../../src/voyager_compiler/pipeline.py).

## Implementation decisions

| Problem | Next action |
| --- | --- |
| Normalized ops need a target arithmetic form | Consume established layouts/qparams, rewrite the supported signature, then compare against the pre-lowering reference. |
| One logical op needs several target calls | Preserve the region input/output/state boundary and reconstruction. Use a recipe builder for schedule-dependent splitting; fixed semantic decomposition can remain here. |
| Two target realizations are both legal | Keep a common operation/region for their recipes. Do not destroy the alternative by selecting an ISA mode in this hook. |

## Do

- Use `lower(prepared, config, options)` after shared normalization and before search.
- Consume established operation definitions, scales and layouts; preserve logical outputs and state.
- Leave competing ISA/algorithm realizations discoverable through implementation recipes.

## Do not

- Do not recalibrate, redo layout selection, select software tiles or allocate buffers here.
- Do not confuse this graph hook with ImplementationSpec.build, which constructs bufferized execution for a candidate.

## Handoff and validation

Return explicit target operations for matching and implementation search. Validate the normalized-to-target arithmetic boundary independently of emitted output.

[Role index](../roles.md) · [Shared transformations](../compilation/rewrites.md)

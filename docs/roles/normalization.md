# Normalization policy

Use for: Choose shared layout, alignment and graph-fusion options.

## Extension and inputs

Binding: `TargetSpec.normalization`.

Legal tensor graph, target layout/pattern policy and retained-constant/storage constraints.

Read the [extension catalog](../extensions.md) and the relevant
[compiler contract](../compilation.md#related-optimizations-across-the-pipeline).
Source: [compiler/normalize.py](../../src/voyager_compiler/compiler/normalize.py).

## Implementation decisions

| Problem | Next action |
| --- | --- |
| Layout or alignment prevents a good mapping | Check operand layout and native tail capabilities; select existing options and measure remaining transpose/padding work. Inspect pass guards before adding a transform. |
| Static weights/scales require a reshape or transpose | Request a graph transform and let shared constant preparation decide immutability and retained-byte accounting. Do not assume every parameter transform is free. |
| Fusion removes an intermediate but changes execution choices | Check escaping users, Q/DQ and instruction legality. Group equivalent computations without prematurely selecting competing algorithms; use recipes for that choice. |

## Do

- Bind normalization option pairs. The shared normalizer owns pass order; caller normalize_options override target defaults.
- Keep broad fusion patterns target-owned and reuse the same table in frontends and tests.
- Let shared constant preparation charge retained representations. Use the padding reference to separate alignment, tile filling and physical storage.

## Do not

- Do not derive fusion policy from hardware topology, or treat the pattern table as support for every signature.
- Do not run another layout optimizer in quantization, builders or emission. Do not disable alignment without a supported tail implementation.

## Handoff and validation

Return the normalized graph to custom-operation lowering and search. Compare valid outputs and inspect layouts/materialization; reduced graph size alone is not a performance result.

[Role index](../roles.md) · [Shared transformations](../compilation/rewrites.md)

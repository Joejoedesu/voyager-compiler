# Target runtime

Use for: Submit and execute the emitted program at a specific endpoint.

## Extension and inputs

Binding: `TargetSpec.runtime`.

CompilationResult, arguments/state, emitted artifacts, endpoint/ABI revision and numerical reference.

Read the [extension catalog](../extensions.md) and the relevant
[compiler contract](../compilation.md#execution-model-and-converter-consistency).
Source: [runtime.py](../../src/voyager_compiler/runtime.py).

## Implementation decisions

| Problem | Next action |
| --- | --- |
| Target has no runtime adapter | Identify launch, input transfer, output/state readback and completion API. Submit the emitted artifact for one small case; do not substitute a vendor demo or host result. |
| Runtime inserts transfers or synchronization | Compare their scope to the selected execution. Separate necessary ABI work from a loss of intended residency/overlap. |
| Timing or state varies across calls | Check warmup, synchronization, persistent state and batching; make the execution boundary explicit before handing timing to characterization. |

## Do

- Implement `runtime(result, args, kwargs)` to return outputs. Call
  `run(result, target=target, reference=expected, rtol=..., atol=...)` explicitly;
  omitting the target selects host replay, and omitting the reference skips comparison.
  See [execution/comparison](../extensions.md#execution-and-comparison-are-explicit).
- Preserve selected command order, transfers, waits and declared native ownership. Check runtime-inserted moves against reuse/overlap assumptions.
- Record endpoint identity and comparison boundary. The run helper's elapsed
  seconds include host overhead; hand attributable device timing to characterization.

## Do not

- Do not reselect tiles/ISA or replace intermediate values with reference outputs.
- Do not call host replay device execution or treat native invocation as asynchronously complete without a token contract.

## Handoff and validation

Return outputs/state and endpoint evidence. Validate submitted behavior and numerical results; emission syntax, host replay and measured device execution are distinct claims.

[Role index](../roles.md) · [Shared transformations](../compilation/rewrites.md)

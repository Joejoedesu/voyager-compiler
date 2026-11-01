# Input preprocessing

Use for: Improve mapping and hardware utilization by changing the input
representation or operation graph before quantization.

## Binding and inputs

`TargetSpec.input_preprocessing`, with a per-compilation override. Read the
[contract](../extensions.md#input-preprocessing) for ordering, samples and references.
Load the graph, input boundary, target capabilities and calibration source.
Sources: [dispatch](../../src/voyager_compiler/compiler/preprocess.py) and
[shared transforms](../compilation/rewrites.md#input-boundary-rewrites).

## Implementation decisions

| Problem | Action |
| --- | --- |
| Input representation wastes hardware capacity or work | Reuse a graph rewrite and compare complete compilations, including packing and movement. |
| Caller can supply prepared inputs | Declare the input contract, adapt calibration and validate deployment preparation. |
| Preparation must execute on the accelerator | Keep it in the graph; builders can materialize tiles or absorb work into loads. |
| Only immutable weights/scales need reformatting | Use normalization and shared constant preparation. |

## Do

- Preserve outputs/state, matching examples/calibration and the chosen input contract.
- Reuse layouts and transformations; declare temporary storage and expanded constants.
- Compare useful workload utilization and complete cost under the same accuracy
  and timing boundary; follow [performance diagnosis](../compilation/performance-models.md#required-diagnosis-when-utilization-or-selected-performance-is-poor).

## Do not

- Hide internal computation in host preparation or assume packing is free.
- Quantize, select final tiles/addresses, duplicate frontend rewrites or replace constant folding.

## Handoff and validation

Pass the executable graph and input contract to quantization. Check rewritten
outputs against `result.original_reference`, then quantized accuracy and the
actual deployed input path. Report remaining packing, traffic and storage costs.

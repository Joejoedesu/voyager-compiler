# Bufferized kernel construction

Use for: Kernel reuse, loops/copies, stateful control, residency or native invocation construction.

## Binding and ownership

`ImplementationSpec.build`. Construct a trial candidate using shared templates; the planner selects among the results. TargetSpec.refine has a separate reader for post-build refinement.

## Load for this task

Candidate parameters and any provisional `TileResult`, tensor-operation graph and effects,
target primitive semantics and resource reservations. See the
[graph/decision definitions](../compilation.md#tensor-graph-execution-graph-and-compilation-decisions).

Read the relevant [shared contract](../compilation.md#search-and-reusable-bufferized-kernels) and the selected
target's bindings. Start with these source owners; follow imports as needed:

- [Builders, movement and residency](../../src/voyager_compiler/compiler/bufferize/)
- [Buffer and invocation contracts](../../src/voyager_compiler/ir/buffer.py)
- [Target kernel builders](../../src/voyager_compiler/targets/)

## Implementation decisions

| Problem | Next action |
| --- | --- |
| Existing template almost fits | Compare its loop/reduction lifetime, copy placement, layout and primitive hooks. Specialize the smallest supported hook; do not copy the whole scheduling pipeline. |
| Target has multiple engines or distinct local memories | Declare where operands, partial sums and temporaries live, when transfers happen and which engines/dependencies permit overlap. Pair every extra buffer or transpose with the model. |
| Native macro hides internal control, or algorithm has a new loop structure | Use explicit invocation ownership for native work. Add a builder when shared control cannot express the algorithm; unsupported effects/completion require a scoped IR change. |

For a fixed region assembled from sequential kernels, reuse
[`build_sequential_region`](../compilation/algorithm-search.md#reuse-sequential-component-kernels).
It returns bufferized FX with ordered component choices and a complete sequential
cost. A streaming or recomputation schedule needs its own builder/model; summing
component latencies does not describe overlapping execution.

Grouped/depthwise convolution describes connectivity, not an execution unit.
Select a group-aware or native recipe. The shared dense convolution template
cannot split independent groups as a dense C reduction; `whole_tensor=True`
explicitly reuses its whole-operation form, including fused epilogues. Target
models decide search support; the shared search does not skip depthwise ops.

## Do

1. Reuse templates when differences are layout, placement or primitive realization. Add a builder for a different algorithmic control structure, not for every PyTorch spelling.

2. Materialize explicit buffer roots, views, copies, loops and dependencies. Preserve combined outputs when splitting a logical operation.

3. Residency must preserve observable stores, including consumers through a backing/view. Respect complete transfer matching, live capacity, reduction lifetime and asynchronous completion.

4. Native macros declare private memory/control ownership. Keep them visible in the selected plan and cost. Builders do not own a competing final allocator or hidden fallback pipeline.

Use the [async/native completion contract](../compilation.md#asynchronous-buffer-operations-and-native-completion)
for waits and lifetime boundaries, and the [padding guide](../compilation/padding.md) to
distinguish tile/halo filling from shape legalization and physical storage.

## Do not

- Re-select tiles or ISA mode behind the planner, or create a competing final allocator.
- Drop observable stores/waits or treat native submission as asynchronous completion without the required completion contract.

## Handoff and validation

Return executable buffers/control, scratch/native reservations, changed movement and semantic checks. Handoff actual placement to allocation and modeled movement to performance/emission.

Replay affected kernels/regions, check state and view-observable results, and verify allocation constraints. Include emitted/submitted checks when changing target execution assumptions.

[Role index](../roles.md) · [Adoption and task handoffs](../adoption.md)

When reusing a search engine, separately check that this builder can realize its
full mapping: spatial tiling, reduction order, buffering and retained operands.
The shared matrix builder's reduction-inner restriction is not a universal target
constraint. Custom builders can use [the matrix search adapter](../compilation/performance-models.md#target-owned-matrix-search-adapters).

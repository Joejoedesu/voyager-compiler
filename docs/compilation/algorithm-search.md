# Compare algorithms for one region

An operation or grouped FX subgraph defines **what must be computed**. Its
implementation recipes define **how to compute it**. The external operands,
outputs, effects and numerical requirements stay fixed while intermediate
storage, recomputation, loops and local mappings may differ.

## Extension ownership

```text
Recognized operation / fixed region
  → TargetSpec.implementations
      → recipe A: build + component tile searches + complete cost
      → recipe B: different build + paired search/model + complete cost
  → shared planner compares legal candidates
  → ImplementationChoice (recipe, parameters, cost, component choices)
  → winning bufferized graph → refinement → allocation → freeze → emission
```

Use `ImplementationSpec` for each algorithm; there is no separate algorithm
registry or mandatory new graph IR. `matches` checks the computation, `candidates`
enumerates algorithm parameters, `build` constructs execution, and `estimate`
prices the complete region when one `TileResult.cost` cannot describe it.
Competing recipes need finite costs in the same target units. The configured
runtime tolerance still permits a lower-traffic candidate near the fastest one;
set it to zero for strict modeled-latency selection.

Normalization owns recognition and the observable region boundary. A region is
not allowed to hide an intermediate that another consumer needs. Arbitrary
overlapping region partitions and rewrite-sequence search remain outside this
fixed-region comparison. Start with an existing operation when its contract fits.

## Reuse sequential component kernels

[`build_sequential_region`](../../src/voyager_compiler/compiler/bufferize/regions.py)
accepts a normalized, shape-propagated FX graph with flat tuple/list outputs, a
build context, and an explicit set of component implementation recipes. It copies
the graph, searches/builds its kernels, and returns bufferized FX. Do not include
the outer recipe recursively in its component set.

The standard component builders finish their transfers/computation before the
next kernel executes. `estimate_sequential_region` therefore sums their complete
latencies and, when known for every component, their DRAM traffic. Missing cost
rejects the candidate. This helper does not model overlap between component
kernels. An algorithm that overlaps stages needs a matching builder and model.

The returned graph carries `region_choices` and `region_cost`. The planner saves
the ordered component choices in `ImplementationChoice.components`; the outer
cost already includes them. Reporting must not add them again. The expanded graph
retains actual buffers and movement; the shared allocator still determines final
addresses, banks and lifetimes. A `NoLegalMapping` rejects a candidate while other
algorithms remain searchable; unexpected model/builder errors still propagate.

Component kernel scopes remain distinct beneath the region. Their bank groups,
mapping records and reporting identities must not collapse into one kernel.
An allocation failure in a child scope replans its nearest selected region owner.

## Attention example and present scope

The shared [`attention_implementations`](../../src/voyager_compiler/compiler/bufferize/attention_algorithms.py)
factory provides two recipes for an existing unquantized SDPA operation:

- **Materialized:** reuse GEMM, scaling and softmax kernels; keep the full
  intermediate tensors between kernels. Shared per-operation padding restores
  logical shapes before softmax, so padding does not add keys to its denominator.
- **Streaming:** reuse the FA3 builder and attention tile search, keeping running
  softmax/output state instead of the full intermediate matrix. Its model must
  price that execution structure, not the sequential baseline.

This comparison currently accepts FP32 rank 2–4 operands with matching batch/head
dimensions and equal Q/K/V head widths, no masks, dropout, causal mode or GQA.
Shapes without a legal streaming product mapping may still use materialization.
The [default target guide](../../src/voyager_compiler/targets/voyager/README.md#opt-in-attention-algorithm-comparison)
shows opt-in registration; its default recipe remains unchanged.

Deliberate recomputation is another recipe, not a synonym for streaming. A future
builder can recompute scores across passes to save storage; it must count repeated
products and transfers and preserve the numerical boundary. That recipe and
recognition of decomposed `matmul → softmax → matmul` are not implemented here.

## Validate an addition

Force each algorithm through build, allocation and host replay against the same
reference. Exercise supported shapes/tails and reject unsupported numerics.
Then enable comparison: inspect alternatives, full costs, component mappings,
rejection reasons and the selected recipe. Verify a cost change can change the
winner and an infeasible mapping does not hide a valid alternative. Host replay
does not establish device performance; validate emitted/runtime execution before
claiming measured speed or fitting its model.

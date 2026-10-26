# Building and reusing performance models

Use for: choosing reusable search/model components or implementing a model from
hardware facts. The [performance role](../roles/performance.md) owns decisions;
this reference supplies the construction details.

> **Creating a model from scratch? Start at [Create and register the model](#create-and-register-the-model).**
> Produce a required target model factory, compatible search/recipe choices,
> legality and placement checks, comparable costs, and evidence that the builder
> realizes those assumptions. Measurements and fitting follow when the claim needs them.

## Starting a model for a new target

This route works without an existing target implementation. Start from the shared
interfaces and the requested hardware/ISA documentation. An installed default
Voyager model is an optional example, not a prerequisite. Inspect the source
before copying any model. Reuse is a decision about execution assumptions, not
whether two targets both have matrix multiplication.

### Choose what to reuse

| Situation | Reuse | Target-owned work |
| --- | --- | --- |
| Hierarchical GEMM/conv mapping matches the builder | Interstellar search through `compiler/search/interstellar.py`, `MatrixSpace`, shared GEMM/conv builders where applicable | Hardware resource/schedule description, operation `fit` and `cost`, scratch and supported restrictions |
| Dimension-wise tiling fits a vector/reduction kernel | Shared grid search and compatible bufferized builder | `GridSpace`, capacity/placement checks and timing for the actual reduction/vector execution |
| Streaming attention matches the existing execution family | Attention tile search, nested product mappings and compatible builder | `prepare_attention`, `AttentionSpace`, state/scratch requirements and coupled costs |
| A native macro or another algorithm has its own candidate parameters | Implementation recipe planner; shared components where applicable | `candidates`, `build`, legality and a whole-region `estimate`; an Interstellar mapping is not mandatory |
| An existing model has compatible movement/control but some different costs | Only the verified formulas or model components | Override differing capacity, compute, transfer, startup and overlap assumptions; retain explicit provenance for fitted data |

Interstellar is not the target latency model in the current adapter. Its runtime
callback calls `OpModel.cost(mapping, placement).latency`; its capacity callback
calls `OpModel.fit(mapping)`. Interstellar still supplies mapping enumeration,
architecture-based energy/access accounting and selection. Its energy ranking
also needs appropriate resource/cost parameters; borrowed energy coefficients
are not measured target energy. There is no universal calibrated accelerator
model obtained simply by changing the array dimensions.

Reuse the default Voyager model as an implementation example, not an automatic
base class for every target. Its bank behavior, formats, vector primitives and
transfer/overlap assumptions are target-specific. A helper is reusable only after
checking those dependencies. Extract a shared helper when a concrete second
consumer establishes a common contract; do not create forwarding wrappers or a
large abstract model hierarchy in advance.

### Create and register the model

1. **Define the supported execution.** Start with the operations/signatures needed
   by the requested workload. Pair the model with the actual builder and endpoint
   behavior: data movement, accumulator lifetime, buffering, padding, scratch and
   overlap. Do not require complete operation coverage before providing a useful
   analytical model.
2. **Choose the search representation.** `RuntimeModel.prepare(node)` returns an
   `OpModel` with a compatible `MatrixSpace` or `GridSpace`; attention uses
   `prepare_attention`. A recipe can instead supply its own candidates and
   whole-region estimate. Use an existing space only when its choices can be
   realized by the builder. General engine registration is not a current target
   hook; see the [search role](../roles/search.md).
3. **Implement fit and placement.** `fit(tiling)` returns a `Placement` or `None`.
   Include live operands, accumulators, packed data, buffer slots and scratch.
   The final shared allocator still validates expanded capacity, banks and
   lifetimes. In the matrix adapter, fit runs before loop ordering/partitioning;
   it may depend on per-level total extents, not the later loop order. The current
   matrix builder also requires the L3 reduction loop to be innermost among the
   outer grid loops. These are constraints of this adapter/builder, not universal
   accelerator requirements. Use recipe-local candidates or a scoped shared
   extension if the target's required execution cannot be represented.
4. **Implement costs in consistent units.** `cost(tiling, placement)` returns
   `Cost`, with whole-operation latency in target cycles and traffic when used
   for selection. Supply per-step `compute` when nested attention/reporting
   consumes it. Include startup, repeated transfers, tails, reduction work and
   unavoidable materialization. Sum serialized work; overlap only what the
   builder/runtime can actually overlap. `operations / throughput` and
   `bytes / bandwidth` are initial components, not a complete universal formula.
5. **Register and preserve identity.** Always bind `TargetSpec.model`; the pipeline
   constructs it even for recipes that use a whole-region estimate. A native-only
   target can return no local tiling model from `prepare` while its recipes
   provide complete legality/build/cost. `ImplementationSpec.model` overrides
   the target model for an alternative execution family. Use
   `ImplementationSpec.estimate` for the complete region when a local tile cost
   does not cover it. Competing recipes need finite comparable costs. Cache keys
   must distinguish relevant hardware, formats, execution policy and model/profile
   changes. Matrix search workers consume plain model data; do not retain FX
   nodes or dispatch Torch operations in those worker callbacks.
6. **Check applicability, then fit if needed.** Check known feasible and infeasible
   candidates, force representative alternatives, and trace modeled movement
   through builder and emitted/runtime work. Test ranking under controlled costs
   independently of numerical correctness. Without hardware measurements, label
   the result analytical. Reuse applicable measurements with provenance, or use
   the [characterization role](../roles/characterization.md) and `performance/` to fit
   corrections and evaluate held-out rankings when performance tuning needs it.

The default runtime tolerance is 2%: matrix search considers energy among
near-fastest mappings; grid/recipe selection generally uses traffic (some grid
spaces use relative ties). A zero tolerance requests the fastest modeled result,
with secondary criteria among equal-latency choices. Selection policy does not
repair an incorrect model or guarantee fastest measured hardware performance.

Source references: [model interfaces](../../src/voyager_compiler/extensions.py),
[search/result records](../../src/voyager_compiler/ir/plan.py),
[Interstellar adapter](../../src/voyager_compiler/compiler/search/interstellar.py),
[default model assembly](../../src/voyager_compiler/targets/voyager/implementations/model.py),
[recipe planner](../../src/voyager_compiler/compiler/search/planner.py).


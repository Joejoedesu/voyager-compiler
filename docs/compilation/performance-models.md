# Building and reusing performance models

Use for: choosing reusable search/model components or implementing a model from
hardware facts. The [performance role](../roles/performance.md) owns decisions;
this reference supplies the construction details.

> **Creating a model from scratch? Start at [Create and register the model](#create-and-register-the-model).**
> Produce a required target model factory, compatible search/recipe choices,
> legality and placement checks, comparable costs, and evidence that the builder
> realizes those assumptions. Measurements and fitting follow when the claim needs them.

## Timing units and observation scope

Measure what the endpoint exposes: device elapsed time may be in microseconds or
milliseconds, while a cycle-accurate simulator can report cycles. Retain the unit,
completion boundary, repetitions/statistic and included work with each result.
Use comparable units and scope when ranking candidates; do not infer device cycles
from an unknown clock or treat host submission time as completed device time.

The existing `Cost` documentation and cycle-based `performance/tuning.py` fitter
use target cycles; the fitter requires a known `clock_ghz`. This does not require
all targets to measure cycles. For a time-only endpoint, retain measured time and
use a target-owned empirical model/adapter, documenting the conversion into its
search cost convention and any cycle-assuming consumers it bypasses. Do not feed
fabricated clocks to the cycle fitter. The
[adoption harness](../adoption/harness.md) records timing with explicit units and
does not invoke that fitter or convert measurements.

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

The default runtime tolerance is **0**: select the fastest modeled result.
Traffic/energy can break exact latency ties, but cannot buy a 1–2% slowdown.
A positive `runtime_tolerance` is an explicit opt-in to near-fastest energy
(matrix) or traffic (grid/recipe) selection. Selection policy does not repair an
incorrect model or guarantee fastest measured hardware performance. Changing
tiling can also change floating-point reduction order. Recheck numerical results
and preserve failures; artifact differences or pre-existing warnings do not
authorize silently replacing numerical regression expectations.

Source references: [model interfaces](../../src/voyager_compiler/extensions.py),
[search/result records](../../src/voyager_compiler/ir/plan.py),
[Interstellar adapter](../../src/voyager_compiler/compiler/search/interstellar.py),
[default model assembly](../../src/voyager_compiler/targets/voyager/implementations/model.py),
[recipe planner](../../src/voyager_compiler/compiler/search/planner.py).

## Target-owned matrix search adapters

Reuse has three independent parts: **enumeration engine**, **target adapter/model**,
and **execution builder**. Reusing Interstellar does not require the default
Voyager architecture, loop restrictions or bufferized kernel.

| Decision | Extension owner |
| --- | --- |
| Convert operation geometry into Interstellar `Layer` | Target recipe adapter; `matrix_layer` is an optional shared FX adapter |
| Resources, partitioning and loop hints | `OpModel.space = MatrixSpace(architecture, schedule, mapping_filter=...)` |
| Extent/capacity legality before loop ordering | `OpModel.fit(mapping)`; `TileConstraint` adds exact/multiple extents at `fit_level` |
| Legality depending on the complete loop order/partitioning | Optional `MatrixSpace.mapping_filter(mapping) -> bool` |
| Score a legal complete mapping | `OpModel.cost(mapping, placement)` |
| Realize its control, movement and retention | Paired `ImplementationSpec.build` |
| Compare different ISA/algorithm/retention policies | Recipe candidates and complete comparable estimates |

A custom builder can call the public adapter directly:

```python
from voyager_compiler.compiler.search.interstellar import search_matrix
from voyager_compiler.ir.plan import MatrixSpace

# Illustrative: resource/schedule/filter are owned by this target's recipe.
op.space = MatrixSpace(resource, schedule, mapping_filter=legal_full_mapping)
selected = search_matrix(op, layer, runtime_tolerance=context.tiler.runtime_tolerance)
# Build the selected mapping, placement and reuse; preserve selected in the plan.
# Do not flatten it to the shared builder's grid unless that conversion is valid.
```

The helper returns `TileResult`; it does not register a new scheduler, select a
builder or assign final addresses. The direct caller also supplies semantic
constraints (for example, whole quantization blocks or attention heads); it does
not run the shared FX adapter that discovers those constraints. `fit` is queried before loop orders exist;
use the full-mapping filter for order-dependent rejection. Callbacks must be pure
plain-data operations usable in forked workers. Include hardware, schedule,
filter policy and model/profile revisions in `OpModel.key`; tolerance is included
in the shared matrix cache key. `NoLegalMapping` permits another recipe to compete;
programming errors must propagate. The adapter installs its fit callback on a
local copy of `Layer`, leaving the caller's sizing callback unchanged.

The filter rejects mappings; it cannot supply a different placement for each loop
order. `fit` must produce a placement valid for the accepted orders sharing those
extents. If that is insufficient, use separate execution recipes or extend the
placement contract; do not hide order-dependent placement inside the cost model.

The shared FX-to-grid route (`get_tiling`) still requires the L3 reduction
innermost because its pipelined builder keeps one reduction live at a time.
`search_matrix` does not impose that builder restriction. The reusable adapter
accepts a target-owned hierarchy and explicit internal `MatrixSpace.fit_level`
(default 2). For example, array → on-chip → DRAM uses `fit_level=1`.
Only the shared FX/grid builder requires four levels and fit at L2.
Interstellar enumerates factor-based tiles, not every arbitrary partial-tile size.
Pad the search geometry only when the builder handles the corresponding tails;
keep logical tensor extents and physical footprint explicit.

### Connect search to executable code

Use the existing `TileResult`, not a second schedule record:

```python
# Inside a recipe builder. Each ISA/retention policy can run its own tile search.
selected = search_matrix(op, layer, runtime_tolerance=context.tiler.runtime_tolerance)
node.meta["tile_result"] = selected  # planner retains it in ImplementationChoice.tile
return build_from_mapping(node, selected.tiling, selected.placement)
```

For a native call, retain the mapping even when it becomes one bufferized
invocation. The emitter reads `program.plan.choices[scope].tile`; projection into
ISA tile fields must be deterministic. Share geometry/physical-footprint functions
between fit, cost and emission instead of re-deriving them in each owner. Keep
actual device arithmetic/encoding target-owned. A recipe's outer candidate budget
does not need to enumerate every internal matrix tile.

`memory_level(config, (input_memory, output_memory, weight_memory), ...)` derives
Interstellar capacities and sharing indices from named hardware memories. Explicit
widths convert bytes to element slots; byte-based fit callbacks use width one and
account for actual formats themselves. Mixed-width operands sharing an element
pool require a byte-based callback. This helper does not derive fusion or loop
policy from hardware. Keep energy coefficients and overlap assumptions explicit.

If native execution fixes an order, `Schedule(..., order_priority={level: loops})`
accepts every loop once, innermost first, and compresses away inactive loops.
Unspecified levels still search orders. Use this restriction only when justified
by the paired builder. `search_matrix` validates selected factor coverage and
active-loop orders. Its `access_list` counts logical Interstellar accesses;
physical DMA estimates belong in `Cost`, and observed traffic belongs in target
traces. These are different evidence.

For convolution, investigate both output spatial axes, output/input channels,
filter geometry and weight retention. Small fixed candidate lists are search
budgets or heuristics, not proven legality. A candidate that reuses weights across
spatial tiles needs matching storage lifetime, loop structure and cost; adding a
`retain_weights` flag without realizing reuse is insufficient. Retain a known
feasible comparator before pruning, and report which useful choices were omitted.

## Required diagnosis when utilization or selected performance is poor

Start with the dominant cost-producing decision. If numerical granularity forces
narrow operations or repeated materialization, compare a different valid numerical
assignment together with its executable lowering; retuning tiles within the same
expensive form cannot answer that question. Preserve fixed accuracy/coverage
constraints for such comparisons, and preserve the integer contract when only
execution changes. Keep unresolved opportunities in the
[task work record](../adoption/harness.md), separate from pending validation.

For a performance task, investigate the cause of a material gap before declaring
the target inherently underutilized or the search adequate. If evidence or budget
runs out, report the unresolved question and missing evidence. This requirement
is scoped to performance work, not a prerequisite for every registration edit.

1. **Fix the comparison boundary.** Match shapes, numerical behavior, mapped
   operations, clock and timing scope. State whether initial input packing is
   permitted outside the measured accelerator region. See [input rewrites](rewrites.md#input-boundary-rewrites).
   Report that preprocessing separately; internal activation packing/movement
   remains charged unless the agreed boundary explicitly says otherwise.
2. **Check representation choices.** Use optional
   [input preprocessing](../roles/input-preprocessing.md) when representation limits
   useful utilization; compare its full cost with the original path. Consider im2col/packing,
   layouts, padded weights and native versus decomposed operations. Preserve
   quantization scales, rounding, bias and the independent raw-input reference.
3. **Check reachability before cost.** Can the candidate generator, adapter and
   builder represent a known good candidate? Inspect spatial rows AND columns,
   channel/reduction tiles, loop order, buffering, reuse and retention. For each
   exclusion distinguish hardware legality, builder limitation, policy and search
   budget. Do not merely raise the candidate budget over the same narrow list.
4. **Check ranking.** Force representative alternatives, including the comparator;
   retain mapping, launches, physical traffic, predicted latency and measured
   latency when available. Determine whether the good candidate was absent,
   rejected, scored poorly or lost in selection. A per-family positive affine
   correction cannot reverse ordering within that family; missing geometry or
   reuse terms need a better target cost model.
5. **Check realization.** Compare model assumptions with bufferized loops,
   serialized IR, target conversion and submitted work: loads/stores per loop,
   accumulator lifetime, weight retention, launch count, scratch and dependencies.
   Use traces when available; static structure cannot prove actual overlap.
6. **Retest the explanation.** Change the responsible adapter, builder or model;
   compare held-out shapes/candidates. Report measured selection regret where
   possible, plus unresolved limitations. Lower launch count alone is not proof
   of higher measured performance.

A useful record is one row per compared candidate: semantic region, recipe,
spatial/channel tiles, retention, legality/rejection, launch/traffic estimate,
predicted time, measured time (or explicitly unmeasured), and output check.
This is diagnostic evidence, not a mandatory new compiler IR or status tracker.

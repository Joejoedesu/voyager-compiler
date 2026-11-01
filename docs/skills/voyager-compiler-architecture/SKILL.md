---
name: voyager-compiler-architecture
description: Navigate and implement Voyager's shared bufferized compiler stages, workload and target extension contracts, implementation search, and pinned regression validation.
---

# Voyager compiler architecture

## Locate the selected checkout

Use the user-selected checkout and its current source; preserve unrelated changes.
This editable skill is packaged in `docs/skills/` and installs into
`.agents/skills/` in that checkout. Relative links resolve to the same checkout
from either location. Do not follow a global skill's links to another tree.

Start with the [documentation router](../../../docs/README.md), then the
[role index](../../../docs/roles.md). Load only the applicable
[target registration](../../../docs/adoption/registration.md) or
[workload × target binding](../../../docs/adoption/workload-binding.md) route.
If no target exists, begin with authoritative ISA/API/configuration and the
fresh-target route; a donor model or target guide is not a prerequisite.

For model/search creation, start with the
[creation instructions](../../../docs/compilation/performance-models.md#create-and-register-the-model)
and the [performance role](../../../docs/roles/performance.md).
The shared search enumerates; the target model estimates; the paired builder
realizes execution. Interstellar is not a universal target latency formula.

Shared instructions remain target agnostic. When present, the selected target's
own README supplies its bindings, constraints and scoped evidence. Helpers are
not additional public hooks. Use existing contracts; do not add target-name
switches or a second compilation pipeline. One source file per hook is not required.

## Follow the shared lifecycle

`pipeline.compile_workload()` owns prepare/export → optional input preprocessing → quantize → target legalize
→ common normalize → optional target custom-operation lowering → search/bufferize → target refine → allocate/finalize
invocations → validate/freeze → encode → save bundle → optional target emit.
`encode` captures common IR and target records/payloads from frozen decisions;
`emit` reads the saved bundle without live compiler objects. Test relocation and
fresh-process emission. Neither hook selects a new implementation or allocation.
See the [emission role](../../../docs/roles/emission.md) for signatures and commands.
`runtime.run()` uses host replay unless the caller passes a target with a runtime
hook. Pass the reference and chosen tolerances explicitly for comparison.
`compile_bufferized()` accepts an already prepared/normalized FX graph;
public `transform()`/`compile()` remain compatibility entry points.

| Responsibility | Canonical owner |
| --- | --- |
| Model construction, assets, export/reference | `compiler/prepare.py`, `WorkloadSpec` |
| Optional input preprocessing | `compiler/preprocess.py`, `TargetSpec.input_preprocessing` |
| Shared quantization and normalization | `compiler/quantize.py`, `compiler/normalize.py` |
| Alias/effect analysis | `compiler/analysis.py` |
| Recipe and mapping/tile search | `compiler/search/`, `ImplementationSpec` |
| Buffers, movement, control and residency | `compiler/bufferize/` |
| Final placement, banks and lifetimes | `compiler/allocate.py`, `compiler/banked_allocation.py` |
| Frozen encoding and bundle | `compiler/emit.py`, `artifacts.py` |
| Offline measurement and fitting | `performance/` |

Public protocols live in `extensions.py`, physical device records in `hardware.py`,
and lifecycle records in `ir/{graph,plan,buffer}.py`. Import canonical owners;
removed forwarding paths (`backend/`, `hardware_config.py`, old tiling/bufferize
packages) are not compatibility APIs. Do not duplicate these contracts into a
parallel hierarchy. Substantive existing graph, quantization and reporting helpers
remain usable.

## Preserve representation and decision ownership

- Distinguish the tensor-operation graph, implementation/allocation decisions
  (`CompilationPlan`), and bufferized execution graph (`BufferProgram`). Analysis
  derives relationships; builders materialize buffers/movement; allocation assigns
  addresses/bank cells. Address reuse does not make values semantic aliases.
- Packed `BufferComponent` values can share physical backing and lifetime without
  being ordinary tensor views. `physical_shape` reserves storage; it does not
  enlarge the logical computation. Bank-port conflicts can prevent sharing even
  at disjoint entries. See the shared padding reference for ownership choices.
- `TargetInvoke` allocation and scheduling ownership are independent. Declare
  private memory and implicit traffic. Native boundaries are synchronous; native
  async calls need a new explicit completion-token contract. Async buffer operands
  remain live through waits/drains. Private arenas cannot overlap live allocations.
- Freeze graph, plan and binding identities before emission. Emitters consume
  selected tiling/ISA/placement; they must not retile or invent unmodeled scratch.
  Artifact checksums protect saved files; graph identity does not hash arbitrary
  tensor `.data` mutations.
- Keep hardware facts, numerical policy, execution options, selected decisions and
  empirical profiles separate. Unknown clocks remain unknown; host wall time is
  not accelerator performance.

## Choose the smallest extension

A workload normally starts with one `WorkloadSpec` module: `build` and
`example_inputs`. Assets, calibration, state and custom reference hooks are
optional. Targets register a `TargetSpec` through their package or `voyager.targets`
entry point; `describe_target()` exposes recipes. Workloads have an equivalent
`voyager.workloads` entry point.

An `ImplementationSpec` pairs match, candidates, model, build, optional estimate
and validation. Competing ISA/algorithm forms need separate discoverable candidates
and finite comparable costs. First-match dispatch is not search. Preserve candidate
isolation, rejection reasons and selected parameters. Capacity failures can trigger
bounded replanning, not silent fallback to a donor compiler.

For [fixed-region algorithms](../../../docs/compilation/algorithm-search.md), reuse
`build_sequential_region` when component kernels execute sequentially; its parent
cost already includes `ImplementationChoice.components`. Pair streaming/recomputation
with its actual execution model. `NoLegalMapping` rejects only the infeasible
candidate; do not classify arbitrary implementation errors as search failures.

Reuse shared bufferized templates for layout, tile, placement or primitive changes.
`MatrixSpace`, `GridSpace` and `AttentionSpace` select different search engines but
produce `TileResult` for the same lifecycle. A new algorithmic control structure may
need a builder; new effects, aliasing or completion semantics need a scoped shared
contract change. An independent engine registry or richer result schema is not
created by adding a target option; inspect the current scheduler contract first.

For input-representation utilization problems, see
[input preprocessing](../../../docs/roles/input-preprocessing.md), before quantization.

`TargetSpec.quantization` configures shared PT2E defaults and observer hooks, or
provides `QuantizationPolicy.run(prepared, config, options)` for a target-owned
quantization flow. The override receives the same resolved hardware configuration
as later compilation and returns an executable PreparedWorkload with quantization
provenance; the shared driver refreshes its reference without converting again.
Explicit caller quantizer configuration wins, including None to skip preparation.
Workload bindings own samples/assets/saved profiles; targets own numerical formats,
exact arithmetic and optional recalibration. Repeated passes need replayable
samples. Value-preserving physical packing does not require recalibration. There
is no mandatory calibration-pass DSL or format-converter registry.

Quantization does not select layout, tiling or addresses. `TargetSpec.lower` maps
the normalized graph into custom target operations after shared layout/constant
preparation; preserve its scales and reference semantics. Broad fusion patterns
are explicit target-owned tables bound through normalization and shared by
frontends/tests. Do not derive them from hardware topology or ISA-pipeline
records, or add a configuration-aware resolver without a concrete missing need.
Builder instruction fusion remains a separate mechanism. See the extension
catalog for the ordered signatures.

Shared constant preparation owns immutability and stored-byte accounting. Target
layout policy requests graph transforms; it must not bypass the storage limit by
silently manufacturing another constant. Retained originals require accounting for
both representations or a remaining view/runtime transform. Target packing and
physical padding remain distinct from static preparation.

## Validate the requested claim

Prioritize correctly supported workload coverage and improve useful utilization
on mapped regions as coverage grows; 100% coverage is not a tuning gate. A coverage
percentage needs a stable semantic-work denominator. Introduced copies, padding
and fused calls must not inflate it; eliminated work is not unsupported work.

The engine enumerates/selects, the model evaluates, builders realize execution,
and conversion/runtime preserve it. Review these together when transfer scope,
reuse or overlap changes. Choose target-appropriate traces/checks; no universal
execution checker or fresh characterization campaign is required. Fit models only
after checking structural agreement. Reuse applicable measurements with provenance;
assess held-out ranking as well as error when fitting.

Use affected [test contracts](../../../tests/README.md) and agreed target gates; consult the target guide
when present, or establish the requested checks for a new target.
Preserve pinned comparators, approved replacements and numerical warnings. Do not
edit source during a gate. Separate host replay, emitted artifacts, endpoint
correctness and measured performance. Reuse evidence only when relevant behavior
and configuration still apply; keep historical evidence historical.

Update the canonical doc owner with interface/ownership changes. General guides
and skills stay target agnostic; target guides describe actual bindings and local
constraints; dated reports hold run status. Old TODOs, reports and retired R/W/X
labels are not instructions to reimplement existing code. Inspect source before
acting on a stated gap. Follow the user's commit/push scope.

For requested performance work, investigate material utilization gaps: permitted
initial-input rewrites, candidate/adapter/builder expressiveness, ranking, and
model–conversion–runtime agreement. Follow the checkout's
[model construction and required diagnosis](../../../docs/compilation/performance-models.md).
Default selection is strict modeled latency. Reuse Interstellar through
`search_matrix` with target-owned constraints; do not inherit the default target's
schedule or call a narrow candidate list a hardware limit.

# Endpoint-constrained data-movement search

Voyager now has an opt-in movement implementation search before final ISA
collateral emission. A request fixes the source and destination memory, dtype,
axis layout, shape and whether a new materialized value is required. The search
chooses a legal instruction chain, its tile geometry, traversal and intermediate
materialization. It applies to HBM loads/stores, local operand staging, result
eviction, local assembly and layout conversion; transpose is one edge in this
movement graph, not the search abstraction.

## Facts, choices, and costs

`trainium/movement_search.py` separates:

- `Endpoint` / `Request`: the required values at both ends. Current layouts are
  two-dimensional partition/free axes, with identity or swapped orientation.
- `Mode`: a directed memory route, ISA instruction, engine, permitted dtypes,
  tile limits, and any pinned-SDK storage-encoding requirement.
- `Chain`: the selected sequence of modes, tile shape, row/column traversal,
  and tile-by-tile or stage-by-stage materialization.
- `MovementSelector`: selected request-to-chain bindings and rejected options.
- `search`: bounded beam search over bindings, using the allocated full kernel
  dependency graph and the existing shared graph evaluator.

There are **no nanosecond constants in movement search**. The existing hardware
profile supplies primitive issue/occupancy/completion and DMA parameters. The
93 ns observation remains a measured parameter for one FP32 32×32 stream mode;
it is not a universal movement rate and does not select an implementation.
A test changes the hardware profile and verifies that route selection changes.
Unknown movement timing is reported and excluded from automatic selection,
rather than treated as zero. Unrelated, pre-existing compute timing gaps remain
visible in the full-kernel score.

## Examples of legal chains

| Endpoint requirement | Possible chains |
| --- | --- |
| HBM → SBUF, preserve axes | DMA load |
| HBM → SBUF, swap axes | DMA → Tensor transpose → Scalar/Vector copy; DMA → Vector stream transpose |
| SBUF → SBUF, preserve axes | Scalar copy; Vector copy; two transpose/copy pairs that restore the original axes |
| SBUF → SBUF, swap axes | Tensor transpose → Scalar/Vector copy; Vector stream transpose |
| PSUM → SBUF | Scalar or Vector copy, optionally converting the output dtype |
| SBUF → HBM, swap axes | Tensor transpose → Scalar/Vector copy → DMA; Vector stream transpose → DMA |
| PSUM → HBM | Scalar/Vector copy to SBUF → DMA; no nonexistent PSUM-to-HBM DMA edge |

Enumeration follows memory/layout/dtype states, not a kernel-name preset.
Paths have at most four operations; repeated intermediate states and implicit
HBM spill workspace are excluded. Dtype conversion is restricted to supported
copy edges and the requested output dtype, preventing hidden lossy round trips.
An alias is legal only when materialization is explicitly unnecessary.

Tensor routes consider 128×128, 64×64 and 32×32 instruction tiles. Copy/DMA routes
also consider wider free extents. Vector transpose currently participates only
for the measured contiguous FP32 32×32 SBUF form. Larger tiles can be decomposed;
unsupported tails/dtypes/routes are excluded.

A chain can process every stage for one small tile before moving on, or
materialize each stage before the next. For example, **one 128×128 DMA followed
by sixteen 32×32 stream transposes** is distinct from sixteen small DMAs each
followed by a transpose. Intermediate partition/PSUM geometry is checked as well
as instruction shape. Final allocation checks real SBUF capacity, PSUM banks,
lifetimes, overlapping storage and reuse completion dependencies.

A discovered SDK constraint is recorded separately from hardware capacity:
stream instructions in the pinned 2.22 compiler fail its in-place allocation
checker with the monolithic arena encoding in the tested kernels. Their selected
chain now requires disjoint arenas, obtained through the existing size-class
allocator. The allocated graph is rescored with that storage. The formal emitter
rejects a stream plan with the unsupported monolithic encoding. This is a
conservative supported-realization constraint, not a claim that the physical
hardware lacks such accesses.

## Compiler integration

Movement requests are created in `InstructionPlanner` for load/store layout
paths, operand staging, PSUM eviction and explicit assembly copies. Each search
candidate builds the actual typed instruction plan, allocates it, validates it,
and is scored by `analyze_selected`. The winning instructions, placements and
bindings are persisted in `instructions.json`, `selection.json` and
`movement-search.json`. The formal converter only encodes that selected result;
it does not search again.

The existing shared software tile/loop schedule is held fixed in this experiment.
The new search is the ISA implementation stage within target plan selection;
it is **not yet a joint search of software tiles, cross-kernel fusion, retained
weight panels and movement recipes**. Source/destination views that cannot be
represented by the supported affine two-dimensional movement contract are
rejected. Whole-kernel bank/queue scheduling and some compute completion laws
remain approximate.

The default search budget is zero, preserving existing compilation behavior.
`TrainiumTuning(movement_search_budget=64, movement_search_beam=2)` enables it in
normal target compilation. The CLI below applies it to saved shared schedules,
without changing original artifacts or their references.

## Hardware experiment, 2026-10-07

Trainium2 / NeuronCore-v3, neuronx-cc 2.22.12471.0+b4a00d10, `--target=trn2 --lnc=1`.
Hardware output was checked independently; the official CPU simulator was not
used. Timings are device execution, with 10 warmups and 100 iterations per repeat,
three repeats. Actual device execution is serialized with the shared lock.

Both search inputs use 128×128 matrices. SwiGLU is the complete small workload
with four 128×128 inputs, not the full 4096×3072×1024 benchmark.

| Kernel | Baseline predicted µs | Baseline hardware µs | Selected predicted µs | Selected hardware µs | Hardware latency reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEMM 128×128×128 | 19.119 | 20 | 18.284 | 18 | 10% |
| Small SwiGLU | 57.936 | 75 | 56.088 | 69 | 8% |

Each design space contains **122,304 nominal calibrated recipe combinations**;
64 candidates were evaluated per kernel. This is a budgeted search, not exhaustive
validation or a global-optimum claim. The same winner remained after enforcing
the SDK storage constraint; its final source, instruction plan, hardware record,
model and reference hashes exactly match the measured artifacts.

The winner chose two Tensor-transpose/Scalar-copy pairs for some SBUF assembly
moves, restoring the original layout. This counterintuitive choice adds commands
but changes resource use and allocation/reuse dependencies. The compiled GEMM
binary contains eight transpose commands and two regular backend matmul commands;
the extra transposes were not optimized away. Hardware confirms the performance
improvement, but this experiment does not uniquely attribute it to one pipeline
or allocation effect. Source command count alone is not the selection objective.

Alternative GEMM choices check ranking rather than only the chosen winner:

| Movement alternative | Predicted µs | Hardware µs |
| --- | ---: | ---: |
| Vector assembly copy | 18.981 | 20 |
| Coarse DMA → tiled stream transpose for loads, disjoint storage | 26.097 | 28 |
| Stream transpose for GEMM operand orientation, disjoint storage | 22.112 | 23 |

All seven final kernels (two baselines, two winners, three alternatives) passed
hardware correctness. Initial monolithic-storage stream variants failed compiler
allocation checks; their failure logs are retained. They are not counted as
successful implementations. The selected GEMM is the fastest of the measured
GEMM alternatives. The small-SwiGLU absolute prediction remains optimistic by
about 19%; search improvements do not establish universal model accuracy.

## Reproduce and inspect

From this checkout, with the compiler environment:

```sh
PYTHONPATH=src python scripts/trainium_search_movement.py \
  --input results/trainium/retirement-matrix-fp32/gemm128 \
  --output /tmp/movement-gemm --budget 64 --beam 2
```

The output must be a new directory. `--budget 0` creates the unchanged baseline.
`--bindings FILE.json` replays a recorded candidate for diagnostic comparison.
Hardware validation uses the NKI environment:

```sh
python scripts/trainium_run_hardware.py \
  --artifacts /tmp --case movement-gemm --skip-simulator --repeats 3
```

Results, candidate scores, endpoint bindings, rejected laws, correctness checks,
NEFF/NTFF files, source hashes and failure evidence are retained under
`results/trainium/movement-search-2026-10-07/`. See `report.json` for a compact
comparison and `validation.json` for test/default-regression status.

Final validation: **135 tests and 23 subtests passed**. Seven final hardware
variants passed correctness. With search disabled, the generated GEMM and small
SwiGLU sources are byte-for-byte identical to their saved baselines. The 11 marked
default-target compilation cases still fail before compilation with the same
pre-existing gated ImageNet/Llama access and GLUE URI errors as the fixed
reference; emitted-program equivalence for that suite remains unverified.
Shared/default compiler source files are unchanged.


## Full benchmark follow-up, 2026-10-07

The full-shape experiment is retained in
`results/trainium/movement-search-full-2026-10-07/`. Its input manifest is
the frozen `baseline-manifest.json`, copied from
`/home/ubuntu/ML/trainium-prior-model-experiment/manifest.json`; original source,
model, instruction-plan and hardware-result hashes are recorded before each run.
It uses the same saved shared software schedules and current primitive timing
profile, with independently checked device outputs and three timing repeats.
The four matrix-heavy cases use an eight-candidate budget. LayerNorm exhausts its
four admitted choices; maxpool exposes no variable endpoint requests in this
lowering, so its search reproduces the existing plan. These are bounded movement
experiments, not joint software-schedule or layout searches.

All six selected kernels and four fresh seed controls passed hardware correctness.
The selected kernels have exact modeled/profiled HBM-byte agreement; compiled
replay reports zero unsupported instructions and zero unresolved encoded waits
for every case. This establishes mapping and traffic coverage, not complete
timing calibration. No timing parameters were fitted to these full applications.

Times below are milliseconds. Prior and previous Voyager columns are authenticated
saved hardware measurements; selected hardware is the median of three fresh p50
measurements. The plan estimate is the search score, while compiled replay uses
static instructions and dependencies extracted from the measured binary.

| Kernel | Prior hardware | Previous Voyager hardware | Selected plan estimate | Compiled replay | Selected hardware |
| --- | ---: | ---: | ---: | ---: | ---: |
| LayerNorm | 1.728 | 1.812 | 1.797 | 1.763 | 1.812 |
| Maxpool | 1.358 | 1.016 | 1.321 | 1.074 | 1.016 |
| BMM → softmax | 14.591 | 108.930 | 92.440 | 118.122 | 108.764 |
| Residual → RMSNorm → GEMM | 4.114 | 9.140 | 7.773 | 7.210 | 9.392 |
| GEMM → residual → RMSNorm | 8.466 | 21.197 | 14.863 | 16.692 | 22.149 |
| SwiGLU | 4.082 | 52.925 | 36.748 | 40.214 | 55.458 |

Fresh controls isolate the effect of the selected movement bindings from source
regeneration and measurement drift. Positive changes mean slower execution:

| Kernel | Fresh seed hardware (ms) | Predicted latency change | Measured latency change |
| --- | ---: | ---: | ---: |
| BMM → softmax | 108.969 | −0.25% | −0.19% |
| Residual → RMSNorm → GEMM | 9.147 | −5.78% | +2.68% |
| GEMM → residual → RMSNorm | 21.209 | −6.44% | +4.43% |
| SwiGLU | 53.008 | −5.37% | +4.62% |

Thus this bounded search yields three ranking regressions and one small
improvement among the four matrix-heavy cases. LayerNorm and maxpool retain their
previous measured performance. Compiled replay improves some absolute estimates,
but the two RMSNorm/GEMM kernels and SwiGLU remain underestimated by 23–27%.

Full cases exposed two implementation gaps absent from the small examples.
Movement slicing now preserves affine source/destination strides, rather than
requiring unit steps on each root axis. Address resolution maps selected points
backwards through reshape/index chains, avoiding full HBM coordinate arrays,
including the especially costly flattened-root case. Tests cover strided views,
wrapped-view rejection, and selection through a 268-million-element HBM reshape.
Admitted chains are cached per request within a candidate. Progress records
include per-candidate build/analysis time. None of these changes introduces a
new timing rate.

The focused regression run after those fixes passed **138 tests and 23 subtests**
(`regression.log`). Compiler source hashes are frozen in `experiment-config.json`;
resumed runs verify them before rebuilding candidates. BMM's first seven scores
were recovered from its saved progress log, the eighth was freshly evaluated,
and the previous winner was rebuilt with an exactly matching score before
native compilation. The selected candidate remains number five of eight.

The normalization/matmul comparisons expose a ranking failure. Both winners
switch SBUF staging copies from VectorE to ScalarE. Fresh seed measurements rule
out source regeneration as the main explanation. For residual→RMSNorm→matmul,
1,024 copies read a free-axis stride of 8 and write unit-stride destinations;
their summed profiled instruction durations rise from 632,016 ns to 1,137,763 ns.
The model's copy laws do not distinguish those source/destination strides.
`ranking-diagnosis.json` retains this evidence. Profile durations are diagnostic
outputs only: they are not fed into selection or the compiled-static predictor,
and no application-specific timing constant was fitted.

The GEMM-first RMSNorm case has the same failure with source free-axis stride
16. Its 2,048 staging copies have median observed durations of 591 ns on VectorE
and 1,045 ns on ScalarE; their summed durations increase from 1,210,964 ns to
2,140,941 ns. Summed instruction durations may overlap and are not a critical-path
decomposition. Per-case `measurement-copy-groups.json` files preserve geometry,
engine, counts and measured durations, separately from timing-free
`compiled_static.json` inputs to the predictor.

Full SwiGLU repeats show a 4.62% regression: seed p50 values are
53.027/52.990/53.008 ms, versus 55.458/55.536/55.457 ms for the selected plan.
The 4,608 changed staging copies comprise 3,072 source-stride-8 copies and 1,536
source-stride-24 copies, all 128×512 FP32. Their summed observed durations increase
from 3.128 ms to 5.650 ms. The selected plan still contains 14,784 transposes and
30,624 copies in total; changing the staging engine does not remove those
layout-induced instructions. Selected-plan prediction prefers the slower kernel
(36.748 versus 38.835 ms), while compiled-static replay predicts the correct
ordering (40.214 versus 39.875 ms) but underestimates the absolute times.

BMM changes a different class: 4,096 unit-stride 128×128 staging copies move from
VectorE to ScalarE. Its selected p50 repeats are 108.765/108.719/108.764 ms versus
108.929/108.975/108.969 ms for the seed. Both the selected-plan model and compiled
replay rank this small improvement correctly. Compiled replay predicts
118.122 ms for the selected binary and 118.318 ms for the seed, overestimating
selected hardware by 8.60%; the selected-plan estimate underestimates it by 15.01%.
The replay still marks 2,048 reduction/activation operations as missing calibrated
completion laws. This unit-stride case should not be attributed to the same
strided-copy issue as the three regressions.

The final comparison is reproduced with:

```sh
python scripts/trainium_full_movement_report.py \
  --results results/trainium/movement-search-full-2026-10-07 \
  --manifest results/trainium/movement-search-full-2026-10-07/baseline-manifest.json
```

The report authenticates the frozen manifest, compiler source snapshot, prior
results, newly measured sources/NEFF/NTFF, input identity and compiler flags.
`selected_predicted_us` is the pre-compilation selected-plan estimate used by
search. `compiled_static_prediction.prediction_us` replays the backend's emitted
instruction order and dependencies using the same timing model, without measured
durations or timestamps. Hardware latencies are the median of three repeated
p50 measurements; the separate profiler capture supplies traffic and engine
activity evidence, not the benchmark latency. Native compiler debug artifacts
are retained for the resumed SwiGLU and BMM runs.

Further work needs stride-aware primitive characterization and timing admission,
calibration of the remaining compute completion laws, and validation of ranking
across the alternative chains. The broad software schedule/layout/retention
choices remain outside this fixed-schedule experiment. Full instruction-graph
rebuilding is also expensive: a BMM→softmax candidate takes several minutes.
The current implementation therefore cannot treat a large nominal combination
count as evidence that that space has been searched adequately.

## Recorded plan/replay and schedule-reachability gaps (2026-10-07)

The timing profile is the same, but replay consumes the native compiler's
instruction graph: engine PC order, semaphore completion dependencies, concrete
operands/DMA descriptors and backend instruction realization. It does not consume
measured instruction durations, timestamps, observed waits or application latency.
The signed differences below are **not yet attributed event by event**; they do
not establish that compilation inserted equivalent amounts of extra computation.

| Kernel | Plan (ms) | Replay (ms) | Replay minus plan (ms) | Relative difference |
| --- | ---: | ---: | ---: | ---: |
| LayerNorm | 1.797 | 1.763 | −0.034 | −1.90% |
| Maxpool | 1.321 | 1.074 | −0.247 | −18.70% |
| BMM → softmax | 92.440 | 118.122 | +25.683 | +27.78% |
| Residual → RMSNorm → GEMM | 7.773 | 7.210 | −0.563 | −7.24% |
| GEMM → residual → RMSNorm | 14.863 | 16.692 | +1.830 | +12.31% |
| SwiGLU | 36.748 | 40.214 | +3.466 | +9.43% |

`results/trainium/movement-search-full-2026-10-07/schedule-gap/plan-replay-gap.json`
records the open attribution task. `comparison.json` authenticates compiled
inputs and captures instruction inventories and search coverage. Reproduce with
`scripts/trainium_schedule_gap_report.py --results
results/trainium/movement-search-full-2026-10-07 --prior-experiment
/home/ubuntu/ML/trainium-prior-model-experiment` using the compiler environment.
The saved source hashes identify the model version used for this audit.

Replaying prior binaries with that same current model predicts 1.731, 1.375,
13.456, 4.062, 8.405 and 4.058 ms respectively. Thus compiled replay prefers prior
work wherever prior hardware wins, and prefers Voyager for maxpool. This does
not establish production candidate scoring accuracy: the good compiled schedules
are not necessarily representable in the production search.

The shared bufferized path **is reused**. Its FlashAttention/FA3 builders dispatch
on SDPA(Q,K,V), including inside the per-kernel flow. This benchmark instead
returns the full softmax(A@B) matrix, with no V. Its saved transformed graph has
separate matmul and softmax nodes; no SDPA builder is dispatched. Standalone
producer/reduction fusion could still avoid the intermediate score write, but
requires a suitable shared builder/rewrite. The separate Trainium restriction to
`flow="per_kernel"` also excludes the generic resident flow; merely removing
that guard would not supply the specialized reduction buffering it needs.
No claim is made here that the existing SDPA builder has been validated on
Trainium. Further softmax-path work is deferred at the user's request.

For the other kernels the obstacles are concrete:

- **Search scope:** the movement experiment preserved every `model.txt` byte.
  It did not rerun shared fusion, Interstellar tiling or bufferization. Every
  tested matrix-case alternative changed only one request-class binding;
  budget eight stopped before combined choices, and no stream-transpose
  candidate was evaluated despite such chains being admitted.
- **Fusion/retention:** Trainium's configured fusion patterns cover matrix/add
  with ReLU and maximum chains, not RMSNorm/GEMM or SwiGLU. Shared allocations
  therefore materialize residual and normalized tensors for the GEMM-last case,
  matrix and residual outputs for GEMM-first, and five separate 4096×3072
  intermediate objects for SwiGLU. These are logical allocations, not summed
  peak memory. Prior RMSNorm kernels fuse within row tiles; prior SwiGLU retains
  useful local intermediates but also explicitly spills one intermediate to HBM.
- **Layout/realization:** GEMM uses weight-stationary [N,M] result panels, stages
  both operands for larger software tiles and transposes non-transposed weights
  inside the panel loop. Movement bindings preserve those required endpoints;
  they cannot eliminate the conversion by choosing a different producer layout,
  operand orientation or reuse lifetime.
- **Model ranking:** the stride-dependent copy regressions above remain real.
  Wider enumeration alone is not a remedy for incorrectly ranked primitives.

For SwiGLU, both compiled programs perform 4,608 logical matmuls. Prior work has
256 Tensor transposes and 320 raw COPY opcodes, versus 14,784 and 30,624 in
Voyager. Prior/current HBM traffic is 349,241,344/666,959,872 bytes. For the two
RMSNorm/GEMM cases, prior Tensor transpose counts are zero, with 16,384/32,768
STREAM_TRANSPOSE operations instead; current Tensor transpose counts are
3,072/5,632. Raw COPY counts exclude STREAM_TRANSPOSE and movement fused into
ACTIVATE, so they are not counts of every semantic data transfer.

LayerNorm is already close (1.812 versus prior 1.728 ms); prior uses a different
variance formula. Maxpool is already faster (1.016 versus 1.358 ms), so there is
no missing superior prior schedule for that case. Further work should expose
legal shared fusion/layout/retention alternatives, align their model and
realization, then evaluate candidate ranking and hardware correctness. Prior
kernels are diagnostic comparators, not hardcoded production schedules.

## Operand reuse and shared pointwise fusion follow-up

The follow-up experiment in `results/trainium/operand-reuse-2026-10-07` excludes
BMM→softmax. It adds reconstructible target options to `TrainiumTuning`:

- `matmul_operands="staged"` preserves the existing default.
- `matmul_operands="direct"` consumes the selected local operand views without
  the unconditional per-panel SBUF staging copies.
- `matmul_operands="reuse"` also retains a converted weight panel for subsequent
  M panels **within the same shared software GEMM tile**. The cache ends at that
  GEMM invocation, so later writes/reloads cannot reuse stale weights.
- `pointwise_fusion=True` enables the existing shared fusion machinery for
  sigmoid→multiply→multiply chains. Shared fusion legality and group selection,
  tiling, bufferization, lifetime analysis and ISA realization remain in use.
  This removes the two internal HBM objects in a SiLU/gating chain; it does not
  fuse whole GEMMs or reductions. The default remains false; this is an
  explicit compilation policy.

The direct/reuse alternatives are explicit candidate policies, not benchmark
shape presets. Full workloads rerun ordinary shared search without fixed tiles.
The current experiment compares policy configurations; the compiler does not yet
jointly enumerate these policy switches inside each mapping search. The formal
converter still consumes the chosen typed plan deterministically.

`compute_graph` represents the omitted staging and reused-weight completion
edges. Matrix storage additionally reserves a converted weight tile, while final
physical allocation validates exact live ranges and capacity. The instruction
count audit now derives compute-transpose counts from that graph, avoiding a
second stale count formula. Capacity-rejected candidates return infinite memory
cost rather than accessing a missing traffic estimate. No primitive timing rate
was changed or fitted.

The first attempt to change policy on an already-selected saved schedule was
correctly rejected by the dependency-template audit; the experiment then reran
shared compilation. Early full-search attempts exposed the capacity-cost and
transpose-count bugs above. Their failed logs are retained and are not hardware
results.

Three small unfused workloads and small fused SwiGLU pass device correctness.
Held-out reuse probes also pass: FP32 1024×512×512 with a diagnostic
1024×256×256 software tile, split K and one buffer slot; BF16 768×384×256 with a
768×384×256 tile, including a partial final M instruction panel. These fixed
probes test legality and reuse, and are not the full benchmark search results.
The fused small SwiGLU measures 57 µs versus 75 µs with operand reuse alone.

For full comparisons, the runner uses the exact authenticated seed reference
inputs via `--reference-root`. A runner bookkeeping fix distinguishes that
external validation reference from the local reference authenticated by the
reusable binary manifest. Latencies remain the median of three device p50s under
`--target=trn2 --lnc=1`, with the official CPU simulator skipped. The initial
RMSNorm→GEMM run on newly generated input values is retained separately;
matched-input reruns supply the comparison.

Reproduction (compiler environment for generation/reporting, NKI environment for
hardware):

```sh
python scripts/trainium_advanced.py --output NEW_OUTPUT \
  --cases add_rmsnorm_matmul matmul_add_rmsnorm swiglu --matmul-operands reuse
python scripts/trainium_advanced.py --output NEW_FUSED_OUTPUT \
  --cases swiglu --matmul-operands reuse --pointwise-fusion
python scripts/trainium_run_hardware.py --artifacts NEW_OUTPUT \
  --reference-root results/trainium/movement-search-full-2026-10-07/seed \
  --skip-simulator --repeats 3
python scripts/trainium_operand_report.py --roots NEW_OUTPUT NEW_FUSED_OUTPUT \
  --baseline results/trainium/movement-search-full-2026-10-07 \
  --prior-experiment /home/ubuntu/ML/trainium-prior-model-experiment \
  --output NEW_REPORT.json
```

Run the hardware command for `NEW_FUSED_OUTPUT` too. The report authenticates
source, binaries, traces, references and compiler configuration, checks modeled
versus profile HBM bytes, and keeps compiled replay separate from measurement.
LayerNorm and maxpool retain their previous implementations and measurements;
these matrix-operand changes do not address LayerNorm's remaining small gap,
and Voyager maxpool already beats the prior comparator.

The final focused suite passes **145 tests and 23 subtests** (28 warnings),
including shared hardware/configuration/execution contracts. Twenty-four default
matrix dependency graphs and instruction-count comparisons are identical to the
pre-change implementation across FP32/BF16, transposed weights, short K and
multi-panel shapes (`default-graph-equivalence.json`). Regenerated LayerNorm and
maxpool NKI sources are byte-identical to the prior measured versions
(`nonmatrix-default-check/comparison.json`). ImageNet/Llama regressions remain
waived as requested; no new device measurements are claimed for those two
unchanged kernels.

Compilation-time stack samples show that full SwiGLU spends substantial time in
shared candidate evaluation and subsequent selected-ISA expansion. The reuse-only
full generated source is 7.2 MB. This is a remaining compiler scalability issue,
separate from device latency; no candidate timings were replaced by measured
kernel times to make search finish faster.

The retained-weight capacity charge is currently a **conservative whole-tile
policy reserve**, including cases whose converted views have shorter actual
lifetimes. It is not a physical SRAM limit. Final allocation checks exact
instruction lifetimes. Tightening that reserve and selecting retention per GEMM,
rather than one policy for the entire compilation, remain useful search-space
extensions. In the reuse-only SwiGLU run, search changes the down-projection tile
from M512/N1024/K3072 to M1024/N128/K3072 and increases HBM traffic; its hardware
comparison therefore includes both policy and software-tile changes.

The new hardware runs explicitly expose NeuronCore 0. The authenticated earlier
seed records used the runtime's default core visibility; both use the same
Trainium2 host and single-core compiler flags. Runtime environment fields remain
in the report so that this difference is visible rather than silently equated.

### Measured full-size results

All times below are milliseconds. “Previous” is the authenticated earlier seed,
not the slower movement-search winner. New runs use those exact reference
inputs. The first two rows use operand reuse; SwiGLU also enables shared
pointwise fusion. All three pass device correctness at the existing tolerances.

| Kernel | Prior work | Previous Voyager | New Voyager | Latency reduction | Plan estimate | Compiled replay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Residual → RMSNorm → GEMM | 4.114 | 9.147 | **8.318** | 9.06% | 5.999 | 5.372 |
| GEMM → residual → RMSNorm | 8.466 | 21.209 | **17.647** | 16.79% | 11.016 | 12.608 |
| SwiGLU, reused operands + fused pointwise | 4.082 | 53.008 | **34.765** | 34.42% | 20.538 | 22.931 |

Matched-input p50 repeats are 8.326/8.316/8.318 ms,
17.647/17.647/17.622 ms, and 34.733/34.765/34.769 ms respectively. Maximum absolute
error for full fused SwiGLU is 1.1921e−6. No tolerance was relaxed.

| Kernel | Previous Tensor transposes | New Tensor transposes | Previous raw COPY | New raw COPY | New HBM bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Residual → RMSNorm → GEMM | 3,072 | 2,560 | 6,562 | 4,002 | 167,841,792 |
| GEMM → residual → RMSNorm | 5,632 | 4,608 | 12,452 | 7,332 | 302,063,616 |
| Fused SwiGLU | 14,784 | 8,448 | 30,624 | 13,280 | 453,050,368 |

Selected-plan and compiled-model HBM byte counts match the profile in every
completed full case. RMSNorm/GEMM traffic is unchanged. Fused SwiGLU traffic is
32.07% lower than the previous 666,959,872 bytes; relative to the new reuse-only
schedule, fusion removes 251,658,240 bytes (two intermediate writes, their reads,
and the duplicate gate-input read). Raw COPY again excludes transfers fused into
other opcodes. `report.json`, per-case `compiled_static.json`, prediction files
and extraction provenance retain the detailed evidence.

These are measured improvements, **not closure of the prior-work or timing-model
gaps**. The selected-plan model still underpredicts these cases by roughly
28–41%; compiled replay underpredicts by roughly 29–35%. Prior kernels remain
about 2× faster for the RMSNorm/GEMM cases and 8.5× faster for SwiGLU. Whole-tile
GEMM/reduction fusion, compatible producer/consumer layouts and operand
orientation, more precise retention footprints, and joint policy/mapping
selection are still missing. Increasing the movement-binding budget alone
cannot express those transformations.


The completed full SwiGLU ablation with operand reuse but no pointwise fusion
passes correctness at **47.719 ms**, with repeats 47.766/47.714/47.719 ms. That is
a 9.98% reduction from the previous 53.008 ms. Enabling the shared pointwise
fusion lowers this further to 34.765 ms, another 27.15% reduction relative to
the reuse-only variant. Both variants retain 4,608 logical matmuls. This isolates
the additional benefit of shared pointwise fusion while preserving the same
matrix policy and searched GEMM tiles. All four full-case result rows remain
in the final report, including the slower SwiGLU ablation.

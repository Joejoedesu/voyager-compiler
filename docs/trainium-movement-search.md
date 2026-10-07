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
`/home/ubuntu/ML/trainium-prior-model-experiment/manifest.json`; original source,
model, instruction-plan and hardware-result hashes are recorded before each run.
It uses the same saved shared software schedules and current primitive timing
profile, with independently checked device outputs and three timing repeats.
The four matrix-heavy cases use an eight-candidate budget. LayerNorm exhausts its
four admitted choices; maxpool exposes no variable endpoint requests in this
lowering, so its search reproduces the existing plan. These are bounded movement
experiments, not joint software-schedule or layout searches.

Full cases exposed two implementation gaps absent from the small examples.
Movement slicing now preserves affine source/destination strides, rather than
requiring unit steps on each root axis. Address resolution maps selected points
backwards through reshape/index chains, avoiding full HBM coordinate arrays,
including the especially costly flattened-root case. Tests cover strided views,
wrapped-view rejection, and selection through a 268-million-element HBM reshape.
Admitted chains are cached per request within a candidate. Progress records
include per-candidate build/analysis time. None of these changes introduces a
new timing rate.

The normalization/matmul comparisons expose a ranking failure. Both winners
switch SBUF staging copies from VectorE to ScalarE. Fresh seed measurements rule
out source regeneration as the main explanation. For residual→RMSNorm→matmul,
1,024 copies read a free-axis stride of 8 and write unit-stride destinations;
their summed profiled instruction durations rise from 632,016 ns to 1,137,763 ns.
The model's copy laws do not distinguish those source/destination strides.
`ranking-diagnosis.json` retains this evidence. Profile durations are diagnostic
outputs only: they are not fed into selection or the compiled-static predictor,
and no application-specific timing constant was fitted.

Further work needs stride-aware primitive characterization and timing admission,
calibration of the remaining compute completion laws, and validation of ranking
across the alternative chains. The broad software schedule/layout/retention
choices remain outside this fixed-schedule experiment. Full instruction-graph
rebuilding is also expensive: a BMM→softmax candidate takes several minutes.
The current implementation therefore cannot treat a large nominal combination
count as evidence that that space has been searched adequately.

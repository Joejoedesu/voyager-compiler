# Trainium ISA movement and dependency model

The current [ISA mapping and measured timing inventory](trainium-isa-model.md)
includes the integrated stream-transpose and short-K FP32 laws.

## Dependency-aware physical scoring (2026-10-10)

The opt-in `scheduled-ready` model scores the fully expanded selected ISA with
slice hazards and a predicted compiler issue order. Enable it through the target
policy:

```python
from voyager_compiler.trainium.execution import TrainiumTuning

# Pass this tuning object to TrainiumMappingPolicy for compilation/search.
tuning = TrainiumTuning(physical_model="scheduled-ready", reorder_window=16)
```

`program_analysis.analyze_selected(..., execution_model="scheduled-ready",
reorder_window=16)` also supports analysis of an existing selected program.
Expanded region search uses this same analysis to score candidates and retains
its model identity in the selected plan. The default physical model is unchanged.

The analysis has two additional steps:

1. `trainium/region_dependencies.py` reconstructs read-after-write,
   write-after-read and write-after-write hazards using root-relative tensor
   coordinates. Strided disjoint slices no longer depend on the unrelated last
   writer of their entire buffer. Only fully covered accesses can be retired;
   partial writes retain hazards on untouched elements. Unknown views and views
   above the coordinate limit conservatively overlap the whole root. Affine
   Cartesian views use a fast path; general small views use explicit coordinates.
   Physical allocation reuse edges and dependencies beyond the Builder's logical
   whole-root edges remain required.
2. `trainium/issue_schedule.py` chooses a dependency-ready static order with a
   bounded source lookahead per engine. Existing operand handoff laws participate
   in this choice. The final order is scored with the existing per-engine issue,
   cumulative completion and readiness laws. The lookahead is compiler modeling
   policy, not a measured hardware queue capacity. It is recorded in the model
   identity together with hashes of both mechanisms.

These steps operate on an analysis copy. They do not alter emitted source,
physical addresses, or the selected buffer layout, and do not invoke the native
compiler or read NEFF/profile data during search. They predict the reorder that
native compilation may perform; they do not enforce an identical native order.
Primitive timing coefficients remain unchanged. Geometry/context classification
currently precedes predicted reordering, so history-sensitive primitive laws are
still an approximation. The version-1 dependency record does not distinguish an
explicit scheduling edge identical to a Builder root edge; such custom ordering
constraints need separate provenance before this refinement can support them.

A whole-kernel `ncc.no_reorder()` guard is a separate hardware diagnostic. It
changes native scheduling constraints and must be evaluated as a distinct binary;
its performance is not the expected result of enabling `scheduled-ready`.
The model remains experimental: guarded native replay still has unexplained
latency, and validation on retained selected streams is not a full search-ranking
or ISA timing-coverage validation.

## Historical baseline

The text below records the 10-05 baseline. See
[the 10-06 selected-plan implementation](trainium-selected-plan.md) for the
formal converter, physical allocation, row reductions, pool and new evidence.

Implemented in `voyager-trainium-10-05-isa`. The parallel
`voyager-trainium-10-05` remains the language-interface baseline. Selection is
minimum predicted latency; energy is not part of the objective.

## Integration and ownership

The compiler still uses shared Interstellar enumeration, target `evaluate()`,
`CandidateEvaluation`, shared bufferization, and target `realize()`. The reusable
`codegen/transform/tiling/execution.py` evaluator schedules an immutable repeated
graph of operation events. It is not a new search algorithm or a mandatory pass
for Voyager/Gemmini. Trainium caches equivalent graph evaluations during search.

Hardware IR `OperationImplementation` definitions specify typed operands,
intermediate values, primitive operations, supported memory routes, instruction
expansion counts, and compiler applicability. `trainium/operations.py` owns the
Trainium catalog; `isa.py` resolves its expansion. An implementation containing
several dependent instructions is distinct from an ISA-supported fused call.

`trainium/dependencies.py` binds each candidate to operation implementations and
a repeated graph. Shared traversal supplies load reuse and repetition periods.
Bufferization still owns aliases, waits, lifetimes, and correctness. The target
converter emits the selected panels and explicit NKI ISA operations.

Dependencies are therefore hybrid: bufferization supplies storage/lifetime and
region relationships; the target expands operations and their timing; the shared
evaluator scores the resulting chains during search. Whole-program boundary
materializations are analyzed after lowering in `program_analysis.py`.

## Required converter boundary

The Trainium converter must perform formal emission of a fully selected plan.
This is the required architecture; the current implementation does not yet meet
it for all operations. In particular, naming an early arithmetic recipe is not
sufficient when layouts, transfers or temporaries are still chosen during emission.

| Decision | Owner before conversion |
| --- | --- |
| Legal operand/result layouts, ISA implementations, expansion rules and timing | Hardware IR and target lowering definitions |
| Chosen implementation, concrete tile layouts and required conversions | Target lowering instantiated within shared search/bufferization |
| Transfer geometry and recurrence, residency, buffering, order and dependencies | Shared search/bufferization with target policy hooks |
| SBUF offsets, PSUM banks, aliases and safe reuse of all explicit/implicit temporaries | Shared memory planner with target placement constraints |
| NKI syntax, symbols, addresses and encoding of the already selected instructions | Converter |

The converter must not select a different layout, split/fuse a compute region,
choose an engine, insert a materialization, change buffering or residency, or
reorder operations, even when a reordering preserves mathematical dependencies.
Legal instruction subdivision and any fixed encoding expansion must be specified
and costed in the selected implementation before conversion. Missing layout,
placement or transfer decisions must be rejected, not inferred heuristically.
A rejection must identify the operation and missing plan field.

The same selected expansion must drive candidate cost and emission. Repeated
instruction templates remain compact during search; startup/steady-state/tail
scoring does not require expanding every iteration. Template instantiation and
address expression substitution are mechanical conversion, not new scheduling.

Current gaps found in the 2026-10-06 audit:

- `Converter.storage_shape` and `index` impose generic feature-partition storage.
- `Converter.copy` chooses transfer subdivision and layout adapters from tuning.
- `Converter.transpose` resolves copy-engine policy during emission.
- `realize_reduction` introduces row-layout adapters and temporary tensors late.
- Physical addresses from shared placement are not bound to NKI direct allocation;
  default `nl.sbuf`/`nl.psum` allocation remains compiler-managed.
- Nonmatrix timing reconstructs transfer repetition from compute records, losing
  retained-parameter reuse in LayerNorm.

These are outstanding implementation changes, not guarantees supplied by the
existing realization audit. The NKI backend also performs final compilation and
hardware scheduling after conversion; explicit placement/dependencies and NEFF
checks are needed to establish that the resulting executable honors the modeled
contract. A formal Voyager converter alone cannot certify backend timing.

## Four movement costs, not one transfer duration

For each emitted DMA panel, `movement.py` represents:

1. **Issue/admission service:** time occupying the shared DMA request path.
2. **Dispatch latency:** time before payload execution can begin. Independent
   requests can issue during this delay.
3. **Bandwidth occupancy:** bytes divided by the target bandwidth, adjusted for
   which partition engines carry the rectangle. Payloads can overlap; an
   observed payload's elapsed duration is not exclusive engine service.
4. **Completion:** characterized payload duration plus endpoint notification,
   after which dependent transpose, copy, or compute may consume the data.

Store panels depend on explicit output layout work. Loads feed explicit layout
work. The one shared transpose identity constant contributes 16 KiB of HBM reads;
its endpoint uses a conservative 64-KiB FP32 SBUF service budget (BF16 may write 32 KiB).

DMA command admission is currently one shared `DMAIssue` resource. Finite
command queues and backend reordering are not inferred from its issue interval.
The evaluator uses dependency-ready scheduling, so a later-ready command does
not block all future commands as it might in a physical submission stream.
This remaining approximation is exposed in the report.

## Throughput versus dependent completion

`trainium/timing.py` and `timing_trainium2.json` hold a versioned Trainium2 timing
profile for neuronx-cc 2.22.12471. Profile fields participate in the hardware
fingerprint. The measurements are isolated primitives, not application latency
fits or selected-tile presets. `scripts/trainium_characterize.py` saves source,
correctness checks, NEFF, NTFF, decoded profiles and hashes;
`scripts/trainium_calibrate.py` reproduces the timing profile from that evidence.

Documented back-to-back TensorE service remains separate from result readiness.
For example, FP32 128x128 matmul service is about 213 ns, while measured dependent
completion is about 808 ns. A PSUM-to-SBUF Scalar copy then needs approximately
274 ns before its consumer is ready. Multiplying documented service by the
number of expanded LDWEIGHTS/MATMUL instructions would double-count pipelining.
Instead, expansion supplies counts and the completion law covers the group's
result latency. The generic hardware `OperationTiming` override still takes
precedence over a characterized law.

Ready-operand streams exhibit different initial and later spacing. Their traces
are retained as evidence, but mixed compiler scheduling effects are not treated
as uniquely identified intrinsic instruction issue intervals. Vector/transpose
completion uses observed operation spans; unisolated notification effects,
partial-partition extrapolation, and uncharacterized engine/dtype alternatives
remain limitations. Missing completion laws are explicitly reported.

A separately measured fixed device term is approximately 9.40 us. It accounts for
front/end control outside modeled dataflow; it is not host invocation time.
Application predictions were frozen before final benchmarking. The graph models
finite fill, dependent chains, and drain; its final iteration spacing alone is
not a certificate that steady state has been reached.

## Buffering and physical allocation

The default `buffer_allocation="compiler"` contract accurately describes emitted
SSA tensors. The Neuron compiler chooses their physical storage and reuse.
Voyager's logical depth remains a declaration/capacity choice, but receives no
latency credit from fictitious enforced slot reuse. `--force-buffer-depth 1|2`
provides a diagnostic pair with otherwise identical tile constraints.

`legacy_logical` remains an explicitly named replay approximation for older
artifacts. Direct NKI allocation cannot be partially mixed with automatic
allocation; a true physical multi-buffer backend would require a complete
allocation implementation. Current SBUF/PSUM checks are conservative capacity
checks, not bank-conflict or physical-slot certificates.

## Ragged boundaries and audits

The common two-dimensional trailing pad and unit-step slice paths emit explicit
rectangular DMA and VectorE fill operations. A partially valid tile is filled,
its valid rectangle is loaded, and the full output rectangle is stored. Padding
and slicing still materialize intermediate HBM arrays; their reads and writes
are now included in whole-program traffic. This change does not remove those
intermediates or claim residency across separately lowered regions.

`program_analysis.py` parses actual emitted source and independently checks
matmul, transpose, and DMA call counts against realization records. It sums
scheduled traffic, boundary materialization and shared constants, and composes
boundary graphs around matrix bodies. Unsupported boundary expansion is marked
incomplete. General nonmatrix-only kernels, such as add+ReLU, retain correctness
and traffic validation but have no claimed whole-program latency prediction.

The source audit checks emitted calls, not the final backend schedule. NEFF/NTFF
validation separately checks expansion, HBM counters, control/command timing,
and active intervals. Backend-folded assembly copies, scalar constants, spills,
physical banks, throttling and full semaphore timing cannot be certified from
source templates. Service/peak fractions and profiler-active occupancy are
reported separately because they are different utilization metrics.

## Reproduction and evidence

```sh
source ../activate-compiler.sh voyager-trainium-10-05-isa
NEURON_RT_VISIBLE_CORES=0 /home/ubuntu/ML/.venv-nki/bin/python \
  scripts/trainium_characterize.py --output results/trainium/movement-probes
python scripts/trainium_calibrate.py results/trainium/movement-probes \
  --output /tmp/reproduced-timing.json
cmp /tmp/reproduced-timing.json src/voyager_compiler/trainium/timing_trainium2.json
python scripts/trainium_generate.py --output results/trainium/example
NEURON_RT_VISIBLE_CORES=0 /home/ubuntu/ML/.venv-nki/bin/python \
  scripts/trainium_run_hardware.py --artifacts results/trainium/example --repeats 3
```

Final application evidence is under `results/trainium/movement-final-corrected/`.
Each variant contains generated source, plan, reference, simulation/hardware
checks, three benchmark repetitions, and matching NEFF/NTFF. `validation.json`,
`validation.csv`, and `report.md` provide detailed comparisons. The earlier
`movement-final/` is a superseded experiment which incorrectly serialized
isolated payload duration; it is retained for diagnosis and not final evidence.
Older `dependency-*` evidence remains unchanged. ImageNet and Llama regression
runs remain excluded as requested.

Final evidence: 13 kernels passed simulation/device correctness and identical
reconversion; 12 matrix predictions have 8.89% mean absolute error and 16.47%
worst error. HBM totals match exactly for 11/13 cases; ragged differs by 128
backend constant bytes, and one held-out case has 196,608 bytes of backend load
deduplication. 86 tests and 10 subtests passed. The additional unrestricted large
search was stopped after more than 15 minutes; its whole-tile comparator was
fully measured. See the report and `coverage.json` for details.

## Bounded search scoring

Long candidate graphs use a neighboring-iteration window, retaining the actual
one-time initial loads, periodic transfers, split-K phases, buffer reuse edges
and final stores. Search estimates `T(N) = T(W) + (N-W) * II`: the sampled finite
window includes startup and drain, and `II` is the observed steady interval.
This does not estimate host launch or runtime initialization overhead.

The sampling period covers operation cadences and reuse distances. Three period
increments must agree with their mean within 1%; doubling the window must agree
with the extrapolation within 0.2%. Small scheduling jitter is therefore allowed;
this is an approximate search score, not a proof of eventual periodicity. The
longer check provides the final slope. The window grows when startup persists,
and a slope below sustained resource demand is rejected as transient. Resource service and traffic counts retain
the full problem size. Short graphs, interior one-time phases and failure to
converge fall back to finite replay. Selected matrix plans are always evaluated
with finite replay; whole-program realization analysis also retains finite replay.
The candidate set and performance-only objective are unchanged.

Reproduce graph-level accuracy and scoring time separately from device error:

```sh
PYTHONPATH=src python scripts/trainium_validate_steady.py \
  results/trainium/advanced-fixed \
  --output results/trainium/steady-validation.json
PYTHONPATH=src python -m pytest -q test/test_trainium_steady.py
```

Scaled validation graphs preserve once-only prefix/drain work and increase the
number of periodic iterations. They are synthetic model-scaling checks based on
actual selected plans, not additional hardware benchmark results. Finite replay
and bounded scoring share the same ISA service/dependency assumptions; agreement
between them does not validate those assumptions against hardware.

An infrequent reload does not require expanding its entire reuse interval.
For aligned first/last phases with exactly scalable dependency distances, the
scorer also shortens the steady interiors between reloads. For example, sixteen
4096-iteration runs are sampled as sixteen runs of 4, 8, 16 and 32 iterations.
All sixteen reload boundaries, the original prefix/drain, and local reuse edges
remain explicit. Their measured slopes must agree within 1%, and the final
window check within 0.2%. Otherwise the ordinary periodic-window path applies.
Full-problem resource counts are unchanged; this remains an approximation whose
selected matrix result is checked by finite replay.


## Native scan, register move, and accumulator readback

The compiled-ISA analyzer also consumes `native_timing.py` and the packaged
`native_characterization.json`. This extends analysis of existing NEFF instruction
streams; it does not add NKI lowering or a new search candidate family.

- `TENSOR_TENSOR_SCAN` uses the documented `max(64, 2N)` VectorE cycles,
  a measured pipeline drain, and an additional operand-read cost when its initial
  value comes from SBUF instead of an immediate zero. Completion characterization
  is bounded to contiguous FP32 multiply/add, 128 partitions, and free widths
  64–2048. Other shapes retain analytical service and unknown completion.
- `ACTIVATION_READ_ACCUMULATOR` has separate issue and output-completion costs.
  A reduction-producing activation supplies an internal forwarding milestone;
  readback waits for it, and a subsequent activation waits until readback releases
  the hidden state. These RAW/WAR dependencies survive disabling engine ordering.
  Continuation requires a recognized producer; an ordinary activation invalidates
  the tracked reduction state. The calibrated producer is FP32-source EXP+sum,
  with explicit shape bounds; unsupported producers keep their existing partial
  timing. No extra full reduction is charged on top of readback.
- GpSimd `MOVE` currently covers a scalar `uint32` immediate-to-register move.
  Its measured cost is deliberately separate from tensor-copy throughput.
  Other move forms are rejected rather than assigned the same cost.

Calibration contains per-probe source/binary/trace hashes and documented versus
measured quantities. Twelve isolated hardware probes check scan width, seed
kind, and reduction/readback; full-kernel runtime is not a fitting target. The
32-partition probes are coverage checks, not permission to extrapolate the
128-partition completion laws. Runtime durations remain forbidden prediction
inputs. Unknown completion laws remain visible in analysis results.

Native `CAST` and long dtype spellings are normalized inside the production
adapter. A load-elided matmul consumes the preceding TensorE stationary operand
without duplicating its load or semaphore notifications. Existing paired-matmul
service laws remain unchanged; this does not claim a new independently timed
LDWEIGHTS pipeline model.

A packed static DMA descriptor can repeat a partition: four 1 KiB fragments on
three partitions are not four thirds of a KiB per partition. For descriptors
that fail the old equal-partition byte accounting, `native_dma.py` validates the
supported contiguous-byte block form, preserves the exact bytes, and uses the
busiest eight-partition DMA-engine group. The descriptor still contributes one
request. Other descriptor forms fail explicitly. Existing uniform-panel rules
are unchanged.

Unit coverage: `test/test_trainium_native_ops.py`. Characterization data is
packaged with the compiler; external experiment scripts and result reports are
not required to use these laws. Remaining affine-select, predicated-copy,
mask/shuffle, and wide native transpose forms are not covered by this extension.

# Trainium backend

This package now owns the compiler-2 Trainium implementation. Registration is
in `target.py`; import this package as `voyager_compiler.targets.trainium`.
The former `backend/trainium` compatibility package has been removed.
See the [compilation guide](../../../../docs/compilation.md) for the shared
pipeline, extension policy and current acceptance scope. Its coverage-first
objective and target-specific model–converter debugging guidance apply to NKI
lowering. Choose checks for Trainium's actual execution mechanisms; the framework
does not require a universal execution checker. Trainium-specific conventions
below preserve the shared semantic and selected-plan contracts.

`compile_workload(workload, registry.target("trainium"), output)` applies the
registered normalization policy, compares implementation recipes, freezes the
buffer program and automatically emits NKI under `output/nki/`. FP8 matrix
operations expose `trainium.gemm.single_row` and `trainium.gemm.double_row`;
`implementations.py` pairs each choice with its model and carries the selected
mode through `TargetInvoke` into NKI. Both paths have host replay and emitted-mode
checks in `tests/test_trainium_alternatives.py`. They have not been rerun on a
Neuron simulator or device in this integration. Performance fitting is available
through `performance/`; no measured profile is installed by default. This describes
the integration's evidence, not an incomplete framework mechanism. Registration
and ordinary compilation can use analytical models. For performance evaluation or
fitting, reuse applicable donor measurements with their provenance or collect new
timings when changed execution assumptions or the requested claim require them.

The detailed backend notes below retain the inherited instruction and runtime
contracts. Their historical verification results belong to the compiler-2 donor,
not the current integration.

Trainium uses the same search, bufferization and final-plan lifecycle as other
targets. The NKI translator implements the selected recipe and its instruction
expansion; it does not select new software tiles or silently change execution
policy. NKI's downstream scheduling must still realize the dependencies and
overlap assumptions being priced by the model.

## Extension points used

| Extension | Binding and responsibility |
| --- | --- |
| `hardware` | `config.py:TrainiumConfig`, TRN2/TRN3: engines, partitions and capacities |
| `operators` | `scan.py:selective_scan`: unbatched FP32 state-space recurrence with explicit CPU and fake semantics |
| `normalization` | `config.py:LAYOUT` and `fusion.py:FUSION_PATTERNS`: operand layouts and explicit target-owned broad fusion patterns |
| `legalize` / `lower` | `normalization.py`, `dag.py`: guarded RMS recognition, inverse-transpose cancellation, numerical-policy tagging and equal-shape FP32 DAG outlining |
| `implementations` / `model` | `implementations.py`, `compensated.py`, `model.py`, GEMM/attention/vector models and builders: competing arithmetic forms plus shared recipes |
| `encode` | `implementations.py:encode`: selected row mode in common IR; immediate constants in binary payloads (including non-finite values) and translation limits in saved target records |
| `emit` | `target.py:emit`, `translate.py`: read saved IR and translation records to generate NKI, without live FX or configuration objects |

No separate `input_preprocessing`, `quantization`, `refine`, `runtime` or `characterize` hook is
registered. Shared defaults apply; inherited runtime tools are invoked separately.
Broad graph patterns live in `fusion.py`; registered compilation and both benchmark
frontends consume that table, with no imports from tests or hardware-derived
patterns. Instruction patterns in `patterns.py` belong to target builders/translation
and remain a separate mechanism.

RMS prologues and last-axis softmax epilogues use the existing shared GEMM
builder. `normalization.py` also registers separately priced materialized
alternatives through the common sequential-region builder; their costs include
the component kernels and intermediate movement. This adds no scheduler or
extension contract. The same composition also covers GEMM followed by RMSNorm:
when a full-row fused epilogue cannot fit, GEMM and add/RMS can use their own
legal tiles and materialize the intermediate. The fused candidate remains in
the search; this is not a dimension-specific fallback.

`legalize` recognizes private FP32 `softmax((Q @ K.T) * scale) @ V` expressions
(including division by a positive literal) as unmasked attention. It preserves
the explicit scale and reaches the existing shared attention search/builder.
Observed scores/probabilities, non-last-axis softmax, and low-precision
intermediate rounding are excluded. Streaming reassociates FP32 arithmetic;
numerical equivalence is accuracy-qualified, not bitwise. The FP32 builder omits
redundant same-dtype casts so scale/store/row-reduction instruction fusion remains
visible. Actual dtype conversions remain explicit.

Unweighted RMS recognition is limited to FP32, power-of-two
last dimensions and the exact `x / sqrt(sum(square(x) / width))` expression with
keepdim. It is accuracy-qualified, not bitwise equivalent across reduction
orders, overflow or underflow. Other expressions retain their original graph.
The ordinary row and spread translators support absent RMS weights; this does
not establish support for every K-major representation.

The November 2 shared-queue evidence validates full-size RMSNorm+Matmul and
Add+RMSNorm+Matmul against the original NumPy functions at relative L2 < 2e-5.
Held-out shapes, materialized alternatives, saved emission and an unchanged-path
lifecycle comparison are covered by `tests/test_trainium_alternatives.py`.
Model latency is in cycles; conversion to nanoseconds divides by the configured
frequency in GHz. The matrix change adds no performance formula or fitted parameter.

The Set2 compilation repair passes full-output relative L2 < 2e-5 on hardware:
case 08 takes 4.373250 ms and case 20 takes 143.917178 ms. The same attention
recognition improves Set1 GQA from 15.592859 to 9.009785 ms; the other 13 Set1
plans and emitted programs remain identical. These are LNC1 resident-device
measurements, 2 x (10 warmups + 40 timed calls), excluding host glue/transfers.
The large attention still emits 1.29 million instructions and requires a
memory-aware source-parser/isolated-backend compilation workflow. See the
[repair evidence](../../../../../reports/set2-compile-fixes-20261104/REVIEW.txt)
for measurements, compiler-memory limits and reproduction scripts.

Emitter layout analysis deduplicates identical `OpUse` constraints, whose unions
and maxima are independent of multiplicity. Repeated row access syntax is cached
within one kernel emission pass using the resolved view, reshape dimensions,
slot and requested row/column range. Both caches reset with the emission pass;
they do not cache candidate predictions or alter schedules, buffers or placement.

`dag.py` groups dtype-preserving FP32 or BF16 pointwise DAGs, including branches and
multiple live outputs. Its implementation uses `build_pipelined_buffers` for
movement, slots, waits and loops. Ordinary SRAM destinations and last-use coloring
represent intermediates; it introduces no native whole-kernel invocation or new
scheduler. The target `lower` hook exposes eligible short fused chains and outlines
connected regions. Inputs may also be contiguous suffixes broadcast over leading
dimensions. The builder visits outer indices inside each suffix tile, allowing the
shared pipeline to reuse the loaded broadcast tile. It does not expand the input
in HBM. Other broadcast geometries, strided inputs and mixed types retain the
materialized path. The external allocation retains the logical output shape;
internal transfer maps are flat or two-dimensional. Export-only example clones
handle disjoint aliased inputs without introducing executable copies.

Dtype is a semantic constraint, not an automatic precision choice. Every operation
writes a destination of its declared dtype, preserving intermediate BF16 rounding.
The candidate's buffer capacity, DMA bytes and engine service use that dtype;
division's reciprocal scratch remains FP32. Selecting BF16 for an FP32 graph would
need a separate explicit numerical policy. The direct `compile_bufferized` entry
expects an already lowered graph and does not run this outlining hook itself.

Oversized DAG searches rank all fitting parameter sets with the existing
analytical cost before building at most 63, reserving the default planner's
64th trial for the materialized comparator. Smaller searches retain their
original order. Stable latency ordering retains the modeled winner and its
tie order without changing costs; allocation refinement can try the remaining
shortlisted alternatives. Region inlining binds actual positional and keyword
arguments, preserving repeated operands introduced by transpose cancellation.

For ordinary pointwise tiles spread over SBUF partitions, the DMA model counts
contiguous HBM runs and applies the existing row overhead to those runs. It uses
the dense NKI argument shape, matching the emitter and runtime boundary, rather
than a host view's parent strides. This corrects fragmented-access ranking without
changing bandwidth constants. It does not model every emitted command, Walrus
packet, hoisted load or queue dependency; geometry correction is not a claim of
cycle-accurate prediction. The November4 implementation/access report records
the model error, isolated controls and unchanged-path regression evidence.

Candidates pair a model with physical width, two/three pipeline slots and optional
ScalarEngine placement for literal add/multiply. Widths use a bounded 128-column
lattice up to8192, with exact divisibility and the existing usable-SBUF limit;
this is a search policy, not a hardware limit. `TargetInvoke` on individual ordinary
primitives freezes width/engine selection into saved PrimOp records. It does not
own private allocation. Emission consumes these records without making a new
tiling decision. A separately priced materialized region remains available.

The optional `compile_workload` option `trainium_approximate_reciprocal=True`
also permits ScalarEngine activation reciprocal in this DAG family. It is off
by default: the SDK documents lower precision than VectorEngine reciprocal and
the supported input range `±[2^-42, 2^42]`. The caller must qualify its numerical
contract and input domain; host replay alone cannot validate approximate hardware
arithmetic. The benchmark harness exposes `--dag-approximate-reciprocal` explicitly.
Search retains the original VectorEngine form and scalar-literal-only control.
This is a numerical candidate, not an emit-time rewrite. It shares the existing
reciprocal scratch and buffer plan. Ordinary standalone reciprocal also emits
the VectorEngine form unless the saved choice requests activation.

The DAG cost reuses `TrainiumModel` engine/DMA/sweep services, accounts for every
input/output transfer, and bounds compute by both engine totals and dependencies,
including scratch reuse. Vector reciprocal prices eight elements of work per
logical element, following the Neuron2.31 documented `max(MIN_II,8*N)` throughput
versus `max(MIN_II,N)` for ScalarEngine activation. Tile widths here exceed the
documented approximate64-cycle initiation interval. The shared engine clocks and
overheads remain unchanged; this adds instruction work rather than a fitted
kernel factor. It is not a cycle-exact scheduler: two versus three slots
currently share the overlap estimate; backend automatic engine placement and
instruction-specific service costs remain limitations. The November2 DAG evidence
under `reports/implementation-dag-20261102` retains those estimation errors rather
than installing per-kernel timing factors. Tests in `test_trainium_dag.py` cover
branches, retained intermediate outputs, disjoint views, guards, host replay and
fresh saved emission on held-out shapes. Hardware evidence is scoped separately
to the benchmark RoPE, AdamW and SiLU shapes.

The optional `trainium_compensated_matmul=True` admits
`trainium.gemm.compensated` alongside ordinary FP32 GEMM. It is off by default.
The recipe splits FP32 operands into high and residual halves (FP16 or BF16),
then accumulates high×high, high×low and low×high in FP32. It omits low×low;
rounding, FP16 range and residual underflow require application-specific numerical
qualification. It is not an exact FP32 replacement or a range-checked fallback.
Search compares both formats at their modeled cost; a tie does not establish
equal numerical quality. The benchmark harness exposes `--compensated-matmul`.

This recipe subclasses the existing GEMM step/model and uses `_gemm_plan`,
`build_pipelined_buffers`, shared operand reuse, transpose loads, PSUM planning
and eviction. Four ordinary SRAM destinations hold high/residual components;
decomposition repeats when an input tile changes. Selected dtype and all three
products are explicit in the frozen buffer program and survive saved emission.
There is no native whole-kernel body, new public hook or separate DMA scheduler.
Its current scope is two-input FP32 matrix contractions whose operands land
K-major, including batched and input-transpose forms. Bias, computed prologues,
epilogues, MX representations and transposed outputs retain other recipes.

The paired cost uses three half-precision TensorEngine products, FP32 conversion
passes, input-reuse counts, explicit split-buffer capacity and the existing
DMA/sweep services. FP32 input transposes are charged once per loaded tile.
These are resource costs with an overlap assumption, not a cycle-exact backend
schedule. The held-out small-matrix latency error remains substantial even when
a full-size case is predicted closely. `test_trainium_compensated.py` covers
split-K, batch, format, numerical-policy guards, input preservation and fresh
saved emission. Dated hardware evidence is in
`reports/implementation-compensated-20261102` outside this checkout.

Rank-two `numpy_T` canonicalization and cancellation of inverse transposes across
a region boundary preserve other external uses. These layout changes are
independent of the numerical option. Existing host padding/cropping is still
outside resident-device timings and must be disclosed in comparisons.

## Static preparation, padding and local checks

K-major broadcast/RMS-scale relayouts request ordinary graph transforms through
common normalization. Shared constant preparation decides immutability and the
stored-byte budget. If an original remains live, a runtime transpose may remain:
the transposing load uses DMA into SBUF, Tensor Engine through PSUM, then SBUF.
The activation-layout heuristic treats a constant relayout as free only when one
immutable representation replaces all readers; otherwise it charges the
transposed-block estimate. This does not claim measured speedup.

SBUF/PSUM partition, MX packing and attention constraints remain target-specific.
Kernel manifests record partially written output ranges for per-kernel checking.
The completed program still requires full-output comparison; partial MX writes
remain unsupported. Older manifests without ranges retain whole-output checking.
Partial ordinary HBM outputs receive the existing destination as a kernel input,
so subsequent calls preserve the ranges written by earlier calls. Mutable
destination inputs are cloned during per-call comparison to protect references.
Not every generic halo/partial-transfer form translates to NKI. `patterns.py`
implements instruction fusions, such as paired scalar ALU operations in one
`tensor_scalar`; translation packs MX codes/scales with required padding. These
runtime/encoding transformations are distinct from generic constant folding.

From the repository root, run `PYTHONPATH=src python
validation/compare_trainium_constants.py OUTPUT` against the unchanged sibling
`voyager-compiler-2` (or supply `--donor`). It checks eager/bufferized outputs,
stored constants, selected mappings/model estimates and emitted NKI, not Neuron
execution or measured performance. Shared changes also need affected target gates.
Donor ISA comparator: `9bdf4ee9cb11d2b5eaa0fc9a46201746c2bbf79c`.

## How a model gets there

`workloads.mamba.SelectiveScan` exposes `trainium_ops::selective_scan` for
`delta/u [channels, time]`, `A [channels, states]` and `B/C [states, time]`.
Its CPU reference follows NumPy FP32 exponential and serial recurrence rounding.
It does not recognize an arbitrary Python-unrolled recurrence. The native recipe
currently requires FP32 and channel counts divisible by 128. Recipe search
selects a sequence tile, declares private SBUF capacity and implicit HBM traffic,
and prices scalar/vector/DMA work. Saved `TargetInvoke` records that tile;
emission uses `tensor_tensor_scan` with explicit state carried between tiles.
NKI owns private allocation and scheduling. The estimate is analytical and does
not imply a fitted scan throughput. Delta and U are retained explicitly across
states; their full sequence storage is included in the private reservation.
The model uses separate vector/scalar/DMA work budgets, exposed fill/drain and
one two-ALU scan instruction overhead, with a one-element carry-copy allowance.
It does not sum all engines as sequential work. This overlap assumption was
checked with Trn2 NTFF profiles; it is not a general promise for every native
macro or a dependency-exact downstream schedule. Saved-emission and
semantic tests are in `tests/test_trainium_alternatives.py`.

Use `pipeline.compile_workload(workload, registry.target("trainium"), output,
config=TRN2)` (or `TRN3`) with a prepared `WorkloadSpec`. The registered target
applies `LAYOUT`, pairs search with `TrainiumModel` and emits `kernels.py`,
`program.py` and `manifest.json` under `output/nki/`. Do not invoke `translate()`
a second time as part of this pipeline.

`runtime/check.py`, `runtime/trace.py` and `runtime/run.py` retain the inherited
Neuron simulator, NEFF tracing and device endpoints. They require the appropriate
Neuron environment and recorded tensor inputs; availability of these scripts
does not mean the integration has run them. `benchmarks/trainium/compile_lm.py`
is an inherited manual frontend/reference workflow; inspect its actual invocation
and imported modules when using it. Historical environment paths and donor
measurements are not setup instructions or fresh acceptance evidence here.

## Design rules

Apply these Trainium-specific rules within the shared compilation contract.
Other targets may need different primitive representations or execution models.

The selected NKI program is not necessarily the final instruction inventory.
Downstream allocation, DMA optimization, copy elimination and synchronization
lowering can change transfers, alias copies and engine waits. In particular,
`CompileOptions.disable_backend_optimizations()` changes two NKI pipeline options;
it does not suppress every later Neuron compiler pass. `dma_copy` with unknown
DGE mode also leaves descriptor generation to that compiler. Keep its installed
version and options with any model-validation evidence. Inspect compiler IR and
NTFF together before treating a source-level copy as one hardware instruction,
or a source-level DMA subdivision as fewer physical transfers. Small-DMA and
reuse behavior is conditional on shape, layout, dtype, lifetimes and capacity;
do not generalize a measured threshold from one descriptor path to all kernels.

1. **Translate the selected execution.** Instruction/transfer expansion follows
   the chosen primitive's limits, including FP8 row mode and MX contraction
   format. It does not select a different tiling, fusion or submission policy.
   Model the resulting command geometry, movement and dependencies; fixed ISA
   expansion does not excuse an unpriced change in execution behavior.
2. **PSUM is in the IR.** It is the IR memory level
   `MEMORY_LEVEL_LOCAL_BUFFER` (space tag `"Local"`). The builders allocate
   the PSUM buffers and state which one each matmul writes, the copies from
   PSUM to SBUF, a split-K sum kept in PSUM and attention scores kept
   there. NKI places on-chip buffers itself, so memory planning gives a
   PSUM buffer no address.
3. **A kernel that computes differently gets its own builder.** When a
   kernel's computation differs from the shared builder's in more than
   details, register its builder with the paired model/cost as an
   `ImplementationSpec` in `TargetSpec.implementations`. A kernel with the shared
   schedule subclasses the shared builder and overrides only its steps.
   Shared code changes only where a hook is needed, and Voyager's
   `model.txt` stays the same.
4. **Two levels of fusion, one representation.** General fusion runs in
   `transform()` on the model's own ops (`fuse_operator`, e.g. a GEMM and
   its dequantize tail). Instruction-level fusion groups the ops a backend
   builder creates into single NKI instructions (`fuse_patterns`); it runs
   inside each builder on the subgraph it just built, before bufferization
   splices it in, so the groups still get dtype stamping, renaming and
   layer tags. Both match a pattern the same way: a sequence of
   `OpMatcher`, one per op along a producer-consumer chain, each with an
   optional check of that op's operands. An instruction-level pattern also
   has a name, the NKI instruction form: the fused node records it,
   `model.txt` carries it in the fused op's `pattern` field, and the
   translator dispatches on it and raises on an unknown one. Builders emit
   plain aten / `quantized_ops` ops and do not decide the grouping.
   Voyager's patterns have no name, so its `model.txt` is unchanged.
5. **Prefer existing semantic ops for Trainium instruction forms.** Use named
   patterns/selected parameters for ordinary instruction realization; whole-kernel
   swaps may need a custom op. This is a Trainium convention, not a framework-wide
   ban on target primitives. Other targets may register primitive operators with
   explicit numerical, alias and effect semantics.
6. **Operand layouts are an argument, not an op variant.** A GEMM op takes
   one `layout` string, `"<activation>,<weight>[,<output>]"`, in the
   `--gemm_layout` letters (`x` the rows M, `c` the contraction K, `k` the
   output columns N): `"cx,ck"`, or `"cx,ck,kx"` for an output stored
   `[N, M]`. Likewise a GEMM writing PSUM says so by its `out_dtype`
   argument (float32): its product is summed and returned in it.
7. **Fit proposes; expanded allocation validates.** The model's placement
   records buffer/scratch requirements for the candidate. Builders materialize
   the selected recipe, and the shared allocator validates its actual expanded
   footprint before freezing. Share sizing rules to avoid divergent estimates.
   The translator must not add scratch absent from the selected buffer program.
8. **Backend logic lives in backend code.** A shared option or hook that
   exists only to carry one backend's behaviour is that backend hardcoded
   in shared code. Shared code changes only by small, backend-neutral
   extensions.
9. **Trainium-only paths live in the Trainium subclass.** Everything
   K-major in flash attention -- the query and key stored
   `[head_dim, tokens]`, their tiles' addressing, a one-token query's
   transposing load, a transposed output -- is in `TrainiumFlashAttention`.
   The shared FA3 builder keeps Voyager's schedule, the improvements every
   backend takes and the hooks the subclass overrides.
10. **A transposing load is the operand's load functions.** The shared
    scheduler keeps all its loading logic; an operand's spec may carry
    `load_buffers` (called once before the loop) and `load` (called in
    place of `async_copy`). Trainium's GEMM builder sets both for an
    operand it transposes on TensorE (`load_transposed`); Voyager sets
    none.

## What is shared and what is custom

| Stage | Trainium |
|---|---|
| Export, PT2E quantization, `transform()` passes | shared as is; `LAYOUT` stores every GEMM weight contraction-major (`ck`, [in, out]), the activations between matrix-matrix GEMMs contraction-major too (`gemm_layout="cx,ck"`), and keeps feature maps NCHW |
| Tile search (interstellar for GEMMs, the shared grid search for vector ops and GEMVs) | shared; it asks the target's `RuntimeModel` / `OpModel` contracts (`extensions.py`) to prepare each op, then whether each candidate fits and what it costs |
| Runtime model | custom: `TrainiumModel` in `model.py`, preparing per op a `TrainiumGemm` or `TrainiumGemv` (`gemm.py`), a `TrainiumAttention` (`attention.py`) or a `TrainiumVector` or `TrainiumPool` (`vector.py`) from the graph facts in `compiler/search/lowering.py` |
| Bufferization, memory planning, `model.txt` | shared, but a dense GEMM (`linear`, `matmul`, their MX forms, a decode GEMV) lowers by Trainium's own builder, `gemm.py`, and a K-major flash attention by `attention.py` (`TrainiumFlashAttention`, a subclass of the shared FA3 pipeline), selected through `TargetSpec.implementations`; FP8 alternatives are compared rather than first-match dispatched |
| Translation to NKI | custom: `translate.py` |
| Simulation, trace, device runs | custom: `runtime/` |

## Files

One file per op, holding its op class (fit and cost) and its builder:

- `gemm.py`: `TrainiumGemm` and `TrainiumGemv`, and the GEMM builder
  (`build_gemm`).
- `attention.py`: `TrainiumAttention`, and the flash-attention builder
  (`TrainiumFlashAttention`).
- `vector.py`: `TrainiumVector` and `TrainiumPool`, which the shared
  builders lower.
- `utils.py`: what the op files share: `TrainiumPlacement`, the SBUF and
  PSUM sizes of placed buffers, the scratch regions the translator computes
  in, and `TrainiumGrid`, the fit of a vector op and a GEMV.
- `model.py`: `TrainiumModel`, which hands each op to its op class and
  prices what they share (an engine instruction, a DMA, an MX tile, a
  sweep). It imports the op files; they import only `utils.py`.
- `patterns.py`: the instruction patterns; `config.py`: `TrainiumConfig`,
  `TRN2` / `TRN3`; `target.py` / `implementations.py`: recipe registration;
  `translate.py`: NKI projection; `runtime/`: simulation, tracing and device runs.

## How the IR becomes NKI

- **Control** (loops, conditions, index arithmetic) runs at translation
  time. A kernel is the straight-line sequence of transfers and compute
  steps the schedule makes. Semaphores and waits are dropped: NKI orders
  instructions by their data dependences. This relies on the generated NKI
  preserving data and storage-alias dependencies; dropping semaphore syntax is
  not itself proof of correctness or modeled overlap. Check the lowered endpoint
  when changes affect buffer reuse, command admission or synchronization.
- **SBUF tiles** normally have rows (all dims but the last, flattened) on the
  partitions and the last dim free. Eligible flat elementwise tiles spread
  across partitions even with multiple rows. Broadcast operands with fewer rows
  are replicated onto their reading partitions; incompatible tilings are rejected.
  The vector model accounts for the same layout, footprint and repeated loads.
  A tile one
  access pattern walks moves in one `dma_copy`, except a load of 1 MiB or
  more: that moves in pieces of 256 KiB or more of whole 128-row blocks,
  the last piece whatever blocks remain, so a GEMM's first matmuls start
  on the first piece of a weight tile and the next tile's pieces land as
  the current one's blocks free up. A tile no single pattern walks moves
  in one `dma_copy` per 128 rows, split further until a pattern walks each
  part. `transform` runs with `tensor_layout="nchw"`, so a pool's H lands
  on the partitions while the GEMM weights stay `"ck"`.
- **A 2-D max pool** loads each output row's kH input rows side by side
  onto that row's partition, at most 128 rows per tile, and runs kH-1
  vertical and kW-1 horizontal maxes on VectorE.
- **Access patterns** come from the elements' offsets: each instruction's
  operands are mapped element by element and factored into one shared
  nesting (`fit`). Views and relayouts fused into a kernel need no special
  case.
- **A GEMM step** is written out in the IR by Trainium's GEMM builder
  (`gemm.py`), on top of the shared plan and synchronous schedule. Its
  output tile is cut into slices of at most 128 rows and 512 columns, each
  one GEMM into a PSUM box, summed in float32 (`out_dtype`). With the
  reduction in one step the slices' GEMMs write the 4 rotating
  `[128, 512]` float32 accumulators in turn,
  each slice evicted at once: the fused tail run on it, or a copy, writes
  the output tile from PSUM. A reduction split over steps keeps the whole
  output tile's sum in PSUM, a bank per slice: the first step's GEMMs write
  it, the later ones add to it (`matmul_accumulate`), and the last step
  evicts each slice at once -- or every slice after its GEMMs, when the
  output tile is the sum itself. A row-major activation is transposed on
  TensorE per 128 rows, its 128 x 128 blocks side by side into a PSUM
  transposition tile of half a bank, each tile copied out whole into an
  SBUF tile the GEMMs read contraction-major. A tail ending in a transpose
  of the last two dims evicts the slices into a staging tile, then
  transposes the finished tile on TensorE into the output. The
  instruction patterns then fuse the builder's ops.
- **Buffers** are decided once, by each op class's fit, and
  listed in its `TrainiumPlacement`: `buffers`, the SBUF and PSUM tiles a
  Trainium builder allocates beside its streams (`gemm_buffers` for a GEMM
  or GEMV, `TrainiumAttention.buffers` for a flash attention), and
  `scratch`, the regions the translator computes an op group's
  intermediates in (`scratch_regions`: a chain's float32 pairs, a
  `quantize_mx`'s rearranged input and transposes, a K-major RMS norm's
  ones matrix and sums, a norm's statistics), which the builders attach to
  the op with `with_scratch`. An op built whole without a search takes its
  scratch from its op class (`OpModel.scratch`). A kernel's PSUM must fit
  the core's `psum_banks` x `psum_bank_bytes` (`fits_psum`; NKI reserves
  no bank for a kernel that never asks it to lower a `dma_transpose`), and
  the fit rejects a tiling whose buffers do not fit PSUM or SBUF. No SBUF
  is reserved (`scratchpad_offset` is 0): the translator declares the IR's
  boxes and nothing else.
- **PSUM boxes**: a box the IR places at `MEMORY_LEVEL_LOCAL_BUFFER` is
  declared in PSUM (`nl.psum`), laid out as an SBUF tile is but with an
  8-bit element two bytes apart. The ops writing it, and the fused groups
  that are one instruction, are in "Fused patterns" below.
- **A GEMM prologue**, the ops fused ahead of a row-major GEMM that
  compute its activation (an add, then an RMS norm), runs on each 128-row
  chunk into an SBUF tile before the transposes, a row op over whole
  rows, so `TrainiumModel` keeps the contraction whole. It reruns for
  every N tile, which the model prices.
- **A row-wise op** (layer norm, RMS norm, softmax) runs 128 rows at a
  time, one row per partition. A norm takes its rows' moments with
  `bn_stats` / `bn_aggr`; an RMS norm's mean square is `mean² + variance`,
  and one pass then scales each row by its rstd and the weight. The ops
  fused before it write its rows to the destination first, and it then
  runs there in place. A GEMM tail ending in one evicts each 128-row
  chunk's slices into the output tile, then the builder runs it on the
  chunk's rows, so `TrainiumModel` keeps an output tile's rows whole.
- **A K-major GEMM** (`linear` or `matmul` of layout `"cx,ck"`) reads
  its activation `[..., K, M]` as stored, the contraction already on the
  partitions, so it transposes nothing: the activation is the stationary
  operand, or, when the GEMM writes its output transposed (`[..., N, M]`,
  layout `"cx,ck,kx"`), the moving one and the weight the stationary, and
  the slices run over 128 of N and 512 of M.
- **A K-major RMS norm** (`rms_norm_kmajor`) normalizes over the
  partitions: per 512 columns, the 128-row blocks are squared on ScalarE,
  8 blocks an instruction, and a ones-matmul on TensorE sums the squares
  over the blocks and the partitions into PSUM, landing the sum on every
  partition; one VectorE pass per block then scales by the rstd and the
  `[H, 1]` weight.
- **RoPE's rotate-half** is a K-major `matmul` by the `[D, D]` signed
  permutation constant, an ordinary K-major GEMM whose tail (`* sin`, `+ x *
  cos`) reads each slice from PSUM.
- **A flash attention** (`sdpa_kmajor`) lowers by Trainium's own builder,
  `attention.py`: `TrainiumFlashAttention`, a subclass of the shared FA3
  pipeline (`bufferize/attention_v3.py`) that keeps its walk, DMA streams
  and semaphores and states each step's instructions in the IR, per block
  of 128 query rows, written out unrolled. Everything K-major -- the query
  and key stored `[head_dim, tokens]`, their tiles' addressing, a
  one-token query's load and a transposed output -- is in
  `TrainiumFlashAttention`; the shared builder lowers only Voyager's
  attentions. QKᵀ writes the scores into PSUM in float32, a bank at a
  time; on a step that adds a mask (under `is_causal`, a pair crossing the
  diagonal), a matmul through an identity of `1 / scale`, built once on
  chip (`scaled_identity`), adds it onto them there (`matmul_accumulate`).
  Per block, one VectorE `tensor_scalar_reduce` per bank evicts the scores
  scaled with their row max, a `tensor_scalar` makes the bias `-max(m,
  max)`, and one ScalarE `activation` takes the exponentials in place with
  their row sum. Alpha, `m` and `l` then run over every block's float32
  columns at once. P is transposed on TensorE into PSUM and copied into an
  SBUF Pᵀ, which P·V reads; P·V's result stays in PSUM for the rescale `o
  = alpha·(o + pv)` to read. A scaling by alpha or `l` runs on VectorE,
  since ScalarE runs the exponentials. Scores wider than a PSUM bank are
  evicted a bank at a time, each bank's row max into a column of its own,
  then one VectorE pass takes the max of those columns.
- **A decode step's flash attention** is the same kernel over one query
  token. The layout pass stores the key cache `[head_dim, positions]`, so
  QKᵀ reads it as stored, and the GQA fold puts the query heads of a KV
  head (4 in Llama-3.1-8B) on one query tile, loaded row-major and
  transposed on TensorE through PSUM once per KV head, and its output
  transposed back the same way, all stated in the IR. A 1152-slot cache is
  one key block, so each KV head is one step, and the mask, broadcast over
  the heads, is one tile loaded once.
- **A transposing load** (an operand a GEMM reads transposed from how it
  is stored) lands as stored into staging slots by DMA, then one commit
  transposes it on TensorE through a PSUM tile into its slot
  (`load_transposed`, the operand spec's `load` and `load_buffers`): the
  blocks of one block row of the result go side by side into a PSUM tile
  of up to a bank, as the kernel's other PSUM tiles leave room for, and
  are copied out together. TensorE reads a block from partition 0 only,
  so a load of several leading elements whose rows do not fill whole
  128-row runs (per-head rows under 128) is transposed by the DMA instead,
  element by element.
- **A layer norm** normalizes, scales and shifts in its output tile and
  keeps its statistics in the scratch regions its op class lists.
- **Immediates** (constants the IR names but does not hold, such as the
  attention scale) are written into the kernel as literals.
- **DRAM-only ops** (embedding, slice, cat) run in the host program as
  torch ops.

- **MX (Trn3)** arrays -- the codes and E8M0 block scales a `linear_mx`
  multiplies -- are stored the way `nc_matmul_mx` reads them (`Mx` in
  `translate.py`): 4 consecutive contraction elements per 32-bit word, a
  chunk of 512 on the 128 partitions, each further chunk beside it, and
  each block's scale on the first 4 partitions of its quadrant.  In HBM
  that is `[128, K/512, X]` int32 codes and `[16, K/512, X]` uint8 scales,
  `X` the activation's rows or the weight's columns; `record` saves the
  references packed this way.  A `linear_mx` slice runs one
  `nc_matmul_mx` per 512 of the contraction, with no transposes (a single
  row is padded to the even width the instruction needs).  An op ending
  in `quantize_mx` writes its result in BF16, transposes every 4th column
  onto the partitions on TensorE and quantizes there with
  `nisa.quantize_mx`.  The tile search keeps MX contractions in whole
  chunks of 512 (`TrainiumModel`).

- **Host statements** are everything outside a kernel: DRAM-only ATen ops
  (embedding, slice, cat, a KV-cache `index_copy_`) called by schema
  keyword, an in-place op as its functional twin rebinding what it
  mutates, scalar arithmetic on host values, a compute op writing an HBM
  window, and HBM allocations no kernel makes. The parameters the program
  writes (its state) are stored back so a trace can alias them.

## Fused patterns

Fusion runs at two levels, with one representation: a `Pattern`
(`voyager_compiler`), the `OpMatcher` of each op along a producer-consumer
chain, each optionally checking that op's operands, and an optional name.

- **General fusion** runs in `transform()` on the model's own ops
  (`fuse_operator`), by the unnamed patterns it is given: a GEMM or a norm
  and the ops after it, any of which a group may skip.
- **Instruction-level fusion** runs inside each Trainium builder on the
  subgraph it just built: `fuse_patterns(gm, INSTRUCTION_PATTERNS)`, in that
  subgraph and the loop bodies and branches it calls. Each pattern in
  `patterns.py` is named for the NKI instruction it is. A chain matches only
  with an op for every matcher, and where two patterns match one op the
  earlier takes it. A group's results are its values read outside it, in
  program order: its last op's, and the tile a reduction marked
  `keeps_input` reads, which the instruction stores as it reduces it (a
  chain whose tile is read only by the reduction stays unfused). The group
  runs where its last op stood, so a builder reduces a tile before storing
  it.

The fused node records the pattern's name (`meta['pattern']`), `model.txt`
carries it as the fused op's `pattern`, and the translator emits the group
as that instruction, one per 128 rows (`PATTERN_HANDLERS`), raising on a
name it does not know. A custom op library is only for swapping a whole
pattern for a dedicated kernel, the way `aten.sdpa` stands for a whole
attention, never for a single instruction.

| pattern | ops in the group | NKI instruction | why it is one instruction |
|---|---|---|---|
| `tensor_scalar` | an ALU op (`add`, `sub`, `mul`, `maximum`) of a tile and a column of one value per row, then one of its result and a constant | `tensor_scalar` with two operators, on VectorE | the engine applies two operators to each element in one pass, e.g. the softmax bias `-max(old max, new max)` as `maximum` then `mul` by -1 |
| `tensor_scalar_reduce` | an ALU op of a tile and a constant, a cast (`to`) when it is stored in another type, then `amax` or `sum` over its last dim, kept | `tensor_scalar_reduce` | the engine reduces each row of its result as stored in the same pass, e.g. scores copied out of PSUM scaled, with their row max |
| `activation_reduce` | `add` of a tile and a column, `exp`, then `sum` over its last dim, kept | `activation` with `reduce_op=nl.add` | ScalarE adds the bias before the function and sums each row of the result in the same pass, e.g. a softmax's exponentials with their row sums |
| `scaled_identity` | `eye`, then `mul` by a constant | `memset` of the constant and `affine_select` keeping the diagonal | the identity a mask is added through, scaled by `1 / scale`, is made on chip once |
| `matmul_accumulate` | a K-major GEMM (a `matmul`, or a `linear` without a bias, of activation layout `"cx"`) or an MX one (a `linear_mx` without a bias), then `add` of its product and the PSUM partial sum the group writes | `nc_matmul` (`nc_matmul_mx`) with `accumulate=True` | TensorE adds its product into the PSUM tile it writes, e.g. a split reduction's later steps, or a mask added onto scores by a matmul through an identity |

To add an instruction: append a named `Pattern` to `INSTRUCTION_PATTERNS`,
one `OpMatcher` per op with a predicate on that op's operands (the ones in
`patterns.py`, or a new small one), marking a reduction the instruction
makes of the tile it stores `keeps_input=True`; a form whose op is optional
is two patterns of one name. Then add the `Kernel` method emitting it to
`PATTERN_HANDLERS` in `translate.py` under the name, and a test in
`test/unit/test_trainium.py` that `fuse_patterns` forms the group from plain ops
and that it translates to the instruction.

A lone op writing a PSUM box is the instruction that writes PSUM: an
`aten.transpose` one `nc_transpose` per 128 x 128 block, a K-major GEMM one
`nc_matmul` per 128 rows, 512 columns and 128-deep contraction chunk, a
`linear_mx` one `nc_matmul_mx` per 128 rows, 512 columns and 512-deep
chunk. An `aten.eye` is a `memset` of ones and an `affine_select` keeping
the diagonal, per 128 rows. An op reading a PSUM box of whole 128-row
blocks runs on all its blocks at once, side by side in their bank.

## Historical donor verification (not rerun in this integration)

Kernels passing the CPU simulator against Voyager's reference, and
compiling for the target (distinct kernels): trn2 for BF16 and FP8, trn3
for MX (MXFP8 E4M3).

| Design point | Llama-3.1-8B simulator | Llama-3.1-8B compiled | Qwen3-4B simulator | Qwen3-4B compiled |
|---|---|---|---|---|
| BF16 prefill | 20/20 | 17/17 | 22/22 | 20/20 |
| BF16 decode | 37/37 | 30/30 | 39/39 | 33/33 |
| FP8 prefill | 20/20 | 20/20 | 22/22 | 22/22 |
| FP8 decode | 34/34 | 32/32 | 36/36 | 34/34 |
| MXFP8 prefill | 20/20 | 17/17 | 22/22 | 20/20 |
| MXFP8 decode | 34/34 | 29/29 | 36/36 | 32/32 |

Every step traces into one NEFF for its target.  All eight Trn2 steps run
on a Trn2 device and match Voyager's output (cosine 0.9993 to 0.99998,
the same predicted token).  The MX steps have only run in the simulator:
whole steps at cosine 0.99965 and 0.99959 (Llama and Qwen prefill),
0.99943 and 0.99836 (decode).  An MX output is checked as values, block
by block: an element may round one FP8 step the other way where its BF16
input differs by a rounding, and those flips add up over a step's four
quantizations, so a whole MX step passes at 0.998 (E4M3) rather than
0.999, and at 0.992 for E5M2, whose step is twice as coarse.  MXFP8 E5M2
passes the same checks for Llama prefill (whole step 0.99797).

FP8 is Trn2's own E4M3, the dtype `fp8_e4m3_ieee`: the IEEE kind, which
keeps its top exponent for inf and NaN and so tops out at 240 (Voyager's
`fp8_e4m3` is OCP's E4M3FN, up to 448).

The BF16 prefill programs run whole in the simulator at cosine 0.99993
(Llama: argmax and top-5 equal), and every design point's step traces into
one trn2 NEFF.  The 32-layer Llama prefill translates to 578 calls of the
same 17 kernels and traces into one trn2 NEFF in about two minutes: the
decoder-layer markers let the compiler build one layer for all 32.

A GEMM writing PSUM sums in float32 in the reference too (`out_dtype`),
so a split-K GEMM rounds once, as the kernel does.  An FP8 result may
round one step the other way.

A value has one destination in `model.txt`: decode's rotary
`cat(freqs, freqs)` stores its angles with two clones, one per half
(`bufferization._dedup_regions` never merges two stored values, and
`emit` rejects a second store of one).

A reference through a reshape of a quantized box gives the view's
physical dtype (`bfloat16`) rather than the box's (`fp8_e4m3`), a
difference in the shared compiler; the translator takes each box's dtype
from its declaration.

The simulator is functional only and lenient: four hardware rules surfaced
only in the trn2 compile -- no `mod` or `divide` ALU op, per-row scalar
operands in FP32, no free dim crossing partitions, FP8 transposes at stride
2 -- so every translation is compiled too (`check.py --neff`).

An `fp8_e4m3_ieee` step crosses into XLA as torch's `float8_e4m3fn`,
which has the same bits up to 240: `trace.py` sets NKI's non-OCP mode and
NxD Inference's `--experimental-unsafe-fp8e4m3fn-as-fp8e4m3`.

A traced decode step is specialized to its example position: the host
program reads the position as a Python scalar, which a trace cannot, so
it takes the value the reference run recorded -- a decode is compiled
with references (without `--no-reference`).

## Automatic copy service attribution

The GEMM model can share unpinned, same-type FP32 PSUM-to-SBUF copies
between VectorE and ScalarE on Trn2, for one row group per pipeline step
(`rows <= partitions`). It collects their service separately and balances it
after the fixed arithmetic, store issue and, for compensated products, input
conversion/relayout work. Existing engine rates and overheads are unchanged.
This is a steady-state resource approximation, not a replay of Walrus scheduling
or an instruction-engine directive. Casts, half-width copies, other targets and
multiple row groups retain their previous pricing.

The measured reason for that scope matters: unconstrained balancing selected a
256-row softmax tile that regressed from 4.470 to 6.252 ms. Its trace exposed less
overlap despite fewer copies and DMAs. The scoped model retains the 128-row
softmax kernel and estimates 4.904 ms instead of 7.535 ms. It also selects a
128-row compensated BMM that passes full numerics and improves 3.472 to 3.021 ms.
Nine other qualified kernels have identical saved emission; 33 affected tests
pass. The softmax scaling cost is deliberately unchanged. Exact pass snapshots,
rejected control and shared-queue measurements are in the
[copy-engine review](../../../../../reports/implementation-copy-engine-20261102/REVIEW.txt).

## Selective-scan numerical algorithm

The native `trainium.selective_scan` recipe accepts the explicit compilation
option `options={"trainium_scan_algorithm": "integrating_factor"}`. The default
is `"direct"`; unknown values fail rather than silently selecting a policy.
The option applies to the semantic selective-scan workload, not arbitrary
Python-unrolled recurrences. Both paths require FP32, channel multiples of 128,
and complete time panels. The existing recipe search chooses a panel, declares
private SBUF and traffic, and freezes the algorithm and width for saved emission.

The integrating-factor path computes a tile-local prefix of delta and positive
and negative exponentials of A times that prefix. Cumulative sums then reconstruct
the carried state. This avoids compounding independently rounded exponentials
over thousands of direct recurrence steps. The CPU semantic reference remains
the original serial recurrence. The native algorithm remains an opaque scheduling
boundary; this does not add a shared bufferized recurrence builder.

This is a numerical opt-in: product association, prefix sums and state evaluation
order change, and large positive or negative prefix exponents can overflow or
underflow even when a direct recurrence is finite. There is no runtime range
check or universal-error guarantee. Qualify the caller's inputs and tolerance.
The resource model prices the declared ScalarE, VectorE and DMA work using the
existing service rates and fill/drain assumptions. It does not fit a per-kernel
correction or reconstruct every dependency; tested panel rankings can disagree
with hardware. The retained direct path still fails the Set1 long-scan relative-L2
gate, while the integrating option passes the recorded benchmark and held-outs.
See the [numerical diagnosis and qualification](../../../../../reports/mamba-numerics-20261103/REVIEW.txt).

## Not supported yet

Kernels that read an HBM box while writing it, partial MX outputs, transfers with
a non-zero pad, generic partial transfers, MXFP4, convolution, pooling other than an
unpadded NCHW max_pool2d, and a fused attention other than a K-major flash
attention (`sdpa_mx`, a split KV cache's, is not lowered).
# Reuse experiment: composable products and bufferized scan

The 2026-11-03 reuse experiment is recorded under
`/home/ubuntu/AGEN-voyager/reports/reuse-mamba-20261103`. Its source snapshot,
paired LNC1 measurements and candidate identities are retained there. It does
not integrate the separate DMA-expansion predictor.

Compensated arithmetic now specializes operand preparation and slice accumulation
inside `GemmSteps`, rather than duplicating the matrix loop. The same traversal
owns activation prologues, transposition, split-K, residual/normalization/softmax
tails and final eviction. Extra high/residual buffers enter the ordinary matrix
fit calculation through a target-local helper. `trainium_compensated_matmul=True`
remains required. Bias, MX, unsupported weight layouts and transposed outputs
remain excluded. Shared control does not imply every fused arithmetic candidate
is faster or qualified over every FP32 input range.

`trainium_scan_execution="bufferized"`, together with
`trainium_scan_algorithm="integrating_factor"`, enables the shared scan experiment.
The existing `build_pipelined_buffers` owns the channel/time/state grid, input and
output slots, movement and persistent scratch. Carry and panel intermediates are
ordinary explicit SRAM buffers. The two target primitives are a tile cumulative
sum with optional incoming state and a tile scaled exponential. Their emitter
consumes the saved tile views and allocates no private scratch. B/C rows use the
existing partition-broadcast DMA path, without host replication. Capacity pricing
includes those physical replicas even though their logical buffer shape is one row.
This uses the existing pipeline, not a new recurrence scheduler or native macro.

The default execution remains `native` while this experiment is evaluated. Direct
scan remains the default numerical algorithm and its known long-case L2 failure
is unchanged. Integrating factors retain their input-range/overflow limitations.
Saved emission and multi-panel carry tests cover the new path; see the dated
report for exact hardware shapes and numerical qualification.

The native integrating-factor model now charges one vector service for its
cumulative instruction, and surviving boundary copies to ScalarE, using matching
NTFF evidence. The shared scan model prices its own actual pipeline work and
slot capacity using the existing hardware rates. Neither model receives a
benchmark-specific correction factor. Native-versus-shared ranking remains a
separate question from ranking panels within the shared path.

## Enumerated physical decode-cache capacity

The full-model Trainium benchmark defaults to `--decode-cache-policy search`
for homogeneous, batch-one BF16 single-token SDPA decode on Trn2 with the
default layouts. Other paths retain their existing capacity. `exact` retains
the original context-plus-generation alignment. An optional
`--decode-cache-max-bytes` constrains total persistent BF16 K+V storage across
the model's layers. Logical context, generation budget and update positions
are unchanged; the workload's normal mask covers the extra physical slots.

`decode_capacity.search_decode_capacity` builds shape-only normalized attention
nodes and invokes the existing shared `attention_op_tiling`, including its
product mapping searches and target fit/cost model. It does not force a tile
into compilation, adjust hardware service rates, or fuse attention/projection.
The resulting full model is still tiled by the ordinary compiler. Compare its
saved choices with the representation-search receipt before reusing a forecast.

The search exhaustively visits aligned capacities through the largest first
capacity that accommodates any PSUM-bounded aligned key tile. Two score
parities bound a tile by `(psum_banks // 2) * moving_free_max`; actual fitting
also includes all other PSUM and SBUF buffers. For each tile, its smallest
capacity is the first multiple of `lcm(tile, cache_alignment)` covering the
required context and generation budget. Further multiples add identical steps,
traffic and storage under this single-token model. The finite bound therefore
covers every undominated capacity/tile choice in the supported representation;
it does not claim to search alternative algorithms, partial final tiles or
different attention schedules. All interior aligned capacities are visited too.

The objective is strict modeled **attention** latency summed over the layers;
ties prefer less traffic and persistent memory. KV storage is a budget and
tie-break, not a latency coefficient. This estimate excludes other full-model
operators, input preparation and host overhead. The benchmark writes
`cache-capacity-search.json` with the domain, memory usage, scored/rejected tile
trials and selected capacity. It uses no model-family or preferred-capacity
table. P2048 selecting2304/1152 is a search result, not a DMA-size requirement.

Tests are in `tests/test_trainium_decode_capacity.py`; dated full-model evidence
and source-isolation checks are under
`/home/ubuntu/AGEN-voyager/reports/cache-capacity-search-20261103`.



### Optional resident decode attention/projection

`trainium_resident_handoff="auto"` in `compile_workload` options (or
`compile_lm.py --resident-handoff auto`) compares priced HBM and SBUF region
recipes. The default is `off`; `hbm` and `sbuf` force the boundary for controlled
qualification. Initial scope is BF16 batch-one, single-token decode, 128-wide
heads, an absent/BF16 mask, and a single plain output-projection consumer through
order-preserving views. Unsupported patterns retain ordinary compilation.

`handoff.py` specializes the existing FA3 output boundary and GEMV operand
reader. Shared `bufferize.residency.compose_resident_stages` joins completed
producer/consumer stages; the ordinary allocator owns their storage. The
Trainium emitter preserves live on-chip values in one kernel scope from frozen
IR. It does not select fusion or invent prefetch. Bufferized stages are ordered,
but the lower compiler can overlap independent weight DMA and ready projection
products with late attention work. The initial component-sum estimate does not
credit that overlap; see `LOWER-COMPILER-OVERLAP.txt` in the evidence directory.

The resident output is packed `[128, heads]` in SBUF (8 KiB for 32 heads), after
the original BF16 rounding. Local transpose/copy and persistent capacity are
priced; removed output/input DMAs are removed from the existing engine and
traffic calculations before recomputing latency. Costs use the unchanged
hardware resource parameters. HBM remains a comparable candidate and capacity
fallback. Qualification evidence is under
`/home/ubuntu/AGEN-voyager/reports/onchip-handoff-20261103`; consult its state for
measured scope and limitations.

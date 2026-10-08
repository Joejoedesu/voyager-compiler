# Trainium selected ISA plan, 2026-10-06

[Endpoint-constrained movement search](trainium-movement-search.md) optionally
selects alternative data-movement chains before final plan emission.

This checkout was copied from the **dirty working tree** of
`voyager-trainium-10-05-isa`. [provenance.json](../provenance.json) records the source
revision and file digests. The older checkout and results are preserved.

## Implemented boundary

Production still uses PyTorch export, shared transforms, Interstellar search,
shared bufferization and collateral emission. `TrainiumBackend.compile` then
selects a typed ISA program before `realize`/`convert`.

`instruction_plan.py` represents tensors, static address expressions, selected
ISA implementations/engines, operand bindings, completion edges, lifetimes,
physical SBUF offsets and PSUM banks. JSON contains these values, not generated
Python source. Loop iteration spans are retained as repeated-region metadata.
`compact_encoding.py` selects sequential loop encodings before conversion when
repeated bodies differ only in affine HBM addresses or validated periodic
quotient/remainder traversal addresses, including two nested wrap levels. Conversion verifies the
recorded encoding expands exactly to the selected instructions. Shapes, local
addresses, opcodes and engines cannot vary. Full maxpool source shrinks to
8.0 KB for the separable maxpool expansion; hardware correctness passes at
1016 µs. The preceding eight-maximum version was 11.5 KB and 2027 µs.
All temporaries in the supported lowering are typed, including retained-parameter
replication, PSUM clears and explicit transpose identities.

`planning.py` instantiates target templates from shared scheduled control. Its
compatibility front end types each address/ISA statement immediately.
`converter.py` never calls that planner: it authenticates model/hardware/plan
hashes, checks the hardware ISA catalog, validates memory/layout compatibility and the
complete plan, and invokes
`plan_emitter.py`. It preserves selected instruction order and cannot repair
missing selections. The explicitly requested language baseline is isolated in
`legacy_converter.py` and is outside the strict ISA contract. Historical policy
records can still be inspected, but a missing ISA-mode field cannot implicitly
select legacy conversion: that requires explicit `isa_lowering=False`.

Hardware IR contains operation implementations, reduction layout signatures,
expansion steps and the pinned ISA catalog. Shared `ImplementationValue.layout`
has a compatible default for other targets. An early target annotation prevents
shared vector padding from changing a row reduction's exact feature extent.

## Physical allocation

By default (`strict_realization=True`), every local value binds an explicit NKI allocation. A uint8 SBUF arena starts at
partition/address zero; typed accesses bind selected per-partition offsets.
Each used PSUM bank has one allocation at partition/address zero. BF16 transpose
destinations retain BF16; ordinary matmul accumulators are FP32. Identity
constants have explicit HBM DMA loads. There is no automatic local tensor
allocation in this path.

An optional preconversion `disjoint_arenas` encoding groups the connected union
of overlapping SBUF placements. Separate arenas have disjoint physical ranges;
all aliases and reuse of an overlapping range stay in one arena. Exact offsets,
instructions and dependencies remain unchanged. Small LayerNorm, BMM→softmax,
SwiGLU, padded maxpool and BF16 GEMM pass device correctness. Full production residual→RMSNorm→GEMM passes at 9.135 ms, versus 9.140 ms
with the single arena, which is not a meaningful speedup. The disjoint encoding
remains opt-in; it does not change the analytical dependency graph.

Shared largest-first best-fit planning assigns addresses using inclusive
instruction lifetimes. An opt-in size-class diagnostic uses that same planner
within separate byte-size classes, preventing differently sized generations
from partially overlapping. Full residual→RMSNorm→GEMM still passes correctness,
but its median is **10.155 ms**, 11.1% slower than the default's 9.140 ms, with
more SBUF reserved. This experiment does not change the production default and
is recorded as an allocation diagnostic, not a new shared-search result. A lifetime index accelerates overlap queries without
changing placement choices; randomized tests compare against the original
algorithm. Conservatively, every value reserves all 128 partitions. Capacity is
28 MiB SBUF and eight PSUM banks, each at most 2 KiB per partition. PSUM
placement chooses the least recently occupied legal bank, spreading independent
operations across the available banks before reuse.

RAW/WAR/WAW edges track generations. Physical reuse adds completion edges from
all outstanding accesses of the latest owner of each byte range, retaining
partial-range owners. Source-order last use alone cannot retire independent
readers on different engines. Earlier
generations are covered transitively, avoiding quadratic dependency growth.
Missing types, layouts, implementations, bindings, dependencies or placements,
overlapping live values, and illegal banks/addresses fail before emission.
Partition broadcasting requires an explicit ISA expansion; `broadcast_to` is
rejected. Negative-infinity fills use a typed arena bitcast before indexing,
because the pinned SDK cannot bitcast an indexed access-pattern object.

The pinned direct-allocation ABI requires internal HBM buffers to be I/O. They
are recorded as workspace outputs following semantic outputs. The runner checks
semantic results while retaining workspace traffic. Physical addresses are
controlled; exact engine issue times remain the Neuron backend's responsibility.
Logical buffering depth alone receives no speed credit.

## Optional native allocation (2026-10-08)

`TrainiumTuning(strict_realization=False)`, exposed as
`--no-strict-realization` by `trainium_generate.py` and `trainium_advanced.py`,
keeps shared search, bufferization, software tiles, layouts and operand policy.
The selected program uses `encoding_storage="compiler"`: SBUF/PSUM tensors
remain logical values, with no physical placements or allocator-induced reuse
edges. The emitter declares `nl.sbuf` / `nl.psum` buffers and the native compiler
chooses their addresses and final instruction schedule. Logical dependencies,
shapes, memory endpoints and individual tile capacities are still validated.
The saved policy is restored during conversion; mismatched allocation modes fail.
Strict realization remains the default. `--no-isa` and movement-chain search
cannot be combined with this mode.

The pinned SDK rejects compiler-managed explicit identity-matmul transposes
with `NCC_ITEN404: identical memlocSet name`. Relaxed mode therefore selects the
native TensorE `nisa.nc_transpose`, with a typed PSUM result and an explicit
PSUM-to-SBUF copy. The SDK owns its shared uint8 128x128 HBM identity and casts it
on load. That implicit DMA is counted separately from source calls; selected
replay includes its traffic and dependency. This means relaxed versus strict
compares allocation **and native realization**, not physical addresses alone.
No legacy converter is used. The native compiler can remove clears and fold
copies; its placement, folding, reuse and spills are not certified by the
logical replay estimate, which is explicitly marked incomplete.

Fresh one-core Trainium2 checks pass for FP32 GEMM128, GEMM256 and GEMM512,
with medians 17, 27 and 42 microseconds, versus strict 20, 79 and 169.
All three retain byte-identical shared `model.txt` programs and identical input
arrays. BF16 GEMM128 also passes (15 microseconds; one timing repeat).
This validation does not establish coverage for the six full application kernels.

The controlled `scripts/trainium_allocation_probe.py` experiment changes only
SBUF addresses and the consequent physical reuse edges, keeping instruction
order/operands and PSUM placement. For GEMM512, 1 MiB becomes 11.5625 MiB and
hardware falls from 169 to 44 microseconds. The selected-DAG prediction changes
from 137.325 to 44.654 microseconds. Native profiles preserve 128 MATMUL, 160
COPY, 72 MEMSET and 33 DMA commands and identical HBM traffic; neither spills.
The first eight weight-load issue spacings change from a median 3.695 to 0.599
microseconds. Repeated-address ScalarE waits disappear. This isolates address
reuse serialization as the main regression; reducing operation counts is a
smaller additional improvement in this experiment. This fully distinct storage
policy is diagnostic and is not a scalable production allocator.

Evidence, failed SDK probes and reproduction artifacts are in
[`results/trainium/relaxed-realization-2026-10-08`](../results/trainium/relaxed-realization-2026-10-08).
The remaining production work is to search a bounded number of temporary
buffer generations and score their physical reuse dependencies **before**
selecting the mapping. Compact candidate estimates currently see this loss of
concurrency only in the subsequent selected-plan audit. Payload retention via
`matmul_operands="reuse"` is independent; these comparisons use `"staged"`.

## Bounded temporary-buffering knob (2026-10-08)

`TrainiumTuning(temporary_buffer_depth=N)`, exposed as
`--temporary-buffer-depth N` in `trainium_generate.py` and
`trainium_advanced.py`, controls strict SBUF placement. N is a positive integer;
1 preserves the existing allocation exactly. It is independent of shared
software-tile `max_buffer_depth` and retained-weight payload reuse. N above 1
requires strict ISA realization and cannot be combined with native allocation.

For N above 1, collect SBUF values by their aligned bytes per partition. In each
size class, the existing shared allocator determines the minimum number of
slots covering inclusive source-order lifetimes. Reserve at most N times that
slot count, capped by the number of values, and assign values in source order
to the least recently occupied legal slot. A live value cannot be overwritten.
The existing physical-reuse analysis then adds completion edges between owners.
PSUM placement and selected ISA operands/order are unchanged. The whole plan
must fit the physical 28 MiB SBUF; overflow is rejected rather than silently
reducing N. The count is a pool multiplier, not N total buffers or a guarantee
of N panels running concurrently.

For fixed size classes and peak live-value counts, this storage bound is
independent of loop length. Tests compare 16 and 160 iterations, verify every
physical reuse dependency and reject capacity overflow. Settings are saved in
compiler policy and restored at conversion; the manifest records depth, actual
reserved bytes and allocation strategy. This first version applies to the
already selected logical program. Joint software-tile/pool-depth search remains
future work; compact search estimates do not yet score this depth.

Fresh FP32 Trainium2 device measurements (microseconds):

| Kernel | Default 1 | Depth 2 | Depth 4 | Depth 8 | Depth 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEMM128 | 20 | 17 | 17 | 17 | unmeasured |
| GEMM256 | 79 | 42 | 35 | 31 | unmeasured |
| GEMM512 | 169 | 76 | 62 | 49 | 46 |

The default timings are authenticated previous results: regenerated model.txt
and emitted NKI are byte-identical. Every nondefault row is fresh hardware
validation with three p50 repeats (100 iterations after 10 warmups), using the
same inputs, software schedule, staged operands and PSUM placements. No CPU
simulator was used. GEMM512 reserves 1, 1.5, 2.5, 3.5 and 5.5 MiB respectively.
Fully distinct storage took 44 microseconds at 11.5625 MiB; native allocation
was 42 microseconds with additional instruction folding.

The model predicts GEMM512 137.325, 67.114, 46.614, 44.127 and 44.164 microseconds.
It captures the major trend but underestimates intermediate settings and ranks
8 ahead of 16 by only 0.038 microseconds; that selection would be 6.5% slower
on hardware within this measured candidate set. No timing coefficients were
changed to fit these kernels. Depth 4's native trace retains all 128 MATMUL,
160 COPY, 72 MEMSET and 33 DMA commands. Its first weight loads can overlap,
but physical reuse elsewhere still constrains the whole program.

Evidence and the reproducible `trainium_buffering_report.py` output are in
[`results/trainium/bounded-buffering-2026-10-08`](../results/trainium/bounded-buffering-2026-10-08).
Validation: 118 focused tests, 33 shared tests plus 23 subtests; all three
default GEMM model/NKI files unchanged. The six full application kernels and
the full default-model regression suite were not rerun for this target-local
change. Depth 1 remains the default.

## LayerNorm walkthrough

For FP32 4096×8192, hardware lowering selects centered LayerNorm: mean,
subtraction, centered square, variance, reciprocal square root, normalization,
gamma and beta. RMSNorm has a separate recipe without centering. Shared search
selects a 128×8192 row tile, repeated 32 times. Rows map to partitions and
features to the free axis, so layout round trips disappear.

Each tile has two input DMA calls and two stores. Gamma/beta load once, then
explicit ones tiles, TensorE matmuls and PSUM copies replicate them. Replicated
parameters stay live across data tiles. Total: 132 DMA calls and no layout
transposes. The converter binds the selected physical plan mechanically.

Full hardware correctness passed at atol=rtol=1e-3, maximum absolute error
0.00049305, and median **1812 µs** with the revised PSUM bank policy (1871 µs before
that change). The unchanged matching prior kernel measured
**1728 µs**, so this implementation remains 4.9% slower. The prior uses a different
variance formula; both passed the same input reference and tolerance.
Profile/model HBM traffic agrees exactly at 268,500,992 bytes, with no spills.
The final trace is 1.814 ms, with 1.692 ms of VectorE activity and no spills.

## Maxpool walkthrough

The 3×3 stride-one pool remains one region through graph preparation. Shared
spatial bufferization computes the halo. Selection uses one partition per output
row and three vertically shifted strips on the free axis: three halo loads,
four local maxima and one store per tile. Two maxima combine the three
vertical strips; two combine horizontal views of that intermediate. This
separable expansion is shared by costing and typed instruction lowering.
Padding uses explicit negative infinity plus bounded valid loads. The
single-channel NHWC/NCHW boundary is a shape-only view.

An initial full schedule selected 89×178 output tiles, repeated 1058 times.
Its cost graph incorrectly allowed successive tiles to overlap without retiring
the reused arena. Adding the completion edge changed production selection to
**89×4094**, repeated 46 times. DMA calls fell from 4232 to 184 and maximum calls
from 8464 to 368. This was a dependency fix, not a prescribed preferred tile.

The initial halo kernel passed exactly at **2027 µs**. The separable expansion
halves maximum calls again, from 368 to 184, and passes exactly at **1016 µs**
in all three timing repeats. The unchanged matching prior is **1358 µs**, a
**1.34× speedup**. The padded ragged 33×197 case passes at 18 µs. Traffic is
268,271,632 bytes versus the old materialized graph's 2,279,490,080 bytes. Three
vertically replicated loads still exceed an ideal single input pass. The profile
has no spills in the final separable kernel. Its 1.017 ms trace includes
0.986 ms of DMA activity overlapping 0.814 ms of VectorE activity. The selected-DAG model currently predicts
1321 µs, 30.0% high. The candidate template is still more conservative about
iteration retirement than the final two-input-buffer realization; this is an
explicit remaining abstraction gap, not a fitted kernel penalty.

## Models and evidence

Search retains compact startup/neighboring-steady/tail analysis with exact
fallback. Candidate templates preserve parameter recurrence and count retained
DMA bytes once. Matrix/movement templates include explicit PSUM clears.
Final `analyze_selected` replays the selected ISA DAG, including physical-reuse
edges, with the same primitive laws. It independently sums DMA bytes and rejects
disagreement with transfer accounting. Template and selected-DAG predictions are
both retained; differences expose abstraction gaps rather than being hidden in
an application timing constant. Unknown completion times remain explicit.

Eight isolated VectorE probes measured FP32 binary/scalar multiplication at
widths 128, 512, 2048 and 8192, with 128 partitions and 16 dependent instructions.
The first three widths fit a service scale of 0.5 relative to the previous
documented-cycle estimate. Completion intercepts are 158.667 ns (binary) and
160.333 ns (scalar). Held-out width 8192 differs by 0.5 ns and 0 ns respectively.
This is primitive calibration, not application fitting. Smaller/strided cases
remain extrapolations. See
[calibration records](../results/trainium/vector-characterization/calibration.json).

An additional isolated ScalarE SBUF-copy chain measured 297, 617, 1897 and
7017 ns at the same four widths. Widths 128/512/2048 fit a 190.333 ns completion
intercept plus the existing 1.2 GHz service term. Held-out width 8192 matches
exactly. This fills the previously unknown SBUF-to-SBUF ScalarE copy timing;
[probe records and calibration](../results/trainium/scalar-copy-characterization/calibration.json)
retain the source, reference, NEFF and NTFF hashes. Application timing tables
are explicitly reanalyzed under these primitive laws without changing measured
kernels or refitting coefficients to them.


Hardware validation is the default at the user's request; the official CPU
simulator is opt-in with `--simulate`. Voyager's performance model still runs.
A pinned-SDK resume path (`--reuse-compiled`) authenticates source, model,
hardware, instructions, reference, compiler flags/version and NEFF hashes before
recovering the same argument ABI and rerunning device correctness. The runner automatically saves this manifest before device execution. A deliberately
tampered source digest is rejected before execution. Execution is serialized with `/tmp/voyager-trainium-device.lock`, on visible core
zero, with `--target=trn2 --lnc=1`. Every correctness check uses device output.
Timing is the median of three p50 samples, each after ten warmups and 100 device
iterations, excluding host invocation. The runner reuses one compiled NEFF for
the correctness check and all timing repeats through the pinned benchmark
execution API, avoiding benchmark recompilation. Small BMM→softmax (43 µs)
and BF16 GEMM (18 µs) passed this execution path, including workspace outputs. Large SwiGLU profiling needed RAM-backed temporary storage (`TMPDIR`) because
its compiler debug sections expanded beyond the free disk space. Both variants
passed device correctness before that profiling failure; failed attempts are
preserved beside the resumed runs. The report authenticates independent
source/model/hardware/plan/reference hashes and NEFF/NTFF evidence.

- [Per-case table](../results/trainium/selected-plan-comparison.md)
- [Detailed JSON](../results/trainium/selected-plan-comparison.json)
- [CSV table](../results/trainium/selected-plan-comparison.csv)
- [Focused test log](../results/trainium/focused-tests.log)
- [Full LayerNorm](../results/trainium/retirement-current-full/layernorm/result.json)
- [Corrected full maxpool](../results/trainium/separable-final-full/maxpool/result.json)
- [Matched prior measurements](../results/trainium/final-prior/)

The focused suite has 76 passing tests, including independent-reader retirement,
separable pooling, cached candidate equivalence and exact-final-evaluation
separation. Small hardware checks cover all six advanced kernels, FP32/BF16 matrices,
split-K at logical depths one and two, held-out ragged GEMM, padded ragged pool
and ragged LayerNorm. The final table contains **37 timed device correctness
passes**, all with exact HBM-byte and backend matmul-expansion agreement.
Thirty-five have complete current recorded dependency plans; two historical
comparators are explicitly marked as having incomplete original dependencies.
Every requested full kernel family was measured. The three unmeasured rows are
superseded generation/encoding attempts, retained for provenance.

## Measured fused-kernel regression

The fixed 512×512×1024 residual→RMSNorm→GEMM diagnostic remains about
13.0 ms (12.996 ms with the revised bank policy), versus 4.113 ms for the
matching prior. The earlier allocation measured 12.957 ms. Extra PSUM bank
rotation therefore did not resolve this full-case regression. GEMM→residual→
RMSNorm with the same diagnostic tile measured 31.549 ms versus 8.465 ms prior.
These are correctness passes and performance regressions.

Unrestricted production search improves these timings but remains slower than
the priors: GEMM→residual→RMSNorm measured **21.197 ms**, versus **8.466 ms**
for its matching prior; residual→RMSNorm→GEMM measured **9.140 ms**, versus
**4.114 ms**. Each prior was rerun on the exact production reference inputs.
The current primitive-calibrated selected-DAG predictions are 15.893 ms and
8.251 ms, respectively; these are model replays of unchanged measured kernels.

The authenticated residual→RMSNorm→GEMM trace has 193,007,616 HBM bytes,
exactly matching explicit transfers, and no spills. It contains 3456 transpose
matmuls, 2052 regular matmul commands, 7330 copies and 3588 clears. Median
transpose-matmul duration is 472 ns, but median spacing is 3863 ns. Scalar
copies have median duration 266 ns and reported event wait 2520 ns. These waits
overlap and cannot be summed as an extra kernel constant. They show repeated
producer/consumer and allocation dependencies, not a one-time startup effect.
The dependency-ready analytical scheduler still underpredicts this trace.
See the [instruction timing audit](../results/trainium/diagnostic-full/add_rmsnorm_matmul/dependency_profile_audit.json).

Full production SwiGLU passes device correctness at **52.925 ms** (p50 repeats
52.933/52.925/52.874 ms), versus **4.082 ms** for the matching prior. Its
selected-DAG prediction is 38.878 ms, 26.5% low; the preallocation template was
8.632 ms. The fixed 512×512×1024 SwiGLU diagnostic also passes correctness,
with p50 samples 69.366/69.366/69.475 ms and a **69.366 ms** median, versus
4.082 ms for its independently matched prior. This large template/physical-plan difference is material to search
ranking and remains an unresolved compiler-contract issue.

The SwiGLU profile has exactly **666,959,872 HBM bytes** and no spills. It
contains 14,784 transpose matmuls, 9,216 regular matmul commands, 21,312 ScalarE
copies, 9,312 VectorE copies and 15,233 clears. Median transpose duration is
474 ns, but median start spacing is 3,873 ns. ScalarE copies have 265 ns median
duration and 2,428 ns median event wait; clears have 165 ns duration and 2,010 ns
wait. These overlapping waits identify repeated execution overhead; they do
not establish a unique cause or justify adding a fitted kernel constant.
[Authenticated profile](../results/trainium/cached-search/swiglu/profile_audit.json)
retains the instruction groups and original NEFF/NTFF hashes.

Full **BMM→softmax** (B16, M4096, N4096, K64 padded to 128) passes with
p50 samples **108.975/108.930/108.930 ms**, median **108.930 ms**, versus
**14.591 ms** for the matching prior. Shared search selected M4096/N128/K128.
The current selected-DAG model predicts **92.673 ms**, 14.9% low, versus a
58.550 ms template estimate. An additional
[binary correctness replay](../results/trainium/grouped-bmm/bmm_softmax/binary_correctness.json)
checks the exact saved benchmark NEFF, with maximum absolute error 7.87e-6 at
atol=rtol=1e-3. Profile and model agree on **3,389,063,168 HBM bytes**, with
no spills. This is a full hardware result and a performance regression.

FP32 matrix profiles confirm two regular backend matmul commands per selected
matmul; BF16 confirms one. Both have one transpose command per explicit
transpose realization. For example, GEMM512 has 32 regular source matmuls and
64 transposes in FP32, producing 64 regular and 64 transpose commands. The
BF16 search selects a different tile: 16 regular source matmuls and 48
transposes produce 16 regular and 48 transpose commands. The model already
accounts for the FP32 expansion; multiplying its service by two again would
double-count it.

## Remaining limitations

- The formal converter boundary is implemented, but exact temporary allocation
  is instantiated after shared bufferization, not for every candidate. Compact
  search templates and the selected physical DAG can still differ in concurrency.
  The final audit exposes this; the full early-realization/search contract is not
  yet established for every operation.
- Search and eligible emitted loops are compact, but the selected JSON retains
  expanded instructions for allocation and independent validation. Non-affine
  or locally changing bodies remain expanded. Full unrestricted fused search and
  backend compilation can still be expensive. Production BMM-softmax source
  shrank from 20.65 MB to 3.06 MB with verified encoding, while full SwiGLU
  remains about 8.43 MB for the fixed diagnostic (7.2 MB for production search).
  The pinned SDK documents that sequential loops may unroll in the backend;
  current full BMM and SwiGLU compiles still expand to over 97,000 backend
  instructions and spend many minutes reducing alias dependencies. Compact
  source therefore does not guarantee compact native compilation. Encoding does not alter
  layout, allocation, schedules or dependencies to manufacture a repeat.
- The optimized ISA subset is pinned to Trainium2 and the installed SDK.
  Reductions are FP32 on the full final axis; the optimized pool route is
  single-channel and unit-stride. Unsupported language-only boundary paths fail
  explicitly. Direct allocation does not certify exact backend engine scheduling.
- Primitive completion/queue behavior is incompletely characterized. Source
  calls, backend command expansion, model service and profiler activity are
  distinct quantities. Good traffic agreement is not proof of timing agreement.
- Consult the evidence table for the status of full fused and BMM-softmax runs.
  Fixed-tile diagnostics are distinguished from unrestricted production search.
  ImageNet and Llama regressions were explicitly waived.

## 8 October: general row regions and maxpool coverage

`BufferizationOptions(row_regions=True)` / `trainium_advanced.py --row-regions`
enables a shared composition pass before per-operation HBM buffers are created.
The pass is in `codegen/transform/bufferize/row_regions.py`. It derives an
independent row axis from pointwise broadcasting roles, normalization reduction
axes, and matrix operand roles. It does not recognize benchmark names or encode
separate residual/RMSNorm/GEMM orderings. All six permutations pass compilation
and hardware correctness on M=192, K=128, N=256 (two 96-row tiles). A multiplication
in place of residual addition also passes the compiler test.

The shared exact-divisor enumerator proposes row tiles; the target's
`row_region_candidate` hook checks partition-rounded storage, invariant operands,
named reduction scratch and temporary reserve, and scores existing instruction
templates. The selected row tile and rejected candidates are recorded in
`hardware.json:row_regions`. The existing `build_pipelined_buffers` scheduler
loads invariant weights once, streams row-varying inputs, allocates intermediate
SBUF destinations, and stores only the final result. The initial implementation
uses one slot and keeps K and N whole, so a reduction never sees a partial axis.

Early ISA planning propagates compatible row layouts across local same-shape
pointwise edges and explicitly converts matrix result panels to row layout or
row tiles to TensorE's K-partition layout. `--matmul-operands reuse` retains each
converted activation panel across output-column blocks. These are actual
payload reuses, not merely reuse of an address. The formal converter remains
policy-free. Strict mode still binds physical addresses in Voyager.

Full FP32 hardware results (median of three p50 repeats, one Trainium2 core):

| Workload | Previous Voyager | Row region | Prior work | HBM before → after |
| --- | ---: | ---: | ---: | ---: |
| Residual → RMSNorm → GEMM | 8.318 ms | 4.042 ms | 4.114 ms | 167,841,792 → 75,567,104 B |
| GEMM → residual → RMSNorm | 17.647 ms | 15.868 ms | 8.466 ms | 302,063,616 → 117,514,240 B |

Both have one semantic HBM output, no HBM intermediate workspace outputs, exact
agreement between selected and profiled HBM byte counts, and zero profiled
spills. Before retaining converted activation panels, the first region took
13.861 ms: reduced HBM traffic alone did not produce a faster kernel.

Search/selected/hardware estimates are respectively 2.677/3.516/4.042 ms and
5.053/9.515/15.868 ms. Search uses serial boundary/stage templates and does not
include the final physical-reuse edges. Selected analysis still reports missing
reduce/activation completion laws. These estimates remain incomplete; no
hardware timing coefficient was fitted to the new measurements.

The prior GEMM-first algorithm also uses 128 rows, but its TensorE operand
orientation produces 128×512 output tiles directly in the row layout. Voyager's
fixed opposite orientation produces 128×128 panels in this region: 8,192 logical
GEMMs versus 2,048, plus local layout conversion. Invariant weight payload stays
local, but its panel transposes repeat per row tile. Alternate matrix orientation,
retaining converted invariant weights, and physical-reuse-aware joint scoring
remain open; their individual timing effects have not been isolated here.

Initial scope: FP32, one 2-D GEMM per discovered region, same-shape pointwise
tensor operands and full-feature normalization, invariant matrix fitting SBUF,
exact-divisor row tiles up to 128. Padding/view boundaries and existing fused
submodules may break regions. Larger invariant matrices, multi-GEMM regions,
streaming feature reductions, softmax/attention, and choosing region versus
unfused execution are not implemented. Unsupported proposals preserve the
ordinary per-kernel path and record their rejection. The option defaults off.

Maxpool has a separate proven enumeration gap. Output height 4,094 permits
89-row exact tiles but not 128-row tiles. A fixed-template probe retains the
existing four-max expansion, double-buffered halo, and 16 MiB explicit arena:

| Row tile | Hardware p50 | DMA commands | HBM bytes |
| --- | ---: | ---: | ---: |
| 89 (control) | 1.016 ms | 184 | 268,271,632 |
| 128, final tile 126 | 0.729 ms | 128 | 268,271,632 |

Both are exactly correct. The current model ranks the excluded 128-row schedule
ahead of 89; this missed speedup is a shared tile representation/enumeration
restriction, not a bad model ranking. Production maxpool remains unchanged.
Ragged pool enumeration/bufferization and rolling halo reuse remain open.

For the existing maxpool, the serial search template is 2.045 ms before 9.403 µs
fixed overhead. The selected ISA model is 1.321 ms, compiled replay 1.074 ms,
hardware 1.016 ms. Omitting physical-reuse edges diagnostically gives 1.077 ms;
changing only the template recurrence distance to two gives 1.175 ms. These
localize overly conservative dependencies but are not legal new allocation
plans or a completed replacement model. Safe refinement requires byte-range
read/write lifetime evidence.

Evidence and commands:

- `results/trainium/row-regions-2026-10-08/report.json`: authenticated full and
  all-order measurements, final source/ISA/reference equivalence, constraints.
- `results/trainium/maxpool-gap-2026-10-08/analysis.json`: original NEFF/NTFF
  authentication, counterfactual model graphs, measured probes and static replay.
- `scripts/trainium_row_region_probe.py`, `trainium_pool_tile_probe.py`,
  `trainium_maxpool_gap.py`, and `trainium_row_region_report.py` reproduce probes,
  analysis and the update in `../trainium-kernel-analysis-2026-10-07.html`.
- `regression-final.log`: 162 focused/shared tests and 23 subtests passed. Default
  GEMM128/512 `model.txt` and emitted NKI match the preserved comparator exactly.
  No CPU simulator or waived ImageNet/Llama regression was run.

## TensorE operand orientation search (2026-10-08)

`TrainiumTuning.matmul_orientation` and the generation CLI option
`--matmul-orientation auto|weights|activations` expose both legal TensorE
orientations. Auto is the Trainium default. `weights` preserves the earlier
weight-stationary expansion; `activations` uses activation-stationary panels.
This changes local ISA panels inside the shared software tile, not the HBM
software tile or the shared bufferization schedule. Convolution retains its
existing weight-stationary lowering.

| Mode | Stationary operand | Moving operand | Logical panel limits | Physical result |
| --- | --- | --- | --- | --- |
| weights | W, K×N | X transpose, K×M | M≤512, N≤128, K≤128 | N×M |
| activations | X transpose, K×M | W, K×N | M≤128, N≤512, K≤128 | M×N |

`trainium/orientation.py` prices both choices with existing primitive timing
laws. It includes row-to-matrix activation conversion, the operand reuse policy,
weight preparation, wider moving-panel assembly, PSUM eviction, and the output
conversion required by the consumer. Generic output storage partitions N;
row reductions partition M. The selected orientation is recorded in ordinary
matrix candidate diagnostics (`matrix_choice`) or the row-region candidate
(`matrix_choices`). `InstructionPlanner` consumes and checks those bindings.
A mismatched binding fails; the formal NKI emitter makes no orientation choice.
No new measured constant or fitted rate was introduced.

Storage remains conservative: the existing matrix certificate includes input,
weight and output slots, converted-weight storage under the reuse policy,
a default 4 MiB temporary allowance, and three PSUM banks. Row-region capacity
checks sum boundary/local buffers, reduction scratch and temporary allowance.
The final typed ISA allocator checks actual SBUF/PSUM placement before emission.
This does not yet jointly search temporary-buffer depth or weight-buffer layout.
Orientation ranking uses the local conversion/compute dependency graph; outer
transfer overlap and final physical reuse can change its realized cost.

The controlled full-size FP32 experiment is under
`results/trainium/orientation-2026-10-08/`. Both full candidates use shared row
regions, `matmul_operands=reuse`, strict addresses, and temporary depth 1.
Activation-stationary reduced logical matmul calls from 8192 to 2048 for
GEMM→residual→RMSNorm and from 4096 to 1024 for the reverse order, but was slower
on hardware: 16.560 ms and 9.715 ms respectively. Auto retains weight-stationary;
the preserved baseline is 15.868 ms and 4.042 ms respectively. See `report.json`
for the fresh automatic-mode measurements, three timing repetitions, hashes,
search/selected predictions and comparison with prior work.

This result does **not** establish that the prior kernel's schedule is now
fully represented. The row-region invariant weight still uses the existing
N-partitioned SBUF layout. The alternate orientation transposes its 128-column
fragments and assembles wider K-partitioned moving panels within every row tile.
The prior kernel's direct K-partitioned loads and reusable converted-weight
layout are still missing choices. The model ranks these two current realizations
correctly; absolute predictions retain the earlier physical-reuse and primitive
completion gaps. Both orientations have identical boundary HBM traffic in each
full comparison (75,567,104 or 117,514,240 bytes including the identity constant).

Validation: 167 tests and 23 subtests pass in `regression-final.log`. Hardware
checks cover both small application kernels, both full kernels in both modes,
all six operation orders at M=192/N=256/K=128 using activation-stationary,
a rectangular FP32 GEMM at M=256/N=512/K=256, and BF16 whole-K GEMM at
M=128/N=256/K=256. No CPU simulator was used. A separate BF16 external-K-split
probe failed the same shared-bufferized CPU reference check for both orientations
(32/32768 elements); this existing partial-sum rounding limitation is not claimed
as validated by the new mode. Network regressions remain outside the user-agreed
scope.

Reproduce a forced full candidate with:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ../.venv-compiler/bin/python scripts/trainium_advanced.py \
  --output results/trainium/orientation-2026-10-08/full-activations \
  --cases matmul_add_rmsnorm add_rmsnorm_matmul --row-regions \
  --matmul-operands reuse --matmul-orientation activations
NEURON_RT_VISIBLE_CORES=0 ../../.venv-nki/bin/python \
  scripts/trainium_run_hardware.py \
  --artifacts results/trainium/orientation-2026-10-08/full-activations \
  --save-compiler-artifacts
```

## Joint invariant-weight layout and orientation (2026-10-08 follow-up)

The preceding fixed-weight-layout restriction is now relaxed for eligible
row regions. `TrainiumTuning.matmul_weight_layout` / CLI
`--matmul-weight-layout auto|generic|k_partitioned` defaults to auto. The target
compares both physical representations for each shared row-tile candidate,
including their best allowed operand orientation. The generic representation
remains available as a fallback and as a forced diagnostic comparator.

The new representation maps logical W[k,n] to physical
`[k % P, (k // P) * N + n]`, where `P=min(128,K)`. A full invariant load issues
ordinary DMA rectangles with at most 128 K rows and the configured DMA column
limit. It populates the final K-partitioned SBUF buffer directly, replacing the
old buffer rather than retaining an additional converted copy. Both stationary
and moving TensorE operands can read checked panel views from it. The shared
region's one load/wait and invariant lifetime allow payload reuse across row
tiles; this is not merely a smaller allocation footprint.

The conservative boundary is explicit:

- Whole invariant 2-D FP32 weights in the existing row-region contract, with
  one matrix consumer inside the region. Conflicting uses retain the generic
  candidate; ambiguous saved consumer bindings are rejected rather than guessed.
- Each layout is charged its partition-rounded footprint. K blocking can use
  **more** space than N blocking. A test with a partial K block makes only the
  generic candidate fit and verifies that auto falls back to it.
- Selected bindings must match the matrix geometry and consumer row layouts.
  The planner binds storage before DMA and instruction expansion, validates
  full unpadded contiguous loads, and rejects views crossing a physical K block.
- Shared bufferization, invariant lifetime, generation changes on writes,
  and final physical SBUF/PSUM allocation checks remain in force. Strict mode
  continues to assign local addresses in Voyager.
- Oversized weight streaming, multiple-GEMM regions, arbitrary alias layouts,
  and fusion across padding boundaries are not added by this extension.

Boundary estimates now distinguish row-compatible/direct K loads from the
existing generic transpose-on-load/store path. The old row estimator had charged
all boundaries as direct rectangles. The new graph removes only weight
conversions that the selected physical representation actually avoids. It uses
existing DMA and primitive timing laws; no application-fitted rate was added.
The estimate still has conservative norm-layout charges and incomplete modeling
of physical reuse and primitive completion.

Authenticated results in `results/trainium/weight-layout-2026-10-08/report.json`:

| FP32 full kernel | Before: generic weights | Auto: K layout / weights stationary | K layout / activations stationary | Prior-work reference |
| --- | ---: | ---: | ---: | ---: |
| GEMM → residual → RMSNorm | 15.863 ms | 6.016 ms | **3.567 ms** | 8.466 ms |
| Residual → RMSNorm → GEMM | 4.042 ms | **2.456 ms** | 2.744 ms | 4.114 ms |

All candidates pass hardware correctness with identical input arrays and the
same strict allocation policy. Automatic-mode speedups are 2.64× and 1.65×.
The first row exposes a remaining **model ranking error**, not an absent schedule:
auto still prefers weight-stationary. Its search estimates are 4.698 vs 5.283 ms;
selected-ISA estimates are 4.234 vs 4.276 ms. The measured alternate is faster.
The selector was not modified to fit this application's winner. For the second
row the ranking is correct (search 2.681 vs 3.025 ms; selected ISA 2.059 vs
2.556 ms). These are explicitly separate prediction scopes.

The automatic programs reduce transposes from 9472 to 1024 and from 4992 to 768.
Full-kernel profiles confirm 117,514,240 and 75,567,104 HBM bytes respectively,
unchanged from before, and zero spill save/reload bytes. One full invariant
weight load feeds all row tiles. The alternate GEMM-first program has 512
transposes and 2048 logical GEMM calls, versus auto's 8192 calls.

Validation: 171 tests and 23 subtests pass. Hardware covers both full candidates
in both orientations, both small applications, and all six operation orders at
M=192/N=512/K=256 (two 96-row tiles, multiple K blocks). Ordinary GEMM128 and
GEMM512 `model.txt`, `instructions.json`, and `nki/program.py` remain byte-identical
to the preserved pre-change artifacts. A partial-dimension M=192/N=320/K=192
probe encounters existing padding boundaries and falls back; it is not claimed
as new-layout hardware coverage. No CPU simulator or network regressions ran.

For a controlled fast GEMM-first candidate, use the earlier generation command
with `--row-regions --matmul-operands reuse --matmul-weight-layout k_partitioned
--matmul-orientation activations`. Both options also accept `auto`; use
`--matmul-weight-layout generic` to retain the prior layout family.

## Operand-geometry timing experiment, 2026-10-08

The Trainium2 FP32 matmul model now distinguishes moving and stationary operand
free-axis strides and pipeline startup versus steady execution. The implementation
is in `timing.MatmulGeometryTiming`; coefficients live in the hardware-owned
`timing_trainium2.json`. `isa.matmul` returns effective issue, completion and
accumulator-forwarding timing without changing instruction expansion. Candidate
graphs infer strides from their layouts and materializations; selected-ISA analysis
resolves actual physical views. Compiled-static analysis also consumes operand
strides. None of these paths consumes application profile durations.

Seventeen matched direct-address hardware probes varied moving width and both
operand strides independently. Twelve probes define/check the model; five fresh
probes were withheld from parameter fitting. `trainium_geometry_calibration.py`
reproduces every coefficient from isolated measurements and the existing FP32
arithmetic lower bound. The five withheld launch-spacing errors are at most 6.50%.
All seventeen probes passed numerical checks. Evidence and source hashes are in
`results/trainium/matmul-geometry-2026-10-08/calibration.json`.

The characterized scope is FP32, K=N=128, moving width 128–512, and regular free
strides 1/2/4/8/16. The empirical feed factor is one for strides 1–2 and two for
strides 4–16. This is an effective feed regime, not a claim about undocumented
SBUF bank topology. In TensorE cycles:

```
moving_feed    = 4 * moving_width * moving_access_factor
stationary_feed = 4 * stationary_width * stationary_access_factor
issue          = max(moving_feed, stationary_feed) + 24
steady_result  = 432 + max(moving_feed, stationary_feed) + moving_feed / 2
forward        = issue
```

Convert cycles to ns using the target TensorE clock. Startup retains the earlier
isolated completion bound; a source-order TensorE interruption or freshly prepared
operand breaks the modeled stream. This is an explicit approximation, not a
finite-queue simulator. Compiled-static replay recognizes TensorE interruptions
but does not have the selected plan's producer-generation context. All physical
memory reuse and operand-read retirement still wait for completion; only a true
same-accumulator edge can use forwarding. BF16, short-K, unknown strides and
uncharacterized geometry keep the previous timing laws.

For unchanged, correctness-validated full programs (milliseconds):

| Kernel/orientation | Old selected-ISA prediction | New prediction | Hardware |
|---|---:|---:|---:|
| GEMM → residual → RMSNorm, weights | 4.234 | 6.057 | 6.016 |
| GEMM → residual → RMSNorm, activations | 4.276 | 3.684 | 3.567 |
| Residual → RMSNorm → GEMM, weights | 2.059 | 2.086 | 2.456 |
| Residual → RMSNorm → GEMM, activations | 2.556 | 2.179 | 2.744 |

Fresh search selects activations for GEMM-first and weights for RMSNorm-first.
The selected NKI sources and input arrays exactly match the previous measured
programs, so their hardware evidence is authenticated reuse. Compact search
predicts 4.690 and 2.691 ms respectively; these are distinct from the selected-ISA
predictions above. Correct orientation ranking does not establish accurate
whole-kernel timing: RMSNorm-first activation timing actually becomes less
accurate, and other non-matmul/overlap costs remain unresolved.

Twenty-four fixed historical programs were re-scored against unchanged measured
artifacts, including standalone FP32/BF16 and bounded-buffering variants. Fresh
old/new search and hardware runs under identical settings gave:

| Standalone case (M,N,K), temporary depth 2 | Old selection | New selection |
|---|---:|---:|
| GEMM128 (128,128,128) | 17 us | 17 us |
| GEMM256 (256,256,256) | 42 us | 37 us |
| GEMM512 (512,256,512) | 76 us | 82 us |

The GEMM512 regression is retained explicitly. Re-scoring its old tile
(128,256,512) and new tile (256,256,512) with the same new model gives compact
search costs 41.513/41.123 us, but selected physical ISA costs 63.176/71.860 us.
The selected-ISA audit ranks the pair correctly; compact search reverses it.
Candidate realization/overlap accounting is therefore still a blocker. Do not
claim the geometry correction makes general schedule ranking reliable or tune
another primitive rate to hide this disagreement.

The full evidence is in `results/trainium/matmul-geometry-2026-10-08/report.json`
and the parent `trainium-kernel-analysis-2026-10-07.html` geometry section. Focused
regression: 148 tests passed, followed by 19 affected-context checks after the
assembled-weight stride correction. Hardware validation used the pinned compiler,
one Trainium2 core, three benchmark repeats, and no CPU simulator. No emitter,
address-allocation policy, kernel-rule restriction, or hardware ISA was changed.

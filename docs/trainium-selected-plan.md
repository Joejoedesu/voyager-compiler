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

Every local value binds an explicit NKI allocation. A uint8 SBUF arena starts at
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

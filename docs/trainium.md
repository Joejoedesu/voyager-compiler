# Trainium integration on the shared compiler

For the 10-06 refactor, read [selected ISA plans](trainium-selected-plan.md).
The baseline description below is retained as historical context.

This is the ISA experiment checkout, copied independently from
`../voyager-trainium-10-05` on 2026-10-06. New compilation defaults to explicit
NKI ISA lowering and the ISA expansion model. Use `--no-isa` only for a baseline
comparison; stored artifacts always restore their recorded mode. Prior results
were copied with their original provenance and have not been rerun merely by
copying them. See [dependency timing plan](trainium-dependency-model.md).

This checkout starts from `voyager-base` commit
`0a446d7eaed864c7ffbaf8c12cdb2f9fa411371c` (the integrated Gemmini base).
The Trainium adapter was ported from `voyager-trainium` commit
`365bb65438307100565a43aead085fcd35b111a5`. The original checkouts are unchanged.

## Movement and dependency model, 2026-10-06

The ISA checkout declares operation implementations in hardware IR and uses the
shared search hook to evaluate repeated operation chains. The current model
separates DMA request issue, bandwidth service and completion; characterizes
primitive dependent completion; and includes explicit ragged pad/slice traffic.
Compiler-managed logical buffer depth receives no assumed physical overlap credit.
Read the [implementation and validation scope](trainium-dependency-model.md).
Final measurements are under `results/trainium/movement-final-corrected/`.
The older results below are historical evidence, not the current model's results.

## Execution contract

`CompilerContext` resolves hardware, performance-only policy and conversion
options once. The Trainium target uses shared Interstellar enumeration, matrix
traversal, buffer plans, bufferization, lifetimes and allocation. It supplies
its own capacity checks and analytical engine model. It does not instantiate
Voyager's runtime equations or use energy to select a mapping. Pointwise search
also honors the target's `speed_only` declaration; the default Voyager objective
retains its existing tolerance/traffic selection rule.

`execution.py` shares DMA panel geometry, TensorE panel geometry and storage
sizing between estimation and realization. `CandidateEvaluation` returns the
selected plan and a snapshot of its diagnostics. The report reads that selected
snapshot, never the cost calculator's last candidate. Conversion restores the
policy from `compilation.json` and rejects a conflicting hardware context.

The NKI converter expands the selected software tiles into instructions. It
preserves the selected transfers, buffer identities, generations and reduction
order. It does not select different software tiles. NKI owns final physical
allocation and instruction scheduling. The shared allocation is a capacity
certificate, not a promise that NKI uses the same physical addresses.

## Hardware facts and policy

`trainium-v3` denotes NeuronCore-v3, used by Trainium2. It does not mean
Trainium3. The validation host uses one physical core: `--target=trn2 --lnc=1`.
The model uses 28 MiB SBUF over 128 partitions, 2 MiB PSUM over eight banks,
TensorE at 2.4 GHz, VectorE at 0.96 GHz, and 368 GB/s aggregate DMA service.
The FP32 compute normalization is 19.6608 TFLOP/s, counting a MAC as two FLOPs.
Hardware provenance is recorded in `hardware.py`.

Policy is separate: the default SBUF temporary reserve is 4 MiB; buffer depth
candidates are one and two; DMA transpose is optional. Physical SBUF capacity
is not reduced to encode the reserve. SBUF allocations reserve all partitions
and round each partition's pitch to 16 bytes. Matrix candidates reserve three
whole PSUM banks for the current accumulator and staged transpose results.
This is a conservative live-capacity check, not a bank-conflict or spill model.

The old blanket GEMM software-tile restriction
`M * ceil(N / 128) <= 4096` is removed. Instruction limits remain
`M <= 512, N <= 128, K <= 128`; larger software tiles are subdivided by the
shared panel helper. Convolution's current converter limits remain explicit
software restrictions rather than altered hardware sizes.

One-slot changing operands require retiring the previous async compute/store
before submitting work that overwrites the slot. The shared async builder now
orders this correctly. Retained one-slot operands do not force serialization.

## Analytical timing and its limits

DMA payload occupancy and cross-engine completion latency are distinct. A DMA
engine owns eight consecutive partitions. Transfer geometry determines the
busiest engine's bytes; reads and writes share the aggregate service budget.
The documented 1300 ns completion delay is no longer charged as serialized
engine service for every panel. The current approximation charges exposed
first-load/final-store completion and models steady-state overlap analytically.
It does not establish command issue rate or prove all intervening delays hide.

An analytical event schedule accounts for dependencies between gathers,
transposes, matmuls and accumulator eviction on TensorE and VectorE. These
are performance-model events, not invented async ISA operations. Actual data
movement and transposes are emitted as NKI operations. Engine service sums,
critical-path duration, useful-work/peak utilization, and profiler active time
are separately named quantities.

No fitted shape-specific coefficients or measured-latency lookup table guide
search. Unknown instruction issue/setup, ScalarE/control work, descriptor
throughput, finite queues, bank/port contention, compiler rewrites and spills
remain limitations. In particular, the model currently underestimates absolute
latency: correcting an unjustified DMA startup charge does not supply the
missing engine overhead. Do not treat reported utilization as calibrated.

The current scope is per-kernel execution. PSUM stays resident across instruction
K panels within one software tile. External software split-K still uses shared
SBUF partial results. Cross-software-tile PSUM retention needs an explicit
accumulator lifetime/reduction contract and matching lowering; a larger capacity
setting alone cannot enable it. Cross-operator layout preservation and graph
transpose rewrites are future extensions. DMA transpose and explicit instruction
selection are lowering choices, not PyTorch semantic rewrites.

## Reproduction

From `/home/ubuntu/ML/AGEN-voyager`:

```sh
source activate-compiler.sh voyager-trainium-10-05
cd voyager-trainium-10-05
python scripts/trainium_generate.py --output results/trainium/reproduction
NEURON_RT_VISIBLE_CORES=0 /home/ubuntu/ML/.venv-nki/bin/python \
  scripts/trainium_run_hardware.py --artifacts results/trainium/reproduction
/home/ubuntu/ML/.venv-nki/bin/python scripts/trainium_report.py \
  --artifacts results/trainium/reproduction --output results/trainium/report
```

Generation verifies the bufferized graph against PyTorch, checks matrix DMA and
matmul panel counts for aligned cases, and reconverts byte-identically using the
persisted context. The hardware runner independently executes CPU simulation and
the actual device program, retains failures, NEFF and NTFF, and measures three
sets of 100 executions after ten warmups. Device timings exclude host invocation.
The report checks source/collateral hashes and preserves profile artifact hashes.

`--tile M N K` is a diagnostic constraint applied to shared search, not a
production preset or separate scheduler. `--buffer-depth 1`, `--dma-transpose`,
`--legacy-isa` (matmul selection), `--dtype bfloat16`, and `--shape M N K` support controlled probes.
BF16 references store exactly rounded inputs as FP32 arrays, then restore BF16
before execution; tolerances are recorded explicitly in `generation.json`.
Ragged kernels may contain graph-boundary work outside the matrix estimate;
the report retains this scope difference.

Focused compiler validation:

```sh
python -m pytest -q test/test_trainium.py test/test_hardware_config.py \
  test/test_compilation.py test/test_extensibility.py test/test_gemmini.py \
  test/test_mapping_policy.py test/test_gemmini_scheduling.py \
  test/test_gemmini_dma.py test/test_interstellar_selection.py
```

ImageNet and Llama regressions are excluded at the user's request.

## Sources

- [Trainium2 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/architecture/trainium2_arch.html)
- [DMA bandwidth and completion delay](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/nki/deep-dives/nki-dma-bandwidth-guide.html)
- [TensorE matmul timing](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.26.1/nki/api/generated/nki.isa.nc_matmul.html)
- [Profiler metric definitions](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.32.0/tools/neuron-explorer/overview-summary-page.html)

## SDK transpose correctness

On the installed neuronx-cc 2.22.12471, explicitly selecting
`nisa.nc_transpose(..., engine=nisa.tensor_engine)` produced stale output tiles
when SBUF slots were overwritten and reused. CPU simulation passed. Three
controlled substitutions on the same generated schedule isolated the path:
replacing only matmul or tensor_copy did not fix it; replacing transpose did.
The converter therefore uses `nl.transpose` for staged transposes, including
when matmul uses explicit ISA. The exact SDK root cause remains unproven.
Failed probes and their outputs remain under `results/trainium/candidate-*`;
corrected runs use distinct directories. No hardware capacity is changed to
hide this realization failure.

A broader split-K/single-slot sweep subsequently reproduced stale results even
with generic matmul and transpose. Fully overwriting a logical SBUF slot now
creates a fresh NKI tensor definition; references to the logical slot resolve
to its latest definition. The emitted copies, arithmetic and logical slot counts
are unchanged. This resolves the failing repeated-slot cases on device. It does
not force a physical address: NKI may coalesce, reorder or spill these definitions.
Consequently buffer-depth timing remains an analytical policy estimate, not a
verified constraint on the SDK's final overlap or physical live-memory peak.
Both the shared semaphore fix and correct tensor definitions are required.

The native-transpose isolation can be reproduced using the retained failing
program (the final converter no longer emits that program):

```sh
NEURON_RT_VISIBLE_CORES=0 /home/ubuntu/ML/.venv-nki/bin/python \
  scripts/trainium_probe_transpose.py \
  results/trainium/candidate-128-256-512/gemm512
```

## Final validation, 2026-10-06

79 shared/Gemmini tests and 12 Trainium tests pass (plus 23 subtests in the shared suite). Dependency checking, formatting and `git diff --check` pass. All 13 final generated programs pass independent PyTorch-reference checks in CPU simulation and on Trainium2. Generation also verifies byte-identical collateral reconversion.

The final artifacts and hash-authenticated profile comparisons are in `results/trainium/final-report/report.json` and `measurements.csv`. These are device-kernel measurements, not application throughput.

| Variant | Case | Model µs | Device p50 µs |
| --- | --- | ---: | ---: |
| final-ssa | add_relu | — | 15 |
| final-ssa | gemm128 | 4.36 | 16 |
| final-ssa | gemm256 | 10.58 | 24 |
| final-ssa | gemm512 | 24.63 | 43 |
| final-ssa | split_k | 5.53 | 19 |
| ssa-old | gemm512 | 29.41 | 46 |
| ssa-small | gemm512 | 41.11 | 51 |
| ssa-single | gemm512 | 26.06 | 40 |
| ssa-bfloat16 | gemm128 | 3.93 | 15 |
| ssa-ragged | gemm_custom | 8.40 | 66 |
| ssa-dma | gemm128 | 3.99 | 20 |
| ssa-dma | gemm512 | 20.13 | 37 |
| ssa-large-tile | gemm_custom | 98.63 | 259 |

For FP32 M=512, N=256, K=512, the previous backend measured 46 µs. The new default selects a 512×128×512 tile and measures 43 µs. The same tile with one buffer measures 40 µs; the smaller 128×128×128 tile measures 51 µs. The model selects the right larger tile but misranks buffer depth: 7.5% slower than the best measured candidate under the default instruction-mode policy. No measured winner is hardcoded into production selection.

Optional DMA transpose selects a 256×128×512 software tile and measures 37 µs for that larger workload, but slows the 128×128×128 workload from 16 to 20 µs. It remains opt-in. The analytical model currently predicts a small-case benefit that hardware does not deliver; automatic joint instruction-mode selection is not justified by these results.

The forced 1024×1024×256 software tile exceeds the retired M*ceil(N/128)<=4096 cap and executes correctly. This proves representability and legality, not optimality: it measures 259 µs against a 98.63 µs matrix estimate. The trace reports 80,664,576 HBM bytes versus 6,291,456 modeled bytes (12.82×). The subsequent instruction audit below confirms compiler spill/reload traffic and isolates result assembly as its cause. At the configured 368 GB/s, that actual traffic alone requires about 219 µs. Logical capacity checks therefore cannot certify that NKI emits a spill-free program.

Absolute latency and utilization remain unreliable. For the default larger GEMM, the estimate is 24.63 µs versus 43 µs measured. Matrix estimates omit the separate boundary operations in the ragged case (three pad/slice operations); its 8.40 µs matrix estimate must not be interpreted as a whole-program prediction against its 66 µs execution. Pointwise search has a performance score but does not yet emit the matrix-style estimate report.

Remaining priorities are compiled instruction/descriptor issue, short-instruction setup, actual SDK overlap and live-storage behavior, strided on-chip copy service, and bank/port contention. Full bank placement, cross-software-tile PSUM retention and cross-operator layout rewrites are not implemented. These require matching execution contracts and evidence, rather than a timing scale factor or changed physical memory capacity.

## Instruction and traffic audit, 2026-10-06

Evidence is retained under `results/trainium/gap-audit/analysis.json`, with full
profile exports and a reproducible diagnostic source transformation. Original
profiles warned about absent dynamic-DMA metadata. Re-running the unchanged
large kernel with `NEURON_RT_ENABLE_DGE_NOTIFICATIONS=1` removes that warning
and reproduces the same byte counts and a 259 us median benchmark latency.

The 80,664,576 bytes decompose exactly into 2,097,152 input bytes, 65,536
identity-constant bytes, 4,194,304 output bytes, 50,331,648 spill writes
(12 stores of 4 MiB), and 23,975,936 scratch reload bytes (46 descriptors).
Instructions explicitly carry the SPILL label; dynamic DMA variables include
SpillSave names. The zero DRAM Spill allocation summary field therefore did
not establish absence of compiler spills.

A controlled diagnostic removes assembly of the interleaved 128x8192 SBUF
result, retaining its 16 independent 128x512 result panels and reading those
panels in the same output-store order. Input loads, source matmuls and HBM
output panel operations are unchanged. Both CPU simulation and real hardware
pass the independent reference. All spill DMA disappears: traffic falls to
6,356,992 bytes (logical I/O plus the identity constant), and all three
benchmark medians are 108 us. This isolates a converter temporary/lifetime
problem, rather than an intrinsic inability to execute the large software tile.
The diagnostic is not yet integrated into the production converter. Its
inherited 98.63 us baseline estimate is not a recomputed variant prediction.

The FP32 128-cube produces two regular MATMULs and four transpose MATMULs,
each with a matching LDWEIGHTS instruction. The model prices one source
matmul with an FP32 throughput factor and four transpose operations, but
does not model the expanded instruction issue/load pipeline. Five copies
execute on ScalarE; the model attributes copy work to VectorE. In the default
larger GEMM, 40 ScalarE copies are absent from its engine-resource model.
Thus source matmul counts and compiled MATMUL counts are different metrics,
and the FP32 factor must not be applied twice when expansion is introduced.

Conversely, the small 128-cube software tile for the 512x256x512 problem
models 4,718,592 bytes but compiles to 2,162,688 bytes. Its 24 input load
descriptors each read a unique panel once, showing reuse beyond the logical
schedule counted by the analytical model. The default and single-buffer
candidates have identical input/output traffic and matmul counts but different
engine timelines; software buffer depth alone does not establish SDK overlap.

Required fixes are a shared lowering/storage contract for result panels and
load reuse, complete mode-aware TensorE/ScalarE/VectorE expansion costs,
payload-size-dependent DMA issue/completion modeling, and a dependency model
that reflects actual physical buffers and SDK scheduling. Peak-bandwidth or
latency multipliers cannot repair these mismatches.

## Explicit NKI ISA path, 2026-10-06

The new path is the default in `scripts/trainium_generate.py` and
`TrainiumTuning()`. `--isa` remains accepted; `--no-isa` selects the language
baseline for comparison. The ISA contract is pinned to Trainium2 and neuronx-cc 2.22.12471.
It still uses the shared Interstellar search and bufferized program. No new
production scheduler, measured-winner tile preset, or shape-fitted timing
coefficient was introduced.

The path emits `nisa.dma_copy`, native TensorE `nisa.nc_transpose`,
`nisa.nc_matmul`, and explicit-engine `nisa.tensor_copy`. Elementwise add and
activation use ISA APIs. Independent GEMM result panels are bound to logical
output slots and consumed directly by output DMA or split-K arithmetic;
there is no interleaved full-result assembly. ScalarE handles transpose/result
eviction by default, while gathers use VectorE. `--copy-policy balanced`
is an optional alternative with VectorE compute/output eviction and ScalarE
input packing. Its policy is persisted and restored with the collaterals.

On this installed SDK, explicit SBUF destinations alone were insufficient:
Neuron's middle-end transformations could produce an invalid PSUM operand
for DMA. The ISA source therefore uses
`nki.compiler.skip_middle_end_transformations` above `nki.jit`. Native
transpose and explicit copies pass hardware correctness with this path. The
legacy route continues to use its previously validated generic transpose.
Failed intermediate probes are retained under `isa-probe*`; they are not final
validation results. Back-end allocation, scheduling and DMA deduplication
still occur, so this decorator is not a raw-ISA scheduling guarantee.

`isa.py` defines shared dtype/mode expansion rules. Search and conversion
agree on regular MATMUL, transpose MATMUL and LDWEIGHTS counts. Documented
steady-state TensorE costs already include the FP32 pipeline factor; expansion
counts are not multiplied into that cost a second time. The model now counts
the native transpose's shared 16 KiB identity constant and separately tracks
ScalarE and VectorE service. Result-panel aliasing removes the modeled
whole-result placement copy.

Reproduction:

```sh
PYTHONPATH=src /home/ubuntu/ML/AGEN-voyager/.venv-compiler/bin/python \
  scripts/trainium_generate.py --isa --output results/trainium/isa-reproduction
NEURON_RT_VISIBLE_CORES=0 /home/ubuntu/ML/.venv-nki/bin/python \
  scripts/trainium_run_hardware.py --artifacts results/trainium/isa-reproduction
/home/ubuntu/ML/.venv-nki/bin/python scripts/trainium_report.py \
  --artifacts results/trainium/isa-reproduction --output results/trainium/isa-reproduction-report
```

The runner enables dynamic-DMA notifications by default. Reports retain full
instruction traces, warnings, hashes, source-level fallbacks, expanded ISA
counts, spill counts, byte errors and latency errors. TensorE math-interval
union is reported separately from engine-active time, which includes SDK
control instructions. Pipeline-overlapping instruction durations are not
simply summed.

Final evidence: `results/trainium/isa-report/report.json`, `measurements.csv`
and `comparison.json`. There are 21 correct hardware/simulator programs
across the two copy policies, including FP32, BF16, split-K, large tiles,
buffering alternatives, optional DMA transpose and ragged boundaries. All
21 reconvert byte-identically with the final code. All 19 matrix programs
match the predicted expanded TensorE instruction counts. HBM bytes match
exactly in 16 of 19; the exceptions are the small software tile, larger DMA
transpose case, and ragged graph. Repeated input loads are still removed by
the Neuron backend in the first two cases. The ragged case includes pad/slice
work outside the matrix estimate and still has profiled spill instructions.

| Case | Previous model / device us | ISA model / device us | Previous / ISA actual HBM bytes |
| --- | ---: | ---: | ---: |
| FP32 128-cube | 4.36 / 16 | 4.14 / 17 | 262,144 / 212,992 |
| FP32 512x256x512, selected mappings | 24.63 / 43 | 22.26 / 43 | 2,162,688 / 2,113,536 |
| FP32 1024x1024x256, fixed whole tile | 98.63 / 259 | 74.22 / 107 | 80,664,576 / 6,307,840 |

The large case is 2.42x faster with zero spill instructions; predicted HBM
bytes now match exactly. Relative latency error improves from 61.9% to
30.6%. The normal small/medium cases do not improve absolute latency
accuracy: explicit ISA fixes expansion/storage accounting, not instruction
completion and scheduling costs. For the four scalar-policy candidates on
the 512x256x512 problem, analytical selection picks the measured best (43 us
versus 45, 46 and 52 us), but this does not prove global optimality. The
optional balanced policy still misranks buffering (48 versus 44 us).

Important remaining mechanisms are SDK prologue/epilogue, short-instruction
completion versus documented steady-state initiation intervals, DMA packet
issue/completion, physical bank/port conflicts and backend scheduling. The
128-cube predicts 0.43 us TensorE service against roughly 4.7 us overall
TensorE activity; the latter includes setup/control as well as math. The
model must not be presented as calibrated occupancy or accurate end-to-end
kernel latency. A future timing refinement should use independently
characterized primitive latency/issue parameters and held-out schedules,
not an overall multiplier fitted to these GEMMs.

Validation: 96 focused Trainium/shared/Gemmini tests and 23 subtests pass;
formatting, final collateral reconversion and `git diff --check` pass. ImageNet
and Llama remain excluded as requested. Original checkouts are unchanged.

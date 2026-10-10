# Trainium operand modeling, 2026-10-10

The current extension is in `trainium/operand_timing.py` and
`trainium/operand_characterization.json`. It supplies bounded completion laws
for forms that were missing in the previous sweep. It is shared by selected
physical ISA scoring and static compiled-ISA analysis in context modes.
The compiler does not read hardware timings or benchmark names while scoring.

## Instructions and operands

Lookup distinguishes opcode, engine, input/output dtype, SBUF/PSUM endpoints,
activation/reduction function and effective reduction rank. Binary operations
also distinguish the second input's dtype and memory. Geometry supplies input
cardinality, output cardinality, partition count, strides and broadcast reads.
Singleton reduction axes do not add a reduction stage. `POOL AVERAGE` maps to
scaled addition; `MIN` is distinct from `MAX`.

| Added or extended form | Timing and domain |
| --- | --- |
| Affine select on GpSimd | Measured FP32 lane throughput and drain; documented 150-cycle startup. Contiguous widths 64–1024. |
| Predicated scalar copy on Vector | FP32 output and integer predicates; widths 64–1024, including 4-partition probes. Predicate geometry is checked. |
| Stream shuffle and mask load | BF16/UINT8 SBUF paths at widths 64–512. Measured mask load. Mask values and shape are checked. |
| Copies and casts | Additional Scalar/Vector dtype and SBUF/PSUM paths, including long FP32→BF16 Scalar casts through width 8192. Each path has its own measured width bounds. |
| Memset | UINT16 initialization now has a completion law; existing FP32/BF16 laws remain. |
| Scalar activations | Copy/identity, square, sqrt, rsqrt, log, SiLU and mixed-dtype EXP forms. Broadcast inputs and PSUM reads are represented separately. Long square and affine-copy probes extend through width 8192. |
| Reductions | Add/max/min, BF16→FP32, PSUM input, multi-output and multi-axis forms. Long FP32 reductions extend through width 8192; overlapping 3×3 max windows were separately characterized. |
| Scan | Existing FP32 multiply/add scan law validated at widths 4096, 7168 and 8192. |
| Activation accumulator | New 4-partition EXP→BF16/readback characterization at widths 64–1024, alongside the previous 128-partition path. The partial path conservatively waits for full producer completion; early forwarding was not observed. |

The JSON is the authoritative list of path-specific bounds and coefficients.
Pointwise forms require contiguous endpoints. Reduction service counts consumed
input elements, rather than only the reduced outputs. The reduction extension
does not establish cycle accuracy for every legal strided layout or bank conflict.
Partition counts not directly observed rely on the documented parallel-lane
execution assumption; partial-partition probes validate several important forms.

## Completion and initiation are separate

New completion laws use startup/drain plus per-element and, where measured,
broadcast-read terms. Existing issue/occupancy laws retain the documented
streaming work estimate unless an instruction-specific characterization exists. Newly characterized
Vector binary paths use the documented single-N service for parallel PSUM/SBUF
reads or contiguous packed BF16 add/multiply/subtract, rather than the generic
two-N SBUF read estimate. The second operand matters for this choice.
The new fallback conservatively bounds completion by its estimated occupancy;
raw measured durations can be shorter than that analytical estimate. This bound
is a model limitation, not a universal hardware rule. A longer completion is
not automatically a longer issue interval. This matters
for streams of independent small reductions and mixed Vector instructions.

Previously validated laws have priority within their original operand domain.
The old single-output reduction law is no longer applied to multi-output or
multi-axis reductions. Activation `COPY` uses Scalar affine characterization;
it cannot accidentally select the `tensor_copy` law of the same short name.
Selected NKI activation descriptors account for the pinned compiler's implicit
bias/constant pointer; a test compares selected and native activation latency.

## Hidden state and setup

Mask-load/shuffle dependencies and GpSimd fill-register dependencies include
read-after-write, write-after-read and write-after-write edges. These remain
when source-order constraints are disabled. Missing mask/register producers
fail explicitly. Existing Scalar accumulator dependencies remain separate.

Measured activation-table loads take approximately 1283 ns. Their common setup
is already included in the existing fixed kernel startup budget, so the graph
records a covered setup marker without charging the same setup twice. This
is lumped startup accounting, not a universally free instruction.

## Analytical domains remain visible

The pinned NKI SDK rejects explicit BF16 PSUM allocations or their legalization
(`NCC_IMSA300`). Narrow BF16 PSUM copies, BF16 PSUM copy activations and the
one-element SiLU form therefore use explicitly labeled aliases over measured
PSUM word paths. Their JSON entries carry `derived_from` and state that exact
operand-form characterization is unavailable. Compiled analysis reports their
counts as `derived_operand_models` and distinguishes analytical coverage from
direct characterization. Broader uncharacterized variants remain unknown.

The compiler's wide FP32 `TRANSPOSE` tag in the historical optimized-attention
NEFF represents a LOW/LOW_HIGH identity-matmul sequence, not a legal wide NKI
transpose. Its two-pass service is analytical and reported as an assumption.
This does not widen the lowerer's transpose constraints. That historical
kernel failed correctness and its timing is diagnostic only.

## Validation and reproduction

Characterization uses real Trainium2 / NeuronCore-v3, neuronx-cc
2.22.12471.0+b4a00d10 and one logical core. The extension contains evidence from
93 numerically checked probe programs and 60 path/function models, including
derived aliases. Fitting uses endpoint widths; 1738 interior-width or
partial-partition instruction samples are held out from fitting. This is
per-instruction characterization, with no whole-kernel latency fitting or
additional throttling factor. COPY conversion outliers remain evidence of
unmodeled context; low median error does not imply uniformly accurate timing.

All 49 retained streams are rescored using timing-free static metadata. Existing
hardware measurements are reused; this is not a new generated-kernel sweep.
Two historically incorrect kernels remain labeled incorrect and are excluded
from accuracy conclusions. Removing unknown completion laws does not resolve
all whole-kernel gaps. In particular, reference Conv1D was already predicted
at about 3.405 ms versus 6.011 ms measured before this extension. Its mixed
Vector stream remains underpredicted despite complete opcode coverage.

From the checkout root:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/ubuntu/ML/AGEN-voyager/.venv-compiler/bin/python \
  -m voyager_compiler.trainium.compiled_analysis \
  /path/to/compiled-static.json.gz \
  --context-model --execution-model context-ready \
  --output /tmp/trainium-prediction.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/ubuntu/ML/AGEN-voyager/.venv-compiler/bin/python \
  -m pytest -q -p no:cacheprovider \
  test/test_trainium*.py test/test_execution_contracts.py
```

Validation passed 256 regression tests. The final operand-domain checks passed
37 focused tests. Among 47 correctness-passed retained cases, nine still exceed
0.5 ms prediction error; the median absolute error is 0.113 ms. Asset-dependent
default `run_ci` compilation was not rerun for this extension.

Experiments and the standalone comparison report are outside this checkout at
`/tmp/trainium-complete-model-2026-10-10/`. The existing collateral archive and
downloaded ZIP are unchanged. Model parameters and their evidence hashes are
part of the code checkout; native binaries and execution logs are not.

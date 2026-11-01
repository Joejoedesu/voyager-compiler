# Default Voyager target

Registration is [`target.py`](target.py). It retains the tiled/bufferized
quantization, sparse/outlier and attention flow and downstream bitstream contract.
The [shared compilation guide](../../../../docs/compilation.md) owns compiler
invariants; this page owns the default target's bindings and regression scope.

## Extension points used

| Extension | Binding and responsibility |
| --- | --- |
| `hardware` | Shared `AcceleratorConfig` describes the default device |
| `implementations` | Shared dense/vector recipes, pointwise and GEMM overrides, and explicit whole-operation grouped/depthwise recipes; GEMM retains its shape-only tail for artifact compatibility |
| `model` | `implementations:VoyagerModel` and paired operation models for fit/cost evaluation |
| `operators` | `implementations/kernels.py:OPERATORS` declares primitive semantics |
| `encode` | `codegen.py:encode` produces the established model projection |

Other hooks retain shared defaults. Workload/frontends own export/calibration;
shared search, bufferization and allocation own their normal stages.

With `accumulate_fp32`, split reductions produce FP32 partials before adding them
and round the completed result at its output boundary. GEMV search reserves the
corresponding FP32 accumulator. MX matrix operations combine the dense product,
scaling, bias and sparse outlier contribution before the final output rounding.

## Grouped and depthwise dispatch

`voyager.depthwise` and `voyager.grouped` explicitly select the existing
whole-operation convolution path. `VoyagerModel.prepare` leaves these operations
untiled; shared search has no target-independent depthwise exclusion. The recipe
reuses the convolution template with `whole_tensor=True` for fused MX epilogues
and the existing untiled template otherwise. Dense recipes are disjoint. This
preserves the established control and instruction format; it does not add a new
grouped-convolution tiling search or establish hardware support for new signatures.

## Opt-in attention algorithm comparison

```python
from voyager_compiler.targets.voyager.target import with_attention_algorithms

target = with_attention_algorithms()
# Pass target to compile_workload(..., target, ..., config=config).
```

This target value compares `voyager.attention.materialized` with
`voyager.attention.streaming` for supported FP32 SDPA regions. The first reuses
Voyager GEMM/vector recipes; the second uses its FA3 builder/attention model.
The registered default target and quantized attention recipe remain unchanged.
Supply the complete hardware configuration, including local buffer capacities,
as for normal matrix search. `runtime_tolerance=0` selects strict modeled latency;
this is now the default. A positive tolerance explicitly enables near-fastest
traffic/energy selection.

Current comparison excludes masks, dropout, causal mode, GQA and differing value
head widths. Materialization can handle per-GEMM padding; streaming rejects shapes
whose products have no legal mapping. Inspect `plan.alternatives`, `plan.rejected`
and the winner's `components`. Costs are analytical estimates, not measurements.
See [algorithm comparison](../../../../docs/compilation/algorithm-search.md) and
`tests/test_region_algorithms.py` for the shared contract and forced-recipe checks.

## Padding

Common normalization handles matrix/quantization alignment; selected builders
handle convolution halos, tile tails and attention padding under their legality
rules. The workload-specific `test/utils/models/vit.py:pad_vit_embeddings_output`
extends embedding sequence length and rewrites downstream reshapes. It is not a
general attention-padding pass: check masks, positions and reference outputs
before reusing it. See the shared [padding decisions](../../../../docs/compilation/padding.md).

## Pinned development gate

The active reference is `reference-point/voyager-latency0-2026-10-31`, selected
by `validation/reference-baselines.json`. Its 11 development cases explicitly
pin `--runtime_tolerance 0.0`: minimize modeled latency, with secondary criteria
only breaking equal-latency ties. The user approved this replacement after the
compiler default changed from 2% to 0%. The historical 26-case snapshot and the
October 23 decode correction remain archived; the active snapshot includes both
required concatenation copies.

Eight cases match their numerical references; three Llama cases retain pinned
numerical warnings. Passing this gate establishes artifact/status/warning-summary
compatibility, not model accuracy. Changing a reduction tile can change BF16
rounding; a new search policy needs its own numerical acceptance check.

Run from the repository root into a fresh output directory:

```bash
/home/zhouhua/Research/ML/ml-env/bin/python validation/run_references.py \
  results/voyager-suite-RUN --offline --jobs 2 --dependency-path ../.validation-packages
```

The runner reads the baseline and commands named by the pins, checks their
checksums, and records the explicit search policy. `validation/run_case.py`
supplies the two ImageNet-dependent cases with local images in pinned order,
verifying their checksums and retaining the original frontend transforms.
The image paths are recorded in the snapshot's `imagenet-inputs.json`.
GLUE/WikiText reuse existing Arrow caches; offline runs stage file links under
the output directory so cache locks and new writes stay local. Set
`HF_DATASETS_CACHE` to select another existing cache source. No dataset download,
sitecustomize adapter or report-directory environment setup is required.
Optional dataset download-count telemetry is disabled. The existing CPU-capable
environment and small compatibility dependency directory remain sufficient.

Source base: compiler-2 `cef1e7e42b31e1cfcc01a1e4e02cedab11e3e9ef`.

## Fusion policy

`fusion.py:FUSION_PATTERNS` owns the broad graph policy. Registration and the
historical test frontend consume this same table. Hardware ISA-pipeline records
do not derive these patterns; no configuration-aware resolver is used.

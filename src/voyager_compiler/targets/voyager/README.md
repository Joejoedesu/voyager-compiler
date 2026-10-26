# Default Voyager target

Registration is [`target.py`](target.py). It retains the tiled/bufferized
quantization, sparse/outlier and attention flow and downstream bitstream contract.
The [shared compilation guide](../../../../docs/compilation.md) owns compiler
invariants; this page owns the default target's bindings and regression scope.

## Extension points used

| Extension | Binding and responsibility |
| --- | --- |
| `hardware` | Shared `AcceleratorConfig` describes the default device |
| `implementations` | Shared recipes, with `build_pointwise` and `build_gemm` overrides; GEMM retains its shape-only tail for artifact compatibility |
| `model` | `implementations:VoyagerModel` and paired operation models for fit/cost evaluation |
| `operators` | `implementations/kernels.py:OPERATORS` declares primitive semantics |
| `encode` | `codegen.py:encode` produces the established model projection |

Other hooks retain shared defaults. Workload/frontends own export/calibration;
shared search, bufferization and allocation own their normal stages.

## Padding

Common normalization handles matrix/quantization alignment; selected builders
handle convolution halos, tile tails and attention padding under their legality
rules. The workload-specific `test/utils/models/vit.py:pad_vit_embeddings_output`
extends embedding sequence length and rewrites downstream reshapes. It is not a
general attention-padding pass: check masks, positions and reference outputs
before reusing it. See the shared [padding decisions](../../../../docs/compilation/padding.md).

## Pinned development gate

The Voyager gate selects the user's 11 cases from the 26-case historical
snapshot. The historical KIVI failure is outside that selection. Some selected
cases have pinned numerical warnings: preserving those warnings passes artifact
compatibility, not numerical accuracy. The approved decode replacement at
`../reference-point/voyager-decode-corrected-2026-10-23` retains both concatenation
copies and supersedes only that one artifact. All other comparators remain fixed.

Run from the repository root using the existing environment and a fresh output:

```bash
/home/zhouhua/Research/ML/ml-env/bin/python validation/run_references.py \
  ../reports/voyager-suite-RUN --jobs 2 --dependency-path ../.validation-packages
```

Use a new `RUN` directory each time. The two ImageNet-dependent Voyager cases
use an ImageNet input (three calibration steps for ResNet INT8, one example
image for ViT). The original frontend uses streaming, but the current static-
preparation validation runs entirely locally: `--offline`, the existing GLUE/
WikiText Arrow caches with workspace-local lock files, and an explicit local
ImageNet adapter preserving TFDS validation order. The report evidence records
the adapter, image checksums and exact invocations under
`../reports/voyager-static-preparation-2026-10-24/`. Both ImageNet cases must
still match the original pinned model hashes and numerical summaries. Disable
optional dataset download-count requests with `HF_UPDATE_DOWNLOAD_COUNTS=0`;
those S3 HEAD requests are not required for input loading or compilation. The environment is Python 3.13/Torch
2.12.1+cu130/TorchAO 0.17, running on CPU. The dependency path contains only small
existing compatibility dependencies, not another Torch/CUDA environment.


Source base: compiler-2 `cef1e7e42b31e1cfcc01a1e4e02cedab11e3e9ef`.

## Fusion policy

`fusion.py:FUSION_PATTERNS` owns the broad graph policy. Registration and the
historical test frontend consume this same table. Hardware ISA-pipeline records
do not derive these patterns; no configuration-aware resolver is used.

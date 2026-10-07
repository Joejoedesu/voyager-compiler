# Trainium ISA mapping and measured timing

This inventory describes `voyager-trainium-10-06-isa`, Trainium2 / NeuronCore-v3,
with neuronx-cc 2.22.12471. The selected-plan instruction registry is in
`trainium/operations.py`; compiled prior-work instructions are mapped by
`trainium/compiled_analysis.py`. These are different coverage boundaries.
Mapping means the operation and its engine/dependencies are represented; it
does not imply complete measured timing or support for every legal NKI variant.

## Data movement and layout

| NKI ISA operation | Engine / compiled form | Placement and purpose | Coverage and timing |
| --- | --- | --- | --- |
| `dma_copy` | DMA; `DMA_DIRECT2D` or static DMA trigger/descriptors | HBM ↔ SBUF; panel loads/stores | Emitted and analyzed. Measured issue, dispatch, payload and notification laws; uses actual panel bytes/geometry. |
| `tensor_copy` | VectorE or ScalarE; `COPY` | SBUF → SBUF staging/assembly, PSUM → SBUF result eviction; can convert result dtype | Emitted and analyzed. Measured FP32 SBUF copies on VectorE/ScalarE, BF16 SBUF copies on VectorE, FP32/BF16 PSUM copies on ScalarE. Other routes/dtypes retain analytical service and unknown completion. |
| `memset` | VectorE; `MEMSET` | Initialize SBUF or PSUM, including accumulator clears and padding | Emitted and analyzed. FP32/BF16 timing laws exist; other dtypes can be mapped but lack completion calibration. |
| Tensor transpose (`nc_transpose` semantics; current selected lowering uses `nc_matmul(..., is_transpose=True)`) | TensorE; `LDWEIGHTS` + `MATMUL` with TRANSPOSE type | SBUF → PSUM, then a separate `tensor_copy` to SBUF; tiles ≤128×128 | Emitted and analyzed as the identity-matrix transpose recipe. FP32/BF16 laws are available for the ScalarE-copy recipe. |
| Vector `nc_transpose` | VectorE; `STREAM_TRANSPOSE` | Partition/free-axis transpose directly SBUF → SBUF | **Compiled analysis and opt-in endpoint movement search.** Measured support is restricted to contiguous FP32 32×32 SBUF tiles, with source/destination partition-group context. Other forms are explicitly unsupported. |

Changing the transpose engine alone does not remove redundant layout conversions
or staging copies. A 128×128 Vector transpose requires sixteen 32×32 operations;
layout/engine selection is now available through the opt-in [endpoint movement search](trainium-movement-search.md).
The legacy `dma_transpose` option is not part of the active explicit-ISA mapping.

## Compute

| NKI ISA operation | Engine / compiled form | Work represented | Timing coverage |
| --- | --- | --- | --- |
| `nc_matmul` | TensorE; grouped `LDWEIGHTS` + regular `MATMUL` | SBUF stationary/moving operands → FP32 PSUM accumulation; GEMM/BMM and convolution panels | FP32/BF16 measured laws plus analytical shape service. FP16 recipe exists but lacks a measured completion law. New exact FP32 K=64, stationary-free=128, moving-free=512 law applies in candidate graphs, selected-plan analysis and compiled analysis. |
| `tensor_tensor` | VectorE; `TENSOR_TENSOR` | Elementwise add/subtract/multiply/maximum; also separable max-pool lowering | FP32 measured law; other dtype completion laws incomplete. |
| `tensor_scalar` | VectorE or ScalarE; `TENSOR_SCALAR` | Scalar/per-partition broadcast arithmetic, scaling, affine operations, normalization | FP32 VectorE measured law; ScalarE and other dtype completion laws incomplete. Current arithmetic lowering normally selects VectorE. |
| `tensor_reduce` | VectorE; `TENSOR_REDUCE`, plus compiled `POOL` reduction forms | Free-axis sums/maxima for softmax and normalization | Analytical service; measured completion remains unknown. Voyager's current max-pool implementation uses `tensor_tensor` maxima, not a separate NKI pool API. |
| `activation` | ScalarE; `ACTIVATE` | exp, rsqrt, sigmoid, tanh, ReLU; SiLU decomposes into sigmoid and multiply | Analytical service; measured completion remains unknown. |
| `reciprocal` | VectorE; `RECIPROCAL` | Inverse of the softmax denominator | Analytical service; measured completion remains unknown. |

Arithmetic recipes operate on local tiles and normally produce SBUF values;
matmul produces PSUM. Host code and GpSimd are not substitutes for these engines.
`EVENT_SEMAPHORE` and `NOP` provide ordering/control rather than data or compute
work. Encoded semaphore waits become completion edges. `ACT_TABLE_LOAD` is
tracked as setup covered only by the existing fixed device term. Arbitrary
`WRITE`, mask-control opcodes and unrecognized instructions fail explicitly.

## Integrated calibration, 2026-10-07

The new timing profile is `trn2-primitives-stream32-shortK64-2026-10-07`.
`isa.Expansion` retains the physical implementation separately from its timing
implementation, so shape-specific timing does not alter backend instruction
counts or hardware capability validation. Both candidate dependency graphs and
selected-plan replay use that timing selection.

| Measured form | Issue / occupancy | Completion / dependency-ready latency |
| --- | ---: | ---: |
| FP32 stationary (64,128), moving (64,512) | 1,707 ns | 2,765 ns |
| FP32 32×32 stream transpose, same partition-group pair | 93 ns | 233 ns |
| FP32 32×32 stream transpose, first operation or changed pair | 234 ns | 234 ns (233 ns clamped to occupancy) |

Stream context follows static Vector-engine issue order. Wait-only instructions
preserve it; other Vector payloads reset it. The group is the physical partition
index divided by 32. Repeated transfers between two different but fixed groups
still use the steady rate. The context effect is empirical; the measurements do
not establish a specific internal bank or pipeline mechanism.

The K=64 rule does not extrapolate to other shapes/dtypes. The previous formula
omitted K from throughput and predicted 853 ns for both K=64 and K=128. Isolated
probes measured 1,707 ns versus 860 ns, with both independent outputs and
accumulation. Sixteen-operation held-out K=64 probes reproduced the correction.
Stream held-outs reproduce instruction intervals, but short whole-probe runtime
predictions still have overhead error; this is not universal cycle accuracy.

All constants came from isolated hardware probes, not application latencies.
Fourteen probe result/source records and authenticated artifact hashes are in
`results/trainium/model-integration-2026-10-07/calibration/` and
`calibration-provenance.json`. Original NEFF/NTFF/reference artifacts remain in
`/home/ubuntu/ML/trainium-prior-model-experiment/followup/`. Hardware measurements
are authenticated reuse; this integration does not claim new device executions.

## Evaluation

The repository predictor exactly reproduces the separate experiment's six
predictions, DMA bytes, encoded waits, selected laws and unknown completion counts.
Shuffling static input records leaves every reported value unchanged.

| Prior-work kernel | Predicted ms | Hardware ms | Error |
| --- | ---: | ---: | ---: |
| LayerNorm | 1.731 | 1.728 | +0.20% |
| Maxpool | 1.375 | 1.358 | +1.22% |
| SwiGLU | 4.058 | 4.082 | −0.59% |
| BMM → softmax | 13.456 | 14.591 | −7.78% |
| Residual → RMSNorm → GEMM | 4.062 | 4.114 | −1.26% |
| GEMM → residual → RMSNorm | 8.405 | 8.466 | −0.72% |

BMM's remaining 1.135 ms error is unresolved. The new law explains approximately
73% of the original error, but completion timing and backend scheduling remain
incomplete. No generated kernel, allocation or layout choice changes here.

## Reproduction

From the checkout root, use the compiler environment and set `PYTHONPATH=src`:

```sh
PYTHONPATH=src python -m voyager_compiler.trainium.compiled_analysis \
  /home/ubuntu/ML/trainium-prior-model-experiment/artifacts/bmm_softmax/compiled_static.json \
  --output /tmp/bmm-prediction.json
PYTHONPATH=src python scripts/trainium_validate_calibrated_isa.py \
  --experiment /home/ubuntu/ML/trainium-prior-model-experiment \
  --output results/trainium/model-integration-2026-10-07/prior-work-validation.json
PYTHONPATH=src python -m pytest -q test/test_trainium_calibrated_isa.py
```

The predictor accepts static compiled metadata only, with allowlists for
instruction, DMA and audit fields. Runtime timestamps/durations and measured
application latency are not inputs. Unsupported stream forms return no complete
prediction; unknown completion laws are reported, not silently labeled calibrated.

Validation on integration: **124 tests and 23 subtests passed**, compared with
109 tests and 23 subtests before the change. The 11 marked default compilation
cases were run against an authenticated pre-change source copy; all fail before
compilation in both revisions with identical terminal errors: six gated ImageNet
dataset failures, three gated Llama model failures and two Hugging Face GLUE URI
errors. Emitted-program equivalence is therefore not established by that suite.
Shared/default compiler sources remain byte-identical. Full logs and comparison
records are in `results/trainium/model-integration-2026-10-07/validation.json`.

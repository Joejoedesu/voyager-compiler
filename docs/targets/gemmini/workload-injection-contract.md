<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# ResNet-18 quantized-value ReLU × Gemmini injection contract

This W-stage contract binds ordinary shared PT2E operations to existing
Gemmini software helpers. Canonical `model.txt` operations and tensor dumps
are the only semantic inputs. The adapter validates, packs, and projects those
facts; it does not invent quantization domains or target-local schedules.

This is a **partial-control injection point**. The `_auto` helpers privately
choose performance-relevant tiling, traversal, buffering, DMA, scratchpad and
accumulator windows, local addresses, and RoCC expansion after Voyager emits
each logical call.

## Conv/ReLU regions

An admitted Conv region has an optional input `quantize`, followed by
`conv2d`, accumulator `dequantize`, final output `quantize`, and ordinary
out-of-place `relu`. The stem may end with `max_pool2d`. ReLU consumes the
final signed-INT8 qcode tensor directly. There is no intermediate output-Q,
DQ, or floating-ReLU storage domain, and no fused-member result dump is
required. The fused output is the ReLU or MaxPool INT8 tensor.

The output quantize uses a positive scalar FLOAT32 scale, implicit zero point
zero, nearest rounding, and bounds `[-128, 127]`. Conv weight is INT8 OIHW,
bias is INT32, and accumulator DQ has a positive scalar FLOAT32 scale.
Geometry is read unchanged from ordinary Conv2d kwargs.

The adapter emits one `tiled_conv_auto` call. Its activation is `RELU`, its
store multiplier is `accumulator_scale / output_scale`, and optional pooling
comes from the canonical MaxPool kwargs. Input/weight packing and static input
quantization remain adapter-owned; inner traversal, DMA, local addresses, and
RoCC commands remain helper-owned.

## Residual Add/ReLU regions

An admitted complete residual is `DQ, DQ, Add, Q, ReLU`. The terminal form is
`Add, Q, ReLU`, with both input qcodes and scales resolved through canonical
top-level producer DQs. ReLU consumes the final output qcodes. The removed
legacy `Q, DQ, ReLU, Q` round trip is not reconstructed by collateral.

The adapter emits one `tiled_resadd_auto` call. Its input multipliers and
single output multiplier are ratios derived only from the two input scales
and the one canonical output scale. It enables the helper's ReLU flag.

## Other established helpers

The existing `DQ, AdaptiveAvgPool2d, Q` global-average and connected
`Linear, DQ, Q` bindings remain unchanged. Normalization remains unavailable
because the pinned target configuration lacks `HAS_NORMALIZATIONS`.

## Qualification

Every admitted static region is executed independently on the pinned Gemmini
Spike stack. The complete logical signed-INT8 output is compared to the
target-legalized canonical output with `rtol=0` and `atol=1`. Exact mismatch
count and maximum error are retained but exact equality is not required.
Those counts are W evidence only. This public-helper contract does not exercise
the same W binding through a full-control interface. X cannot be claimed
through this partial-control point even if output is exact.
Results are fixed-static region evidence, not connected-model accuracy,
arbitrary-input equivalence, or a Gemmini hardware change.

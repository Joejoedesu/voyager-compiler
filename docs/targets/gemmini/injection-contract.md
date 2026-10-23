<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Gemmini shared-Q/DQ Conv2d injection contract

This R-level contract defines one fail-closed projection from shared Voyager
semantics to a public Gemmini helper. It proves one static functional instance;
it does not claim arbitrary inputs, bit-exactness, performance, or ownership of
Gemmini's internal schedule.

**Injection control:** partial-control. Voyager emits one logical Conv2d helper
call, while `tiled_conv_auto` privately selects tiling, traversal, buffering,
DMA, scratchpad and accumulator windows, local addresses, and RoCC expansion.
Those hidden choices materially affect the mapping cost. This point can support
R and W evidence, but cannot support X even if a result compares bit-for-bit.

## Canonical graph

The canonical package contains exactly two top-level operations:

1. one `fused_op` with members `quantize`, `conv2d`, `dequantize`, and
   `quantize`, in that order; and
2. one terminal `dequantize` that consumes the fused signed-INT8 output and
   produces the public FLOAT32 result.

The corresponding shared targets are `quantized_ops.quantize`, ordinary
`aten.conv2d` (or shared `quantized_ops.conv2d`), and
`quantized_ops.dequantize`. No Gemmini-private arithmetic operator, schedule,
tile, local address, DMA command, or raw accumulator output appears in
`model.txt`.

Member kwargs are limited to:

| Member | Kwargs |
| --- | --- |
| quantize | `input`, `scale`, `rounding`, `quant_min`, `quant_max` |
| Conv2d | `input`, `weight`, `stride`, `padding`, `dilation`, `groups` |
| dequantize | `input`, `scale` |

Quantization is signed INT8, symmetric, zero point zero, nearest-even, with
bounds `[-128, 127]`. Four positive scalar FLOAT32 operands describe input
quantization, accumulator dequantization, output quantization, and terminal
dequantization. Output and terminal scales must be bit-identical. The weight
scale is `accumulator_scale / input_scale` and is retained in oracle evidence,
not added to the fused ABI.

## Frozen registration instance

| Fact | Value |
| --- | --- |
| input | NCHW FLOAT32 `[1, 16, 4, 4]` |
| weight | OIHW signed INT8 `[16, 16, 3, 3]` |
| bias | absent |
| stride / padding / dilation | `[1, 1]` / `[1, 1]` / `[1, 1]` |
| groups | `1` |
| fused output | NCHW signed INT8 `[1, 16, 4, 4]` |
| terminal output | NCHW FLOAT32 `[1, 16, 4, 4]` |

The accepted nonunit scales are input `0.5`, weight `0.25`, accumulator
`0.125`, and output/terminal `0.06200787425041199`.

Tensor dumps are contiguous little-endian FLOAT32 containers, including
logical INT8 tensors. The adapter validates declared dtypes, shapes, finite
values, and integer code ranges before packing.

A fused operation materializes only its final output. The package therefore
contains fused signed-INT8 and terminal FLOAT32 output dumps, but no dump for
any fused member result. Input, weight, scales, and other referenced support
tensors remain ordinary artifacts. `Tensor.memory` values are compiler storage
facts; they are not native pointers or Gemmini-local addresses.

## Native projection

The adapter performs only these target-facing transformations:

- statically quantize the frozen FLOAT32 input with
  `clamp_s8(RNE_FP32(input / input_scale))`;
- pack input NCHW to NHWC and weight OIHW to HWIO;
- allocate a zeroed signed-INT8 output; and
- emit one non-inlined wrapper containing one public `tiled_conv_auto` call.

The helper receives the canonical shape and convolution geometry, no bias,
`NO_ACTIVATION`, no pooling, `WS`, and the FLOAT32 multiplier
`accumulator_scale / output_scale`. Gemmini owns tiling, traversal, buffering,
DMA, scratchpad and accumulator windows, local addresses, and RoCC expansion.
The terminal dequantize remains outside the native call.

Input, weight, output, and comparison-oracle arrays must be 16-byte aligned,
non-overlapping ELF symbols. The linker owns their addresses. Layout mappings
must be complete and bijective; the oracle is comparison-only and is never a
Gemmini input.

## Numerical acceptance

The canonical package contains independent pre-transform PT2E references for
the fused INT8 codes and terminal FLOAT32 output. The converter validates and
hash-binds them; it must not derive or repair either reference.

Shared qmap rounding normalizes through BF16, while static packing and the
Gemmini helper use FP32. Boundary values can differ by one signed-INT8 code.
Acceptance therefore checks all 256 fused-output codes with `rtol=0`,
`atol=1`; exact mismatch count and maximum error are recorded but need not be
zero. The accepted instance has 252 exact values, four one-code differences,
and no tolerance failures.

The native runner must prove one generated helper call, one observed wrapper
entry, complete ordered readback, and no CPU-produced output. Helper-call and
dynamic RoCC-instruction counts are separate evidence.

## Runtime lock and ownership

`conversion/gemmini/stack-manifest.json` is a portable lock for
the qualified Gemmini, RoCC-test, RISC-V-test, libgemmini, Spike, tool, and
header identities. The caller supplies the stack root. Compile and link flags
are implementation-owned constants in `conversion/gemmini/run.py`; the
semantic contract is not duplicated in the runtime lock.

Voyager owns shared arithmetic, qparams, graph order, logical tensors, and
canonical outputs. Conversion owns fail-closed validation, static input
quantization, packing, generated symbols, and manifests. The public helper
owns accelerator-local control. The runner owns immutable build, launch,
readback, inverse layout, and comparison.

This contract does not claim raw accumulator output, arbitrary runtime FLOAT32
input, arbitrary shape or scale, per-channel or MX quantization, bias,
activation, pooling, grouped or dilated convolution, QAT, connected workload
execution, cycles, traffic, utilization, energy, RTL behavior, or
Voyager-selected performance mapping.

## Source anchors

- `decomposed.py`: shared quantize, Conv2d, dequantize, and qmap behavior.
- `gemmini/hardware_pipeline.py`: target-owned PyTorch-semantic matchers.
- `gemmini/lowering.py`: exact region structure and no-local-control validation.
- `gemmini/serialization.py`: canonical metadata validation.
- `mapping_utils.py`: FLOAT32 tensor-container serialization.
- `conversion/gemmini/convert.py`: endpoint ABI validation and helper projection.
- `conversion/gemmini/run.py`: locked build, Spike execution, and comparison.
- public Gemmini `gemmini.h`: `tiled_conv_auto`, the first native consumer.

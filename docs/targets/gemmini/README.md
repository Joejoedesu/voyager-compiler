<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Gemmini

Gemmini is a registered Voyager compilation target backed by the public
Gemmini software helpers and a pinned functional Spike stack. Voyager keeps
the model semantics in shared operators and projects the finalized canonical
collateral into the helper ABI under `conversion/gemmini/`.

This page is the entry point for the target. Detailed contracts, commands, and
results live in the linked documents rather than being repeated here.

## Adoption status

| Phase | Status | Qualified claim |
| --- | --- | --- |
| R: target registration | subsumed by W for this scope | a separate shared Q/Conv/DQ/Q anchor is retained for maintenance |
| W: ResNet-18 binding | complete for the fixed pair | 30 mapped regions execute independently at `rtol=0`, `atol=1` |
| X: bit-accurate full-control binding | not started | the selected `_auto` helper is partial-control; the W binding has not been rebound through a full-control point |

The fixed ResNet-18 binding achieved W and therefore also satisfies R for this
injection point. A separate operation anchor isolates the registration path for
maintenance. The retained endpoint-validated package is a review-time
golden-promotion candidate until its source changes are reviewed. Its 30/30
native campaign binds the predecessor workload-contract digest. The current
contract adds the injection-control/X classification without changing the
helper or numerical boundary; no current-contract native rerun is claimed.
The retained results do not prove connected full-model execution, arbitrary
shapes or inputs, exact arithmetic, or target performance.

Gemmini X has not started. The current W/public-helper path stops at the public
`_auto` helper ABI, a partial-control injection point where the Gemmini software
stack privately selects accelerator-local tiling, DMA, scratchpad placement,
and RoCC expansion. X cannot be claimed through this point even if output is
exact.
Voyager has not rebound the same frozen ResNet-18 scope, semantic reference,
and comparison boundary through a full-control interface. The exact and
one-code mismatch counts are W accounting only, not an active X gap.

## Documentation

- [Operation inventory](operations.yaml) records the compact W regression
  snapshot plus supported, unsupported, and unverified semantic forms without
  conflating support with check results.
- [R injection contract](injection-contract.md) defines the canonical shared
  Q/Conv/DQ/Q boundary and its `tiled_conv_auto` projection.
- [Workload injection contract](workload-injection-contract.md) defines the
  ResNet-18 Conv, residual Add, global-average, and Linear projections.
- [Injection flow](injection-flow.md) explains target-stack setup, canonical
  conversion, collateral generation, and runtime execution.
- [Regression plan](regression.md) contains the maintained compiler,
  converter, and Spike checks.
- [ResNet-18 report](resnet18.md) owns the frozen W-stage workload,
  assignments, numerical results, and limitations.

## Compiler boundary

Gemmini reuses shared Voyager semantics. Final FX and `model.txt` contain
ordinary quantize, dequantize, Conv2d, Add, pooling, ReLU, and Linear members.
They do not contain a Gemmini-private arithmetic operator or accelerator-local
schedule.

The target backend under
`src/voyager_compiler/codegen/target_backends/gemmini/` owns:

- the default-derived `GemminiHardwareConfig`;
- target quantization-domain preparation;
- an `OpMatcher` hardware pipeline for supported linear PyTorch semantics;
- target grouping and final-FX structural validation; and
- fail-closed serialization metadata checks.

It does not own Gemmini-local loop order, tile selection, DMA traversal,
scratchpad addresses, or RoCC commands.

`hardware_pipeline.py` is the single inventory of ordinary linear PyTorch
semantics that the hardware can match. Its Conv core also derives the stricter
required public-injection grouping. Target quantization forms richer workload
groups, including the residual DAG, and `lowering.py` validates their exact
final-FX structure and dependencies. The concrete helper symbol, accepted ABI
arguments, tensor shapes, and packing remain adapter-owned. These groupings
are neither one-instruction claims nor physical schedules, and they are not
serialized as a second execution IR. The current exact-pair adapter keeps
optional generic fusion disabled and uses the target-owned grouping path.

Only code-domain ReLU forms are helper-executable. A retained
`Q -> DQ -> ReLU -> Q` round trip is not grouped as one helper operation: it
denotes two observable INT8 saturations unless exact equal-domain legalization
has already removed it.

## Conversion and runtime boundary

The reusable adapter under `conversion/gemmini/` consumes canonical compiler
collateral. It validates declared tensors and qparams, performs required
layout packing and target scale projection, and emits calls to existing public
helpers:

| Semantic region | Public helper |
| --- | --- |
| Conv with optional ReLU or stem MaxPool | `tiled_conv_auto` |
| residual Add with ReLU | `tiled_resadd_auto` |
| global average | `tiled_global_average_auto` |
| Linear | `tiled_matmul_auto` |

The helper library chooses the accelerator-local tiling and emits the RoCC
commands. The generated program and runtime runner build an ELF, launch Spike,
read back the complete logical output, and compare it with the canonical
reference.

`stack-manifest.json` is only a portable identity lock for the
external Gemmini sources, headers, tools, libgemmini, and Spike binary. It is
not a second semantic contract.

## Qualified numerical scope

The R and W comparisons use complete logical outputs with `rtol=0` and
`atol=1`. Exact mismatch counts and maximum error remain reported separately;
exact equality is not required. Those statistics characterize W and do not
turn the partial-control helper point into an X binding.

The fixed ResNet-18 mapping contains:

- 20 Conv-family regions;
- eight residual Add regions;
- one global-average region; and
- one Linear region.

Inference BatchNorm is folded into Conv parameters because the qualified
Gemmini configuration does not expose normalization support. Do not emit or
claim `tiled_norm_auto` while `HAS_NORMALIZATIONS` is absent.

## Maintained entry points

Use the standard Voyager compiler and exact workload-target dispatch for W
canonical collateral. The maintained boundaries are:

- `test/test_codegen.py resnet18 --target gemmini` dispatches the exact
  ResNet-18 x Gemmini compiler/capture adapter;
- `conversion.gemmini.workload` converts and runs W regions;
- `conversion/gemmini/convert.py` and `conversion/gemmini/run.py` convert and
  run the R operation anchor; and
- `conversion/gemmini/test_convert.py`, `test_workload.py`, and
  `test_compiler_quantization.py` cover fail-closed adapter, capture, and
  quantization-domain legalization behavior.

There is no separate `test/run_gemmini_operation_regression.py`. R and W reuse
the same compiler, converter, and runtime entry points as the other registered
targets.

## Current limitations

The current evidence does not establish:

- arbitrary runtime FLOAT32 inputs, shapes, scales, or qschemes;
- per-channel, MX, QAT, or raw-accumulator output support;
- Gemmini normalization support;
- connected end-to-end ResNet-18 execution or workload accuracy;
- a full-control binding of the same frozen ResNet-18 scope and reference;
- bit-exact target arithmetic or Voyager-owned accelerator-local control; or
- target cycles, traffic, utilization, energy, or RTL behavior.

Treat any broader form as unverified until it passes the applicable compiler,
conversion, endpoint-execution, and complete-output comparison gates.

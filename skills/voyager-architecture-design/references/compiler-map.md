<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Voyager Compiler Ownership Map

Verify current symbols with `rg` before relying on this map. It describes
responsibility boundaries, not a promise that line numbers remain stable.

## Front-end and quantization

| Responsibility | Primary location |
| --- | --- |
| Qscheme, qspec fields, parser, validation | `src/voyager_compiler/quantizer/quantizer.py` |
| Operator annotation and qconfig policy | `src/voyager_compiler/quantize_pt2e.py` and `quantizer/` |
| Observer state, fake quantization, STE | `src/voyager_compiler/fake_quantize.py` |
| Target-neutral/reference stateless custom-op schemas and behavior | `src/voyager_compiler/decomposed.py` |
| Prepare and fake-quant replacement | `src/voyager_compiler/quantize_pt2e.py` |
| Model-specific mixed precision | `test/utils/models/*.py` |

Use `QuantizationSpec` to describe one edge's requested scheme. Use the
quantizer to decide which operator edges receive which specs. Do not place a
network's layer-name policy in generic qspec or conversion code.

During prepare/calibration/QAT, fake-quant modules return reconstructed floating
tensors so ordinary PyTorch operators remain executable. During conversion,
replace those modules with explicit payload and qparam nodes. A custom
operator's `register_fake` function is FakeTensor shape/meta behavior, not QAT
fake quantization.

## Transform and lowering

| Responsibility | Primary location |
| --- | --- |
| Public transform pipeline and pass order | `src/voyager_compiler/__init__.py::transform` |
| High-level decomposition/legalization | `codegen/passes/lowering.py` |
| Padding | `codegen/passes/padding.py` |
| Tiling | `codegen/passes/tiling.py` |
| Data-layout transforms | `codegen/passes/data_layout.py` |
| Compilation-target constants | `codegen/targets.py` |
| Shared hardware-pipeline matching and datapath projection | `codegen/hardware.py` |
| Target final-FX classification | `codegen/target_backends/<target>/lowering.py` when needed |
| Target hardware-pipeline declarations | `codegen/target_backends/<target>/hardware_pipeline.py` |

`transform()` is the pass orchestrator. Functions in `passes/lowering.py` are
individual rewrites. For example, Magnet Alpha channel-shuffle recognition belongs
in target dispatch and runs before generic cat/view rewrites destroy its
pattern.

Keep reference semantics executable after every transform. Re-run shape
propagation after creating target-specific nodes when later passes consume
shape metadata.

The final target FX graph is the semantic instruction carrier. A target may
use the existing late memory-rewrite callback to insert executable physical
movers before execution order, storage planning, and scheduler workload
construction, then bind the existing operation classification. Do not add a
shared IR, target hook, or protobuf field solely for that target-local split.

## Mapping and code generation

| Responsibility | Primary location |
| --- | --- |
| Memory allocation, aliases, liveness, copies | `codegen/mapping.py` |
| Operation classification and field mapping | `codegen/mapping_utils.py` |
| Protobuf definitions | `codegen/param.proto` |
| Default serialization | `codegen/mapping.py::gen_code` |
| Bufferized lowering | `codegen/lowering/` |
| External collateral projection | `conversion/<target>/` |

Let mapping decide whether a logical data movement aliases storage or requires
a physical copy. Let serialization emit the decision already represented by
the graph, schedule, storage plan, allocation, and explicit qparams. Keep an
external runtime's route, disposition, fusion owner, descriptor, epilogue,
skip, and encoding policy in `conversion/<target>/`; do not use codegen as a
hidden external-schema or qparam pass.

The bufferized path is a terminal alternative that creates explicit loops and
memory primitives. A default-path change does not automatically support it.

## Change-routing examples

### Add or extend a quantization scheme

Inspect, and change only as required:

1. qspec fields, parser, and validation;
2. operator-axis annotation and legal combinations;
3. fake-quant forward state and STE;
4. stateless decomposed arithmetic;
5. conversion to explicit graph values;
6. tests for parsing, math, gradients, conversion, and consumers;
7. mapping/codegen only when new operands or output structures cross that
   boundary.

### Add a target intrinsic

Apply the R/W/X gate in `SKILL.md` before creating it. During W, a target
semantic intrinsic requires evidence that, after adapter and endpoint
projection are otherwise correct, the shared arithmetic itself exceeds one
output-code LSB. A representation-only intrinsic or target adapter may instead
express a real ABI or legality constraint while preserving shared arithmetic.
For X-oriented work, exact target arithmetic is target-owned and
performance-relevant mapping decisions use their normal Voyager scheduler,
mapper, storage, or allocation owners at a full-control injection point. Then
define executable and FakeTensor behavior, recognize or
create the intrinsic in a target rewrite, and teach mapping and serialization
only what the target requires. Test generic-target no-op behavior, target
rewrite idempotence, forward equivalence, and emitted collateral.

### Propagate quantization through data movement

Define semantic transfer rules for identity, reshape, permute, slice, split,
chunk, concat, and fan-out. Transform payload and scale components together.
Insert explicit requantization when domains are incompatible. Do not add a new
ad hoc upstream search to every consumer.

### Support QAT

Keep deploy arithmetic stateless and share it between training simulation and
converted execution within its proper semantic owner; gated target-specific
arithmetic remains target-owned rather than generic compiler behavior. Put
observer scheduling, scale buffer updates, enable/disable controls, and STE
only in fake-quant modules. Test forward equivalence with converted execution,
gradient passage, state-dict round trips, observer freeze, and repeated
invocation.

## Current VSQ example

The current reference slice uses:

```text
int4,qs=vsq,coarse=int8,bs=32,scale=uint8
```

It composes per-tensor coarse quantization with fine MX block quantization. The
atomic converted operation returns coarse scale, fine scale, and fine payload.
Conversion currently multiplies coarse and fine scales and reuses MX compute
operators. Treat immediate GEMM-consumer support as a bounded vertical slice,
not proof that encoded VSQ values propagate through arbitrary data movement.

Keep AoV model loading, PTQ orchestration, and network-level I4/I8 policy in
the exact-pair adapter
`test/utils/models/yolo5_face_nv_magnet_alpha.py`. Extract shared workload
helpers only after multiple target adapters demonstrate real reuse. Keep
Magnet Alpha fixed-point requantization separate from generic VSQ fake and
reference behavior.

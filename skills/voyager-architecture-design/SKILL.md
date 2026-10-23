---
name: voyager-architecture-design
description: Plan, implement, or review Voyager compiler architecture with minimal changes, explicit layer ownership, target separation, and shared semantic reuse. Use for quantization and QAT design, hierarchical or encoded values, PT2E conversion, custom operators, graph transforms, lowering, tiling, memory mapping, code generation, target-specific behavior, and deciding which Voyager file should own a change.
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Voyager Architecture Design

Keep one semantic source of truth while preserving the boundaries between
model policy, generic compiler behavior, and architecture-specific lowering.

Read [references/compiler-map.md](references/compiler-map.md) before assigning
file ownership or proposing a cross-layer change.

## Respect R/W/X adoption boundaries

Read `docs/backend-adoption/adoption-levels.md` for the normative injection-
point and R/W/X definitions; do not recreate them in a design proposal. R and W
may use a partial- or full-control point. X requires a full-control point, the
same-scope W proof, zero-tolerance target output against the frozen
workload-semantic reference, and provenance for any packaged expected stream.
A schedule, allocation, exact operation, or X-capable ABI is only a subcheck.

During W, begin with executable shared semantics such as `conv2d_mx`. Keep the
shared operator when the target path meets `rtol=0`, `atol=1` output-code LSB;
add target-specific semantic lowering only after evidence attributes a failure
of that boundary to shared semantics rather than the adapter or runtime.

Do not confuse semantic lowering with representation or legality work. A target
adapter, packing rule, graph-visible tile, or semantic-preserving decomposition
may be required to execute the shared arithmetic without redefining it.

Full-control work belongs to the normal scheduler, mapper, storage, allocation,
and target-lowering owners. Tiling and allocation may be evidence for a target,
but are not a universal checklist. A complete X proof subsumes W; never create
an additive `W + X` status.

## Define the contract first

Write the smallest explicit contract before editing code:

- logical input and output values;
- physical tensors or encoded components;
- scale, axis, block, dtype, rounding, and saturation semantics;
- calibration versus runtime state;
- supported consumers and data-movement behavior;
- generic reference behavior versus target hardware behavior;
- required fake-quant, converted, transformed, and compiled equivalence.

Do not use metadata as the only storage for a numerical value needed by later
passes. Make runtime scales and encoded components explicit FX values.

## Assign one owner per responsibility

Use these ownership rules:

- Keep model-specific layer lists and mixed-precision policy in model helpers.
- Keep configuration parsing and validation declarative.
- Keep observer state, fake dequantization, and straight-through gradients in
  fake-quant modules.
- Keep target-neutral/reference stateless operator semantics in shared
  decomposed/custom operators.
- Keep gated target-specific W correction and X exact arithmetic in
  target-owned lowering or operators.
- Keep fake-quant replacement and encoded graph construction in PT2E
  conversion.
- Keep semantic graph canonicalization in transform passes.
- Keep architecture selection in target dispatch, not model-name branches.
- Keep final target FX as the semantic instruction carrier. Bind existing
  classification after required late target rewrites; do not add a shared
  execution IR merely to coordinate target files.
- Insert required physical movers before execution order and storage planning;
  keep placement and addresses with mapping and the selected allocator.
- Keep serialization canonical. Put deterministic external route, fusion,
  descriptor, and encoding projection in target-owned conversion tooling.
- Treat the bufferized and non-bufferized backends as separate consumers that
  need explicit support and tests.

## Minimize without creating shortcuts

Prefer the smallest vertical slice that proves the semantic contract:

1. parse and validate one configuration;
2. simulate it in fake quantization;
3. convert it to explicit operators and values;
4. execute one Conv, Linear, or Matmul reference path;
5. verify fake versus converted behavior;
6. extend transform, mapping, and codegen only when an attributed semantic gap
   or the downstream representation requires them.

Reuse an existing compute operator when its contract naturally accepts the new
payload and scale operands. During W, create a target-specific semantic operator
only after correct adapter and endpoint projection attributes a greater-than-
one-LSB failure to the shared arithmetic. A materially different output or ABI
representation may instead require a representation-only target operation or
adapter projection that preserves the shared arithmetic. Do not overload an
existing operator with target-only semantics.

Avoid adding a general framework before one complete concrete case works, but
do not encode model names, fixed hierarchy depth, or consumer-specific graph
walks into generic code.

## Share semantic functions

Place pure, stateless math at the lowest neutral layer and call it from both
training simulation and converted execution. Examples include codebook mapping,
scale calculation, block expansion, rounding, clamping, and reconstruction.

Keep wrappers distinct when their responsibilities differ:

- fake quantization owns mutable observer buffers and STE backward behavior;
- converted operators receive finalized qparams and expose encoded components;
- target semantic lowering converts generic reference semantics into
  hardware-specific integer or fixed-point operations only at the active
  stage's gate: an attributed shared-arithmetic greater-than-one-LSB W gap or X
  exactness.

Add an equivalence test whenever two paths claim the same forward semantics.
Test gradients separately for QAT.

## Preserve architecture separation

Maintain up to two conceptual IR levels when target-sensitive features require
them; W may remain entirely on executable shared semantics plus the native
injection stack:

```text
reference quantized IR
  explicit payloads and scales, executable with PyTorch

target IR
  placement, packing, DMA, fixed-point multiplier/shift, and target intrinsics
```

Do not leak Magnet or Alpha packing, Q8.8, multiplier width, address, or copy
rules into generic fake quantization or qspec parsing. Parameterize target
arithmetic and lower it in a target-owned pass.

Run semantic domain legalization before padding, tiling, fusion, or placement
removes easy access to graph-edge meaning. Run target pattern recognition before
generic rewrites destroy the pattern, then rerun shape propagation.

## Handle encoded and hierarchical values

Do not assume one logical quantized value is one tensor. Track payload and all
required scale/codebook components explicitly. For reshape, permute, slice,
split, concat, add, and fan-out, define a domain-transfer rule or insert an
explicit requantization.

Avoid rediscovering relationships with repeated local graph traversals. Add a
central transfer/legalization mechanism once encoded values cross more than an
immediate compute consumer.

For hierarchy, describe levels as data rather than adding fields named for
every future scheme. Keep each level's granularity, axes, block size, dtype,
and calibration mode explicit. Permit named profiles to construct the general
representation.

## Implement and review safely

1. Read the complete repository instructions and inspect `git status`.
2. Trace the existing end-to-end path before editing.
3. State the owner files, invariants, unsupported cases, and validation plan.
4. Preserve user and other-agent work in the dirty tree.
5. Make the smallest coherent patch with one source for shared defaults.
6. Run syntax/format checks and focused tests.
7. Run fake-versus-converted and pre/post-transform equivalence as applicable.
8. Run a compile/serialization test only when downstream representation changes.
9. Document intentional limitations; do not silently accept unsupported paths.

Reject a design when it:

- parses hierarchy but loses its structure before consumers;
- duplicates rounding or scale math between fake and converted paths;
- derives target fixed-point data in the serializer;
- serializes an external runtime taxonomy as authoritative compiler semantics;
- keys generic behavior on a model or module name;
- stores required numerical qparams only in node metadata;
- relies on a graph walk that breaks after fusion or data movement;
- updates only the non-bufferized path while claiming universal backend support;
- tests only node presence without forward equivalence.

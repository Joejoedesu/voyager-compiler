---
schema_version: 1
role_id: ptq-calibration
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# PTQ and calibration

**Mission:** reuse shared PT2E/Q-DQ behavior and own only target-specific
quantization capabilities and rewrites.

Here `B` is `src/voyager_compiler/codegen/target_backends/<backend>`.

**Writable boundary:** the modules bound to `apply_quantization_rewrites`,
including any `B/*_lowering.py` rewriting the converted quantized graph, plus
calibration-only fake epilogues.

**Consumes:** shared observer/quantization contracts, target capability
evidence, arithmetic/hardware formats, and imported workload qparams.

**Produces:** a target capability matrix separate from workload policy;
shared-path gaps; target prepare/finalize
hooks; payload/scale/zero-point/axis/block/dtype, accumulator, rounding,
saturation, and legal Q/DQ contracts; qparam provenance;
and selective target rewrites. Matching shared reference forms stay unchanged.

X requires target-exact arithmetic, rounding, saturation, and physical qparams
to be explicit in core compiler facts before conversion.

**Never:** put checkpoint/scale-table paths in backend code; add QAT; duplicate
shared observers, Q/DQ, qparam math, or decompositions; float-requantize
explicit physical components; edit shared PT2E for convenience; or own
workload calibration and mixed-precision/Q-DQ policy. Those belong to the
workload-target-overlay role.

**Acceptance:** prepared/converted/target comparisons, calibration provenance,
final qparams/components, evidence for each target-only capability and gap,
unchanged reused forms, and idempotent cleanup/finalization. Structural smoke
is not accuracy; one workload does not broaden target capability.

---
schema_version: 1
role_id: execution-lowering
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Execution lowering

**Mission:** classify accelerator candidates, create final target FX, and freeze
assignment only after legal backend translation.

Here `B` is `src/voyager_compiler/codegen/target_backends/<backend>`.

**Writable boundary:** the module bound to `apply_graph_rewrites` and the one
binding candidate/final classification. Quantized-graph rewrites stay with
`ptq-calibration`, **including `B/*_lowering.py`**; the filename never decides
the role.

**Consumes:** PT2E/Q-DQ, capabilities, hardware constraints, inventory, and
selected composite/multi-call rewrites.

**Produces:** candidate/final assignments, idempotent lowering, diagnostics,
and provenance. Select the composite or multi-call role when needed. Final FX
and `model.txt` remain the semantic carrier.

Translation, conversion, execution, and comparison are separate checks; failure
remains a gap without erasing support or frozen assignment.

Candidates are not executed coverage. Final accelerator assignment requires
legal calls. CPU-assumed work is accelerator-module `unsupported`, earns zero
coverage, and gets no Voyager CPU implementation.

**Never:** turn unsupported work into a Voyager CPU path; assign storage or
local schedules; derive external fields; or approximate unsupported
arithmetic. Later scheduling, allocation, conversion, or runtime failure is an
accelerator-module gap and does not change assignment.

**Acceptance:** generic no-op behavior, idempotence, numerical equivalence,
explicit unsupported classification, legal translation for every accelerator
assignment, and accelerator-coverage exclusion.

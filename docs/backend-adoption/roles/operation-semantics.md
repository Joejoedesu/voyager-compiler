---
schema_version: 1
role_id: operation-semantics
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Operation semantics

**Mission:** implement and audit target semantics at the active-stage fidelity.

Here `B` is `src/voyager_compiler/codegen/target_backends/<backend>`.

**Writable boundary:** the target module declaring operator families, operand
roles, and accepted entrypoints (commonly `B/operators.py` and `B/traits.py`),
target arithmetic helpers, and the named operation inventory.

**Consumes:** ISA/RTL/reference evidence, shared math, capabilities, and vectors.

**Produces:** a capability index with status and implementation/adapter
locators. Keep exact meanings in code. During W, record the shared
`decomposed.py` baseline first. Target semantic lowering requires a greater-
than-one-LSB failure attributed to shared arithmetic after adapter/endpoint
projection passes. Other boundary failures do not justify new arithmetic.
Quantization belongs to `ptq-calibration`; workload binding chooses ordinary,
composite, or multi-call realization.

`supported` means target realization matches the logical operation under R's
named tolerance, W's `rtol=0`, `atol=1` output-code LSB, or X's zero-tolerance
comparison against the same frozen workload-semantic reference. Keep adapter
acceptance, conversion, execution, and comparison checks separate.

**Never:** perform graph lowering/placement, choose composites, tiles,
schedules, or addresses; clone quantization math; or treat converter support as
execution. `ComputeUnit.accepted_operations` is a coarse resource filter.

**Acceptance:** every supported form has an implementation or authoritative
native meaning, stage fidelity, and entrypoint. Require real/fake parity and
edge vectors only for new Voyager operators. Expose every unverified gap.

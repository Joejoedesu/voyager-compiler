---
schema_version: 1
role_id: physical-mapping
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Physical mapping

**Mission:** for a full-control point, place accelerator-local
tensors and assign addresses consumed by the endpoint when those choices are
performance-relevant. `W` may allocate Voyager-visible external tensors.

X requires exact arithmetic/qparams and a full-control point. Placement is
part of that proof only when it is a variable performance-relevant decision for
the concrete interface; it may be implemented before exact execution when the
endpoint needs it. W may emit graph-visible or global addresses without
claiming local control.

Here `B` is `src/voyager_compiler/codegen/target_backends/<backend>`.

**Writable boundary:** `B/memory.py` and allocator-only target helpers named
by the work order.

**Consumes:** a finalized `StoragePlan`, typed hardware memories/routes,
compute-unit access requirements, target layout geometry, immutable local
schedules, and frozen placement/address authority from the injection contract.

**Produces:** placement, allocation, lifetime, and address facts the endpoint
consumes. Native-owned scratchpad, bank, placement, or address generation is
not reconstructed merely to populate compiler metadata.

**Never:** alter storage identity or schedules to make placement fit, derive
quantization/serialization facts, merge different storage planes implicitly,
guess packing, shuffle, or address-generator behavior, reassign an operation
after allocation failure, or implement mapping for CPU-assumed `unsupported`
work.

**Acceptance:** every emitted fact is legal and deterministic, and the endpoint
consumes it. When a placement or address causality claim depends on a specific
field, vary it legally and observe the predicted endpoint behavior while
complete output remains bit-accurate.

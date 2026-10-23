---
schema_version: 1
role_id: storage-contracts
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Storage contracts

**Mission:** express target physical values/movements in `StoragePlan` before
address assignment.

Here `B` is `src/voyager_compiler/codegen/target_backends/<backend>`.

**Writable boundary:** the target module that owns storage relations, copies,
and liveness, plus named storage helpers. Layout varies by backend: it may be a
dedicated module, the storage-relation parts of `B/graph.py`, or a typed plan
record beside the allocator. Own the concern, not a filename.

**Consumes:** final target FX, quantized components, graph-visible multi-call
values when present, storage identities, operand roles, and memory
capabilities.

**Produces:** materialization/alias/view/copy/direct-write relations, packed
components, physical movers, graph-visible call inputs and outputs, and
liveness inputs for facts Voyager owns. Storage owns size, layout, identity,
relations, and liveness; allocation owns placement and address.

Insert required movers before execution order and storage finalization while
they remain explicit target FX operations.

**Never:** assign addresses; mutate graph tiles or local schedules; infer
native scratchpad residency; equate numerical equality with physical identity;
reconstruct qparams; or invent target-internal movement. Unknown required
movement is a gap.

**Acceptance:** complete multi-output identity/component coverage, legal
relations, deterministic plans, and endpoint evidence for Voyager-inserted
graph-visible DRAM movers. Voyager-owned accelerator-local scratchpad or
residency may be evidence for a full-control point when it materially affects
mapping cost; it is X-oriented work, not an achieved X claim by itself.

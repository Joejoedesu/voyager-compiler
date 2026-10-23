---
schema_version: 1
role_id: serialization
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Serialization

**Mission:** validate finalized compiler decisions and emit canonical target
wire metadata without repair or external-schema projection.

Here `B` is `src/voyager_compiler/codegen/target_backends/<backend>`.

**Writable boundary:** `B/serialization.py`. Generic protobuf changes require
`shared-contract`.

**Consumes:** final target FX, artifact references, hardware, canonical wire
requirements, ordinary R/W global storage/address facts, and X-claimed
target-internal schedule/storage/address facts.

**Produces:** canonical `model.txt`, target artifact declarations,
deterministic dump metadata, and round-trip evidence. A composite serializes as
`fused_op.op_list`; an explicit multi-call rewrite serializes as normal
operations and result construction.

`model.txt` holds typed compiler facts, not arbitrary runtime metadata. Use
`StoragePlan` for resident relations, `PhysicalPayloadRef` for realized
placement, serialized artifacts for nonresident encodings, and supplemental
artifacts for rich metadata.

**Never:** choose assignment; form or repair composites, schedules, graph tiles,
aliases, addresses, layouts, or qparams; substitute logical tensors for
required physical facts; or emit runtime route, disposition, projected owner,
descriptor, epilogue, skip, or encoding policy.

Under X, serialization must receive target-exact arithmetic and physical
qparams from core lowering. It never derives them for the converter.

**Acceptance:** missing/ambiguous facts fail; each physical component is
emitted once; deterministic parse/round trip succeeds; every field traces to
one prior compiler source.

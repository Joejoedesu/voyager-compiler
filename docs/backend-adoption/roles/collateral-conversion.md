---
schema_version: 1
role_id: collateral-conversion
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Collateral conversion

**Mission:** project finalized compiler artifacts into target collateral and,
when contracted, integrate the endpoint.

**Writable boundary:** `conversion/<backend>/**` and its tests.
Launch/readback glue stays here; serialization belongs to `serialization`;
`backend-assembly` only wires.

**Consumes:** final `model.txt`, its referenced compiler artifacts, the frozen
external contract, and any contracted X facts Voyager owns.

**Produces:** deterministic encoding, provenance, fail-closed checks, and
contracted endpoint glue. Emitted descriptors and observed launches remain
distinct evidence.

The [execution contract](../contracts/execution.md#injection-contract) defines
the frozen `injection_contract`; target regressions freeze instances.

**Never:** form composites; change semantics; invent or repair computation,
qparams, graph tiles, schedules, storage, or addresses; write other compiler
files; or equate conversion with runtime success. Under X, bit-accurate
arithmetic and physical qparams already exist in canonical artifacts.

Converter-side capture rules may classify an already-finalized operation or
member sequence and must validate the exact endpoint ABI. They must not create
graph grouping or serve as the target's semantic hardware pipeline.

**Acceptance:** every required semantic/X fact traces to one compiler source;
ambiguity fails, tests pass, every Voyager output is compared, and any retained
stock operation uses the same endpoint contract as the Voyager operation.

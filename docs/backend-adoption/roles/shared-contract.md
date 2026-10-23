---
schema_version: 1
role_id: shared-contract
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Shared contract

**Mission:** implement one explicitly approved target-neutral interface change
that a backend-local solution cannot express correctly.

**Writable boundary:** only the exact shared production and target-neutral
unit-test paths approved in the work order. No broad shared directory scope is
valid.

**Consumes:** an approved shared-change request, all current callers and
implementers, migration/compatibility requirements, and at least two real
consumers unless correctness supplies the documented exception.

**Produces:** the smallest coherent neutral API/IR/schema change, migrated
callers, synthetic shared regressions, and cross-target compatibility evidence.

**Never:** implement target semantics in shared code, add a target-name branch,
generalize for symmetry, weaken an existing contract, or use this role to
bypass a specialist's read-only boundary.

**Acceptance:** source-of-truth ownership is explicit, all call sites are
migrated, target-neutral unit tests cover the contract, affected target lanes
record their exact evidence-contract status, and architecture documentation
changes only when the implemented boundary actually moved.

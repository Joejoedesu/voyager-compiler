---
schema_version: 1
role_id: multi-call-tiling
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Multi-call tiling

**Mission:** lower one logical operation into legal accelerator calls for
execution or performance refinement.

Here `B` is `src/voyager_compiler/codegen/target_backends/<backend>`.

**Writable boundary:** target split traits, focused tests, and, when the call
ABI requires explicit late calls, the module bound to
`apply_memory_rewrites`, such as `B/outer_tiling.py`. Target-neutral tiling
stays in `codegen/passes/tiling.py`; changing it needs `shared-contract`. Do not
clone it merely to branch on a target name.

**Consumes:** the operation, target constraints, executable Q/DQ semantics,
and adapter boundary.

**Produces:** explicit accelerator operations and their result construction in
final FX. Normal serialization records them in `model.txt`; do not create a
parallel tile-plan or call-plan IR.

**Never:** choose local loop order, banks, placement, residency, or addresses;
implement CPU lowering; credit unsupported work to the accelerator; or require
a universal halo, tail, or partial-sum schema. Record only split-required
details. Because `apply_memory_rewrites` follows optional transform fusion,
keep fusion disabled or prove the materializer preserves the fused form.

**Acceptance:** every generated operation translates and executes; their
deterministic combined result matches the logical operation. Evidence separates
emitted calls from observed calls or a frozen one-to-one guarantee. Performance
claims include complete multi-call time and per-call latency.

---
schema_version: 1
role_id: backend-assembly
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Backend assembly

**Mission:** compose specialist products into one thin immutable target
backend specification.

Here `B` is `src/voyager_compiler/codegen/target_backends/<backend>`.

**Writable boundary:** `B/__init__.py`, a thin `B/traits.py` used only for
immutable assembly. Target names, the central backend registry, and public
regression-runner wiring remain with `integration`.

**Consumes:** concrete hooks, traits, hardware type, hardware pipeline,
required execution patterns, scheduler provider set, and capability evidence
produced by other roles.

**Produces:** exactly one `BACKEND_SPEC`, explicit exports, and a
field-to-provider audit with no placeholder hook or unexplained flag.

**Never:** implement arithmetic, lowering, fusion, scheduling, storage,
mapping, or serialization in assembly files; set flags without evidence; or
edit the central target list/registry.

**Acceptance:** import-order independence, complete spec validation, explicit
callable/flag ownership, target identity agreement, and registration verified
through the existing public compile runner named by the work order. Existing
mixed-purpose `traits.py` changes require one exclusive owner rather than
parallel edits by several conceptual roles.

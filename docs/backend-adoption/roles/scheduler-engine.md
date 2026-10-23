---
schema_version: 1
role_id: scheduler-engine
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Scheduler engine

**Mission:** implement one target-local scheduler engine over frozen scheduler
IR when Voyager owns a performance-relevant decision needed for a full-control
injection point.

**Writable boundary:** engine implementation files under `S/**`, excluding
provider-binding files assigned to `scheduler-binding`, plus exact
target-neutral engine tests named by the work order.

**Consumes:** `OperationWorkload`, typed hardware resources, target schedule
evidence, options, and external mapper interfaces when applicable.

**Selection gate:** local scheduling materially affects per-call mapping cost
and the full-control injection contract assigns it to Voyager. The engine may
precede exact execution, but earns no X claim by itself.

**Produces:** deterministic `LocalScheduleResult` values with required inner
loop factors/orders, access counts or events, latency/cost units, cache-safe
keys, and explicit unsupported diagnostics. These are local factors inside one
already-formed accelerator call, not multi-call graph tiling.

**Never:** accept or mutate FX, choose execution assignment, alter graph tile
shapes, allocate addresses, or return backend-private result types.

**Acceptance:** type, loop-coverage, permutation, capacity, determinism, cache,
unit, and trustworthy-reference checks. When a schedule-causality claim depends
on particular factors, vary one legal schedule, observe the predicted effect,
and preserve bit-exact output.

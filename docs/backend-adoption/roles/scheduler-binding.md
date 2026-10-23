---
schema_version: 1
role_id: scheduler-binding
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Scheduler binding

**Mission:** project target semantics/hardware into scheduler IR and bind
eligible local engines without target policy in shared dispatch.

Here `B` is `src/voyager_compiler/codegen/target_backends/<backend>` and `S`
is `src/scheduler/<engine>`.

**Writable boundary:** the module bound to `scheduler_providers` and
`classify_scheduler_operation`, plus the provider package `S/`. Algorithms
belong to `scheduler-engine`; shared fields in `scheduler/types.py` need a
`shared-contract` order.

**Consumes:** final classifications, hardware, local-loop authority, fidelity,
engines, and scheduler contracts, after graph rewrites.

**Produces:** projectors, provider adapters, one immutable
`SchedulerProviderSet`, eligibility/default policy, and canonical local factors.

**Selection gate:** local scheduling materially affects per-call mapping cost
and the full-control injection contract assigns it to Voyager. Binding may
precede exact execution, but earns no X claim by itself. Never bind
endpoint-owned local execution at a partial-control point.

**Never:** branch shared dispatch on target IDs; mutate FX/allocation; hide
workload policy; schedule accelerator-unsupported forms; build CPU scheduling
for assumed external execution; turn failure into reassignment or CPU
implementation work; build an engine for helper-selected execution; or claim
exactness when ignored.

**Acceptance:** dispatch/default/error coverage, frozen `OperationWorkload`,
canonical local results, identity agreement, and public compilation for each
provider. When a schedule-causality claim depends on generated factors, require
a legal control variation, causal endpoint consumption, and unchanged
bit-accurate output.

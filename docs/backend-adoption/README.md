<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Voyager backend and workload adoption

This directory coordinates PTQ inference adoption. It adds no compiler path or
agent IR. Workers produce only the ordinary compiler or evidence artifacts
owned by the selected role and lifecycle mode.

New to adoption? Read the [glossary](glossary.md) first — it summarizes `R`,
`W`, `X` and the surrounding vocabulary with worked examples. Then read the
[adoption model](adoption-levels.md), which is the single normative owner of
the lifecycle, the compilation-ownership table, and the invariants that hold
across phases.

This directory adds no rules of its own. It routes.

## Route by responsibility

- Coordinators use the [coordinator contract](contracts/coordinator.md) and
  [role registry](roles/README.md).
- Specialists use one work order, one selected role, and the
  [execution contract](contracts/execution.md).
- Regression owners use the [evidence contract](contracts/evidence.md) and the
  affected target contract.
- Performance campaigns additionally use the
  [performance contract](contracts/performance.md).
- Authors of isolated run instructions start from the
  [R/W/X adoption guide template](run-guide-template.md) and fill its
  repository and environment roots, target injection point, requested
  operations and workload, and X full-control scope.

Open templates only when producing or validating those artifacts. The
repo-local skills in `AGENTS.md` perform the same routing and load less.

## Reading order

A specialist doing one bounded task needs exactly three documents: the
[glossary](glossary.md) once, their work order, and the one role contract that
work order names. The [execution contract](contracts/execution.md) covers the
rest. Do not read the full catalog first.

Ownership boundaries, phase definitions, and evidence rules are **not**
restated here. They live at exactly one owner each:

| Question | Owner |
| --- | --- |
| What do injection point, R, W, X, and their invariants mean? | [adoption model](adoption-levels.md) |
| Where are concise examples? | [glossary](glossary.md) |
| Which files may I write? | [role registry](roles/README.md) |
| How do I plan and split work? | [coordinator](contracts/coordinator.md) |
| How do I execute one work order? | [execution](contracts/execution.md) |
| What counts as proof? | [evidence](contracts/evidence.md) |
| How do I claim coverage or speed? | [performance](contracts/performance.md) |

Roles describe file ownership, not mandatory phases or mandatory files.
Existing targets may keep adjacent responsibilities together; give a shared
file one writer at a time rather than refactoring merely to match the role
catalog.

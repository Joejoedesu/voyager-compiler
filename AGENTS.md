<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Voyager compiler agent instructions

Read `CLAUDE.md` and inspect `git status --short` before editing. Preserve
unrelated work and dirty submodules.

Do not preload every document under `docs/`. Use `docs/README.md` as a router
when ownership or validation is unclear. Read the complete production files
that own the selected behavior and use `rg` to verify current symbols.

Stages use `R -> W -> X`, with same-scope `X => W => R`. R registers a target
and injection point. W binds a workload through a partial- or full-control
point at `rtol=0`, `atol=1` output-code LSB. X needs that W proof, exact output,
oracle provenance, and a full-control point where Voyager determines the
performance-relevant mapping. It is never `W + X`. Definitions:
`docs/backend-adoption/adoption-levels.md`; examples:
`docs/backend-adoption/glossary.md`.

For workload or backend adoption, use the repo-local skills:

- `$voyager-adopt-backend` selects `register_target`, `bind_workload_target`,
  or `deepen_inner_control`, then selects roles and work orders;
- `$voyager-execute-adoption-work-order` loads one role contract and executes
  one bounded assignment; and
- `$voyager-gate-regressions` loads the affected target contracts and judges
  integration evidence.

Those skills define minimal context; do not preload the adoption catalog.

Permanent tests under `test/unit_tests/` are target-neutral. Target behavior
belongs in end-to-end compile lanes and the selected target's regression
contract. A temporary target diagnostic is not durable acceptance evidence.

Before claiming code-review readiness, run the applicable shared checks and
every affected target/workload contract. A concrete dependency makes a
required lane `blocked`; do not relabel it or replace it with a weaker smoke
test.

Update architecture documentation when a compiler stage, owner, persistent
representation, target, or scheduler registration changes.

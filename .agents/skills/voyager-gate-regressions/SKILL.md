---
name: voyager-gate-regressions
description: Gate Voyager registration, direct-first semantic workload-target binding, qualified performance estimates, and bit-accurate full-control claims.
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Gate Voyager Adoption

Judge evidence; do not implement. Read:

1. `AGENTS.md` and `CLAUDE.md`;
2. `docs/backend-adoption/glossary.md` — vocabulary and examples;
3. `docs/backend-adoption/contracts/evidence.md` — the states you assign;
4. the work order and its handoff; and
5. the target contract, plus the performance contract when needed.

Verify repository, environment, hardware, injection point, workload, and
tolerance. Load no unrelated targets. Rerun scope verification, then apply only
the phase's gates.

## Registration

Require a default-derived `HardwareConfig`, overrides, inventories, a Voyager
endpoint run, and an injection contract naming control class, evidence, and
consumed fields. Stock runs are optional provenance.
Every requested kind needs ordinary lowering and R-tolerance comparison before
special decomposition.

## Workload binding

Require exact inputs, PTQ/qscheme candidates, assignments, final FX,
`model.txt`, translation, and W comparison at `rtol=0`, `atol=1` output-code
LSB per accelerator boundary.

Require the shared executable operator and baseline comparison. Target semantic
lowering requires a greater-than-one-LSB failure attributed to shared
arithmetic after adapter and endpoint projection are otherwise correct. Record
representation, adapter, endpoint, and runtime failures separately; none
justifies new target arithmetic. Native software may retain target-internal
control at W.

Require one natural instance of every kind, then every real supported signature.
Decomposition follows a recorded direct failure; serialization precedes fusion,
grouping, or multi-call refinement.

Require trace or endpoint guarantees for executed coverage; candidates never
count. For composites and graph-visible multi-call tiling, verify
complete-boundary comparison, every generated call, and the combined result.

Reject support downgrades after later failures or CPU reassignment after an
accelerator-module gap.

## Performance, when claimed

Rank qscheme candidates by checked accelerator coverage, then estimated total time
at equal coverage. Estimated calls feed time; include per-call launch latency and
conversion cost.
State assumptions; never call coarse W estimates exact cycles. At a
partial-control point, keep hidden native mapping costs unresolved or measured
as a black box and attribute native choices to the native stack.
Utilization needs a qualified numerator, peak, interval, and provenance.

## X bit accuracy and full control

Credit X only when same-scope W, a full-control point, endpoint consumption of
Voyager's mapping, exact target output against the frozen semantic reference,
and packaged-stream provenance pass.
Exact output through a partial-control point remains at most W. Control fields
are evidence, not a universal checklist. Complete X subsumes W; never report
`W + X`.

## Report

A smoke test never replaces a blocked lane. Report support, separate checks,
calls, estimates, gaps, the current stage-gate result, and the strongest
completed stage. Update owned status; only a passing gate
advances the achieved stage.

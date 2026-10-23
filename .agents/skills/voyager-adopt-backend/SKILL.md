---
name: voyager-adopt-backend
description: Coordinate Voyager target registration, direct-first semantic workload-target binding, or bit-accurate full-control workload binding.
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Coordinate Voyager Adoption

Read from the repository root:

1. `AGENTS.md` and `CLAUDE.md`;
2. `docs/backend-adoption/glossary.md` — vocabulary and worked examples, once;
3. `docs/backend-adoption/adoption-levels.md` — only the phase you selected;
4. `docs/backend-adoption/contracts/coordinator.md` — your duties; then
5. the selected role contract and the affected target contract.

Reference these rule owners; never copy them into a work order.

## Select one mode

Choose `register_target` (R), `bind_workload_target` (W), or
`deepen_inner_control` (X-oriented work). A mode names an action, not an
achieved stage. Registration precedes workload mutation. Adoption is
PTQ-only. Each work order names one mode and one owned output.

## Coordinate R

Assign the default-derived `HardwareConfig`, its permitted overrides, and its
configuration diff to one owner; never alter hardware to fit an operation.
Freeze the injection point, classify it as partial-control or full-control with
target evidence, and record the exact fields it consumes. Close the
one-operation anchor before authorizing any workload-specific policy.

## Coordinate W

Freeze the workload, inputs, target, injection point and control class,
reference, tolerance, coverage
denominator, and estimation boundary before candidates run. Name each
quantization and calibration candidate.

Start from shared executable semantics such as `conv2d_mx` and use
the existing target-native software stack at the injection point. Add
target-specific semantic lowering only when a greater-than-one-LSB failure is
attributed to shared arithmetic after adapter and endpoint projection are
otherwise correct. Representation or legality projection may still be
required, but must preserve the shared arithmetic. Keep adapter, endpoint, and
runtime failures at their own boundaries.

Sequence the work so direct execution is attempted first, and so serialization
happens before composite, grouping, or multi-call refinement. Reject a
decomposition whose projected call or collateral growth is unreasonable.

When refining performance, rank candidates by checked coverage, then estimated
total time at equal coverage; estimates are not cycles. W completion requires
coverage, translation, target execution, and the fixed one-LSB comparison.
Semantic lowering stays conditional on an attributed shared-arithmetic failure;
an estimate never completes W.

Create a combined pair adapter only for genuinely pair-specific policy; the
exact tuple remains the evidence unit.

## Coordinate X

Require a full-control point and prove the endpoint consumes Voyager's
performance-relevant mapping. Split exact-lowering and control orders when
ownership requires. Credit X only after same-scope W and exact target output
against the frozen semantic reference pass, with packaged-stream provenance.

## Close out

Route target-neutral gaps through a `shared-contract` order. Assign one
`integration` owner for registry, CLI, target regressions, and status. Invoke
`$voyager-gate-regressions` and record only checks that actually ran.

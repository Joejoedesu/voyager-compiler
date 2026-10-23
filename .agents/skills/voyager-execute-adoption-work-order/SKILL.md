---
name: voyager-execute-adoption-work-order
description: Execute one phase-scoped Voyager adoption work order while preserving compiler, target, evidence, and file-ownership boundaries.
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Execute an Adoption Work Order

Read from the repository root:

1. `AGENTS.md` and `CLAUDE.md`;
2. `docs/backend-adoption/glossary.md` — vocabulary and worked examples, once;
3. your work order and its workspace baseline;
4. the one role contract the work order names; and
5. `docs/backend-adoption/contracts/execution.md`.

Read `docs/backend-adoption/adoption-levels.md` only for the phase your work
order names. Do not preload other roles or targets.

Reject mismatched ownership, tolerance, environment, repositories, hardware,
injection point, or boundary. The work order bounds implementation.

## Implement one owned output

Produce only role-owned artifacts. Keep semantics with operators and lowering,
loops with schedulers, relations with `StoragePlan`, addresses with allocators,
canonical facts with serializers, and ABI projection with converters. In W,
reuse shared semantics. Target semantic lowering requires a greater-than-one-
LSB failure attributed to shared arithmetic after adapter/endpoint projection
passes. Target representation or legality work must preserve that arithmetic;
adapter, endpoint, and runtime failures stay at their boundaries. Converters
encode finalized facts; they never invent computation or qparams.

Do not add QAT, unsupported-operation CPU lowering, guessed hardware,
target-named shared branches, or target-specific shared tests. A target-neutral
gap needs a `shared-contract` order.

## Respect the phase you were given

For `R`, derive `HardwareConfig` from the target default and freeze the
injection point, control classification and evidence, and consumed fields
before lowering anything.

For `W`, work direct-first: every operation kind, then every real metadata
signature, before any composite or decomposition. Serialize before refining.
Start from shared operators such as `conv2d_mx`. Comparison uses `rtol=0`,
`atol=1` output-code LSB, not bit-exactness. You may reuse the injection point's
software stack, and detailed physical controls are not a W requirement.
At a partial-control point, hidden native decisions may remain; at a full-
control point, record Voyager's decisions, the endpoint fields, and consumption
evidence without promoting W to X. Exact output at a partial-control point
remains at most W.

For `X`, use a full-control point where Voyager determines every variable
performance-relevant mapping decision and the endpoint consumes the artifact.
Target output must match the frozen workload-semantic reference exactly, with
packaged-stream provenance. Neither exactness nor control alone earns X. A
complete same-scope X result subsumes W.

## Verify and hand off

Run the role checks and target lane. A smoke test never replaces a blocked lane.

Hand off final FX and `model.txt`, support and tolerance, each check
separately, calls, unsupported accounting, W estimates when performance
refinement is in scope, X exactness, workload-reference provenance, and
full-control evidence, gaps, exact commands, artifacts, and the strongest claim
the evidence actually supports.
Then run the scope verifier.

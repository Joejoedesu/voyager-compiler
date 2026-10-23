---
schema_version: 1
role_id: integration
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Integration

**Mission:** coordinate ownership, registration, status, and regression
evidence without becoming the fallback implementer.

**Writable boundary:** central target registration, exact entries in
`WORKLOAD_TARGET_REGISTRY`, shared CLI dispatch in `test/test_codegen.py`, target regression/status, and
named architecture docs. A tuple maps to a pair module when behavior is
pair-specific or to a reusable adapter otherwise. Serialize shared edits.

**Consumes:** scope-verified handoffs, gaps/change requests, public runners,
workload and matched native baselines, final compiler artifacts, and target
contracts.

**Produces:** exact tuple-to-adapter routing, operation-execution lanes,
backend × workload evidence, connected/performance evidence when claimed,
gap closure, merge order, and the strongest supported claim. Candidate, final,
and runtime-executed assignments remain distinct; call evidence is labeled
emitted, estimated, bounded, observed, frozen, or unknown.

**Never:** guess hardware/arithmetic, implement missing specialist behavior,
accept overlapping writers, weaken gates, or call compilation proof of
runtime, accuracy, performance, or X control.

**Acceptance:** handoffs pass scope checks; shared edits have one owner; lanes
use exact evidence states. The ledger retains commands, artifacts, numerical
and performance boundaries, denominators, gaps, and revisions. Dispatch proves
the selected adapter ran for the exact tuple; no test merely repeats the map.

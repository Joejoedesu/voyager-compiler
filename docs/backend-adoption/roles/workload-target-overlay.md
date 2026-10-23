---
schema_version: 1
role_id: workload-target-overlay
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Workload adapter and target overlay

**Mission:** own executable model, data, PTQ, calibration, and comparison
behavior. Keep shared behavior target-independent; keep inseparable policy in
one workload × target adapter.

**Writable boundary:** work-order-named paths under `examples/<workload>/` or
`test/utils/models/`, including justified `<workload>_<target>.py` helpers.
`integration` owns registry tuples, CLI routing, and regression status.

**Consumes:** model/checkpoint/data provenance, hardware, Voyager APIs,
capability evidence, and evaluation requirements.

**Produces:** deterministic loading, preprocessing, reference/PTQ/evaluation,
provenance, workload demand, native-baseline selection, and a typed result.
Pair adapters may also own pair-specific qspec, projection, calibration, and
comparison policy. They may request target work but do not implement it.

This role does not require a policy-only overlay or new file for every cell.
`yolo5_face_nv_magnet_alpha.py` remains combined because its policies are
inseparable. Extract a reusable helper only after two exact tuples consume it;
do not clone pair files or force extraction for one pair.

**Never:** own registry or CLI routing, construct hardware, implement target
lowering/scheduling/mapping/serialization, add QAT, infer modules from
filenames, substitute random data or zero-step calibration, or present smoke
as PTQ.

**Acceptance:** explicit inputs/outputs and provenance, deterministic
reference/PTQ, frozen baselines, exact tuple resolution, and separate evidence
per tuple. Reuse preserves results; test behavior, not registry text.

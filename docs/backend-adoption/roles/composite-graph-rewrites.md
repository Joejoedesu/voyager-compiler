---
schema_version: 1
role_id: composite-graph-rewrites
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Composite graph rewrites

**Mission:** declare target-supported semantic pipelines and, when activated,
group matched source operations in final target FX without changing semantics.

Here `B` is `src/voyager_compiler/codegen/target_backends/<backend>`.

**Writable boundary:** `B/hardware_pipeline.py`, any datapath-to-`OpMatcher`
projection, `execution_fusion_patterns`, predecessor-input traits,
`prepare_execution_fusion`, and focused tests.

**Consumes:** graph semantics, explicit Q/DQ domains, registered operation and
composite capabilities, and the target adapter boundary.

**Produces:** a deterministic `OpMatcher` inventory with sources, exclusions,
and activation status. Activated work also produces a deterministic `fused_op`
whose `op_list` retains members and required operands. A target may translate
it into one or more accelerator calls.

**Never:** use converter capture rules as the hardware pipeline; add another
composite representation; infer hardware fusion, performance, or call count
from grouping; drop semantics; include unsupported members; or activate before
later tiling and serialization/conversion accept the fused form.

Replacing a match with one target operation that erases its members is
`execution-lowering`, not this role.

**Acceptance:** modeling has source-backed stages, fail-closed predicates,
explicit exclusions, and accurate enabled status. Activation also proves
reference execution, legal target translation, later-consumer compatibility,
and unfused equivalence at `rtol=0`, `atol=1` output-code LSB. Every member is
accelerator-implemented or absorbed. Call evidence states its provenance.

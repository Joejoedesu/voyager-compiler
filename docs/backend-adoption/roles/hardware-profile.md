---
schema_version: 1
role_id: hardware-profile
---

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Hardware profile

**Mission:** translate authoritative architecture, RTL, configuration, or
measured facts into one typed `HardwareConfig` profile.

Here `B` is `src/voyager_compiler/codegen/target_backends/<backend>`.

**Writable boundary:** `B/hardware.py`.

**Consumes:** the hardware schema, backend evidence, and required resources.

**Produces:** typed operation, dtype, capacity, boundary, and measurement facts;
deterministic constructors; and source provenance. Add detailed memory facts
only for a named legality/performance observation or `X` claim. Recording a
resource does not transfer control to Voyager; consumed Voyager choices are
`X`. Utilization records peak, clock/unit meaning, and provenance.

R derives this profile from the target hardware repository's authoritative
default configuration. Record its revision, permitted overrides, the
default-to-derived diff, and proof that the external run configuration and
Voyager `HardwareConfig` describe the same machine. W and X consume that
frozen R profile unless a new R work order changes it.

**Never:** build a generic parser without a real common input format, guess an
unknown fact, encode loop schedules or workload shapes as hardware, or hide a
second target selector in scheduler options.

**Acceptance:** deterministic schema-valid construction, serialized round trip,
claim completeness, measurement provenance, and explicit required unknowns.

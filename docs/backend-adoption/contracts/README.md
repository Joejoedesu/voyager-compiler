<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Backend-adoption contracts

Load the common contract or contracts required by the active responsibility:

- [Coordinator](coordinator.md) for adoption planning and integration
  ownership;
- [Execution](execution.md) for one bounded implementation work order; or
- [Evidence](evidence.md) for independent regression gating and status claims;
  or
- [Performance](performance.md) for matched-boundary performance and mapping
  quality claims.

The `register_target`, `bind_workload_target`, and `deepen_inner_control`
work-order modes advance ordered stages defined by the
[adoption-stage contract](../adoption-levels.md). For one frozen claim scope,
the achieved-stage invariant is `X => W => R`; the modes are actions, not
additive status labels. W is an intermediate milestone toward X. A complete X
proof for the same frozen scope subsumes W and replaces a separate W-only
regression result. R and W may use either a partial- or full-control injection
point; X requires a full-control point, bit-accurate target output, and the
same-scope W proof including its semantic-reference boundary.

Role-specific ownership is separate under [roles](../roles/README.md).
Templates are data schemas and should be opened only when producing or
validating that artifact.

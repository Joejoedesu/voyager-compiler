<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Backend-adoption role registry

A role is a file-ownership contract, not a mandatory compiler phase. Select
only roles that own a concrete output.

| Role ID | Ownership |
| --- | --- |
| [workload-target-overlay](workload-target-overlay.md) | Reusable workload behavior and genuinely pair-specific policy |
| [hardware-profile](hardware-profile.md) | Target capabilities and resources needed by the selected adoption phase |
| [operation-semantics](operation-semantics.md) | Operation meanings, accepted entrypoints, and constraints |
| [execution-lowering](execution-lowering.md) | Candidate and final assignment plus final-FX lowering |
| [ptq-calibration](ptq-calibration.md) | Target quantization capabilities and rewrites |
| [composite-graph-rewrites](composite-graph-rewrites.md) | Composite graph rewrites |
| [multi-call-tiling](multi-call-tiling.md) | Rewrite one logical operation into several accelerator calls |
| [storage-contracts](storage-contracts.md) | Voyager-owned storage relations, copies, and liveness |
| [physical-mapping](physical-mapping.md) | Voyager-owned placement and addresses |
| [scheduler-engine](scheduler-engine.md) | Voyager-owned target-local scheduling or search |
| [scheduler-binding](scheduler-binding.md) | Target projection of a Voyager-owned schedule |
| [serialization](serialization.md) | Canonical `model.txt` and compiler artifacts |
| [collateral-conversion](collateral-conversion.md) | Target ABI validation, collateral projection, and optional endpoint glue |
| [backend-assembly](backend-assembly.md) | Thin immutable `TargetBackendSpec` assembly |
| [shared-contract](shared-contract.md) | One approved target-neutral interface change |
| [integration](integration.md) | Registration, pair routing, status, and regressions |

## How to select a role

Roles are not a pipeline and not a checklist. Pick the role that owns the file
you must change, and only that role.

- Role IDs name **concerns, not filenames**. Resolve the file by reading the
  backend's `__init__.py` spec bindings; layouts range from 3 to 21 modules.
- Select a role only when it owns a **concrete output file**. Do not add a
  scheduler, mapper, converter, or pair adapter to make the role list look
  symmetric.
- One file has one writer per mutation window. Serialize edits to legacy files
  that combine roles. Shared chokepoints require a `shared-contract` order.
- Select `scheduler-*`, `storage-contracts`, or `physical-mapping` only for
  decisions **Voyager** owns. Endpoint-owned local scheduling and memory
  control need no role at all.
- `B/graph.py` has three claimants; split it by what the rewrite produces.
  Grouping operations into a `fused_op` that keeps its members is
  `composite-graph-rewrites` via `B/traits.py`. Replacing a matched pattern with
  one new target operation, which keeps no members, is `execution-lowering`.
  Storage relations, copies, and movement are `storage-contracts`.
- Keep exact `(workload, target)` tuples as the routing, policy, measurement,
  and evidence unit. Create a combined pair adapter only when the behavior is
  genuinely inseparable.
- Put evidence, sources, and read-only context in the work order, not here.

The phase definitions, assignment states, composite and tiling rules, and the
semantic-support-versus-check distinction are **not** restated here. They are
owned by `docs/backend-adoption/adoption-levels.md`; the vocabulary is defined
with examples in `docs/backend-adoption/glossary.md`.

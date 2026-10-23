<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Target-specific integration contracts

This directory contains validation requirements that are concrete to one
backend. The [root agent instructions](../../AGENTS.md),
[shared development guide](../agent-development-guide.md), and
[evidence contract](../backend-adoption/contracts/evidence.md) describe how to
select evidence; they must not accumulate every backend × workload ×
scheduler lane.

Each target may add only the documents justified by a real interface:

- `regression.md` for detailed commands, inputs, policies, schedulers,
  numerical results, and maintenance lanes;
- `operations.yaml` for the compact stage-gate/check snapshot and named
  semantic capabilities;
- `injection-flow.md` whenever the endpoint is external to this repository. The
  frozen `injection_contract` records *what* the boundary is; this runbook
  records *how to reach it*: a boundary diagram naming who owns each hop, the
  pinned external-stack identity with hashes and a preflight check, the
  compile/convert/execute command chain using fresh output roots, the job
  submission procedure, acceptance markers, a failure-to-owning-boundary
  routing table, and explicit non-claims; and
- `ownership.md` only when a legacy file combines role responsibilities and the
  constraint list would crowd the README. The responsibility-to-owner mapping
  itself belongs in the target `README.md` by default.

The inventory is an evidence index, not another specification. It records the
stage currently being evaluated for an explicitly named target x workload
scope, its result, concise form summaries, endpoint ownership, available
observations, and locators for authoritative compiler and adapter behavior.
Only `regression_status.status: pass` makes the named stage achieved. Matching,
arithmetic, ABI fields, and validation algorithms stay in code and interface
contracts. A W or X workload result does not promote broad capability rows;
unknown facts stay `unverified`. Each inventory names its concrete injection
point, and the linked target interface records the partial- or full-control
classification and evidence. R and W may use either; X requires a full-control
point but is not earned by control evidence alone.

In an inventory, `supported` means a semantic implementation or authoritative
native meaning matches the logical form under the inventory's declared stage
tolerance, including any composite-only constraint. Record a named adapter/backend
entrypoint separately. Its acceptance, conversion, execution, and comparison
are checks; failure does not erase semantic support. Compiler recognition
without an established semantic match is `unverified`.

A runtime contract is not implied by successful compilation. If a target has
no named external-runtime contract, that boundary remains unverified. Never
reuse another backend's profiles, launch rules, pass markers, or tolerances by
analogy.

Before expanding a new backend to a representative workload, its contract must
name a target execution anchor:

1. the exact Voyager collateral injection contract, including encoding,
   layout, every field consumed by that endpoint, launch, completion, readback,
   and oracle;
2. one Voyager-generated `model.txt` operation or composite translated and
   executed through that endpoint; and
3. a complete output comparison and qualified accelerator-call evidence: an
   observed count or frozen one-to-one guarantee when available, otherwise an
   emitted or estimated count, justified bound, or `unknown` with provenance.

A separate stock target-native run is useful endpoint/configuration provenance
when available, but is not a second registration gate.

The normative injection-point and control-class meanings are in the
[adoption model](../backend-adoption/adoption-levels.md#injection-points-and-control);
the required recorded contents are in the
[execution contract](../backend-adoption/contracts/execution.md#injection-contract).
Target documents bind those contracts to one concrete endpoint and artifact.

An ISA or operator name is unverified until a scoped target realization matches
the logical semantics under the stage tolerance. Record a named adapter or
backend entrypoint, but report acceptance, conversion, execution, and
comparison as separate checks. An accelerator composite in final FX or
`model.txt` contains only implemented or correctly absorbed members and may
translate into one or more native calls.
Keep unsupported operations outside it as separate assignments assumed to run
on external CPU, with no Voyager CPU implementation. Compare the complete
accelerator-owned boundary and qualify call evidence as observed, guaranteed,
emitted, estimated, bounded, or unknown. Grouping does not imply hardware
fusion.

An initial operation-local execution anchor is sufficient for the
operation-execution milestone. Workload binding then separates candidate
assignment from final assignment: freeze final assignment only after the
target adapter or native stack translates final `model.txt` into legal
executable backend calls. At a partial-control point, target-local scheduling
and mapping may remain native-owned. At a full-control point, Voyager's
scheduler and mapper determine every variable decision that materially affects
per-call mapping performance. X additionally requires the same-scope W proof,
zero-tolerance target output against the frozen workload-semantic reference,
and provenance for any packaged expected stream. See
[adoption model, X](../backend-adoption/adoption-levels.md#x-bit-accurate-full-control-workload-binding).
Registration closes
only when its capability inventory, injection contract, and operation anchor
pass.
A connected workload or accelerator-resident cross-call state is a separate,
stronger contract. Any hybrid CPU/accelerator system runtime is external to
Voyager adoption and must not become CPU implementation work unless the user
explicitly requests that separate system-integration task.

Current target contracts:

- [Gemmini](gemmini/README.md)
- [Magnet Alpha](magnet-alpha/README.md)

## Current control and binding summary

This is a scope-qualified status summary, not a backend-wide capability score.
`Unverified` means the named gate is incomplete even when some of its subchecks
pass.

| Target and injection point | Control | Registration and workload status |
| --- | --- | --- |
| Gemmini public `_auto` helpers | partial-control | ResNet-18 achieved W for 30 independently executed final regions at `rtol=0`, `atol=1`, therefore also satisfying R for the same injection point; X cannot be claimed there. |
| Magnet Alpha compiler and converter | not classified | The maintained package validates compiler serialization and fail-closed collateral conversion. External target execution and R/W/X workload evidence are not included. |
| Generic compiler path | not classified | It is the compiler-native fallback, not a concrete target injection point, so it carries no R/W/X adoption claim. |

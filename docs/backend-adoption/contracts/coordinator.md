<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Adoption coordinator contract

Use this contract to plan and integrate backend adoption. The coordinator owns
scope, file ownership, ordering, and evidence acceptance. It does not implement
missing target semantics.

## Establish scope and ownership

Record the backend, workload when applicable, PTQ/calibration inputs, target
endpoint, native correctness and performance references, source revision, and
pre-existing dirty paths. New adoption work is PTQ-only.

Select roles from the [role registry](../roles/README.md) by concrete output
file. A role is not a mandatory phase. Do not create a scheduler, mapper,
converter, runtime, or pair adapter merely to make the role list symmetric.

One Git-visible file has one writer during a mutation window. Give a legacy
file containing several responsibilities to one worker, serialize later work,
or approve an intentional extraction. Never assign overlapping writable paths
to concurrent workers.

Agent orchestration changes ownership and timing, not compiler semantics.
Compiler and test routing remain deterministic.

## Select one adoption mode

The phase definitions live in the canonical
[adoption model](../adoption-levels.md); the vocabulary is defined with
examples in the [glossary](../glossary.md). Read the phase there, then apply
only the coordination duties below. The mode names the primary change in one
work order, not the achieved backend state. Stages follow working order for the same
frozen claim scope: `X => W => R`. Report the strongest completed stage; do not
construct additive stage labels. W is an intermediate milestone toward X. A
complete X proof carries the same-scope W obligations, so it does not require a
separate W-only campaign or leave a separately reported W status.

### `register_target`

See [adoption model, R](../adoption-levels.md#r-register-a-target) for what
registration must record and prove. As coordinator you additionally own:

- ordering: do not authorize workload-specific policy until the one-operation
  anchor closes;
- authority: an internal compiler, converter, or endpoint-integration gap
  becomes a successor work order, never an excuse to invent a weaker endpoint;
  and
- foundation: assign the derived `HardwareConfig`, its permitted overrides, and
  its configuration diff to one owner during R.

### `bind_workload_target`

Binding is scoped to one exact `(workload, target, injection point)` tuple. For
the direct-first sequence see
[adoption model, W](../adoption-levels.md#w-bind-and-refine-a-workload-target-pair);
do not restate it in a work order. As coordinator you additionally own:

- freezing the inputs every candidate is judged against: exact model/export,
  PTQ inputs, capability inventory, native reference, tolerance, coverage
  denominator, and measurement boundary;
- refusing any separate composite-region, tile-plan, or call-plan compiler
  representation;
- gating the freeze: candidate assignments earn zero coverage, so
  final execution assignments freeze only after the serialized form has a
  proven backend translation; and
- routing the pair. `integration` registers the exact tuple in
  `WORKLOAD_TARGET_REGISTRY` and owns CLI dispatch in `test/test_codegen.py`.
  A target-independent workload helper is optional. Use a combined
  `workload-target-overlay` in `<workload>_<target>.py` only when model, PTQ,
  calibration, or comparison behavior is genuinely inseparable from the target,
  and preserve the legacy path for unregistered cells.

### `deepen_inner_control`

This legacy mode name performs X-oriented work; it does not define X as a list
of inner-control fields or imply that X advanced. For the exactness, full-control, and consumption
requirements see
[adoption model, X](../adoption-levels.md#x-bit-accurate-full-control-workload-binding).
As coordinator you additionally own:

- dependency ordering: issue separate bounded `bit_accurate_core_lowering` and
  `inner_control` work orders as needed. Implementation order follows real
  inputs; neither local result earns an X claim until the same-scope W proof,
  full-control endpoint consumption, exact output against the frozen
  workload-semantic reference, and reference provenance all pass. Do not
  combine independent owned outputs to bypass the one-output rule.
- feasibility: before assigning inner-control implementation, prove the
  endpoint interface can actually accept the decision. Merely serializing a
  field is insufficient.
- scoping: each later work order takes one named decision — atomic target
  lowering, engine routing, local loops, scratchpad or bank placement,
  retained values, transfer sequencing, or physical address and descriptor
  fields.

## Gate scheduler, storage, and mapping work

Ordinary compiler scheduling, storage planning, and graph-visible or global
address generation may run at R and W. At a partial-control point,
accelerator-internal mapping decisions may remain native-owned. At a
full-control point, Voyager must determine every variable decision that
materially affects per-call mapping performance and the endpoint must consume
its representation. Implementing one such decision is X-oriented work, not by
itself an achieved X binding. Then:

1. identify the exact consumed facts and target legality constraints;
2. reuse an existing provider when it represents and enforces them;
3. bind an existing engine when only target projection is missing; or
4. implement an engine or shared contract only when no existing representation
   can express the required facts.

Graph-level multi-call tiling is separate from local scheduling inside each
call.

## Issue bounded work orders

Create one [work order](../templates/work-order.yaml) per concrete output. It
names one adoption mode, intended claim, role ID and contract, inputs,
consumers, writable paths, read-only dependencies, forbidden paths, acceptance
tests, and required evidence.

An instantiated run guide supplies the repository/environment foundations, R
hardware configuration and tolerance, injection boundary, operation lists,
W's fixed one-LSB fidelity, signature inventory, decomposition bounds, qscheme
candidates, and estimation inputs. The work order references that guide
through `run_guide.path` rather than copying it; the schema no longer has
fields for those parameters. They constrain work; they never replace the
selected role or broaden writable scope.

Every non-`register_target` order must name the R claim it builds on in
`run_guide.registration_claim_id`, so the target configuration and injection
contract it consumes are traceable to the registration that established them.

Specialists cannot modify shared compiler chokepoints or target-neutral unit
tests without an approved `shared-contract` order. `integration` owns registry,
CLI, target regression, and central status edits. Serialize these windows.

Before mutation, snapshot the work-order scope with the checker described in
the [execution contract](execution.md). Compare every new work order with
active work and reject overlapping writable paths. Parallelize read-only
discovery or use isolated worktrees.

## Route gaps and integrate

Record every unsupported, failed, blocked, approximated, or unverified
in-scope boundary in a [gap record](../templates/gap-report.yaml). A missing
neutral interface produces a
[shared-change request](../templates/shared-change-request.yaml); it does not
authorize target branches in shared code.

Review handoffs against their exact work orders and scope checks. Assign one
`integration` owner for central registration, pair routing, target regressions,
and status. Run `$voyager-gate-regressions` and record only checks that actually
ran.

Keep the following claims distinct without duplicating compiler semantics:

- target registration and operation anchor;
- candidate and final execution assignment;
- backend translation and target execution;
- connected workload execution;
- checked coverage, estimated calls/time, optional measurements, and
  utilization; and
- injection-point control classification, X exactness, workload-reference
  provenance, and endpoint consumption of Voyager's performance-relevant
  mapping.

When performance refinement is in scope, use estimated coverage/time gaps,
launch cost, or qualified utilization limits to rank refinements. Keep
measurements separate and do not create work merely for role symmetry.

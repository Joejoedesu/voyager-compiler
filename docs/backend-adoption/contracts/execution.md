<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Adoption work-order execution contract

Use this contract with one work order and its exact role file. It grants no
ownership beyond `workspace.writable_paths`.

## Validate before editing

Reject a missing or unsupported version before interpreting an artifact. The
current versions are:

| Artifact | Version |
| --- | ---: |
| Work order | 5 |
| Handoff | 4 |
| Gap record | 2 |
| Operation inventory, including every target `operations.yaml` | 6 |
| Scope baseline | 1 |
| Role contract | 1 |
| Shared-change request | 1 |

`test/unit_tests/test_agent_context_routing.py` is the target-neutral owner of
this table: it validates every template and every target inventory against
these numbers. A target runner that pins its own version duplicates that
check and drifts; bump this table and that test together.

A work order carries only what bounds **your** assignment: the adoption mode
and claim, the role and its contract, one owned output, the write scope, and
the acceptance bar. Campaign parameters — repositories, environment, target
configuration, injection contract, requested operations, tolerances,
inventories, and estimation inputs — live in the instantiated run guide that
`run_guide.path` names. Read the run guide for those; a work order that
inlines them is malformed.

Require a concrete task ID, one adoption mode, one intended claim, a matching
kebab-case role ID and role contract, one owned outcome, named inputs and
consumers, and explicit writable, read-only, and forbidden paths.

Read the role contract, complete owner files, direct consumers, and the
selected target regression. Do not preload unrelated roles or targets. Stop
with a gap when required input, authority, or a shared contract is missing.

## Preserve the compiler path

- Produce only normal Voyager artifacts owned by the selected role and mode.
  Agent coordination records never become compiler input. W may reuse the
  injection point's native software stack and need not produce detailed
  physical schedules, tiling, scratchpad plans, or accelerator-local addresses.
  Record whether that makes the concrete point partial-control; W may use
  either class, whereas X requires a full-control point.
- Keep workload policy outside target code and target behavior outside shared
  generic passes.
- Start W from executable shared PT2E/Q-DQ semantics, including operators such
  as `conv2d_mx` when they already express the workload operation. Use the
  adapter and existing target software stack to execute that form. Add target-
  specific semantic lowering only when a greater-than-one-LSB failure is
  attributed to shared arithmetic after adapter and endpoint projection are
  otherwise correct. Treat ABI projection, packing, and semantic-preserving
  legality rewrites as representation work. Fix adapter, endpoint, and runtime
  failures at their own boundaries rather than inventing target arithmetic.
- Keep final target FX and `model.txt` as the semantic source of truth. Do not
  introduce another shared execution, composite-region, tile-plan, or call-plan
  IR.
- Keep Q/DQ and arithmetic with operators, graph rewrites with lowering,
  local-loop decisions with schedulers, Voyager-visible storage relations with
  `StoragePlan`, Voyager-visible allocations with allocators, canonical
  emission with serializers, and external ABI projection with target
  converters. W may produce ordinary graph-visible/global addresses through the
  normal compiler. Accelerator-local schedule factors, scratchpad placement,
  address-generator fields, and instruction metadata may remain native-owned
  at a partial-control R or W point. At a full-control point, Voyager must own
  every variable decision that materially affects the claimed call mapping.
- Keep payloads, scales, axes, blocks, dtypes, rounding, saturation, copies,
  aliases, and any Voyager-owned addresses explicit for their consumers.
- Do not add QAT, CPU lowering for accelerator-unsupported operations, guessed
  hardware behavior, or target-named branches to shared production paths.
- Preserve numerical correctness under the declared R tolerance. W requires
  `rtol=0`, `atol=1` output-code LSB. X requires bit-accurate target output
  with zero numerical tolerance against the same frozen workload-semantic
  reference and a full-control injection point. When conversion packages the
  reference, retain its provenance; it is not a second oracle.

A complete X result for the same frozen scope subsumes W's mapping, coverage,
execution, and one-LSB comparison. W is an intermediate working boundary, not
a separately maintained result after X passes.

The rules governing your work order's phase — the direct-first W sequence,
composite and multi-call tiling semantics, assignment states, and the
separation of semantic support from later checks — are owned by the
[adoption model](../adoption-levels.md) and defined with examples in the
[glossary](../glossary.md). Read your phase there once. This contract covers
only what changes while you hold the work order.

## Injection contract

This section is the normative recording contract for an `injection_contract`.
The injection-point and control-class definitions are owned by the
[adoption model](../adoption-levels.md#injection-points-and-control). Freeze one
target-specific instance before adapting Voyager output to an authoritative
execution endpoint.

Here **ABI** means the exact machine-consumed fields, bytes, layouts, symbols,
and call convention at an executable boundary. File collateral is also an
external interface even when it is not a process ABI.

A target adapter is the general integration boundary. A converter is the
specific adapter component that projects canonical Voyager output into
collateral accepted by the injection point.

Record:

- target hardware repository/revision, authoritative default configuration,
  derived run configuration and diff, and matching Voyager `HardwareConfig`
  identity established during R;
- accepted input artifacts, such as `model.txt`, converted collateral,
  generated source, object files, or an executable;
- endpoint identity, version, configuration, and supported operation or
  composite forms;
- injection-point identity, its `partial-control` or `full-control`
  classification, and target-specific evidence for that classification;
- the external input/output boundary and required materialization;
- required dtype, layout, packing, alignment, and qparam encodings;
- who selects the graph form and each variable decision that materially affects
  mapping performance, including applicable instructions, local loops,
  scratchpad behavior, addresses, launch, and readback;
- the entrypoint, initialization, completion, timeout, and failure behavior;
- complete output identity, extent, decoding, oracle, and comparison policy;
  and
- available cycle, call, traffic, and utilization observations with their
  provenance.

At R and W, the native stack may own target-internal instruction decomposition,
local scheduling, scratchpad behavior, and accelerator-local address
generation; that is a partial-control point when any such hidden choice
materially affects mapping performance. Ordinary compiler global-address
generation may remain Voyager-owned. A full-control contract instead names the
performance-relevant decisions Voyager owns, their canonical representation,
and the exact fields the endpoint consumes. X additionally names the
bit-accurate core lowering, frozen workload-semantic reference and comparison
boundary, and the provenance of any packaged expected stream. Control fields
alone earn no X claim until the endpoint executes the accepted artifact and
its scoped output matches exactly.

If the selected endpoint cannot execute until Voyager supplies explicit
mapping-control fields, implement those fields to establish that injection
point. This dependency does not by itself make the point full-control or earn
X: classify it from the remaining hidden performance-relevant decisions and
complete the full workload proof. Do not hide the dependency inside conversion.

The frozen instance lives in a tracked target document or linked versioned
artifact. Evidence records its path, source revision, and content hash.
Examples, headers, ISA descriptions, and successful compilation are discovery
inputs, not substitutes for the contract.

## Operating an external endpoint

These rules were learned independently by every adopted backend with an
out-of-repo endpoint. Apply them whenever `injection_contract.accepted_artifact`
is consumed outside this repository.

- The repository is source-self-contained but **not binary-self-contained**. A
  checkout does not include the toolchain, simulator, or shared libraries. No
  converter-side repair substitutes for that setup.
- Pin the external stack in a location-independent lock artifact recording
  endpoint, guest-stack, and target-revision identities with hashes. The caller
  supplies the root. It is an identity record, not a model schema.
- Preflight the environment with executable assertions before compiling. A
  missing shared library or a shadowed `PYTHONPATH` is a host-environment
  failure and must not be reported as a compiler or converter failure.
- Write every run into a fresh, absent output root. Deleting or overwriting an
  old result hides attempt history.
- Keep compilation separate from external endpoint execution so each retained
  artifact has one clear producing boundary.
- Native completion records from independent engine queues may interleave.
  Global completion order is not an acceptance criterion.
- Publish a failure-to-owning-boundary routing table with the run, so a defect
  is routed to the compiler, converter, endpoint, or host rather than guessed.

## Target adaptation and binding evidence

Target adaptation belongs under `conversion/<backend>/**` when canonical
Voyager artifacts are not accepted directly. A converter may validate and
translate one final operation or composite into one or more native calls. It
may not change the operation's semantics, repair Q/DQ, invent computation or
qparams, or invent an unowned schedule, placement, or address decision.
R semantic lowering may create the computation and qparams needed under its
declared tolerance. During W, target-specific semantic lowering may do so only
after a greater-than-one-LSB failure is attributed to shared arithmetic and the
adapter and endpoint projection are otherwise correct. Target-local
representation or legality projection may be added independently when it
preserves those shared semantics. Adapter, endpoint, and runtime failures stay
at their owning boundaries. Under X, target-exact arithmetic and physical
qparams must already be canonical compiler facts before conversion.

For each final accelerator assignment, retain:

- final FX or `model.txt` artifact identity and hash;
- converted target artifact or direct endpoint input;
- emitted accelerator calls as translation evidence;
- actually executed calls from runtime observation, or from a frozen endpoint
  guarantee that emitted calls execute one-to-one;
- evidence that unsupported work was excluded from accelerator coverage;
- evidence that claimed accelerator work executed at the selected target;
- complete output comparison; and
- producer provenance only when connected execution is claimed.

These items prove translation and execution. They do not restate graph
membership, call dependencies, or tiling semantics already represented by
final FX and `model.txt`.

A captured reference input may validate a target operation or region, but it
does not prove a connected workload. Connected execution requires inputs from
the declared workload or earlier executed producers.

## Verify and hand off

Validate shared changes with target-neutral unit tests. Validate target
behavior through the selected target's end-to-end lane. Preserve exact model,
PTQ/calibration, inputs, endpoint, and numerical comparison requirements; a
weaker smoke test cannot replace a blocked lane.

Return the [handoff](../templates/handoff.yaml) plus one gap record per
unsupported, approximated, failed, blocked, or unverified in-scope boundary.
Record exact commands, artifacts, source revision, assumptions, and the
strongest boundary-limited claim.

The coordinator captures scope before work:

```bash
python3 .agents/skills/voyager-execute-adoption-work-order/scripts/check_work_order_scope.py snapshot \
  --work-order <work-order.yaml> \
  --baseline /tmp/<task-id>.workspace.json
```

After writing the handoff, both worker and coordinator run:

```bash
python3 .agents/skills/voyager-execute-adoption-work-order/scripts/check_work_order_scope.py verify \
  --work-order <work-order.yaml> \
  --baseline /tmp/<task-id>.workspace.json \
  --handoff <handoff.yaml>
```

The checker validates schema, adoption-mode/claim binding, frozen role
contract, Git-visible changes, and declared path scope. It does not prove
compiler semantics, target execution, or performance.

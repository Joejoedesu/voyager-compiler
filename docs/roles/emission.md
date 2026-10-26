# Encoding and emission

Use for: Canonical encoding and target collateral generation from frozen decisions.

## Binding and ownership

`TargetSpec.encode / TargetSpec.emit`. Project frozen decisions into the canonical artifact and target collateral. Runtime submission is a separate extension.

## Load for this task

Frozen program/plan/bindings, target endpoint/ABI and selected recipe/policy, reference boundary and runtime revisions.

Read the relevant [shared contract](../compilation.md#execution-model-and-converter-consistency) and the selected
target's bindings. Start with these source owners; follow imports as needed:

- [Frozen emission](../../src/voyager_compiler/compiler/emit.py)
- [Bundle/checksums](../../src/voyager_compiler/artifacts.py)
- [Runtime entry](../../src/voyager_compiler/runtime.py)
- [Target encoders, emitters and runtime](../../src/voyager_compiler/targets/)

## Implementation decisions

| Problem | Next action |
| --- | --- |
| First target emitter is missing | Start with one selected operation and an endpoint-accepted artifact; map frozen operands, formats, dimensions, addresses and waits to its ABI. Include required host/boundary work. |
| Numerics pass but timing violates modeled reuse | Trace one load/compute/store sequence across tile iterations. Repair transfer scope or missing dependencies with model/builder/runtime owners; do not retile in translation. |
| Destination compiler rejects a form the simulator accepts | Check instruction operand/layout/resource restrictions and versioned API rules. Return new legality constraints to the candidate or operation owner. |

## Do

1. Consume finalized arithmetic, mapping and addresses. Projection may validate/encode/pack; it cannot repair missing semantics, retile or choose another ISA mode.

2. Include transfers, views, host conversions and state operations in the complete driver where needed. Native layer files alone may omit observable program work.

3. Hand the command/submission contract to the [runtime role](runtime.md). Review together when inserted loads/stores, fences or launch order may change reuse/overlap.

4. Record which downstream decisions remain native-owned. Preserve reference provenance; an endpoint-generated expected stream is not an independent semantic reference.

## Do not

- Reinvent tiling, choose a different ISA recipe, assign new addresses or repair missing arithmetic in the converter. Return those changes to search, allocation or semantics.
- Insert or move transfers/fences in a way that changes assumed reuse/overlap without coordinating the builder/model contract and validating the affected behavior.

## Handoff and validation

Return encoded artifacts, the complete driver description and emission checks. Hand execution to the runtime owner and report transfer/control differences to model/builder owners.

Check actual operand/address/mode projection plus the endpoint when available. Host replay and syntactically valid code/commands are separate from device correctness and timing.

[Role index](../roles.md) · [Adoption and task handoffs](../adoption.md)

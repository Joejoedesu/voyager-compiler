# Buffer and storage refinement

Use for: Finalize target execution/storage declarations before shared allocation.

## Extension and inputs

Binding: `TargetSpec.refine`.

Selected buffer program and plan, hardware bank/layout constraints and invocation ownership.

Read the [extension catalog](../extensions.md) and the relevant
[compiler contract](../compilation.md#ir-ownership-and-why-these-records-differ).
Source: [ir/buffer.py](../../src/voyager_compiler/ir/buffer.py).

## Implementation decisions

| Problem | Next action |
| --- | --- |
| Builder output lacks physical requirements | Add packed components, physical shapes, bank/layout constraints and native reservations before final allocation. |
| Want producer/consumer data to stay on chip | Inspect existing residency transforms, escaping users and live footprint; update movement assumptions with the builder/model. |
| Allocation rejects an otherwise legal tile | Expose accurate requirements and rejection evidence. Let shared replanning try another candidate; do not quietly shrink buffers or select another schedule here. |

## Do

- Use `refine(program, config)` after bufferization and before allocation/freeze.
- Declare physical shapes, packed components, bank constraints, scratch/native reservations and permitted residency refinements.
- Keep execution choices consistent with the selected recipe and model; expose competing alternatives through recipes when comparison is needed.

## Do not

- Do not implement a private final allocator or treat reserved padding as initialized values.
- Do not hide a new ISA/tiling search or mutate a frozen program.

## Handoff and validation

Return complete declarations to the shared allocator. Check bank/lifetime/capacity and observable transfer behavior. Coordinate changed movement with the model and builder.

[Role index](../roles.md) · [Shared transformations](../compilation/rewrites.md)

# Shared storage analysis and allocation

Use for: Addresses, bank conflicts, lifetime reuse, packed components or native reservations.

## Binding and ownership

`Framework: compiler/analysis.py / compiler/allocate.py / compiler/banked_allocation.py`. No target allocator override exists. Targets supply declarations through builders and refinement.

## Load for this task

Bufferized execution graph and effects, provisional `CompilationPlan`, hardware
memory/bank geometry and native ownership. Use the [padding guide](../compilation/padding.md)
to distinguish logical shape changes from reserved physical storage; allocation
does not initialize padded values.

Read the relevant [shared contract](../compilation.md#ir-ownership-and-why-these-records-differ) and the selected
target's bindings. Start with these source owners; follow imports as needed:

- [Derived value/effect relationships](../../src/voyager_compiler/compiler/analysis.py)
- [Final allocation](../../src/voyager_compiler/compiler/allocate.py)
- [Physical banks](../../src/voyager_compiler/compiler/banked_allocation.py)
- [CompilationPlan and AllocationPlan](../../src/voyager_compiler/ir/plan.py)

## Implementation decisions

| Problem | Next action |
| --- | --- |
| Model says fit but allocation fails | Compare expanded buffers, slot counts, packed components, bank constraints and lifetimes to model requirements; fix the owner of the missing reservation. |
| Output corruption appears after reuse | Check view/alias effects and completion boundaries before adjusting addresses. Disjoint numerical ranges alone do not prove safe lifetime reuse. |
| A real memory rule is not representable | Identify a concrete rejected/unsafe allocation and extend the canonical constraint plus consumer. Do not work around it with a target-private final allocator. |

## Do

1. Derive relationships from the graph; allocate actual expanded buffers. Logical aliasing is different from reuse of an address during disjoint lifetimes.

2. Preserve packed component identities and reserve their common physical backing once. Include declared physical padding and native arenas.

3. Check physical bank cells/ports as well as byte ranges. Whole native outputs must exist before invocation; asynchronous operands stay live until completion/drain.

4. Return capacity failure to the bounded replan path when appropriate. Do not silently change recipes, hardware sizes or alias semantics inside allocation.

## Do not

- Invent aliases, resize hardware or silently change selected recipes to make allocation succeed; return the actual capacity failure.
- Treat physical padding as initialized numerical values, or reuse an asynchronous operand before its completion boundary.

## Handoff and validation

Return AllocationPlan, capacity/bank/lifetime reasoning and any rejection cause. Handoff finalized addresses/reservations to emission and freeze validation.

Check the affected bank/lifetime/native edge case and selected target regression. A legal virtual address range alone does not establish physical-bank legality.

[Role index](../roles.md) · [Adoption and task handoffs](../adoption.md)

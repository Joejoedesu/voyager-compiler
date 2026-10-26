# Target registration and hardware

Use for: A new target/endpoint, capability or hardware configuration.

## Binding and ownership

`TargetSpec.name / TargetSpec.hardware`. Registry assembly and physical configuration. Follow the separately listed hooks for target behavior.

## Load for this task

Authoritative ISA/configuration, native endpoint and source revisions, requested operations and target control boundary.

Read the relevant [shared contract](../compilation.md#extension-policy-and-minimal-examples) and the selected
target's bindings. Start with these source owners; follow imports as needed:

- [TargetSpec and recipe discovery](../../src/voyager_compiler/extensions.py)
- [Physical types](../../src/voyager_compiler/hardware.py)
- [Registered targets](../../src/voyager_compiler/targets/)

## Implementation decisions

| Problem | Next action |
| --- | --- |
| No existing target package | Read the ISA/API, hardware configuration and endpoint examples. Derive units, memory spaces, legal operand forms and submission boundaries; do not infer hardware facts from a compiler donor. |
| Several instructions compute the same result | Look for explicit commands, native loops/macros, dtype modes and dataflow variants. Route legal alternatives to recipe discovery; a vendor default is not evidence that other forms are slower. |
| Registration works but deployment fails | Check the actual hardware/configuration and endpoint ABI against the declared one. Identify which missing fact or hook matters; do not fill every optional hook. |

## Do

1. Separate physical resources, software execution policy, numerical policy and empirical corrections. Record unknown facts; do not resize hardware to fit an inherited model.

2. List semantic forms and shape/dtype/capacity restrictions. Discover alternative ISA/algorithm realizations before fixing a default.

3. Bind needed hooks in TargetSpec; hardware, recipes and model are required.
   Register the spec once from its package or installed entry point; follow the
   [discovery contract](../compilation.md#extension-policy-and-minimal-examples).
   Reuse shared operators/builders before introducing target arithmetic.

4. Freeze the endpoint and ownership of mapping decisions. Use the registration procedure for affected checks; registry presence and emission alone do not prove endpoint execution.

## Do not

- Resize physical memories or invent hardware facts to make a model fit; keep software policies separate.
- Re-register an existing capability because an old checklist is incomplete, or introduce target-name branches in shared stages.

## Handoff and validation

Return sourced hardware/capability facts, discoverable recipes, endpoint contract, supported restrictions and anchor evidence. Handoff alternatives to implementation and performance owners.

Check registration/discovery and explicit unsupported cases. Execute the declared operation anchor where endpoint access exists; otherwise retain the unverified boundary.

[Role index](../roles.md) · [Adoption and task handoffs](../adoption.md)

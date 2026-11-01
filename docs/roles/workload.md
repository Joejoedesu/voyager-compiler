# Workload integration

Use for: Model construction, inputs/state, assets, export/reference and calibration samples.

## Binding and ownership

`WorkloadSpec`. Construction, input/state, assets, export/reference and calibration sample acquisition. Quantization policy is a separate role.

## Load for this task

Frozen model/weights, preprocessing and input/state boundary; target and intended reference/tolerance. Read the workload × target binding procedure when adopting a new pair.

Read the relevant [shared contract](../compilation.md#extension-policy-and-minimal-examples) and the selected
target's bindings. Start with these source owners; follow imports as needed:

- [Preparation defaults](../../src/voyager_compiler/compiler/prepare.py)
- [WorkloadSpec](../../src/voyager_compiler/extensions.py)
- [Existing workloads](../../src/voyager_compiler/workloads/)

## Implementation decisions

| Problem | Next action |
| --- | --- |
| No workload adapter exists | Start with build/example_inputs and an independent reference; export one representative input to inspect actual signatures. Add assets/state hooks only when needed. |
| Select optional input packing or a pre-quantization graph rewrite | Bind [input preprocessing](input-preprocessing.md); keep sample acquisition here and adapt samples if the input contract changes. |
| Coverage stalls despite a working GEMM | Inspect connected regions, non-matrix ops, layouts, quantization boundaries and state updates. Hand unsupported signatures to lowering/recipes; do not hide them in preprocessing. |
| Repeated calls disagree | Check input/state aliasing and mutation before blaming arithmetic. Compare the first differing update and subsequent consumers. |

## Do

1. Start with one workload module: build plus example_inputs. Add asset checks, calibration, custom reference or state hooks only when this workload needs them.

2. Make input and state evolution reproducible. Reference evaluation can mutate
   state; give reference and target runs independent initial state. Keep the
   semantic reference separate from generated expectations and preserve the
   pre-quantization baseline when measuring accuracy. Record input/checkpoint identities.

3. Inventory real exported operation signatures before judging target coverage. Keep pair-specific numerical policy explicit; do not copy a compiler pipeline into the adapter.

4. For static weights/scales, let common preparation apply the storage policy. Do not create extra transformed weight copies in the workload to bypass that decision.

For input or shape changes, follow [padding ownership](../compilation/padding.md). Keep
model preprocessing/reference semantics in the workload; use the selected
target's legalization and builders for hardware alignment and tile tails.

## Do not

- Copy quantization, tiling or allocation into a workload adapter; hand missing compiler support to its owner.
- Pre-store transformed constants to bypass common storage policy, or add hardware padding without checking its semantic owner.

## Handoff and validation

Return the WorkloadSpec, input/reference identities, relevant signatures and evidence for output/state behavior. Hand format/scale/calibration decisions to [quantization](quantization.md), operation behavior/effects to [operation semantics](semantics.md), and unsupported implementation forms to implementation binding.

Exercise the preparation/reference path and affected cases. Check declared assets and state transitions; use the selected target/workload gate for integration.

[Role index](../roles.md) · [Adoption and task handoffs](../adoption.md)

For input-layer utilization problems, follow [input preprocessing](input-preprocessing.md)
and its shared rewrite helpers.

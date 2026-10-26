# Operation semantics

Use for: defining or correcting an operation's computation, supported signatures,
rounding/saturation behavior, mutation and aliasing.

## Extension and inputs

Operation semantics is a definition responsibility, not another pipeline stage.
Torch operator schemas declare signatures, aliases and writes; executable
implementations define the computation. `TargetSpec.operators` registers the
shared `OperatorSemantics` classification (currently compute/elementwise flags).
Those flags alone do not specify complete arithmetic or effects.

Load the operation's intended meaning, supported shapes/dtypes, exact arithmetic
where required, state/effects and an independent reference. Read the target's
actual operation definitions and the [extension catalog](../extensions.md#operation-semantics).

Sources: [operator registration](../../src/voyager_compiler/extensions.py),
[classification](../../src/voyager_compiler/codegen/aten_classifier.py),
[shape/effect propagation](../../src/voyager_compiler/shape_prop.py),
[value analysis](../../src/voyager_compiler/compiler/analysis.py).

## Implementation decisions

| Problem | Next action |
| --- | --- |
| Shared op seems close to the ISA | Compare rounding, saturation, reduction dtype/order, broadcasting and effects. Reuse it when the selected numerical contract fits; otherwise define the smallest missing operation. |
| Only tails or fused sequences disagree | Check invalid-lane values, reduction boundaries and intermediate rounding. Localize the first divergence with an independent reference before changing quantization. |
| An instruction also mutates storage | Declare writes/aliases and executable effects. Instruction encoding alone cannot establish the operation contract. |

## Do

- Reuse existing ATen/shared operations when their behavior fits. Define a custom
  operation for a concrete unsupported computation or effect.
- Specify input/output shapes and dtypes, numerical behavior and supported forms.
  Define overflow, saturation and rounding where the operation requires them.
- Declare reads/writes/aliases accurately and provide executable reference behavior.
  A copy into an existing buffer must expose its write; a name is not an effect contract.
- Preserve valid-lane, mask and reduction semantics when operations handle padded
  values. Use the [padding reference](../compilation/padding.md).
- Check immutable inputs and results before allowing static preparation; stateful
  behavior cannot become an ordinary constant expression.

## Do not

- Acquire calibration samples (the [workload owner](workload.md)), fit scales or
  run quantization policy here (the [quantization owner](quantization.md)).
- Pick tilings/ISA alternatives, allocate buffers or implement hardware scheduling
  in an operation's reference implementation.
- Infer purity from `compute`/`elementwise` flags or treat a schema as sufficient
  without matching executable behavior.
- Repair missing arithmetic in emission or use an emitter-produced expectation
  as the sole independent numerical reference.

## Handoff and validation

Quantization and lowering consume these definitions. `TargetSpec.lower` maps the
normalized graph to target operations; implementation recipes choose how to realize
those operations. Their executable meaning is established here, independently of
which recipe wins. The same operation may have multiple legal implementations.

Validate outputs, shapes, writes and alias behavior. Exercise repeated invocation
when state changes. Use exact code/bit comparison when required by the operation
contract. Host equivalence does not by itself establish endpoint correctness or
performance. Hand concrete arithmetic/effect requirements to builders and emitters.

[Role index](../roles.md) · [Implementation/search role](implementations.md)

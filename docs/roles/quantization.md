# Quantization

Use for: numerical formats, quantization boundaries, scales, observers,
calibration/recalibration and conversion of quantization representations.

## Extension and inputs

The extension is `TargetSpec.quantization`, configured by `QuantizationPolicy`.
It selects shared PT2E defaults/observer hooks or a target-owned `run` override.
Read the [quantization contract](../extensions.md#quantization)
for signatures, caller precedence and the resolved hardware configuration.

Load the model/graph, requested formats, calibration samples or saved profile,
logical input/output contract and numerical comparison boundary. Workload code
owns sample and asset acquisition; this role owns how they inform quantization.

Sources: [stage dispatch](../../src/voyager_compiler/compiler/quantize.py),
[shared conversion](../../src/voyager_compiler/quantization/quantize_pt2e.py),
[domain rules](../../src/voyager_compiler/quantization/regions.py),
[policy declarations](../../src/voyager_compiler/extensions.py).

## Implementation decisions

| Problem | Next action |
| --- | --- |
| New target accepts a conventional integer/float format | Try shared quantizer settings and observer/domain hooks first. Preserve caller overrides and calibrate on workload-provided data. |
| Target representation or calibration order differs | Use the stage override for an actual conversion-flow difference. Define its arithmetic via operation semantics; distinguish a saved profile, recalibration and value-preserving packing. |
| Quantized output drifts | Compare scale domains, axes and the first changed rounding boundary. Check current conversion/folding guards before adding a second quantizer or weakening tolerance. |

## Do

- Reuse the shared quantizer and preserve explicit per-module/operator controls.
  Use the target override when its format/conversion or calibration flow needs it.
- Preserve required Q/DQ rounding boundaries, scale domains and compound-format
  metadata. Distinguish conversion, later Q/DQ folding and physical packing.
- Pass the resolved compile configuration. Repeated calibration needs replayable
  samples; restore saved state without silently fitting a different profile.
- When exact target arithmetic changes observed values, reference its defined
  operations and recalibrate only when the numerical recipe requires it.
- Return an executable graph and quantization provenance; the shared driver
  refreshes the reference and does not convert an override result again.

## Do not

- Select layouts, tiles, ISA alternatives, addresses or physical bank packing.
- Hide a target calibration/conversion flow inside a workload or observer hook.
- Treat value-preserving storage packing as a reason to recalibrate, or equate
  a second calibration pass with proof of bit accuracy.
- Silently redefine an operation's arithmetic to compensate for bad scales or
  relax a reference tolerance to hide a mismatch.

## Handoff and validation

Pass formats, scales/domains, explicit graph arithmetic and profile evidence to
normalization and lowering. Reuse existing operation definitions. If conversion
requires a new operation, hand its signature and exact behavior to
[operation semantics](semantics.md), then use that operation in the quantized graph.

Preserve a pre-quantization reference for accuracy claims: the shared driver
refreshes `prepared.reference` to the converted graph, which tests a different boundary.

Check the first changed numerical boundary, explicit caller overrides, calibration
and conversion on relevant cases, and affected pinned references. For padding or
scale-axis changes, use the [padding numerical rules](../compilation/padding.md#logical-shape-changes-and-numerical-meaning).
Saved-profile restoration, fresh calibration, host equality and endpoint accuracy
are different validation claims.

[Role index](../roles.md) · [Shared transformations](../compilation/rewrites.md)

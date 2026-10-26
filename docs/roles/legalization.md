# Graph legalization

Use for: Rewrite supported forms before shared normalization.

## Extension and inputs

Binding: `TargetSpec.legalize`.

PreparedWorkload, resolved hardware config, options and logical numerical reference.

Read the [extension catalog](../extensions.md) and the relevant
[compiler contract](../compilation.md#the-compilation-flow).
Source: [pipeline.py](../../src/voyager_compiler/pipeline.py).

## Implementation decisions

| Problem | Next action |
| --- | --- |
| A supported computation has an unsupported graph form | Find a semantics-preserving decomposition or canonicalization needed before common layout/fusion; reuse a guarded shared rewrite where possible. |
| Rewrite needs already selected tensor layouts | Hand it to custom-operation lowering. Tile/ISA-dependent alternatives belong to recipes, rather than an unconditional graph rewrite. |
| No equivalent legal form exists | Report the signature and missing operation contract. Do not approximate silently or call a host fallback accelerator support. |

## Do

- Use `legalize(prepared, config, options)` for target legality and rewrites that must precede shared layouts and fusion.
- Reuse shared rewrite helpers under their actual guards; preserve valid outputs, quantization and state.
- Reject unsupported forms with a clear reason; hand new operation definitions to operation semantics.

## Do not

- Do not duplicate the complete normalization pipeline or silently recalibrate.
- Do not select tiles, allocate final addresses or defer missing arithmetic to emission.

## Handoff and validation

Return a legal tensor-operation graph to normalization. Check the changed signature/boundary and its reference; use the transformation catalog before adding a pass.

[Role index](../roles.md) · [Shared transformations](../compilation/rewrites.md)

# Padding: meaning, placement and storage

Start with the original model's logical input/output/state contract. Identify
the consumer that needs a larger extent and choose the owner below. Padding is
not a single global switch or a reason to round every tensor to an array size.
This is a focused reference under [Shared compilation transformations](rewrites.md#padding-and-shape-legalization). The
[Workload × target binding procedure](../adoption/workload-binding.md) links here when binding a pair.


## Meaning and implementation are separate decisions

**Logical padding** adds values or changes executable tensor extents. Model-required
padding (image borders, masks) is part of the workload semantics. Compiler-added
channel/block alignment is an implementation transform that must preserve the
original valid outputs, scales and state, often through masking or slicing.

**Physical storage padding** reserves extra bytes, bank cells or an expanded
physical layout while preserving logical tensor extents. It does not initialize
those bytes or authorize reading them. Address alignment is a storage constraint,
not a graph operation.

**Absorbing or moving padding is an implementation choice**, not a third kind of
padding. The compiler may preserve a padded producer result, move the pad to a
smaller operand, let a consumer fill its temporary tile, or precompute a constant.
Tile padding is executable initialization/masking of local lanes; it is more than
reserving physical storage, but need not enlarge the graph's global tensor.

Ask first what values/extents must be preserved, then where the extra lanes are
created and removed. A padded intermediate may pass through several operations
before a slice restores the logical result. That requires proof that invalid lanes
do not influence valid ones; merely deleting a pad node is insufficient.

## Which mechanism belongs where

| Need | Owner and current mechanism | Meaning and when to use it |
| --- | --- | --- |
| Input/model semantics | Workload preprocessing, model operators and reference | Image letterboxing, token padding/masks, and convolution's spatial padding belong to the declared workload. Preserve coordinate, attention and output interpretation. |
| Numerical algorithm or format representation | Quantization block helpers, Hadamard rotation, CSR/outlier operators | Padding required by the selected numerical algorithm belongs to that operation's contract. Use its block, scale, sentinel and output-shape rules; it does not substitute for hardware alignment. |
| Matrix channel or quantization-block alignment | Shared `compiler/normalize.py` → `codegen/transform/padding.py:pad_matrix_op_dimensions`; or an explicitly registered target legalization pass | Enlarge actual tensor extents for supported operations, pad operands/scales/bias consistently, and restore logical outputs. Use target policy, not a workload-wide hardcoded multiple. |
| Vector-lane alignment | Shared `pad_vector_op_dimensions` | Align supported softmax and dynamic quantization extents; restore original result shapes. This is not a general padder for every vector op. |
| Tile boundary, receptive-field halo or attention head width | Shared bufferized builders and the paired fit/cost model; `async_copy` padding and `pad_head` | Fill or mask a tile's invalid lanes and store only the valid region. Use when the larger extent is local to a kernel rather than a new full-model tensor shape. Support is builder/target-specific. |
| ISA storage layout or packed representation | Target layout/refinement and encoding; `physical_shape`, `BankLayout`, packed components | Reserve padded physical storage while keeping logical values/shapes distinct. Use for bank/ISA representation, not to make an otherwise illegal logical operation valid. |
| Address, slot or store-beat alignment | Shared allocator, `tensor_alloc_bytes`, whole-bank groups and slot strides | Round allocation bytes or addresses to physical requirements. This does not execute a pad, initialize extra bytes, or add tensor elements. |
| Existing workload-specific compatibility rewrite | Explicit pair/frontend policy, for a particular model/reference contract | Reuse only when its model/reference assumptions apply. These helpers are not universal target registration requirements. |

## Logical shape changes and numerical meaning

The [shared padding implementation](../../src/voyager_compiler/codegen/transform/padding.py)
is called by [normalization](../../src/voyager_compiler/compiler/normalize.py):
matrix padding precedes NHWC/GEMM layout selection; vector padding follows layout
selection and static preparation, before fusion. These passes run when `pad_to_hardware=True` and the
configuration supplies a PE array. They act on recognized operations, not all
operators with matrix-like shapes.

For supported GEMM/conv operations, contraction channels align to the least
common multiple of PE rows and the quantization block; output channels align
to PE columns. Bias/scales and output slices must follow. The shared pass has
special cases, including skipping a three-channel first convolution. Do not
infer that an arbitrary target/op is legal just because this pass ran.

The value used to fill a pad is part of correctness:

- GEMM contraction padding must contribute numerical zero. For codebook
  indices, integer code `0` need not represent zero; the shared helper selects
  the codebook entry closest to zero. Preserve the quantized representation's
  actual rule rather than copying a float pad into encoded data.
- Softmax pads logits with negative infinity and slices the result back;
  zero logits would change normalization. Attention padding must also preserve
  masks, valid sequence length and the scale based on the logical head width.
- Convolution halos use the correct numerical-zero representation; max-pool
  halos use negative infinity in the shared builder. Reductions and average
  pooling must retain their original valid-element/denominator semantics.

Quantized payload and scales can have different shapes. Pad the correct axes
and number of blocks, and validate calibration/quantization consequences at
the pair's declared reference boundary. A shape-preserving output slice alone
does not prove unchanged values.

## Padding inside numerical operators

The [MX block helpers](../../src/voyager_compiler/quantization/mx_utils.py)
temporarily pad partial blocks for blockwise computation and undo that padding
when reconstructing the payload. This differs from persistent graph alignment.
The [quantized operators](../../src/voyager_compiler/ops/quantized.py) also contain
two distinct cases: `hadamard_rotate` keeps a padded contraction extent for
paired rotated GEMM operands; `_pad_csr` fills a fixed-capacity sparse stream
with zero data and invalid `-1` column indices. CSR overflow/truncation is a
separate numerical policy, not harmless padding. Select these mechanisms only
as part of their numerical/representation contract, with matching consumers
and reference behavior.

## Tile padding versus physical storage

[Bufferized builders](../../src/voyager_compiler/compiler/bufferize/pipeline.py)
can load a convolution/pooling halo into a temporary tile using
[`async_copy(..., pad, pad_value)`](../../src/voyager_compiler/compiler/bufferize/ops.py).
This avoids materializing a padded whole input. Attention uses
[`AttentionSpace.head_multiple` and `pad_head`](../../src/voyager_compiler/ir/plan.py)
for kernel dimensions. The fit/cost model must account for the actual padded
tile, scratch and buffering slots. Some builders require divisible dimensions
instead of supporting arbitrary tails; inspect their legality checks.

`physical_shape` means **reserve enough backing storage**, not **compute a
larger tensor**. The target must define initialization/masking and transfer
encoding if any extra lanes are accessed. Target-specific logical-extent metadata records the original extent when an
executable operation has been padded; it is not interchangeable with `physical_shape`.
The [allocator](../../src/voyager_compiler/compiler/allocate.py) charges physical
extent, alignment and store-tail slack; target packing can add layout-specific
bytes. Search fit and final allocation have different scopes, so inspect both.

## Absorption, movement and elimination opportunities

The following are existing mechanisms, not a general promise that pads fuse into
arbitrary preceding or following operations. Source owners refer to
[`codegen/transform/padding.py`](../../src/voyager_compiler/codegen/transform/padding.py)
unless otherwise linked.

| Opportunity | Current mechanism | Guard or limit |
| --- | --- | --- |
| Keep an already padded producer result | `_insert_pad`, `_find_cancelling_slice`, `_drop_slice_through_chain` remove compatible slice/re-pad boundaries through supported relayouts | Internal shape-legalization context, matching axis/extents and supported chains. This is not the general identity `pad(slice(x)) = x`; removed values and externally visible uses matter. |
| Delay restoring the logical output | `slice_output` / `_crossable` move the slice after supported elementwise operations or non-block quantization | Extra lanes cannot mix into valid lanes. Additional operands must agree; blocked quantization is explicitly stopped. |
| Pad before expansion | `_hoist_pad_above_repeat` moves padding before eligible repeat/broadcast relayout chains | Track the padded axis through permutations and restate shapes. This reduces the amount of data filled; it is not arbitrary reordering through compute. |
| Prepare immutable operands once | `_insert_pad` materializes padded `get_attr` tensors when their backing is not written | Charge the enlarged constant. Required alignment is separate from the non-expanding budget for optional constant preparation. |
| Widen a state buffer and its writes | `_fold_pad_into_cache` rewrites an eligible cache buffer plus per-token updates | Guarded reader/axis conditions; cannot widen the indexed update axis. Update the state contract consistently; a padded snapshot of mutable data would go stale. |
| Put padding in the consumer's load | Bufferized convolution/pooling builders use `async_copy(..., pad, pad_value)` for halos | Builder-specific support and correct fill value. The consumer's model must include the actual tile, scratch and transfer behavior. This is not a universal graph-pad-to-copy rewrite. |
| Avoid full-tensor alignment by using native tails | `pad_to_hardware=False` plus an implementation supporting partial dimensions | Disabling the shared padder does not create tail support. The recipe, model and emitter must already implement it. |

For example, moving a zero pad across ReLU can preserve the zero fill, while
moving it across sigmoid would turn the added zeros into 0.5. Conversely, delaying
a slice after an elementwise sigmoid can still preserve all **valid** lanes if
extra lanes remain unobserved. These are different transformations with different
proofs. Softmax, reductions and block-scale calculation can mix lanes; their masks,
valid extents and scale grouping must be preserved explicitly.

There is currently no shared search that enumerates all pad placements across
arbitrary operators. Reuse these guarded rewrites and builder mechanisms; a new
absorption rule belongs to the relevant shared pass or target implementation,
with its numerical and use/alias conditions made explicit.

## Target-specific decisions

The selected [target guide](../../src/voyager_compiler/targets/README.md) records
which legality pass, builder, storage layout and emitter implements these rules.
Check the actual operation signature and selected recipe: a generic buffer
primitive does not imply that every target can encode it. Pair-specific workload
preparation belongs in the workload contract; it is not a registration requirement.

## Decision and validation procedure

1. Record logical shapes, dtypes/scales, masks and state. Separate model-required
   padding from the target's representation requirements.
2. Use `compile_workload()` so preparation → quantization → target legalization →
   common normalization → bufferization → allocation run in their intended
   order. `compile_bufferized()` expects an already normalized graph and does
   not add missing frontend/legalization steps for you.
3. Choose the smallest applicable owner from the first table. Prefer existing
   target legality and tile handling. Use a workload-specific rewrite only
   when its model/reference assumptions are explicit; do not stack copies of
   the same padding rule in the workload, target and emitter.
4. Check one aligned and one affected boundary case, or reuse applicable
   evidence. Compare valid outputs and masks/state, inspect fill values and
   slicing, then check actual memory/bank demand and emitted transfers. For
   cache changes, exercise repeated decode/update, not just the first call.
5. Report useful logical work separately from padded execution. Required
   padding can increase stored bytes and runtime work; price that increase.

The shared padding helper can hoist eligible padding into immutable constants
or widen a KV-cache buffer and its per-token writes under its guards. Mutable
cache reshaping is state legalization, not constant folding. The optional
[static preparation budget](../compilation.md#static-tensor-preparation-and-the-storage-limit)
rejects enlarging parameter transforms; required shape legalization and ISA
storage padding have separate, explicitly charged capacity contracts.

[Documentation router](../README.md) · [Bufferization role](../roles/bufferization.md) ·
[Quantization role](../roles/quantization.md) · [Operation semantics role](../roles/semantics.md) · [Allocation role](../roles/allocation.md)

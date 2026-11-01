# Shared compilation transformations

Use this catalog to find an existing transformation before adding target code.
It sits under the [compilation reference](../compilation.md), alongside stage/IR
contracts. [Adoption](../adoption.md) describes how to carry out integration work;
[roles](../roles.md) describe who owns decisions. They link here for compiler
capabilities rather than maintaining separate feature lists.

This is a catalog of supported transformation families, not a new pass manager
or a promise that every exported helper runs automatically. The shared pipeline
selects stage order. A target's registration and each pass's guards determine
what applies. Read only the relevant detail and callable source.

## Features along the compilation flow

| Stage | Existing optimization | Owner and boundary |
| --- | --- | --- |
| Prepare/export (optional) | Input representation rewrite, such as selected conv-to-im2col | `TargetSpec.input_preprocessing`, selected by the binding; before observers/calibration. See [input boundary rewrites](#input-boundary-rewrites). |
| Quantization | Observer conversion, constant-weight quantization and inference conv/BN folding | `quantization/quantize_pt2e.py`; shared quantization or a target override. `freeze_weights` is optional, after calibration; it is not automatically selected by normalization. |
| Normalize: cleanup | Constant generators, autocast inlining, removable operations, attention-operand normalization and scalar index arithmetic | `compiler/normalize.py` invokes `fold_constant_generators`, `inline_autocast_modules`, `remove_prunable_ops`, `normalize_attention_operands`, `scalarize_index_arithmetic`. `keep_fp32=False` explicitly opts into removing FP32 casts. |
| Normalize: quantization folding | Compose compatible scales and move Q/DQ toward producers or supported cache updates | `quant_folding.py:fuse_quantize_dequantize_with_producer` runs before alignment/layout; format, block-axis and boundary guards still apply. |
| Normalize: shape legality | Align matrix channels/quantization blocks and supported vector extents; remove redundant slice/pad boundaries | `padding.py:pad_matrix_op_dimensions`, `pad_vector_op_dimensions`; controlled by `pad_to_hardware` and configured PE geometry. Details below. |
| Normalize: layout | NCHW/NHWC convolution layout, GEMM/GEMV weight layout and eligible activation layout islands | `data_layout.py`; selected by normalization options. These are explicit graph transformations with guards and transpose-cost heuristics, not arbitrary layout enumeration. |
| Normalize: static preparation | Fold eligible immutable parameter/scale transforms after layouts are known; deduplicate equivalent expressions | `rewrites.py:fold_constant_generators(phase="layout")`, `deduplicate_nodes`; retained originals and compound storage count toward the optional storage-growth limit. |
| Normalize: graph fusion | Group target-supported compute chains and eligible input/output relayouts | `operator_fusion.py:fuse_operator`; target-owned pattern table, `skip_op_fusion`, `fuse_reshape`. Group selection considers saved DRAM traffic and escaping users. Hardware descriptions do not derive patterns. |
| Explicit frontend preparation | Conv-to-im2col, RMSNorm recognition, selected mask/interpolation/embedding rewrites and cache splitting | Reusable helpers in `rewrites.py`, `padding.py`, `quant_folding.py`; callers opt in where workload/target semantics justify them. Their presence in the package does not make them universal default passes. |
| Bufferize | Lower views to addressing; redirect eligible concat/stack stores; fill local halo/tail tiles | `compiler/bufferize/view_lowering.py`, `pipeline.py` and selected builders. Keep copies when access, escape or write semantics require them. Tile support is implementation-specific. |
| Buffer refinement | Merge eligible SRAM-resident producer/consumer transfers | `compiler/bufferize/residency.py:merge_resident`; preserve observable stores and satisfy later capacity/lifetime constraints. It is a buffer-graph transformation, not tensor-graph fusion. |
| Target kernel construction | Combine primitives into ISA instruction patterns | Target-owned builders and instruction patterns; optional target behavior, not an automatic generic fusion capability. The emitter consumes selected decisions. |
| Storage and emission | Pack representations, reserve physical padding and reuse non-overlapping allocations | Canonical storage declarations, shared allocation and target encoding. These are related implementation stages; reserving bytes does not execute a tensor pad or establish numerical equivalence. |

Canonical stage order: [compiler/normalize.py](../../src/voyager_compiler/compiler/normalize.py).
Public controls and ownership: [extension catalog](../extensions.md).
The descriptions below distinguish closely related mechanisms.

Convolution classification distinguishes dense, grouped, and depthwise connectivity
(including depthwise channel multipliers). NHWC conversion retains OIHW weights
for grouped operations; channel padding must preserve group membership. These
facts do not select hardware. Target recipes own native/group-aware execution or
an explicit whole-operation builder; see [bufferization](../roles/bufferization.md).


## Padding and shape legalization

There are two questions:

1. **What does the padding mean?** Logical padding adds values or changes executable
   extents; physical padding reserves storage without changing logical extents.
   Model-required padding and compiler alignment have different semantic obligations.
2. **Where is it realized?** An explicit tensor operation, a widened producer,
   a consumer's tile load, precomputed constants, or physical storage declarations.
   Moving or absorbing padding is a placement choice, not a third padding type.

The shared implementation already cancels eligible slice/re-pad boundaries,
delays output slices, moves padding before supported repeat/broadcast chains,
pads immutable constants once and widens eligible cache buffers together with
updates. Selected builders can fill halos directly into local tiles. These are
guarded mechanisms; there is no generic search across all pad placements or
arbitrary producer/consumer absorption.

Read [padding: meaning, placement and storage](padding.md) for exact boundaries,
fill values, quantized scales, existing absorption opportunities and checks.
A physical reservation alone neither fills invalid lanes nor makes their reads safe.

## Layout and static preparation

Layout policy changes how operations read operands. Static preparation decides
whether an immutable transform can be evaluated once and stored. One does not
imply the other: a layout transpose may remain at runtime when its source is
mutable or keeping both versions would exceed the storage policy.

The shared pass supports recognized parameter/scale transforms and constant
generators; it does not fold arbitrary activation-dependent computation. The
[static-preparation contract](../compilation.md#static-tensor-preparation-and-the-storage-limit)
owns eligibility, retained representations, byte accounting and metadata.
Required alignment padding is accounted for separately from optional,
non-expanding parameter preparation.

Source: [rewrites.py](../../src/voyager_compiler/codegen/transform/rewrites.py),
[data_layout.py](../../src/voyager_compiler/codegen/transform/data_layout.py).

## Quantization conversion and folding

Quantization establishes numerical representations and scales. Conversion turns
observers into explicit arithmetic. Later Q/DQ folding can compose compatible
scale operations or place work closer to producers; it must preserve required
rounding boundaries and block/domain relationships. Neither is a synonym for
constant folding or graph fusion.

Hardware-accurate numerical conversion may justify target-specific recalibration;
value-preserving packing does not. See the
[quantization boundary](../extensions.md#quantization-and-custom-operation-lowering).
Shared scale-folding implementation:
[quant_folding.py](../../src/voyager_compiler/codegen/transform/quant_folding.py).

## Fusion and intermediate storage

Graph fusion chooses groups of tensor operations for joint implementation. Target
pattern tables describe legal groups; shared matching and group selection perform
the rewrite. Trailing layouts and repeated operands have separate guarded folding
rules. A group does not guarantee that all internal work becomes one instruction.

Builders can subsequently fuse primitive instructions using target-owned patterns.
Bufferized view/store rewrites and SRAM-resident merging can remove data movement
without changing graph-fusion policy. Final allocation can reuse addresses without
making values aliases. Distinguish these mechanisms when diagnosing an intermediate
allocation or unexpected transfer.

Sources: [operator_fusion.py](../../src/voyager_compiler/codegen/transform/operator_fusion.py),
[view_lowering.py](../../src/voyager_compiler/compiler/bufferize/view_lowering.py),
[residency.py](../../src/voyager_compiler/compiler/bufferize/residency.py),
[allocation](../../src/voyager_compiler/compiler/allocate.py).

## Attention and mutable state

Attention-operand normalization and scalar index rewrites run in the shared
normalizer. Cache splitting is an explicit frontend choice; compatible cache folds
can then participate in quantization rewrites. Padding/layout changes that widen a
cache must update its writes and readers together. An immutable parameter transform
cannot be applied to a snapshot of mutable cache contents.

Read the current callers and state/effect guards before combining these helpers.
Validate repeated updates, not only the initial invocation. This catalog does not
turn model-specific mask or cache recipes into requirements for every workload.

## Choosing a mechanism

Identify the stage, inspect the actual target binding and pass guards, then reuse
the smallest supported transformation. If a needed behavior is absent, extend the
owning pass or implementation rather than adding a duplicate pass in an emitter.
Preserve valid outputs, quantization and state; account for remaining transfers,
padded work and retained storage. Numerical equality and reduced graph-node count
do not by themselves establish device performance.

[Compilation reference](../compilation.md) · [Extension points](../extensions.md) ·
[Documentation entry point](../README.md)

## Input boundary rewrites

**Stage:** prepare/export → optional rewrite → calibrate/quantize. The workload ×
target binding opts in; shared normalization does not enable it automatically.

Use [input preprocessing](../extensions.md#input-preprocessing) to select an
exported-graph rewrite for improved utilization. `WorkloadSpec.prepare` remains
for custom frontends; `WorkloadSpec.preprocess` only changes inputs before export.

[`replace_conv2d_with_im2col`](../../src/voyager_compiler/codegen/transform/rewrites.py)
is a specialized helper for eligible three-channel ATen convolutions with stored
weights. Use `predicate=lambda node: ...` to select nodes; the default rewrites all
eligible nodes. Check parameter-use and following-operation assumptions. This is
not a generic first-layer optimizer or a host-packing extractor.
When the binding needs an aligned packed reduction, `reduction_multiple=N`
zero-pads the unfolded input and materializes a matching padded weight. The
default `1` adds no alignment. This is explicit binding-owned storage expansion;
it does not relax the non-inflating constant-preparation rule.

| Choice | Contract |
| --- | --- |
| Rewrite before quantization | Check activation/weight axes and bias annotations on the rewritten graph. This selects one frontend per compile. |
| Packed GEMM versus native convolution after quantization | Target recipes preserve fixed scales, zero points, rounding and bias under the same input/output contract; no silent recalibration. |

Host packing is excluded from accelerator timing only by the agreed boundary;
report it separately. Internal activation packing remains executable, modeled
work. Declare packed shapes, padded weights and scratch; padded lanes must encode
numerical zero and expanded constants count toward storage. Dynamic im2col is not
constant folding. A graph rewrite alone does not move packing to the host.

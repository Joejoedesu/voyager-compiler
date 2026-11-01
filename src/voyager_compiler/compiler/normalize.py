"""Canonical graph normalization, shared by every target."""

import torch
from torch.utils._pytree import tree_flatten

from voyager_compiler.codegen import (
    deduplicate_nodes,
    fold_constant_generators,
    fuse_operator,
    fuse_quantize_dequantize_with_producer,
    inline_autocast_modules,
    normalize_activation_layout,
    normalize_attention_operands,
    normalize_conv2d_layout,
    normalize_gemm_weight_layout,
    pad_matrix_op_dimensions,
    pad_vector_op_dimensions,
    remove_fp32_casts,
    remove_prunable_ops,
    rename_nodes_with_param_names,
    scalarize_index_arithmetic,
)
from voyager_compiler.hardware import AcceleratorConfig
from voyager_compiler.ops.layout import (
    DEFAULT_GEMM_LAYOUT,
    DEFAULT_TENSOR_LAYOUT,
    GEMM_LAYOUTS,
    GEMV_LAYOUTS,
    TENSOR_LAYOUTS,
    GemmLayout,
)
from voyager_compiler.shape_prop import ShapeProp, fake_like


def transform(
    model: torch.fx.GraphModule,
    example_args,
    example_kwargs=None,
    patterns=None,
    config=None,
    tensor_layout=DEFAULT_TENSOR_LAYOUT,
    gemm_layout=DEFAULT_GEMM_LAYOUT,
    gemv_layout=DEFAULT_GEMM_LAYOUT,
    skip_op_fusion=False,
    fuse_reshape=True,
    keep_fp32=True,
    pad_to_hardware=True,
):
    """Lower ``model`` in place through the graph-level passes.

    The passes run on fake copies of the example inputs and read only their
    shapes (see ``shape_prop``).

    Args:
        model: The exported graph, rewritten in place.
        example_args: Example positional inputs.
        example_kwargs: Example keyword inputs; ``None`` for none.
        patterns: The fusion patterns ``fuse_operator`` applies, each a list
            of ``OpMatcher`` from anchor to tail.
        config: The accelerator's ``AcceleratorConfig``; ``None`` models no
            hardware and skips padding.
        tensor_layout: The memory order of the 4-D feature maps that conv2d,
            the 2-D pools and the ops connected to them read and write:
            ``"nchw"`` (channels first, PyTorch's order) or ``"nhwc"``
            (channels last, with HWIO conv weights). GEMM operands are not
            affected; they follow ``gemm_layout`` and ``gemv_layout``.
        gemm_layout: The layout of a matrix-matrix GEMM's activation, then
            its weight, in C (contraction), K (output features) and X
            (tokens): ``"xc,kc"`` (PyTorch's), ``"xc,ck"`` (the weight
            [contraction, out]) or ``"cx,ck"`` (also the activations between
            GEMMs [features, tokens] wherever that transposes less, for a
            Trainium TensorE; see ``normalize_activation_layout``).
        gemv_layout: The layout of a matrix-vector GEMM (a single activation
            row): ``"xc,kc"`` or ``"xc,ck"``.
        skip_op_fusion: Skip operator fusion.
        fuse_reshape: Let fusion fold a trailing reshape or permute into
            its producer's output.
        pad_to_hardware: Align global matrix/vector dimensions to hardware when True.
            False leaves tail/halo handling to the selected native implementation.
        keep_fp32: ``False`` computes in 16-bit float where the model casts
            up to float32 (``remove_fp32_casts``).

    Returns:
        ``model``.
    """
    if example_kwargs is None:
        example_kwargs = {}

    # A null config (no hardware) skips padding and tiling.
    if config is None:
        config = AcceleratorConfig(pe_array_size=None)

    if not keep_fp32:
        remove_fp32_casts(model)

    flatten_args, spec = tree_flatten((example_args, example_kwargs))
    ShapeProp(model).propagate(*map(fake_like, flatten_args))

    fold_constant_generators(model)
    inline_autocast_modules(model)
    remove_prunable_ops(model)
    normalize_attention_operands(model)
    scalarize_index_arithmetic(model)
    fuse_quantize_dequantize_with_producer(model)

    if pad_to_hardware and config.pe_array_size is not None:
        pad_matrix_op_dimensions(model, config.pe_array_size)

    assert tensor_layout in TENSOR_LAYOUTS, tensor_layout
    if tensor_layout == "nhwc":
        normalize_conv2d_layout(model)

    assert gemm_layout in GEMM_LAYOUTS, gemm_layout
    assert gemv_layout in GEMV_LAYOUTS, gemv_layout
    gemm_activation, gemm_weight, _ = GemmLayout.parse(gemm_layout)
    gemv_weight = GemmLayout.parse(gemv_layout).weight
    normalize_gemm_weight_layout(model, gemm_weight, gemv_weight)
    if gemm_activation == "cx":
        normalize_activation_layout(model)

    # Layout selection can introduce static transposes. Decide after every
    # consumer has its layout, charging any original representation retained.
    deduplicate_nodes(model)
    fold_constant_generators(model, phase="layout")

    if pad_to_hardware and config.pe_array_size is not None:
        pad_vector_op_dimensions(model, config.vector_lanes)

    if not skip_op_fusion:
        fuse_operator(model, patterns, fuse_reshape)

    rename_nodes_with_param_names(model)
    deduplicate_nodes(model)

    return model

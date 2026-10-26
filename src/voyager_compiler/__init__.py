"""Lower a quantized PyTorch model onto a Voyager-generated accelerator.

``transform()`` runs the hardware-lowering passes over an exported FX graph;
``compile()`` bufferizes, plans memory and emits the ``voyager`` IR.  Both
operate in place and leave the graph executable, so every stage can be checked
numerically against the original.
"""

from typing import Callable, Optional, Tuple

import torch
from torch.fx import Node

# Defines torch.ops.quantized_ops.* (and the hardware-layout twins).  This must
# come first: the imports below reference those targets, and the op namespace
# resolves lazily, so an unregistered target fails at call time rather than
# here.
from voyager_compiler import ops as _register_ops  # noqa: F401
from voyager_compiler.cli_args import (
    add_compile_args,
    add_experiment_args,
    add_quantization_args,
)
from voyager_compiler.codegen import (
    deduplicate_nodes,
    extract_input_preprocessor,
    fuse_operator,
    match_and_fuse_rms_norm,
    normalize_attention_operands,
    pad_vit_embeddings_output,
    quantizes_kv_cache,
    remove_fp32_casts,
    remove_zero_attention_mask,
    replace_conv2d_with_im2col,
    replace_interpolate,
    scalarize_index_arithmetic,
    split_kv_cache,
)
from voyager_compiler.codegen import (
    fold_constant_generators as fold_constant_generators,
)
from voyager_compiler.codegen import (
    fuse_quantize_dequantize_with_producer as fuse_quantize_dequantize_with_producer,
)
from voyager_compiler.codegen import (
    gen_compute_graph as gen_compute_graph,
)
from voyager_compiler.codegen import (
    inline_autocast_modules as inline_autocast_modules,
)
from voyager_compiler.codegen import (
    normalize_activation_layout as normalize_activation_layout,
)
from voyager_compiler.codegen import (
    normalize_conv2d_layout as normalize_conv2d_layout,
)
from voyager_compiler.codegen import (
    normalize_gemm_weight_layout as normalize_gemm_weight_layout,
)
from voyager_compiler.codegen import (
    pad_matrix_op_dimensions as pad_matrix_op_dimensions,
)
from voyager_compiler.codegen import (
    pad_vector_op_dimensions as pad_vector_op_dimensions,
)
from voyager_compiler.codegen import (
    remove_prunable_ops as remove_prunable_ops,
)
from voyager_compiler.codegen import (
    rename_nodes_with_param_names as rename_nodes_with_param_names,
)
from voyager_compiler.compiler.bufferize import (
    bufferize_graph as bufferize_graph,
)
from voyager_compiler.compiler.bufferize import (
    flush_tensor_files as flush_tensor_files,
)
from voyager_compiler.compiler.bufferize import (
    gen_code_bufferized as gen_code_bufferized,
)
from voyager_compiler.compiler.bufferize import (
    plan_memory as plan_memory,
)
from voyager_compiler.compiler.bufferize import (
    print_bufferized_graph as print_bufferized_graph,
)
from voyager_compiler.compiler.bufferize import (
    print_layer_table as print_layer_table,
)
from voyager_compiler.compiler.bufferize import (
    shared_dram_layout,
)
from voyager_compiler.compiler.search import (
    DEFAULT_RUNTIME_TOLERANCE as DEFAULT_RUNTIME_TOLERANCE,
)
from voyager_compiler.compiler.search import (
    build_tiler as build_tiler,
)
from voyager_compiler.export_utils import (
    export_model,
    get_aten_graph_module,
    get_conv_bn_layers,
    get_node_name_to_scope,
    merge_sibling_linears,
    print_node_scope_tabular,
)
from voyager_compiler.hardware import AcceleratorConfig
from voyager_compiler.modeling import (
    dispatch_model,
    get_device_map,
    insert_align_device_nodes,
)
from voyager_compiler.ops.layout import (
    DEFAULT_GEMM_LAYOUT as DEFAULT_GEMM_LAYOUT,
)
from voyager_compiler.ops.layout import (
    DEFAULT_TENSOR_LAYOUT as DEFAULT_TENSOR_LAYOUT,
)
from voyager_compiler.ops.layout import (
    GEMM_LAYOUTS as GEMM_LAYOUTS,
)
from voyager_compiler.ops.layout import (
    GEMV_LAYOUTS as GEMV_LAYOUTS,
)
from voyager_compiler.ops.layout import (
    TENSOR_LAYOUTS as TENSOR_LAYOUTS,
)
from voyager_compiler.quantization import (
    DerivedQuantizationSpec,
    DirectCastFakeQuantize,
    FusedAmaxObsFakeQuantize,
    GroupWiseAffineFakeQuantize,
    MXFakeQuantize,
    QScheme,
    QuantizationConfig,
    QuantizationSpec,
    TrainingQuantizers,
    TrainingStep,
    capture_training,
    convert_pt2e,
    derive_bias_qparams_fn,
    disable_observers,
    fold_conv_bn_qat,
    freeze_cache_reads,
    freeze_weights,
    get_default_quantizer,
    gradient_program,
    prepare_for_qat,
    prepare_pt2e,
    prepare_qat_pt2e,
    prepare_training,
    set_batch_norm_training,
    set_training,
    sink_obs_or_fq,
    update_program,
)
from voyager_compiler.quantization.dtypes import (
    quantize_to_nf,
    quantize_to_posit,
)
from voyager_compiler.shape_prop import (
    ShapeProp,
    fetch_attr,
    propagate_shape,
)
from voyager_compiler.shape_prop import (
    constant_value as constant_value,
)
from voyager_compiler.shape_prop import (
    fake_like as fake_like,
)
from voyager_compiler.utils import with_execution_context

__all__ = [
    "AcceleratorConfig",
    "DerivedQuantizationSpec",
    "DirectCastFakeQuantize",
    "FusedAmaxObsFakeQuantize",
    "GroupWiseAffineFakeQuantize",
    "MXFakeQuantize",
    "OpMatcher",
    "Pattern",
    "QScheme",
    "QuantizationConfig",
    "QuantizationSpec",
    "ShapeProp",
    "TrainingQuantizers",
    "TrainingStep",
    "add_compile_args",
    "add_experiment_args",
    "add_quantization_args",
    "capture_training",
    "compile",
    "convert_pt2e",
    "deduplicate_nodes",
    "derive_bias_qparams_fn",
    "disable_observers",
    "dispatch_model",
    "export_model",
    "extract_input_preprocessor",
    "fetch_attr",
    "fold_conv_bn_qat",
    "freeze_cache_reads",
    "freeze_weights",
    "fuse_operator",
    "get_aten_graph_module",
    "get_conv_bn_layers",
    "get_default_quantizer",
    "get_device_map",
    "get_node_name_to_scope",
    "gradient_program",
    "insert_align_device_nodes",
    "match_and_fuse_rms_norm",
    "merge_sibling_linears",
    "normalize_attention_operands",
    "pad_vit_embeddings_output",
    "prepare_for_qat",
    "prepare_pt2e",
    "prepare_qat_pt2e",
    "prepare_training",
    "print_node_scope_tabular",
    "propagate_shape",
    "quantize_to_nf",
    "quantize_to_posit",
    "quantizes_kv_cache",
    "remove_fp32_casts",
    "remove_zero_attention_mask",
    "replace_conv2d_with_im2col",
    "replace_interpolate",
    "scalarize_index_arithmetic",
    "set_batch_norm_training",
    "set_training",
    "shared_dram_layout",
    "sink_obs_or_fq",
    "split_kv_cache",
    "transform",
    "update_program",
    "with_execution_context",
]


class qscheme: ...


# Defined in voyager_compiler/quantizer.h
per_tensor_symmetric: qscheme = QScheme.PER_TENSOR_SYMMETRIC
per_channel_symmetric: qscheme = QScheme.PER_CHANNEL_SYMMETRIC
microscaling: qscheme = QScheme.MICROSCALING
group_wise_affine: qscheme = QScheme.GROUP_WISE_AFFINE


def _get_op_overload(op_name: str):
    all_overloads = []
    for lib in [torch.ops.aten, torch.ops.quantized_ops]:
        # Also check inplace version of the op (e.g., "add_" for "add")
        for name in [op_name, f"{op_name}_"]:
            if (packet := getattr(lib, name, None)) is None:
                continue
            all_overloads.extend(
                [getattr(packet, name) for name in packet.overloads()]
            )
    return all_overloads


class OpMatcher:
    """One op of a fusion pattern: its targets and a check of its operands.

    Args:
        *ops: The op names it matches, every overload of each, in
            ``aten`` and ``quantized_ops``.
        predicate: A check of the op node, e.g. of its operands; ``None``
            for none.
        keeps_input: Whether the op's input, the op before it in the
            chain, is read outside the group too: the instruction-level
            fusion (``fuse_patterns``) fuses the chain only then, returning
            it as a result of the group -- a reduction an instruction makes
            of the tile it stores.
    """

    targets: Tuple[torch._ops.OpOverload]
    predicate: Optional[Callable[[Node], bool]] = None

    def __init__(self, *ops, predicate=None, keeps_input=False):
        self.predicate = predicate
        self.keeps_input = keeps_input

        # Resolve symbolic ops
        targets = []
        for op in ops:
            targets.extend(_get_op_overload(op))

        # Freeze resolved targets
        self.targets = tuple(targets)

    def matches(self, node: Node) -> bool:
        if node.target not in self.targets:
            return False

        return self.predicate(node) if self.predicate else True


class Pattern(list):
    """A fusion pattern: the ``OpMatcher`` of each op along a producer ->
    consumer chain, in order, and an optional name.

    ``transform()`` fuses the model's own ops by unnamed patterns (a plain
    list is one too).  A named pattern is an instruction a backend's builder
    fuses the ops it built into (``fuse_patterns``); the fused node records
    the name as ``meta['pattern']``, and ``model.txt`` carries it.

    Args:
        *ops: The ops' matchers, in chain order.
        name: The instruction form the pattern is; ``None`` for none.
    """

    def __init__(self, *ops, name=None):
        super().__init__(ops)
        self.name = name


from voyager_compiler.compiler.normalize import transform


def compile(
    model: torch.fx.GraphModule,
    example_args,
    example_kwargs=None,
    config=None,
    output_dir=None,
    output_file="compute_graph",
    dump_tensors=True,
    runtime_tolerance=None,
    dram_layout=None,
    accumulate_fp32=False,
    bool_mask=True,
):
    """Bufferize ``model``, plan its memory and write its ``voyager`` IR.

    Writes ``model.txt`` (the IR as text), ``layers.txt`` (a table of its
    layers) and ``<output_file>.svg`` (the compute graph) to ``output_dir``.

    Args:
        model: A graph ``transform`` lowered, bufferized in place.
        example_args: The inputs ``model`` runs on. With ``dump_tensors``,
            one it writes, such as a gradient buffer, is overwritten;
            otherwise only their shapes and dtypes are read.
        example_kwargs: Keyword inputs; ``None`` for none.
        config: The accelerator's ``AcceleratorConfig``; ``None`` models no
            hardware.
        output_dir: The directory the files are written to.
        output_file: The compute-graph drawing's basename.
        dump_tensors: Also write every tensor's values under
            ``output_dir/tensor_files``.
        runtime_tolerance: How much longer than the fastest modeled tiling
            a tiling may run and still be chosen, as a fraction; ``None``
            takes ``DEFAULT_RUNTIME_TOLERANCE``.
        dram_layout: The DRAM ranges ``shared_dram_layout`` agreed for the
            inputs ``model`` shares with other programs, such as a training
            step's gradient and update programs; ``None`` places every
            input freely.
        accumulate_fp32: Accumulate a GEMM split along K in float32 rather
            than its output dtype.
        bool_mask: Store an ``is_causal`` flash attention's mask table as
            int1 codes plus a scale; off, as the additive fill at the
            accumulation dtype.

    Returns:
        The ``voyager`` IR ``Model`` written to ``model.txt``.
    """
    from voyager_compiler.pipeline import compile_bufferized

    return compile_bufferized(
        model,
        example_args,
        example_kwargs,
        config,
        output_dir,
        output_file,
        dump_tensors,
        runtime_tolerance,
        dram_layout,
        accumulate_fp32,
        bool_mask,
    ).model

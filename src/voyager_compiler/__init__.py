# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import os

import torch
from google.protobuf import text_format
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._pytree import tree_flatten

# isort: off
# Import order here is load-bearing, not cosmetic.  ``decomposed`` defines the
# ``quantized_ops`` torch Library, and 22 modules under ``codegen`` resolve
# ``torch.ops.quantized_ops.*`` at module scope.  Alphabetising this block puts
# ``codegen`` first and those lookups fail with AttributeError at import time.
from voyager_compiler.pt2e_utils import (
    WrapperModule,
    convert_scalars_to_attrs,
    deduplicate_nodes,
    dispatch_model,
    dtype_byte_size,
    fetch_attr,
    get_aten_graph_module,
    get_device_map,
    get_node_name_to_scope,
    insert_align_device_nodes,
    new_attr_name,
    print_node_scope_tabular,
    sink_obs_or_fq,
)
from voyager_compiler.decomposed import (
    F,
    Library,
    Union,
    adaptive_avg_pool2d,
    calculate_mx_qparam,
    conv2d,
    conv2d_mx,
    dequantize,
    dequantize_vsq,
    expand,
    filter_outlier,
    impl,
    increment_indices,
    layer_norm,
    linear,
    linear_mx,
    load_tile,
    matmul,
    matmul_mx,
    max_pool2d,
    quantize_mx,
    quantize_mx_outlier,
    quantize_vsq,
    quantized_decomposed_lib,
    slice_csr_tensor,
    spmm_csr,
    store_tile,
    vmap,
)
from voyager_compiler.codegen import (
    COMPILATION_TARGET_META,
    COMPILATION_TARGETS,
    DEFAULT_COMPILATION_TARGET,
    DEFAULT_EXECUTION_ORDER_MODE,
    DP_EXECUTION_ORDER_MODE,
    END_TO_END_SCRATCHPAD_MAPPING,
    EXECUTION_ORDER_MODES,
    EXECUTION_ORDER_RESULT_META,
    FX_EXECUTION_ORDER_MODE,
    GEMMINI_COMPILATION_TARGET,
    GEMMINI_HARDWARE_CONFIG,
    INTERSTELLAR_SCHEDULER,
    LOCAL_SCHEDULERS,
    MAGNET_ALPHA_COMPILATION_TARGET,
    MAGNET_ALPHA_RULE_BASED_SCHEDULER,
    PER_OPERATION_SCRATCHPAD_MAPPING,
    SCRATCHPAD_MAPPING_MODES,
    CollectorConfig,
    ComputeUnit,
    DTypeMode,
    ExecutionOrder,
    GemminiHardwareConfig,
    HardwareConfig,
    HierarchyBandwidth,
    MagnetAlphaHardwareConfig,
    MemoryAllocator,
    MemoryInstance,
    MemoryLevelConfig,
    OpMatcher,
    PEConfig,
    ScratchpadAllocator,
    ShapeProp,
    TensorRole,
    UnrollDimension,
    apply_target_graph_rewrites,
    apply_target_memory_rewrites,
    convert_cat_and_stack_as_stack_on_dim0,
    convert_cat_with_mismatched_shapes_to_stack,
    convert_expand_to_memory_copy,
    create_gemmini_hardware_config,
    create_scratchpad_allocator,
    dump_compile_artifacts,
    dump_hardware_config,
    dump_performance_estimates,
    dump_storage_plan,
    eliminate_reshape_with_no_effect,
    extract_input_preprocessor,
    fuse_operator,
    gen_code,
    gen_compute_graph,
    get_conv_bn_layers,
    get_op_overload,
    get_target_execution_fusion_patterns,
    get_target_hardware_pipeline,
    get_target_operator_traits,
    inline_autocast_modules,
    lower_target_quantization,
    pad_matrix_op_dimensions,
    pad_vector_op_dimensions,
    pad_vit_embeddings_output,
    prepare_output_directory,
    prepare_target_execution_fusion,
    remove_softmax_dtype_cast,
    rename_nodes_with_param_names,
    replace_conv2d_with_im2col,
    replace_interpolate,
    replace_rmsnorm_with_layer_norm,
    rewrite_fx_graph,
    run_matrix_op_l2_tiling,
    run_memory_mapping,
    run_pool_op_l2_tiling,
    run_vector_op_l2_tiling,
    serialize_hardware_config,
    split_multi_head_attention,
    target_allows_qdq_fusion,
    target_supports_bufferized,
    target_uses_generic_aggregate_copies,
    transpose_conv2d_inputs_and_weights,
    transpose_linear_weights,
    validate_execution_order_mode,
)
from voyager_compiler.codegen.mapping_utils import TensorWriteSession
from voyager_compiler.fake_quantize import (
    _DerivedObserverOrFakeQuantize,
    get_vsq_scale_quantization_map,
)
from voyager_compiler.fp8 import (
    _quantize_elemwise_core,
    quantize_to_fp8_e4m3,
    quantize_to_fp8_e5m2,
)
from voyager_compiler.histogram import (
    defaultdict,
    get_grouped_histogram,
    make_interp_spline,
    np,
    os,
    plot_histogram,
    plot_layer_range,
    plt,
)
from voyager_compiler.llm_utils import (
    TorchExportableModuleWithStaticCache,
    convert_and_export_with_split_cache,
    fuse_dequantize_quantize,
    generate,
    swap_llama_attention,
)
from voyager_compiler.normal_float import (
    create_normal_map,
    quantize_to_nf,
)
from voyager_compiler.posit import quantize_to_posit
from voyager_compiler.qconfig import (
    QConfig,
    get_qconfig,
)
from voyager_compiler.quantize import (
    convert,
    prepare,
    propagate_config,
    quantize,
    replace_softmax,
)
from voyager_compiler.quantize_pt2e import (
    COMPILER_ONLY,
    MX_OP_MAPPING,
    PHYSICAL_QUANTIZATION_CONTRACT,
    QUANTIZATION_DOMAIN_DTYPE,
    QUANTIZATION_REGION,
    SOURCE_OP_NAME,
    Any,
    DerivedQuantizationSpec,
    Dict,
    EdgeOrNode,
    FusedAmaxObsFakeQuantize,
    Graph,
    GraphModule,
    List,
    Node,
    ObserverOrFakeQuantize,
    Optional,
    OrderedDict,
    PhysicalQuantizationContract,
    QuantizationConfig,
    QuantizationSpecBase,
    Tensor,
    Tuple,
    XNNPACKQuantizer,
    assert_and_get_unique_device,
    bind_compilation_target,
    cleanup_target_quantization_preparation,
    convert_pt2e,
    copy,
    create_getattr_from_value,
    derive_bias_qparams_fn,
    export_model,
    finalize_pt2e_calibration,
    finalize_target_quantization_calibration,
    fuse_quantize_dequantize_with_previous_op,
    get_arg_value,
    get_default_quantizer,
    get_microscaling_quantizer,
    get_per_channel_act_quantizer,
    get_quantization_map,
    get_vsq_quantizer,
    inherit_physical_quantization_contract,
    is_conv2d,
    is_gemm_op,
    is_indexing_or_concatenation_op,
    is_matmul,
    is_nop,
    is_reshape_op,
    logger,
    logging,
    math,
    operator,
    prepare_pt2e,
    prepare_qat_pt2e,
    prepare_target_quantization,
    propagate_shape,
    qt,
    re,
    replace,
    resolve_compilation_target,
    set_physical_quantization_contract,
    swap_matmul_inputs,
    torch,
    validate_target_hardware_config,
)
from voyager_compiler.quantizer import (
    QScheme,
    QuantizationSpec,
)
from scheduler.types import (
    NUM_LOOPS,
    GemmTilingCandidateCheck,
    GemmWorkload,
    LevelAccessCounts,
    LevelAccessEvents,
    LocalScheduleResult,
    Loop,
    MappingPoint,
    OperandAccessEvents,
    OperationClassification,
    OperationOperand,
    OperationWorkload,
    SchedulerProviderSet,
)
from voyager_compiler.training_args import add_qspec_args
from voyager_compiler.utils import with_execution_context

# isort: on

__all__ = [
    "FusedAmaxObsFakeQuantize",
    "OpMatcher",
    "QConfig",
    "QuantizationSpec",
    "TorchExportableModuleWithStaticCache",
    "add_qspec_args",
    "convert_and_export_with_split_cache",
    "convert",
    "deduplicate_nodes",
    "derive_bias_qparams_fn",
    "dispatch_model",
    "dtype_byte_size",
    "export_model",
    "finalize_pt2e_calibration",
    "fetch_attr",
    "generate",
    "get_aten_graph_module",
    "get_device_map",
    "get_node_name_to_scope",
    "get_op_overload",
    "get_qconfig",
    "get_default_quantizer",
    "insert_align_device_nodes",
    "plot_histogram",
    "plot_layer_range",
    "prepare",
    "prepare_pt2e",
    "prepare_qat_pt2e",
    "print_node_scope_tabular",
    "propagate_config",
    "propagate_shape",
    "quantize",
    "quantize_to_fp8_e4m3",
    "quantize_to_fp8_e5m2",
    "quantize_to_nf",
    "quantize_to_posit",
    "replace_softmax",
    "sink_obs_or_fq",
    "swap_llama_attention",
    "with_execution_context",
]


class qscheme: ...


# Defined in voyager_compiler/quantizer.h
per_tensor_symmetric: qscheme = QScheme.PER_TENSOR_SYMMETRIC
per_channel_symmetric: qscheme = QScheme.PER_CHANNEL_SYMMETRIC
microscaling: qscheme = QScheme.MICROSCALING
group_wise_affine: qscheme = QScheme.GROUP_WISE_AFFINE
vsq: qscheme = QScheme.VSQ


# Private compatibility spelling retained for callers that used the old helper
# before it moved into the leaf module.
_get_op_overload = get_op_overload


def fuse(
    model,
    fusion_pattern,
    example_args,
    example_kwargs=None,
    fuse_reshape=True,
    fake_mode=None,
    require_all_pattern_stages=False,
    operator_traits=None,
):
    if example_kwargs is None:
        example_kwargs = {}

    flatten_args, _ = tree_flatten((example_args, example_kwargs))
    ShapeProp(model, mode=fake_mode).propagate(*flatten_args)

    fuse_operator(
        model,
        fusion_pattern,
        fuse_reshape,
        require_all_pattern_stages=require_all_pattern_stages,
        operator_traits=operator_traits,
    )
    return model


def transform(
    model: torch.fx.GraphModule,
    example_args,
    example_kwargs=None,
    patterns=None,
    unroll_dims=None,
    transform_layout=False,
    transpose_fc=False,
    cache_size=None,
    num_banks=None,
    fuse_operator=True,
    fuse_reshape=True,
    split_spmm=False,
    use_fake_mode=True,
    hardware_config: HardwareConfig = None,
):
    target = resolve_compilation_target(hardware_config)
    validate_target_hardware_config(target, hardware_config)
    bind_compilation_target(
        model,
        hardware_config,
        stage="transform",
    )
    operator_traits = get_target_operator_traits(target)
    if hardware_config is not None:
        legacy_values = (cache_size, num_banks, unroll_dims)
        if any(value is not None for value in legacy_values):
            raise ValueError(
                "hardware_config cannot be combined with legacy transform "
                "hardware arguments"
            )
        unroll_dims = hardware_config.unroll_dims
        num_banks = hardware_config.global_num_banks

    if example_kwargs is None:
        example_kwargs = {}

    fake_mode = (
        FakeTensorMode(allow_non_fake_inputs=True) if use_fake_mode else None
    )

    flatten_args, _ = tree_flatten((example_args, example_kwargs))
    apply_target_graph_rewrites(model, target, hardware_config)
    lower_target_quantization(model, target, hardware_config)
    ShapeProp(model, mode=fake_mode).propagate(*flatten_args)

    # -------------------------------------------------------------------------
    # 1. Lowering & Decomposition
    # -------------------------------------------------------------------------
    # Break down complex operators (like MultiHeadAttention) into simpler
    # primitives and handle memory copy/concat operations.

    split_multi_head_attention(model)

    # TODO Disabled for large models. This will be removed in the future once
    # we can handle stack/cat using DMA properly.
    if len(model.graph.nodes) < 10000:
        convert_expand_to_memory_copy(model)
        convert_cat_and_stack_as_stack_on_dim0(model)
        convert_cat_with_mismatched_shapes_to_stack(model)

    if target_allows_qdq_fusion(target):
        fuse_quantize_dequantize_with_previous_op(model)
    # -------------------------------------------------------------------------
    # 2. Hardware Alignment (Padding)
    # -------------------------------------------------------------------------
    # Pad dimensions to align with hardware unrolling constraints (SIMD,
    # systolic array dimensions, etc.) to ensure efficient execution.
    if unroll_dims is not None:
        if hardware_config is None:
            pad_matrix_op_dimensions(model, *unroll_dims)
        else:
            pad_matrix_op_dimensions(
                model,
                hardware_config=hardware_config,
                operator_traits=operator_traits,
            )
        pad_vector_op_dimensions(model, unroll_dims[1])

    # -------------------------------------------------------------------------
    # 3. Matrix Operation Tiling
    # -------------------------------------------------------------------------
    # Apply L2 tiling logic specifically for matrix operations (GEMM/Conv) to
    # optimize for the specific cache size and memory bank configuration.
    if cache_size is not None or hardware_config is not None:
        run_matrix_op_l2_tiling(
            model,
            unroll_dims if hardware_config is None else None,
            cache_size,
            num_banks,
            hardware_config=hardware_config,
            operator_traits=operator_traits,
        )

    # -------------------------------------------------------------------------
    # 4. Data Layout Transformation
    # -------------------------------------------------------------------------
    # Transform GEMM and convolution inputs/weights into layouts friendly
    # for systolic-array based hardware (e.g., transposing weights).

    if transform_layout:
        transpose_conv2d_inputs_and_weights(
            model,
            operator_traits,
        )

    transpose_linear_weights(
        model,
        transform_layout,
        transpose_fc,
        operator_traits=operator_traits,
    )

    ShapeProp(model, mode=fake_mode).propagate(*flatten_args)

    # Remove redundant reshapes that have no effect on tensor semantics
    eliminate_reshape_with_no_effect(model)

    # -------------------------------------------------------------------------
    # 5. Vector Operation Tiling
    # -------------------------------------------------------------------------
    # TODO: Used for unit test. Will be removed in the future
    from voyager_compiler.codegen.passes.lowering import split_dense_spmm_node

    if split_spmm:
        split_dense_spmm_node(model)

    # Apply L2 tiling logic for vector-based operations.
    if cache_size is not None or hardware_config is not None:
        run_pool_op_l2_tiling(
            model,
            unroll_dims[1],
            cache_size,
            num_banks,
            hardware_config=hardware_config,
        )
        run_vector_op_l2_tiling(
            model,
            unroll_dims[1],
            cache_size,
            num_banks,
            hardware_config=hardware_config,
            operator_traits=operator_traits,
        )

    # -------------------------------------------------------------------------
    # 6. Operator Fusion
    # -------------------------------------------------------------------------
    # Perform final operator lowering and fuse sequences of operations (e.g.,
    # Conv+ReLU) into single kernels to reduce memory access overhead.

    prepare_target_execution_fusion(model, target, hardware_config)
    rename_nodes_with_param_names(model, operator_traits)

    execution_patterns = [
        [OpMatcher(*operators) for operators in pattern]
        for pattern in get_target_execution_fusion_patterns(target)
    ]
    if execution_patterns:
        # These groups define target-required execution regions rather than
        # optional performance fusions. Their physical call or instruction
        # expansion remains target-owned. Keep the groups intact even when
        # generic operator fusion is disabled.
        fuse(
            model,
            execution_patterns,
            flatten_args,
            fuse_reshape=False,
            fake_mode=fake_mode,
            require_all_pattern_stages=True,
            operator_traits=operator_traits,
        )

    if fuse_operator:
        fuse(
            model,
            patterns,
            flatten_args,
            fuse_reshape=fuse_reshape,
            fake_mode=fake_mode,
            operator_traits=operator_traits,
        )

    deduplicate_nodes(model)

    return model


def compile(
    model: torch.fx.GraphModule,
    example_args,
    example_kwargs=None,
    total_memory=None,
    cache_size=None,
    num_banks=None,
    bank_width=None,
    unroll_dims=None,
    output_dir=None,
    output_file="compute_graph",
    dump_tensors=True,
    dump_snapshot=False,
    double_buffered_accum_buffer: bool = False,
    input_buffer_size: int = None,
    weight_buffer_size: int = None,
    accum_buffer_size: int = None,
    hardware_config: HardwareConfig = None,
    scheduler_backend: str | None = None,
    scheduler_options=None,
    bufferize: bool = False,
    scratchpad_mapping: str = PER_OPERATION_SCRATCHPAD_MAPPING,
    execution_order: str = DEFAULT_EXECUTION_ORDER_MODE,
):
    validate_execution_order_mode(
        execution_order,
        bufferized=bufferize,
        end_to_end=(scratchpad_mapping == END_TO_END_SCRATCHPAD_MAPPING),
    )

    target = resolve_compilation_target(hardware_config)
    validate_target_hardware_config(target, hardware_config)
    legacy_scheduler_configured = unroll_dims is not None and all(
        size is not None
        for size in (
            input_buffer_size,
            weight_buffer_size,
            accum_buffer_size,
        )
    )
    if bufferize:
        if scratchpad_mapping != PER_OPERATION_SCRATCHPAD_MAPPING:
            raise NotImplementedError(
                "The bufferized backend does not support model-wide "
                "scratchpad mapping"
            )
        if not target_supports_bufferized(target):
            raise NotImplementedError(
                f"{target} does not support bufferized compilation"
            )
        if (
            hardware_config is not None
            or scheduler_backend is not None
            or scheduler_options is not None
            or legacy_scheduler_configured
            or double_buffered_accum_buffer
        ):
            raise NotImplementedError(
                "The bufferized backend cannot preserve local scheduler "
                "results. Use bufferized lowering without a hardware "
                "profile, scheduler backend/options, complete legacy local "
                "buffer configuration, or scheduler double buffering."
            )
    bind_compilation_target(
        model,
        hardware_config,
        stage="compile",
    )
    operator_traits = get_target_operator_traits(target)
    if hardware_config is not None:
        legacy_values = (
            cache_size,
            num_banks,
            bank_width,
            unroll_dims,
            input_buffer_size,
            weight_buffer_size,
            accum_buffer_size,
        )
        if any(value is not None for value in legacy_values):
            raise ValueError(
                "hardware_config cannot be combined with legacy hardware "
                "arguments"
            )
        unroll_dims = hardware_config.unroll_dims
        num_banks = hardware_config.global_num_banks
        bank_width = hardware_config.memory_alignment_bytes

    prepare_output_directory(output_dir)

    flatten_args, _ = tree_flatten((example_args, example_kwargs))
    ShapeProp(model).propagate(*flatten_args)

    # ---------------------------------------------------------------------
    # Optional add-on: bufferized FX lowering + loop-aware code generation.
    # Rewrites tiled GEMM/pointwise nodes into explicit while_loop nests over
    # buf.* primitives, then emits protobuf / graph / text from that FX graph.
    # Terminal alternative to the per-node run_memory_mapping + gen_code path.
    # ---------------------------------------------------------------------
    if bufferize:
        from voyager_compiler.codegen.lowering import (
            bufferize_graph,
            gen_code_bufferized,
            gen_compute_graph_bufferized,
            print_bufferized_graph,
        )
        from voyager_compiler.codegen.mapping import adjust_tiling

        # Fit each fused submodule's or tiled op's L2 tiling to the
        # scratchpad before bufferizing, so loop nests use SRAM-fit tile
        # sizes. ``adjust_tiling`` parks the node-keyed result on each node;
        # its returned map is unused because this path does no per-node
        # memory allocation. ``run_memory_mapping`` is skipped.
        ShapeProp(model).propagate(*flatten_args)
        named = dict(model.named_modules(remove_duplicate=False))
        for node in list(model.graph.nodes):
            if node.op == "call_module" or "tiled_shapes" in node.meta:
                adjust_tiling(
                    node,
                    named,
                    cache_size,
                    num_banks,
                    bank_width,
                    unroll_dims,
                    hardware_config=hardware_config,
                    operator_traits=operator_traits,
                )

        bufferize_graph(model)
        print_bufferized_graph(model)

        path = os.path.join(output_dir, "tensor_files")
        with TensorWriteSession():
            params = gen_code_bufferized(
                model, flatten_args, path if dump_tensors else None
            )
        with open(os.path.join(output_dir, "model.txt"), "w") as f:
            f.write(text_format.MessageToString(params))
        with open(os.path.join(output_dir, "bufferized_graph.txt"), "w") as f:
            f.write(print_bufferized_graph(model, to_string=True))
        if len(model.graph.nodes) < 10000:
            gen_compute_graph_bufferized(
                model, os.path.join(output_dir, output_file)
            )
        dump_compile_artifacts(
            model,
            hardware_config,
            output_dir,
        )
        return params

    allocator = MemoryAllocator(
        total_memory,
        bank_width=None if hardware_config is not None else bank_width,
    )
    scratchpad_allocator = (
        create_scratchpad_allocator(hardware_config)
        if (
            hardware_config is not None
            and scratchpad_mapping == END_TO_END_SCRATCHPAD_MAPPING
        )
        else None
    )
    run_memory_mapping(
        model,
        allocator,
        cache_size,
        num_banks,
        bank_width,
        unroll_dims,
        double_buffered_accum_buffer=double_buffered_accum_buffer,
        input_buffer_size=input_buffer_size,
        weight_buffer_size=weight_buffer_size,
        accum_buffer_size=accum_buffer_size,
        hardware_config=hardware_config,
        scheduler_backend=scheduler_backend,
        scheduler_options=scheduler_options,
        scratchpad_allocator=scratchpad_allocator,
        scratchpad_mapping=scratchpad_mapping,
        execution_order=execution_order,
    )

    if dump_snapshot:
        allocator.dump_snapshots(os.path.join(output_dir, "memory.png"))
        if scratchpad_allocator is not None:
            scratchpad_allocator.dump_snapshots(
                os.path.join(output_dir, "scratchpad.png")
            )

    path = os.path.join(output_dir, "tensor_files")
    with TensorWriteSession():
        params = gen_code(
            model,
            flatten_args,
            path if dump_tensors else None,
        )

    dump_compile_artifacts(
        model,
        hardware_config,
        output_dir,
    )

    with open(os.path.join(output_dir, "model.txt"), "w") as f:
        f.write(text_format.MessageToString(params))

    operations = [
        (
            operation.op.name
            if operation.WhichOneof("op_type") == "op"
            else operation.fused_op.name
        )
        for operation in params.ops
        if operation.op.op != "nop"
    ]

    with open(os.path.join(output_dir, "layers.txt"), "w") as f:
        f.write("\n".join(operations))

    if len(model.graph.nodes) < 10000:
        gen_compute_graph(model, os.path.join(output_dir, output_file))

    return params

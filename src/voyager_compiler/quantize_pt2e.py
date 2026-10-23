# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import copy
import logging
import math
import operator
import re
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.ao.quantization import ObserverOrFakeQuantize
from torch.ao.quantization.fx.utils import assert_and_get_unique_device
from torch.ao.quantization.quantizer import EdgeOrNode, QuantizationSpecBase
from torch.fx import Graph, GraphModule, Node

import voyager_compiler as qt
from voyager_compiler import quantization_regions as _quantization_regions
from voyager_compiler.codegen.mapping_utils import (
    SOURCE_OP_NAME,
    is_conv2d,
    is_gemm_op,
    is_indexing_or_concatenation_op,
    is_matmul,
    is_nop,
    is_reshape_op,
)
from voyager_compiler.codegen.passes.utils import get_arg_value
from voyager_compiler.codegen.target_backends import (
    cleanup_target_quantization_preparation,
    finalize_target_quantization_calibration,
    prepare_target_quantization,
    validate_target_hardware_config,
)
from voyager_compiler.codegen.targets import (
    bind_compilation_target,
    resolve_compilation_target,
)
from voyager_compiler.fake_quantize import (
    FusedAmaxObsFakeQuantize,
    _DerivedObserverOrFakeQuantize,
    get_quantization_map,
)
from voyager_compiler.pt2e_utils import (
    COMPILER_ONLY,
    create_getattr_from_value,
    export_model,
    propagate_shape,
)
from voyager_compiler.quantization_contracts import (
    PHYSICAL_QUANTIZATION_CONTRACT,
    QUANTIZATION_DOMAIN_DTYPE,
    QUANTIZATION_REGION,
    PhysicalQuantizationContract,
    inherit_physical_quantization_contract,
    set_physical_quantization_contract,
)
from voyager_compiler.quantizer.quantizer import (
    DerivedQuantizationSpec,
    QuantizationSpec,
)
from voyager_compiler.quantizer.xnnpack_quantizer import XNNPACKQuantizer
from voyager_compiler.quantizer.xnnpack_quantizer_utils import (
    QuantizationConfig,
)

# Keep the former private import surface while the new module owns the code.
_CONFLICTING_PREPARED_DOMAIN = (
    _quantization_regions._CONFLICTING_PREPARED_DOMAIN
)
_PREPARED_DOMAIN_JOIN_TARGETS = (
    _quantization_regions._PREPARED_DOMAIN_JOIN_TARGETS
)
_PREPARED_DOMAIN_PASSTHROUGH_TARGETS = (
    _quantization_regions._PREPARED_DOMAIN_PASSTHROUGH_TARGETS
)
_PREPARED_DOMAIN_REGION_ROOT_TARGETS = (
    _quantization_regions._PREPARED_DOMAIN_REGION_ROOT_TARGETS
)
_PREPARED_DOMAIN_TARGET_RULE = (
    _quantization_regions._PREPARED_DOMAIN_TARGET_RULE
)
_QUANTIZATION_DOMAIN_ALIAS = _quantization_regions._QUANTIZATION_DOMAIN_ALIAS
_QUANTIZATION_DOMAIN_ENTRY = _quantization_regions._QUANTIZATION_DOMAIN_ENTRY
_UNKNOWN_PREPARED_DOMAIN = _quantization_regions._UNKNOWN_PREPARED_DOMAIN
_VSQ_JOIN_ALIAS = _quantization_regions._VSQ_JOIN_ALIAS
_VSQ_JOIN_ENTRY = _quantization_regions._VSQ_JOIN_ENTRY
_analyze_prepared_quantization_regions = (
    _quantization_regions._analyze_prepared_quantization_regions
)
_bind_prepared_domain_target_rules = (
    _quantization_regions._bind_prepared_domain_target_rules
)
_create_obs_or_fq_from_qspec = (
    _quantization_regions._create_obs_or_fq_from_qspec
)
_effective_prepared_domain_rule_set = (
    _quantization_regions._effective_prepared_domain_rule_set
)
_finalize_quantization_regions = (
    _quantization_regions._finalize_quantization_regions
)
_find_prepared_data_movement_regions = (
    _quantization_regions._find_prepared_data_movement_regions
)
_get_module = _quantization_regions._get_module
_get_prepared_per_tensor_observer = (
    _quantization_regions._get_prepared_per_tensor_observer
)
_get_prepared_vsq_observer = _quantization_regions._get_prepared_vsq_observer
_insert_prepared_observer = _quantization_regions._insert_prepared_observer
_is_chunk_getitem = _quantization_regions._is_chunk_getitem
_is_prepared_vsq_output_boundary = (
    _quantization_regions._is_prepared_vsq_output_boundary
)
_is_zero_constant_pad = _quantization_regions._is_zero_constant_pad
_merge_prepared_quantization_plans = (
    _quantization_regions._merge_prepared_quantization_plans
)
_move_output_observers_after_epilogues = (
    _quantization_regions._move_output_observers_after_epilogues
)
_node_tensor_shape = _quantization_regions._node_tensor_shape
_normalize_axis = _quantization_regions._normalize_axis
_prepare_quantization_regions = (
    _quantization_regions._prepare_quantization_regions
)
_prepare_vsq_join_regions = _quantization_regions._prepare_vsq_join_regions
_prepared_domain_data_inputs = (
    _quantization_regions._prepared_domain_data_inputs
)
_prepared_domain_rule_set = _quantization_regions._prepared_domain_rule_set
_selected_quantization_output_epilogues = (
    _quantization_regions._selected_quantization_output_epilogues
)
_prepared_domains_compatible = (
    _quantization_regions._prepared_domains_compatible
)
_prepared_vsq_join_exit = _quantization_regions._prepared_vsq_join_exit
_prepared_vsq_join_has_legal_layout = (
    _quantization_regions._prepared_vsq_join_has_legal_layout
)
_prepared_vsq_join_input = _quantization_regions._prepared_vsq_join_input
_selected_prepared_domain_rules = (
    _quantization_regions._selected_prepared_domain_rules
)
_vsq_physical_quantization_contract = (
    _quantization_regions._vsq_physical_quantization_contract
)

logger = logging.getLogger(__name__)


def _get_obs_or_fq_map(
    edge_or_node_to_group_id: Dict[EdgeOrNode, int],
    edge_or_node_to_qspec: Dict[EdgeOrNode, QuantizationSpecBase],
    is_qat: bool,
) -> Dict[EdgeOrNode, ObserverOrFakeQuantize]:
    """Generates the EdgeOrNode to observer/fake_quant instances
    EdgeOrNode values with the same group ID share the same observer or fake
    quantizer instance.
    """
    obs_or_fq_map: Dict[EdgeOrNode, ObserverOrFakeQuantize] = {}
    group_id_to_obs_or_fq: Dict[int, ObserverOrFakeQuantize] = {}
    for edge_or_node, qspec in edge_or_node_to_qspec.items():
        group_id = edge_or_node_to_group_id[edge_or_node]
        if group_id not in group_id_to_obs_or_fq:
            # TODO: use a root qspec map to simplify observer construction.
            group_id_to_obs_or_fq[group_id] = _create_obs_or_fq_from_qspec(
                qspec,
                obs_or_fq_map,
                is_qat,
            )
        obs_or_fq_map[edge_or_node] = group_id_to_obs_or_fq[group_id]
    return obs_or_fq_map


def _set_ch_axis(qspec: Optional[QuantizationSpec], ch_axis: int):
    if qspec is None:
        return None
    return replace(qspec, ch_axis=ch_axis)


def get_microscaling_quantizer(
    activation: Optional[QuantizationSpec],
    weight: Optional[QuantizationSpec],
):
    # Microscaling performs quantization along the reduction dimension
    act_qspec = _set_ch_axis(activation, 1)
    weight_qspec = _set_ch_axis(weight, 1)
    qconfig_conv2d = QuantizationConfig(act_qspec, None, weight_qspec, None)

    act_qspec = _set_ch_axis(activation, -1)
    weight_qspec = _set_ch_axis(weight, -1)
    qconfig_linear = QuantizationConfig(act_qspec, None, weight_qspec, None)

    act0_qspec = _set_ch_axis(activation, -1)
    act1_qspec = _set_ch_axis(activation, -2)
    qconfig_matmul = QuantizationConfig(act0_qspec, None, act1_qspec, None)

    return (
        XNNPACKQuantizer()
        .set_object_type(torch.ops.aten.conv2d.default, qconfig_conv2d)
        .set_object_type(torch.ops.aten.linear.default, qconfig_linear)
        .set_object_type(torch.ops.aten.matmul.default, qconfig_matmul)
    )


def get_vsq_quantizer(
    activation: Optional[QuantizationSpec],
    output_activation: Optional[QuantizationSpec],
    weight: Optional[QuantizationSpec],
):
    """Configure VSQ along each GEMM reduction dimension."""
    act_qspec = _set_ch_axis(activation, 1)
    weight_qspec = _set_ch_axis(weight, 1)
    if (
        weight_qspec is not None
        and weight_qspec.coarse_qscheme == qt.per_channel_symmetric
    ):
        weight_qspec = replace(weight_qspec, coarse_ch_axis=0)
    output_qspec = _set_ch_axis(output_activation, 1)
    qconfig_conv2d = QuantizationConfig(
        act_qspec,
        output_qspec,
        weight_qspec,
        None,
    )

    act_qspec = _set_ch_axis(activation, -1)
    weight_qspec = _set_ch_axis(weight, -1)
    if (
        weight_qspec is not None
        and weight_qspec.coarse_qscheme == qt.per_channel_symmetric
    ):
        weight_qspec = replace(weight_qspec, coarse_ch_axis=0)
    output_qspec = _set_ch_axis(output_activation, -1)
    qconfig_linear = QuantizationConfig(
        act_qspec,
        output_qspec,
        weight_qspec,
        None,
    )

    act0_qspec = _set_ch_axis(activation, -1)
    act1_qspec = _set_ch_axis(activation, -2)
    output_qspec = _set_ch_axis(output_activation, -1)
    qconfig_matmul = QuantizationConfig(
        act0_qspec,
        output_qspec,
        act1_qspec,
        None,
    )

    return (
        XNNPACKQuantizer()
        .set_object_type(torch.ops.aten.conv2d.default, qconfig_conv2d)
        .set_object_type(torch.ops.aten.linear.default, qconfig_linear)
        .set_object_type(torch.ops.aten.matmul.default, qconfig_matmul)
    )


def get_per_channel_act_quantizer(
    input_activation: Optional[QuantizationSpec],
    output_activation: Optional[QuantizationSpec],
    weight: Optional[QuantizationSpec],
    bias: Optional[QuantizationSpec],
):
    # Convolution layer only support per-tensor activation quantization
    act_qspec = replace(input_activation, qscheme=qt.per_tensor_symmetric)
    qconfig_conv2d = QuantizationConfig(
        act_qspec,
        output_activation,
        weight,
        bias,
    )

    # Perform quantization along the outer dimension
    act_qspec = replace(input_activation, ch_axis=-2)
    qconfig_linear = QuantizationConfig(
        act_qspec,
        output_activation,
        weight,
        bias,
    )

    act0_qspec = replace(input_activation, ch_axis=-2)
    act1_qspec = replace(input_activation, ch_axis=-1)
    qconfig_matmul = QuantizationConfig(
        act0_qspec,
        output_activation,
        act1_qspec,
        None,
    )

    return (
        XNNPACKQuantizer()
        .set_object_type(torch.ops.aten.conv2d.default, qconfig_conv2d)
        .set_object_type(torch.ops.aten.linear.default, qconfig_linear)
        .set_object_type(torch.ops.aten.matmul.default, qconfig_matmul)
    )


def derive_bias_qparams_fn(
    obs_or_fqs: List[ObserverOrFakeQuantize],
) -> Tuple[Tensor, Tensor]:
    assert len(obs_or_fqs) == 2, (
        "Expected two observers/fake quantizers, one for activation and one "
        f"for weight; got {len(obs_or_fqs)}"
    )
    act_obs_or_fq = obs_or_fqs[0]
    weight_obs_or_fq = obs_or_fqs[1]
    act_scale = act_obs_or_fq.calculate_qparams()
    weight_scale = weight_obs_or_fq.calculate_qparams()
    return act_scale * weight_scale.flatten()


def get_default_quantizer(
    input_activation: Optional[QuantizationSpec] = None,
    output_activation: Optional[QuantizationSpec] = None,
    weight: Optional[QuantizationSpec] = None,
    bias: Optional[QuantizationSpec] = None,
    record_histogram: bool = False,
    force_scale_power_of_two: bool = False,
    **kwargs: Any,
) -> XNNPACKQuantizer:
    """
    Create a quantizer for the activation and weight quantization specs.

    Parameters:
    - activation: The quantization spec for activations.
    - weight: The quantization spec for weights.
    - record_histogram: Whether to record histogram of input.
    - force_scale_power_of_two: Whether to use a power-of-two scale.

    Returns:
    - A configured XNNPACKQuantizer.
    """

    observer_or_fake_quant_ctr = FusedAmaxObsFakeQuantize.with_args(
        record_histogram=record_histogram,
        force_scale_power_of_two=force_scale_power_of_two,
    )

    qschemes = []
    if input_activation is not None:
        input_activation = QuantizationSpec.from_str(input_activation)
        input_activation.observer_or_fake_quant_ctr = observer_or_fake_quant_ctr
        qschemes.append(input_activation.qscheme)

    if output_activation is not None:
        output_activation = QuantizationSpec.from_str(output_activation)
        output_activation.observer_or_fake_quant_ctr = (
            observer_or_fake_quant_ctr
        )

    if weight is not None:
        weight = QuantizationSpec.from_str(weight)
        weight.observer_or_fake_quant_ctr = observer_or_fake_quant_ctr
        qschemes.append(weight.qscheme)

    qschemes = [qs for qs in qschemes if qs is not None]
    if qt.vsq in qschemes:
        if any(qscheme != qt.vsq for qscheme in qschemes):
            raise ValueError("VSQ cannot be mixed with another qscheme.")
        if bias is not None:
            raise ValueError("VSQ keeps bias unquantized; bias must be None.")
        return get_vsq_quantizer(
            input_activation,
            output_activation,
            weight,
        )

    if len(qschemes) > 0 and qt.microscaling not in qschemes:
        assert bias is not None, (
            "Bias quantization is required when quantizing activations and "
            "weights."
        )

    # We will specify derived_from later in the quantizer.
    # We use bias data type to imply the accumulation data type for the output.
    if bias is not None:
        bias = DerivedQuantizationSpec(
            derived_from=None,
            derive_qparams_fn=derive_bias_qparams_fn,
            dtype=bias,
        )

    if qt.microscaling in qschemes:
        assert len(set(qschemes)) == 1, (
            f"Quantization scheme {qschemes[0]} does not work with "
            f"{qschemes[1]}"
        )
        return get_microscaling_quantizer(input_activation, weight)

    if weight is not None and weight.qscheme == qt.per_channel_symmetric:
        assert weight.ch_axis == 0, (
            "Per-channel weight quantization only supports quantizing output "
            "channel dimension (dim=0)."
        )

    if (
        input_activation is not None
        and input_activation.qscheme == qt.per_channel_symmetric
    ):
        return get_per_channel_act_quantizer(
            input_activation,
            output_activation,
            weight,
            bias,
        )

    conv_output = output_activation
    linear_output = output_activation
    if output_activation is not None and output_activation.qscheme == qt.vsq:
        conv_output = _set_ch_axis(output_activation, 1)
        linear_output = _set_ch_axis(output_activation, -1)

    qconfig = QuantizationConfig(input_activation, conv_output, weight, bias)
    qconfig_linear = QuantizationConfig(
        input_activation,
        linear_output,
        weight,
        bias,
    )
    qconfig_matmul = QuantizationConfig(
        input_activation,
        linear_output,
        input_activation,
        None,
    )
    return (
        XNNPACKQuantizer()
        .set_object_type(torch.ops.aten.conv2d.default, qconfig)
        .set_object_type(torch.ops.aten.linear.default, qconfig_linear)
        .set_object_type(torch.ops.aten.matmul.default, qconfig_matmul)
        .set_object_type(torch.ops.aten.add.Tensor, qconfig)
        .set_object_type(torch.ops.aten.add_.Tensor, qconfig)
    )


def finalize_pt2e_calibration(
    model: GraphModule,
    hardware_config=None,
) -> GraphModule:
    """Finalize observer scales after the last calibration invocation."""
    target = resolve_compilation_target(hardware_config)
    validate_target_hardware_config(target, hardware_config)
    bind_compilation_target(
        model,
        hardware_config,
        stage="finalize_pt2e_calibration",
    )
    finalize_target_quantization_calibration(model, target)
    finalized = set()
    for module in model.modules():
        if not isinstance(module, FusedAmaxObsFakeQuantize):
            continue
        if id(module) in finalized:
            continue
        module.calculate_qparams()
        finalized.add(id(module))
    return model


def _prepare_pt2e(
    model,
    quantizer,
    args=None,
    kwargs=None,
    dynamic_shapes=None,
    hardware_config=None,
    quantization_region_qspec: Optional[QuantizationSpec] = None,
    *,
    qat: bool,
    fold_conv_bn: bool,
):
    from torch.ao.quantization.pt2e import prepare
    from torch.ao.quantization.quantize_pt2e import (
        _disallow_eval_train,
        _get_node_name_to_scope,
    )
    from torch.ao.quantization.quantize_pt2e import (
        prepare_pt2e as torch_prepare_pt2e,
    )
    from torch.ao.quantization.quantize_pt2e import (
        prepare_qat_pt2e as torch_prepare_qat_pt2e,
    )

    stage = "prepare_qat_pt2e" if qat else "prepare_pt2e"
    target = resolve_compilation_target(hardware_config)
    validate_target_hardware_config(target, hardware_config)
    if not isinstance(fold_conv_bn, bool):
        raise TypeError("fold_conv_bn must be a bool")

    if not isinstance(model, GraphModule):
        model = export_model(model, args, kwargs, dynamic_shapes=dynamic_shapes)

    bind_compilation_target(
        model,
        hardware_config,
        stage=stage,
    )
    original_obs_or_fq_map = prepare._get_obs_or_fq_map
    prepare._get_obs_or_fq_map = _get_obs_or_fq_map
    try:
        if fold_conv_bn:
            prepare_impl = torch_prepare_qat_pt2e if qat else torch_prepare_pt2e
            model = prepare_impl(model, quantizer)
        else:
            original_graph_meta = model.meta
            node_name_to_scope = _get_node_name_to_scope(model)
            model = quantizer.transform_for_annotation(model)
            quantizer.annotate(model)
            quantizer.validate(model)
            # PyTorch's public PT2E helpers always fold Conv-BN. This direct
            # prepare path omits that step so callers retain a visible
            # BatchNorm boundary by default.
            model = prepare.prepare(
                model,
                node_name_to_scope,
                is_qat=qat,
                obs_or_fq_callback=quantizer.prepare_obs_or_fq_callback,
            )
            model.meta.update(original_graph_meta)
            model = _disallow_eval_train(model)
    finally:
        prepare._get_obs_or_fq_map = original_obs_or_fq_map
    bind_compilation_target(
        model,
        hardware_config,
        stage=stage,
    )
    return prepare_target_quantization(
        model,
        target,
        quantization_region_qspec,
        hardware_config,
    )


def prepare_pt2e(
    model,
    quantizer,
    args=None,
    kwargs=None,
    dynamic_shapes=None,
    hardware_config=None,
    quantization_region_qspec: Optional[QuantizationSpec] = None,
    fold_conv_bn: bool = False,
):
    """Prepare a model for PT2E conversion with optional Conv-BN folding."""
    return _prepare_pt2e(
        model,
        quantizer,
        args,
        kwargs,
        dynamic_shapes,
        hardware_config,
        quantization_region_qspec,
        qat=False,
        fold_conv_bn=fold_conv_bn,
    )


def prepare_qat_pt2e(
    model,
    quantizer,
    args=None,
    kwargs=None,
    dynamic_shapes=None,
    hardware_config=None,
    quantization_region_qspec: Optional[QuantizationSpec] = None,
    fold_conv_bn: bool = False,
):
    """Prepare a QAT graph with optional Conv-BN folding."""
    return _prepare_pt2e(
        model,
        quantizer,
        args,
        kwargs,
        dynamic_shapes,
        hardware_config,
        quantization_region_qspec,
        qat=True,
        fold_conv_bn=fold_conv_bn,
    )


def _materialize_unobserved_per_channel_scales(
    model: GraphModule,
    modules: Dict[str, torch.nn.Module],
) -> None:
    """Give unobserved per-channel qparams their broadcast shape."""
    materialized_shapes = {}

    for node in model.graph.nodes:
        if node.op != "call_module":
            continue
        module = _get_module(node, modules)
        if not isinstance(module, FusedAmaxObsFakeQuantize):
            continue
        if module.amax_history.numel() != 0:
            continue

        if module.is_per_channel:
            axis = module.ch_axis
            scale_kind = "per-channel"
        elif module.coarse_is_per_channel:
            axis = module.coarse_ch_axis
            scale_kind = "coarse per-channel"
        else:
            continue

        input_node = node.args[0]
        input_value = (
            input_node.meta.get("val") if isinstance(input_node, Node) else None
        )
        if not isinstance(input_value, torch.Tensor):
            raise ValueError(
                f"Cannot shape an uncalibrated {scale_kind} scale "
                "without the logical input tensor"
            )
        if not isinstance(axis, int):
            raise ValueError(f"{scale_kind} quantization requires one axis")
        if axis < 0:
            axis += input_value.ndim
        if not 0 <= axis < input_value.ndim:
            raise ValueError(
                f"{scale_kind} axis {axis} is invalid for rank-"
                f"{input_value.ndim} input"
            )

        channel_size = input_value.shape[axis]
        if not isinstance(channel_size, int):
            raise ValueError(
                f"Uncalibrated {scale_kind} quantization requires a "
                "static channel dimension"
            )
        scale_shape = [1] * input_value.ndim
        scale_shape[axis] = channel_size
        scale_shape = tuple(scale_shape)

        previous_shape = materialized_shapes.get(id(module))
        if previous_shape is not None and previous_shape != scale_shape:
            raise ValueError(
                "A shared per-channel observer cannot represent inputs "
                f"with shapes {previous_shape} and {scale_shape}"
            )
        materialized_shapes[id(module)] = scale_shape

        scale = module.scale.detach().clone()
        if scale.numel() == 1:
            scale = scale.expand(scale_shape).clone()
        elif scale.numel() == channel_size:
            scale = scale.reshape(scale_shape)
        else:
            raise ValueError(
                f"Cannot reshape {scale.numel()} {scale_kind} scale "
                f"values to {scale_shape}"
            )
        module.scale.resize_(scale_shape).copy_(scale)


def _insert_or_update_gemm_dequantize(
    model: GraphModule,
    user: Node,
    scale: Tensor,
    output_dtype: Optional[str],
    device: torch.device,
) -> None:
    graph = model.graph
    user.meta["dtype"] = output_dtype
    node_order = {node: index for index, node in enumerate(graph.nodes)}
    all_users = sorted(
        user.users,
        key=lambda node: node_order.get(node, float("inf")),
    )
    maybe_dequantize = all_users[0]

    if (
        maybe_dequantize.op == "call_function"
        and maybe_dequantize.target
        == torch.ops.quantized_ops.dequantize.default
    ):
        qparam_node = maybe_dequantize.args[1]
        buffer = model.get_buffer(qparam_node.target)
        model.register_buffer(qparam_node.target, scale * buffer)
        return

    quant_map = get_quantization_map(output_dtype, device)
    with graph.inserting_before(maybe_dequantize):
        qparam_node = create_getattr_from_value(
            model,
            graph,
            user.name + "_scale",
            scale,
        )
        get_attr_node = create_getattr_from_value(
            model,
            graph,
            "qmap",
            quant_map,
            compiler_only=True,
        )
        dequantized = graph.call_function(
            torch.ops.quantized_ops.dequantize.default,
            (
                user,
                qparam_node,
                None,
                None,
                None,
                get_attr_node,
            ),
            {},
        )

    for original_user in list(user.users):
        if original_user is not dequantized:
            original_user.replace_input_with(user, dequantized)


def _insert_dequantize_before_user(
    model: GraphModule,
    source: Node,
    user: Node,
    scale: Tensor,
) -> None:
    graph = model.graph
    with graph.inserting_before(user):
        qparam_node = create_getattr_from_value(
            model,
            graph,
            user.name + "_scale",
            scale,
        )
        dequantized = graph.call_function(
            torch.ops.quantized_ops.dequantize.default,
            (source, qparam_node),
            {},
        )
    user.replace_input_with(source, dequantized)


def _materialize_domain_users(
    model: GraphModule,
    observer: Node,
    source: Node,
    users: list[Node],
    scale: Tensor,
    dtype: str,
    output_dtype: Optional[str],
    device: torch.device,
) -> None:
    region = observer.meta.get(QUANTIZATION_REGION)
    for user in users:
        if region is not None and user.meta.get(QUANTIZATION_REGION) == region:
            user.meta["dtype"] = dtype
        elif is_gemm_op(user):
            _insert_or_update_gemm_dequantize(
                model,
                user,
                scale,
                output_dtype,
                device,
            )
        else:
            _insert_dequantize_before_user(
                model,
                source,
                user,
                scale,
            )


def _replace_domain_alias(
    model: GraphModule,
    node: Node,
    modules: Dict[str, torch.nn.Module],
    output_dtype: Optional[str],
) -> None:
    assert isinstance(node.target, str)
    activation_post_process = modules[node.target]
    device = assert_and_get_unique_device(activation_post_process)
    dtype = next(iter(model.parameters())).dtype
    scale = activation_post_process.calculate_qparams().to(dtype)
    users = list(node.users)
    source = node.args[0]
    node.replace_all_uses_with(source)
    model.graph.erase_node(node)
    source.meta["dtype"] = activation_post_process.dtype
    inherit_physical_quantization_contract(source, node)
    _materialize_domain_users(
        model,
        node,
        source,
        users,
        scale,
        activation_post_process.dtype,
        output_dtype,
        device,
    )


def _replace_observer_with_quantize_dequantize_node_decomposed(
    model: torch.fx.GraphModule,
    node: Node,
    modules: Dict[str, torch.nn.Module],
    output_dtype: str = None,
):
    graph = model.graph
    assert modules is not None
    assert isinstance(node.target, str)
    activation_post_process = modules[node.target]
    device = assert_and_get_unique_device(activation_post_process)

    dtype = next(iter(model.parameters())).dtype
    scale = activation_post_process.calculate_qparams().to(dtype)

    orig_fq_users = list(node.users.keys())
    input_node = node.args[0]
    if input_node.op == "get_attr":
        # Quantize weight and remove the fq module
        param = model.get_parameter(input_node.target)
        param.data = torch.ops.quantized_ops.quantize(
            param.data,
            scale,
            qmap=activation_post_process.qmap,
        )
        node.replace_all_uses_with(input_node)

        # Annotate weight dtype
        input_node.meta["dtype"] = activation_post_process.dtype

        # Reshape the scale to match the shape of the output tensor for
        # per-channel weight quantization.
        if scale.ndim == 4:
            scale = scale.view(-1, 1, 1)
        elif scale.ndim == 2:
            scale = scale.view(-1)
    else:
        # Replace fake quant module with a quantize node
        with graph.inserting_before(node):
            qparam_node = create_getattr_from_value(
                model, graph, next(iter(node.users)).name + "_scale", scale
            )
            # TODO quantization map can be shared among multiple quantize nodes?
            get_attr_node = create_getattr_from_value(
                model,
                graph,
                "qmap",
                activation_post_process.qmap,
                compiler_only=True,
            )
            quantized_node = graph.call_function(
                torch.ops.quantized_ops.quantize.default,
                (
                    node.args[0],
                    qparam_node,
                    None,
                    None,
                    None,
                    get_attr_node,
                    None,
                    activation_post_process.rounding,
                    activation_post_process.quant_min,
                    activation_post_process.quant_max,
                ),
            )

        # Annotate input dtype
        quantized_node.meta["dtype"] = activation_post_process.dtype
        for key in (
            QUANTIZATION_REGION,
            _QUANTIZATION_DOMAIN_ENTRY,
            QUANTIZATION_DOMAIN_DTYPE,
            PHYSICAL_QUANTIZATION_CONTRACT,
        ):
            if key in node.meta:
                quantized_node.meta[key] = node.meta[key]

        node.replace_all_uses_with(quantized_node)
    graph.erase_node(node)

    # We don't need to insert dequantize node for bias
    if (
        isinstance(activation_post_process, _DerivedObserverOrFakeQuantize)
        or activation_post_process.qscheme is None
    ):
        return

    _materialize_domain_users(
        model,
        node,
        input_node if input_node.op == "get_attr" else quantized_node,
        orig_fq_users,
        scale,
        activation_post_process.dtype,
        output_dtype,
        device,
    )


MX_OP_MAPPING = {
    torch.ops.aten.conv2d.default: torch.ops.quantized_ops.conv2d_mx.default,
    torch.ops.aten.linear.default: torch.ops.quantized_ops.linear_mx.default,
    torch.ops.aten.matmul.default: torch.ops.quantized_ops.matmul_mx.default,
}


def _record_source_op_name(node: Node, parameter: Node):
    """Record the logical compute name while its parameter is explicit."""
    if parameter.op != "get_attr":
        return
    node.meta[SOURCE_OP_NAME] = parameter.name.removesuffix("_weight")


def _trace_converted_vsq_join(source: Node):
    path = []
    current = source
    while current.target in _PREPARED_DOMAIN_PASSTHROUGH_TARGETS:
        inputs = _prepared_domain_data_inputs(current)
        if len(inputs) != 1:
            return None
        path.append(current)
        current = inputs[0]
    if current.target not in _PREPARED_DOMAIN_JOIN_TARGETS:
        return None
    return current, tuple(reversed(path))


def _converted_vsq_components(node: Node):
    if node.target != torch.ops.quantized_ops.dequantize_vsq.default:
        return None
    return (
        get_arg_value(node, 0, "input"),
        get_arg_value(node, 1, "fine_scale"),
        get_arg_value(node, 2, "coarse_scale"),
        get_arg_value(node, 3, "axes"),
        get_arg_value(node, 4, "block_size"),
        get_arg_value(node, 5, "code"),
    )


def _clone_vsq_join_component(
    graph: Graph,
    join: Node,
    path: tuple[Node, ...],
    inputs: tuple[Node, ...],
) -> tuple[Node, tuple[Node, ...]]:
    args = list(join.args)
    args[0] = inputs
    with graph.inserting_before(join):
        output = graph.call_function(
            join.target,
            tuple(args),
            dict(join.kwargs),
        )
    output.meta = copy.copy(join.meta)
    created = [output]

    previous = join
    for path_node in path:
        args = list(path_node.args)
        if not args or args[0] is not previous:
            raise RuntimeError(
                f"{path_node.name}: malformed VSQ join transfer path"
            )
        args[0] = output
        with graph.inserting_before(path_node):
            next_output = graph.call_function(
                path_node.target,
                tuple(args),
                dict(path_node.kwargs),
            )
        next_output.meta = copy.copy(path_node.meta)
        output = next_output
        created.append(output)
        previous = path_node
    return output, tuple(created)


def _replace_vsq_join_alias(
    model: GraphModule,
    node: Node,
    modules: Dict[str, torch.nn.Module],
) -> None:
    """Convert a calibrated VSQ fan-in into explicit payload/scale joins."""
    graph = model.graph
    activation_post_process = modules[str(node.target)]
    source = node.args[0]
    match = _trace_converted_vsq_join(source)
    if match is None:
        raise RuntimeError(f"{node.name}: cannot recover its VSQ join")
    join, path = match

    dequantizers = _prepared_domain_data_inputs(join)
    components = tuple(
        _converted_vsq_components(input_node) for input_node in dequantizers
    )
    if not components or any(component is None for component in components):
        raise RuntimeError(
            f"{node.name}: VSQ join inputs are not encoded boundaries"
        )

    payloads = tuple(component[0] for component in components)
    fine_scales = tuple(component[1] for component in components)
    coarse_scales = tuple(component[2] for component in components)
    axes = tuple(component[3] for component in components)
    block_sizes = tuple(component[4] for component in components)
    codes = tuple(component[5] for component in components)
    if len(set(map(str, axes))) != 1 or len(set(block_sizes)) != 1:
        raise RuntimeError(f"{node.name}: incompatible VSQ join geometry")
    if len(set(map(str, codes))) != 1:
        raise RuntimeError(f"{node.name}: incompatible VSQ join codebooks")

    payload, payload_nodes = _clone_vsq_join_component(
        graph,
        join,
        path,
        payloads,
    )
    fine_scale, fine_scale_nodes = _clone_vsq_join_component(
        graph,
        join,
        path,
        fine_scales,
    )
    scale_dtype = (
        "fp8_e8m0"
        if activation_post_process.force_scale_power_of_two
        else activation_post_process.scale_dtype
    )
    payload.meta["dtype"] = activation_post_process.dtype
    for fine_scale_node in fine_scale_nodes:
        fine_scale_node.meta["dtype"] = scale_dtype
        fine_scale_node.meta.pop(QUANTIZATION_DOMAIN_DTYPE, None)
        fine_scale_node.meta.pop(PHYSICAL_QUANTIZATION_CONTRACT, None)
    payload.meta[QUANTIZATION_REGION] = node.meta[QUANTIZATION_REGION]
    domain = _vsq_physical_quantization_contract(activation_post_process)
    for payload_node in payload_nodes:
        set_physical_quantization_contract(payload_node, domain)

    original_users = list(node.users)
    node.replace_all_uses_with(payload)
    graph.erase_node(node)

    for user in original_users:
        if (
            user.target not in MX_OP_MAPPING
            and user.target not in MX_OP_MAPPING.values()
        ):
            raise RuntimeError(
                f"{payload.name}: unsupported VSQ join user {user.target}"
            )
        kwargs = OrderedDict(user.kwargs)
        kwargs["input_scale"] = fine_scale
        kwargs["input_coarse_scale"] = coarse_scales[0]
        kwargs["block_size"] = block_sizes[0]
        if codes[0] is not None:
            kwargs["input_code"] = codes[0]
        order = (
            "input_scale",
            "input_coarse_scale",
            "weight_scale",
            "weight_coarse_scale",
            "block_size",
            "input_code",
            "weight_code",
        )
        kwargs = OrderedDict(
            (key, kwargs[key]) for key in order if key in kwargs
        )
        if user.target in MX_OP_MAPPING:
            with graph.inserting_before(user):
                mx_op_node = graph.call_function(
                    MX_OP_MAPPING[user.target],
                    user.args,
                    kwargs,
                )
            mx_op_node.meta = user.meta
            user.replace_all_uses_with(mx_op_node)
            graph.erase_node(user)
        else:
            user.kwargs = kwargs

    for dead in (*reversed(path), join, *dequantizers):
        if not dead.users:
            graph.erase_node(dead)


def _replace_observer_with_quantize_vsq_node_decomposed(
    model: torch.fx.GraphModule,
    node: Node,
    modules: Dict[str, torch.nn.Module],
):
    """Replace one VSQ fake quantizer with an atomic VSQ graph node."""
    graph = model.graph
    assert isinstance(node.target, str)
    activation_post_process = modules[node.target]
    device = assert_and_get_unique_device(activation_post_process)
    input_node = node.args[0]

    axes = activation_post_process.ch_axis
    if isinstance(axes, int):
        axes = (axes,)

    fine_dtype = activation_post_process.dtype
    fine_qmap = get_quantization_map(
        fine_dtype,
        device,
        activation_post_process.quant_min,
        activation_post_process.quant_max,
        activation_post_process.rounding,
    )
    dequant_code, quant_code = None, None
    storage_dtype = fine_dtype

    if isinstance(fine_qmap, tuple):
        matches = re.findall(r"\d+", fine_dtype)
        storage_dtype = f"int{matches[0]}"
        indices, values = fine_qmap
        fine_qmap = indices

        with graph.inserting_before(node):
            dequant_code = create_getattr_from_value(
                model,
                graph,
                "code",
                values,
                compiler_only=True,
            )
            if input_node.op != "get_attr":
                midpoints = (values[:-1] + values[1:]) / 2
                quant_code = create_getattr_from_value(
                    model,
                    graph,
                    "code",
                    midpoints,
                    compiler_only=True,
                )

        if len(matches) > 1:
            dequant_code.meta["dtype"] = f"int{matches[1]}"

    input_value = input_node.meta.get("val")
    model_scale_dtype = (
        input_value.dtype
        if isinstance(input_value, torch.Tensor)
        else torch.float32
    )
    coarse_scale = activation_post_process.calculate_qparams().to(
        model_scale_dtype
    )

    if input_node.op == "get_attr":
        try:
            parameter = model.get_parameter(input_node.target)
        except AttributeError:
            parameter = model.get_buffer(input_node.target)

        if activation_post_process.static_vsq_payload.numel() > 0:
            quantized_value = activation_post_process.quantize_vsq_components(
                parameter.data
            )
        else:
            quantized_value = torch.ops.quantized_ops.quantize_vsq.default(
                parameter.data,
                coarse_scale,
                activation_post_process.coarse_qmap,
                fine_qmap,
                axes,
                activation_post_process.block_size,
                activation_post_process.scale_quant_max,
                activation_post_process.force_scale_power_of_two,
                activation_post_process.scale_qmap,
                None,
                activation_post_process.coarse_scale_quant_max,
            )
        with graph.inserting_before(node):
            coarse_scale_output = create_getattr_from_value(
                model,
                graph,
                input_node.name + "_coarse_scale",
                quantized_value[0],
            )
            fine_scale_output = create_getattr_from_value(
                model,
                graph,
                input_node.name + "_scale",
                quantized_value[1],
            )
            quantized_node = create_getattr_from_value(
                model,
                graph,
                input_node.name + "_" + storage_dtype,
                quantized_value[2],
            )
        vsq_node = None
    else:
        with graph.inserting_before(node):
            coarse_scale_node = create_getattr_from_value(
                model,
                graph,
                input_node.name + "_coarse_scale",
                coarse_scale,
            )
            coarse_qmap_node = create_getattr_from_value(
                model,
                graph,
                "coarse_qmap",
                activation_post_process.coarse_qmap,
                compiler_only=True,
            )
            fine_qmap_node = create_getattr_from_value(
                model,
                graph,
                "fine_qmap",
                fine_qmap,
                compiler_only=True,
            )
            fine_scale_qmap_node = None
            if activation_post_process.scale_qmap is not None:
                fine_scale_qmap_node = create_getattr_from_value(
                    model,
                    graph,
                    "fine_scale_qmap",
                    activation_post_process.scale_qmap,
                    compiler_only=True,
                )

            vsq_node = graph.call_function(
                torch.ops.quantized_ops.quantize_vsq.default,
                (
                    input_node,
                    coarse_scale_node,
                    coarse_qmap_node,
                    fine_qmap_node,
                    axes,
                    activation_post_process.block_size,
                    activation_post_process.scale_quant_max,
                    activation_post_process.force_scale_power_of_two,
                    fine_scale_qmap_node,
                    quant_code,
                    activation_post_process.coarse_scale_quant_max,
                ),
            )
            coarse_scale_output = graph.call_function(
                operator.getitem,
                (vsq_node, 0),
            )
            fine_scale_output = graph.call_function(
                operator.getitem,
                (vsq_node, 1),
            )
            quantized_node = graph.call_function(
                operator.getitem,
                (vsq_node, 2),
            )

    scale_dtype = (
        "fp8_e8m0"
        if activation_post_process.force_scale_power_of_two
        else activation_post_process.scale_dtype
    )
    if vsq_node is not None:
        vsq_node.meta["dtype"] = (None, scale_dtype, storage_dtype)
    if scale_dtype is not None:
        fine_scale_output.meta["dtype"] = scale_dtype
    quantized_node.meta["dtype"] = storage_dtype
    physical_contract = PhysicalQuantizationContract(
        storage_dtype,
        (() if scale_dtype is None else (scale_dtype,)),
        activation_post_process.block_size,
    )
    if vsq_node is not None:
        set_physical_quantization_contract(vsq_node, physical_contract)
        for key in (
            QUANTIZATION_REGION,
            _VSQ_JOIN_ENTRY,
            _VSQ_JOIN_ALIAS,
        ):
            if key in node.meta:
                vsq_node.meta[key] = node.meta[key]
        set_physical_quantization_contract(
            quantized_node,
            physical_contract,
        )

    original_users = list(node.users.keys())

    def is_gemm_operand_user(user):
        if user.target == torch.Tensor.to and len(user.users) == 1:
            user = next(iter(user.users))
        return (
            user.target in MX_OP_MAPPING
            or user.target in MX_OP_MAPPING.values()
        )

    is_gemm_operand = input_node.op == "get_attr" or (
        bool(original_users)
        and all(is_gemm_operand_user(user) for user in original_users)
    )

    if not is_gemm_operand:
        with graph.inserting_before(node):
            dequantized_node = graph.call_function(
                torch.ops.quantized_ops.dequantize_vsq.default,
                (
                    quantized_node,
                    fine_scale_output,
                    coarse_scale_output,
                    axes,
                    activation_post_process.block_size,
                    dequant_code,
                ),
            )

        dequantized_node.meta = copy.copy(node.meta)
        node.replace_all_uses_with(dequantized_node)
        graph.erase_node(node)
        return

    node.replace_all_uses_with(quantized_node)
    graph.erase_node(node)

    if len(input_node.users) == 0:
        graph.erase_node(input_node)

    for user in original_users:
        coarse_for_user = coarse_scale_output
        fine_for_user = fine_scale_output
        code_for_user = dequant_code

        if user.target == torch.Tensor.to:
            user_device = user.args[1]
            with graph.inserting_before(user):
                coarse_for_user = graph.call_function(
                    torch.Tensor.to, (coarse_for_user, user_device)
                )
                fine_for_user = graph.call_function(
                    torch.Tensor.to, (fine_for_user, user_device)
                )
                if code_for_user is not None:
                    code_for_user = graph.call_function(
                        torch.Tensor.to, (code_for_user, user_device)
                    )
                    code_for_user.meta[COMPILER_ONLY] = True
            user = next(iter(user.users))

        kwargs = OrderedDict(user.kwargs)
        kwargs.setdefault("block_size", activation_post_process.block_size)
        is_weight = input_node.op == "get_attr" or id(quantized_node) == id(
            user.args[1]
        )
        if is_weight:
            kwargs.setdefault("weight_code", code_for_user)
            kwargs.setdefault("weight_scale", fine_for_user)
            kwargs.setdefault("weight_coarse_scale", coarse_for_user)
        else:
            kwargs.setdefault("input_code", code_for_user)
            kwargs.setdefault("input_scale", fine_for_user)
            kwargs.setdefault("input_coarse_scale", coarse_for_user)

        order = [
            "input_scale",
            "input_coarse_scale",
            "weight_scale",
            "weight_coarse_scale",
            "block_size",
            "input_code",
            "weight_code",
        ]
        kwargs = OrderedDict(
            (key, kwargs[key]) for key in order if key in kwargs
        )

        if user.target in MX_OP_MAPPING:
            with graph.inserting_before(user):
                mx_op_node = graph.call_function(
                    MX_OP_MAPPING[user.target], user.args, kwargs
                )
            user.replace_all_uses_with(mx_op_node)
            graph.erase_node(user)
            mx_op_node.meta = user.meta
        elif user.target in MX_OP_MAPPING.values():
            mx_op_node = user
            mx_op_node.kwargs = kwargs
        else:
            raise RuntimeError(
                f"Unsupported VSQ user {user.target}; expected a GEMM."
            )

        if is_weight:
            _record_source_op_name(mx_op_node, input_node)


def _replace_observer_with_quantize_mx_node_decomposed(
    model: torch.fx.GraphModule,
    node: Node,
    modules: Dict[str, torch.nn.Module],
):
    graph = model.graph
    assert modules is not None
    assert isinstance(node.target, str)
    activation_post_process = modules[node.target]
    device = assert_and_get_unique_device(activation_post_process)

    input_node = node.args[0]
    input_dtype = activation_post_process.dtype
    node_to_quantize = input_node

    if isinstance(activation_post_process.ch_axis, int):
        activation_post_process.ch_axis = (activation_post_process.ch_axis,)

    if activation_post_process.outlier_threshold is not None:
        outlier_max_pct = (
            math.ceil(activation_post_process.outlier_max_pct * 100) / 100.0
        )
        activation_post_process.outlier_max_pct = max(outlier_max_pct, 0.05)
        logger.info(
            f"{node.target} has maximum outlier percentage "
            f"{outlier_max_pct:.2%}"
        )

    # Can only fuse scale calculation with quantization if along the last dim
    fuse_with_quantize = False
    if len(activation_post_process.ch_axis) == 1:
        axis = activation_post_process.ch_axis[0]
        if any(is_conv2d(user) for user in node.users):
            fuse_with_quantize = axis in (1, -3)
        else:
            fuse_with_quantize = axis == -1 or (
                "val" in input_node.meta
                and axis == input_node.meta["val"].ndim - 1
            )

    quant_map = get_quantization_map(
        activation_post_process.dtype,
        device,
        activation_post_process.quant_min,
        activation_post_process.quant_max,
        activation_post_process.rounding,
    )
    dequant_code, quant_code = None, None

    if isinstance(quant_map, tuple):
        matches = re.findall(r"\d+", activation_post_process.dtype)
        activation_post_process.dtype = f"int{matches[0]}"

        indices, values = quant_map
        activation_post_process.qmap = indices

        with graph.inserting_before(node):
            dequant_code = create_getattr_from_value(
                model,
                graph,
                "code",
                values,
                compiler_only=True,
            )
            if input_node.op != "get_attr":
                midpoints = (values[:-1] + values[1:]) / 2
                quant_code = create_getattr_from_value(
                    model,
                    graph,
                    "code",
                    midpoints,
                    compiler_only=True,
                )

        # NF4_[B] means approximate NormalFloat4 with B-bit integer
        if len(matches) > 1:
            dequant_code.meta["dtype"] = f"int{matches[1]}"

    if input_node.op == "get_attr":
        # quantize model parameter and remove the fq module
        try:
            param = model.get_parameter(input_node.target)
        except AttributeError:
            param = model.get_buffer(input_node.target)

        scale = torch.ops.quantized_ops.calculate_mx_qparam(
            param.data,
            activation_post_process.ch_axis,
            activation_post_process.block_size,
            activation_post_process.scale_quant_max,
            activation_post_process.force_scale_power_of_two,
            activation_post_process.scale_qmap,
        )

        weight = torch.ops.quantized_ops.quantize(
            param.data,
            scale,
            axes=activation_post_process.ch_axis,
            block_size=activation_post_process.block_size,
            qmap=activation_post_process.qmap,
        )

        with graph.inserting_before(node):
            quantized_node = create_getattr_from_value(
                model, graph, input_node.name + "_" + input_dtype, weight
            )
            scale_node = create_getattr_from_value(
                model, graph, input_node.name + "_scale", scale
            )
    elif fuse_with_quantize:
        with model.graph.inserting_before(node):
            get_attr_node = create_getattr_from_value(
                model,
                model.graph,
                "qmap",
                activation_post_process.qmap,
                compiler_only=True,
            )

            scale_qmap = None
            if activation_post_process.scale_qmap is not None:
                scale_qmap = create_getattr_from_value(
                    model,
                    model.graph,
                    "qmap",
                    activation_post_process.scale_qmap,
                    compiler_only=True,
                )

            target = torch.ops.quantized_ops.quantize_mx.default
            args = [
                node_to_quantize,
                get_attr_node,
                activation_post_process.ch_axis,
                activation_post_process.block_size,
                activation_post_process.scale_quant_max,
                activation_post_process.force_scale_power_of_two,
                scale_qmap,
                quant_code,
            ]

            if activation_post_process.outlier_threshold is not None:
                target = torch.ops.quantized_ops.quantize_mx_outlier.default
                args.extend(
                    [
                        activation_post_process.outlier_threshold,
                        activation_post_process.outlier_max_pct,
                    ]
                )
                num_outputs = 5
            else:
                num_outputs = 2

            quantize_mx_node = model.graph.call_function(target, tuple(args))

            output_nodes = [
                graph.call_function(operator.getitem, (quantize_mx_node, i))
                for i in range(num_outputs)
            ]

        scale_dtype = (
            "fp8_e8m0"
            if activation_post_process.force_scale_power_of_two
            else activation_post_process.scale_dtype
        )

        if num_outputs == 5:
            csr_data_node = output_nodes[0]
            csr_indices_node = output_nodes[1]
            csr_indptr_node = output_nodes[2]
            scale_node = output_nodes[3]
            quantized_node = output_nodes[4]
            dtype_tuple = (
                None,
                None,
                None,
                scale_dtype,
                activation_post_process.dtype,
            )
        else:
            scale_node, quantized_node = output_nodes
            dtype_tuple = (scale_dtype, activation_post_process.dtype)

        quantize_mx_node.meta["dtype"] = dtype_tuple
    else:
        with model.graph.inserting_before(node):
            if activation_post_process.outlier_threshold is not None:
                filter_outlier_node = graph.call_function(
                    torch.ops.quantized_ops.filter_outlier.default,
                    (
                        input_node,
                        activation_post_process.outlier_threshold,
                        activation_post_process.outlier_max_pct,
                    ),
                )
                output_nodes = [
                    graph.call_function(
                        operator.getitem,
                        (filter_outlier_node, i),
                    )
                    for i in range(4)
                ]
                (
                    csr_data_node,
                    csr_indices_node,
                    csr_indptr_node,
                    input_node,
                ) = output_nodes

            calculate_qparam_op_inputs = [
                input_node,
                activation_post_process.ch_axis,
                activation_post_process.block_size,
                activation_post_process.scale_quant_max,
                activation_post_process.force_scale_power_of_two,
            ]

            if activation_post_process.scale_qmap is not None:
                get_attr_node = create_getattr_from_value(
                    model,
                    model.graph,
                    "qmap",
                    activation_post_process.scale_qmap,
                    compiler_only=True,
                )
                calculate_qparam_op_inputs.append(get_attr_node)

            scale_node = model.graph.call_function(
                torch.ops.quantized_ops.calculate_mx_qparam.default,
                tuple(calculate_qparam_op_inputs),
                {},
            )

            if activation_post_process.scale_dtype is not None:
                scale_node.meta["dtype"] = activation_post_process.scale_dtype

            get_attr_node = create_getattr_from_value(
                model,
                model.graph,
                "qmap",
                activation_post_process.qmap,
                compiler_only=True,
            )
            quantized_node = model.graph.call_function(
                torch.ops.quantized_ops.quantize.default,
                (
                    input_node,
                    scale_node,
                    None,
                    activation_post_process.ch_axis,
                    activation_post_process.block_size,
                    get_attr_node,
                    quant_code,
                ),
            )

        if activation_post_process.outlier_threshold is not None:
            filter_outlier_node.meta["dtype"] = (None, None, "int16", "int16")

    quantized_node.meta["dtype"] = activation_post_process.dtype

    if activation_post_process.force_scale_power_of_two:
        scale_node.meta["dtype"] = "fp8_e8m0"
    elif activation_post_process.scale_dtype is not None:
        scale_node.meta["dtype"] = activation_post_process.scale_dtype

    orig_fq_users = list(node.users.keys())

    node.replace_all_uses_with(quantized_node)
    graph.erase_node(node)

    if len(input_node.users) == 0:
        graph.erase_node(input_node)

    for user in orig_fq_users:
        # Keep the original nodes for other users
        kwarg1, kwarg2 = dequant_code, scale_node

        # Skip device alignment node
        if user.target == torch.Tensor.to:
            user_device = user.args[1]
            with graph.inserting_before(user):
                if kwarg1 is not None:
                    kwarg1 = graph.call_function(
                        torch.Tensor.to, (dequant_code, user_device)
                    )
                    kwarg1.meta[COMPILER_ONLY] = True
                kwarg2 = graph.call_function(
                    torch.Tensor.to, (scale_node, user_device)
                )
            user = next(iter(user.users))

        kwargs = OrderedDict(user.kwargs)
        kwargs.setdefault("block_size", activation_post_process.block_size)
        is_weight = input_node.op == "get_attr" or id(quantized_node) == id(
            user.args[1]
        )
        if is_weight:
            kwargs.setdefault("weight_code", kwarg1)
            kwargs.setdefault("weight_scale", kwarg2)
        else:
            kwargs.setdefault("input_code", kwarg1)
            kwargs.setdefault("input_scale", kwarg2)

        # Keep kwargs in the sequential order used during MHA splitting.
        order = [
            "input_scale",
            "weight_scale",
            "block_size",
            "input_code",
            "weight_code",
            "A_data",
            "A_indices",
            "A_indptr",
        ]
        kwargs = OrderedDict(
            (key, kwargs[key]) for key in order if key in kwargs
        )

        # Replace the node with its MX counterpart
        mx_op_node = None
        if user.target in MX_OP_MAPPING:
            with graph.inserting_before(user):
                mx_op_node = graph.call_function(
                    MX_OP_MAPPING[user.target], user.args, kwargs
                )

            user.replace_all_uses_with(mx_op_node)
            graph.erase_node(user)

            mx_op_node.meta = user.meta
        elif user.target in MX_OP_MAPPING.values():
            mx_op_node = user
            mx_op_node.kwargs = kwargs
        elif user.target == torch.ops.quantized_ops.spmm_csr.default:
            assert (
                input_node.op == "get_attr"
            ), f"Expect input node to be a get_attr, but found {input_node.op}"
            user.args = user.args[:-1] + (quantized_node,)
            user.kwargs = {
                "B_scale": kwargs.get("weight_scale"),
                "B_code": kwargs.get("weight_code"),
                "block_size": activation_post_process.block_size,
            }
        else:
            raise RuntimeError(
                f"Unsupported user node {user.target} for quantization, "
                f"expected one of {list(MX_OP_MAPPING.keys())}"
            )

        if mx_op_node is not None and is_weight:
            _record_source_op_name(mx_op_node, input_node)

        if (
            activation_post_process.outlier_threshold is not None
            and input_node.op != "get_attr"
        ):
            # For now only support linear layers
            assert mx_op_node.target in [
                torch.ops.aten.linear.default,
                torch.ops.aten.matmul.default,
                torch.ops.quantized_ops.linear_mx.default,
                torch.ops.quantized_ops.matmul_mx.default,
            ], (
                "Only GEMM supports outlier suppression; " f"got {user.target}"
            )

            weight_node = mx_op_node.args[1]

            if mx_op_node.target in [
                torch.ops.quantized_ops.linear_mx.default,
                torch.ops.quantized_ops.matmul_mx.default,
            ]:
                mx_op_node.kwargs = {
                    **mx_op_node.kwargs,
                    "A_data": csr_data_node,
                    "A_indices": csr_indices_node,
                    "A_indptr": csr_indptr_node,
                }
            else:
                with graph.inserting_before(mx_op_node):
                    spmm_node = graph.call_function(
                        torch.ops.quantized_ops.spmm_csr.default,
                        (
                            csr_data_node,
                            csr_indices_node,
                            csr_indptr_node,
                            weight_node,
                        ),
                        {
                            "B_scale": kwargs.get("weight_scale"),
                            "B_code": kwargs.get("weight_code"),
                            "block_size": activation_post_process.block_size,
                        },
                    )

                with graph.inserting_after(mx_op_node):
                    add_node = graph.call_function(
                        torch.ops.aten.add.Tensor,
                        (spmm_node, mx_op_node),
                        {},
                    )

                mx_op_node.replace_all_uses_with(add_node)
                add_node.replace_input_with(add_node, mx_op_node)


def _replace_group_wise_affine_observer(
    model: torch.fx.GraphModule,
    node: Node,
    modules: Dict[str, torch.nn.Module],
):
    graph = model.graph
    assert modules is not None
    assert isinstance(node.target, str)
    activation_post_process = modules[node.target]
    device = assert_and_get_unique_device(activation_post_process)

    if isinstance(activation_post_process.ch_axis, int):
        activation_post_process.ch_axis = (activation_post_process.ch_axis,)

    input_node = node.args[0]

    if input_node.op == "get_attr":
        try:
            param = model.get_parameter(input_node.target)
        except AttributeError:
            param = model.get_buffer(input_node.target)

        activation_post_process(param.data)
        scale, zero_point = activation_post_process.calculate_qparams()
        scale = scale.to(param.data.dtype)
        zero_point = zero_point.to(param.data.dtype)

        weight = torch.ops.quantized_ops.quantize(
            param.data,
            scale,
            zero_point,
            activation_post_process.ch_axis,
            activation_post_process.block_size,
            activation_post_process.qmap,
        )

        with graph.inserting_before(node):
            quantized_node = create_getattr_from_value(
                model,
                graph,
                input_node.name + "_" + activation_post_process.dtype,
                weight,
            )
            scale_node = create_getattr_from_value(
                model, graph, input_node.name + "_scale", scale
            )
            zero_point_node = create_getattr_from_value(
                model, graph, input_node.name + "_zero_point", zero_point
            )
    else:
        raise NotImplementedError

    quantized_node.meta["dtype"] = activation_post_process.dtype

    if activation_post_process.scale_dtype is not None:
        scale_node.meta["dtype"] = activation_post_process.scale_dtype
        zero_point_node.meta["dtype"] = activation_post_process.scale_dtype

    # Insert a dequantize node after the quantize node
    with graph.inserting_before(node):
        dequantized_node = graph.call_function(
            torch.ops.quantized_ops.dequantize.default,
            (
                quantized_node,
                scale_node,
                zero_point_node,
                activation_post_process.ch_axis,
                activation_post_process.block_size,
            ),
        )

    node.replace_all_uses_with(dequantized_node)
    graph.erase_node(node)

    if len(input_node.users) == 0:
        graph.erase_node(input_node)


def _eliminate_dequantize_with_no_effect(model: GraphModule):
    for node in model.graph.nodes:
        if node.target != torch.ops.quantized_ops.dequantize.default:
            continue

        scale_node = node.args[1]
        if scale_node.op != "get_attr":
            continue
        scale = model.get_buffer(scale_node.target)
        if torch.any(scale != 1):
            continue

        # During integer quantization, the dequantize node also perform a
        # quantization to the output dtype
        output_qmap = get_arg_value(node, 6, "output_qmap")
        if output_qmap is not None:
            continue

        node.replace_all_uses_with(node.args[0])
        model.graph.erase_node(node)
        logger.info(f"Eliminate dequantize node {node} with no effect")

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()

    return model


def fuse_quantize_dequantize_with_previous_op(model: GraphModule):
    """
    Move quantize and dequantize nodes up the graph and place them after the
    previous operation (e.g. matmul, conv2d, etc.), so that the quantize and
    dequantize can be fused with the previous operation.
    """
    graph = model.graph

    def find_prev_op_and_move_node(node, prev_node=None, curr_node=None):
        nodes_on_path = []
        prev_node = node.args[0] if prev_node is None else prev_node
        while len(prev_node.users) == 1:
            # If there are multiple input nodes, trace each path separately
            if prev_node.target in [
                torch.ops.aten.stack.default,
                torch.ops.aten.cat.default,
            ]:
                nodes_on_path.append(prev_node)
                for arg in prev_node.all_input_nodes:
                    nodes_on_path.extend(
                        find_prev_op_and_move_node(node, arg, prev_node)
                    )
                return nodes_on_path

            # stack and cat are handled above, so we can safely assume that
            # they won't appear here and there is only one input node
            if (
                not is_nop(prev_node)
                and not is_reshape_op(prev_node)
                and not is_indexing_or_concatenation_op(prev_node)
                and prev_node.target
                not in [
                    torch.ops.aten.expand.default,
                    torch.ops.aten.repeat.default,
                ]
            ):
                break

            assert len(prev_node.all_input_nodes) == 1

            nodes_on_path.append(prev_node)
            curr_node = prev_node
            prev_node = prev_node.args[0]

        # A recursive path may be empty. Check whether traversal stayed at
        # the quantize or dequantize input.
        if id(prev_node) == id(node.args[0]):
            return None

        value_remap = {node.args[0]: prev_node}
        with graph.inserting_before(prev_node.next):
            for n in node.all_input_nodes:
                if n not in value_remap:
                    value_remap[n] = graph.node_copy(n)
            new_node = graph.node_copy(node, lambda n: value_remap[n])

        assert (
            curr_node in prev_node.users
        ), "The last node on the path should be a user of the previous node"
        curr_node.replace_input_with(prev_node, new_node)

        for n in list(value_remap.values()) + [new_node]:
            propagate_shape(n, model)

        # Copy meta data from the original node, e.g. dtype
        new_node.meta = {
            k: copy.deepcopy(v) if k != "val" else v.clone()
            for k, v in node.meta.items()
        }

        return nodes_on_path

    # First, move the dequantize nodes before the stack and view nodes that
    # are inserted during MHA splitting. Then try to move quantize nodes forward
    # to immediately after their previous ops (nodes that are not NOP).
    for node in list(model.graph.nodes):
        if node.target not in [
            torch.ops.quantized_ops.dequantize.default,
            torch.ops.quantized_ops.quantize.default,
        ]:
            continue

        # Only handle per-tensor quantization for now
        block_size = get_arg_value(node, 4, "block_size")
        if block_size is not None and block_size > 1:
            continue

        output_node = node
        nodes_on_path = find_prev_op_and_move_node(output_node)
        if nodes_on_path is None:
            continue

        # Update dtype annotation for all nodes on the path.
        is_dequantize = (
            output_node.target == torch.ops.quantized_ops.dequantize.default
        )
        for n in nodes_on_path:
            if is_dequantize:
                n.meta.pop("dtype", None)
            else:
                n.meta["dtype"] = output_node.meta.get("dtype", None)

        # Remove the nodes from the graph. This has to be done at the end
        # because we may need to copy and insert the nodes multiple times
        # if there is stack or cat node in the path.
        output_node.replace_all_uses_with(output_node.args[0])
        graph.erase_node(output_node)

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()

    return model


def swap_matmul_inputs(model: GraphModule):
    graph = model.graph
    modules = dict(model.named_modules(remove_duplicate=False))

    def get_fake_quant_mod(node: Node):
        if node.op == "call_module":
            mod = _get_module(node, modules)
            if isinstance(mod, torch.ao.quantization.FakeQuantizeBase):
                return mod

        return None

    target = torch.ops.aten.transpose.int

    def transpose_input(node):
        input_node = node.args[0]
        if input_node.target != target:
            with graph.inserting_before(node):
                transposed_node = graph.call_function(
                    target,
                    (input_node, -1, -2),
                )
            node.replace_input_with(input_node, transposed_node)
        else:
            node.replace_input_with(input_node, input_node.args[0])

    for node in list(graph.nodes):
        if not is_matmul(node):
            continue

        input_node = node.args[0]
        other_node = node.args[1]

        input_fq = get_fake_quant_mod(input_node)
        other_fq = get_fake_quant_mod(other_node)

        if other_fq is None or other_fq.outlier_threshold is None:
            continue

        assert (
            input_fq is None or input_fq.outlier_threshold is None
        ), "Only one input of matmul can have outlier filter"

        node.args = (other_node, input_node)

        transpose_input(input_node)
        transpose_input(other_node)

        other_fq.ch_axis = -1
        input_fq.ch_axis = -2

        with graph.inserting_after(node):
            transposed = graph.call_function(target, (node, -1, -2))

        for user in list(node.users):
            if id(user) != id(transposed):
                user.replace_input_with(node, transposed)

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()


def convert_pt2e(
    model: GraphModule,
    output_dtype: str = None,
    eliminate_no_effect: bool = True,
    calibration_steps: int | None = None,
    hardware_config=None,
):
    """Convert prepared fake quantizers into executable reference operators.

    A zero-step calibration run leaves observer scales at their identity
    initialization. Those scale-one values do not prove that dequantization is
    semantically redundant; they only mean calibration was skipped. Preserve
    the nodes so target lowering in ``transform`` can still see the intended
    coarse quantization domains.
    """
    if calibration_steps is not None and calibration_steps < 0:
        raise ValueError("calibration_steps must be non-negative")

    target = resolve_compilation_target(hardware_config)
    validate_target_hardware_config(target, hardware_config)
    bind_compilation_target(
        model,
        hardware_config,
        stage="convert_pt2e",
    )
    finalize_pt2e_calibration(model, hardware_config)
    cleanup_target_quantization_preparation(model, target)
    modules = dict(model.named_modules(remove_duplicate=False))

    swap_matmul_inputs(model)
    _materialize_unobserved_per_channel_scales(model, modules)

    for node in list(model.graph.nodes):
        if node.op == "call_module":
            mod = _get_module(node, modules)
            assert mod is not None
            if isinstance(mod, torch.ao.quantization.FakeQuantizeBase):
                if node.meta.get(_QUANTIZATION_DOMAIN_ALIAS):
                    _replace_domain_alias(
                        model,
                        node,
                        modules,
                        output_dtype,
                    )
                elif node.meta.get(_VSQ_JOIN_ALIAS):
                    _replace_vsq_join_alias(model, node, modules)
                elif mod.qscheme == qt.vsq:
                    _replace_observer_with_quantize_vsq_node_decomposed(
                        model, node, modules
                    )
                elif mod.qscheme == qt.microscaling:
                    _replace_observer_with_quantize_mx_node_decomposed(
                        model,
                        node,
                        modules,
                    )
                elif mod.qscheme == qt.group_wise_affine:
                    _replace_group_wise_affine_observer(
                        model,
                        node,
                        modules,
                    )
                else:
                    _replace_observer_with_quantize_dequantize_node_decomposed(
                        model, node, modules, output_dtype
                    )

    for node in model.graph.nodes:
        domain_dtype = node.meta.get(QUANTIZATION_DOMAIN_DTYPE)
        if domain_dtype is not None:
            value = getattr(node, "value", None)
            if value is None:
                value = node.meta.get("val")
            node.meta["dtype"] = (
                tuple(domain_dtype for _ in value)
                if isinstance(value, (tuple, list))
                else domain_dtype
            )

    if eliminate_no_effect and calibration_steps != 0:
        _eliminate_dequantize_with_no_effect(model)

    model.graph.lint()
    model.graph.eliminate_dead_code(
        is_impure_node=lambda n: n.op in {"placeholder", "output"}
    )
    model.recompile()
    model.delete_all_unused_submodules()

    return model

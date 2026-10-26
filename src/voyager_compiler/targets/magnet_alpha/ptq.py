# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Alpha PTQ conversion over the shared TorchAO preparation frontend.

This owns exact fine/coarse numerical domains; search, graph normalization,
bufferization, allocation and emission remain in the common compiler.
"""

import copy
import logging
import operator
import re
from collections import OrderedDict
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.ao.quantization.fx.utils import assert_and_get_unique_device
from torch.fx import Graph, GraphModule, Node
from torchao.quantization.pt2e import FakeQuantizeBase, ObserverOrFakeQuantize

from voyager_compiler.ir.graph import (
    PHYSICAL_QUANTIZATION_CONTRACT,
    QUANTIZATION_DOMAIN_DTYPE,
    QUANTIZATION_REGION,
    PhysicalQuantizationContract,
    inherit_physical_quantization_contract,
    set_physical_quantization_contract,
)
from voyager_compiler.quantization.qspec import QScheme
from voyager_compiler.quantization.quantize_pt2e import (
    prepare_pt2e as shared_prepare,
)
from voyager_compiler.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
)
from voyager_compiler.quantization.quantizer.xnnpack_quantizer_utils import (
    QuantizationConfig,
)

from . import (
    domains,
    quant_ops,  # noqa: F401 — register Torch operator schemas
)
from .fx import (
    COMPILER_ONLY,
    SOURCE_OP_NAME,
    create_getattr_from_value,
    get_arg_value,
    is_gemm_op,
)
from .hardware import ALPHA_HARDWARE_CONFIG
from .observers import (
    FusedAmaxObsFakeQuantize,
    _DerivedObserverOrFakeQuantize,
    get_quantization_map,
)
from .qspec import DerivedQuantizationSpec, QuantizationSpec
from .quantization import (
    cleanup_quantization_preparation,
    finalize_quantization_calibration,
    prepare_quantization,
)

logger = logging.getLogger(__name__)

_PREPARED_DOMAIN_JOIN_TARGETS = domains._PREPARED_DOMAIN_JOIN_TARGETS

_PREPARED_DOMAIN_PASSTHROUGH_TARGETS = (
    domains._PREPARED_DOMAIN_PASSTHROUGH_TARGETS
)

_QUANTIZATION_DOMAIN_ENTRY = domains._QUANTIZATION_DOMAIN_ENTRY

_VSQ_JOIN_ALIAS = domains._VSQ_JOIN_ALIAS

_VSQ_JOIN_ENTRY = domains._VSQ_JOIN_ENTRY

_get_module = domains._get_module

_prepared_domain_data_inputs = domains._prepared_domain_data_inputs

_vsq_physical_quantization_contract = (
    domains._vsq_physical_quantization_contract
)

MX_OP_MAPPING = {
    torch.ops.aten.conv2d.default: torch.ops.alpha_quant.conv2d_mx.default
}


def _set_ch_axis(qspec: Optional[QuantizationSpec], ch_axis: int):
    if qspec is None:
        return None
    return replace(qspec, ch_axis=ch_axis)


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
        and weight_qspec.coarse_qscheme == QScheme.PER_CHANNEL_SYMMETRIC
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
        and weight_qspec.coarse_qscheme == QScheme.PER_CHANNEL_SYMMETRIC
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
    act_qspec = replace(input_activation, qscheme=QScheme.PER_TENSOR_SYMMETRIC)
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
    if QScheme.VSQ in qschemes:
        if any(qscheme != QScheme.VSQ for qscheme in qschemes):
            raise ValueError("VSQ cannot be mixed with another qscheme.")
        if bias is not None:
            raise ValueError("VSQ keeps bias unquantized; bias must be None.")
        return get_vsq_quantizer(
            input_activation,
            output_activation,
            weight,
        )

    if len(qschemes) > 0 and QScheme.MICROSCALING not in qschemes:
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

    if QScheme.MICROSCALING in qschemes:
        assert len(set(qschemes)) == 1, (
            f"Quantization scheme {qschemes[0]} does not work with "
            f"{qschemes[1]}"
        )
        raise ValueError(
            "Alpha inference PTQ supports integer and VSQ domains; use the shared quantizer for other targets"
        )

    if weight is not None and weight.qscheme == QScheme.PER_CHANNEL_SYMMETRIC:
        assert weight.ch_axis == 0, (
            "Per-channel weight quantization only supports quantizing output "
            "channel dimension (dim=0)."
        )

    if (
        input_activation is not None
        and input_activation.qscheme == QScheme.PER_CHANNEL_SYMMETRIC
    ):
        return get_per_channel_act_quantizer(
            input_activation,
            output_activation,
            weight,
            bias,
        )

    conv_output = output_activation
    linear_output = output_activation
    if (
        output_activation is not None
        and output_activation.qscheme == QScheme.VSQ
    ):
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
        and maybe_dequantize.target == torch.ops.alpha_quant.dequantize.default
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
            torch.ops.alpha_quant.dequantize.default,
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
            torch.ops.alpha_quant.dequantize.default,
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
        param.data = torch.ops.alpha_quant.quantize(
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
                torch.ops.alpha_quant.quantize.default,
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
    if node.target != torch.ops.alpha_quant.dequantize_vsq.default:
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
            quantized_value = torch.ops.alpha_quant.quantize_vsq.default(
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
                torch.ops.alpha_quant.quantize_vsq.default,
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
                torch.ops.alpha_quant.dequantize_vsq.default,
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


def prepare_pt2e(
    model,
    quantizer,
    args=None,
    kwargs=None,
    *,
    region_qspec=None,
    hardware_config=None,
):
    graph = shared_prepare(model, quantizer, args, kwargs)
    # Alpha's frozen inference recipe rounds derived bias codes through the
    # bfloat16 lookup table even when their logical dtype is unspecified.
    for observer in graph.modules():
        if isinstance(observer, _DerivedObserverOrFakeQuantize):
            if observer.qmap is None:
                observer.qmap = get_quantization_map(None)
            observer.qscheme = observer.obs_or_fqs[1].qscheme
    return prepare_quantization(
        graph, region_qspec, hardware_config or ALPHA_HARDWARE_CONFIG
    )


def finalize_pt2e_calibration(model, hardware_config=None):
    finalize_quantization_calibration(model)
    for observer in model.modules():
        if isinstance(observer, FusedAmaxObsFakeQuantize):
            observer.calculate_qparams()
    return model


def convert_pt2e(
    model, output_dtype=None, *, calibration_steps=0, hardware_config=None
):
    if calibration_steps < 0:
        raise ValueError("calibration_steps must be nonnegative")
    finalize_pt2e_calibration(model, hardware_config)
    cleanup_quantization_preparation(model)
    modules = dict(model.named_modules(remove_duplicate=False))
    _materialize_unobserved_per_channel_scales(model, modules)
    for node in list(model.graph.nodes):
        if node.op != "call_module":
            continue
        observer = modules[node.target]
        if not isinstance(observer, FakeQuantizeBase):
            continue
        qscheme = getattr(observer, "qscheme", None)
        if node.meta.get(domains._QUANTIZATION_DOMAIN_ALIAS):
            _replace_domain_alias(model, node, modules, output_dtype)
        elif node.meta.get(domains._VSQ_JOIN_ALIAS):
            _replace_vsq_join_alias(model, node, modules)
        elif qscheme == QScheme.VSQ:
            _replace_observer_with_quantize_vsq_node_decomposed(
                model, node, modules
            )
        elif qscheme in (
            None,
            QScheme.PER_TENSOR_SYMMETRIC,
            QScheme.PER_CHANNEL_SYMMETRIC,
        ):
            _replace_observer_with_quantize_dequantize_node_decomposed(
                model, node, modules, output_dtype
            )
        else:
            raise ValueError(
                f"unsupported Alpha quantization domain: {observer.qscheme}"
            )
    for node in model.graph.nodes:
        dtype = node.meta.get(QUANTIZATION_DOMAIN_DTYPE)
        if dtype is not None:
            value = getattr(node, "value", node.meta.get("val"))
            node.meta["dtype"] = (
                tuple(dtype for _ in value)
                if isinstance(value, (tuple, list))
                else dtype
            )
    # Scale-one nodes carry intended domains even when calibration is skipped.
    model.graph.lint()
    model.graph.eliminate_dead_code(
        is_impure_node=lambda n: n.op in {"placeholder", "output"}
    )
    model.recompile()
    model.delete_all_unused_submodules()
    return model

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
import math
import operator
from typing import List

import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher

from voyager_compiler.codegen.hardware import (
    COMMON_IMAGE_INPUT_CHANNELS,
    HardwareConfig,
    get_node_dtype_width,
    resolve_unroll_dims,
)
from voyager_compiler.codegen.mapping_utils import (
    is_conv2d,
    is_depthwise_conv,
    is_elementwise_op,
    is_gemm_op,
    is_matmul,
)
from voyager_compiler.codegen.passes.utils import (
    get_arg_value,
    inherit_hardware_logical_shape,
    preserve_hardware_logical_shape,
)
from voyager_compiler.codegen.target_backends import (
    get_target_operator_traits,
    pad_target_matrix_output_operands,
)
from voyager_compiler.pt2e_utils import (
    create_getattr_from_value,
    fetch_attr,
    get_aten_graph_module,
    propagate_shape,
)

logger = logging.getLogger(__name__)

__all__ = [
    "pad_matrix_op_dimensions",
    "pad_vector_op_dimensions",
    "pad_vit_embeddings_output",
]


def slicing_and_padding_cancel_out(shape, slice_dim, start, end, pad):
    ndim = len(shape)
    k = len(pad) // 2

    pad_pairs = list(reversed(list(zip(pad[0::2], pad[1::2]))))
    full_pad_pairs = [(0, 0)] * (ndim - k) + pad_pairs

    for dim, (left, right) in enumerate(full_pad_pairs):
        if dim != slice_dim:
            if left != 0 or right != 0:
                return False
        else:
            if left != start or right != shape[slice_dim] - end:
                return False

    return True


def pad_input_node(model, node, input, pad, scale, scale_pad):
    pad_quantize_mx_input = (
        scale is not None
        and input.target == operator.getitem
        and scale.target == operator.getitem
        and input.args[0] == scale.args[0]
        and input.args[0].target
        in {
            torch.ops.quantized_ops.quantize_mx.default,
            torch.ops.quantized_ops.quantize_mx_outlier.default,
        }
    )

    node_to_pad = input.args[0].args[0] if pad_quantize_mx_input else input

    skip_padding = False
    if node_to_pad.target == torch.ops.aten.slice.Tensor:
        arg = node_to_pad.args[0]
        skip_padding = slicing_and_padding_cancel_out(
            arg.value.shape, *node_to_pad.args[1:], pad
        )

    if skip_padding:
        new_input = node_to_pad.args[0]
    else:
        with model.graph.inserting_after(node_to_pad):
            new_input = model.graph.call_function(
                torch.ops.aten.pad.default,
                (node_to_pad, pad),
            )

        propagate_shape(new_input)
        new_input.meta["dtype"] = node_to_pad.meta.get("dtype")
        inherit_hardware_logical_shape(new_input, node_to_pad)

    if pad_quantize_mx_input:
        quantize_mx = input.args[0]
        quantize_args = list(quantize_mx.args)
        quantize_args[0] = new_input
        with model.graph.inserting_before(node):
            padded_quantize_mx = model.graph.call_function(
                quantize_mx.target,
                tuple(quantize_args),
                dict(quantize_mx.kwargs),
            )
            padded_scale = model.graph.call_function(
                operator.getitem,
                (padded_quantize_mx, scale.args[1]),
            )
            padded_input = model.graph.call_function(
                operator.getitem,
                (padded_quantize_mx, input.args[1]),
            )

        padded_quantize_mx.meta["dtype"] = quantize_mx.meta.get("dtype")
        propagate_shape(padded_quantize_mx)
        propagate_shape(padded_scale)
        propagate_shape(padded_input)
        padded_scale.meta["dtype"] = scale.meta.get("dtype")
        padded_input.meta["dtype"] = input.meta.get("dtype")
        inherit_hardware_logical_shape(padded_scale, scale)
        inherit_hardware_logical_shape(padded_input, input)
        node.replace_input_with(scale, padded_scale)
        node.replace_input_with(input, padded_input)
    else:
        node.replace_input_with(node_to_pad, new_input)

        if scale is not None and any(x for x in scale_pad):
            with model.graph.inserting_before(node):
                padded_scale = model.graph.call_function(
                    torch.ops.aten.pad.default,
                    (scale, scale_pad),
                )

            node.replace_input_with(scale, padded_scale)

            propagate_shape(padded_scale)
            padded_scale.meta["dtype"] = scale.meta.get("dtype")
            inherit_hardware_logical_shape(padded_scale, scale)


def _constant_zero_channel_tail_pad(node: Node) -> tuple[Node, int] | None:
    """Parse one rank-four NCHW constant-zero channel-tail pad."""
    if node.target != torch.ops.aten.pad.default:
        return None

    source = get_arg_value(node, 0, "input")
    padding = get_arg_value(node, 1, "pad")
    mode = get_arg_value(node, 2, "mode", "constant")
    value = get_arg_value(node, 3, "value")
    if not isinstance(source, Node):
        return None
    if mode not in (None, "constant"):
        return None
    if value is not None and (
        not isinstance(value, (int, float)) or value != 0
    ):
        return None

    source_shape = tuple(getattr(source, "shape", ()) or ())
    output_shape = tuple(getattr(node, "shape", ()) or ())
    if len(source_shape) != 4 or len(output_shape) != 4:
        return None
    if (
        not isinstance(padding, (tuple, list))
        or len(padding) % 2
        or len(padding) > 2 * len(source_shape)
    ):
        return None

    # Aten lists padding from the last dimension toward the first.  NCHW's
    # channel pair is therefore the third pair in the padding list.
    channel_right_index = 2 * (len(source_shape) - 2) + 1
    if len(padding) <= channel_right_index:
        return None
    if any(
        amount != 0
        for index, amount in enumerate(padding)
        if index != channel_right_index
    ):
        return None

    tail = padding[channel_right_index]
    if isinstance(tail, bool) or not isinstance(tail, int) or tail <= 0:
        return None
    expected_shape = list(source_shape)
    expected_shape[1] += tail
    if tuple(expected_shape) != output_shape:
        return None
    return source, tail


def _matrix_source_for_output_pad(
    pad: Node,
    operator_traits,
) -> Node | None:
    """Trace a direct output pad through declared shape-preserving epilogues."""
    parsed = _constant_zero_channel_tail_pad(pad)
    if parsed is None:
        return None
    current, _tail = parsed
    epilogue_operators = (
        ()
        if operator_traits is None
        else operator_traits.output_padding_epilogue_operators
    )
    component_operators = (
        {}
        if operator_traits is None
        else operator_traits.output_padding_components
    )
    visited = set()

    while current not in visited:
        visited.add(current)
        if is_gemm_op(current, operator_traits):
            return current
        if (
            current.target == operator.getitem
            and len(current.args) > 1
            and isinstance(current.args[0], Node)
        ):
            owner = current.args[0]
            components = component_operators.get(owner.target, ())
            if current.args[1] not in components:
                return None
            current = owner
            continue
        if current.target not in epilogue_operators:
            return None
        source = get_arg_value(current, 0, "input")
        if not isinstance(source, Node):
            return None
        current = source
    return None


def _foldable_matrix_output_pads(
    model: GraphModule,
    hardware_config: HardwareConfig,
    operator_traits,
) -> tuple[dict[Node, int], set[Node]]:
    """Plan direct pads that an exactly sized physical matrix output can own."""
    candidates = {}
    for pad in model.graph.nodes:
        parsed = _constant_zero_channel_tail_pad(pad)
        if parsed is None:
            continue
        source, tail = parsed
        producer = _matrix_source_for_output_pad(pad, operator_traits)
        if producer is None:
            continue

        source_shape = tuple(getattr(source, "shape", ()) or ())
        producer_shape = tuple(getattr(producer, "shape", ()) or ())
        if source_shape != producer_shape or len(producer_shape) != 4:
            continue
        output_channels = producer_shape[1]
        resident_channels = output_channels + tail
        input_node = get_arg_value(producer, 0, "input")
        if not isinstance(input_node, Node):
            continue
        input_bits = get_node_dtype_width(input_node)
        if (
            hardware_config.matrix_output_resident_channels(
                resident_channels,
                input_bits,
            )
            != resident_channels
        ):
            continue
        default_resident = hardware_config.matrix_output_resident_channels(
            output_channels,
            input_bits,
        )
        if default_resident > resident_channels:
            continue
        candidates.setdefault(producer, {}).setdefault(
            resident_channels,
            set(),
        ).add(pad)

    resident_overrides = {}
    folded_pads = set()
    for producer, by_resident in candidates.items():
        if len(by_resident) != 1:
            continue
        resident_channels, pads = next(iter(by_resident.items()))
        resident_overrides[producer] = resident_channels
        folded_pads.update(pads)
    return resident_overrides, folded_pads


def _fold_output_pad_as_view(pad: Node, physical_source: Node) -> None:
    """Keep the padded logical value as a view of its physical producer."""
    logical_shape = tuple(getattr(pad, "shape", ()) or ())
    physical_shape = tuple(getattr(physical_source, "shape", ()) or ())
    if not logical_shape or logical_shape != physical_shape:
        raise ValueError(
            f"{pad.name}: folded output pad shape {logical_shape} does not "
            f"match physical producer shape {physical_shape}"
        )

    preserve_hardware_logical_shape(pad)
    pad.target = torch.ops.aten.view.default
    pad.args = (physical_source, physical_shape)
    pad.kwargs = {}
    propagate_shape(pad)
    pad.meta["dtype"] = physical_source.meta.get("dtype")


def slice_output(
    model,
    output_node,
    slice_args,
    operator_traits=None,
    folded_output_pads=None,
):
    if folded_output_pads is None:
        folded_output_pads = set()
    output_padding_components = (
        {}
        if operator_traits is None
        else operator_traits.output_padding_components
    )
    output_padding_epilogues = (
        ()
        if operator_traits is None
        else operator_traits.output_padding_epilogue_operators
    )
    component_indices = output_padding_components.get(
        output_node.target,
    )
    if component_indices is not None:
        components = {
            user.args[1]: user
            for user in output_node.users
            if (
                user.target == operator.getitem
                and len(user.args) > 1
                and user.args[1] in component_indices
            )
        }
        missing = set(component_indices) - set(components)
        if missing:
            raise ValueError(
                f"{output_node.name}: missing output component(s) "
                f"{sorted(missing)} required for hardware padding"
            )
        for index in component_indices:
            propagate_shape(components[index])
            slice_output(
                model,
                components[index],
                slice_args,
                operator_traits,
                folded_output_pads,
            )
        return

    sliced_output_users = []

    for user in list(output_node.users.keys()):
        if user in folded_output_pads:
            _fold_output_pad_as_view(user, output_node)
            continue

        if user.target in output_padding_epilogues:
            preserve_hardware_logical_shape(user)
            propagate_shape(user)
            slice_output(
                model,
                user,
                slice_args,
                operator_traits,
                folded_output_pads,
            )
            continue

        # Cannot slice microscaling quantization ops.
        if user.target in [
            torch.ops.quantized_ops.dequantize.default,
            torch.ops.quantized_ops.quantize.default,
        ]:
            bs = get_arg_value(user, 4, "block_size")
            if bs is None:
                propagate_shape(user)
                slice_output(
                    model,
                    user,
                    slice_args,
                    operator_traits,
                    folded_output_pads,
                )
                continue

        if is_elementwise_op(user):
            if len(user.all_input_nodes) == 1:
                propagate_shape(user)
                slice_output(
                    model,
                    user,
                    slice_args,
                    operator_traits,
                    folded_output_pads,
                )
                continue

            # If all inputs have the same slice args, strip the redundant slice.
            if all(
                n == output_node
                or (
                    n.target == torch.ops.aten.slice.Tensor
                    and n.args[1:] == slice_args
                )
                for n in user.all_input_nodes
            ):
                for n in user.all_input_nodes:
                    if n.target == torch.ops.aten.slice.Tensor:
                        user.replace_input_with(n, n.args[0])

                propagate_shape(user)
                slice_output(
                    model,
                    user,
                    slice_args,
                    operator_traits,
                    folded_output_pads,
                )
                continue

        sliced_output_users.append(user)

    if sliced_output_users:
        with model.graph.inserting_after(output_node):
            slice_node = model.graph.call_function(
                torch.ops.aten.slice.Tensor,
                (output_node, *slice_args),
            )

        propagate_shape(slice_node)
        slice_node.meta["dtype"] = output_node.meta.get("dtype")
        slice_node.meta["hardware_padding_slice"] = True

        for n in sliced_output_users:
            n.replace_input_with(output_node, slice_node)


def _pad_output_dequantize_scales(
    model: GraphModule,
    node: Node,
    output_channels: int,
    output_padding: int,
) -> None:
    """Zero-fill static per-channel scales for inactive output lanes."""
    padded_targets = set()

    for user in node.users:
        if user.target != torch.ops.quantized_ops.dequantize.default:
            continue
        scale = get_arg_value(user, 1, "scale")
        if not isinstance(scale, Node) or scale.op != "get_attr":
            continue
        if scale.target in padded_targets:
            continue

        scale_param = fetch_attr(model, scale.target)
        if scale_param.numel() == 1:
            continue
        if scale_param.shape[0] == output_channels + output_padding:
            continue
        if scale_param.shape[0] != output_channels:
            raise ValueError(
                f"Dequantization scale {scale.target} has shape "
                f"{tuple(scale_param.shape)}, expected leading dimension "
                f"{output_channels}"
            )

        scale_pad = [0, 0] * (scale_param.ndim - 1)
        scale_pad.extend((0, output_padding))
        scale_param.data = F.pad(
            scale_param.data,
            scale_pad,
            value=0.0,
        )
        propagate_shape(scale, model)
        padded_targets.add(scale.target)


def _pad_folded_output_scale(
    model: GraphModule,
    node: Node,
    output_channels: int,
    output_padding: int,
    operator_traits=None,
) -> None:
    """Pad a target-owned folded per-output scale with inactive lanes."""
    folded_scale_operators = (
        ()
        if operator_traits is None
        else operator_traits.folded_scale_operators
    )
    if node.target not in folded_scale_operators:
        return
    scale = node.kwargs.get("folded_coarse_scale")
    if not isinstance(scale, Node) or scale.op != "get_attr":
        return

    scale_param = fetch_attr(model, scale.target)
    if scale_param.numel() == 1:
        return
    if scale_param.numel() == output_channels + output_padding:
        return
    if scale_param.numel() != output_channels:
        raise ValueError(
            f"Folded scale {scale.target} contains {scale_param.numel()} "
            f"values, expected {output_channels}"
        )

    scale_param.data = F.pad(
        scale_param.reshape(-1),
        (0, output_padding),
        value=0.0,
    )
    propagate_shape(scale, model)


def pad_matrix_op_dimensions(
    model: GraphModule,
    C_unroll=None,
    K_unroll=None,
    *,
    hardware_config: HardwareConfig | None = None,
    operator_traits=None,
) -> GraphModule:
    """
    Pad inputs and weights to conv2d nodes in a torch.fx.GraphModule so that
    the input channels (C) and output channels (K) are multiples of the
    provided unroll factors.

    Parameters:
        model (torch.fx.GraphModule): The FX graph module to transform.
        C_unroll (int): Fixed input-channel unroll for the legacy path.
        K_unroll (int): Fixed output-channel unroll for the legacy path.
        hardware_config: Dtype-aware hardware profile. Mutually exclusive
            with fixed unroll factors.

    Returns:
        torch.fx.GraphModule: The transformed FX graph module.
    """
    if operator_traits is None and hardware_config is not None:
        operator_traits = get_target_operator_traits(hardware_config.target)
    resident_overrides = {}
    folded_output_pads = set()
    if hardware_config is not None:
        resident_overrides, folded_output_pads = _foldable_matrix_output_pads(
            model,
            hardware_config,
            operator_traits,
        )

    for node in list(model.graph.nodes):
        if not is_gemm_op(node, operator_traits):
            continue

        is_dw = is_depthwise_conv(node, operator_traits)
        is_conv = is_conv2d(node, operator_traits)
        is_mm = is_matmul(node)

        input = node.args[0]
        C_in = input.shape[1] if is_conv else input.shape[-1]

        # A dedicated partial-vector input path masks unused matrix lanes.
        legacy_partial_input = (
            hardware_config is None and C_in in COMMON_IMAGE_INPUT_CHANNELS
        )
        configured_partial_input = (
            hardware_config is not None
            and hardware_config.supports_partial_matrix_input(C_in)
        )
        uses_partial_matrix_input = is_conv and (
            legacy_partial_input or configured_partial_input
        )

        node_C_unroll, node_K_unroll = resolve_unroll_dims(
            input,
            (C_unroll, K_unroll) if hardware_config is None else None,
            hardware_config,
        )
        node_input_bits = (
            None if hardware_config is None else get_node_dtype_width(input)
        )

        weight = node.args[1]
        weight_C_out = weight.shape[-1] if is_mm else weight.shape[0]
        pad_depthwise_channels = (
            hardware_config is None or hardware_config.pad_depthwise_channels
        )
        if is_dw and pad_depthwise_channels:
            channel_multiplier = weight_C_out // C_in
            output_group_alignment = node_K_unroll // math.gcd(
                node_K_unroll, channel_multiplier
            )
            group_alignment = math.lcm(
                node_C_unroll,
                output_group_alignment,
            )
            pad_C = (-C_in) % group_alignment
        elif is_dw:
            channel_multiplier = weight_C_out // C_in
            pad_C = 0
        elif uses_partial_matrix_input:
            channel_multiplier = None
            pad_C = 0
        else:
            channel_multiplier = None
            pad_C = (-C_in) % node_C_unroll

        bs = node.kwargs.get("block_size", 1)

        # Pad input along C dimension
        if pad_C:
            input_scale = node.kwargs.get("input_scale")

            scale_pad_C = (C_in + pad_C + bs - 1) // bs - (C_in + bs - 1) // bs

            if is_conv:
                input_pad = [0, 0, 0, 0, 0, pad_C]
                scale_pad = [0, 0, 0, 0, 0, scale_pad_C]
            else:
                input_pad = [0, pad_C]
                scale_pad = [0, scale_pad_C]

            pad_input_node(
                model, node, input, input_pad, input_scale, scale_pad
            )

        C_in = weight.shape[-2] if is_mm else weight.shape[1]
        C_out = weight.shape[-1] if is_mm else weight.shape[0]

        if is_dw:
            C_in *= node.args[6]
            node.args = node.args[:-1] + (node.args[-1] + pad_C,)
            pad_K = pad_C * channel_multiplier
        else:
            pad_C = 0 if uses_partial_matrix_input else (-C_in) % node_C_unroll
            if hardware_config is None:
                pad_K = (-C_out) % node_K_unroll
            else:
                resident_K = hardware_config.matrix_output_resident_channels(
                    C_out,
                    node_input_bits,
                )
                resident_K = resident_overrides.get(node, resident_K)
                pad_K = resident_K - C_out

        # Pad weight along K and C dimensions
        if pad_C or pad_K:
            if pad_K:
                preserve_hardware_logical_shape(node)
            weight_scale = node.kwargs.get("weight_scale")

            scale_pad_C = (C_in + pad_C + bs - 1) // bs - (C_in + bs - 1) // bs

            if is_dw:
                weight_pad = [0, 0, 0, 0, 0, 0, 0, pad_K]
                ws_pad = [0, 0, 0, 0, 0, 0, 0, pad_K]
            elif is_conv:
                weight_pad = [0, 0, 0, 0, 0, pad_C, 0, pad_K]
                ws_pad = [0, 0, 0, 0, 0, scale_pad_C, 0, pad_K]
            elif is_mm:
                weight_pad = [0, pad_K, 0, pad_C]
                ws_pad = [0, pad_K, 0, scale_pad_C]
            else:
                weight_pad = [0, pad_C, 0, pad_K]
                ws_pad = [0, scale_pad_C, 0, pad_K]

            if weight.op == "get_attr":
                logger.debug(f"Pad {weight} with {weight_pad}")
                preserve_hardware_logical_shape(weight)
                param = fetch_attr(model, weight.target)
                param.data = F.pad(param.data, weight_pad)
                propagate_shape(weight, model)

                if weight_scale is not None:
                    logger.debug(f"Pad {weight_scale} with {ws_pad}")
                    preserve_hardware_logical_shape(weight_scale)
                    scale_param = fetch_attr(model, weight_scale.target)
                    scale_param.data = F.pad(scale_param.data, ws_pad)
                    propagate_shape(weight_scale, model)
            else:
                pad_input_node(
                    model, node, weight, weight_pad, weight_scale, ws_pad
                )

            if pad_K and len(node.args) > 2 and node.args[2] is not None:
                bias = node.args[2]
                preserve_hardware_logical_shape(bias)
                bias_param = fetch_attr(model, bias.target)
                bias_param.data = F.pad(bias_param.data, [0, pad_K])
                propagate_shape(bias, model)

            if pad_K:
                _pad_folded_output_scale(
                    model,
                    node,
                    C_out,
                    pad_K,
                    operator_traits,
                )
                _pad_output_dequantize_scales(
                    model,
                    node,
                    C_out,
                    pad_K,
                )
                if hardware_config is not None:
                    pad_target_matrix_output_operands(
                        model,
                        node,
                        C_out,
                        pad_K,
                        hardware_config.target,
                    )

        propagate_shape(node)

        if pad_K:
            slice_dim = 1 if is_conv else -1
            slice_output(
                model,
                node,
                (slice_dim, 0, C_out),
                operator_traits,
                folded_output_pads,
            )

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def pad_layer_norm_to_hardware_unroll_size(
    model: GraphModule,
    node: Node,
    unroll: int,
) -> GraphModule:
    input = node.args[0]
    normalize_shape = node.args[1]
    weight = node.args[2]
    bias = node.args[3] if len(node.args) > 3 else None

    orig_k = input.shape[-1]
    pad_k = (-orig_k) % unroll
    if pad_k == 0:
        return model

    logger.info(f"Padding layer_norm {node} last dimension with {pad_k}")

    def pad_param(attr_node: Node):
        if attr_node.op == "get_attr":
            param = fetch_attr(model, attr_node.target)
            new_param = F.pad(param, [0, pad_k])
            with model.graph.inserting_after(attr_node):
                new_attr = create_getattr_from_value(
                    model, model.graph, f"{attr_node.target}_padded", new_param
                )
        else:
            with model.graph.inserting_after(attr_node):
                new_attr = model.graph.call_function(
                    torch.ops.aten.pad.default,
                    (attr_node, [0, pad_k]),
                )
        propagate_shape(new_attr, model)
        return new_attr

    new_weight = pad_param(weight)
    new_bias = pad_param(bias) if bias is not None else None

    with model.graph.inserting_before(node):
        new_input = model.graph.call_function(
            torch.ops.aten.pad.default,
            (input, [0, pad_k]),
        )
        layer_norm = model.graph.call_function(
            torch.ops.quantized_ops.layer_norm.default,
            (new_input, normalize_shape, new_weight, new_bias) + node.args[4:],
        )
        slice_node = model.graph.call_function(
            torch.ops.aten.slice.Tensor,
            (layer_norm, -1, 0, orig_k),
        )

    propagate_shape(new_input)
    new_input.meta["dtype"] = input.meta.get("dtype")

    propagate_shape(layer_norm)

    propagate_shape(slice_node)
    node.replace_all_uses_with(slice_node)
    model.graph.erase_node(node)


def pad_calculate_mx_qparam(model, node, unroll):
    input, axes, bs = node.args[:3]
    ndim = len(input.shape)

    pad_dims = {}

    orig_k = input.shape[-1]
    if orig_k % unroll != 0:
        pad_dims[ndim - 1] = (-orig_k) % unroll

    for axis in axes:
        size = input.shape[axis]
        if size % bs != 0:
            pad_dims[axis % ndim] = (bs - (size % bs)) % bs

    if not pad_dims:
        return

    logger.info(f"Padding calculate_mx_qparams {node} with {pad_dims}")

    min_pad_dim = min(pad_dims.keys())
    pad_tuple = []
    for dim in range(ndim - 1, min_pad_dim - 1, -1):
        pad_tuple.extend([0, pad_dims.get(dim, 0)])

    with model.graph.inserting_before(node):
        new_input = model.graph.call_function(
            torch.ops.aten.pad.default,
            (input, pad_tuple, "constant", 0),
        )

    propagate_shape(new_input)
    new_input.meta["dtype"] = input.meta.get("dtype")
    node.replace_input_with(input, new_input)

    propagate_shape(node)

    def slice_node(n):
        with model.graph.inserting_after(n.next):
            for dim in pad_dims.keys():
                n = model.graph.call_function(
                    torch.ops.aten.slice.Tensor,
                    (n, dim, 0, input.shape[dim]),
                )
            propagate_shape(n)
            n.meta["dtype"] = node.meta.get("dtype")
        return n

    scale_node = slice_node(node)

    for user in list(node.users):
        if user.target == torch.ops.quantized_ops.quantize.default:
            users = list(user.users)
            new_q_node = slice_node(user)
            for u in users:
                u.replace_input_with(user, new_q_node)
            user.replace_input_with(input, new_input)
        elif user != scale_node:
            user.replace_input_with(node, scale_node)


def pad_vector_op_dimensions(
    model: GraphModule,
    K_unroll,
) -> GraphModule:
    """
    Pad inputs to vector operations to multiples of the hardware unroll size.
    Only support softmax operation for now.

    Parameters:
        model (torch.fx.GraphModule): The FX graph module to transform.
        K_unroll (int): Unroll factor for the output channels.

    Returns:
        torch.fx.GraphModule: The transformed FX graph module.
    """
    for node in list(model.graph.nodes):
        if node.target == torch.ops.aten.layer_norm.default:
            pad_layer_norm_to_hardware_unroll_size(model, node, K_unroll)
            continue

        if node.target == torch.ops.quantized_ops.calculate_mx_qparam.default:
            pad_calculate_mx_qparam(model, node, K_unroll)
            continue

        if node.target != torch.ops.aten.softmax.int:
            continue

        input = node.args[0]
        reduction_dim = input.shape[-1]

        pad_k = (K_unroll - (reduction_dim % K_unroll)) % K_unroll

        if not pad_k:
            continue

        with model.graph.inserting_after(input):
            new_input = model.graph.call_function(
                torch.ops.aten.pad.default,
                (input, [0, pad_k], "constant", float("-inf")),
            )

        propagate_shape(new_input)
        new_input.meta["dtype"] = input.meta.get("dtype")
        node.replace_input_with(input, new_input)

        propagate_shape(node)

        with model.graph.inserting_after(node):
            slice_node = model.graph.call_function(
                torch.ops.aten.slice.Tensor,
                (node, -1, 0, reduction_dim),
            )

        node.replace_all_uses_with(slice_node)
        slice_node.replace_input_with(slice_node, node)

        propagate_shape(slice_node)
        slice_node.meta["dtype"] = node.meta.get("dtype")

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    return model


def pad_vit_embeddings_output(
    model: GraphModule,
    embeddings,
    example_inputs,
    dynamic_shapes=None,
    unroll=32,
):
    original_graph = model.graph

    pattern = get_aten_graph_module(
        embeddings, example_inputs, dynamic_shapes=dynamic_shapes
    )
    pattern_graph = pattern.graph

    matcher = SubgraphMatcher(
        pattern_graph,
        match_output=False,
        match_placeholder=False,
        remove_overlapping_matches=True,
        ignore_literals=False,
    )
    _matches: List[InternalMatch] = matcher.match(original_graph)
    logger.info(f"Found {len(_matches)} matches")

    if not _matches:
        return model

    vit_embed_out = _matches[0].returning_nodes[0]
    orig_dim = vit_embed_out.meta["val"].shape[-2]
    pad = (unroll - (orig_dim % unroll)) % unroll
    logger.info(f"Padding {vit_embed_out} with {pad}")

    with model.graph.inserting_after(vit_embed_out):
        pad_node = model.graph.call_function(
            torch.ops.aten.pad.default,
            (vit_embed_out, [0, 0, 0, pad]),
        )

    propagate_shape(pad_node)
    pad_node.meta["val"] = pad_node.value

    for user in list(vit_embed_out.users):
        if id(user) != id(pad_node):
            user.replace_input_with(vit_embed_out, pad_node)

    for node in model.graph.nodes:
        if node.target in [
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
        ]:
            new_size = [x if x != orig_dim else x + pad for x in node.args[1]]
            node.args = (node.args[0], new_size)

    model.graph.lint()
    model.recompile()
    return model

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Magnet Alpha PE and GEMM target lowering."""

import copy
import logging
import operator

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.ir.graph import (
    PHYSICAL_QUANTIZATION_CONTRACT,
    QUANTIZATION_REGION,
    PhysicalQuantizationContract,
    get_physical_dtype,
    get_physical_quantization_contract,
    set_physical_quantization_contract,
)
from voyager_compiler.targets.magnet_alpha.domains import (
    _QUANTIZATION_DOMAIN_ALIAS,
    _QUANTIZATION_DOMAIN_ENTRY,
)
from voyager_compiler.targets.magnet_alpha.fixed_point import (
    ALPHA_Q8_8_REFERENCE_FORMAT,
    alpha_output_domain,
    alpha_w8_accumulator_real_bound,
    encode_alpha_vsq_scales,
    encode_selected_alpha_pe_pipeline,
    reconstruct_accumulator_bias,
    select_alpha_pe_normalization,
)
from voyager_compiler.targets.magnet_alpha.fx import (
    COMPILATION_TARGET_META,
    MAGNET_ALPHA_COMPILATION_TARGET,
    create_getattr_from_value,
    fetch_attr,
    get_arg_value,
    get_identity_slice_source,
    get_leading_channel_slice_source,
    propagate_shape,
)
from voyager_compiler.targets.magnet_alpha.hardware import (
    ALPHA_VSQ_BLOCK_VALUES,
)
from voyager_compiler.targets.magnet_alpha.lowering_utils import (
    clone_bias_free_conv as _clone_bias_free_conv,
)
from voyager_compiler.targets.magnet_alpha.lowering_utils import (
    node_shape,
    relu_reaches_alpha_eadd,
)
from voyager_compiler.targets.magnet_alpha.lowering_utils import (
    output_quantizer_info as _output_quantizer_info,
)
from voyager_compiler.targets.magnet_alpha.lowering_utils import (
    output_quantizers as _output_quantizers,
)
from voyager_compiler.targets.magnet_alpha.lowering_utils import (
    resolve_static_tensor as _resolve_static_tensor,
)
from voyager_compiler.targets.magnet_alpha.lowering_utils import (
    vsq_scale_endpoints as _vsq_scale_endpoints,
)
from voyager_compiler.targets.magnet_alpha.operators import (
    PPU_FIXED_RELU_OPERATOR,
    PVS_QUANTIZE_OPERATOR,
    VSQ_CONV2D_OPERATOR,
    VSQ_SCALE_ENCODE_OPERATOR,
)

logger = logging.getLogger(__name__)


def _static_vsq_coarse_scale_operand(value):
    """Return the static attribute that declares a VSQ coarse scale."""
    if not isinstance(value, Node):
        return None

    if (
        value.target == operator.getitem
        and value.args[1] == 0
        and isinstance(value.args[0], Node)
        and value.args[0].target == torch.ops.alpha_quant.quantize_vsq.default
    ):
        value = get_arg_value(value.args[0], 1, "coarse_scale")

    if not isinstance(value, Node) or value.op != "get_attr":
        return None
    return value


def _resolve_static_vsq_coarse_scale(model: GraphModule, value):
    """Resolve the declared coarse scale carried by a VSQ value."""
    value = _static_vsq_coarse_scale_operand(value)
    if value is None:
        return None
    resolved = fetch_attr(model, value.target)
    return resolved if isinstance(resolved, torch.Tensor) else None


def _converted_pe_boundary(
    model: GraphModule,
    producer: Node,
) -> Node | None:
    """Recognize one Magnet Alpha PE boundary from generic converted Q/DQ."""
    if (
        model.meta.get(COMPILATION_TARGET_META)
        != MAGNET_ALPHA_COMPILATION_TARGET
    ):
        return None
    current = producer
    users = tuple(current.users)
    if (
        len(users) == 1
        and users[0].target == torch.ops.alpha_quant.dequantize.default
    ):
        current = users[0]
        users = tuple(current.users)

    boundary = (
        users[0]
        if (len(users) == 1 and users[0].target == torch.ops.aten.relu.default)
        else current
    )
    if _output_quantizer_info(model, boundary) is None:
        return None
    return boundary


def lower_magnet_alpha_vsq_convolutions(
    model: GraphModule,
) -> GraphModule:
    """Lower VSQ Conv2d nodes recognized from generic converted Q/DQ."""
    graph = model.graph
    lowered = 0
    required_scales = (
        "input_scale",
        "input_coarse_scale",
        "weight_scale",
        "weight_coarse_scale",
    )

    for node in list(graph.nodes):
        if node.target != torch.ops.alpha_quant.conv2d_mx.default:
            continue
        groups = int(get_arg_value(node, 6, "groups", 1))
        if groups != 1 or _converted_pe_boundary(model, node) is None:
            logger.debug(
                "%s: leaving VSQ Conv2d without a recognized Magnet Alpha PE "
                "output domain in its semantic form",
                node.name,
            )
            continue
        if not all(
            node.kwargs.get(name) is not None for name in required_scales
        ):
            continue
        if node.kwargs.get("input_code") is not None:
            continue
        if node.kwargs.get("weight_code") is not None:
            continue

        input_value = get_arg_value(node, 0, "input")
        weight_value = get_arg_value(node, 1, "weight")
        input_scale = node.kwargs["input_scale"]
        weight_scale = node.kwargs["weight_scale"]
        input_coarse_scale = node.kwargs["input_coarse_scale"]
        weight_coarse_scale = node.kwargs["weight_coarse_scale"]
        semantic_input_coarse_scale = _static_vsq_coarse_scale_operand(
            input_coarse_scale
        )
        semantic_weight_coarse_scale = _static_vsq_coarse_scale_operand(
            weight_coarse_scale
        )
        block_size = node.kwargs["block_size"]
        bias = get_arg_value(node, 2, "bias")
        input_coarse = _resolve_static_vsq_coarse_scale(
            model,
            input_coarse_scale,
        )
        weight_coarse = _resolve_static_vsq_coarse_scale(
            model,
            weight_coarse_scale,
        )
        if input_coarse is None or weight_coarse is None:
            logger.debug(
                "%s: leaving dynamic VSQ coarse scales unlowered",
                node.name,
            )
            continue
        if input_coarse.numel() != 1:
            logger.debug(
                "%s: leaving non-scalar input coarse scale unlowered",
                node.name,
            )
            continue

        input_fine_endpoint, input_coarse_endpoint = _vsq_scale_endpoints(
            input_scale
        )
        weight_fine_endpoint, weight_coarse_endpoint = _vsq_scale_endpoints(
            weight_scale
        )
        if int(block_size) == ALPHA_VSQ_BLOCK_VALUES:
            input_encoding = encode_alpha_vsq_scales(
                input_coarse.new_ones(1),
                input_coarse,
                fine_quant_max=input_fine_endpoint,
                coarse_scale_quant_max=input_coarse_endpoint,
            )
            weight_encoding = encode_alpha_vsq_scales(
                weight_coarse.new_ones(1),
                weight_coarse,
                fine_quant_max=weight_fine_endpoint,
                coarse_scale_quant_max=weight_coarse_endpoint,
            )
            folded_value = input_encoding.coarse_scale.detach().reshape(-1)[
                0
            ] * weight_encoding.coarse_scale.detach().reshape(-1)
        else:
            # Non-Alpha block sizes remain a reusable semantic reference.
            # Do not impose Alpha's uint8 scale encoding on those graphs.
            folded_value = input_coarse.detach().reshape(-1)[
                0
            ] * weight_coarse.detach().reshape(-1)
        source_name = node.meta.get("source_op_name", node.name)

        with graph.inserting_before(node):
            folded_coarse_scale = create_getattr_from_value(
                model,
                graph,
                f"{source_name}_folded_coarse_scale",
                folded_value,
            )
            target_node = graph.call_function(
                VSQ_CONV2D_OPERATOR,
                (
                    input_value,
                    weight_value,
                    bias,
                    get_arg_value(node, 3, "stride", (1, 1)),
                    get_arg_value(node, 4, "padding", (0, 0)),
                    get_arg_value(node, 5, "dilation", (1, 1)),
                    groups,
                ),
                {
                    "input_scale": input_scale,
                    "weight_scale": weight_scale,
                    "folded_coarse_scale": folded_coarse_scale,
                    "block_size": block_size,
                    "input_fine_quant_max": input_fine_endpoint,
                    "input_coarse_scale_quant_max": (input_coarse_endpoint),
                    "weight_fine_quant_max": weight_fine_endpoint,
                    "weight_coarse_scale_quant_max": (weight_coarse_endpoint),
                    "semantic_input_coarse_scale": (
                        semantic_input_coarse_scale
                    ),
                    "semantic_weight_coarse_scale": (
                        semantic_weight_coarse_scale
                    ),
                },
            )

        propagate_shape(folded_coarse_scale, model)
        folded_coarse_scale.meta["magnet_alpha_instruction_operand"] = True
        for semantic_operand in (
            semantic_input_coarse_scale,
            semantic_weight_coarse_scale,
        ):
            if isinstance(semantic_operand, Node):
                semantic_operand.meta["magnet_alpha_semantic_operand"] = True

        target_node.meta = copy.copy(node.meta)
        if hasattr(node, "shape"):
            target_node.shape = node.shape
        if hasattr(node, "value"):
            target_node.value = node.value
        node.replace_all_uses_with(target_node)
        graph.erase_node(node)
        lowered += 1

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    logger.info("Lowered %d Magnet Alpha VSQ convolutions", lowered)
    return model


def _static_w8_accumulator_real_bound(
    model: GraphModule,
    conv: Node,
    folded_scale: torch.Tensor,
) -> torch.Tensor:
    """Bound every W8 PE accumulator from frozen physical weight codes."""
    weight_value = _resolve_static_tensor(
        model,
        get_arg_value(conv, 1, "weight"),
    )
    if weight_value is None or weight_value.ndim < 2:
        raise ValueError(
            f"{conv.name}: direct-output PE requires one static W8 weight"
        )
    try:
        return alpha_w8_accumulator_real_bound(
            weight_value,
            folded_scale,
        )
    except ValueError as error:
        raise ValueError(f"{conv.name}: {error}") from error


def lower_magnet_alpha_fixed_point_epilogues(
    model: GraphModule,
) -> GraphModule:
    """Derive explicit Alpha PE operands from generic converted Q/DQ."""
    graph = model.graph
    lowered = 0
    replaced_quantizers: set[Node] = set()
    normalization_counts = {
        "direct_output": 0,
        "fixed_q8_8": 0,
        "fallback_q8_8": 0,
    }
    normalization_fallbacks = []

    for reference_input in list(graph.nodes):
        if reference_input.target == VSQ_CONV2D_OPERATOR:
            conv = reference_input
            if int(get_arg_value(conv, 6, "groups", 1)) != 1:
                continue
            folded_value = conv.kwargs.get("folded_coarse_scale")
            vsq_uint8_fine_scales = True
        elif reference_input.target == torch.ops.alpha_quant.dequantize.default:
            conv = get_arg_value(reference_input, 0, "input")
            if (
                not isinstance(conv, Node)
                or conv.target
                not in (
                    torch.ops.aten.conv2d.default,
                    torch.ops.quantized_ops.conv2d.default,
                )
                or int(get_arg_value(conv, 6, "groups", 1)) != 1
            ):
                continue
            folded_value = get_arg_value(reference_input, 1, "scale")
            vsq_uint8_fine_scales = False
        else:
            continue

        boundary = _converted_pe_boundary(model, reference_input)
        if boundary is None:
            continue
        relu = (
            boundary if boundary.target == torch.ops.aten.relu.default else None
        )
        quantizer_info = _output_quantizer_info(model, boundary)
        assert quantizer_info is not None
        (
            output_scale,
            output_precision,
            uses_vector_scale,
            fine_endpoint,
            coarse_endpoint,
        ) = quantizer_info
        alpha_vsq = (
            vsq_uint8_fine_scales
            and int(conv.kwargs.get("block_size", 0)) == ALPHA_VSQ_BLOCK_VALUES
        )
        exact_w8_o8 = (
            not vsq_uint8_fine_scales
            and not uses_vector_scale
            and output_precision == 8
        )
        exact_w8_o4 = (
            not vsq_uint8_fine_scales
            and uses_vector_scale
            and output_precision == 4
        )
        exact_vsq_o4 = alpha_vsq and uses_vector_scale and output_precision == 4
        exact_vsq_o8 = (
            alpha_vsq and not uses_vector_scale and output_precision == 8
        )
        if alpha_vsq and not exact_vsq_o4 and not exact_vsq_o8:
            raise ValueError(
                f"{boundary.name}: exact Alpha VSQ PE requires O4/PVS "
                "or scalar O8"
            )
        integer_accumulator = (
            exact_w8_o4 or exact_w8_o8 or exact_vsq_o4 or exact_vsq_o8
        )
        output = alpha_output_domain(
            output_scale,
            output_precision,
            uses_vector_scale,
            fine_quant_max=fine_endpoint,
            coarse_scale_quant_max=coarse_endpoint,
        )

        folded = _resolve_static_tensor(
            model,
            folded_value,
        )
        if folded is None:
            continue
        bias_node = get_arg_value(conv, 2, "bias")
        bias = _resolve_static_tensor(model, bias_node)
        if bias is not None and not vsq_uint8_fine_scales:
            # Generic integer conv stores bias in accumulator units.  Its
            # following dequant scale converts both the MAC result and bias
            # back to real values.  VSQ conv, by contrast, retains real bias.
            try:
                bias = reconstruct_accumulator_bias(bias, folded)
            except ValueError as error:
                raise ValueError(f"{boundary.name}: {error}") from error
        selected_normalization = select_alpha_pe_normalization(
            output,
            vsq_uint8_fine_scales=vsq_uint8_fine_scales,
            relu=relu is not None,
        )
        accumulator_real_amax = None
        if selected_normalization.direct_output:
            try:
                accumulator_real_amax = _static_w8_accumulator_real_bound(
                    model,
                    conv,
                    folded,
                )
            except ValueError as error:
                raise ValueError(f"{boundary.name}: {error}") from error
        pipeline, normalization = encode_selected_alpha_pe_pipeline(
            folded,
            bias,
            output,
            vsq_uint8_fine_scales=vsq_uint8_fine_scales,
            relu=relu is not None,
            accumulator_real_amax=accumulator_real_amax,
        )
        if normalization != selected_normalization and not (
            selected_normalization.direct_output
            and normalization.fallback_reason is not None
        ):
            raise AssertionError("Alpha PE normalization selection drifted")
        assert pipeline.secondary is not None
        bias_payload = pipeline.bias_relu.bias
        source_name = conv.meta.get("source_op_name", conv.name)
        if normalization.direct_output:
            normalization_kind = "direct_output"
        elif normalization.fallback_reason is not None:
            normalization_kind = "fallback_q8_8"
            normalization_fallbacks.append(
                {
                    "boundary": boundary.name,
                    "source": source_name,
                    "reason": normalization.fallback_reason,
                }
            )
        else:
            normalization_kind = "fixed_q8_8"
        normalization_counts[normalization_kind] += 1
        target_input = (
            _clone_bias_free_conv(graph, conv)
            if integer_accumulator
            else reference_input
        )
        if integer_accumulator:
            output_quantizers = _output_quantizers(model, boundary)
            if len(output_quantizers) != 1:
                raise ValueError(
                    f"{boundary.name}: exact Alpha PE requires one "
                    "output quantizer"
                )
            if exact_w8_o8 or exact_vsq_o8:
                replaced_quantizers.update(output_quantizers)
        boundary_users = tuple(boundary.users)
        instruction_kwargs = {
            "accumulator_bias": None,
            "output_scale": float(pipeline.output.scale[0].item()),
            "integer_accumulator": integer_accumulator,
            "semantic_step": normalization.semantic_step,
            "rounding": ALPHA_Q8_8_REFERENCE_FORMAT.rounding.value,
            "saturate": ALPHA_Q8_8_REFERENCE_FORMAT.saturate,
            "primary_shift": pipeline.primary.shift,
            "secondary_multiplier": int(pipeline.secondary.multiplier[0]),
            "secondary_shift": pipeline.secondary.shift,
            "output_precision": pipeline.output.precision,
            "relu": pipeline.bias_relu.relu,
        }
        with graph.inserting_before(boundary):
            primary_attr = create_getattr_from_value(
                model,
                graph,
                f"{source_name}_ppu_primary_multiplier",
                pipeline.primary.multiplier.to(torch.float32),
            )
            bias_attr = None
            if bias_payload is not None:
                bias_attr = create_getattr_from_value(
                    model,
                    graph,
                    f"{source_name}_ppu_bias",
                    bias_payload.to(torch.float32),
                )
        insertion_context = (
            graph.inserting_after(reference_input)
            if (not integer_accumulator and boundary is reference_input)
            else graph.inserting_before(boundary)
        )
        with insertion_context:
            target_node = graph.call_function(
                PPU_FIXED_RELU_OPERATOR,
                (target_input, primary_attr, bias_attr),
                instruction_kwargs,
            )

        for attribute in (primary_attr, bias_attr):
            if attribute is None:
                continue
            propagate_shape(attribute, model)
            attribute.meta["magnet_alpha_instruction_operand"] = True
        target_node.meta = copy.copy(boundary.meta)
        if exact_w8_o8 or exact_vsq_o8:
            output_quantizer = output_quantizers[0]
            for key in (
                QUANTIZATION_REGION,
                _QUANTIZATION_DOMAIN_ENTRY,
                _QUANTIZATION_DOMAIN_ALIAS,
            ):
                if key in output_quantizer.meta:
                    target_node.meta[key] = copy.copy(
                        output_quantizer.meta[key]
                    )
        target_node.meta["magnet_alpha_pe_normalization"] = normalization_kind
        if normalization.fallback_reason is not None:
            target_node.meta["magnet_alpha_pe_normalization_fallback"] = (
                normalization.fallback_reason
            )
        if exact_w8_o8 or exact_vsq_o8:
            target_node.meta["dtype"] = "int8"
            set_physical_quantization_contract(
                target_node,
                PhysicalQuantizationContract("int8"),
            )
        for user in boundary_users:
            user.replace_input_with(boundary, target_node)
        if relu is not None:
            graph.erase_node(relu)
        lowered += 1

    for quantizer in replaced_quantizers:
        source = get_arg_value(quantizer, 0, "input")
        if not isinstance(source, Node):
            raise ValueError(
                f"{quantizer.name}: Alpha output quantizer has no input"
            )
        quantizer.replace_all_uses_with(source)
        graph.erase_node(quantizer)

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    model.delete_all_unused_submodules()
    model.meta["magnet_alpha_pe_normalization_summary"] = {
        "counts": normalization_counts,
        "fallbacks": tuple(normalization_fallbacks),
    }
    logger.info(
        "Lowered %d Magnet Alpha PE fixed-point epilogues with "
        "normalizations %s",
        lowered,
        normalization_counts,
    )
    return model


def pad_matrix_output_operands(
    model: GraphModule,
    node: Node,
    output_channels: int,
    output_padding: int,
) -> None:
    """Zero-extend Alpha PPU operands for software-resident K padding."""
    semantic_weight_scale = node.kwargs.get("semantic_weight_coarse_scale")
    if (
        isinstance(semantic_weight_scale, Node)
        and semantic_weight_scale.op == "get_attr"
    ):
        value = fetch_attr(model, semantic_weight_scale.target)
        expected_channels = output_channels + output_padding
        if value.numel() == 1:
            pass
        elif value.shape[0] == output_channels:
            padding = [0, 0] * (value.ndim - 1)
            padding.extend((0, output_padding))
            value.data = torch.nn.functional.pad(
                value,
                padding,
                value=0.0,
            )
            propagate_shape(semantic_weight_scale, model)
        elif value.shape[0] != expected_channels:
            raise ValueError(
                "Semantic weight coarse scale "
                f"{semantic_weight_scale.target} has leading dimension "
                f"{value.shape[0]}, expected {output_channels}"
            )

    queue = list(node.users)
    visited = set()
    ppu_nodes = []
    while queue:
        user = queue.pop()
        if user in visited:
            continue
        visited.add(user)
        if user.target == PPU_FIXED_RELU_OPERATOR:
            ppu_nodes.append(user)
        elif user.target == torch.ops.alpha_quant.dequantize.default:
            queue.extend(user.users)

    padded_targets = set()
    for ppu in ppu_nodes:
        for operand in ppu.args[1:3]:
            if (
                not isinstance(operand, Node)
                or operand.op != "get_attr"
                or operand.target in padded_targets
            ):
                continue
            value = fetch_attr(model, operand.target)
            if value.numel() == 1:
                continue
            if value.numel() == output_channels + output_padding:
                continue
            if value.numel() != output_channels:
                raise ValueError(
                    f"PPU operand {operand.target} contains "
                    f"{value.numel()} values, expected {output_channels}"
                )
            value.data = torch.nn.functional.pad(
                value.reshape(-1),
                (0, output_padding),
                value=0.0,
            )
            propagate_shape(operand, model)
            padded_targets.add(operand.target)


def _is_physical_vsq_scale(
    node: Node,
    visited: set[Node] | None = None,
) -> bool:
    """Recognize physical Alpha mantissas through legal data movement."""
    if visited is None:
        visited = set()
    if node in visited:
        return False
    visited.add(node)
    if node.meta.get("magnet_alpha_vsq_physical_scale"):
        return True
    if node.target == VSQ_SCALE_ENCODE_OPERATOR:
        return True
    if (
        node.target == operator.getitem
        and int(node.args[1]) == 1
        and isinstance(node.args[0], Node)
        and node.args[0].target == PVS_QUANTIZE_OPERATOR
    ):
        return True
    if node.target == torch.ops.aten.cat.default:
        inputs = tuple(node.args[0])
        return bool(inputs) and all(
            _is_physical_vsq_scale(value, visited.copy()) for value in inputs
        )
    if node.target == torch.ops.aten.stack.default:
        inputs = tuple(get_arg_value(node, 0, "tensors"))
        return bool(inputs) and all(
            isinstance(value, Node)
            and _is_physical_vsq_scale(value, visited.copy())
            for value in inputs
        )
    source = get_identity_slice_source(node)
    if source is None:
        source = get_leading_channel_slice_source(node)
    if source is not None:
        return _is_physical_vsq_scale(source, visited)
    if node.target in {
        torch.ops.aten.clone.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.permute.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.upsample_nearest2d.vec,
        torch.ops.aten.view.default,
    }:
        source = get_arg_value(node, 0, "input")
        return isinstance(source, Node) and _is_physical_vsq_scale(
            source,
            visited,
        )
    return False


def finalize_magnet_alpha_vsq_operands(model: GraphModule) -> GraphModule:
    """Materialize each physical Alpha fine-scale operand exactly once."""
    graph = model.graph
    encoded_by_source: dict[tuple[Node, float, float], Node] = {}
    encoded_count = 0

    for conv in list(graph.nodes):
        if conv.target != VSQ_CONV2D_OPERATOR:
            continue
        kwargs = dict(conv.kwargs)
        source_name = conv.meta.get("source_op_name", conv.name)
        for role in ("input_scale", "weight_scale"):
            scale = kwargs.get(role)
            if not isinstance(scale, Node):
                raise ValueError(
                    f"{conv.name}: Alpha VSQ {role} must be one tensor node"
                )
            if _is_physical_vsq_scale(scale):
                scale.meta["magnet_alpha_vsq_physical_scale"] = True
                continue
            fine_endpoint, coarse_endpoint = _vsq_scale_endpoints(scale)
            encoding_key = (scale, fine_endpoint, coarse_endpoint)
            encoded = encoded_by_source.get(encoding_key)
            if encoded is None:
                static_value = _resolve_static_tensor(model, scale)
                with graph.inserting_before(conv):
                    if static_value is not None:
                        encoded_value = VSQ_SCALE_ENCODE_OPERATOR(
                            static_value,
                            fine_endpoint,
                            coarse_endpoint,
                        )
                        encoded = create_getattr_from_value(
                            model,
                            graph,
                            f"{source_name}_{role}_alpha_mantissa",
                            encoded_value,
                        )
                    else:
                        encoded = graph.call_function(
                            VSQ_SCALE_ENCODE_OPERATOR,
                            (
                                scale,
                                fine_endpoint,
                                coarse_endpoint,
                            ),
                        )
                encoded.meta = copy.copy(scale.meta)
                encoded.meta["dtype"] = "uint8"
                encoded.meta["magnet_alpha_vsq_physical_scale"] = True
                if (
                    static_value is not None
                    or getattr(scale, "value", None) is not None
                ):
                    propagate_shape(encoded, model)
                scale.meta["magnet_alpha_semantic_operand"] = True
                encoded_by_source[encoding_key] = encoded
                encoded_count += 1
            kwargs[role] = encoded
        conv.kwargs = kwargs

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    logger.info(
        "Materialized %d Magnet Alpha physical VSQ scale operands",
        encoded_count,
    )
    return model


def _ppu_relu_commutes_with_secondary(ppu: Node) -> bool:
    """Prove ReLU commutes with Alpha's monotone secondary scale stage."""
    multiplier = ppu.kwargs.get("secondary_multiplier")
    shift = ppu.kwargs.get("secondary_shift")
    return not (
        type(multiplier) is not int
        or not 1 <= multiplier <= 65535
        or type(shift) is not int
        or not 0 <= shift <= 255
    )


def _verified_physical_int8_relu_value(
    ppu: Node,
    relu: Node,
) -> bool:
    """Prove the exact PPU-with-ReLU rewrite, or fail closed."""
    if (
        ppu.op != "call_function"
        or ppu.target != PPU_FIXED_RELU_OPERATOR
        or tuple(ppu.users) != (relu,)
        or relu.op != "call_function"
        or relu.target != torch.ops.aten.relu.default
        or get_arg_value(relu, 0, "self") is not ppu
        or len(relu.all_input_nodes) != 1
        or ppu.kwargs.get("relu") is not False
        or ppu.kwargs.get("integer_accumulator") is not True
        or type(ppu.kwargs.get("output_precision")) is not int
        or ppu.kwargs["output_precision"] != 8
        or ppu.kwargs.get("saturate") is not True
        or relu_reaches_alpha_eadd(relu)
        or not _ppu_relu_commutes_with_secondary(ppu)
    ):
        return False

    int8_contract = PhysicalQuantizationContract("int8")
    ppu_contract = get_physical_quantization_contract(ppu)
    relu_contract = get_physical_quantization_contract(relu)
    ppu_region = ppu.meta.get(QUANTIZATION_REGION)
    relu_region = relu.meta.get(QUANTIZATION_REGION)
    if (
        ppu_contract not in (None, int8_contract)
        or relu_contract != int8_contract
        or get_physical_dtype(ppu) != "int8"
        or get_physical_dtype(relu) != "int8"
        or relu_region is None
        or (ppu_region is not None and ppu_region != relu_region)
    ):
        return False

    ppu_shape = node_shape(ppu)
    ppu_input = get_arg_value(ppu, 0, "input")
    if not ppu_shape and isinstance(ppu_input, Node):
        ppu_shape = node_shape(ppu_input)
    relu_shape = node_shape(relu)
    if not ppu_shape or ppu_shape != relu_shape:
        return False
    return True


def _refresh_ppu_output_facts(
    ppu: Node,
    relu: Node,
) -> None:
    """Retain PPU provenance while adopting the verified ReLU output."""
    output_fact_keys = (
        "dtype",
        "tensor_meta",
        "val",
        PHYSICAL_QUANTIZATION_CONTRACT,
        QUANTIZATION_REGION,
    )
    for key in output_fact_keys:
        if key in relu.meta:
            ppu.meta[key] = copy.copy(relu.meta[key])
        elif key in ("tensor_meta", "val"):
            ppu.meta.pop(key, None)
    if hasattr(relu, "shape"):
        ppu.shape = relu.shape
    if hasattr(relu, "value"):
        ppu.value = copy.copy(relu.value)
    elif hasattr(ppu, "value"):
        del ppu.value
    ppu.type = relu.type


def fold_magnet_alpha_physical_int8_relus(
    model: GraphModule,
) -> GraphModule:
    """Fold only exact sole-user O8 PPU-to-ReLU boundaries."""
    graph = model.graph
    folded = 0
    for ppu in list(graph.nodes):
        users = tuple(ppu.users)
        if len(users) != 1:
            continue
        relu = users[0]
        if not _verified_physical_int8_relu_value(ppu, relu):
            continue

        kwargs = dict(ppu.kwargs)
        kwargs["relu"] = True
        ppu.kwargs = kwargs
        _refresh_ppu_output_facts(ppu, relu)
        relu.replace_all_uses_with(ppu)
        graph.erase_node(relu)
        folded += 1

    if folded:
        graph.lint()
        model.recompile()
    logger.info(
        "Folded %d exact physical INT8 PPU-to-ReLU boundaries",
        folded,
    )
    return model


__all__ = [
    "finalize_magnet_alpha_vsq_operands",
    "fold_magnet_alpha_physical_int8_relus",
    "lower_magnet_alpha_fixed_point_epilogues",
    "lower_magnet_alpha_vsq_convolutions",
    "pad_matrix_output_operands",
]

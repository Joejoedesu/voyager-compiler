# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Magnet Alpha NMP fixed-point target lowering."""

import copy
import logging

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.ir.graph import (
    PHYSICAL_QUANTIZATION_CONTRACT,
    QUANTIZATION_REGION,
    PhysicalQuantizationContract,
    set_physical_quantization_contract,
)
from voyager_compiler.targets.magnet_alpha.fixed_point import (
    alpha_output_domain,
    encode_alpha_nmp_pipeline,
)
from voyager_compiler.targets.magnet_alpha.fx import (
    create_getattr_from_value,
    get_arg_value,
    propagate_shape,
)
from voyager_compiler.targets.magnet_alpha.lowering_utils import (
    clone_bias_free_conv,
    node_shape,
    output_quantizer_info,
    output_quantizers,
    resolve_static_tensor,
)
from voyager_compiler.targets.magnet_alpha.operators import (
    INT8_CONV2D_ACCUMULATOR_OPERATOR,
    NMP_FIXED_REQUANTIZE_OPERATOR,
)

logger = logging.getLogger(__name__)


def materialize_magnet_alpha_depthwise_requantize(
    model: GraphModule,
    anchor: Node,
    source: Node,
    *,
    weight_name: str,
    weight_value: torch.Tensor,
    accumulator_scale_name: str,
    accumulator_scale_value: float,
    scale_factor: int,
    binary_point: int,
    output_scale: float,
    output_precision: int,
    relu: bool,
    output_meta: dict,
    bias_factor: int = 0,
) -> Node:
    """Materialize one proved Alpha depthwise accumulator/NMP instruction."""
    if weight_value.ndim != 4 or int(weight_value.shape[1]) != 1:
        raise ValueError(
            f"{anchor.name}: Alpha DWC weight must have shape Kx1xRxS"
        )
    if output_precision not in (4, 8):
        raise ValueError(f"{anchor.name}: Alpha DWC output must be O4 or O8")
    channels = int(weight_value.shape[0])
    graph = model.graph
    with graph.inserting_before(anchor):
        weight = create_getattr_from_value(
            model,
            graph,
            weight_name,
            weight_value,
        )
        semantic_scale = create_getattr_from_value(
            model,
            graph,
            accumulator_scale_name,
            torch.tensor([accumulator_scale_value], dtype=torch.float32),
        )
        accumulator = graph.call_function(
            INT8_CONV2D_ACCUMULATOR_OPERATOR,
            (source, weight, None, (1, 1), (0, 0), (1, 1), channels),
        )
        output = graph.call_function(
            NMP_FIXED_REQUANTIZE_OPERATOR,
            (accumulator, None, semantic_scale),
            {
                "scale_factor": scale_factor,
                "binary_point": binary_point,
                "bias_factor": bias_factor,
                "output_scale": output_scale,
                "output_precision": output_precision,
                "relu": relu,
            },
        )

    propagate_shape(weight, model)
    weight.meta["dtype"] = "int8"
    propagate_shape(semantic_scale, model)
    semantic_scale.meta["magnet_alpha_semantic_operand"] = True
    accumulator.meta = copy.copy(output_meta)
    accumulator.meta.pop(PHYSICAL_QUANTIZATION_CONTRACT, None)
    accumulator.meta.pop(QUANTIZATION_REGION, None)
    accumulator.meta["dtype"] = "int32"
    output.meta = copy.copy(output_meta)
    if output_precision == 8:
        output.meta["dtype"] = "int8"
        set_physical_quantization_contract(
            output,
            PhysicalQuantizationContract("int8"),
        )
    else:
        output.meta.pop(PHYSICAL_QUANTIZATION_CONTRACT, None)
        output.meta.pop(QUANTIZATION_REGION, None)
    return output


def lower_magnet_alpha_depthwise_fixed_point(
    model: GraphModule,
) -> GraphModule:
    """Lower W8 depthwise output scaling into explicit Alpha NMP operands."""
    graph = model.graph
    lowered = 0

    for dequantize in list(graph.nodes):
        if dequantize.target != torch.ops.alpha_quant.dequantize.default:
            continue
        conv = get_arg_value(dequantize, 0, "input")
        if not isinstance(conv, Node) or conv.target not in (
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_ops.conv2d.default,
        ):
            continue
        groups = int(get_arg_value(conv, 6, "groups", 1))
        weight = get_arg_value(conv, 1, "weight")
        weight_shape = node_shape(weight)
        if (
            groups <= 1
            or len(weight_shape) != 4
            or int(weight_shape[1]) != 1
            or int(weight_shape[0]) != groups
        ):
            continue

        relu = next(
            (
                user
                for user in dequantize.users
                if user.target == torch.ops.aten.relu.default
            ),
            None,
        )
        boundary = relu if relu is not None else dequantize
        quantizer_info = output_quantizer_info(model, boundary)
        if quantizer_info is None:
            raise ValueError(
                f"{boundary.name}: Alpha DWC requires a static O4/O8 output "
                "quantizer"
            )
        (
            output_scale,
            output_precision,
            uses_vector_scale,
            fine_endpoint,
            coarse_endpoint,
        ) = quantizer_info
        output = alpha_output_domain(
            output_scale,
            output_precision,
            uses_vector_scale,
            fine_quant_max=fine_endpoint,
            coarse_scale_quant_max=coarse_endpoint,
        )
        accumulator_scale_node = get_arg_value(dequantize, 1, "scale")
        accumulator_scale = resolve_static_tensor(
            model,
            accumulator_scale_node,
        )
        if accumulator_scale is None:
            raise ValueError(
                f"{boundary.name}: Alpha DWC requires a static accumulator "
                "scale"
            )

        accumulator_bias_node = get_arg_value(conv, 2, "bias")
        accumulator_bias = resolve_static_tensor(
            model,
            accumulator_bias_node,
        )
        try:
            pipeline = encode_alpha_nmp_pipeline(
                accumulator_scale,
                accumulator_bias,
                output,
                relu=relu is not None,
            )
        except ValueError as error:
            raise ValueError(f"{boundary.name}: {error}") from error
        assert pipeline.bias_relu.bias is not None
        bias_factor = int(pipeline.bias_relu.bias.reshape(-1)[0].item())
        output_quantizer_nodes = output_quantizers(model, boundary)
        if not output_quantizer_nodes or any(
            get_arg_value(quantizer, 0, "input") is not boundary
            for quantizer in output_quantizer_nodes
        ):
            raise ValueError(
                f"{boundary.name}: Alpha DWC output quantizer must directly "
                "follow its activation boundary"
            )

        instruction_kwargs = {
            "scale_factor": int(pipeline.primary.multiplier[0]),
            "binary_point": pipeline.primary.shift,
            "bias_factor": bias_factor,
            "output_precision": pipeline.output.precision,
            "relu": pipeline.bias_relu.relu,
        }
        raw_conv = clone_bias_free_conv(graph, conv)
        with graph.inserting_before(boundary):
            target_node = graph.call_function(
                NMP_FIXED_REQUANTIZE_OPERATOR,
                (
                    raw_conv,
                    # The scalar hardware bias was already encoded above.
                    # Do not retain the source per-channel tensor as a
                    # physical operand of the target instruction.
                    None,
                    accumulator_scale_node,
                ),
                {
                    **instruction_kwargs,
                    "output_scale": float(pipeline.output.scale[0].item()),
                },
            )

        target_node.meta = copy.copy(boundary.meta)
        accumulator_scale_node.meta["magnet_alpha_semantic_operand"] = True
        if pipeline.output.precision == 8:
            target_node.meta["dtype"] = "int8"
            set_physical_quantization_contract(
                target_node,
                PhysicalQuantizationContract("int8"),
            )
            for quantizer in output_quantizer_nodes:
                quantizer.replace_all_uses_with(target_node)
                graph.erase_node(quantizer)
        else:
            for quantizer in output_quantizer_nodes:
                quantizer.replace_input_with(boundary, target_node)
        if not boundary.users:
            graph.erase_node(boundary)
        if relu is not None and not dequantize.users:
            graph.erase_node(dequantize)
        lowered += 1

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    logger.info("Lowered %d Magnet Alpha DWC fixed-point epilogues", lowered)
    return model


__all__ = [
    "lower_magnet_alpha_depthwise_fixed_point",
    "materialize_magnet_alpha_depthwise_requantize",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Magnet Alpha residual INPUT_JOIN and EADD lowering."""

import copy
import logging
from dataclasses import dataclass

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.passes.utils import get_arg_value
from voyager_compiler.codegen.target_backends.magnet_alpha.fixed_point import (
    prove_exact_identity_dwc_encoding,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.lowering_utils import (
    RESIDUAL_ADD_TARGETS,
    RESIDUAL_RELU_TARGETS,
    direct_residual_output_scale,
    node_shape,
    output_quantizers,
    residual_dequantized_path,
    residual_quantizer_scale,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.nmp_lowering import (
    materialize_magnet_alpha_depthwise_requantize,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.operators import (
    NMP_FIXED_REQUANTIZE_OPERATOR,
    PPU_FIXED_RELU_OPERATOR,
    SATURATING_ADD_OPERATOR,
)
from voyager_compiler.quantization_contracts import (
    PHYSICAL_QUANTIZATION_CONTRACT,
    QUANTIZATION_DOMAIN_DTYPE,
    QUANTIZATION_INPUT_REGION,
    QUANTIZATION_REGION,
    PhysicalQuantizationContract,
    has_physical_payload,
    set_physical_quantization_contract,
)
from voyager_compiler.quantization_regions import (
    _QUANTIZATION_DOMAIN_ALIAS,
    _QUANTIZATION_DOMAIN_ENTRY,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResidualOperandPlan:
    """One residual operand represented as signed-INT8 payload codes."""

    source: Node
    source_scale: float
    rescale_output_scales: tuple[float, ...]
    is_direct_pe_output: bool


@dataclass(frozen=True)
class ResidualOutputPlan:
    """One physical O8 result plus its semantic output boundaries."""

    scalar_quantizers: tuple[Node, ...]
    pvs_quantizers: tuple[Node, ...]
    output_scale: float
    scale_operand: Node


def _plan_residual_output(
    model: GraphModule,
    relu: Node,
) -> ResidualOutputPlan:
    quantizers = output_quantizers(model, relu)
    scalar_quantizers = tuple(
        node
        for node in quantizers
        if node.target == torch.ops.quantized_ops.quantize.default
    )
    pvs_quantizers = tuple(
        node
        for node in quantizers
        if node.target == torch.ops.quantized_ops.quantize_vsq.default
    )
    if len(scalar_quantizers) + len(pvs_quantizers) != len(quantizers):
        raise ValueError(
            f"{relu.name}: Alpha residual ReLU has an unknown output "
            "quantizer"
        )
    if not scalar_quantizers:
        raise ValueError(
            f"{relu.name}: Alpha residual ReLU has no scalar INT8 "
            "output domain"
        )

    output_scales = tuple(
        residual_quantizer_scale(model, quantizer)
        for quantizer in scalar_quantizers
    )
    output_scale = output_scales[0]
    if any(scale != output_scale for scale in output_scales[1:]):
        raise ValueError(
            f"{relu.name}: Alpha residual ReLU scalar output qparams are "
            "unequal: " + ", ".join(str(scale) for scale in output_scales)
        )
    for quantizer in pvs_quantizers:
        if get_arg_value(quantizer, 0, "self") is not relu:
            raise ValueError(
                f"{quantizer.name}: Alpha residual PVS boundary must "
                f"directly consume {relu.name}"
            )

    scale_operand = get_arg_value(
        scalar_quantizers[0],
        1,
        "scale",
    )
    if not isinstance(scale_operand, Node):
        raise ValueError(
            f"{scalar_quantizers[0].name}: Alpha residual output scale "
            "must be one explicit FX value"
        )
    return ResidualOutputPlan(
        scalar_quantizers=scalar_quantizers,
        pvs_quantizers=pvs_quantizers,
        output_scale=output_scale,
        scale_operand=scale_operand,
    )


def _materialize_pvs_dequantize(
    graph,
    relu: Node,
    post_relu: Node,
    quantizer: Node,
    scale_operand: Node,
) -> Node:
    """Restore the semantic O8-to-PVS boundary consumed by Alpha."""
    if get_arg_value(quantizer, 0, "self") is not post_relu:
        raise AssertionError("residual PVS input replacement drifted")
    with graph.inserting_before(quantizer):
        dequantize = graph.call_function(
            torch.ops.quantized_ops.dequantize.default,
            (post_relu, scale_operand),
        )
    dequantize.meta = copy.copy(relu.meta)
    for key in (
        "dtype",
        PHYSICAL_QUANTIZATION_CONTRACT,
        QUANTIZATION_DOMAIN_DTYPE,
    ):
        dequantize.meta.pop(key, None)
    quantizer.replace_input_with(post_relu, dequantize)
    return dequantize


def _plan_residual_operand(
    model: GraphModule,
    node: Node,
    output_scale: float,
) -> ResidualOperandPlan:
    if node.target == PPU_FIXED_RELU_OPERATOR:
        source_scale = direct_residual_output_scale(node)
        assert source_scale is not None
        if (
            not has_physical_payload(node, "int8")
            or not bool(node.kwargs.get("integer_accumulator", False))
            or int(node.kwargs.get("output_precision", 0)) != 8
            or bool(node.kwargs.get("relu", True))
            or source_scale != output_scale
        ):
            raise ValueError(
                f"{node.name}: residual PE must emit non-ReLU O8 directly "
                "in the EADD domain"
            )
        return ResidualOperandPlan(node, source_scale, (), True)

    if node.target == NMP_FIXED_REQUANTIZE_OPERATOR:
        source_scale = direct_residual_output_scale(node)
        assert source_scale is not None
        if (
            not has_physical_payload(node, "int8")
            or int(node.kwargs.get("output_precision", 0)) != 8
            or source_scale != output_scale
        ):
            raise ValueError(
                f"{node.name}: residual NMP output does not match the EADD "
                "signed-INT8 domain"
            )
        return ResidualOperandPlan(node, source_scale, (), False)

    if node.target == torch.ops.quantized_ops.quantize.default:
        quantized_scale = residual_quantizer_scale(model, node)
        if quantized_scale != output_scale:
            raise ValueError(
                f"{node.name}: residual quantizer scale {quantized_scale} "
                f"does not equal EADD scale {output_scale}"
            )
        dequantize = get_arg_value(node, 0, "input")
        if (
            not isinstance(dequantize, Node)
            or dequantize.target != torch.ops.quantized_ops.dequantize.default
        ):
            raise ValueError(
                f"{node.name}: residual rescale must be an explicit "
                "Q(DQ(code)) chain"
            )
        path, source_scale = residual_dequantized_path(model, dequantize)
        output_scales = path.rescale_output_scales
        if source_scale != output_scale:
            output_scales = (*output_scales, output_scale)
        return ResidualOperandPlan(
            path.source,
            path.source_scale,
            output_scales,
            path.source.target == PPU_FIXED_RELU_OPERATOR and not output_scales,
        )

    if node.target == torch.ops.quantized_ops.dequantize.default:
        path, source_scale = residual_dequantized_path(model, node)
        output_scales = path.rescale_output_scales
        if source_scale != output_scale:
            output_scales = (*output_scales, output_scale)
        return ResidualOperandPlan(
            path.source,
            path.source_scale,
            output_scales,
            path.source.target == PPU_FIXED_RELU_OPERATOR and not output_scales,
        )

    raise ValueError(
        f"{node.name}: Alpha residual operand is not a direct PE/NMP code "
        "or an explicit scalar Q/DQ rescale"
    )


def _validate_eadd_destination_lifetime(
    graph,
    destination: Node,
    add: Node,
) -> None:
    """Require EADD's mutable operand2 to be dead after the operation."""
    positions = {node: index for index, node in enumerate(graph.nodes)}
    add_position = positions[add]
    late_users = tuple(
        user
        for user in destination.users
        if user is not add and positions[user] > add_position
    )
    if late_users:
        raise ValueError(
            f"{add.name}: Alpha EADD overwrites operand2, but "
            f"{destination.name} has post-add users: "
            + ", ".join(user.name for user in late_users)
        )


def materialize_identity_dwc(
    model: GraphModule,
    anchor: Node,
    source: Node,
    source_scale: float,
    output_scale: float,
    *,
    relu: bool,
    prefix: str,
    output_meta: dict,
    proved_encoding: tuple[int, int] | None = None,
    proved_shape: tuple[int, ...] | None = None,
    output_precision: int = 8,
) -> Node:
    """Emit one explicit Alpha NMP 1x1 depthwise identity call."""
    if output_precision not in (4, 8):
        raise ValueError(
            f"{anchor.name}: Alpha identity DWC output must be O4 or O8"
        )
    shape = node_shape(source) or node_shape(anchor)
    if proved_shape is not None:
        if shape and shape != proved_shape:
            raise ValueError(
                f"{anchor.name}: proved Alpha identity DWC shape "
                f"{proved_shape} conflicts with graph shape {shape}"
            )
        shape = proved_shape
    if len(shape) != 4 or shape[0] != 1 or shape[1] <= 1:
        raise ValueError(
            f"{anchor.name}: Alpha identity DWC requires static "
            "NCHW shape with N=1 and C>1"
        )
    channels = shape[1]
    if proved_encoding is None:
        scale_factor, binary_point = prove_exact_identity_dwc_encoding(
            source_scale,
            output_scale,
            relu=relu,
            context=anchor.name,
        )
    else:
        scale_factor, binary_point = proved_encoding
        if (
            type(scale_factor) is not int
            or not 1 <= scale_factor <= 65535
            or type(binary_point) is not int
            or not 0 <= binary_point <= 255
        ):
            raise ValueError(
                f"{anchor.name}: proved Alpha NMP encoding is not legal"
            )
    return materialize_magnet_alpha_depthwise_requantize(
        model,
        anchor,
        source,
        weight_name=f"{prefix}_identity_dwc_weight",
        weight_value=torch.ones(
            (channels, 1, 1, 1),
            dtype=torch.float32,
        ),
        accumulator_scale_name=(f"{prefix}_identity_dwc_accumulator_scale"),
        accumulator_scale_value=source_scale,
        scale_factor=scale_factor,
        binary_point=binary_point,
        output_scale=output_scale,
        output_precision=output_precision,
        relu=relu,
        output_meta=output_meta,
    )


def lower_magnet_alpha_saturating_adds(
    model: GraphModule,
) -> GraphModule:
    """Normalize proved INT8 residual codes, then emit EADD and NMP ReLU."""
    graph = model.graph
    lowered = 0
    for add in list(graph.nodes):
        if add.target not in RESIDUAL_ADD_TARGETS:
            continue
        input1 = get_arg_value(add, 0, "input")
        input2 = get_arg_value(add, 1, "other")
        alpha = get_arg_value(add, 2, "alpha", 1)
        input_join_region = add.meta.get(QUANTIZATION_INPUT_REGION)
        has_input_join = (
            type(input_join_region) is int and input_join_region >= 0
        )
        if not all(isinstance(value, Node) for value in (input1, input2)):
            continue
        if not has_input_join:
            continue
        add_users = tuple(add.users)
        if (
            len(add_users) != 1
            or add_users[0].target not in RESIDUAL_RELU_TARGETS
        ):
            continue
        relu = add_users[0]
        if alpha != 1:
            raise ValueError(
                f"{add.name}: Alpha residual input join requires alpha=1"
            )
        output_shape = node_shape(add)
        input_shapes = (node_shape(input1), node_shape(input2))
        if (
            len(output_shape) != 4
            or output_shape[0] != 1
            or output_shape[1] <= 1
            or any(extent <= 0 for extent in output_shape)
            or any(shape and shape != output_shape for shape in input_shapes)
        ):
            raise ValueError(
                f"{add.name}: Alpha EADD requires equal static NCHW inputs "
                "with N=1 and C>1"
            )

        input_scales = []
        for operand in (input1, input2):
            if operand.target == torch.ops.quantized_ops.dequantize.default:
                _, scale = residual_dequantized_path(model, operand)
            elif operand.target == torch.ops.quantized_ops.quantize.default:
                scale = residual_quantizer_scale(model, operand)
            else:
                scale = direct_residual_output_scale(operand)
                if scale is None:
                    raise ValueError(
                        f"{operand.name}: Alpha input join requires a "
                        "direct PE/NMP code or explicit scalar INT8 "
                        "Q/DQ boundary"
                    )
            input_scales.append(scale)
        if input_scales[0] != input_scales[1]:
            raise ValueError(
                f"{add.name}: Alpha EADD input-join qparams are unequal: "
                f"{input_scales[0]}, {input_scales[1]}"
            )
        input_scale = input_scales[0]
        output = _plan_residual_output(model, relu)
        output_scale = output.output_scale

        operands = (
            _plan_residual_operand(model, input1, input_scale),
            _plan_residual_operand(model, input2, input_scale),
        )
        pe_indices = tuple(
            index
            for index, operand in enumerate(operands)
            if operand.is_direct_pe_output
        )
        if not pe_indices:
            raise ValueError(
                f"{add.name}: Alpha EADD must consume one direct PE output; "
                "standalone EADD is unsupported"
            )
        if pe_indices[0] == 1:
            operands = (operands[1], operands[0])
        if not operands[1].rescale_output_scales:
            _validate_eadd_destination_lifetime(
                graph,
                operands[1].source,
                add,
            )

        for index, operand in enumerate(operands):
            current_scale = operand.source_scale
            for stage, stage_output_scale in enumerate(
                operand.rescale_output_scales
            ):
                prove_exact_identity_dwc_encoding(
                    current_scale,
                    stage_output_scale,
                    relu=False,
                    context=(
                        f"{add.name} operand {index + 1} rescale "
                        f"stage {stage + 1}"
                    ),
                )
                current_scale = stage_output_scale
            if current_scale != input_scale:
                raise ValueError(
                    f"{add.name}: normalized operand {index + 1} scale "
                    f"{current_scale} does not equal EADD scale {input_scale}"
                )
        prove_exact_identity_dwc_encoding(
            input_scale,
            output_scale,
            relu=True,
            context=f"{add.name} post-EADD ReLU",
        )
        normalized = []
        for index, operand in enumerate(operands):
            source = operand.source
            current_scale = operand.source_scale
            for stage, stage_output_scale in enumerate(
                operand.rescale_output_scales
            ):
                source = materialize_identity_dwc(
                    model,
                    add,
                    source,
                    current_scale,
                    stage_output_scale,
                    relu=False,
                    prefix=(
                        f"{add.name}_operand_{index + 1}_stage_{stage + 1}"
                    ),
                    output_meta=add.meta,
                )
                current_scale = stage_output_scale
            normalized.append(source)

        with graph.inserting_before(add):
            eadd = graph.call_function(
                SATURATING_ADD_OPERATOR,
                tuple(normalized),
            )
        eadd.meta = copy.copy(add.meta)
        eadd.meta["dtype"] = "int8"
        set_physical_quantization_contract(
            eadd,
            PhysicalQuantizationContract("int8"),
        )

        post_relu_meta = copy.copy(relu.meta)
        for key in (
            QUANTIZATION_REGION,
            _QUANTIZATION_DOMAIN_ENTRY,
            _QUANTIZATION_DOMAIN_ALIAS,
        ):
            if key in output.scalar_quantizers[0].meta:
                post_relu_meta[key] = copy.copy(
                    output.scalar_quantizers[0].meta[key]
                )
        post_relu = materialize_identity_dwc(
            model,
            relu,
            eadd,
            input_scale,
            output_scale,
            relu=True,
            prefix=f"{add.name}_post_eadd_relu",
            output_meta=post_relu_meta,
        )
        relu.replace_all_uses_with(post_relu)
        for quantizer in output.pvs_quantizers:
            _materialize_pvs_dequantize(
                graph,
                relu,
                post_relu,
                quantizer,
                output.scale_operand,
            )
        for quantizer in output.scalar_quantizers:
            if get_arg_value(quantizer, 0, "input") is post_relu:
                quantizer.replace_all_uses_with(post_relu)
                graph.erase_node(quantizer)
        graph.erase_node(relu)
        graph.erase_node(add)
        lowered += 1

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    logger.info(
        "Lowered %d normalized Magnet Alpha EADD/ReLU residuals",
        lowered,
    )
    return model


__all__ = [
    "lower_magnet_alpha_saturating_adds",
    "materialize_identity_dwc",
]

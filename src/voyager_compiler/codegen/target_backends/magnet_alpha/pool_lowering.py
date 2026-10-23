# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Magnet Alpha adaptive-pooling target lowering."""

import logging
import math
from dataclasses import dataclass

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.passes.utils import get_arg_value
from voyager_compiler.codegen.storage_predicates import is_address_only_view
from voyager_compiler.codegen.target_backends.magnet_alpha.fixed_point import (
    prove_exact_adaptive_average_dwc_encoding,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.lowering_utils import (
    direct_residual_output_scale,
    node_shape,
    residual_dequantized_code,
    residual_quantizer_scale,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.nmp_lowering import (
    materialize_magnet_alpha_depthwise_requantize,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.operators import (
    NMP_FIXED_REQUANTIZE_OPERATOR,
)
from voyager_compiler.quantization_contracts import (
    has_physical_payload,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptiveAveragePlan:
    """One proved nonnegative INT8 global-average target rewrite."""

    pool: Node
    source: Node
    source_scale: float
    output_quantizer: Node
    output_scale: float
    channels: int
    kernel_size: tuple[int, int]
    scale_factor: int
    binary_point: int


def _adaptive_average_output_quantizer(pool: Node) -> Node | None:
    current = pool
    visited = set()
    while current not in visited and len(current.users) == 1:
        visited.add(current)
        user = next(iter(current.users))
        if user.target == torch.ops.quantized_ops.quantize.default:
            return user
        if not is_address_only_view(user, current):
            return None
        current = user
    return None


def lower_magnet_alpha_adaptive_average_pool2d(
    model: GraphModule,
) -> GraphModule:
    """Lower proved nonnegative INT8 7x7 global average into DWC plus NMP."""
    graph = model.graph
    plans: list[AdaptiveAveragePlan] = []
    for pool in list(graph.nodes):
        if pool.target != torch.ops.aten.adaptive_avg_pool2d.default:
            continue
        dequantize = get_arg_value(pool, 0, "self")
        if (
            not isinstance(dequantize, Node)
            or dequantize.target != torch.ops.quantized_ops.dequantize.default
        ):
            continue
        source = get_arg_value(dequantize, 0, "input")
        if not has_physical_payload(source, "int8"):
            continue
        if (
            source.target != NMP_FIXED_REQUANTIZE_OPERATOR
            or int(source.kwargs.get("output_precision", 0)) != 8
            or not bool(source.kwargs.get("relu", False))
        ):
            raise ValueError(
                f"{pool.name}: Alpha adaptive average requires a proved "
                "nonnegative physical signed-INT8 NMP producer"
            )

        source_shape = node_shape(source)
        output_shape = node_shape(pool)
        output_size = tuple(int(value) for value in pool.args[1])
        if (
            len(source_shape) != 4
            or source_shape[0] != 1
            or source_shape[1] <= 1
            or source_shape[-2:] != (7, 7)
            or output_size != (1, 1)
            or output_shape != (source_shape[0], source_shape[1], 1, 1)
        ):
            raise ValueError(
                f"{pool.name}: Alpha adaptive average requires static "
                "batch-one Cx7x7 input and one-by-one output"
            )
        if tuple(dequantize.users) != (pool,):
            raise ValueError(
                f"{pool.name}: Alpha adaptive-average dequantizer must be "
                "owned only by the pool"
            )

        _, source_scale = residual_dequantized_code(model, dequantize)
        producer_scale = direct_residual_output_scale(source)
        if producer_scale != source_scale:
            raise ValueError(
                f"{pool.name}: Alpha adaptive-average source scale "
                f"{producer_scale} does not equal dequantization scale "
                f"{source_scale}"
            )

        quantizer = _adaptive_average_output_quantizer(pool)
        if quantizer is None:
            raise ValueError(
                f"{pool.name}: Alpha adaptive average requires one linear "
                "address-only suffix and one INT8 output quantizer"
            )
        output_scale = residual_quantizer_scale(model, quantizer)
        kernel_size = source_shape[-2:]
        scale_factor, binary_point = prove_exact_adaptive_average_dwc_encoding(
            source_scale,
            output_scale,
            math.prod(kernel_size),
            context=pool.name,
        )
        plans.append(
            AdaptiveAveragePlan(
                pool=pool,
                source=source,
                source_scale=source_scale,
                output_quantizer=quantizer,
                output_scale=output_scale,
                channels=source_shape[1],
                kernel_size=kernel_size,
                scale_factor=scale_factor,
                binary_point=binary_point,
            )
        )

    for plan in plans:
        output = materialize_magnet_alpha_depthwise_requantize(
            model,
            plan.pool,
            plan.source,
            weight_name=f"{plan.pool.name}_alpha_dwc_weight",
            weight_value=torch.ones(
                (plan.channels, 1, *plan.kernel_size),
                dtype=torch.float32,
            ),
            accumulator_scale_name=(
                f"{plan.pool.name}_alpha_accumulator_scale"
            ),
            accumulator_scale_value=(
                plan.source_scale / math.prod(plan.kernel_size)
            ),
            scale_factor=plan.scale_factor,
            binary_point=plan.binary_point,
            output_scale=plan.output_scale,
            output_precision=8,
            relu=False,
            output_meta=plan.pool.meta,
        )
        plan.output_quantizer.replace_all_uses_with(output)
        graph.erase_node(plan.output_quantizer)

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    logger.info(
        "Lowered %d exact Magnet Alpha adaptive-average operations",
        len(plans),
    )
    return model


__all__ = ["lower_magnet_alpha_adaptive_average_pool2d"]

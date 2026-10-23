# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Magnet Alpha executable physical-movement lowering."""

import copy
import logging

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.codegen.passes.utils import get_arg_value
from voyager_compiler.codegen.target_backends.magnet_alpha.operators import (
    UPSAMPLE_NEAREST2D_OPERATOR,
)
from voyager_compiler.quantization_contracts import (
    get_physical_quantization_contract,
    set_physical_quantization_contract,
)

logger = logging.getLogger(__name__)


def lower_magnet_alpha_integer_data_movement(
    model: GraphModule,
) -> GraphModule:
    """Give physical INT8 nearest upsample an exact executable reference."""
    graph = model.graph
    lowered = 0
    for node in list(graph.nodes):
        if node.target != torch.ops.aten.upsample_nearest2d.vec:
            continue
        source = get_arg_value(node, 0, "input")
        if not isinstance(source, Node):
            continue
        domain = get_physical_quantization_contract(source)
        if domain is None or domain.payload_dtype != "int8":
            continue
        with graph.inserting_before(node):
            replacement = graph.call_function(
                UPSAMPLE_NEAREST2D_OPERATOR,
                node.args,
                node.kwargs,
            )
        replacement.meta = copy.copy(node.meta)
        set_physical_quantization_contract(replacement, domain)
        node.replace_all_uses_with(replacement)
        graph.erase_node(node)
        lowered += 1

    graph.lint()
    model.recompile()
    logger.info(
        "Lowered %d physical INT8 nearest-upsample operations",
        lowered,
    )
    return model


__all__ = ["lower_magnet_alpha_integer_data_movement"]

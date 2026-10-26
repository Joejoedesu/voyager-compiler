"""Explicit historical Voyager numerical recipe for the sealed September suite.

This is an approximation policy, not semantics-preserving normalization:
RMSNorm was replaced by LayerNorm and softmax used its input dtype. New
workloads retain native normalization unless they request this recipe.
"""

import logging
from typing import List

import torch
from torch.fx import GraphModule
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher

from voyager_compiler.export_utils import get_aten_graph_module
from voyager_compiler.quantization.quantizer.xnnpack_quantizer_utils import (
    _convert_scalars_to_attrs,
)

logger = logging.getLogger(__name__)


def replace_rmsnorm_with_layer_norm(
    model: GraphModule,
    layer_norm: torch.nn.Module,
    example_input,
    convert_scalars_to_attrs=False,
):
    """Replace LLaMA RMSNorm with ATen layer_norm"""
    original_graph = model.graph

    pattern = get_aten_graph_module(layer_norm, example_input)
    if convert_scalars_to_attrs:
        _convert_scalars_to_attrs(pattern)
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

    weight_node = next(
        iter(n for n in pattern_graph.nodes if n.target == "weight")
    )

    for match in _matches:
        input_node = match.placeholder_nodes[0]
        output_node = match.returning_nodes[0]
        input_shape = input_node.meta["val"].shape
        new_weight_node = match.nodes_map[weight_node]
        layer_norm_inputs = [input_node, [input_shape[-1]], new_weight_node]

        with original_graph.inserting_before(output_node):
            new_node = original_graph.call_function(
                torch.ops.aten.layer_norm.default, tuple(layer_norm_inputs), {}
            )

        output_node.replace_all_uses_with(new_node)
        original_graph.erase_node(output_node)

        new_node.meta = output_node.meta

    original_graph.lint()
    original_graph.eliminate_dead_code()
    model.recompile()


def remove_softmax_dtype_cast(model: torch.fx.GraphModule):
    graph = model.graph
    for node in list(model.graph.nodes):
        if node.target == torch.ops.aten.softmax.int:
            node.args = node.args[:2]

    graph.lint()
    model.recompile()
    return model

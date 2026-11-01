"""Opt-in algorithm recipes for one existing SDPA region.

The materialized form reuses the registered GEMM/vector recipes; the streaming
form reuses the FA3 builder and its paired attention model. This first contract
is dropout-free FP32 attention with equal batch/head dimensions and no mask.
Quantized attention, masking, GQA and arbitrary decomposed-pattern recognition
need their own numerical/legality validation before joining this comparison.
"""

import math

import torch
from torch.fx import Graph, GraphModule

from voyager_compiler.codegen.node_info import get_arg_value
from voyager_compiler.compiler.bufferize.attention_v3 import build_attention_fa3
from voyager_compiler.compiler.bufferize.regions import (
    build_sequential_region,
    estimate_sequential_region,
)
from voyager_compiler.extensions import CandidateRejected, ImplementationSpec
from voyager_compiler.shape_prop import ShapeProp, fake_like

SDPA = torch.ops.aten.scaled_dot_product_attention.default


def _check_contract(node):
    q, k, v = (operand.value for operand in node.args[:3])
    if (
        any(
            t.dtype != torch.float32 or t.ndim not in (2, 3, 4)
            for t in (q, k, v)
        )
        or q.shape[:-2] != k.shape[:-2]
        or k.shape[:-2] != v.shape[:-2]
        or q.shape[-1] != k.shape[-1]
        or k.shape[-2] != v.shape[-2]
        or q.shape[-1] != v.shape[-1]
        or any(d <= 0 for t in (q, k, v) for d in t.shape)
    ):
        raise CandidateRejected(
            "attention algorithms require FP32 rank 2–4 operands with equal "
            "batch/head dimensions, equal Q/K/V head widths, and nonempty axes"
        )
    if (
        get_arg_value(node, 3, "attn_mask", None) is not None
        or get_arg_value(node, 4, "dropout_p", 0.0) != 0
        or get_arg_value(node, 5, "is_causal", False)
        or node.kwargs.get("enable_gqa", False)
    ):
        raise CandidateRejected(
            "attention algorithm comparison currently excludes masks, dropout, "
            "causal attention and GQA"
        )
    scale = node.kwargs.get("scale")
    if scale is not None and not math.isfinite(scale):
        raise CandidateRejected("attention scale must be finite")


def attention_implementations(components, *, prefix="attention"):
    """Return materialized/streaming specs, registered together by a target.

    ``components`` are the target's GEMM/vector and layout recipes, excluding
    these outer recipes. The caller must supply a runtime model that prices the
    FA3 execution structure as well as those components. Use separate models via
    ``ImplementationSpec.model`` when needed. No target registration is mutated.
    """
    components = tuple(components)

    def matches(node, anchor):
        return node is anchor and node.target == SDPA

    def materialized(node, context, parameters):
        from voyager_compiler.compiler.normalize import transform

        _check_contract(node)
        graph = Graph()
        operands = {
            source: graph.placeholder(f"input_{i}")
            for i, source in enumerate(node.all_input_nodes)
        }
        q, k, v = (operands[source] for source in node.args[:3])
        kt = graph.call_function(torch.ops.aten.transpose.int, (k, -2, -1))
        scores = graph.call_function(torch.ops.aten.matmul.default, (q, kt))
        scale = node.kwargs.get("scale")
        if scale is None:
            scale = 1 / math.sqrt(node.args[0].value.shape[-1])
        scores = graph.call_function(torch.ops.aten.mul.Tensor, (scores, scale))
        probabilities = graph.call_function(
            torch.ops.aten._softmax.default, (scores, -1, False)
        )
        result = graph.call_function(
            torch.ops.aten.matmul.default, (probabilities, v)
        )
        graph.output((result,))
        region = GraphModule(torch.nn.Module(), graph)
        args = tuple(fake_like(source.value) for source in node.all_input_nodes)
        ShapeProp(region).propagate(*args)
        # Shared per-operation padding restores logical shapes before softmax;
        # padded key positions must never join its normalization denominator.
        # Group operand views with their GEMM so the shared builder can bind
        # them to external operands. An empty table enables that structural
        # grouping without fusing compute stages across the region.
        transform(
            region,
            args,
            config=context.tiler.config,
            patterns=(),
        )
        return build_sequential_region(region, context, components)

    def streaming(node, context, parameters):
        _check_contract(node)
        graph = build_attention_fa3(node, context, accumulate_fp32=True)
        if graph is None:
            raise CandidateRejected(
                "FA3 builder cannot realize this attention region"
            )
        return graph

    return (
        ImplementationSpec(
            f"{prefix}.materialized",
            matches,
            materialized,
            estimate=estimate_sequential_region,
            description="SDPA as sequential GEMM, scale, softmax and GEMM kernels",
        ),
        ImplementationSpec(
            f"{prefix}.streaming",
            matches,
            streaming,
            description="SDPA with the shared FA3 running-softmax pipeline",
        ),
    )

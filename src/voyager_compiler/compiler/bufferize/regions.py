"""Compose sequential kernels into one fixed-region implementation candidate.

Region recognition belongs to normalization; this helper consumes an already
chosen, executable region. It neither chooses overlapping graph partitions nor
allocates addresses. A different algorithm can build a different region body
while preserving the same external inputs, outputs and numerical contract.
"""

import copy
import math

from voyager_compiler.extensions import CandidateRejected
from voyager_compiler.ir.plan import CompilationPlan, Cost


def build_sequential_region(graph, context, implementations):
    """Bufferize a private region body and retain its component decisions.

    ``graph`` is normalized FX with populated shapes, in its external operand
    order. Supply the component recipes explicitly; do not recursively include
    the outer recipe. The ordinary builders drain each component before the next
    component executes. Their complete costs can therefore be summed. Algorithms
    with cross-component overlap need their own builder and cost model instead.

    Returned metadata ``region_choices`` and ``region_cost`` is consumed by the
    planner and ``estimate_sequential_region``. All compute components must have
    finite, nonnegative costs in the same target units. No missing cost is zero.
    Final capacity/bank/lifetime validation remains the shared allocator's job.
    """
    from voyager_compiler.compiler.bufferize.bufferization import (
        bufferize_graph,
    )
    from voyager_compiler.compiler.search.tiler import Tiler
    from voyager_compiler.shape_prop import ShapeProp, fake_like

    if context.tiler is None:
        raise CandidateRejected("sequential region requires a runtime model")
    tensors = (*graph.parameters(), *graph.buffers())
    candidate = copy.deepcopy(graph, {id(t): t for t in tensors})
    ShapeProp(candidate).propagate(
        *(
            fake_like(n.value)
            for n in graph.graph.nodes
            if n.op == "placeholder"
        )
    )
    plan = CompilationPlan("region")
    tiler = Tiler(context.tiler.model, context.tiler.runtime_tolerance)
    bufferize_graph(
        candidate,
        pipelined=context.num_slots > 1,
        tiler=tiler,
        async_pipeline=context.async_pipeline,
        single_buffer_tail=context.single_buffer_tail,
        flash_attention_v3=context.flash_attention_v3,
        bool_mask=context.bool_mask,
        implementations=implementations,
        compilation_plan=plan,
    )
    choices = tuple(plan.choices.items())
    if not choices or any(
        choice.cost is None
        or not math.isfinite(choice.cost.latency)
        or choice.cost.latency < 0
        for _, choice in choices
    ):
        raise CandidateRejected(
            "every region component needs a finite modeled cost"
        )
    traffic = [choice.cost.dram_bytes for _, choice in choices]
    candidate.meta["region_choices"] = choices
    candidate.meta["region_cost"] = Cost(
        sum(choice.cost.latency for _, choice in choices),
        sum(traffic) if all(t is not None for t in traffic) else None,
    )
    return candidate


def estimate_sequential_region(node, graph, context, parameters):
    """Price the complete region once, retaining component costs for inspection."""
    return graph.meta["region_cost"]

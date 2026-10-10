"""Static, dependency-ready list scheduling before per-engine order is fixed.

The lookahead is compiler policy, not a claim about hardware queue capacity.
Input timings guide ordering but are never fitted to a measured kernel here.
"""

from collections import defaultdict
from dataclasses import replace

from voyager_compiler.codegen.transform.tiling.execution import (
    Dependency,
    RepeatedGraph,
    evaluate_graph,
)


def schedule(graph, *, window=16):
    if type(window) is not int or window < 1:
        raise ValueError("Reorder window must be a positive integer")
    if graph.repetitions != 1:
        raise ValueError("Issue scheduling requires expanded finite ISA")
    from .physical_context import transform

    # A bounded source horizon exposes upcoming work; actual data and physical
    # reuse dependencies remain intact. Add existing readiness laws before
    # choosing an order, rather than ignoring handoffs during scheduling.
    nodes, history = [], defaultdict(list)
    for i, node in enumerate(graph.nodes):
        previous = history[node.resource]
        edges = node.dependencies
        if len(previous) >= window:
            edges += (Dependency(previous[-window], milestone="issue"),)
        nodes.append(replace(node, dependencies=edges))
        previous.append(i)
    bounded = RepeatedGraph(tuple(nodes))
    gated, _ = transform(bounded, "_readiness-only")
    starts = {}

    def observe(node, start):
        starts[node.name] = start
        return node

    evaluate_graph(gated, event_timing=observe)
    order = sorted(range(len(nodes)), key=lambda i: (starts[nodes[i].name], i))
    remap = {old: new for new, old in enumerate(order)}
    # Retain admission edges as well as all original hazards.
    result = RepeatedGraph(
        tuple(
            replace(
                nodes[i],
                dependencies=tuple(
                    replace(d, source=remap[d.source])
                    for d in nodes[i].dependencies
                ),
            )
            for i in order
        )
    )
    return result, dict(
        reorder_window=window,
        reordered_events=sum(i != old for i, old in enumerate(order)),
        scheduler="dependency-ready ASAP with bounded source horizon",
    )

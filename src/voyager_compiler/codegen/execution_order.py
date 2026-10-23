# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Validated execution order for a transformed FX graph."""

from dataclasses import dataclass

from torch.fx import GraphModule, Node

FX_EXECUTION_ORDER_MODE = "fx"
DP_EXECUTION_ORDER_MODE = "dp"
DEFAULT_EXECUTION_ORDER_MODE = FX_EXECUTION_ORDER_MODE
EXECUTION_ORDER_MODES = (
    FX_EXECUTION_ORDER_MODE,
    DP_EXECUTION_ORDER_MODE,
)
EXECUTION_ORDER_RESULT_META = "execution_order_result"


def validate_execution_order_mode(
    mode: str,
    *,
    bufferized: bool,
    end_to_end: bool,
) -> None:
    """Validate the representation/residency scope of one order mode."""
    if mode not in EXECUTION_ORDER_MODES:
        raise ValueError(f"Unknown execution order mode: {mode}")
    if mode == DP_EXECUTION_ORDER_MODE and (bufferized or not end_to_end):
        raise ValueError(
            "DP execution order is available only for operator_graph "
            "+ end_to_end compilation"
        )


@dataclass(frozen=True)
class ExecutionOrder:
    """One immutable flat execution sequence.

    Positions are deliberately derived by consumers instead of being stored
    redundantly in this value.
    """

    nodes: tuple[Node, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, tuple):
            raise TypeError("ExecutionOrder.nodes must be a tuple")


def fx_execution_order(model: GraphModule) -> ExecutionOrder:
    """Capture the graph's current linked-list order."""
    return ExecutionOrder(tuple(model.graph.nodes))


def execution_positions(order: ExecutionOrder) -> dict[Node, int]:
    """Derive node positions for one execution-order consumer."""
    if not isinstance(order, ExecutionOrder):
        raise TypeError("expected an ExecutionOrder")
    return {node: position for position, node in enumerate(order.nodes)}


def validate_execution_order(
    model: GraphModule,
    order: ExecutionOrder,
) -> None:
    """Require an exact topological permutation of the current FX graph."""
    if not isinstance(order, ExecutionOrder):
        raise TypeError("expected an ExecutionOrder")
    if not all(isinstance(node, Node) for node in order.nodes):
        raise TypeError("ExecutionOrder may contain only FX nodes")
    if len(set(order.nodes)) != len(order.nodes):
        raise ValueError("ExecutionOrder contains duplicate nodes")

    graph_nodes = tuple(model.graph.nodes)
    graph_node_set = set(graph_nodes)
    ordered_node_set = set(order.nodes)
    foreign = ordered_node_set - graph_node_set
    missing = graph_node_set - ordered_node_set
    if foreign or missing:
        details = []
        if missing:
            details.append(
                "missing " + ", ".join(sorted(node.name for node in missing))
            )
        if foreign:
            details.append(
                "foreign " + ", ".join(sorted(node.name for node in foreign))
            )
        raise ValueError(
            "ExecutionOrder is not an exact graph permutation: "
            + "; ".join(details)
        )

    positions = execution_positions(order)
    for node in order.nodes:
        for predecessor in node.all_input_nodes:
            if positions[predecessor] >= positions[node]:
                raise ValueError(
                    f"ExecutionOrder places {node.name} before dependency "
                    f"{predecessor.name}"
                )


def execution_order_for_model(model: GraphModule) -> ExecutionOrder:
    """Return the selected plan order, or FX order before planning."""
    plan = getattr(model, "meta", {}).get("storage_plan")
    if plan is None:
        return fx_execution_order(model)
    order = getattr(plan, "execution_order", None)
    if order is None:
        raise ValueError("storage plan has no execution order")
    validate_execution_order(model, order)
    return order


__all__ = [
    "DEFAULT_EXECUTION_ORDER_MODE",
    "DP_EXECUTION_ORDER_MODE",
    "EXECUTION_ORDER_MODES",
    "EXECUTION_ORDER_RESULT_META",
    "ExecutionOrder",
    "FX_EXECUTION_ORDER_MODE",
    "execution_order_for_model",
    "execution_positions",
    "fx_execution_order",
    "validate_execution_order_mode",
    "validate_execution_order",
]

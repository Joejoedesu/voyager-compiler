# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from voyager_compiler.codegen.hardware import MemoryLevelConfig, TensorRole
from voyager_compiler.codegen.memory import (
    MemorySpace,
    Segment,
    align_size,
    compute_tensor_size,
)
from voyager_compiler.codegen.operands import (
    classify_operand,
    require_allocation,
)

logger = logging.getLogger(__name__)


class BankingPolicy(Enum):
    """
    Policy for handling tensors that are not explicitly listed in any partition.
    """

    SEPARATE_ALIGNED = auto()
    SEPARATE_UNALIGNED = auto()


@dataclass(frozen=True)
class BankPartition:
    """
    A bank partition groups multiple logical tensor roles into a single
    bank-aligned allocation unit.

    - tensors: logical tensor roles (each role maps to at most one node)
    - max_banks: optional constraint on how many banks this partition may span
    - align: whether to apply bank-size rounding to the aggregate size
    """

    tensors: Tuple[str, ...]
    max_banks: Optional[int] = None
    align: bool = True


def get_tensor_role(
    operation,
    tensor,
    operand_key=None,
    operator_traits=None,
) -> TensorRole:
    """Classify one operation operand for physical memory selection."""
    return classify_operand(
        operation,
        tensor,
        operand_key,
        operator_traits,
    ).role


@dataclass(frozen=True)
class BankingStrategy:
    """
    A banking strategy is an ordered list of partitions.
    Earlier strategies are assumed to be higher-performance.
    """

    partitions: Tuple[BankPartition, ...]
    unspecified_policy: BankingPolicy = BankingPolicy.SEPARATE_ALIGNED

    def evaluate(
        self,
        key_to_node,
        node,
        tile_shapes,
        bank_width,
        bank_size,
        unroll_dim=None,
        memory_level: MemoryLevelConfig | None = None,
        operator_traits=None,
        size_node_overrides=None,
    ):
        if memory_level is not None:
            return self._evaluate_memory_level(
                key_to_node,
                node,
                tile_shapes,
                unroll_dim,
                memory_level,
                operator_traits,
                size_node_overrides,
            )

        size_node_overrides = size_node_overrides or {}

        specified_nodes = set()

        node_to_segment: Dict[Any, Segment] = {}
        current_offset = 0

        # Only keep tiled shapes for input and output nodes
        tile_shapes = {
            n: s
            for n, s in tile_shapes.items()
            if (
                n in node.all_input_nodes
                and require_allocation(n, operator_traits)
            )
            or n is node
        }

        logger.debug("Evaluating banking strategy:")

        for part in self.partitions:
            nodes = [
                key_to_node[key] for key in part.tensors if key in key_to_node
            ]

            if not nodes:
                continue

            logger.debug(
                f"  Partition tensors: {part.tensors} -> nodes: {nodes}"
            )
            specified_nodes.update(nodes)

            # Make sure the partition starts from a new bank
            if part.align:
                current_offset = align_size(current_offset, bank_size)

            for n in nodes:
                # For fused operation the output node of the first node becomes
                # an intermediate and will not appear in tile_shapes
                if n not in tile_shapes:
                    continue

                t_size = compute_tensor_size(
                    size_node_overrides.get(n, n),
                    tile_shapes[n],
                    n is node,
                    bank_width,
                    unroll_dim,
                    operator_traits=operator_traits,
                )

                node_to_segment[n] = Segment(
                    start=current_offset,
                    end=current_offset + t_size,
                    memory_space=MemorySpace.SCRATCHPAD,
                    node=n,
                    storage_plane=MemorySpace.SCRATCHPAD,
                    memory_instance="Scratchpad",
                )

                current_offset += t_size

            if part.align:
                current_offset = align_size(current_offset, bank_size)

        unspecified_nodes = [n for n in tile_shapes if n not in specified_nodes]
        if not unspecified_nodes:
            return current_offset, node_to_segment

        if self.unspecified_policy == BankingPolicy.SEPARATE_ALIGNED:
            current_offset = align_size(current_offset, bank_size)

        for n in unspecified_nodes:
            t_size = compute_tensor_size(
                size_node_overrides.get(n, n),
                tile_shapes[n],
                n is node,
                bank_width,
                unroll_dim,
                operator_traits=operator_traits,
            )

            node_to_segment[n] = Segment(
                start=current_offset,
                end=current_offset + t_size,
                memory_space=MemorySpace.SCRATCHPAD,
                node=n,
                storage_plane=MemorySpace.SCRATCHPAD,
                memory_instance="Scratchpad",
            )

            current_offset += t_size

            if self.unspecified_policy == BankingPolicy.SEPARATE_ALIGNED:
                current_offset = align_size(current_offset, bank_size)

        logger.debug(f"  Unspecified tensors: {unspecified_nodes}")
        logger.debug(f"    Unspecified policy: {self.unspecified_policy}")
        logger.debug("    Scratchpad segments:")
        for n, seg in node_to_segment.items():
            logger.debug(f"      Node: {n}, Segment: [{seg.start}, {seg.end})")
        logger.debug(f"    Total scratchpad used: {current_offset} bytes")

        return current_offset, node_to_segment

    def _evaluate_memory_level(
        self,
        key_to_node,
        node,
        tile_shapes,
        unroll_dim,
        memory_level,
        operator_traits,
        size_node_overrides,
    ):
        """Evaluate this strategy using role-specific physical banks."""
        size_node_overrides = size_node_overrides or {}
        tile_shapes = {
            tensor: shape
            for tensor, shape in tile_shapes.items()
            if (
                tensor in node.all_input_nodes
                and require_allocation(tensor, operator_traits)
            )
            or tensor is node
        }
        key_by_node = {tensor: key for key, tensor in key_to_node.items()}
        specified_nodes = set()
        node_to_segment = {}
        offsets = {memory.name: 0 for memory in memory_level.instances}

        def place(nodes, align):
            nodes_by_memory = {}
            for tensor in nodes:
                if tensor not in tile_shapes:
                    continue
                role = get_tensor_role(
                    node,
                    tensor,
                    key_by_node.get(tensor),
                    operator_traits,
                )
                memory_name = memory_level.memory_name_for_role(role)
                nodes_by_memory.setdefault(memory_name, []).append(tensor)

            for memory_name, tensors in nodes_by_memory.items():
                memory = memory_level.memory(memory_name)
                if align:
                    offsets[memory_name] = align_size(
                        offsets[memory_name],
                        memory.bank_capacity_bytes,
                    )

                for tensor in tensors:
                    size = compute_tensor_size(
                        size_node_overrides.get(tensor, tensor),
                        tile_shapes[tensor],
                        tensor is node,
                        memory.allocation_alignment_bytes,
                        unroll_dim,
                        operator_traits=operator_traits,
                    )
                    start = offsets[memory_name]
                    partition = memory_level.memory_index(memory_name)
                    node_to_segment[tensor] = Segment(
                        start=start,
                        end=start + size,
                        memory_space=partition,
                        node=tensor,
                        storage_plane=MemorySpace.SCRATCHPAD,
                        memory_instance=memory_name,
                    )
                    offsets[memory_name] += size

                if align:
                    offsets[memory_name] = align_size(
                        offsets[memory_name],
                        memory.bank_capacity_bytes,
                    )

        for partition in self.partitions:
            nodes = [
                key_to_node[key]
                for key in partition.tensors
                if key in key_to_node
            ]
            specified_nodes.update(nodes)
            place(nodes, partition.align)

        unspecified_nodes = [
            tensor for tensor in tile_shapes if tensor not in specified_nodes
        ]
        if self.unspecified_policy == BankingPolicy.SEPARATE_ALIGNED:
            for tensor in unspecified_nodes:
                place([tensor], True)
        else:
            place(unspecified_nodes, False)
        return sum(offsets.values()), node_to_segment

    def print_banking_info(
        self,
        key_to_node,
        node,
        operator_traits=None,
    ):
        log_lines = []
        handled_nodes = set()

        for i, part in enumerate(self.partitions):
            entries = []
            for key in part.tensors:
                if key in key_to_node:
                    n = key_to_node[key]
                    handled_nodes.add(n)
                    entries.append(f"{n.name} ({key})")

            if entries:
                entry_str = ", ".join(entries)
                log_lines.append(f"Partition {i}: {entry_str}")

        if self.unspecified_policy == BankingPolicy.SEPARATE_UNALIGNED:
            all_relevant = list(node.all_input_nodes) + [node]
            unspecified_nodes = []

            for n in all_relevant:
                if n not in handled_nodes and require_allocation(
                    n,
                    operator_traits,
                ):
                    role = "output" if n is node else "input"
                    unspecified_nodes.append(f"{n.name} ({role})")

            if unspecified_nodes:
                entry_str = ", ".join(unspecified_nodes)
                log_lines.append(f"Partition Unspecified: {entry_str}")

        if log_lines:
            logger.warning(
                f"[BANK_CONFLICT]: {node.name}: Not all nodes are placed in "
                f"separate banks; performance may be impacted."
            )
            logger.warning("\n".join(log_lines))
        else:
            logger.info(f"{node}: All nodes are placed in separate banks.")


class BankingStrategyRegistry:
    """
    Global registry mapping op_kind -> list of banking strategies.
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, List[BankingStrategy]] = {}

    def register(
        self, op_kind: Any, strategies: Sequence[BankingStrategy]
    ) -> None:
        self._strategies[op_kind] = list(strategies)

    def get(self, op_kind: Any) -> List[BankingStrategy]:
        if op_kind in self._strategies:
            return self._strategies[op_kind]

        return get_default_strategies(op_kind)


BANKING_STRATEGY_REGISTRY = BankingStrategyRegistry()


def register_banking_strategies(targets: Any) -> Callable:
    """
    Decorator for registering banking strategies for one or more op kinds.

    The decorated function must return a list (or tuple) of BankingStrategy.
    """
    target_list = targets if isinstance(targets, (list, tuple)) else [targets]

    def decorator(fn: Callable[[], Sequence[BankingStrategy]]):
        for target in target_list:
            strategies = fn(target)
            if not isinstance(strategies, (list, tuple)):
                raise TypeError(
                    "Banking strategy function must return a list or tuple of BankingStrategy"
                )

            for s in strategies:
                if not isinstance(s, BankingStrategy):
                    raise TypeError("All elements must be BankingStrategy")

            BANKING_STRATEGY_REGISTRY.register(target, strategies)
        return fn

    return decorator


def _get_scope(target) -> str:
    if hasattr(target, "_schema"):
        scope = target._schema.name.split("::")[1]
    else:
        scope = str(target)
    return scope


def get_default_strategies(target) -> List[BankingStrategy]:
    scope = _get_scope(target)
    return [
        BankingStrategy(
            partitions=(),
            unspecified_policy=BankingPolicy.SEPARATE_ALIGNED,
        ),
        BankingStrategy(
            partitions=(BankPartition((f"{scope}::output",)),),
            unspecified_policy=BankingPolicy.SEPARATE_UNALIGNED,
        ),
        BankingStrategy(
            partitions=(),
            unspecified_policy=BankingPolicy.SEPARATE_UNALIGNED,
        ),
    ]


@register_banking_strategies(
    [
        torch.ops.aten.conv2d.default,
        torch.ops.aten.linear.default,
        torch.ops.quantized_ops.conv2d.default,
        torch.ops.quantized_ops.linear.default,
    ]
)
def _(target) -> List[BankingStrategy]:
    scope = _get_scope(target)
    return [
        BankingStrategy(
            partitions=(),
            unspecified_policy=BankingPolicy.SEPARATE_ALIGNED,
        ),
        BankingStrategy(
            (BankPartition((f"{scope}::weight", f"{scope}::bias")),)
        ),
        BankingStrategy(
            (
                BankPartition(
                    (f"{scope}::weight", f"{scope}::bias", f"{scope}::output")
                ),
            )
        ),
        BankingStrategy(
            partitions=(),
            unspecified_policy=BankingPolicy.SEPARATE_UNALIGNED,
        ),
    ]


@register_banking_strategies(torch.ops.aten.matmul.default)
def _(target) -> List[BankingStrategy]:
    return [
        BankingStrategy(
            partitions=(),
            unspecified_policy=BankingPolicy.SEPARATE_ALIGNED,
        ),
        BankingStrategy((BankPartition(("matmul::other", "matmul::output")),)),
        BankingStrategy(
            partitions=(),
            unspecified_policy=BankingPolicy.SEPARATE_ALIGNED,
        ),
    ]


def _scale_bearing_conv2d_strategies(
    target,
) -> List[BankingStrategy]:
    scope = _get_scope(target)
    input_scale = f"{scope}::input_scale"
    weight_scale = f"{scope}::weight_scale"
    folded_scale = f"{scope}::folded_coarse_scale"
    weight_partition = (
        f"{scope}::weight",
        weight_scale,
        f"{scope}::bias",
        folded_scale,
    )
    return [
        BankingStrategy(
            partitions=(),
            unspecified_policy=BankingPolicy.SEPARATE_ALIGNED,
        ),
        BankingStrategy((BankPartition(weight_partition),)),
        BankingStrategy(
            (
                BankPartition((f"{scope}::input", input_scale)),
                BankPartition(weight_partition),
            )
        ),
        BankingStrategy(
            (
                BankPartition(
                    (
                        f"{scope}::input",
                        input_scale,
                        *weight_partition,
                    )
                ),
            )
        ),
        BankingStrategy(
            partitions=(),
            unspecified_policy=BankingPolicy.SEPARATE_UNALIGNED,
        ),
    ]


register_banking_strategies(torch.ops.quantized_ops.conv2d_mx.default)(
    _scale_bearing_conv2d_strategies
)


@register_banking_strategies(torch.ops.quantized_ops.linear_mx.default)
def _(target) -> List[BankingStrategy]:
    return [
        BankingStrategy(
            (
                BankPartition(("linear_mx::weight", "linear_mx::weight_scale")),
                BankPartition(
                    (
                        "linear_mx::A_data",
                        "linear_mx::A_indices",
                        "linear_mx::A_indptr",
                    )
                ),
            )
        ),
        BankingStrategy(
            (
                BankPartition(
                    (
                        "linear_mx::weight",
                        "linear_mx::weight_scale",
                        "linear_mx::bias",
                    )
                ),
                BankPartition(
                    (
                        "linear_mx::A_data",
                        "linear_mx::A_indices",
                        "linear_mx::A_indptr",
                    )
                ),
            )
        ),
        BankingStrategy(
            (
                BankPartition(("linear_mx::input", "linear_mx::input_scale")),
                BankPartition(
                    (
                        "linear_mx::weight",
                        "linear_mx::weight_scale",
                        "linear_mx::bias",
                    )
                ),
                BankPartition(
                    (
                        "linear_mx::A_data",
                        "linear_mx::A_indices",
                        "linear_mx::A_indptr",
                    )
                ),
            )
        ),
        BankingStrategy(
            (
                BankPartition(
                    (
                        "linear_mx::input",
                        "linear_mx::input_scale",
                        "linear_mx::weight",
                        "linear_mx::weight_scale",
                        "linear_mx::bias",
                        "linear_mx::A_data",
                        "linear_mx::A_indices",
                        "linear_mx::A_indptr",
                    )
                ),
            )
        ),
        BankingStrategy(
            partitions=(),
            unspecified_policy=BankingPolicy.SEPARATE_UNALIGNED,
        ),
    ]


@register_banking_strategies(torch.ops.quantized_ops.matmul_mx.default)
def _(target) -> List[BankingStrategy]:
    return [
        BankingStrategy(
            (
                BankPartition(("matmul_mx::other", "matmul_mx::weight_scale")),
                BankPartition(
                    (
                        "matmul_mx::A_data",
                        "matmul_mx::A_indices",
                        "matmul_mx::A_indptr",
                    )
                ),
            )
        ),
        BankingStrategy(
            (
                BankPartition(("matmul_mx::input", "matmul_mx::input_scale")),
                BankPartition(("matmul_mx::other", "matmul_mx::weight_scale")),
                BankPartition(
                    (
                        "matmul_mx::A_data",
                        "matmul_mx::A_indices",
                        "matmul_mx::A_indptr",
                    )
                ),
            )
        ),
        BankingStrategy(
            (
                BankPartition(
                    (
                        "matmul_mx::input",
                        "matmul_mx::input_scale",
                        "matmul_mx::other",
                        "matmul_mx::weight_scale",
                        "matmul_mx::A_data",
                        "matmul_mx::A_indices",
                        "matmul_mx::A_indptr",
                    )
                ),
            )
        ),
        BankingStrategy(
            partitions=(),
            unspecified_policy=BankingPolicy.SEPARATE_UNALIGNED,
        ),
    ]


def get_banking_strategies_for_op(
    op_kind: Any,
    operator_traits=None,
) -> List[BankingStrategy]:
    """Retrieve strategies admitted by generic or selected-target policy."""
    scale_bearing_operators = (
        ()
        if operator_traits is None
        else operator_traits.scale_bearing_conv2d_operators
    )
    if op_kind in scale_bearing_operators:
        return _scale_bearing_conv2d_strategies(op_kind)
    return BANKING_STRATEGY_REGISTRY.get(op_kind)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Shared operation-operand identity and tensor-role classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.fx import Node
from torch.fx.operator_schemas import normalize_function

from voyager_compiler.codegen.hardware import TensorRole
from voyager_compiler.codegen.storage_predicates import (
    is_address_only_view,
    is_zero_copy_storage_op,
)
from voyager_compiler.pt2e_utils import COMPILER_ONLY

_WEIGHT_OPERAND_KEYS = frozenset({"weight"})
_BIAS_OPERAND_KEYS = frozenset({"bias", "bias_payload"})
_SCALE_OPERAND_KEYS = frozenset({"scale"})


@dataclass(frozen=True)
class OperandDescriptor:
    """Identify one logical operand and its physical-memory role."""

    tensor: Node
    key: str | None
    role: TensorRole
    occurrence: tuple[object, ...] = ()


def _canonical_operand_key(operand_key: str | None) -> str | None:
    if operand_key is None:
        return None
    return operand_key.rsplit("::", 1)[-1]


def _is_matrix_rhs(operation: Node | None, key: str | None) -> bool:
    if operation is None or key != "other":
        return False
    target = str(operation.target).lower()
    return "linear" in target or "matmul" in target


def _role_from_key(
    operation: Node | None,
    tensor: Node,
    key: str | None,
    role_overrides,
) -> TensorRole | None:
    if (operation is not None and tensor is operation) or key == "output":
        return TensorRole.OUTPUT
    if key in role_overrides:
        return TensorRole(role_overrides[key])
    if key in _WEIGHT_OPERAND_KEYS or _is_matrix_rhs(operation, key):
        return TensorRole.WEIGHT
    if (
        key in _BIAS_OPERAND_KEYS
        or (key is not None and key.endswith("_bias"))
        or (key is not None and key.endswith("_bias_payload"))
    ):
        return TensorRole.BIAS
    if key in _SCALE_OPERAND_KEYS or (
        key is not None
        and key.endswith(
            (
                "_code",
                "_multiplier",
                "_qmap",
                "_scale",
            )
        )
    ):
        return TensorRole.SCALE
    return None


def classify_operand(
    operation: Node | None,
    tensor: Node,
    operand_key: str | None = None,
    operator_traits=None,
    occurrence: tuple[object, ...] = (),
) -> OperandDescriptor:
    """Return the canonical descriptor for one operation tensor operand."""

    key = _canonical_operand_key(operand_key)
    role_overrides = (
        {}
        if operator_traits is None
        else operator_traits.operand_role_overrides
    )
    role = _role_from_key(
        operation,
        tensor,
        key,
        role_overrides,
    )
    if role is None:
        # A name is not a semantic port. Unknown schema operands use the
        # ordinary activation path unless the selected target explicitly
        # overrides their normalized key.
        role = TensorRole.ACTIVATION
    return OperandDescriptor(
        tensor=tensor,
        key=key,
        role=role,
        occurrence=occurrence,
    )


def _node_occurrences(value, path=()):
    if isinstance(value, Node):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _node_occurrences(
                item,
                (*path, str(key)),
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _node_occurrences(item, (*path, index))


def _node_leaves(value):
    for _path, node in _node_occurrences(value):
        yield node


def _outer_tensor_sources(
    tensor: Node,
    outer_by_placeholder: dict[Node, object],
    outer_graph,
    extra_transparent_source=None,
) -> tuple[Node, ...]:
    """Trace an inner tensor identity to visible outer graph operands.

    Only storage-transparent operations are traversed. Propagating through
    arbitrary computation would incorrectly assign a downstream operand role
    to every value that contributed to the computed intermediate.
    """
    pending = [tensor]
    seen = set()
    sources = []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)

        outer = outer_by_placeholder.get(current)
        if outer is not None:
            sources.extend(_node_leaves(outer))
            continue

        source_node = current.meta.get("source_node")
        if isinstance(source_node, Node):
            if source_node.graph is outer_graph:
                sources.append(source_node)
            elif source_node.graph is current.graph:
                pending.append(source_node)
            continue

        source = _transparent_source(current)
        if source is None and extra_transparent_source is not None:
            source = extra_transparent_source(current)
        if source is not None:
            pending.extend(_node_leaves(source))

    return tuple(dict.fromkeys(sources))


def _transparent_source(node: Node):
    """Return the sole logical source of a target-independent zero-copy op."""
    if node.op != "call_function" or not node.args:
        return None
    if is_address_only_view(node):
        return node.args[0]
    if node.target in {
        torch.ops.aten.dropout.default,
        torch.ops.aten.to.dtype,
    } and is_zero_copy_storage_op(node):
        return node.args[0]
    return None


def _submodule_boundary(
    operation: Node,
    model,
):
    if operation.op != "call_module":
        raise TypeError(
            f"Cannot inspect submodule operands for {operation.name} "
            f"({operation.op})"
        )
    submodule = (
        operation.meta.get("submodule")
        if model is None
        else model.get_submodule(str(operation.target))
    )
    graph = getattr(submodule, "graph", None)
    if graph is None:
        raise TypeError(
            f"Cannot inspect non-graph submodule {operation.target}"
        )
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    outer_by_placeholder = {
        placeholder: (
            operation.args[index]
            if index < len(operation.args)
            else operation.kwargs.get(str(placeholder.target))
        )
        for index, placeholder in enumerate(placeholders)
    }
    return graph, outer_by_placeholder


def submodule_operand_descriptors(
    operation: Node,
    inner_operation: Node,
    model,
    *,
    extra_transparent_source=None,
    operator_traits=None,
) -> tuple[OperandDescriptor, ...]:
    """Project one inner operation's operands through a module boundary."""
    graph, outer_by_placeholder = _submodule_boundary(operation, model)
    if inner_operation.graph is not graph:
        raise ValueError(
            f"{inner_operation.name} is not inside {operation.target}"
        )
    if (
        inner_operation.op != "call_function"
        or _transparent_source(inner_operation) is not None
    ):
        return ()

    descriptors = []
    for descriptor in operand_descriptors(
        inner_operation,
        operator_traits=operator_traits,
    ):
        outer_sources = _outer_tensor_sources(
            descriptor.tensor,
            outer_by_placeholder,
            operation.graph,
            extra_transparent_source,
        )
        for outer_index, outer in enumerate(outer_sources):
            occurrence = descriptor.occurrence
            if len(outer_sources) > 1:
                occurrence = (*occurrence, outer_index)
            descriptors.append(
                OperandDescriptor(
                    tensor=outer,
                    key=descriptor.key,
                    role=descriptor.role,
                    occurrence=occurrence,
                )
            )
    return tuple(descriptors)


def operand_descriptors(
    operation: Node,
    model=None,
    *,
    operator_traits=None,
) -> tuple[OperandDescriptor, ...]:
    """Describe every operand occurrence without collapsing repeated nodes."""
    if operation.op == "call_function":
        normalized = normalize_function(
            operation.target,
            operation.args,
            operation.kwargs,
            normalize_to_only_use_kwargs=True,
        )
        if normalized is not None:
            arguments = normalized.kwargs.items()
        else:
            schema = getattr(operation.target, "_schema", None)
            if schema is None:
                raise TypeError(
                    f"Cannot normalize operands for {operation.name} "
                    f"({operation.target})"
                )
            arguments = tuple(
                (
                    argument.name,
                    (
                        operation.args[index]
                        if index < len(operation.args)
                        else operation.kwargs[argument.name]
                    ),
                )
                for index, argument in enumerate(schema.arguments)
                if (
                    index < len(operation.args)
                    or argument.name in operation.kwargs
                )
            )
        return tuple(
            classify_operand(
                operation,
                tensor,
                key,
                operator_traits,
                occurrence,
            )
            for key, value in arguments
            for occurrence, tensor in _node_occurrences(value)
        )

    graph, _outer_by_placeholder = _submodule_boundary(operation, model)
    descriptors = []
    for inner in graph.nodes:
        descriptors.extend(
            submodule_operand_descriptors(
                operation,
                inner,
                model,
                operator_traits=operator_traits,
            )
        )
    if not descriptors:
        raise TypeError(
            f"Cannot resolve operands through submodule {operation.target}"
        )
    return tuple(descriptors)


def classify_storage_role(
    node: Node,
    operator_traits=None,
) -> TensorRole:
    """Classify a stored graph value using the canonical descriptors."""
    explicit_role = node.meta.get("tensor_role")
    if explicit_role is not None:
        return TensorRole(explicit_role)
    if node.op == "placeholder":
        return TensorRole.ACTIVATION
    if node.op != "get_attr":
        return TensorRole.OUTPUT

    roles = []
    model = getattr(node.graph, "owning_module", None)
    for user in node.users:
        if user.op not in {"call_function", "call_module"}:
            continue
        try:
            descriptors = operand_descriptors(
                user,
                model,
                operator_traits=operator_traits,
            )
        except (RuntimeError, TypeError):
            descriptors = ()
        roles.extend(
            descriptor.role
            for descriptor in descriptors
            if descriptor.tensor.meta.get(
                "source_node",
                descriptor.tensor,
            )
            is node
        )

    for role in (
        TensorRole.WEIGHT,
        TensorRole.BIAS,
        TensorRole.SCALE,
        TensorRole.ACTIVATION,
    ):
        if role in roles:
            return role
    return classify_operand(
        None,
        node,
        operator_traits=operator_traits,
    ).role


def require_allocation(
    node: Node,
    operator_traits=None,
) -> bool:
    """Return whether a logical tensor needs independently allocated storage.

    Allocation eligibility is a property of an operand, not of a banking
    strategy or allocator.  Keeping the predicate beside operand identity and
    role classification gives tiling, banking, and model-wide allocation one
    target-qualified source of truth.
    """
    if node.meta.get(COMPILER_ONLY, False):
        return False

    folded_scale_operators = (
        ()
        if operator_traits is None
        else operator_traits.folded_scale_operators
    )
    non_allocating_operand_meta = (
        ()
        if operator_traits is None
        else operator_traits.non_allocating_operand_meta
    )
    is_instruction_operand = any(
        user.target in folded_scale_operators
        and user.kwargs.get("folded_coarse_scale") is node
        for user in node.users
    )
    if (
        any(node.meta.get(key, False) for key in non_allocating_operand_meta)
        or is_instruction_operand
    ):
        return False

    if node.op == "get_attr" and not node.users:
        return False

    value = getattr(node, "value", None)
    if value is None:
        return True
    if not isinstance(value, torch.Tensor):
        return False
    if node.op == "get_attr" and value.numel() == 1:
        return False
    return True

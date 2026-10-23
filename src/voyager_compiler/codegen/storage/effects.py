# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Generic FX storage effects used by logical-to-physical lowering."""

from __future__ import annotations

import operator

import torch
from torch.fx import Node

from voyager_compiler.codegen.storage.types import (
    StorageRelation,
    StorageRelationKind,
    ValueRef,
    ViewTransform,
)
from voyager_compiler.codegen.storage_predicates import (
    StorageEffectKind,
    is_address_only_view,
    is_zero_copy_storage_op,
    node_value,
    select_view,
    storage_op_traits,
)


def _first_input(
    node: Node,
    node_inputs: dict[Node, tuple[ValueRef, ...]],
) -> ValueRef | None:
    inputs = node_inputs.get(node, ())
    return inputs[0] if inputs else None


def _mutable_storage_source(
    node: Node,
    node_inputs: dict[Node, tuple[ValueRef, ...]],
) -> ValueRef | None:
    """Return the single logical value mutated by an operator schema."""
    schema = getattr(node.target, "_schema", None)
    if schema is None or not schema.is_mutable:
        return None
    written = [
        (index, argument)
        for index, argument in enumerate(schema.arguments)
        if (argument.alias_info is not None and argument.alias_info.is_write)
    ]
    if len(written) != 1:
        raise ValueError(
            f"{node.name}: storage analysis requires exactly one mutable "
            "tensor operand"
        )
    index, argument = written[0]
    value = (
        node.args[index]
        if index < len(node.args)
        else node.kwargs.get(argument.name)
    )
    if not isinstance(value, Node):
        raise ValueError(
            f"{node.name}: mutable tensor operand is not an FX value"
        )
    candidates = tuple(
        ref for ref in node_inputs.get(node, ()) if ref.node is value
    )
    if len(candidates) != 1:
        raise ValueError(
            f"{node.name}: mutable tensor operand must identify one "
            "logical tensor"
        )
    return candidates[0]


def normalize_getitem_index(
    producer: Node,
    index: int,
    producer_outputs: tuple[ValueRef, ...],
) -> int:
    """Normalize a Python sequence index against one FX aggregate result."""
    value = node_value(producer)
    if isinstance(value, (tuple, list)):
        length = len(value)
    else:
        top_level_indices = {
            output.path[0] for output in producer_outputs if output.path
        }
        length = max(top_level_indices, default=-1) + 1
    if index < 0 and -length <= index:
        return length + index
    return index


def _normalized_view_transform(node: Node) -> ViewTransform:
    """Describe addressable geometry without leaving FX syntax to lowering."""
    selection = select_view(node)
    if selection is not None:
        return ViewTransform(
            "contiguous_block",
            parameters=(
                ("block_index", selection.index),
                ("selected_dimension", selection.dim),
            ),
        )
    return ViewTransform(
        "identity_view",
        parameters=(("operator", str(node.target)),),
    )


def generic_storage_relation(
    ref: ValueRef,
    node_outputs: dict[Node, tuple[ValueRef, ...]],
    node_inputs: dict[Node, tuple[ValueRef, ...]],
) -> StorageRelation:
    """Return the conservative generic storage effect for one result."""
    node = ref.node
    source = _first_input(node, node_inputs)
    mutable_source = _mutable_storage_source(node, node_inputs)
    storage_traits = storage_op_traits(node.target)

    if node.op in {"placeholder", "get_attr"}:
        return StorageRelation(ref, StorageRelationKind.MATERIALIZE)

    if mutable_source is not None:
        kind = (
            StorageRelationKind.ALIAS
            if node.target == torch.ops.aten.detach_.default
            else StorageRelationKind.WRITE_INTO
        )
        return StorageRelation(
            ref,
            kind,
            source=mutable_source,
            transform=ViewTransform(
                "in_place",
                parameters=(("operator", str(node.target)),),
            ),
        )

    if (
        storage_traits.effect is StorageEffectKind.SPLIT_VIEW
        and source is not None
    ):
        return StorageRelation(
            ref,
            StorageRelationKind.VIEW,
            source=source,
            transform=ViewTransform(
                "split",
                parameters=(("result_index", ref.path),),
            ),
        )

    if node.target == operator.getitem and source is not None:
        producer = node.args[0] if node.args else None
        index = node.args[1] if len(node.args) > 1 else None
        producer_value = (
            node_value(producer) if isinstance(producer, Node) else None
        )
        if not isinstance(producer_value, (tuple, list)):
            raise NotImplementedError(
                f"{node.name}: tensor-valued operator.getitem must be "
                "normalized to aten.select/slice before storage analysis"
            )
        producer_outputs = (
            node_outputs.get(producer, ()) if isinstance(producer, Node) else ()
        )
        if isinstance(index, int) and len(producer_outputs) > 1:
            index = normalize_getitem_index(
                producer,
                index,
                producer_outputs,
            )
            selected_path = (index, *ref.path)
            selected = next(
                (
                    output
                    for output in producer_outputs
                    if output.path == selected_path
                ),
                None,
            )
            if selected is not None:
                return StorageRelation(
                    ref,
                    StorageRelationKind.ALIAS,
                    source=selected,
                )
        return StorageRelation(
            ref,
            StorageRelationKind.VIEW,
            source=source,
            transform=ViewTransform(
                "getitem",
                parameters=(("index", index),),
            ),
        )

    if node.meta.get("physical_copy", False) and source is not None:
        return StorageRelation(
            ref,
            StorageRelationKind.COPY,
            source=source,
        )

    if storage_traits.effect is StorageEffectKind.COPY and source is not None:
        return StorageRelation(
            ref,
            StorageRelationKind.COPY,
            source=source,
        )

    if (
        storage_traits.effect is StorageEffectKind.VIEW_OR_COPY
        and source is not None
    ):
        if is_address_only_view(node, source):
            return StorageRelation(
                ref,
                StorageRelationKind.VIEW,
                source=source,
                transform=_normalized_view_transform(node),
            )
        return StorageRelation(
            ref,
            StorageRelationKind.COPY,
            source=source,
        )

    if (
        storage_traits.effect is StorageEffectKind.ALIAS_OR_MATERIALIZE
        and source is not None
        and is_zero_copy_storage_op(node, source)
    ):
        return StorageRelation(
            ref,
            StorageRelationKind.ALIAS,
            source=source,
        )

    return StorageRelation(ref, StorageRelationKind.MATERIALIZE)


__all__ = [
    "generic_storage_relation",
    "is_zero_copy_storage_op",
    "node_value",
]

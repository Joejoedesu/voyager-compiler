"""Derived semantic relations and executable graph fingerprints."""

from __future__ import annotations

import hashlib
import json

from voyager_compiler.ir.graph import ValueRelation, ValueRelations


def graph_digest(graph):
    records = []
    for name, module in graph.named_modules():
        if not hasattr(module, "graph"):
            continue
        records.append((name, str(module.graph)))
        for node in module.graph.nodes:
            for key in ("memory", "scratchpad"):
                segment = node.meta.get(key)
                if segment is not None:
                    records.append(
                        (name, node.name, key, segment.start, segment.end)
                    )
            records.append(
                (
                    name,
                    node.name,
                    str(node.meta.get("space")),
                    node.meta.get("slot_count"),
                    node.meta.get("slot_stride"),
                    node.meta.get("memory_instance"),
                )
            )
            if "target_invoke" in node.meta:
                from voyager_compiler.artifacts import to_data

                records.append(
                    (name, node.name, to_data(node.meta["target_invoke"]))
                )
            if "bank_layout" in node.meta:
                from voyager_compiler.artifacts import to_data

                records.append(
                    (
                        name,
                        node.name,
                        to_data(node.meta["bank_layout"]),
                        node.meta.get("bank_entry"),
                        node.meta.get("bank_ranges"),
                    )
                )
            for key in (
                "physical_component",
                "physical_shape",
                "buffer_layout_key",
                "instruction_constant",
                "target_schedule",
                "target_layout",
            ):
                if key in node.meta:
                    from voyager_compiler.artifacts import to_data

                    records.append(
                        (name, node.name, key, to_data(node.meta[key]))
                    )
    return hashlib.sha256(
        json.dumps(records, sort_keys=True).encode()
    ).hexdigest()


def analyze_values(graph):
    """Derive required aliasing from operator schemas; do not assign storage.

    A reshape is not assumed to alias: its schema permits an alias but may
    allocate. Keep that distinction visible to later bufferization.
    """
    relations = []
    for scope, module in graph.named_modules():
        if not hasattr(module, "graph"):
            continue
        for node in module.graph.nodes:
            schema = getattr(node.target, "_schema", None)
            if schema is None:
                continue
            returns = set().union(
                *(
                    r.alias_info.after_set
                    for r in schema.returns
                    if r.alias_info
                )
            )
            for index, argument in enumerate(schema.arguments):
                alias = argument.alias_info
                if alias is None or not returns.intersection(alias.after_set):
                    continue
                source = (
                    node.args[index]
                    if index < len(node.args)
                    else node.kwargs.get(argument.name)
                )
                if hasattr(source, "name"):
                    kind = "write_alias" if alias.is_write else "may_alias"
                    relations.append(
                        ValueRelation(
                            f"{scope}/{node.name}",
                            f"{scope}/{source.name}",
                            kind,
                        )
                    )
    return ValueRelations(graph_digest(graph), tuple(relations))


# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Leaf-level predicates shared by storage analysis and code emission."""


from dataclasses import dataclass
from enum import Enum

import torch
from torch.fx import Node


class StorageEffectKind(str, Enum):
    """Physical-storage effect of one otherwise generic FX operator."""

    MATERIALIZE = "materialize"
    COPY = "copy"
    VIEW_OR_COPY = "view_or_copy"
    ALIAS_OR_MATERIALIZE = "alias_or_materialize"
    SPLIT_VIEW = "split_view"


class AddressRule(str, Enum):
    """Rule proving that an FX result can reuse its source address."""

    NEVER = "never"
    ALWAYS = "always"
    CONTIGUOUS_RESHAPE = "contiguous_reshape"
    AS_STRIDED = "as_strided"
    IDENTITY_INDEX = "identity_index"
    SELECT = "select"
    SAME_DTYPE = "same_dtype"
    INFERENCE_DROPOUT = "inference_dropout"


@dataclass(frozen=True)
class StorageOpTraits:
    """One source of truth for generic storage and view semantics."""

    effect: StorageEffectKind
    address_rule: AddressRule = AddressRule.NEVER
    forwards_aggregate_writes: bool = False


_MATERIALIZE = StorageOpTraits(StorageEffectKind.MATERIALIZE)
_STORAGE_OP_TRAITS = {
    torch.ops.aten.clone.default: StorageOpTraits(StorageEffectKind.COPY),
    torch.ops.aten.lift_fresh_copy.default: StorageOpTraits(
        StorageEffectKind.COPY
    ),
    torch.ops.aten.chunk.default: StorageOpTraits(StorageEffectKind.SPLIT_VIEW),
    torch.ops.aten.split.Tensor: StorageOpTraits(StorageEffectKind.SPLIT_VIEW),
    torch.ops.aten.split_with_sizes.default: StorageOpTraits(
        StorageEffectKind.SPLIT_VIEW
    ),
    torch.ops.aten.detach.default: StorageOpTraits(
        StorageEffectKind.VIEW_OR_COPY,
        AddressRule.ALWAYS,
        True,
    ),
    torch.ops.aten.detach_.default: StorageOpTraits(
        StorageEffectKind.VIEW_OR_COPY,
        AddressRule.ALWAYS,
        True,
    ),
    torch.ops.aten.squeeze.dim: StorageOpTraits(
        StorageEffectKind.VIEW_OR_COPY,
        AddressRule.ALWAYS,
        True,
    ),
    torch.ops.aten.squeeze.dims: StorageOpTraits(
        StorageEffectKind.VIEW_OR_COPY,
        AddressRule.ALWAYS,
        True,
    ),
    torch.ops.aten.unsqueeze.default: StorageOpTraits(
        StorageEffectKind.VIEW_OR_COPY,
        AddressRule.ALWAYS,
        True,
    ),
    torch.ops.aten.view.default: StorageOpTraits(
        StorageEffectKind.VIEW_OR_COPY,
        AddressRule.ALWAYS,
        True,
    ),
    torch.ops.aten.contiguous.default: StorageOpTraits(
        StorageEffectKind.VIEW_OR_COPY,
        AddressRule.CONTIGUOUS_RESHAPE,
        True,
    ),
    torch.ops.aten.flatten.using_ints: StorageOpTraits(
        StorageEffectKind.VIEW_OR_COPY,
        AddressRule.CONTIGUOUS_RESHAPE,
        True,
    ),
    torch.ops.aten.reshape.default: StorageOpTraits(
        StorageEffectKind.VIEW_OR_COPY,
        AddressRule.CONTIGUOUS_RESHAPE,
        True,
    ),
    torch.ops.aten.as_strided.default: StorageOpTraits(
        StorageEffectKind.VIEW_OR_COPY,
        AddressRule.AS_STRIDED,
    ),
    torch.ops.aten.expand.default: StorageOpTraits(
        StorageEffectKind.VIEW_OR_COPY,
        AddressRule.IDENTITY_INDEX,
        True,
    ),
    torch.ops.aten.slice.Tensor: StorageOpTraits(
        StorageEffectKind.VIEW_OR_COPY,
        AddressRule.IDENTITY_INDEX,
        True,
    ),
    torch.ops.aten.select.int: StorageOpTraits(
        StorageEffectKind.VIEW_OR_COPY,
        AddressRule.SELECT,
        True,
    ),
    # Voyager has no general strided-segment representation, so these remain
    # logical views but require physical copies.
    torch.ops.aten.permute.default: StorageOpTraits(
        StorageEffectKind.VIEW_OR_COPY
    ),
    torch.ops.aten.transpose.int: StorageOpTraits(
        StorageEffectKind.VIEW_OR_COPY
    ),
    torch.ops.aten.to.dtype: StorageOpTraits(
        StorageEffectKind.ALIAS_OR_MATERIALIZE,
        AddressRule.SAME_DTYPE,
    ),
    torch.ops.aten.dropout.default: StorageOpTraits(
        StorageEffectKind.ALIAS_OR_MATERIALIZE,
        AddressRule.INFERENCE_DROPOUT,
    ),
}


def storage_op_traits(target) -> StorageOpTraits:
    """Return generic storage semantics for ``target``."""
    return _STORAGE_OP_TRAITS.get(target, _MATERIALIZE)


def forwards_aggregate_writes(target) -> bool:
    """Return whether a full-coverage result can forward aggregate writes."""
    return storage_op_traits(target).forwards_aggregate_writes


def node_value(node: Node):
    """Return the propagated tensor value attached to an FX node."""
    value = getattr(node, "value", None)
    if value is not None:
        return value
    return node.meta.get("val")


def _source_node(source) -> Node | None:
    if isinstance(source, Node):
        return source
    return getattr(source, "node", None)


def _normalized_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return str(dtype).removeprefix("torch.")
    if isinstance(dtype, str):
        return dtype.removeprefix("torch.")
    return dtype


@dataclass(frozen=True)
class SelectView:
    """Normalized parameters for one tensor-valued ``aten.select``."""

    source: Node
    shape: tuple[int, ...]
    dim: int
    index: int


def select_view(node: Node) -> SelectView | None:
    """Parse and bounds-check positional or keyword ``aten.select`` args."""
    if node.target != torch.ops.aten.select.int or not node.args:
        return None
    source = _source_node(node.args[0])
    if source is None:
        return None
    shape = getattr(source, "shape", None)
    if shape is None:
        value = node_value(source)
        shape = getattr(value, "shape", None)
    if shape is None:
        return None
    shape = tuple(int(size) for size in shape)
    dim = int(node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0))
    if dim < 0:
        dim += len(shape)
    if dim < 0 or dim >= len(shape):
        return None
    index_value = (
        node.args[2] if len(node.args) > 2 else node.kwargs.get("index")
    )
    if index_value is None:
        return None
    index = int(index_value)
    if index < 0:
        index += shape[dim]
    if index < 0 or index >= shape[dim]:
        return None
    return SelectView(source, shape, dim, index)


def is_zero_copy_storage_op(node: Node, source=None) -> bool:
    """Return whether the operation can reuse its input's physical address.

    Unknown layout metadata is deliberately treated as copy-producing.
    Keeping this predicate in a leaf module lets relation analysis and
    instruction emission share the exact decision without an import cycle.
    """
    rule = storage_op_traits(node.target).address_rule
    if rule is AddressRule.ALWAYS:
        return True
    if rule is AddressRule.NEVER:
        return False

    source_node = _source_node(source)
    if source_node is None:
        source_node = next(iter(node.all_input_nodes), None)
    if source_node is None:
        return False

    if rule is AddressRule.SAME_DTYPE:
        force_copy = (
            node.args[3]
            if len(node.args) > 3
            else node.kwargs.get("copy", False)
        )
        memory_format = (
            node.args[4]
            if len(node.args) > 4
            else node.kwargs.get("memory_format")
        )
        if force_copy or memory_format is not None:
            return False
        input_dtype = getattr(source_node, "dtype", None)
        if input_dtype is None:
            input_value = node_value(source_node)
            if isinstance(input_value, torch.Tensor):
                input_dtype = input_value.dtype
        output_dtype = (
            node.args[1] if len(node.args) > 1 else node.kwargs.get("dtype")
        )
        return _normalized_dtype(input_dtype) == _normalized_dtype(output_dtype)

    if rule is AddressRule.INFERENCE_DROPOUT:
        training = (
            node.args[2]
            if len(node.args) > 2
            else node.kwargs.get("train", True)
        )
        return training is False

    if rule is AddressRule.CONTIGUOUS_RESHAPE:
        input_value = node_value(source_node)
        output_value = node_value(node)
        if not all(
            isinstance(value, torch.Tensor)
            for value in (input_value, output_value)
        ):
            return False
        if input_value.numel() != output_value.numel():
            return False
        if node.target != torch.ops.aten.contiguous.default:
            return input_value.is_contiguous() and output_value.is_contiguous()
        memory_format = (
            node.args[1]
            if len(node.args) > 1
            else node.kwargs.get(
                "memory_format",
                torch.contiguous_format,
            )
        )
        if memory_format is None:
            memory_format = torch.contiguous_format
        try:
            return input_value.is_contiguous(
                memory_format=memory_format,
            )
        except (TypeError, RuntimeError):
            # Unknown or unsupported formats must materialize rather than
            # silently reuse storage with the wrong physical strides.
            return False

    if rule is AddressRule.AS_STRIDED:
        input_value = node_value(source_node)
        output_value = node_value(node)
        if not all(
            isinstance(value, torch.Tensor)
            for value in (input_value, output_value)
        ):
            return False
        storage_offset = (
            node.args[3]
            if len(node.args) > 3
            else node.kwargs.get("storage_offset")
        )
        if storage_offset is None:
            storage_offset = input_value.storage_offset()
        return (
            int(storage_offset) == int(input_value.storage_offset())
            and input_value.numel() == output_value.numel()
            and input_value.is_contiguous()
            and output_value.is_contiguous()
        )

    if rule is AddressRule.IDENTITY_INDEX:
        return is_identity_indexing_op(node)

    if rule is AddressRule.SELECT:
        selection = select_view(node)
        if selection is None:
            return False
        # A singleton selected dimension aliases the complete payload. When
        # all preceding dimensions are singleton, the existing byte-offset
        # lowering can also represent a nonzero selection.
        return selection.shape[selection.dim] == 1 or all(
            size == 1 for size in selection.shape[: selection.dim]
        )

    raise AssertionError(f"Unhandled storage address rule: {rule}")


def is_identity_indexing_op(node: Node) -> bool:
    """Return whether an indexing operation preserves the complete value."""
    if node.target == torch.ops.aten.slice.Tensor:
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        start = node.args[2] if len(node.args) > 2 else node.kwargs.get("start")
        end = node.args[3] if len(node.args) > 3 else node.kwargs.get("end")
        step = (
            node.args[4] if len(node.args) > 4 else node.kwargs.get("step", 1)
        )
        if (start is not None and start != 0) or step != 1:
            return False
        source = node.args[0] if node.args else None
        shape = getattr(source, "shape", None)
        if end is not None and shape is not None:
            return end >= shape[dim]
        return (start is None and end is None) or end == 0x7FFFFFFFFFFFFFFF

    if node.target == torch.ops.aten.expand.default:
        sizes = (
            node.args[1] if len(node.args) > 1 else node.kwargs.get("size", ())
        )
        return bool(sizes) and all(size in (1, -1) for size in sizes)

    return False


def is_address_only_view(node: Node, source=None) -> bool:
    """Return whether Voyager can represent a view without a new backing.

    This is deliberately narrower than PyTorch alias semantics. Operations
    such as a partial slice, transpose, or permute may return an eager tensor
    view, but Voyager currently serializes them as executable operations and
    has no general strided-segment representation. Those operations therefore
    need a distinct physical result in the compiler memory plan.
    """
    traits = storage_op_traits(node.target)
    return (
        traits.effect is StorageEffectKind.VIEW_OR_COPY
        and is_zero_copy_storage_op(node, source)
    )


__all__ = [
    "AddressRule",
    "StorageEffectKind",
    "StorageOpTraits",
    "forwards_aggregate_writes",
    "is_address_only_view",
    "is_identity_indexing_op",
    "is_zero_copy_storage_op",
    "node_value",
    "select_view",
    "SelectView",
    "storage_op_traits",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Graph helpers for Alpha's inference lowering; no compiler pipeline."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from voyager_compiler.codegen.node_info import get_arg_value
from voyager_compiler.codegen.node_info import is_conv2d as _is_conv2d
from voyager_compiler.codegen.node_info import is_gemm_op as _is_gemm_op
from voyager_compiler.codegen.node_info import is_matmul as _is_matmul
from voyager_compiler.codegen.node_info import is_nop as is_nop
from voyager_compiler.codegen.node_info import is_reshape_op as is_reshape_op
from voyager_compiler.export_utils import (
    create_getattr_from_value as _create_getattr,
)
from voyager_compiler.export_utils import export_model as export_model
from voyager_compiler.shape_prop import fetch_attr as fetch_attr
from voyager_compiler.shape_prop import propagate_shape as propagate_shape

COMPILER_ONLY = "compiler_only"
COMPILATION_TARGET_META = "target"
MAGNET_ALPHA_COMPILATION_TARGET = "magnet-alpha"
SOURCE_OP_NAME = "source_op_name"
TARGET_INSTRUCTION_NOP = "target_instruction_nop"


def is_conv2d(node, operator_traits=None):
    return _is_conv2d(node) or str(node.target) in (
        "alpha_quant.conv2d_mx.default",
        "magnet_alpha.conv2d_vsq.default",
        "magnet_alpha.conv2d_int8_accumulator.default",
    )


def is_gemm_op(node, operator_traits=None):
    return _is_gemm_op(node) or is_conv2d(node)


def is_matmul(node, operator_traits=None):
    return _is_matmul(node)


def new_attr_name(module, prefix):
    prefix = prefix.replace(".", "_")
    name, index = prefix, 0
    while hasattr(module, name):
        index += 1
        name = f"{prefix}_{index}"
    return name


def create_getattr_from_value(
    module, graph, prefix, value, *, compiler_only=False
):
    node = _create_getattr(module, graph, prefix, value)
    if compiler_only:
        node.meta[COMPILER_ONLY] = True
    return node


def declare_unsupported_operation(node, reason):
    raise NotImplementedError(f"{node.name}: {reason}")


def is_indexing_or_concatenation_op(node):
    return node.target in (
        torch.ops.aten.cat.default,
        torch.ops.aten.stack.default,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.select.int,
    )


def _slice_source_and_shapes(node: torch.fx.Node):
    if node.target != torch.ops.aten.slice.Tensor:
        return None
    source = node.args[0]
    if not isinstance(source, torch.fx.Node):
        return None

    def node_shape(value):
        shape = getattr(value, "shape", None)
        if shape is None:
            tensor_meta = value.meta.get("tensor_meta")
            shape = getattr(tensor_meta, "shape", None)
        if shape is None:
            shape = getattr(value.meta.get("val"), "shape", ())
        return tuple(shape)

    source_shape = node_shape(source)
    result_shape = node_shape(node)
    if not source_shape or len(source_shape) != len(result_shape):
        return None
    return source, source_shape, result_shape


def get_identity_slice_source(node: torch.fx.Node):
    """Return the source when an Aten slice preserves the complete shape."""
    slice_info = _slice_source_and_shapes(node)
    if slice_info is None:
        return None
    source, source_shape, result_shape = slice_info
    step = get_arg_value(node, 4, "step", 1)
    return source if step == 1 and result_shape == source_shape else None


def get_leading_channel_slice_source(node: torch.fx.Node):
    """Return the source of a zero-offset rank-four NCHW channel crop."""
    slice_info = _slice_source_and_shapes(node)
    if slice_info is None:
        return None
    source, source_shape, result_shape = slice_info
    dim = get_arg_value(node, 1, "dim", 0)
    start = get_arg_value(node, 2, "start", 0)
    step = get_arg_value(node, 4, "step", 1)
    if dim == -3:
        dim = 1
    if (
        dim != 1
        or start not in (None, 0)
        or step != 1
        or len(source_shape) != 4
        or source_shape[0] != result_shape[0]
        or source_shape[2:] != result_shape[2:]
        or source_shape[1] < result_shape[1]
    ):
        return None
    return source


def update_submod_user_meta(model, node, named_modules=None):
    """
    Update the metadata of all user nodes that consume the given node.
    """
    if named_modules is None:
        named_modules = dict(model.named_modules())

    for user in list(node.users):
        if user.op != "call_module":
            continue

        index = user.all_input_nodes.index(node)

        submod = named_modules[user.target]
        placeholders = [n for n in submod.graph.nodes if n.op == "placeholder"]

        assert index < len(placeholders)

        placeholder = placeholders[index]
        placeholder.name = node.name
        placeholder.meta["source_node"] = node


def _freeze_attribute(value, field_name: str):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_attribute(item, field_name) for item in value)
    raise TypeError(f"{field_name} must contain only scalar or tuple values")


@dataclass(frozen=True)
class OperationClassification:
    """Target-owned operation identity and frozen scalar facts."""

    kind: str
    attributes: tuple[tuple[str, object], ...] | Mapping = ()

    def __post_init__(self):
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("operation kind must be non-empty")
        items = (
            self.attributes.items()
            if isinstance(self.attributes, Mapping)
            else self.attributes
        )
        attributes = tuple(
            sorted(
                (
                    (
                        str(name),
                        _freeze_attribute(
                            value,
                            f"operation attribute {name!r}",
                        ),
                    )
                    for name, value in items
                ),
                key=lambda item: item[0],
            )
        )
        names = tuple(name for name, _value in attributes)
        if len(set(names)) != len(names):
            raise ValueError("operation attribute names must be unique")
        object.__setattr__(self, "attributes", attributes)

    def attribute(self, name: str, default=None):
        """Return one frozen scalar fact."""
        return next(
            (
                value
                for attribute_name, value in self.attributes
                if attribute_name == name
            ),
            default,
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import collections.abc
from itertools import repeat
from typing import Any, Optional

import torch
import torch.nn as nn

__all__ = [
    "get_hardware_logical_shape",
    "get_arg_value",
    "get_conv_bn_layers",
    "get_identity_slice_source",
    "get_leading_channel_slice_source",
    "get_named_arg",
    "inherit_hardware_logical_shape",
    "preserve_hardware_logical_shape",
]


HARDWARE_LOGICAL_SHAPE = "hardware_logical_shape"


def preserve_hardware_logical_shape(node: torch.fx.Node) -> tuple:
    """Record a tensor's semantic shape before hardware-only padding.

    Matrix padding may mutate a ``get_attr`` tensor in place.  The resulting
    FX value is the physical parameter consumed by the accelerator, while
    converters and validation still need the original logical extents.  Keep
    that distinction on the value itself before the first mutation.
    """
    shape = tuple(getattr(node, "shape", ()) or ())
    previous = node.meta.setdefault(HARDWARE_LOGICAL_SHAPE, shape)
    previous = tuple(previous)
    if len(previous) != len(shape):
        raise ValueError(
            f"{node.name}: hardware logical rank changed from "
            f"{len(previous)} to {len(shape)}"
        )
    return previous


def get_hardware_logical_shape(node: torch.fx.Node) -> tuple:
    """Return the pre-padding semantic shape, or the current logical shape."""
    return tuple(
        node.meta.get(
            HARDWARE_LOGICAL_SHAPE,
            getattr(node, "shape", ()) or (),
        )
    )


def inherit_hardware_logical_shape(
    target: torch.fx.Node,
    source: torch.fx.Node,
) -> tuple:
    """Carry a source's semantic shape onto a padded replacement value."""
    logical_shape = get_hardware_logical_shape(source)
    target.meta[HARDWARE_LOGICAL_SHAPE] = logical_shape
    return logical_shape


def get_arg_value(
    node: torch.fx.Node,
    arg_number: int,
    kwarg_name: Optional[str] = None,
    default=None,
) -> Any:
    return (
        node.args[arg_number]
        if len(node.args) > arg_number
        else node.kwargs.get(kwarg_name, default)  # type: ignore[arg-type]
    )


def get_named_arg(node: torch.fx.Node, name: str):
    """Return one schema-named FX operand, independent of call spelling."""
    if name in node.kwargs:
        return node.kwargs[name]

    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return None
    for index, argument in enumerate(schema.arguments):
        if argument.name == name and index < len(node.args):
            return node.args[index]
    return None


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


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_pair = _ntuple(2, "_pair")


def get_conv_bn_layers(model):
    layers = []
    module_names = list(model._modules)
    for k, name in enumerate(module_names):
        if len(list(model._modules[name]._modules)) > 0:
            conv_bn_pairs = get_conv_bn_layers(model._modules[name])
            layers.extend(
                [
                    [f"{name}.{conv}", f"{name}.{bn}"]
                    for conv, bn in conv_bn_pairs
                ]
            )
        elif isinstance(model._modules[name], nn.BatchNorm2d) and isinstance(
            model._modules[module_names[k - 1]], nn.Conv2d
        ):
            layers.append([module_names[k - 1], name])
    return layers

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Magnet Alpha graph-pattern recognition before generic lowering."""

import logging
from dataclasses import dataclass

import torch
from torch.fx import GraphModule, Node

from voyager_compiler.compiler.analysis import is_address_only_view
from voyager_compiler.ir.graph import (
    QUANTIZATION_DOMAIN_DTYPE,
    QUANTIZATION_REGION,
    get_physical_dtype,
    get_physical_quantization_contract,
    inherit_physical_quantization_contract,
)
from voyager_compiler.targets.magnet_alpha.fx import (
    TARGET_INSTRUCTION_NOP,
    declare_unsupported_operation,
    get_arg_value,
    propagate_shape,
    update_submod_user_meta,
)
from voyager_compiler.targets.magnet_alpha.hardware import (
    ALPHA_OUTPUT_LANES,
    MagnetAlphaHardwareConfig,
)
from voyager_compiler.targets.magnet_alpha.operators import (
    CHANNEL_SHUFFLE_COPY_OPERATOR,
    CHANNEL_SHUFFLE_OPERATOR,
    INT8_CONV2D_ACCUMULATOR_OPERATOR,
    PPU_FIXED_RELU_OPERATOR,
    SATURATING_ADD_OPERATOR,
    VSQ_CONV2D_OPERATOR,
)

logger = logging.getLogger(__name__)


ALPHA_SHUFFLE_COPY_KIND = "channel_shuffle_copy"


_TRANSPARENT_TENSOR_META = (
    "dtype",
    "nn_module_stack",
    QUANTIZATION_DOMAIN_DTYPE,
    QUANTIZATION_REGION,
)


@dataclass(frozen=True)
class ShuffleCopyMovement:
    """One physical channel-block movement after a two-way shuffle."""

    index: int
    source_output: int
    source_entry: int
    destination_entry: int


def magnet_alpha_meta(node: Node):
    return node.meta.setdefault("magnet_alpha", {})


def is_shuffle_copy(node: Node) -> bool:
    return node.target == CHANNEL_SHUFFLE_COPY_OPERATOR


def shuffle_copy_movements(
    channel_blocks: int,
) -> tuple[ShuffleCopyMovement, ...]:
    """Return the in-place copy sequence for a two-way channel shuffle."""
    if channel_blocks < 2 or channel_blocks % 2:
        raise ValueError(
            "Alpha shuffle requires a positive even channel-block count"
        )

    half = channel_blocks // 2
    movements = []
    copy_index = 1
    for source_entry in range(half - 1, 0, -1):
        movements.append(
            ShuffleCopyMovement(
                index=copy_index,
                source_output=0,
                source_entry=source_entry,
                destination_entry=2 * source_entry,
            )
        )
        copy_index += 1

    for source_entry in range(half - 1):
        movements.append(
            ShuffleCopyMovement(
                index=copy_index,
                source_output=1,
                source_entry=source_entry,
                destination_entry=2 * source_entry + 1,
            )
        )
        copy_index += 1
    return tuple(movements)


def _shuffle_copy_count(
    shuffle: Node,
    block_channels: int,
) -> int:
    shape = getattr(shuffle, "shape", None)
    if shape is None or len(shape) != 4:
        raise ValueError(
            f"Magnet Alpha shuffle {shuffle.name} requires a rank-4 "
            "output shape"
        )

    channels = shape[1]
    groups = magnet_alpha_meta(shuffle).get("groups")
    if not isinstance(channels, int) or channels <= 0:
        raise ValueError(
            f"Magnet Alpha shuffle {shuffle.name} has invalid channels: "
            f"{channels}"
        )
    if groups != 2:
        raise ValueError(
            f"Magnet Alpha shuffle {shuffle.name} requires two groups: {groups}"
        )
    if channels % block_channels:
        raise ValueError(
            f"Magnet Alpha shuffle {shuffle.name} channels must be "
            "divisible by "
            f"{block_channels}: {channels}"
        )

    channel_blocks = channels // block_channels
    return len(shuffle_copy_movements(channel_blocks))


def _inherit_tensor_contract(target: Node, source: Node) -> None:
    """Preserve one encoded tensor's domain across transparent movers."""
    for key in _TRANSPARENT_TENSOR_META:
        if key in source.meta:
            target.meta[key] = source.meta[key]
    inherit_physical_quantization_contract(target, source)


def _int8_dtype(dtype) -> bool:
    return dtype in {"int8", torch.int8} or str(dtype) in {
        "int8",
        "torch.int8",
    }


def _static_pair(value) -> tuple[int, int] | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return (value, value)
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in value
        )
    ):
        return tuple(value)
    return None


def _physical_padded_maxpool_geometry(pool: Node, block_channels: int):
    """Return one Alpha-compatible static padded max-pool geometry."""
    if pool.target != torch.ops.aten.max_pool2d.default:
        return None
    source = get_arg_value(pool, 0, "self")
    if not isinstance(source, Node):
        return None
    input_shape = _positive_static_nchw(source)
    output_shape = _positive_static_nchw(pool)
    kernel = _static_pair(get_arg_value(pool, 1, "kernel_size"))
    stride = _static_pair(get_arg_value(pool, 2, "stride", kernel))
    padding = _static_pair(get_arg_value(pool, 3, "padding", 0))
    dilation = _static_pair(get_arg_value(pool, 4, "dilation", 1))
    if (
        input_shape is None
        or output_shape is None
        or kernel is None
        or stride is None
        or padding is None
        or dilation is None
        or min(*kernel, *stride, *dilation) <= 0
        or min(*padding) < 0
        or not any(padding)
        or get_arg_value(pool, 5, "ceil_mode", False) is not False
        or not _int8_dtype(get_physical_dtype(source))
        or not _int8_dtype(get_physical_dtype(pool))
    ):
        return None
    if input_shape[:2] != output_shape[:2]:
        raise ValueError(f"{pool.name}: max-pool must preserve N and C")
    expected_output = tuple(
        (
            input_shape[axis + 2]
            + 2 * padding[axis]
            - dilation[axis] * (kernel[axis] - 1)
            - 1
        )
        // stride[axis]
        + 1
        for axis in range(2)
    )
    if output_shape[2:] != expected_output:
        raise ValueError(
            f"{pool.name}: static max-pool output {output_shape[2:]} "
            f"does not match {expected_output}"
        )
    if block_channels <= 0 or input_shape[1] % block_channels:
        raise ValueError(
            f"{pool.name}: padded Alpha max-pool channels "
            f"{input_shape[1]} must be divisible by NMP width "
            f"{block_channels}"
        )
    physical_shape = (
        input_shape[0],
        input_shape[1],
        input_shape[2] + 2 * padding[0],
        input_shape[3] + 2 * padding[1],
    )
    return (
        source,
        output_shape,
        physical_shape,
        kernel,
        stride,
        dilation,
        padding,
        input_shape[1] // block_channels,
    )


def legalize_physical_padded_maxpools(
    model: GraphModule,
    hardware: MagnetAlphaHardwareConfig,
) -> GraphModule:
    """Materialize Alpha pool borders and emit one call per NMP C block."""
    if not isinstance(hardware, MagnetAlphaHardwareConfig):
        raise TypeError(
            "Magnet Alpha max-pool legalization requires "
            "MagnetAlphaHardwareConfig"
        )
    block_channels = hardware.nmp_lanes(8)
    graph = model.graph
    rewritten = 0
    for pool in tuple(graph.nodes):
        geometry = _physical_padded_maxpool_geometry(pool, block_channels)
        if geometry is None:
            continue
        (
            source,
            output_shape,
            physical_shape,
            kernel,
            stride,
            dilation,
            padding,
            tile_count,
        ) = geometry
        blocked_shape = (
            physical_shape[0],
            tile_count,
            block_channels,
            physical_shape[2],
            physical_shape[3],
        )
        tiled_outputs = []
        with graph.inserting_before(pool):
            padded = graph.call_function(
                torch.ops.aten.constant_pad_nd.default,
                (
                    source,
                    (padding[1], padding[1], padding[0], padding[0]),
                    -128,
                ),
            )
            padded.name = f"{pool.name}_input_padded"
            propagate_shape(padded, model)
            _inherit_tensor_contract(padded, source)
            if tuple(padded.shape) != physical_shape:
                raise ValueError(
                    f"{pool.name}: physical pad produced "
                    f"{tuple(padded.shape)}, expected {physical_shape}"
                )
            declare_unsupported_operation(
                padded,
                reason_code="alpha_maxpool_spatial_pad_materialization",
                detail=(
                    "Alpha maxpool requires an external INT8 spatial-pad "
                    "materialization before zero-padding pool calls"
                ),
                stage="magnet_alpha_maxpool_legalization",
                facts={
                    "logical_input_shape": tuple(source.shape),
                    "physical_input_shape": tuple(padded.shape),
                    "spatial_padding": tuple(padding),
                    "fill_value": -128,
                },
            )
            blocked = graph.call_function(
                torch.ops.aten.reshape.default,
                (padded, blocked_shape),
            )
            blocked.name = f"{pool.name}_input_blocked"
            propagate_shape(blocked, model)
            _inherit_tensor_contract(blocked, padded)
            for tile_index in range(tile_count):
                tile_input = graph.call_function(
                    torch.ops.aten.select.int,
                    (blocked, 1, tile_index),
                )
                tile_input.name = f"{pool.name}_input_tile{tile_index}"
                propagate_shape(tile_input, model)
                _inherit_tensor_contract(tile_input, padded)
                tiled = graph.call_function(
                    pool.target,
                    (tile_input, kernel, stride, (0, 0), dilation, False),
                )
                tiled.name = f"{pool.name}_tile{tile_index}"
                propagate_shape(tiled, model)
                _inherit_tensor_contract(tiled, pool)
                tiled.meta["source_fn_stack"] = [
                    ("magnet_alpha_maxpool_multicall", tiled.target)
                ]
                tiled.meta["magnet_alpha"] = {
                    "kind": "physical_padded_maxpool_multicall",
                    "multicall_parent": pool.name,
                    "tile_count": tile_count,
                    "tile_index": tile_index,
                }
                tiled_outputs.append(tiled)
            stacked = graph.call_function(
                torch.ops.aten.stack.default,
                (tuple(tiled_outputs), 1),
            )
            stacked.name = f"{pool.name}_tiles"
            propagate_shape(stacked, model)
            _inherit_tensor_contract(stacked, pool)
            stacked.meta[TARGET_INSTRUCTION_NOP] = True
            reconstructed = graph.call_function(
                torch.ops.aten.reshape.default,
                (stacked, output_shape),
            )
            reconstructed.name = f"{pool.name}_reconstructed"
            propagate_shape(reconstructed, model)
            _inherit_tensor_contract(reconstructed, pool)
        pool.replace_all_uses_with(reconstructed)
        update_submod_user_meta(model, reconstructed)
        graph.erase_node(pool)
        rewritten += 1
    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    logger.info(
        "Legalized %d physically padded Alpha max-pools",
        rewritten,
    )
    return model


def apply_memory_rewrites(
    model: GraphModule,
    hardware: MagnetAlphaHardwareConfig,
) -> GraphModule:
    """Declare target multi-calls and copies before storage analysis."""
    if not isinstance(hardware, MagnetAlphaHardwareConfig):
        raise TypeError(
            "Magnet Alpha memory rewrites require MagnetAlphaHardwareConfig"
        )
    model = legalize_physical_padded_maxpools(model, hardware)
    block_channels = hardware.nmp_lanes(8)
    graph = model.graph
    inserted = 0

    for shuffle in list(graph.nodes):
        if shuffle.target != CHANNEL_SHUFFLE_OPERATOR:
            continue
        if any(is_shuffle_copy(user) for user in shuffle.users):
            continue

        copy_count = _shuffle_copy_count(shuffle, block_channels)
        if copy_count == 0:
            continue

        original_users = list(shuffle.users)
        previous = shuffle
        for copy_index in range(1, copy_count + 1):
            with graph.inserting_after(previous):
                copy_node = graph.create_node(
                    "call_function",
                    CHANNEL_SHUFFLE_COPY_OPERATOR,
                    (previous,),
                    name=f"{shuffle.name}_cpy{copy_index}",
                )

            for meta_key in ("dtype", "nn_module_stack"):
                if meta_key in previous.meta:
                    copy_node.meta[meta_key] = previous.meta[meta_key]
            copy_node.meta["physical_copy"] = True
            copy_node.meta["magnet_alpha"] = {
                "block_channels": block_channels,
                "copy_index": copy_index,
                "kind": ALPHA_SHUFFLE_COPY_KIND,
                "logical_parent": shuffle.name,
            }
            propagate_shape(copy_node, model)
            previous = copy_node
            inserted += 1

        for user in original_users:
            user.replace_input_with(shuffle, previous)

    graph.lint()
    model.recompile()
    logger.info("Inserted %d Magnet Alpha shuffle-copy operations", inserted)
    return model


def _resolved_view_shape(node: Node):
    """Return an exact view shape, resolving PyTorch's single ``-1``."""
    declared = get_arg_value(node, 1, "size")
    if not isinstance(declared, (tuple, list)):
        return None
    declared = tuple(declared)
    if not all(
        isinstance(extent, int) and not isinstance(extent, bool)
        for extent in declared
    ):
        return None
    if -1 not in declared:
        return declared
    if declared.count(-1) != 1:
        return None

    inferred = getattr(node, "shape", None)
    if inferred is None:
        inferred = getattr(getattr(node, "value", None), "shape", None)
    if inferred is None:
        inferred = getattr(node.meta.get("val"), "shape", None)
    if inferred is None:
        inferred = getattr(
            node.meta.get("tensor_meta"),
            "shape",
            None,
        )
    if inferred is None:
        return None
    inferred = tuple(inferred)
    if len(inferred) != len(declared) or not all(
        isinstance(extent, int) and not isinstance(extent, bool) and extent >= 0
        for extent in inferred
    ):
        return None
    if any(
        expected != -1 and expected != actual
        for expected, actual in zip(declared, inferred)
    ):
        return None
    return inferred


def _match_channel_shuffle(output: Node):
    """Match one exclusive two-input channel-shuffle spine."""
    view_targets = {
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
    }
    if output.target not in view_targets:
        return None

    contiguous = output.args[0]
    if not isinstance(contiguous, Node) or (
        contiguous.target != torch.ops.aten.contiguous.default
    ):
        return None

    transpose = contiguous.args[0]
    if not isinstance(transpose, Node) or (
        transpose.target != torch.ops.aten.transpose.int
    ):
        return None
    if (
        get_arg_value(transpose, 1, "dim0") != 1
        or get_arg_value(transpose, 2, "dim1") != 2
    ):
        return None

    grouped_view = transpose.args[0]
    if (
        not isinstance(grouped_view, Node)
        or grouped_view.target not in view_targets
    ):
        return None

    grouped_shape = _resolved_view_shape(grouped_view)
    if grouped_shape is None or len(grouped_shape) != 5:
        return None
    groups = grouped_shape[1]
    if groups != 2:
        return None
    output_shape = _resolved_view_shape(output)
    if output_shape is None:
        return None
    if len(output_shape) != 4:
        return None
    expected_output_shape = (
        grouped_shape[0],
        grouped_shape[2] * groups,
        grouped_shape[3],
        grouped_shape[4],
    )
    if tuple(output_shape) != expected_output_shape:
        return None

    cat = grouped_view.args[0]
    if not isinstance(cat, Node):
        return None
    if cat.target != torch.ops.aten.cat.default:
        return None
    if get_arg_value(cat, 1, "dim", 0) != 1:
        return None

    inputs = get_arg_value(cat, 0, "tensors")
    if (
        not isinstance(inputs, (tuple, list))
        or len(inputs) != 2
        or not all(isinstance(node, Node) for node in inputs)
    ):
        return None

    spine = (cat, grouped_view, transpose, contiguous, output)
    if any(
        set(node.users) != {spine[index + 1]}
        for index, node in enumerate(spine[:-1])
    ):
        return None

    return tuple(inputs), groups, spine


def _match_channel_shuffle_cat(cat: Node):
    """Return the complete shuffle match rooted at ``cat``, if present."""
    if cat.target != torch.ops.aten.cat.default:
        return None

    output = cat
    for _ in range(4):
        if len(output.users) != 1:
            return None
        output = next(iter(output.users))

    match = _match_channel_shuffle(output)
    if match is None or match[2][0] is not cat:
        return None
    return match


def is_channel_shuffle_cat(node: Node) -> bool:
    """Return whether ``node`` starts one complete Alpha shuffle spine."""
    return _match_channel_shuffle_cat(node) is not None


def _channel_shuffle_name(output: Node) -> str:
    """Preserve a model-scoped name when export metadata provides one."""
    module_scope = None
    for path, module_type in output.meta.get("nn_module_stack", {}).values():
        if "ShuffleV2Block" in str(module_type):
            module_scope = path
    if module_scope is None:
        return f"{output.name}_shuffle"
    if module_scope.startswith("L['self']."):
        module_scope = module_scope[len("L['self'].") :]
    return f"{module_scope.replace('.', '_')}_shuffle"


def collapse_channel_shuffles(model: GraphModule) -> GraphModule:
    """Replace structural shuffle spines with Alpha logical operations."""
    graph = model.graph
    rewritten = 0

    for cat in list(graph.nodes):
        match = _match_channel_shuffle_cat(cat)
        if match is None:
            continue

        inputs, groups, spine = match
        output = spine[-1]
        node_name = _channel_shuffle_name(output)

        with graph.inserting_before(output):
            shuffle = graph.create_node(
                "call_function",
                CHANNEL_SHUFFLE_OPERATOR,
                (*inputs, groups),
                name=node_name,
            )

        shuffle.meta.update(output.meta)
        shuffle.meta["source_fn_stack"] = [
            ("magnet_alpha_channel_shuffle", shuffle.target)
        ]
        shuffle.meta["magnet_alpha"] = {
            "groups": groups,
            "kind": "channel_shuffle",
        }

        output.replace_all_uses_with(shuffle)
        for node in reversed(spine):
            graph.erase_node(node)
        rewritten += 1

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()
    logger.info("Collapsed %d channel-shuffle spines", rewritten)
    return model


def _positive_static_shape(
    node: Node,
    rank: int,
) -> tuple[int, ...] | None:
    shape = getattr(node, "shape", None)
    if shape is None:
        tensor_meta = node.meta.get("tensor_meta")
        shape = getattr(tensor_meta, "shape", None)
    if shape is None:
        value = node.meta.get("val", getattr(node, "value", None))
        shape = getattr(value, "shape", None)
    if shape is None or len(shape) != rank:
        return None
    shape = tuple(shape)
    if not all(
        isinstance(extent, int) and not isinstance(extent, bool) and extent > 0
        for extent in shape
    ):
        return None
    return shape


def _positive_static_nchw(node: Node) -> tuple[int, ...] | None:
    return _positive_static_shape(node, 4)


def _exclusive_pe_output(node: Node, user: Node) -> bool:
    """Return whether ``node`` is one complete direct PE output chain."""
    if node.target != PPU_FIXED_RELU_OPERATOR or set(node.users) != {user}:
        return False
    accumulator = get_arg_value(node, 0, "input")
    if not isinstance(accumulator, Node) or accumulator.target not in {
        INT8_CONV2D_ACCUMULATOR_OPERATOR,
        VSQ_CONV2D_OPERATOR,
    }:
        return False
    return (
        set(accumulator.users) == {node}
        and _positive_static_nchw(node) is not None
        and get_arg_value(node, 5, "integer_accumulator") is True
        and get_arg_value(node, 12, "output_precision") == 8
        and get_arg_value(node, 13, "relu") is False
    )


def _exclusive_eadd_pe_output(node: Node, eadd: Node) -> bool:
    """Return whether ``node`` is EADD's exclusive direct PE output."""
    return _exclusive_pe_output(node, eadd)


def _exclusive_channel_cat(node: Node, eadd: Node) -> bool:
    """Return whether ``node`` is a private channel concatenation."""
    if (
        node.target != torch.ops.aten.cat.default
        or set(node.users) != {eadd}
        or get_arg_value(node, 1, "dim", 0) != 1
    ):
        return False
    inputs = get_arg_value(node, 0, "tensors")
    if (
        not isinstance(inputs, (tuple, list))
        or len(inputs) < 2
        or not all(isinstance(value, Node) for value in inputs)
    ):
        return False
    output_shape = _positive_static_nchw(node)
    tile_shapes = tuple(_positive_static_nchw(value) for value in inputs)
    return (
        output_shape is not None
        and output_shape[0] == 1
        and all(shape is not None for shape in tile_shapes)
        and all(set(value.users) == {node} for value in inputs)
        and all(
            shape[0] == 1 and shape[2:] == output_shape[2:]
            for shape in tile_shapes
        )
        and sum(shape[1] for shape in tile_shapes) == output_shape[1]
    )


def _exclusive_stacked_channel_aggregate(node: Node, eadd: Node) -> bool:
    """Match the exact batch-one K-tile stack/reshape reconstruction."""
    if node.target != torch.ops.aten.reshape.default or set(node.users) != {
        eadd
    }:
        return False
    stacked = get_arg_value(node, 0, "self")
    if (
        not isinstance(stacked, Node)
        or stacked.target != torch.ops.aten.stack.default
        or set(stacked.users) != {node}
        or get_arg_value(stacked, 1, "dim", 0) != 0
    ):
        return False
    tiles = get_arg_value(stacked, 0, "tensors")
    if (
        not isinstance(tiles, (tuple, list))
        or len(tiles) < 2
        or not all(isinstance(value, Node) for value in tiles)
        or not all(set(value.users) == {stacked} for value in tiles)
    ):
        return False

    output_shape = _positive_static_nchw(node)
    stacked_shape = _positive_static_shape(stacked, 5)
    tile_shapes = tuple(_positive_static_nchw(value) for value in tiles)
    if (
        output_shape is None
        or output_shape[0] != 1
        or stacked_shape is None
        or any(shape is None for shape in tile_shapes)
        or any(shape != tile_shapes[0] for shape in tile_shapes[1:])
    ):
        return False
    tile_shape = tile_shapes[0]
    return (
        tile_shape[0] == 1
        and stacked_shape == (len(tiles), *tile_shape)
        and output_shape
        == (
            1,
            len(tiles) * tile_shape[1],
            tile_shape[2],
            tile_shape[3],
        )
    )


def _exclusive_channel_aggregate(node: Node, eadd: Node) -> bool:
    """Return whether ``node`` is one exact private K reconstruction."""
    return _exclusive_channel_cat(
        node,
        eadd,
    ) or _exclusive_stacked_channel_aggregate(node, eadd)


@dataclass(frozen=True)
class _KTileAggregate:
    """One exact private stack/reshape of contiguous PE K tiles."""

    output: Node
    stacked: Node
    tiles: tuple[Node, ...]
    channel_ranges: tuple[tuple[int, int], ...]


def _same_optional_meta(nodes: tuple[Node, ...], key: str) -> bool:
    present = tuple(key in node.meta for node in nodes)
    if any(present) and not all(present):
        return False
    if not any(present):
        return True
    first = nodes[0].meta[key]
    return all(node.meta[key] == first for node in nodes[1:])


def _compatible_optional_meta(
    anchor: Node,
    nodes: tuple[Node, ...],
    key: str,
) -> bool:
    """Reject explicit metadata that conflicts with ``anchor``."""
    if key not in anchor.meta:
        values = tuple(node.meta[key] for node in nodes if key in node.meta)
        return not values or all(value == values[0] for value in values[1:])
    return all(
        key not in node.meta or node.meta[key] == anchor.meta[key]
        for node in nodes
    )


def _same_physical_int8_format(nodes: tuple[Node, ...]) -> bool:
    """Require one explicit physical INT8 payload format."""
    contracts = tuple(
        get_physical_quantization_contract(node) for node in nodes
    )
    return (
        all(contract is not None for contract in contracts)
        and all(contract == contracts[0] for contract in contracts[1:])
        and all(_int8_dtype(get_physical_dtype(node)) for node in nodes)
    )


def _same_physical_int8_domain(nodes: tuple[Node, ...]) -> bool:
    """Require one explicit physical INT8 format and input domain."""
    return (
        _same_physical_int8_format(nodes)
        and _same_optional_meta(nodes, QUANTIZATION_DOMAIN_DTYPE)
        and _same_optional_meta(nodes, QUANTIZATION_REGION)
    )


def _exact_k_tiled_pe_aggregate(
    node: Node,
    eadd: Node,
) -> _KTileAggregate | None:
    """Match exclusive equal-width PE tiles covering one full NCHW C."""
    if not _exclusive_stacked_channel_aggregate(node, eadd):
        return None
    stacked = get_arg_value(node, 0, "self")
    tiles = tuple(get_arg_value(stacked, 0, "tensors"))
    output_shape = _positive_static_nchw(node)
    if output_shape is None:
        return None

    channel_extent = output_shape[1]
    channel_ranges = []
    expected_start = 0
    tile_width = None
    for tile in tiles:
        if not _exclusive_pe_output(tile, stacked):
            return None
        accumulator = get_arg_value(tile, 0, "input")
        channel_range = accumulator.meta.get("output_channel_tile")
        full_extent = accumulator.meta.get("output_channel_extent")
        if (
            not isinstance(channel_range, (tuple, list))
            or len(channel_range) != 2
            or not all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in channel_range
            )
            or full_extent != channel_extent
        ):
            return None
        start, end = channel_range
        shape = _positive_static_nchw(tile)
        width = end - start
        if (
            shape is None
            or start != expected_start
            or end <= start
            or end > channel_extent
            or width != shape[1]
            or (tile_width is not None and width != tile_width)
            or start % ALPHA_OUTPUT_LANES
            or end % ALPHA_OUTPUT_LANES
        ):
            return None
        tile_width = width
        expected_start = end
        channel_ranges.append((start, end))

    if expected_start != channel_extent or tile_width is None:
        return None
    if (
        get_physical_quantization_contract(node) is None
        or not _int8_dtype(get_physical_dtype(node))
        or not all(_int8_dtype(get_physical_dtype(tile)) for tile in tiles)
        or not _compatible_optional_meta(
            node,
            tiles,
            QUANTIZATION_DOMAIN_DTYPE,
        )
        or not _compatible_optional_meta(node, tiles, QUANTIZATION_REGION)
    ):
        return None
    return _KTileAggregate(
        node,
        stacked,
        tiles,
        tuple(channel_ranges),
    )


def _returns_alias(node: Node) -> bool:
    schema = getattr(node.target, "_schema", None)
    returns = () if schema is None else schema.returns
    return bool(returns and returns[0].alias_info is not None)


def _safe_tiled_eadd_destination(
    destination: Node,
    eadd: Node,
    aggregate: _KTileAggregate,
    positions: dict[Node, int],
) -> bool:
    """Prove the earlier disjoint writes cannot clobber another reader."""
    output_shape = _positive_static_nchw(aggregate.output)
    if (
        output_shape is None
        or _positive_static_nchw(destination) != output_shape
        or destination.op not in {"placeholder", "call_function"}
        or destination is aggregate.output
        or positions.get(destination, positions[aggregate.stacked])
        >= positions[aggregate.stacked]
        or _returns_alias(destination)
        or not _same_physical_int8_domain((aggregate.output, destination))
        or not _same_physical_int8_format((aggregate.output, eadd))
    ):
        return False

    write_position = positions[aggregate.stacked]
    pending = [destination]
    visited = set()
    while pending:
        value = pending.pop()
        if value in visited:
            continue
        visited.add(value)
        for user in value.users:
            if user is eadd:
                continue
            if is_address_only_view(user, value):
                pending.append(user)
                continue
            if positions.get(user, write_position) >= write_position:
                return False
    return True


def _distribute_k_tiled_eadd(
    model: GraphModule,
    eadd: Node,
    aggregate: _KTileAggregate,
    destination: Node,
) -> None:
    """Move one full-channel EADD into each exclusive PE K tile."""
    graph = model.graph
    tiled_outputs = []
    output_shape = _positive_static_nchw(aggregate.output)
    assert output_shape is not None
    tile_width = aggregate.channel_ranges[0][1]
    blocked_shape = (
        output_shape[0],
        len(aggregate.tiles),
        tile_width,
        output_shape[2],
        output_shape[3],
    )
    with graph.inserting_before(aggregate.stacked):
        blocked_destination = graph.call_function(
            torch.ops.aten.reshape.default,
            (destination, blocked_shape),
        )
        blocked_destination.name = f"{eadd.name}_input2_blocked"
        propagate_shape(blocked_destination, model)
        _inherit_tensor_contract(blocked_destination, destination)

        for index, (tile, (start, end)) in enumerate(
            zip(
                aggregate.tiles,
                aggregate.channel_ranges,
            )
        ):
            destination_tile = graph.call_function(
                torch.ops.aten.select.int,
                (blocked_destination, 1, index),
            )
            destination_tile.name = f"{eadd.name}_input2_channels_{start}_{end}"
            propagate_shape(destination_tile, model)
            _inherit_tensor_contract(destination_tile, destination)

            tiled_eadd = graph.call_function(
                SATURATING_ADD_OPERATOR,
                (tile, destination_tile),
            )
            tiled_eadd.name = f"{eadd.name}_channels_{start}_{end}"
            _inherit_tensor_contract(tiled_eadd, eadd)
            propagate_shape(tiled_eadd, model)
            magnet_alpha_meta(tiled_eadd).update(
                {
                    "kind": "k_tiled_eadd",
                    "multicall_parent": eadd.name,
                    "output_channel_tile": (start, end),
                }
            )
            tiled_outputs.append(tiled_eadd)

    aggregate.stacked.update_arg(0, tuple(tiled_outputs))
    propagate_shape(aggregate.stacked, model)
    _inherit_tensor_contract(aggregate.stacked, eadd)
    assert tuple(aggregate.stacked.shape) == (
        len(aggregate.tiles),
        output_shape[0],
        tile_width,
        output_shape[2],
        output_shape[3],
    )
    propagate_shape(aggregate.output, model)
    _inherit_tensor_contract(aggregate.output, eadd)
    eadd.replace_all_uses_with(aggregate.output)
    graph.erase_node(eadd)


def canonicalize_eadd_execution_ports(model: GraphModule) -> GraphModule:
    """Make proved post-K-tiling EADDs executable, then canonicalize ports.

    A private stack/reshape of complete PE K tiles is distributed into
    disjoint mutable channel views of the other operand.  Remaining EADDs are
    commuted only when an exclusive direct PE output can own the operation.
    """
    distributed = 0
    commuted = 0
    for eadd in tuple(model.graph.nodes):
        if (
            eadd.target != SATURATING_ADD_OPERATOR
            or len(eadd.args) != 2
            or eadd.kwargs
        ):
            continue
        input1, input2 = eadd.args
        if not all(isinstance(value, Node) for value in (input1, input2)):
            continue

        aggregates = tuple(
            (index, aggregate)
            for index, value in enumerate((input1, input2))
            if (aggregate := _exact_k_tiled_pe_aggregate(value, eadd))
            is not None
        )
        if len(aggregates) == 1:
            index, aggregate = aggregates[0]
            destination = (input2, input1)[index]
            positions = {
                node: position
                for position, node in enumerate(model.graph.nodes)
            }
            if _safe_tiled_eadd_destination(
                destination,
                eadd,
                aggregate,
                positions,
            ):
                # Alpha's operation-local allocator proves these disjoint
                # K-major writes against one resident output backing.  Only
                # that target-owned proof permits eliding the stack itself.
                aggregate.stacked.meta[TARGET_INSTRUCTION_NOP] = True
                _distribute_k_tiled_eadd(
                    model,
                    eadd,
                    aggregate,
                    destination,
                )
                distributed += len(aggregate.tiles)
                continue

        if _exclusive_eadd_pe_output(input1, eadd):
            continue
        input1_shape = _positive_static_nchw(input1)
        input2_shape = _positive_static_nchw(input2)
        if not (
            _exclusive_channel_aggregate(input1, eadd)
            and _exclusive_eadd_pe_output(input2, eadd)
            and input1_shape == input2_shape
        ):
            continue
        eadd.args = (input2, input1)
        commuted += 1

    model.graph.lint()
    model.graph.eliminate_dead_code()
    model.recompile()
    logger.info(
        "Distributed %d Alpha K-tile EADD calls and commuted %d owners",
        distributed,
        commuted,
    )
    return model


def apply_graph_rewrites(model: GraphModule) -> GraphModule:
    """Run Magnet Alpha recognition before generic graph lowering."""
    return collapse_channel_shuffles(model)


__all__ = [
    "ALPHA_SHUFFLE_COPY_KIND",
    "ShuffleCopyMovement",
    "apply_graph_rewrites",
    "apply_memory_rewrites",
    "canonicalize_eadd_execution_ports",
    "collapse_channel_shuffles",
    "is_shuffle_copy",
    "is_channel_shuffle_cat",
    "legalize_physical_padded_maxpools",
    "magnet_alpha_meta",
    "shuffle_copy_movements",
]

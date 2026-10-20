"""Fold quantize / dequantize nodes into the ops around them.

Two passes.  ``split_kv_cache`` runs on the exported decode graph before
quantization: it splits each KV cache into the completed, block-aligned
chunks (which the quantizer then sees as one tensor) and a small
full-precision residual holding the chunk being filled.
``fuse_quantize_dequantize_with_producer`` runs inside ``transform()`` over
an already-quantized graph and moves a quantize or dequantize rather than
compute it: it hoists a quantize into its producer (so a value is stored
already narrow), replays it above a relayout, folds one (a ``quantize_mx``
or a group-wise ``quantize_affine``) into a KV cache write, and collapses a
``get_attr -> dequantize -> layout ops -> quantize`` chain into a single
``dequantize`` with pre-multiplied scales.

Both work purely on the ``quantized_ops`` schema and FX, not on the
quantizer that produced the graph.
"""

import copy
import logging
import operator
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import map_arg

from voyager_compiler.codegen.aten_classifier import is_compute_op
from voyager_compiler.codegen.node_info import (
    _BROADCAST_OPS,
    QUANTIZE_FAMILY_OPS,
    get_arg_value,
    is_gemm_op,
    is_mha_qkv_permute,
    is_nop,
    is_reshape_op,
)
from voyager_compiler.codegen.subgraph import get_new_node_name_with_prefix
from voyager_compiler.export_utils import create_getattr_from_value
from voyager_compiler.ops import repeat_per_block
from voyager_compiler.shape_prop import (
    constant_value,
    fetch_attr,
    propagate_shape,
    set_node_value,
)

logger = logging.getLogger(__name__)

__all__ = [
    "fuse_quantize_dequantize_with_producer",
    "quantizes_kv_cache",
    "sink_cache_folds",
    "split_kv_cache",
]


_SLICE = torch.ops.aten.slice.Tensor
_EXPAND = torch.ops.aten.expand.default
_UNSQUEEZE = torch.ops.aten.unsqueeze.default
_RESHAPE = torch.ops.aten.reshape.default
_INDEX_COPY = torch.ops.aten.index_copy_.default
_MATMUL = torch.ops.aten.matmul.default
_COND = torch.ops.higher_order.cond
_SDPA = torch.ops.aten.scaled_dot_product_attention.default
_SDPA_MX = torch.ops.quantized_ops.sdpa_mx.default
_ATTENTION_OPS = (_SDPA, _SDPA_MX)
_DEQUANTIZE = torch.ops.quantized_ops.dequantize.default
_QUANTIZE = torch.ops.quantized_ops.quantize.default
_QUANTIZE_MX = torch.ops.quantized_ops.quantize_mx.default
_QUANTIZE_AFFINE = torch.ops.quantized_ops.quantize_affine.default
_QUANTIZE_MX_OUTLIER = torch.ops.quantized_ops.quantize_mx_outlier.default

_HOISTABLE_OPS = (
    _SLICE,
    torch.ops.aten.select.int,
    _EXPAND,
    torch.ops.aten.repeat.default,
)

# Ops that regroup dims without moving an element: they preserve row-major
# order, so a quantize can be lifted over one even when it cuts across the axis
# the quantize blocks along -- as long as the blocks come out the same
# (``_blocks_survive_regroup``).  These are also the only ops whose size
# argument ``_replay_relayout`` knows how to rebuild for the scale.
_REGROUP_OPS = (
    _RESHAPE,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
)

_REPEAT_OPS = (_UNSQUEEZE, _EXPAND, *_REGROUP_OPS)

_SQUEEZE = torch.ops.aten.squeeze.default

# What a quantize or a dequantize may cross (``_map_block_axes``): ops that
# re-address a tensor's elements without changing any, and the slices,
# selects and repeats a cache or GQA read takes of them.
_RELAYOUT_OPS = (
    *_REPEAT_OPS,
    _SQUEEZE,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.dims,
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.unflatten.int,
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
    torch.ops.aten.movedim.int,
    torch.ops.aten.swapaxes.default,
    torch.ops.aten.swapdims.default,
    torch.ops.aten.alias.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.contiguous.default,
    torch.ops.aten.lift_fresh_copy.default,
    _SLICE,
    torch.ops.aten.select.int,
    torch.ops.aten.repeat.default,
)

# MHA head splitting rejoins the per-head results with one of these, so a
# quantize lifted over it has to be duplicated onto every branch.
_FORK_OPS = (
    torch.ops.aten.stack.default,
    torch.ops.aten.cat.default,
)

# A dynamic quantize returns its qparams beside the value, the value last:
# ``quantize_mx`` gives ``(scale, value)``, ``quantize_affine`` ``(scale,
# zero_point, value)``.  The value inherits the relayout ops the quantize was
# lifted over; each qparam gets a replayed copy of them.  A cache baked through
# one becomes a buffer per output, named ``<cache>_<suffix>``.
_OUTPUT_NAMES = {
    _QUANTIZE_MX: ("scale", "full"),
    _QUANTIZE_AFFINE: ("scale", "zero_point", "full"),
}

# The KV-cache buffers of a Hugging Face static-cache decode export, and the
# residual buffers ``split_kv_cache`` puts beside them.
_KV_CACHE = re.compile(r"^(key|value)_cache_(\d+)$")


def _is_relayout(node) -> bool:
    return (
        isinstance(node, Node)
        and (
            is_nop(node) or is_reshape_op(node) or node.target in _HOISTABLE_OPS
        )
        and len(node.all_input_nodes) == 1
    )


def _on_meta(node: Node):
    """``(input, output)`` of ``node``'s op run on an empty contiguous input
    of its input's shape, on the ``meta`` device."""
    x = torch.empty(_shape_of(node.args[0]), device="meta")
    return x, node.target(x, *node.args[1:], **node.kwargs)


def _map_block_axes(
    node: Node, axes, block_size: Optional[int], from_output=True
) -> Optional[Tuple[int, ...]]:
    """Map a blocked quantization's axes across op ``node``.

    The quantization groups ``block_size`` consecutive elements along each of
    ``axes`` (one size for all; ``None`` means one group per index, as per
    channel).  With ``from_output``, ``axes`` are on ``node``'s output and are
    mapped to its input; otherwise the reverse.  The block size itself does
    not change.  A reshape may regroup a blocked dim when its size on both
    sides is a multiple of ``block_size``; a slice may cut one, mapping to the
    output only, when it starts and ends on block boundaries.

    Returns:
        The mapped axes, counted from the end (-1 is the last dim), or None if
        ``node`` splits a block.
    """
    if node.target not in _RELAYOUT_OPS or len(node.all_input_nodes) != 1:
        return None
    # On a contiguous input, a dim carried whole keeps its size and stride.
    x, y = _on_meta(node)
    src, dst = (y, x) if from_output else (x, y)
    block = block_size or 1
    restated = []
    for a in axes:
        size, stride = src.shape[a], src.stride(a)
        dims = [
            d
            for d in range(dst.ndim)
            if dst.stride(d) == stride and (dst.shape[d] > 1) == (size > 1)
        ]
        if not dims:
            if size == 1:
                continue
            return None
        d = dims[0]
        sizes = sorted((size, dst.shape[d]))
        if sizes[0] != sizes[1]:
            start = y.storage_offset() // stride
            regrouped = (
                node.target in _REGROUP_OPS
                and block_size is not None
                and sizes[0] % block_size == 0
                and sizes[1] % block_size == 0
            )
            cut = (
                not from_output
                and node.target is _SLICE
                and d == a % src.ndim
                and start % block == 0
                and (sizes[0] % block == 0 or start + sizes[0] == sizes[1])
            )
            if not (regrouped or cut):
                return None
        restated.append(d - dst.ndim)
    return tuple(restated)


def _relayout_path(
    src,
    axes: Tuple[int, ...],
    block_size: Optional[int],
    keep_head_permute=False,
):
    """Walk up from ``src`` over relayout ops that keep the blocks along
    ``axes`` whole (``_map_block_axes``) and have no other reader.

    Returns ``(src, path, src_axes)``: the first node not crossed, the ops
    crossed (nearest the start first) and the block axes at ``src``.  With
    ``keep_head_permute`` it also stops at an MHA head permute, where a
    multi-output quantize fuses with the GEMM that stores through it.
    """
    path = []
    while len(src.users) == 1:
        if keep_head_permute and is_mha_qkv_permute(src):
            break
        above = _map_block_axes(src, axes, block_size)
        if above is None:
            break
        path.append(src)
        src, axes = src.args[0], above
    return src, path, axes


def _copy_quantize_above(model: GraphModule, node: Node, src: Node, axes):
    """A copy of quantize ``node`` reading ``src``, inserted right after it."""

    graph = model.graph
    remap = {node.args[0]: src}
    with graph.inserting_before(src.next):
        for n in node.all_input_nodes:
            if n not in remap:
                remap[n] = graph.node_copy(n)
        new = graph.node_copy(node, lambda n: remap[n])

    if node.target is _QUANTIZE_MX:
        args = list(new.args)
        args[2] = list(axes)
        new.args = tuple(args)

    for n in list(remap.values()) + [new]:
        propagate_shape(n, model)
    new.meta = {
        k: copy.deepcopy(v) if k != "val" else v.clone()
        for k, v in node.meta.items()
    }
    return new


def _replay_relayout(
    graph: Graph, node: Node, src: Node, extents: dict
) -> Node:
    """Copy relayout ``node`` onto ``src``, resized to ``extents``.

    ``extents`` maps an axis of ``node``'s output to its extent on ``src``.  A
    reshape or view restates its shape with them, and so does a bare squeeze,
    as a reshape, since it would also drop a dim ``src`` holds one block of.
    An expand names only the dims it grows (``-1`` keeps one), so it needs
    none.
    """
    new = graph.node_copy(node, lambda n: src if n is node.args[0] else n)
    out_shape = _shape_of(node)
    in_shape = _shape_of(node.args[0])
    if node.target in _REGROUP_OPS or node.target is _SQUEEZE:
        shape = list(out_shape)
        for a, extent in extents.items():
            shape[a] = extent
        new.target = _RESHAPE if node.target is _SQUEEZE else node.target
        new.args = (src, shape)
    elif node.target is _EXPAND:
        lead = len(out_shape) - len(in_shape)
        new.args = (
            src,
            [
                o if k < lead or o != in_shape[k - lead] else -1
                for k, o in enumerate(out_shape)
            ],
        )
    return new


def _replay_path(model, src, path, axes, block_size: int, dtype) -> Node:
    """``src``, a per-block qparam, carried through relayouts ``path``
    (oldest first; ``axes`` are the block axes at its end).  A block axis
    takes its extent over ``block_size``, rounded up, as
    ``_reshape_to_blocks`` pads a partial block.
    """
    axes_at = [axes]  # at each relayout's output
    for n in reversed(path[1:]):
        axes_at.insert(0, _map_block_axes(n, axes_at[0], block_size))
    for n, n_axes in zip(path, axes_at):
        shape = _shape_of(n)
        extents = {a: -(-shape[a] // block_size) for a in n_axes}
        src = _replay_relayout(model.graph, n, src, extents)
        _typed(model, src, dtype)
    return src


def _annotate(path, source: Node) -> None:
    """The relayout ops now sit *below* the quantize, so they carry the dtype of
    the tensor that flows through them -- or none at all, once a dequantize has
    put it back in the clear.  ``source`` is that tensor: the quantize itself,
    or, for a multi-output one, the single output the path was rewired onto (not
    the op, whose ``dtype`` is the pair it returns).
    """
    is_dequantize = source.target is _DEQUANTIZE
    for n in path:
        if is_dequantize:
            n.meta.pop("dtype", None)
        else:
            n.meta["dtype"] = source.meta.get("dtype", None)


def _hoist_forked(model: GraphModule, node: Node) -> List[Node]:
    """Lift a single-output quantize over the relayout ops feeding it.

    A ``stack`` / ``cat`` on the way up (MHA head splitting rejoining its heads)
    forks the walk: every branch is quantized on its own, and the concat then
    joins pieces that are already quantized.

    Returns:
        The copies of ``node`` it put in its place, empty if it moved none.
    """
    graph = model.graph
    on_path, copies = [], []
    todo = [(node.args[0], node)]  # (tensor to lift over, the node reading it)

    while todo:
        start, reader = todo.pop()
        src, path, _ = _relayout_path(start, (), None)
        on_path.extend(path)
        if path:
            reader = path[-1]

        if src.target in _FORK_OPS and len(src.users) == 1:
            on_path.append(src)
            todo.extend((a, src) for a in src.all_input_nodes)
            continue

        if not path and reader is node:
            continue  # already sitting on its producer

        new = _copy_quantize_above(model, node, src, ())
        reader.replace_input_with(src, new)
        copies.append(new)

    if not copies:
        return copies
    for n in sorted(dict.fromkeys(on_path)):
        propagate_shape(n, model)
    _annotate(on_path, node)
    node.replace_all_uses_with(node.args[0])
    graph.erase_node(node)
    return copies


def _writes_buffer(node) -> bool:
    """Whether ``node`` is an ``index_copy_`` into a module buffer."""
    return (
        isinstance(node, Node)
        and node.target is _INDEX_COPY
        and isinstance(node.args[0], Node)
        and node.args[0].op == "get_attr"
    )


def _cache_write_below(node: Node):
    """``(index_copy_, prelude)`` for the KV-cache write ``node`` quantizes, or
    ``(None, [])``.  ``prelude`` is the ops between them, nearest ``node``
    first: a ``pad`` widening the cache to the array.  They apply to the cache
    as a whole, so they can be applied to the buffer and to the token written
    into it instead of to every position, every step."""
    prelude = []
    curr = node.args[0]
    while isinstance(curr, Node) and curr.target is torch.ops.aten.pad.default:
        prelude.append(curr)
        curr = curr.args[0]

    if not _writes_buffer(curr):
        return None, []
    return curr, prelude


def _shape_of(node: Node):
    """``node``'s tensor shape, from ``ShapeProp``'s value or, on a graph
    fresh from export, the fake value export stamped."""
    value = getattr(node, "value", None)
    if not isinstance(value, torch.Tensor):
        value = node.meta["val"]
    return tuple(value.shape)


def _typed(model: GraphModule, node: Node, dtype) -> Node:
    """``node``, typed ``dtype`` and shape-propagated through ``model``."""
    node.meta["dtype"] = dtype
    propagate_shape(node, model)
    return node


def _map_dim_to_output(node: Node, dim: int) -> Optional[int]:
    """Map input dim ``dim`` of op ``node`` to the output dim that holds it.

    A dim merged with inner dims maps to the merged dim; a dim split into
    several maps to the first of them.  A slice over the full range keeps it.

    Returns:
        The output dim, counted from the start, or None if ``node`` removes
        ``dim`` (squeeze, select), cuts it (a slice along it) or merges it as
        an inner factor.
    """
    if len(node.all_input_nodes) != 1:
        return None
    # On a contiguous input a dim spans stride x size elements, and so does
    # the output dim holding it, alone or as its outer factor.
    x, y = _on_meta(node)
    extent = x.stride(dim) * x.shape[dim]
    for d in range(y.ndim):
        if y.shape[d] > 1 and y.stride(d) * y.shape[d] == extent:
            return d
    return None


def _attention_cone(idx: Node):
    """``(relayouts, consumer)``: the relayout ops from cache write ``idx``
    down to the one op that reads the cache -- an ``aten.matmul`` as its
    right operand, or an attention node as its key or value -- in graph
    order; ``(None, None)`` if the write reaches anything else."""
    relayouts, frontier, consumers, seen = [], [idx], [], {idx}
    while frontier:
        for user in frontier.pop().users:
            if user in seen:
                continue
            seen.add(user)
            if user.target is _MATMUL or user.target in _ATTENTION_OPS:
                consumers.append(user)
            elif _is_relayout(user):
                relayouts.append(user)
                frontier.append(user)
            else:
                return None, None
    if len(consumers) != 1:
        return None, None
    consumer = consumers[0]
    operands = consumer.args[1:3] if consumer.target is not _MATMUL else ()
    operands = operands or (consumer.args[1],)
    if not any(a in seen for a in operands):
        return None, None
    return sorted(relayouts), consumer


def _stamp_fake(node: Node, fake_mode, like: Optional[Node] = None) -> None:
    """Give a node built after export the metadata export would have stamped.

    ``meta["val"]`` is the op run on its operands' fake values, under the
    graph's own fake mode; the quantizer's annotation and observer insertion
    read it.  With ``like``, the node also takes the module scope ``like`` was
    traced from, so the quantizer's scope- and order-based filters see it
    beside ``like``.
    """
    for key in ("nn_module_stack", "source_fn_stack"):
        if like is not None and key in like.meta:
            node.meta[key] = like.meta[key]
    if node.op == "get_attr":
        value = fetch_attr(node.graph.owning_module, node.target)
        node.meta["val"] = fake_mode.from_tensor(value, static_shapes=True)
        return

    def load(a):
        return map_arg(a, lambda n: n.meta["val"])

    with fake_mode:
        node.meta["val"] = node.target(*load(node.args), **load(node.kwargs))


def _fold_chunk(model, cache, dim, idx, offsets, pred, after, fake_mode):
    """After the attention reads the main cache, copy the residual into it on
    the step that completes a chunk.  Spelled without a branch, as a
    write-back of the residual or of what the chunk already holds, so the
    graph stays plain tensor ops; the lowering makes it a conditional.

    Args:
        model: The graph module.
        cache: The main cache's ``get_attr``.
        dim: The cache's position axis.
        idx: The residual write, whose value is the residual buffer.
        offsets: The chunk's positions in the cache.
        pred: Whether this step completes the chunk.
        after: The node that reads the main cache; the fold follows it.
        fake_mode: The graph's fake mode, for ``_stamp_fake``.
    """
    graph = model.graph
    with graph.inserting_before(after.next):
        handle = graph.get_attr(cache.target)
        _stamp_fake(handle, fake_mode)
        window = graph.call_function(
            torch.ops.aten.index_select.default, (handle, dim, offsets)
        )
        folded = graph.call_function(
            torch.ops.aten.where.self, (pred, idx, window)
        )
        fold = graph.call_function(_INDEX_COPY, (handle, dim, offsets, folded))
        for n in (window, folded, fold):
            _stamp_fake(n, fake_mode, idx)


def _additive_mask(graph, mask, dtype, fake_mode):
    """A bool mask as the additive one the attention kernel adds to its
    scores: zero where the mask keeps, the dtype's minimum elsewhere."""
    zeros = graph.call_function(
        torch.ops.aten.zeros_like.default, (mask,), {"dtype": dtype}
    )
    dropped = graph.call_function(torch.ops.aten.logical_not.default, (mask,))
    filled = graph.call_function(
        torch.ops.aten.masked_fill.Scalar,
        (zeros, dropped, torch.finfo(dtype).min),
    )
    for n in (zeros, dropped, filled):
        _stamp_fake(n, fake_mode)
    return filled


def _split_attention_operand(
    model, node, cache, idx, relayouts, offsets, fake_mode
):
    """Hand attention ``node`` the residual of the cache it reads as its key
    or value: the operand becomes a view of the cache buffer -- a GQA
    repeat between the two is dropped for ``enable_gqa``, so the kernel
    folds the head group; a view rather than the buffer itself, so the
    quantizer's convert sees an activation and emits the dynamic quantize
    the fold pass bakes into the cache, not a parameter it would quantize
    once -- and the residual write ``idx`` goes onto its ``sdpa_mx`` twin
    as ``key_residual`` / ``value_residual``.  The first cache split
    retargets the node and gives it the query as ``query_residual`` (read
    unquantized) and its masks: the main mask with the chunk's window
    dropped, and ``residual_mask``, the window's own slice.  Both are
    additive (``_additive_mask``).  The residual operands are positional,
    as the quantizer requires.

    Returns:
        The attention node, retargeted.

    Raises:
        ValueError: Relayouts other than a GQA repeat sit between the cache
            and the attention.
    """
    graph = model.graph
    position = next(
        i for i in (1, 2) if node.args[i] is cache or node.args[i] in relayouts
    )
    operand = node.args[position]
    residual_position = position + 6  # query_residual leads at 6
    kwargs = dict(node.kwargs)
    if operand is not cache:
        # ``repeat_kv``: unsqueeze -> expand -> reshape, multiplying the
        # head dim and nothing else, head ``h`` reading KV head ``h // g``.
        b, hkv, *rest = _shape_of(cache)
        b_out, heads, *rest_out = _shape_of(operand)
        if (
            any(n.target not in _REPEAT_OPS for n in relayouts)
            or (b_out, rest_out) != (b, rest)
            or heads % hkv
        ):
            raise ValueError(
                f"{cache.target}: {operand} between the cache and {node} is "
                "not a GQA repeat"
            )
        kwargs["enable_gqa"] = True
    args = list(node.args)
    with graph.inserting_before(node):
        args[position] = graph.call_function(
            torch.ops.aten.view.default, (cache, list(_shape_of(cache)))
        )
        _stamp_fake(args[position], fake_mode, node)
    if node.target is _SDPA_MX:
        args[residual_position] = idx
        node.args = tuple(args)
        node.kwargs = kwargs
        _stamp_fake(node, fake_mode)
        return node

    query = args[0]
    dtype = query.meta["val"].dtype
    mask = get_arg_value(node, 3, "attn_mask", None)
    with graph.inserting_before(node):
        # The residual reads the query unquantized: a view of its own, so
        # the observer the quantizer puts on the query does not cover it.
        residual_query = graph.call_function(
            torch.ops.aten.view.default, (query, list(_shape_of(query)))
        )
        _stamp_fake(residual_query, fake_mode, node)
        if mask is None:
            rows, keys = _shape_of(query)[-2], _shape_of(cache)[-2]
            mask = graph.call_function(
                torch.ops.aten.ones.default,
                ((1, 1, rows, keys),),
                {"dtype": torch.bool, "device": query.meta["val"].device},
            )
            _stamp_fake(mask, fake_mode)
        main = graph.call_function(
            torch.ops.aten.index_fill.int_Scalar, (mask, -1, offsets, False)
        )
        window = graph.call_function(
            torch.ops.aten.index_select.default, (mask, -1, offsets)
        )
        _stamp_fake(main, fake_mode)
        _stamp_fake(window, fake_mode)
        main = _additive_mask(graph, main, dtype, fake_mode)
        window = _additive_mask(graph, window, dtype, fake_mode)
        # (query, key, value, attn_mask, dropout_p, is_causal, then the
        # residual's query, key, value and mask)
        dropout_p = get_arg_value(node, 4, "dropout_p", 0.0)
        is_causal = get_arg_value(node, 5, "is_causal", False)
        args = args[:3] + [main, dropout_p, is_causal, residual_query]
        args += [None, None]
        args.append(window)
        args[residual_position] = idx
        kwargs.pop("dropout_p", None)
        kwargs.pop("is_causal", None)
        new = graph.call_function(_SDPA_MX, tuple(args), kwargs)
        _stamp_fake(new, fake_mode, node)
    node.replace_all_uses_with(new)
    graph.erase_node(node)
    return new


def _split_one_cache(
    model: GraphModule,
    idx: Node,
    residual_length: int,
    context_len: int,
    fake_mode,
) -> None:
    """Split the cache written by ``idx`` (see ``split_kv_cache``)."""
    graph = model.graph
    cache, dim, index, token = idx.args
    length = residual_length
    relayouts, matmul = _attention_cone(idx)
    if matmul is None:
        raise ValueError(
            f"{cache.target}: the cache write does not reach a single "
            "attention matmul or attention node through relayout ops"
        )
    dim = dim % len(_shape_of(idx))
    attention = matmul.target in _ATTENTION_OPS

    # Follow the written axis down to the matmul's right operand: reaching
    # its last dim makes the residual's scores replace a window of the main
    # scores; reaching the reduction dim makes them add.  An attention node
    # takes the residual as operands and joins it itself.
    join = None
    if not attention:
        axis_of = {idx: dim}
        for n in relayouts:
            src = next(a for a in n.all_input_nodes if a in axis_of)
            axis_of[n] = _map_dim_to_output(n, axis_of[src])
            if axis_of[n] is None:
                raise ValueError(
                    f"{cache.target}: {n} loses the cache's position axis"
                )
        weight = matmul.args[1]
        weight_rank = len(_shape_of(weight))
        if axis_of[weight] == weight_rank - 1:
            join = "scores"
        elif axis_of[weight] == weight_rank - 2:
            join = "sum"
        else:
            raise ValueError(
                f"{cache.target}: position axis {axis_of[weight]} of the "
                "attention matmul's operand is neither its output nor its "
                "reduction axis"
            )

    # Split the contents: completed chunks stay, zeros above them; the tail
    # moves into the residual buffer.
    contents = fetch_attr(model, cache.target)
    cache_len = contents.shape[dim]
    if cache_len % length:
        raise ValueError(
            f"{cache.target}: {cache_len} positions is not a multiple of "
            f"the residual length {length}"
        )
    done = context_len // length * length
    residual = torch.zeros_like(contents.narrow(dim, 0, length))
    if context_len > done:
        residual.narrow(dim, 0, context_len - done).copy_(
            contents.narrow(dim, done, context_len - done)
        )
    contents.narrow(dim, done, cache_len - done).zero_()

    # The write lands in the residual at ``p mod R``; the attention's main
    # half reads the cache buffer directly.  The residual keeps the cache's
    # name as its prefix, which is what a traffic report files it under.
    # A step writing several positions (a speculative-decoding verification)
    # fills consecutive slots from its first position's and completes the
    # chunk when its last position lands on slot ``R - 1``; it must not
    # cross a chunk boundary.
    tokens = _shape_of(index)[0]
    if tokens > length:
        raise ValueError(
            f"{cache.target}: a step writes {tokens} positions, more than "
            f"the {length}-slot residual holds"
        )
    with graph.inserting_before(idx):
        residual_attr = create_getattr_from_value(
            model, graph, f"{cache.target}_residual", residual
        )
        _stamp_fake(residual_attr, fake_mode)
        new_nodes = []
        first = index
        if tokens > 1:
            first = graph.call_function(
                torch.ops.aten.select.int, (index, 0, 0)
            )
            new_nodes.append(first)
        slot = graph.call_function(
            torch.ops.aten.remainder.Scalar, (first, length)
        )
        chunk = graph.call_function(
            torch.ops.aten.floor_divide.default, (first, length)
        )
        base = graph.call_function(torch.ops.aten.mul.Tensor, (chunk, length))
        arange = graph.call_function(
            torch.ops.aten.arange.default,
            (length,),
            {"dtype": torch.int64, "device": index.meta["val"].device},
        )
        offsets = graph.call_function(torch.ops.aten.add.Tensor, (base, arange))
        new_nodes += [slot, chunk, base, arange, offsets]
        slots = last = slot
        if tokens > 1:
            steps = graph.call_function(
                torch.ops.aten.arange.default,
                (tokens,),
                {"dtype": torch.int64, "device": index.meta["val"].device},
            )
            slots = graph.call_function(
                torch.ops.aten.add.Tensor, (slot, steps)
            )
            last = graph.call_function(
                torch.ops.aten.add.Tensor, (slot, tokens - 1)
            )
            new_nodes += [steps, slots, last]
        pred = graph.call_function(torch.ops.aten.eq.Scalar, (last, length - 1))
        new_nodes.append(pred)
        for n in new_nodes:
            _stamp_fake(n, fake_mode, idx)
    idx.replace_all_uses_with(cache)
    idx.update_arg(0, residual_attr)
    idx.update_arg(2, slots)
    _stamp_fake(idx, fake_mode)

    if attention:
        joined = _split_attention_operand(
            model, matmul, cache, idx, relayouts, offsets, fake_mode
        )
        _fold_chunk(model, cache, dim, idx, offsets, pred, joined, fake_mode)
        logger.info(
            f"Split {cache.target}: {done} positions in completed chunks, "
            f"{context_len - done} in the {length}-slot residual; the "
            "attention takes the residual as an operand"
        )
        return

    # The residual half: the same relayouts and matmul, over the residual's
    # ``R`` positions, right after the main matmul so the quantizer counts
    # them main, residual, main, residual within the layer.  The main half
    # now reads the cache where it read the write.  A matmul that read the
    # write directly (no relayout: multi-head attention's values) keeps it.
    remap = {cache: idx, idx: idx}
    with graph.inserting_before(matmul.next):
        for n in relayouts:
            remap[n] = _replay_relayout(
                graph, n, remap[n.args[0]], {axis_of[n]: length}
            )
            _stamp_fake(remap[n], fake_mode)
        if join == "scores":
            lhs = matmul.args[0]
        else:
            lhs = graph.call_function(
                torch.ops.aten.index_select.default,
                (matmul.args[0], -1, offsets),
            )
            _stamp_fake(lhs, fake_mode, matmul)
        residual_matmul = graph.node_copy(matmul, lambda a: remap.get(a, a))
        residual_matmul.args = (lhs, remap[weight])
        _stamp_fake(residual_matmul, fake_mode)
        if join == "scores":
            joined = graph.call_function(
                torch.ops.aten.index_copy.default,
                (matmul, -1, offsets, residual_matmul),
            )
        else:
            joined = graph.call_function(
                torch.ops.aten.add.Tensor, (matmul, residual_matmul)
            )
        _stamp_fake(joined, fake_mode, matmul)
    matmul.replace_all_uses_with(
        joined, delete_user_cb=lambda u: u is not joined
    )
    _fold_chunk(model, cache, dim, idx, offsets, pred, joined, fake_mode)
    logger.info(
        f"Split {cache.target}: {done} positions in completed chunks, "
        f"{context_len - done} in the {length}-slot residual; residual "
        f"{'replaces a window of' if join == 'scores' else 'adds to'} the "
        "main attention"
    )


def quantizes_kv_cache(model: GraphModule, quantizer) -> bool:
    """Whether ``quantizer`` quantizes the attention's read of a KV cache,
    the condition for ``split_kv_cache``: only a cache stored quantized
    needs a full-precision residual.

    Annotates ``model`` and clears the annotations again; a cache is read
    quantized when an operand that views it -- through its write and
    single-operand relayouts such as a transpose or the GQA repeat -- takes
    a spec.

    Args:
        model: Decode graph from ``convert_and_export_with_cache``, its
            caches not yet split.
        quantizer: The quantizer ``prepare_pt2e`` will annotate ``model``
            with.

    Returns:
        True when a cache operand of an attention op is quantized.
    """
    quantizer.annotate(model)
    annotations = [
        n.meta.pop("quantization_annotation")
        for n in model.graph.nodes
        if "quantization_annotation" in n.meta
    ]
    for annotation in annotations:
        for operand, qspec in annotation.input_qspec_map.items():
            if qspec is None:
                continue
            # A cache is the first input of its write and of each relayout.
            while (
                operand.target is _INDEX_COPY
                or len(operand.all_input_nodes) == 1
            ):
                operand = operand.all_input_nodes[0]
            if _KV_CACHE.match(str(operand.target)):
                return True
    return False


def split_kv_cache(
    model: GraphModule, residual_length: int, context_len: int
) -> int:
    """Split each KV cache of an exported decode graph into its completed
    chunks and a full-precision residual.

    The main cache buffer keeps only whole chunks of ``residual_length``
    positions, zeros above them; a residual buffer of ``residual_length``
    slots, in the cache's dtype, holds the chunk being filled, the token at
    position ``p`` in slot ``p mod residual_length``.  The attention reads
    both: the residual's scores replace the main scores at the chunk's
    positions (``q @ K^T``) and its values add to the main ones (``P @ V``,
    over the same window of ``P``).  After the attention has read the
    cache, chunk ``p // residual_length`` of the main cache is rewritten:
    with the residual on the step that fills its last slot, with what it
    already holds otherwise.  A quantizer then annotates the main cache's
    read and leaves the residual alone; the lowering turns the rewrite
    into a conditional store of the quantized chunk.

    A step may write several consecutive positions -- a speculative-decoding
    verification step -- at most ``residual_length`` of them and never
    across a chunk boundary: they take consecutive residual slots and the
    chunk completes when the last one lands on its last slot.

    Args:
        model: The decode graph from ``convert_and_export_with_cache``, its
            cache buffers loaded with ``context_len`` positions; rewritten
            in place.
        residual_length: Positions per chunk, a multiple of the quantizer's
            block size along the sequence.
        context_len: Positions already written in each cache.

    Returns:
        The number of caches split.

    Raises:
        ValueError: A cache is not a multiple of ``residual_length`` long,
            its write does not reach a single attention matmul, or a step
            writes more positions than the residual holds.
    """
    graph = model.graph
    fake_mode = next(
        n.meta["val"].fake_mode
        for n in graph.nodes
        if isinstance(n.meta.get("val"), FakeTensor)
    )
    count = 0
    for idx in list(graph.nodes):
        if (
            not _writes_buffer(idx)
            or _KV_CACHE.match(str(idx.args[0].target)) is None
        ):
            continue
        _split_one_cache(model, idx, residual_length, context_len, fake_mode)
        count += 1
    graph.eliminate_dead_code()
    graph.lint()
    model.recompile()
    return count


def _fold_quantize_into_cache(model: GraphModule, node: Node) -> Optional[Node]:
    """Fold a dynamic quantize over a KV cache write into the cache itself.

    The write puts one token in; the quantize then sweeps all of it, every
    step.  Since each token's blocks are its own, quantizing at write time is
    the same arithmetic -- so the buffer is baked already quantized and the
    quantize moves onto the token the write carries.  The quantize returns its
    qparams beside the value, so the one cache buffer becomes one per output
    (``_OUTPUT_NAMES``), each with its own write.  A ``pad`` above the write
    folds in the same way: the buffer is baked wide, and only the token still
    pays for it.

    Only when the blocked axis is not the one the write indexes.  Otherwise a
    token lands mid-block and its block's qparams depend on tokens not yet
    written -- that is what ``split_kv_cache``'s residual is for.

    Returns:
        The quantize the token now takes, or ``None`` if nothing folded.
    """

    graph = model.graph
    idx, prelude = _cache_write_below(node)
    if idx is None:
        return None

    if any(u.target is not operator.getitem for u in node.users):
        return None
    outs = {u.args[1]: u for u in node.users}

    rank = len(node.args[0].value.shape)
    dim = idx.args[1] % rank
    if any(a % rank == dim for a in node.args[2]):
        # Blocked along the written axis a token lands mid-block, so the
        # cache cannot be baked whole.
        logger.debug(f"Skip folding {node}: blocked along the written axis.")
        return None

    cache = idx.args[0]
    q_args = node.args[1:]

    # Bake the cache: the same pad + quantize, run once, on its contents.
    baked = fetch_attr(model, cache.target)
    for op in reversed(prelude):
        baked = _run(model, op, baked)
    baked_outputs = _run(model, node, baked)

    with graph.inserting_before(node):
        buffers = {
            i: create_getattr_from_value(
                model, graph, f"{cache.target}_{name}", baked_outputs[i]
            )
            for i, name in enumerate(_OUTPUT_NAMES[node.target])
        }
        for buffer in buffers.values():
            propagate_shape(buffer, model)

        # The token pays for the pad and the quantize now, in its own right.
        token = idx.args[3]
        for op in reversed(prelude):
            token = graph.call_function(op.target, (token, *op.args[1:]))
            propagate_shape(token, model)

        new_q = graph.call_function(node.target, (token, *q_args), node.kwargs)
        new_q.meta = {
            k: copy.deepcopy(v) if k != "val" else v
            for k, v in node.meta.items()
            if k != "val"
        }
        propagate_shape(new_q, model)

    # One write per output, into the buffer that now holds it.
    for i, old in outs.items():
        with graph.inserting_before(node):
            part = graph.call_function(operator.getitem, (new_q, i))
            _typed(model, part, old.meta.get("dtype", None))
            written = graph.call_function(
                _INDEX_COPY, (buffers[i], idx.args[1], idx.args[2], part)
            )
            _typed(model, written, old.meta.get("dtype", None))
        buffers[i].meta["dtype"] = old.meta.get("dtype", None)
        old.replace_all_uses_with(written)

    # The old write mutates a buffer nothing reads now, but ``index_copy_`` is
    # side-effecting, so dead-code elimination will not collect it -- nor the
    # cache it keeps alive.  Erase the cone by hand, users first.
    for n in [*outs.values(), node, *prelude, idx, cache]:
        if not n.users:
            graph.erase_node(n)
    return new_q


class _FoldBranch:
    """The branches of a split cache's fold, lowered to a ``torch.cond``.

    On a chunk-completing step the ``true`` branch quantizes the residual
    and stores the outputs into the baked cache buffers at the chunk's own
    entries; the ``false`` branch is empty.  Both take the same operands,
    positionally: a top-level node the ``true`` branch reads gets a
    placeholder in each (``placeholder``).  Nodes are built in the ``true``
    graph ahead of its output and shape-propagated as they are made, on
    copies of the operands' values, so a store executed during propagation
    never reaches the module's buffers.
    """

    def __init__(self, model: GraphModule, cache: str, chunk_index: Node):
        self.model = model
        self.cache = cache
        self.chunk_index = chunk_index
        self.operands = []
        self.placeholders = {}
        self.offsets = {}
        self.cond = None
        self.true = GraphModule(torch.nn.Module(), Graph())
        self.false = GraphModule(torch.nn.Module(), Graph())
        self.outputs = [g.graph.output((0,)) for g in (self.true, self.false)]
        self.last_placeholder = [None, None]
        self.handles = []
        for prefix, branch in (
            ("true_graph", self.true),
            ("false_graph", self.false),
        ):
            name = get_new_node_name_with_prefix(prefix)(model)
            setattr(model, name, branch)
            self.handles.append(name)

    @property
    def graph(self) -> Graph:
        return self.true.graph

    def placeholder(self, node: Node) -> Node:
        """``node``, a top-level node, as the ``true`` branch reads it."""
        if node not in self.placeholders:
            self.operands.append(node)
            made = []
            for k, branch in enumerate((self.true, self.false)):
                anchor = self.last_placeholder[k]
                with (
                    branch.graph.inserting_after(anchor)
                    if anchor is not None
                    else branch.graph.inserting_before(self.outputs[k])
                ):
                    made.append(branch.graph.placeholder(node.name))
                made[k].meta["dtype"] = node.meta.get("dtype")
                set_node_value(made[k], node.value)
                self.last_placeholder[k] = made[k]
            self.placeholders[node] = made
        return self.placeholders[node][0]

    def arguments(self, args) -> tuple:
        """``args`` as the branch takes them: a top-level node becomes its
        placeholder, anything else passes as is."""
        return tuple(
            self.placeholder(a) if isinstance(a, Node) else a for a in args
        )

    def call(self, target, args, dtype=None, kwargs=None) -> Node:
        """A node computed in the ``true`` branch, stamped with its value."""
        with self.graph.inserting_before(self.outputs[0]):
            node = self.graph.call_function(target, args, kwargs)
        return _typed(self.true, node, dtype)

    def store(self, source: Node, buffer: Node, axis: int) -> Node:
        """Store ``source``, a chunk-sized branch value, into ``buffer``, a
        baked cache buffer, at the chunk's own entries along ``axis``:
        ``index_copy_`` at the ``span`` entries from ``chunk_index * span``,
        ``span`` being the entries a chunk covers there."""
        span = source.value.shape[axis]
        if span not in self.offsets:
            start = self.call(
                operator.mul, (self.placeholder(self.chunk_index), span)
            )
            end = self.call(operator.add, (start, span))
            self.offsets[span] = self.call(
                torch.ops.aten.arange.start_step,
                (start, end, 1),
                dtype=None,
                kwargs={
                    "dtype": torch.int64,
                    # The buffer itself: ShapeProp's value may be a copy
                    # elsewhere.
                    "device": fetch_attr(self.model, buffer.target).device,
                },
            )
        return self.call(
            _INDEX_COPY,
            (self.placeholder(buffer), axis, self.offsets[span], source),
            dtype=buffer.meta.get("dtype"),
        )

    def unstore(self, write: Node) -> None:
        """Remove a store, and its offsets once nothing else reads them."""
        self.graph.erase_node(write)
        for span, offsets in list(self.offsets.items()):
            if offsets.users:
                continue
            start, end = offsets.args[:2]
            for n in (offsets, end, start):
                self.graph.erase_node(n)
            del self.offsets[span]

    def place(self, predicate, before: Node) -> None:
        """Put the ``cond`` running the branches on ``predicate`` into the
        top-level graph, before ``before``."""
        graph = self.model.graph
        with graph.inserting_before(before):
            handles = [graph.get_attr(name) for name in self.handles]
            for handle in handles:
                propagate_shape(handle, self.model)
            self.cond = graph.call_function(_COND, (predicate, *handles, ()))
        # Scoped like a bufferized nest, so its branch nodes get model-wide
        # unique names and it lands in the layer table as the cache's fold.
        self.cond.meta["scope"] = (f"{self.cache}_fold", None)
        set_node_value(self.cond, (0,))
        self.bind()

    def bind(self) -> None:
        """Hand the ``cond`` the operands the branches take now, dropping
        a placeholder nothing reads any more."""
        for node, made in list(self.placeholders.items()):
            if made[0].users:
                continue
            for k, branch in enumerate((self.true, self.false)):
                if self.last_placeholder[k] is made[k]:
                    self.last_placeholder[k] = made[k].prev
                    if self.last_placeholder[k].op != "placeholder":
                        self.last_placeholder[k] = None
                branch.graph.erase_node(made[k])
            del self.placeholders[node]
            self.operands.remove(node)
        self.cond.args = (*self.cond.args[:3], tuple(self.operands))


@dataclass
class _Fold:
    """A split KV cache's write, as the lowering rebuilds it: ``branch``, a
    ``torch.cond`` that quantizes the residual on the completing step.

    ``op`` is that quantize's op; ``parts`` its outputs and ``stores`` the
    ``index_copy_`` of each into its baked buffer, both by output index;
    ``scale_qmap`` the table its scale rounds through, or ``None``.
    """

    cache: str
    dim: int
    op: object
    branch: _FoldBranch
    parts: dict
    stores: dict
    scale_qmap: Optional[Node]


def _fold_write(graph: Graph, target: str) -> Optional[Node]:
    """The masked write-back ``split_kv_cache`` left on cache ``target``:
    ``index_copy_(cache, dim, offsets, where(pred, residual, window))``."""
    for n in graph.nodes:
        if (
            _writes_buffer(n)
            and n.args[0].target == target
            and isinstance(n.args[3], Node)
            and n.args[3].target is torch.ops.aten.where.self
        ):
            return n
    return None


def _is_split_cache(graph: Graph, node) -> bool:
    """Whether ``node`` reads a main KV cache ``split_kv_cache`` left a fold
    on -- the buffer a dynamic quantize on the read folds into."""
    return (
        isinstance(node, Node)
        and node.op == "get_attr"
        and _fold_write(graph, node.target) is not None
    )


def _fold_quantize_into_split_cache(
    model: GraphModule, node: Node, folds: dict
) -> bool:
    """Fold a quantize (``quantize_affine``, ``quantize_mx`` or a plain
    ``quantize``) over a split KV cache into the cache.

    The quantize sweeps the whole main cache every step, though only whole,
    block-aligned chunks are ever written into it.  So the buffer is baked
    quantized once, into one buffer per quantize output
    (``<cache>_scale`` [/ ``_zero_point``] / ``_full``; a plain quantize has
    its codes alone, ``_full``), the read path takes those, and the fold's
    ``cond`` quantizes the residual instead -- ``R`` positions, on the
    completing step only -- storing its outputs into the baked buffers at
    the chunk's entries.  The chunk's blocks are its own because ``R`` is a
    multiple of the block size, and a plain quantize's scale is a constant,
    so this is the arithmetic the sweep did, on the chunk alone.

    Returns:
        ``True`` if ``node`` read a split cache and was folded.

    Raises:
        ValueError: The residual is not a whole number of blocks long.
        RuntimeError: The main cache already holds data where this step's
            fold lands, i.e. the graph was run past a fold before
            ``transform``.
    """
    graph = model.graph
    cache = node.args[0]
    if not isinstance(cache, Node) or cache.op != "get_attr":
        return False
    write = _fold_write(graph, cache.target)
    if write is None:
        return False
    # A plain quantize returns its codes alone; the others return their
    # qparams beside the value, each read through a getitem.
    plain = node.target is _QUANTIZE
    if plain:
        names, outs = ("full",), {0: node}
    else:
        if any(u.target is not operator.getitem for u in node.users):
            return False
        names = _OUTPUT_NAMES[node.target]
        outs = {u.args[1]: u for u in node.users}

    _, dim, offsets, folded = write.args
    residual = folded.args[1]
    dim %= cache.value.ndim
    length = residual.value.shape[dim]
    block_size = get_arg_value(node, 4, "block_size") if plain else node.args[3]
    if block_size is not None and length % block_size:
        raise ValueError(
            f"{cache.target}: a {length}-position residual is not a whole "
            f"number of {block_size}-wide blocks"
        )
    # The offsets are real only when folded from a constant; in a fake
    # propagation nothing has run the graph past a fold, so there is
    # nothing to check.
    contents = fetch_attr(model, cache.target)
    chunk = (
        constant_value(model, offsets)
        if isinstance(offsets, Node) and offsets.op == "get_attr"
        else offsets.value
    )
    if not isinstance(chunk, FakeTensor):
        chunk = chunk.to(contents.device)
        if contents.index_select(dim, chunk).abs().sum() != 0:
            raise RuntimeError(
                f"{cache.target} already holds data at the chunk this step "
                "folds: the graph was run past a fold before transform"
            )

    baked = _run(model, node, contents)
    if plain:
        baked = (baked,)
    with graph.inserting_before(node):
        buffers = {
            i: create_getattr_from_value(
                model, graph, f"{cache.target}_{name}", baked[i]
            )
            for i, name in enumerate(names)
        }
    for i, old in outs.items():
        _typed(model, buffers[i], old.meta.get("dtype"))
        old.replace_all_uses_with(buffers[i])

    # The chunk index and the completing-step predicate as control-
    # processor scalars.  ``scalarize_index_arithmetic`` has usually
    # already made the offsets an ``arange`` from ``c * R`` and the
    # predicate a host-written ``full``; otherwise they are read out.
    with graph.inserting_before(write):
        if offsets.target is torch.ops.aten.arange.start_step:
            chunk_index = graph.call_function(
                operator.floordiv, (offsets.args[0], length)
            )
            new_nodes = (chunk_index,)
        else:
            first = graph.call_function(_SLICE, (offsets, 0, 0, 1))
            chunk_tensor = graph.call_function(
                torch.ops.aten.floor_divide.default, (first, length)
            )
            chunk_index = graph.call_function(
                torch.ops.aten._local_scalar_dense.default, (chunk_tensor,)
            )
            new_nodes = (first, chunk_tensor, chunk_index)
        flag = folded.args[0]
        if flag.target is torch.ops.aten.full.default:
            predicate = flag.args[1]
        else:
            predicate = graph.call_function(
                torch.ops.aten._local_scalar_dense.default, (flag,)
            )
            new_nodes = (*new_nodes, predicate)
        for n in new_nodes:
            propagate_shape(n, model)

    branch = _FoldBranch(model, cache.target, chunk_index)
    chunk = branch.call(
        node.target,
        (branch.placeholder(residual), *branch.arguments(node.args[1:])),
        dtype=node.meta.get("dtype"),
    )
    if plain:
        parts = {0: chunk}
    else:
        parts = {
            i: branch.call(
                operator.getitem, (chunk, i), dtype=outs[i].meta.get("dtype")
            )
            for i in sorted(outs)
        }
    fold = _Fold(
        cache=cache.target,
        dim=dim,
        op=node.target,
        branch=branch,
        parts=parts,
        stores={},
        scale_qmap=(
            get_arg_value(node, 6, "scale_qmap")
            if node.target is _QUANTIZE_AFFINE
            else None
        ),
    )
    for i, part in fold.parts.items():
        fold.stores[i] = branch.store(part, buffers[i], dim)
    branch.place(predicate, before=write)

    # The masked write-back's own reads -- the ``where``, the window it
    # selects from the cache, and the handle on the cache -- then the
    # quantize and its outputs.
    dead = (folded, folded.args[2], write.args[0], *outs.values(), node)
    graph.erase_node(write)
    for n in dict.fromkeys(dead):
        if not n.users:
            graph.erase_node(n)
    folds[buffers[len(names) - 1].target] = fold
    logger.info(
        f"Folded {cache.target}: baked quantized, {length}-position chunks "
        "quantized on their fold"
    )
    return True


def sink_cache_folds(model: GraphModule) -> GraphModule:
    """Move each split cache's fold below the last reader of the buffers it
    writes.

    The fold's ``cond`` stands where the eager graph wrote the chunk back:
    after the attention read the cache.  Operator fusion can carry a read
    past it, since a fused group lands at its last op and the GEMV reading
    the cache may fuse with ops that follow the fold.  A read after the
    fold would see the completing chunk in both the main cache and the
    residual, so the ``cond`` and its branch handles go back after the
    last reader of any buffer the ``true`` branch writes in place.

    Runs at the end of ``fuse_operator``.

    Args:
        model: The graph module to reorder in place.

    Returns:
        ``model``, reordered in place.
    """
    graph = model.graph
    for cond in list(graph.nodes):
        if cond.target is not _COND:
            continue
        true_graph = getattr(model, cond.args[1].target).graph
        placeholders = [n for n in true_graph.nodes if n.op == "placeholder"]
        written = {
            cond.args[3][placeholders.index(n.args[0])].target
            for n in true_graph.nodes
            if n.target is _INDEX_COPY
        }
        last = cond
        for n in graph.nodes:
            if n.op != "get_attr" or n.target not in written:
                continue
            for user in n.users:
                if user > last:
                    last = user
        if last is cond:
            continue
        reader = last
        for moved in (*cond.args[1:3], cond):
            last.append(moved)
            last = moved
        logger.info(f"Sunk {cond} below {reader}")
    graph.lint()
    model.recompile()
    return model


def _hoist_microscaling(
    model: GraphModule, node: Node, folds: dict
) -> Optional[Node]:
    """Lift a ``quantize_mx`` over the relayout ops feeding it, so it quantizes
    the tensor they re-address rather than the one they hand on.

    Those ops move no element, so quantizing above them is the same arithmetic
    on less data -- and what they were going to do (broadcast a KV head, lay a
    tile out for the MXU) the consumer folds into its addressing rather than
    materializing.  Two things halt the walk: an op the quantization blocks do
    not survive (``_map_block_axes``), and an MHA head permute, where the
    quantize wants to stop -- the GEMM below stores straight through that
    permute, and the quantize fuses onto the end of it
    (``fuse_reshape_with_output``).

    The op returns its qparams beside the value (``_OUTPUT_NAMES``): the
    value keeps the relayout ops it was lifted over and each qparam gets a
    replayed copy of them.  It is never forked: a
    concat can move the axis it blocks along, and head splitting -- the only
    thing that forks -- never runs where microscaling is used.

    A cache a plain quantize was folded into (``folds``) still counts as
    that quantize, a fusable anchor.

    Returns:
        The quantize lifted above the relayouts, or None if it stays.
    """

    graph = model.graph
    if any(u.target is not operator.getitem for u in node.users):
        return None
    outs = {u.args[1]: u for u in node.users}
    value_index = len(_OUTPUT_NAMES[node.target]) - 1
    if value_index not in outs:
        return None

    block_size = node.args[3]
    src, path, src_axes = _relayout_path(
        node.args[0], node.args[2], block_size, keep_head_permute=True
    )
    if not path:
        return None

    # TODO: we only move quantize_mx when there is a fusable anchor or the
    # op/param gets repeated in the memory. However this check is not robust.
    # In the future we should move this into operator fusion.
    fold = folds.get(src.target)
    if (
        not is_compute_op(src)
        and not is_mha_qkv_permute(src)
        and not any(n.target in _BROADCAST_OPS for n in path)
        and not _is_split_cache(graph, src)
        and (fold is None or fold.op is not _QUANTIZE)
    ):
        logger.debug(f"Skip moving {node} because there is no fusable anchor.")
        return None

    new = _copy_quantize_above(model, node, src, src_axes)

    def unpack(i: int) -> Node:
        """Output ``i`` of the hoisted quantize.  ``quantize_mx``'s ``dtype`` is
        the *pair* it returns, so each output takes its own element of it -- the
        one the ``getitem`` it replaces carried."""
        out = graph.call_function(operator.getitem, (new, i))
        return _typed(model, out, outs[i].meta.get("dtype", None))

    # The value keeps the relayout ops it was lifted over: rewire them onto it.
    with graph.inserting_before(path[-1]):
        value = unpack(value_index)
    path[-1].replace_input_with(src, value)
    for n in reversed(path):
        propagate_shape(n, model)
    _annotate(path, outs[value_index])
    outs[value_index].replace_all_uses_with(path[0])

    # A qparam is one element per block, so it needs its own copy of them.
    for i, old in outs.items():
        if i == value_index:
            continue
        with graph.inserting_before(node):
            cur = _replay_path(
                model,
                unpack(i),
                path[::-1],
                node.args[2],
                block_size,
                old.meta.get("dtype", None),
            )
        old.replace_all_uses_with(cur)

    return new


def _run(model: GraphModule, node: Node, *args):
    """``node``'s op on real tensors: ``args`` in place of its leading
    operands, the rest read from the buffers they name."""

    def load(a):
        return map_arg(a, lambda n: fetch_attr(model, n.target))

    return node.target(
        *args, *load(node.args[len(args) :]), **load(node.kwargs)
    )


def _per_block(t, axes, block_size):
    """One entry of ``t`` per block of ``block_size`` along each of
    ``axes``.  When ``block_size`` is ``None``, one entry per channel
    along ``axes`` (size 1 along the other dims), or a single entry when
    there are none."""
    if block_size is None:
        if not axes:
            return t.reshape(-1)[0]
        return t[
            tuple(slice(None) if a in axes else slice(1) for a in range(t.ndim))
        ]
    return t[
        tuple(
            slice(None, None, block_size) if a in axes else slice(None)
            for a in range(t.ndim)
        )
    ]


def _stored(model: GraphModule, prefix: str, value, dtype) -> Node:
    """A ``get_attr`` of a new buffer holding ``value``, typed ``dtype``."""
    node = create_getattr_from_value(
        model, model.graph, prefix, value.contiguous()
    )
    return _typed(model, node, dtype)


def _gqa_repeat(path):
    """``(head axis, group)`` of a GQA ``repeat_kv`` on ``path``, the head
    axis in ``q``'s input layout: an ``expand`` broadcasting one dim
    ``group`` times, which a later relayout merges into the dim before it.
    ``None`` without one."""
    for k, n in enumerate(path):
        if n.target is not _EXPAND:
            continue
        sizes = zip(_shape_of(n.args[0]), _shape_of(n))
        grown = [i for i, (a, b) in enumerate(sizes) if a != b]
        if len(grown) != 1 or grown[0] == 0:
            return None
        head = grown[0] - 1
        for m in path[k + 1 :]:
            head = _map_dim_to_output(m, head)
            if head is None:
                return None
        return head, _shape_of(n)[grown[0]]
    return None


def _stored_per_kv_head(model, prefix, value, dtype, gqa) -> Node:
    """``value`` stored as a buffer, as the graph reads it.  When it repeats
    exactly over each group of ``gqa = (head axis, group)`` heads, it is
    stored once per KV head and broadcast back by
    ``unsqueeze -> expand -> reshape``."""
    if (
        gqa is not None
        and value.ndim > gqa[0]
        and value.shape[gqa[0]] % gqa[1] == 0
    ):
        head, group = gqa
        grouped = value.unflatten(head, (-1, group))
        base = grouped.narrow(head + 1, 0, 1)
        if torch.equal(grouped, base.expand_as(grouped)):
            node = _stored(model, prefix, base.squeeze(head + 1), dtype)
            for target, arg in (
                (_UNSQUEEZE, head + 1),
                (_EXPAND, list(grouped.shape)),
                (_RESHAPE, list(value.shape)),
            ):
                node = model.graph.call_function(target, (node, arg))
                _typed(model, node, dtype)
            return node
    return _stored(model, prefix, value, dtype)


def _repeat_in_branch(branch, x: Node, axis: int, block_size: int, dtype):
    """``x``, a value of the fold ``branch``, with each entry repeated
    ``block_size`` times along ``axis``."""
    shape = list(x.value.shape)
    shape[axis] *= block_size
    sizes = [-1] * (len(shape) + 1)
    sizes[axis + 1] = block_size
    x = branch.call(_UNSQUEEZE, (x, axis + 1), dtype)
    x = branch.call(_EXPAND, (x, sizes), dtype)
    return branch.call(_RESHAPE, (x, shape), dtype)


def _fold_reencode(model, fold, d, q, axes, column, buffers) -> None:
    """Extend split-cache ``fold`` to store its chunk as the fused pair reads
    it.

    The branch stops storing the chunk's affine scale and zero point.  It
    dequantizes the chunk like ``d``, re-encodes it like ``q`` along
    ``axes`` (cache layout), and stores into ``buffers`` -- the fused scale,
    zero point and block scale ``get_attr`` -- the chunk's fused scale (the
    affine scale over the block scale, rounded through ``fold.scale_qmap``
    when there is one), its zero point, and its block scale, repeated per
    column along ``column`` unless that is ``None``.
    """
    branch = fold.branch
    for i in (0, 1):
        branch.unstore(fold.stores.pop(i))
    scale, zero_point, codes = (fold.parts[i] for i in range(3))
    block_size = q.args[3]
    scale_dtype = q.meta["dtype"][0]
    decoded = branch.call(
        d.target, (codes, scale, zero_point, *branch.arguments(d.args[3:]))
    )
    reencoded = branch.call(
        q.target,
        (decoded, *branch.arguments((q.args[1], axes, *q.args[3:]))),
        q.meta["dtype"],
    )
    block_scale = branch.call(operator.getitem, (reencoded, 0), scale_dtype)
    grid = block_scale
    for axis, size in enumerate(scale.value.shape):
        if grid.value.shape[axis] != size:
            grid = _repeat_in_branch(
                branch, grid, axis, block_size, scale_dtype
            )
    fused = branch.call(
        torch.ops.aten.div.Tensor, (scale, grid), buffers[0].meta["dtype"]
    )
    if fold.scale_qmap is not None:
        with model.graph.inserting_before(branch.cond):
            unit = _stored(
                model,
                f"{fold.cache}_unit_scale",
                fetch_attr(model, buffers[0].target).new_ones(1),
                None,
            )
        fused = branch.call(
            _QUANTIZE,
            (
                fused,
                *branch.arguments((unit, None, None, None, fold.scale_qmap)),
            ),
            buffers[0].meta["dtype"],
        )
    if column is not None:
        block_scale = _repeat_in_branch(
            branch, block_scale, column, block_size, scale_dtype
        )
    for source, buffer in zip((fused, zero_point, block_scale), buffers):
        branch.store(source, buffer, fold.dim)
    branch.bind()


def _qparams_in_input_layout(model, d, q, path, axes, block_size):
    """The fused pair's qparams in ``q``'s input layout.

    Both scales are spread over every element, carried through ``path``
    (``d``'s relayouts to ``q``, nearest ``d`` first) and divided; the fused
    scale and zero point keep one entry per ``axes`` group.

    Returns:
        ``({"scale": ..., ["zero_point": ...]}, q's scale)``, or ``None``
        when a fused qparam is not constant over its groups.
    """
    codes = fetch_attr(model, d.args[0].target)

    def carried(t):
        """``t``, a tensor over the codes' elements, in q's input layout."""
        for n in path:
            t = _run(model, n, t)
        return t

    def spread(t, like):
        """Per-block ``t`` repeated over the elements of ``like``."""
        if block_size is None:
            return t.expand(like.shape)
        return repeat_per_block(t, like, block_size)

    x = carried(_run(model, d, codes))
    if q.target is _QUANTIZE_MX:
        block_scale = _run(model, q, x)[0]
    else:
        block_scale = fetch_attr(model, get_arg_value(q, 1, "scale").target)
    scale = fetch_attr(model, d.args[1].target)
    full = {"scale": carried(spread(scale, codes)) / spread(block_scale, x)}
    zero_point = get_arg_value(d, 2, "zero_point")
    if zero_point is not None:
        zero_point = fetch_attr(model, zero_point.target)
        full["zero_point"] = carried(spread(zero_point, codes))
    qparams = {k: _per_block(v, axes, block_size) for k, v in full.items()}
    if any(not torch.equal(spread(qparams[k], x), v) for k, v in full.items()):
        return None
    return qparams, block_scale


def _qparams_in_cache_layout(model, d, q, fold, q_axes, block_size):
    """The fused pair's qparams in split-cache layout, ``q`` re-encoding the
    cache on its own blocks: the affine scale over the block scale, rounded
    through ``fold.scale_qmap`` when there is one, and ``d``'s zero point.

    Returns:
        ``(qparams, block scale, q's axes, q's last blocked axis)``, the
        axes in cache layout.

    Raises:
        RuntimeError: The relayouts from the cache to ``q`` cut its blocks.
    """
    src, _, cache_axes = _relayout_path(
        q.args[0], tuple(sorted(q_axes)), block_size
    )
    if src is not d:
        raise RuntimeError(f"{fold.cache}: the relayouts to {q} cut its blocks")
    codes, scale, zero_point = (fetch_attr(model, n.target) for n in d.args[:3])
    rank = codes.ndim
    column = cache_axes[-1] % rank
    cache_axes = sorted(a % rank for a in cache_axes)
    qmap = fetch_attr(model, q.args[1].target)
    decoded = _run(model, d, codes)
    block_scale = _run(model, q, decoded, qmap, cache_axes)[0]
    fused = scale / repeat_per_block(block_scale, scale, block_size)
    if fold.scale_qmap is not None:
        table = fetch_attr(model, fold.scale_qmap.target)
        fused = _QUANTIZE(fused, fused.new_ones(1), qmap=table)
    qparams = {"scale": fused, "zero_point": zero_point}
    return qparams, block_scale, cache_axes, column


def _fuse_pair(model: GraphModule, q: Node, folds: dict) -> bool:
    """Fuse quantize ``q`` with the dequantize of a stored tensor it reads,
    through relayouts, into one dequantize, when the two can be.

    A quantize with a zero point of its own is left alone.  Over a split
    cache's baked codes (``folds``), a plain quantize fuses with a plain
    fold, whose scale is constant, and a ``quantize_mx`` with an affine fold,
    which it then extends; no other pair over a fold does.

    A broadcast qparam (GQA's ``repeat_kv``) is stored once per KV head rather
    than once per query head, with the repeat put back as graph ops: a smaller
    buffer for a longer graph, which pays off because the GEMM folds the repeat
    into its tile addressing instead of copying.

    A chain reading a split KV cache's baked codes (``folds``, keyed by the
    codes buffer's target) gets the same fused dequantize, with three
    differences.  Its qparams are kept in cache layout, like the codes --
    the block scale taken on the cache's own blocks, the group scale and
    zero point as the cache holds them -- with the relayouts to the GEMM
    put back into the graph on the qparams, so the fold can store a chunk's
    qparams as its quantize returns them.  Its fused scale and zero point
    take the affine qparams' dtype, the fused scale rounded through the
    affine scale's table when there is one, since the fold stores a run-time
    quotient into that buffer.  And the cache's fold is extended to store
    the chunk's block scale, fused scale and zero point beside its codes
    (``_fold_reencode``).

    Returns:
        Whether it fused ``q``.
    """
    path, d = [], q.args[0]
    while len(d.users) == 1 and _map_block_axes(d, (), None) is not None:
        path.insert(0, d)
        d = d.args[0]
    mx = q.target is _QUANTIZE_MX
    if (
        d.target is not _DEQUANTIZE
        or d.args[0].op != "get_attr"
        or len(d.users) != 1
        or any(
            n.op != "get_attr"
            for n in d.all_input_nodes + q.all_input_nodes[1:]
        )
    ):
        return False
    if not mx and get_arg_value(q, 2, "zero_point") is not None:
        logger.warning(
            f"Not fusing {d} into {q}: a quantize with its own zero point is "
            "not supported yet"
        )
        return False
    block_size = get_arg_value(d, 4, "block_size") or 1
    q_block_size = q.args[3] if mx else get_arg_value(q, 4, "block_size")
    if block_size != (q_block_size or 1):
        return False
    # One element per block is no blocking at all.
    block_size = block_size if block_size > 1 else None

    # The axes each side groups along, in q's input layout; the fused
    # dequantize groups along those both do.
    codes_node, scale_node = d.args[0], d.args[1]
    codes = fetch_attr(model, codes_node.target)
    rank = len(_shape_of(q.args[0]))
    q_axes = q.args[2] if mx else get_arg_value(q, 3, "axes")
    q_axes = {a % rank for a in q_axes or ()}
    d_axes = get_arg_value(d, 3, "axes") or ()
    for n in path:
        d_axes = _map_block_axes(n, d_axes, block_size, from_output=False)
        if d_axes is None:
            return False
    d_axes = {a % rank for a in d_axes}
    # One side's groups must nest in the other's, or the fused scale would
    # be one per element; a size-1 axis groups nothing either way.
    wide = {a for a, size in enumerate(_shape_of(q.args[0])) if size > 1}
    if not (q_axes & wide <= d_axes or d_axes & wide <= q_axes):
        return False
    axes = tuple(sorted(q_axes & d_axes))

    fold = folds.get(codes_node.target)
    if fold is not None and fold.op is _QUANTIZE and not mx:
        fold = None
    elif fold is not None and not (fold.op is _QUANTIZE_AFFINE and mx):
        logger.warning(
            f"Not fusing {d} into {q}: fusing with a {fold.op} cache write is "
            "not supported yet"
        )
        return False
    if fold is None:
        found = _qparams_in_input_layout(model, d, q, path, axes, block_size)
        if found is None:
            return False
        qparams, block_scale = found
        column, width = rank - 1, _shape_of(q.args[0])[-1]
    else:
        qparams, block_scale, cache_axes, column = _qparams_in_cache_layout(
            model, d, q, fold, q_axes, block_size
        )
        width = codes.shape[column]

    # A GEMM takes its weight scale one entry per output column.
    outs = {u.args[1]: u for u in q.users} if mx else {}
    columns = (
        0 in outs
        and rank - 1 in q_axes
        and any(is_gemm_op(u) for u in outs[0].users)
    )
    if columns:
        block_scale = block_scale.repeat_interleave(block_size, column)
        block_scale = block_scale.narrow(column, 0, width)

    graph = model.graph
    gqa = _gqa_repeat(path)
    attrs = []

    def read(name, value, dtype, grid):
        """``value`` stored as ``<codes>_<name>``, as q's input layout reads
        it; ``grid`` are its per-block axes there."""
        prefix = f"{codes_node.target}_{name}"
        if fold is None:
            return _stored_per_kv_head(model, prefix, value, dtype, gqa)
        attrs.append(_stored(model, prefix, value, dtype))
        return _replay_path(model, attrs[-1], path, grid, block_size, dtype)

    d.replace_all_uses_with(codes_node)
    for n in path:
        n.meta["dtype"] = codes_node.meta.get("dtype")
    dtype = scale_node.meta.get("dtype")
    with graph.inserting_before(q):
        scale_read = read("scale", qparams["scale"], dtype, axes)
        zp_read = None
        if "zero_point" in qparams:
            zp_read = read("zero_point", qparams["zero_point"], dtype, axes)
        qmap = graph.node_copy(q.args[1] if mx else get_arg_value(q, 5, "qmap"))
        new = graph.call_function(
            _DEQUANTIZE,
            (q.args[0], scale_read, zp_read, axes, block_size, None, qmap),
        )
        new.meta["dtype"] = q.meta["dtype"][1] if mx else q.meta["dtype"]
        propagate_shape(qmap, model)
        propagate_shape(new, model)
        if 0 in outs:
            grid = tuple(q_axes - {rank - 1} if columns else q_axes)
            scale_dtype = outs[0].meta.get("dtype")
            outs[0].replace_all_uses_with(
                read("scale", block_scale, scale_dtype, grid)
            )

    if not mx:
        q.replace_all_uses_with(new)
        graph.erase_node(q)
    elif 1 in outs:
        outs[1].replace_all_uses_with(new)
    if fold is not None:
        column = column if columns else None
        _fold_reencode(model, fold, d, q, cache_axes, column, attrs)
    graph.erase_node(d)
    return True


def fuse_quantize_dequantize_with_producer(model: GraphModule):
    """Take quantization work off the run time by moving each quantize or
    dequantize next to the op it belongs with.

    Each one is visited once, in graph order, and the first step that applies
    is taken:

    1. Cache write: a quantize on a KV cache moves into the cache write, so the
       cache is stored quantized and only new tokens are quantized at run time
       (``_fold_quantize_into_split_cache``, ``_fold_quantize_into_cache``).
    2. Pair fusion: a quantize that reads a stored tensor's dequantize,
       through shape-only ops, becomes one dequantize with the combined scale
       (``_fuse_pair``).
    3. Hoist: the quantize moves up over shape-only ops to sit right after the
       op that computed its input, so the two fuse into one kernel
       (``_hoist_microscaling`` for a blocked quantize, ``_hoist_forked`` for
       a per-tensor one, which also forks over a concat).  If it then sits on
       a KV cache, step 1 applies.

    A blocked or per-channel plain quantize and ``quantize_mx_outlier`` stay
    where they are.

    Args:
        model: The graph module to rewrite in place.

    Returns:
        ``model``, rewritten in place.
    """
    graph = model.graph
    # A dead reader would stop the walks below.
    graph.eliminate_dead_code()
    folds = {}
    for node in list(graph.nodes):
        if node.target not in QUANTIZE_FAMILY_OPS:
            continue
        if node.target is _QUANTIZE_AFFINE:
            if _fold_quantize_into_split_cache(model, node, folds):
                continue
            # Folded into a cache written token by token, the quantize
            # moves onto the token and is hoisted from there.
            node = _fold_quantize_into_cache(model, node) or node
        elif node.target in (_QUANTIZE, _QUANTIZE_MX):
            if _fuse_pair(model, node, folds):
                continue
        elif node.target is _QUANTIZE_MX_OUTLIER:
            # The sparse quantize's CSR outputs have no relayout replay; it
            # stays where the quantizer put it.
            continue

        if node.target in _OUTPUT_NAMES:
            quantizes = [_hoist_microscaling(model, node, folds) or node]
        elif (
            get_arg_value(node, 3, "axes")
            or (get_arg_value(node, 4, "block_size") or 1) > 1
        ):
            # A blocked or per-channel plain quantize would need the same
            # axis bookkeeping as quantize_mx, which it does not have; only
            # per-tensor is lifted.
            quantizes = [node]
        else:
            quantizes = _hoist_forked(model, node) or [node]
        for quantize in quantizes:
            if quantize.target is _QUANTIZE_MX:
                if not _fold_quantize_into_split_cache(model, quantize, folds):
                    _fold_quantize_into_cache(model, quantize)
            elif quantize.target is _QUANTIZE:
                _fold_quantize_into_split_cache(model, quantize, folds)

    graph.lint()
    graph.eliminate_dead_code()
    model.recompile()

    return model

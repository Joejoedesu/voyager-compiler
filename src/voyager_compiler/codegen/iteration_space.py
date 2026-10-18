from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import fx

from voyager_compiler.codegen.node_info import (
    DYNAMIC_QUANTIZE_OPS,
    NORM_OPS,
    QUANTIZE_FAMILY_OPS,
    SOFTMAX_OPS,
    ancestors,
    get_anchor_node,
    get_arg_value,
    is_aliasing_op,
    is_elementwise_op,
    is_nop,
    is_reshape_op,
    quant_param_arg_nodes,
    reshape_preserves_full_blocks,
)
from voyager_compiler.shape_prop import (
    ShapeProp,
    propagate_shape,
    set_node_value,
)

Shape = Tuple[int, ...]
# One dimension of an address map: row-major ``(radix, stride)`` digits.
Digits = Tuple[Tuple[int, int], ...]


def _canonical(digits) -> Digits:
    """Drop radix-1 digits and merge neighbours that read consecutively.

    Equal maps then have equal digits; a size-1 dim is ``((1, 0),)``.
    """
    out = []
    for radix, stride in digits:
        if radix == 1:
            continue
        if out and out[-1][1] == radix * stride:
            out[-1] = (out[-1][0] * radix, stride)
        else:
            out.append((radix, stride))
    return tuple(out) or ((1, 0),)


@dataclass(frozen=True)
class AddressMap:
    """The flat index of its external input each fused-value position reads.

    ``dims[d]`` is the row-major list of ``(radix, stride)`` digits of dim
    ``d``: an index along it adds each of its mixed-radix digits times that
    digit's stride.  Held as digits, a map stays a few tuples and every shape
    op keeps it exact: a permute reorders dims, an expand is a zero-stride
    digit, and a reshape regroups digits, splitting one a new boundary falls
    inside.  A dim of one digit is read at a fixed stride; one of several
    (e.g. a head axis with a merged GQA repeat) is not.
    """

    dims: Tuple[Digits, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dims", tuple(_canonical(d) for d in self.dims)
        )

    @classmethod
    def arange(cls, shape: Sequence[int]) -> "AddressMap":
        """The identity map over a contiguous input of ``shape``."""
        dims = []
        stride = 1
        for size in reversed([int(s) for s in shape]):
            dims.append(((size, stride),))
            stride *= max(size, 1)
        return cls(tuple(reversed(dims)))

    @property
    def shape(self) -> Shape:
        return tuple(math.prod(r for r, _ in d) for d in self.dims)

    def reshape(self, shape: Sequence[int]) -> "AddressMap":
        """Regroup the digits into ``shape``.

        All dims' digits form one merged row-major list, so only an address
        jump, not an old dim boundary, limits the regrouping.  A new boundary
        inside a digit splits it.

        Raises:
            ValueError: If ``shape`` has another element count, or cuts a
                digit at no multiple of its radix.
        """
        shape = [int(s) for s in shape]
        if math.prod(shape) != math.prod(self.shape):
            raise ValueError(f"cannot reshape {self.shape} to {shape}")
        flat = list(_canonical(d for dim in self.dims for d in dim))
        dims = []
        for size in shape:
            taken = []
            remaining = size
            while remaining > 1:
                radix, stride = flat.pop(0)
                if remaining % radix and radix % remaining:
                    raise ValueError(
                        f"reshape of {self.shape} to {shape} cuts a "
                        "dimension inside a digit"
                    )
                if radix <= remaining:
                    taken.append((radix, stride))
                    remaining //= radix
                else:
                    taken.append((remaining, stride * (radix // remaining)))
                    flat.insert(0, (radix // remaining, stride))
                    remaining = 1
            dims.append(tuple(taken))
        return AddressMap(tuple(dims))

    def permute(self, order: Sequence[int]) -> "AddressMap":
        return AddressMap(tuple(self.dims[o] for o in order))

    def expand(self, sizes: Sequence[int]) -> "AddressMap":
        """Broadcast to ``sizes``; a size-1 or new leading dim gets stride 0.

        Raises:
            ValueError: If the map does not broadcast to ``sizes``.
        """
        lead = len(sizes) - len(self.dims)
        if lead < 0:
            raise ValueError(f"cannot expand {self.shape} to {sizes}")
        dims = [((size, 0),) for size in sizes[:lead]]
        for size, digits, current in zip(sizes[lead:], self.dims, self.shape):
            if size == current:
                dims.append(digits)
            elif current == 1:
                dims.append(((size, 0),))
            else:
                raise ValueError(f"cannot expand {self.shape} to {sizes}")
        return AddressMap(tuple(dims))


# Parent relayouts are aten ops, which the compiler treats as aliasing nops.
_RELAYOUT_OPS = {
    "view": torch.ops.aten.view.default,
    "reshape": torch.ops.aten.reshape.default,
}

# Ops whose data input (args[0]) alone carries the address map: they keep its
# elements in place and index their other operands (qparams, tables, a norm's
# weight and bias) by block or normalized dim, not by position.
_PASS_THROUGH_OPS = (
    NORM_OPS | SOFTMAX_OPS | QUANTIZE_FAMILY_OPS | {torch.ops.aten.to.dtype}
)

# The shape ops that name their output shape outright.
_VIEW_OPS = (
    torch.ops.aten.view.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten._unsafe_view.default,
)

# The calibrated quantize and dequantize, which take their qparams as operands.
_QDQ_OPS = (
    torch.ops.quantized_ops.quantize.default,
    torch.ops.quantized_ops.dequantize.default,
)


class NormalizationError(RuntimeError):
    """Raised when a fused child cannot use one iteration space."""

    def __init__(self, message: str, *, node: Optional[fx.Node] = None) -> None:
        if node is not None:
            message = (
                f"{message} [node={node.name}, op={node.op}, "
                f"target={node.target!r}]"
            )
        super().__init__(message)
        self.node = node


def _moves_elements(node: fx.Node) -> bool:
    """Whether ``node`` is an ``as_strided`` other than the identity view."""
    if node.target is not torch.ops.aten.as_strided.default:
        return False
    source = node.args[0].value
    size, stride = node.args[1:3]
    offset = get_arg_value(node, 3, "storage_offset")
    return not (
        tuple(size) == tuple(source.shape)
        and source.is_contiguous()
        and torch.empty_strided(size, stride, device="meta").is_contiguous()
        and offset in (None, source.storage_offset())
    )


def _replay(node: fx.Node, value: AddressMap) -> AddressMap:
    """``value`` read through the shape op ``node``.

    A transpose or permute reorders dims, an expand broadcasts, and any other
    op (view, reshape, squeeze, unsqueeze, size-1 select) regroups digits.

    Raises:
        NormalizationError: If the map cannot take the op, as for any
            ``as_strided``.
    """
    target = node.target
    if target is torch.ops.aten.as_strided.default:
        raise NormalizationError("Unsupported shape no-op", node=node)
    try:
        if target is torch.ops.aten.permute.default:
            return value.permute(node.args[1])
        if target is torch.ops.aten.transpose.int:
            order = list(range(len(value.dims)))
            a, b = node.args[1:3]
            order[a], order[b] = order[b], order[a]
            return value.permute(order)
        if target is torch.ops.aten.expand.default:
            return value.expand(node.shape)
        return value.reshape(node.shape)
    except ValueError as exc:
        raise NormalizationError("Unsupported shape no-op", node=node) from exc


def _boundary_shape(value: AddressMap, shape: Shape, node: fx.Node) -> Shape:
    """The iteration ``shape`` with each dim ``value`` broadcasts set to 1.

    That is the input's shape at the group boundary: ``value`` regrouped to
    ``shape`` must read it row-major, broadcast to ``shape``.

    Raises:
        NormalizationError: Blaming ``node``, if ``value`` reads otherwise.
    """
    try:
        value = value.reshape(shape)
    except ValueError as exc:
        raise NormalizationError(
            f"Address map of shape {value.shape} cannot be regrouped into "
            f"{tuple(shape)}",
            node=node,
        ) from exc
    boundary = tuple(
        1 if all(stride == 0 for _, stride in digits) else extent
        for digits, extent in zip(value.dims, shape)
    )
    if value != AddressMap.arange(boundary).expand(shape):
        raise NormalizationError(
            "The input is not read in row-major order, broadcast to the "
            "iteration shape",
            node=node,
        )
    return boundary


def _propagate(
    seed: fx.Node, output: fx.Node, anchor: Optional[fx.Node]
) -> Optional[AddressMap]:
    """``seed``'s address map at ``output``, or ``None`` if it is not read.

    Maps stop at ``anchor`` unless it is ``seed``, so only the tail is
    followed.

    Raises:
        NormalizationError: If an op on the way has no map, or one input
            reaches an elementwise op through two different maps.
    """
    # The nodes ``output`` reads, without looking past the anchor.
    tail = set()
    stack = [output]
    while stack:
        node = stack.pop()
        if node not in tail:
            tail.add(node)
            if node is not anchor:
                stack.extend(node.all_input_nodes)

    env: Dict[fx.Node, Optional[AddressMap]] = {}
    for node in seed.graph.nodes:
        if node is seed:
            env[node] = AddressMap.arange(seed.shape)
        elif (
            node not in tail
            or node.op in ("placeholder", "get_attr")
            or node is anchor
        ):
            env[node] = None
        elif (
            is_aliasing_op(node)
            or is_reshape_op(node)
            or node.target is torch.ops.aten.expand.default
        ):
            source = env[node.all_input_nodes[0]]
            env[node] = None if source is None else _replay(node, source)
        elif _moves_elements(node):
            raise NormalizationError(
                "An as_strided that moves elements has no address map",
                node=node,
            )
        elif is_nop(node) or node.target in _PASS_THROUGH_OPS:
            env[node] = env.get(node.args[0])
        elif is_elementwise_op(node):
            try:
                maps = {
                    env[arg].expand(node.shape)
                    for arg in node.all_input_nodes
                    if env[arg] is not None
                }
            except ValueError as exc:
                raise NormalizationError(
                    "An address map cannot broadcast to the op's shape",
                    node=node,
                ) from exc
            if len(maps) > 1:
                raise NormalizationError(
                    "One external input reaches the same pointwise "
                    "operation through incompatible index mappings",
                    node=node,
                )
            env[node] = maps.pop() if maps else None
        elif node.op != "output":
            raise NormalizationError(
                "Cannot propagate address map through node", node=node
            )
        if node is output:
            return env[node]


def _strip_output_relayout(
    value: fx.Node,
) -> Tuple[fx.Node, Tuple[fx.Node, ...]]:
    """Peel a trailing single-use relayout chain off ``value``.

    Returns the core and the chain, output-first, but only when the chain
    holds a transpose, permute or non-identity ``as_strided``, which no
    parent view can express; otherwise ``(value, ())``.
    """
    chain = []
    node = value
    while (
        (is_reshape_op(node) or is_aliasing_op(node) or _moves_elements(node))
        and len(node.users) == 1
        and node.all_input_nodes
    ):
        chain.append(node)
        node = node.all_input_nodes[0]
    if not any(is_reshape_op(n) or _moves_elements(n) for n in chain):
        return value, ()
    return node, tuple(chain)


def _arg_shapes(node: fx.Node) -> list:
    """The shape of each node argument of ``node``, in argument order."""
    shapes = []
    fx.map_arg(node.args, lambda a: shapes.append(getattr(a, "shape", None)))
    return shapes


def _moves_nothing(node: fx.Node) -> bool:
    """Whether ``node``, not a view or reshape, moves no element as traced.

    E.g. a shape-keeping nop or a transpose of size-1 dims.
    """
    source = node.args[0] if node.args else None
    if not isinstance(source, fx.Node) or node.target in _VIEW_OPS:
        return False
    if is_reshape_op(node):
        moved = _replay(node, AddressMap.arange(source.shape))
        return moved == AddressMap.arange(node.shape)
    return (
        is_nop(node)
        and tuple(node.shape) == tuple(source.shape)
        and not _moves_elements(node)
    )


def _moved_dim(
    node: fx.Node,
    dim: int,
    old_shape: Shape,
    new_shape: Shape,
    block_size: Optional[int] = None,
) -> int:
    """The dim of ``new_shape`` holding what ``dim`` of ``old_shape`` holds.

    Both shapes order the same elements row-major.  The match has the same
    extent and as many elements after it or, given ``block_size``, keeps the
    same blocks whole; it counts from the same end as ``dim``.

    Raises:
        NormalizationError: If no dim of ``new_shape`` does.
    """
    old_dim = dim % len(old_shape)
    after = math.prod(old_shape[old_dim + 1 :])
    for new_dim in reversed(range(len(new_shape))):
        if block_size is None:
            holds = (
                new_shape[new_dim] == old_shape[old_dim]
                and math.prod(new_shape[new_dim + 1 :]) == after
            )
        else:
            holds = reshape_preserves_full_blocks(
                old_shape, old_dim, new_shape, new_dim, block_size
            )
        if holds:
            return new_dim if dim >= 0 else new_dim - len(new_shape)
    raise NormalizationError(
        f"Dim {dim} of {old_shape} has no counterpart in {new_shape}",
        node=node,
    )


def _same_broadcast(old_in, old_out, new_in, new_out) -> bool:
    """Whether two broadcasts read the same elements in row-major order.

    Compares ``old_in`` -> ``old_out`` with ``new_in`` -> ``new_out``.
    """
    try:
        old = AddressMap.arange(old_in).expand(old_out).reshape(new_out)
        return old == AddressMap.arange(new_in).expand(new_out)
    except ValueError:
        return False


def _check_operands(
    node: fx.Node,
    old_shapes: Sequence[Shape],
    new_shapes: Sequence[Shape],
    old_out: Shape,
) -> None:
    """Reject ``node`` unless every operand broadcasts as traced."""
    pairs = [
        (old, new)
        for old, new in zip(old_shapes, new_shapes)
        if new is not None
    ]
    try:
        new_out = torch.broadcast_shapes(*(new for _, new in pairs))
    except RuntimeError as exc:
        raise NormalizationError(
            "The operands no longer broadcast together", node=node
        ) from exc
    for old, new in pairs:
        if not _same_broadcast(old, old_out, new, new_out):
            raise NormalizationError(
                f"An operand of shape {tuple(new)} broadcasts onto other "
                "elements",
                node=node,
            )


def _restate(
    node: fx.Node, old_shapes: Sequence[Shape], old_out: Shape
) -> None:
    """Restate ``node`` for node arguments that changed shape.

    Each argument still holds its traced elements row-major.  A named dim
    moves to its counterpart (``_moved_dim``); a norm's normalized shape
    becomes the shortest suffix with as many elements, kept as traced if the
    norm is affine; a blocked (de)quantize must keep the dims its scale spans,
    and an outlier quantize its last two (the CSR rows); an expand, an
    elementwise op and a broadcast (de)quantize need each operand to
    broadcast onto its traced elements.

    Args:
        old_shapes: The traced shapes of its node arguments, in order.
        old_out: Its traced output shape.

    Raises:
        NormalizationError: If the op would act on other elements.
    """
    target = node.target
    args = list(node.args)
    new_shapes = _arg_shapes(node)
    old_in, new_in = tuple(old_shapes[0]), tuple(new_shapes[0])
    if target in _VIEW_OPS:
        return
    if target is torch.ops.aten.transpose.int:
        args[1:3] = [_moved_dim(node, d, old_in, new_in) for d in args[1:3]]
    elif target in SOFTMAX_OPS:
        args[1] = _moved_dim(node, args[1], old_in, new_in)
    elif target in NORM_OPS:
        size = math.prod(args[1])
        suffixes = (new_in[k:] for k in reversed(range(len(new_in))))
        restated = next((s for s in suffixes if math.prod(s) == size), None)
        affine = any(isinstance(a, fx.Node) for a in args[2:4])
        if restated is None or (affine and restated != tuple(args[1])):
            raise NormalizationError(
                f"The normalized dims {tuple(args[1])} have no place in "
                f"{new_in}",
                node=node,
            )
        args[1] = list(restated)
    elif target is torch.ops.aten.expand.default:
        lead = (1,) * max(len(new_in) - len(old_out), 0)
        args[1] = list(lead + tuple(old_out))
        if not _same_broadcast(old_in, old_out, new_in, args[1]):
            raise NormalizationError(
                "The expand broadcasts onto other elements", node=node
            )
    elif target in DYNAMIC_QUANTIZE_OPS:
        outlier = torch.ops.quantized_ops.quantize_mx_outlier.default
        if target is outlier and old_in[-2:] != new_in[-2:]:
            raise NormalizationError(
                "The CSR rows of an outlier quantize change", node=node
            )
        args[2] = [
            _moved_dim(node, axis, old_in, new_in, args[3]) for axis in args[2]
        ]
    elif target in _QDQ_OPS and len(args) > 4 and args[4] is not None:
        # A blocked scale is expanded to the input dim by dim.
        span = max(len(q.shape) for q in args[1:3] if isinstance(q, fx.Node))
        if len(new_in) < span or old_in[-span:] != new_in[-span:]:
            raise NormalizationError(
                f"The scale spans the last {span} dims of {old_in}, which "
                f"become {new_in}",
                node=node,
            )
        args[3] = [
            _moved_dim(node, axis, old_in, new_in, args[4])
            for axis in args[3] or ()
        ]
    elif is_elementwise_op(node) or target is torch.ops.aten.to.dtype:
        # A (de)quantize broadcasts its input, scale and zero point; the
        # quantization tables after them are read whole.
        count = len(new_shapes)
        if target in _QDQ_OPS:
            count = sum(isinstance(a, fx.Node) for a in args[:3])
        _check_operands(node, old_shapes[:count], new_shapes[:count], old_out)
    else:
        raise NormalizationError(
            "Cannot restate the op for its new input shapes", node=node
        )
    node.args = tuple(args)


def _rewrite_child(
    child: fx.GraphModule,
    anchor: Optional[fx.Node],
    relayout: Tuple[fx.Node, ...],
    inputs: Dict[fx.Node, object],
) -> None:
    """Move the tail's shape nops out of ``child`` to the boundary views.

    A value that changes shape keeps its traced elements row-major, so only
    ops whose input shapes change are touched: a view or reshape is kept, an
    op that moved no element as traced is dropped, an output relayout reads
    the core through a view to its traced shape, and any other op is
    restated (``_restate``).

    Args:
        anchor: The strong anchor, or ``None`` in an anchorless group.
        relayout: The output relayout kept inside the child.
        inputs: The value each placeholder takes.

    Raises:
        NormalizationError: If an op cannot act on its new input shapes.
    """
    # Keep the anchor's operand prelude and the output relayout.
    keep = ancestors(anchor) | set(relayout)
    removable = [
        node
        for node in child.graph.nodes
        if is_aliasing_op(node) and node not in keep
    ]
    traced = {
        node: (_arg_shapes(node), getattr(node, "shape", None))
        for node in child.graph.nodes
        if node.op == "call_function"
    }
    identities = {node for node in traced if _moves_nothing(node)}

    for node in removable:
        node.replace_all_uses_with(node.all_input_nodes[0])
        child.graph.erase_node(node)

    for placeholder, value in inputs.items():
        set_node_value(placeholder, value)
    for node in list(child.graph.nodes):
        if node not in traced or _arg_shapes(node) == traced[node][0]:
            continue
        arg_shapes, shape = traced[node]
        if node in identities:
            node.replace_all_uses_with(node.args[0])
            child.graph.erase_node(node)
            continue
        if node in relayout and node.target not in _VIEW_OPS:
            source = node.args[0]
            with child.graph.inserting_before(node):
                view = child.graph.call_function(
                    torch.ops.aten.view.default,
                    (source, list(arg_shapes[0])),
                )
            propagate_shape(view, child)
            node.replace_input_with(source, view)
        else:
            _restate(node, arg_shapes, shape)
        try:
            propagate_shape(node, child)
        except (RuntimeError, IndexError) as exc:
            raise NormalizationError(
                "The op cannot run on its restated inputs", node=node
            ) from exc


def _check_mx_restore(
    value: fx.Node, internal: Sequence[Shape], external: Sequence[Shape]
) -> None:
    """Reject a ``quantize_mx`` output restore that breaks a block.

    The op must quantize along one axis, and the restore, a row-major view
    from ``internal`` to ``external`` shapes, must keep every block of the
    data output (the last) whole along that axis, counted from the end.

    Raises:
        NormalizationError: If the restore breaks the blocks.
    """
    axes, block_size = value.args[2:4]
    data_internal, data_external = tuple(internal[-1]), tuple(external[-1])
    dim = axes[0] if len(axes) == 1 else None
    if dim is not None and dim >= 0:
        dim -= len(data_internal)
    if (
        dim is None
        or len(data_external) < -dim
        or not reshape_preserves_full_blocks(
            data_internal,
            dim + len(data_internal),
            data_external,
            dim + len(data_external),
            block_size,
        )
    ):
        raise NormalizationError(
            "MX output restore would regroup non-unit dimensions and change "
            "quantization block semantics",
            node=value,
        )


def _input_shapes(
    placeholders: Sequence[fx.Node],
    anchor: Optional[fx.Node],
    output: fx.Node,
    iteration_shape: Shape,
) -> Dict[str, Tuple[Shape, Shape]]:
    """Each input the tail reads -> its traced and boundary shape.

    A quantization table (indexed by value), a scalar of at most the
    iteration rank, and an input ``output`` does not read pass whole.

    Args:
        anchor: The strong anchor, or ``None`` in an anchorless group.
        output: The node the tail ends at.

    Raises:
        NormalizationError: If an anchor input is also read by the tail, or
            an input has no boundary shape.
    """
    anchor_inputs = ancestors(anchor)
    quant_tables = set()
    for node in output.graph.nodes:
        quant_tables |= quant_param_arg_nodes(node)
    shapes = {}
    for placeholder in placeholders:
        shape = placeholder.shape
        # An anchor input is checked even if it is a table or a scalar.
        scalar = math.prod(shape) == 1 and len(shape) <= len(iteration_shape)
        broadcasts = placeholder in quant_tables or scalar
        if broadcasts and placeholder not in anchor_inputs:
            continue
        seen = _propagate(placeholder, output, anchor)
        if seen is None:
            continue
        if placeholder in anchor_inputs:
            raise NormalizationError(
                "An anchor operand is also reused by the pointwise tail "
                "outside the anchor",
                node=placeholder,
            )
        boundary = _boundary_shape(seen, iteration_shape, placeholder)
        shapes[placeholder.name] = (shape, boundary)
    return shapes


def _output_shapes(
    value: fx.Node, external_shapes: Sequence[Shape]
) -> Tuple[Tuple[Shape, Shape], ...]:
    """Each output's shape as the rewritten child returns it, and as traced.

    Raises:
        NormalizationError: If a restore would change an output's element
            count or break a dynamic quantize's blocks.
    """
    mx = value.target in DYNAMIC_QUANTIZE_OPS
    internal_shapes = (
        tuple(tuple(s) for s in value.shape) if mx else (tuple(value.shape),)
    )
    for internal, external in zip(internal_shapes, external_shapes):
        if math.prod(internal) != math.prod(external):
            raise NormalizationError(
                "Output restore would change element count: "
                f"internal={internal}, external={external}",
                node=value,
            )
    if mx:
        _check_mx_restore(value, internal_shapes, external_shapes)
    return tuple(zip(internal_shapes, external_shapes))


def _restore_outputs(
    parent: fx.GraphModule,
    call_node: fx.Node,
    output_shapes: Sequence[Tuple[Shape, Shape]],
    relayout_op,
) -> None:
    """Restore each output the parent reads at another shape.

    A tuple output is read through ``getitem`` nodes; once any item changes
    shape, every item re-takes its dtype and value from the call.

    Args:
        call_node: The fused ``call_module`` node, its value already set.
        output_shapes: Each output's returned and traced shape.
        relayout_op: The parent op that restores a shape.
    """
    reads = [(call_node, *output_shapes[0])]
    if len(output_shapes) > 1:
        reads = []
        if any(i != tuple(e) for i, e in output_shapes):
            dtypes = call_node.meta.get("dtype")
            for item in list(call_node.users):
                index = item.args[1]
                if (
                    isinstance(dtypes, (tuple, list))
                    and index < len(dtypes)
                    and dtypes[index] is not None
                ):
                    item.meta["dtype"] = dtypes[index]
                propagate_shape(item, parent)
                reads.append((item, *output_shapes[index]))
    for node, internal, external in reads:
        if internal == tuple(external):
            continue
        users = list(node.users)
        with parent.graph.inserting_after(node):
            restored = parent.graph.call_function(
                relayout_op, (node, list(external))
            )
        # Shallow copy: a deepcopy would touch a FakeTensor ``val``.
        restored.meta = dict(node.meta)
        propagate_shape(restored, parent)
        for user in users:
            user.replace_input_with(node, restored)


@dataclass(frozen=True)
class NormalizationResult:
    """What one ``normalize_group`` call chose.

    Attributes:
        iteration_shape: The group's iteration shape.
        input_shapes: Each reshaped input -> its traced and boundary shape.
        output_shapes: Each output's returned and traced shape.
    """

    iteration_shape: Shape
    input_shapes: Dict[str, Tuple[Shape, Shape]]
    output_shapes: Tuple[Tuple[Shape, Shape], ...]


@dataclass
class NormalizationConfig:
    """Configuration for :func:`normalize_group`.

    Attributes:
        require_contiguous_inputs: Under ``"reshape"``, reject an input with
            no view of its boundary shape instead of copying it.
        parent_shape_method: The parent op, ``"view"`` or ``"reshape"``,
            that reshapes inputs to the boundary and outputs back.
    """

    require_contiguous_inputs: bool = True
    parent_shape_method: str = "view"

    def __post_init__(self) -> None:
        if self.parent_shape_method not in {"view", "reshape"}:
            raise ValueError("parent_shape_method must be 'view' or 'reshape'")


def normalize_group(
    parent: fx.GraphModule,
    call_node: fx.Node,
    config: Optional[NormalizationConfig] = None,
) -> NormalizationResult:
    """Rewrite fused ``call_node`` and its parent to one iteration shape.

    Each fused group runs one loop nest, so every operand must align from
    the right with one iteration shape: the strong anchor's (gemm, conv, norm
    or softmax), or the output's if there is none.  The tail must read the
    anchor's output in order, once per position, and every other input a
    boundary shape row-major, broadcast to the iteration shape (exemptions in
    ``_input_shapes``).  Shape ops move from the child to parent views; an
    output relayout that moves elements stays in the child, and any other
    output shape change is undone by a parent view.  The parent changes only
    after every check, so a rejection leaves it intact.  The child carries
    ``ShapeProp`` values, binds placeholders by position, and returns a
    tensor or a terminal dynamic quantize's tuple.

    Args:
        parent: The graph module holding ``call_node``.
        call_node: The fused ``call_module`` node.
        config: How the parent reshapes inputs and outputs.

    Returns:
        The iteration shape and the shapes moved to the boundary.

    Raises:
        NormalizationError: If the group cannot use one iteration space.
    """
    config = config or NormalizationConfig()
    method = config.parent_shape_method
    child = parent.get_submodule(str(call_node.target))
    placeholders = [n for n in child.graph.nodes if n.op == "placeholder"]
    output_node = next(n for n in child.graph.nodes if n.op == "output")

    # The tail ends at the output, the quantize's input for a terminal
    # dynamic quantize, or the core under an output relayout.
    value = output_node.args[0]
    if not isinstance(value, fx.Node):
        raise NormalizationError(
            "Only tensor outputs are supported", node=output_node
        )
    mx = value.target in DYNAMIC_QUANTIZE_OPS
    core, relayout = _strip_output_relayout(value.args[0] if mx else value)
    map_target = value if mx and not relayout else core
    external_shapes = value.shape if mx else (value.shape,)

    # A pointwise or shape-nop "anchor" means the group is anchorless.
    anchor = get_anchor_node(call_node)
    if anchor is None or is_elementwise_op(anchor) or is_aliasing_op(anchor):
        anchor = None
        iteration_shape = core.shape
    else:
        iteration_shape = anchor.shape
        stream = _propagate(anchor, map_target, anchor)
        if stream is None:
            raise NormalizationError(
                "The fused output does not depend on the strong anchor",
                node=anchor,
            )
        boundary = _boundary_shape(stream, iteration_shape, anchor)
        if boundary != tuple(iteration_shape):
            raise NormalizationError(
                "The tail changes the order or multiplicity of the "
                "anchor-produced stream",
                node=anchor,
            )
    input_shapes = _input_shapes(
        placeholders, anchor, map_target, iteration_shape
    )

    # The value each placeholder takes once the parent's views run.
    views = {}
    inputs = {}
    for index, (placeholder, arg) in enumerate(
        zip(placeholders, call_node.args)
    ):
        inputs[placeholder] = getattr(arg, "value", arg)
        shape, boundary = input_shapes.get(placeholder.name, (None, None))
        if shape == boundary:
            continue
        views[index] = boundary
        try:
            inputs[placeholder] = arg.value.view(boundary)
        except RuntimeError as exc:
            if method == "view" or config.require_contiguous_inputs:
                raise NormalizationError(
                    f"Input cannot be viewed as {boundary}", node=arg
                ) from exc
            inputs[placeholder] = arg.value.reshape(boundary)

    _rewrite_child(child, anchor, relayout, inputs)
    child.graph.eliminate_dead_code()
    child.graph.lint()
    child.recompile()
    result = ShapeProp(child).propagate(*inputs.values())

    output_shapes = _output_shapes(output_node.args[0], external_shapes)

    # Only now does the parent change: input views, then output restores.
    args = list(call_node.args)
    with parent.graph.inserting_before(call_node):
        for index, boundary in views.items():
            view = parent.graph.call_function(
                _RELAYOUT_OPS[method], (args[index], list(boundary))
            )
            # Shallow copy: a deepcopy would touch a FakeTensor ``val``.
            view.meta = dict(args[index].meta)
            propagate_shape(view, parent)
            args[index] = view
    call_node.args = tuple(args)
    set_node_value(call_node, result)

    _restore_outputs(parent, call_node, output_shapes, _RELAYOUT_OPS[method])

    return NormalizationResult(
        iteration_shape=tuple(iteration_shape),
        input_shapes=input_shapes,
        output_shapes=output_shapes,
    )

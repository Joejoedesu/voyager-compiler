import logging
import math
import re
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx import Graph, GraphModule, Interpreter, Node
from torch.fx.node import map_arg
from torch.fx.passes.utils.matcher_utils import InternalMatch, SubgraphMatcher
from torch.library import Library, impl
from torchao.quantization.pt2e.utils import _get_aten_graph_module_for_pattern

from voyager_compiler.codegen.aten_classifier import is_elementwise_op
from voyager_compiler.codegen.node_info import (
    _pair,
    dtype_byte_size,
    get_arg_value,
    is_gemm_op,
    is_nop,
    is_prunable_op,
)
from voyager_compiler.codegen.subgraph import replace_node_with_graph_module
from voyager_compiler.codegen.transform.operator_fusion import _nodes_sequential
from voyager_compiler.export_utils import (
    create_getattr_from_value,
    get_aten_graph_module,
)
from voyager_compiler.quantization.quantizer.xnnpack_quantizer_utils import (
    _convert_scalars_to_attrs,
)
from voyager_compiler.shape_prop import (
    fetch_attr,
    propagate_shape,
    run_op,
    set_node_value,
    written_values,
)

logger = logging.getLogger(__name__)

__all__ = [
    "deduplicate_nodes",
    "replace_interpolate",
    "match_and_fuse_rms_norm",
    "replace_conv2d_with_im2col",
    "extract_input_preprocessor",
    "inline_autocast_modules",
    "fold_constant_generators",
    "remove_fp32_casts",
    "remove_prunable_ops",
    "remove_zero_attention_mask",
    "normalize_attention_operands",
]


class WrapperModule(torch.nn.Module):
    """Wrap a callable in a ``Module`` so it can be exported."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def deduplicate_nodes(model: GraphModule):
    """Collapse identical nodes — same op, target, args and kwargs — into one.

    Args:
        model: Graph module to rewrite in place.

    Returns:
        A mapping from each erased node to the node that replaced it.
    """
    seen = {}
    mapping = {}
    graph = model.graph

    for node in list(graph.nodes):
        if node.op in ("placeholder", "output"):
            continue

        key = (
            node.op,
            node.target,
            tuple(node.args),
            frozenset(node.kwargs.items()),
        )

        if key in seen:
            orig = seen[key]
            node.replace_all_uses_with(orig)
            graph.erase_node(node)
            mapping[node] = orig
        else:
            seen[key] = node

    for old, new in mapping.items():
        logger.debug(f"Deduplicated {old} to {new}")

    graph.lint()
    model.recompile()
    return mapping


def remove_prunable_ops(model: GraphModule) -> None:
    """Delete identity ops — a full ``slice``, a unit ``expand``, a same-dtype
    ``to``, a zero-prob ``dropout`` (see ``is_prunable_op``) — by rewiring each
    to its input.  They survive fusion (they have users, so dead-code
    elimination skips them); dropping them shrinks the graph every later pass
    and the bufferizer must walk.
    """
    graph = model.graph
    removed = 0
    for n in list(graph.nodes):
        if is_prunable_op(n):
            n.replace_all_uses_with(n.all_input_nodes[0])
            graph.erase_node(n)
            removed += 1
    if removed:
        graph.lint()
        model.recompile()
    logger.debug("[transform] removed %d prunable ops", removed)


def replace_interpolate():
    template = (
        "interpolate(Tensor input, SymInt[] size, float[]? scale_factor = None,"
        "str mode = 'nearest', bool? align_corners = None, "
        "bool? recompute_scale_factor = None, bool antialias = False) -> Tensor"
    )

    global m
    m = Library("custom", "DEF")
    m.define(template)

    orig_interpolate = torch.nn.functional.interpolate

    @impl(m, "interpolate", "CompositeExplicitAutograd")
    def interpolate(*args, **kwargs):
        return orig_interpolate(*args, **kwargs)

    torch.nn.functional.interpolate = torch.ops.custom.interpolate


def match_and_fuse_rms_norm(
    model: GraphModule,
    rms_norm: torch.nn.Module,
    example_input,
    convert_scalars_to_attrs=False,
):
    """Fuse each decomposed RMSNorm in ``model`` into one ``aten.rms_norm``.

    The pattern is ``rms_norm`` traced to ATen; literals must match, so a
    match carries the module's eps, which the fused op takes.  The fused
    op's weight is what the pattern scales the normalized input by: the
    ``weight`` itself, or an expression of it (Gemma's ``1 + weight``) cast
    to the input's dtype, which constant folding later collapses.

    Args:
        model: The graph to rewrite.
        rms_norm: An RMSNorm module of the model (e.g. ``LlamaRMSNorm``),
            with a ``weight`` and a ``variance_epsilon`` or ``eps``.
        example_input: Inputs to trace ``rms_norm`` with.
        convert_scalars_to_attrs: Turn the pattern's scalars into attributes
            before matching.
    """
    original_graph = model.graph

    pattern = get_aten_graph_module(rms_norm, example_input)
    if convert_scalars_to_attrs:
        _convert_scalars_to_attrs(pattern)
    pattern_graph = pattern.graph

    matcher = SubgraphMatcher(
        pattern_graph,
        match_output=False,
        match_placeholder=False,
        remove_overlapping_matches=True,
        ignore_literals=False,
    )
    _matches: List[InternalMatch] = matcher.match(original_graph)
    logger.info(f"Found {len(_matches)} matches")

    # The scale is the operand of the last multiply that descends from the
    # weight; the other operand is the normalized input.
    from_weight = set()
    scale = None
    for n in pattern_graph.nodes:
        if n.target == "weight" or from_weight & set(n.all_input_nodes):
            from_weight.add(n)
        if n.target is torch.ops.aten.mul.Tensor:
            weighted = [a for a in n.args if a in from_weight]
            if len(weighted) == 1:
                scale = weighted[0]
    x = next(n for n in pattern_graph.nodes if n.op == "placeholder")
    cast = scale.meta["val"].dtype != x.meta["val"].dtype
    eps = getattr(rms_norm, "variance_epsilon", None) or rms_norm.eps

    for match in _matches:
        input_node = match.placeholder_nodes[0]
        output_node = match.returning_nodes[0]
        input_val = input_node.meta["val"]
        weight = match.nodes_map[scale]
        with original_graph.inserting_before(output_node):
            if cast:
                weight = original_graph.call_function(
                    torch.ops.aten.to.dtype, (weight, input_val.dtype)
                )
                propagate_shape(weight, model)
            new_node = original_graph.call_function(
                torch.ops.aten.rms_norm.default,
                (input_node, [input_val.shape[-1]], weight, eps),
            )

        output_node.replace_all_uses_with(new_node)
        original_graph.erase_node(output_node)

        new_node.meta = output_node.meta

    original_graph.lint()
    original_graph.eliminate_dead_code()
    model.recompile()


def _get_im2col_gemm_pattern(
    output_shape: Tuple[int],
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    reduction_multiple: int = 1,
):

    def _im2col_gemm_pattern(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
    ) -> Tuple[torch.Tensor]:
        inp_unf = F.unfold(input, weight.shape[-2:], dilation, padding, stride)
        wt = weight.view(weight.size(0), -1)
        matrix = inp_unf.transpose(-1, -2)
        tail = (-wt.shape[-1]) % reduction_multiple
        if tail:
            matrix = F.pad(matrix, (0, tail))
            wt = F.pad(wt, (0, tail))
        out_unf = F.linear(matrix, wt, bias)
        out = out_unf.transpose(-1, -2).view(*output_shape)
        return out

    return WrapperModule(_im2col_gemm_pattern)


def replace_conv2d_with_im2col(
    model: GraphModule, *, predicate=None, reduction_multiple=1
):
    """
    Replace Conv2d operations that has input channel dimension equal to 3 with
    In2col operations in the given FX graph module.  Usually this is the first
    Conv2D layer in torchvision models.

    Args:
        model (GraphModule): The FX graph module to transform, before quantization.
        predicate: Optional ``(conv_node) -> bool`` selecting eligible nodes.
            The caller owns workload/target policy and the host input boundary;
            this helper only rewrites the graph. Without a predicate, rewrite
            all eligible three-channel convolutions as before.
        reduction_multiple: Optional packed GEMM reduction alignment. Zero-pad
            both the unfolded input and stored weight explicitly, before
            quantization. The binding must permit the input storage expansion.

    Returns:
        GraphModule: The transformed FX graph module.
    """
    if not isinstance(reduction_multiple, int) or reduction_multiple < 1:
        raise ValueError("reduction_multiple must be a positive integer")
    graph = model.graph

    def get_shape(n):
        if n.meta and "val" in n.meta:
            return n.meta["val"].shape
        return getattr(n, "shape", None)

    for node in list(graph.nodes):
        if node.target != torch.ops.aten.conv2d.default:
            continue

        input_node = node.args[0]
        weight_node = node.args[1]
        bias_node = get_arg_value(node, 2, "bias")
        stride = get_arg_value(node, 3, "stride", 1)
        padding = get_arg_value(node, 4, "padding", 0)
        dilation = get_arg_value(node, 5, "dilation", 1)
        group = get_arg_value(node, 6, "groups", 1)

        input_shape = get_shape(input_node)
        weight_shape = get_shape(weight_node)
        output_shape = get_shape(node)

        if (
            input_shape is None
            or input_shape[1] != 3
            or output_shape is None
            or group != 1
        ):
            continue

        if predicate is not None and not predicate(node):
            continue

        logger.info(f"Replacing Conv2d node {node} with Im2col + GEMM")

        _example_inputs = (
            torch.randn(input_shape),
            torch.randn(weight_shape),
            torch.randn((weight_shape[0],)) if bias_node is not None else None,
        )

        match_pattern = _get_im2col_gemm_pattern(
            output_shape,
            _pair(stride),
            _pair(padding),
            _pair(dilation),
            reduction_multiple,
        )
        match_pattern = _get_aten_graph_module_for_pattern(
            match_pattern,
            _example_inputs,
        )

        val_maps = {}
        output = replace_node_with_graph_module(
            model, node, match_pattern, val_maps
        )[0]
        graph.erase_node(node)

        # Materialize the explicitly requested weight representation. This is
        # binding-owned padding, not the non-inflating constant-preparation pass.
        linear_node = next((n for n in val_maps.values() if is_gemm_op(n)))
        view_node = linear_node.args[1]

        val = fetch_attr(model, weight_node.target).detach()
        val = val.reshape(val.size(0), -1)
        tail = (-val.shape[-1]) % reduction_multiple
        if tail:
            val = F.pad(val, (0, tail))

        with graph.inserting_before(view_node):
            new_weight = create_getattr_from_value(
                model, graph, f"{weight_node.target}_im2col", val
            )

        propagate_shape(new_weight, model)
        view_node.replace_all_uses_with(new_weight)

        # Move elementwise operations after view to after the linear node
        order = {n: i for i, n in enumerate(graph.nodes)}
        fusable_ops = []
        next_node = next(iter(output.users))
        while is_elementwise_op(next_node):
            chain = fusable_ops + [next_node]
            if _nodes_sequential(chain, order):
                fusable_ops.append(next_node)
            else:
                break
            # Stop fusing if last node is a quantize op
            if (
                len(next_node.users) != 1
                or next_node.target == torch.ops.quantized_ops.quantize.default
            ):
                break
            next_node = next(iter(next_node.users))

        if not fusable_ops:
            continue

        linear_node.replace_all_uses_with(fusable_ops[-1])
        fusable_ops[0].replace_input_with(output, linear_node)
        next_node.replace_input_with(fusable_ops[-1], output)

        for n in reversed(fusable_ops):
            linear_node.append(n)

    graph.eliminate_dead_code()
    graph.lint()
    model.recompile()
    return model


def extract_input_preprocessor(
    model: GraphModule, input_name=None, *, additional_ops=()
):
    """
    Extract the input preprocessing operations from the given FX GraphModule
    and create a separate GraphModule.

    Args:
        model (GraphModule): The FX graph module to transform.
        input_name: Target of the placeholder to peel preprocessing from.
            Defaults to the first placeholder (the activation input); pass
            e.g. ``"attention_mask"`` to extract that input's quantize.

    Returns:
        GraphModule: The transformed FX graph module with the input
            preprocessor extracted.
    """
    placeholder = next(
        iter(
            n
            for n in model.graph.nodes
            if n.op == "placeholder"
            and (input_name is None or n.target == input_name)
        )
    )
    preprocess_nodes = [placeholder]

    user = next(iter(placeholder.users))

    while (
        is_nop(user)
        or user.target in additional_ops
        or user.target
        in [
            torch.ops.aten.permute.default,
            torch.ops.aten.transpose.int,
            torch.ops.aten.im2col.default,
            torch.ops.aten.pad.default,
            torch.ops.quantized_ops.quantize.default,
        ]
    ):
        preprocess_nodes.extend(
            n for n in user.all_input_nodes if n not in preprocess_nodes
        )
        preprocess_nodes.append(user)
        user = next(iter(user.users))

    m = torch.nn.Module()

    new_graph = torch.fx.Graph()
    value_remap = {}
    for node in preprocess_nodes:
        if node.op == "placeholder":
            value_remap[node] = new_graph.placeholder(node.name)
        else:
            value_remap[node] = new_graph.node_copy(
                node, lambda n: value_remap[n]
            )

            if node.op == "get_attr":
                param = fetch_attr(model, node.target)
                m.register_buffer(node.target, param)
    new_graph.output(value_remap[preprocess_nodes[-1]])
    new_graph.lint()
    new_graph.print_tabular()

    with model.graph.inserting_before(placeholder):
        new_placeholder = model.graph.placeholder(
            f"{placeholder.name}_preprocess"
        )
    preprocess_nodes[-1].replace_all_uses_with(new_placeholder)

    new_placeholder.meta["dtype"] = preprocess_nodes[-1].meta.get("dtype")
    set_node_value(new_placeholder, preprocess_nodes[-1].value)

    model.graph.lint()
    model.graph.eliminate_dead_code()
    # Placeholder node needs to be manually erased
    model.graph.erase_node(placeholder)
    model.recompile()
    return model, GraphModule(m, new_graph)


def inline_autocast_modules(model: torch.fx.GraphModule):
    """Inline wrap HOPs (``autocast`` / ``set_grad_enabled``) by replacing the
    wrap node with a direct ``call_module`` to its wrapped submodule.

    torch.export emits ``WrapWithAutocast`` for ``torch.autocast`` regions and
    ``wrap_with_set_grad_enabled`` for ``@torch.no_grad`` regions (e.g. Llama's
    ``LlamaRotaryEmbedding.forward``).  Both wrap a submodule that the rest of
    the lowering can only handle once flattened into the parent graph.  The
    wrapped function is at ``args[fn_idx]`` and its operands follow it.
    """
    graph = model.graph
    named_modules = dict(model.named_modules())
    set_grad = torch.ops.higher_order.wrap_with_set_grad_enabled

    for node in list(graph.nodes):
        if isinstance(
            node.target, torch._higher_order_ops.wrap.WrapWithAutocast
        ):
            fn_idx = 4
        elif node.target is set_grad:
            fn_idx = 1
        else:
            continue

        wrapped_func = node.args[fn_idx]
        mod = named_modules.get(wrapped_func.target, None)
        if mod is None:
            continue

        with graph.inserting_before(node):
            new_node = graph.call_module(
                wrapped_func.target, tuple(node.args[fn_idx + 1 :])
            )
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)

        replace_node_with_graph_module(model, new_node, mod)

    graph.eliminate_dead_code()
    model.graph.lint()
    model.recompile()


# Deliberately finite: an unrecognized function, random op or stateful op
# must not become a constant just because its FX inputs happen to be static.
_STATIC_TRANSFORMS = {
    "view",
    "reshape",
    "_unsafe_view",
    "flatten",
    "unflatten",
    "squeeze",
    "unsqueeze",
    "transpose",
    "permute",
    "t",
    "movedim",
    "swapaxes",
    "swapdims",
    "alias",
    "detach",
    "clone",
    "contiguous",
    "lift_fresh_copy",
    "slice",
    "narrow",
    "select",
    "index_select",
    "gather",
    "expand",
    "_to_copy",
    "to",
    "type_as",
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "abs",
    "clamp",
    "clamp_min",
    "clamp_max",
    "minimum",
    "maximum",
    "reciprocal",
    "sqrt",
    "rsqrt",
    "pow",
    "exp",
    "log",
    "sin",
    "cos",
    "sum",
    "mean",
    "amax",
    "amin",
}
_CONSTANT_GENERATORS = {
    "arange",
    "zeros",
    "ones",
    "full",
    "eye",
    "scalar_tensor",
}
_CASTS = {"_to_copy", "to", "type_as"}


def constant_storage_bytes(node: Node) -> int:
    """Encoded payload bytes of one independently stored constant.

    Quantized metadata, explicit physical padding and sub-byte rounding count.
    Distinct attributes are charged separately even if their eager tensors
    alias: the emitter stores them separately. This is not compiler peak RAM
    or a promise about a target's later packing/alignment overhead.
    """
    value = node.value
    count = max(
        value.numel(), math.prod(node.meta.get("physical_shape", value.shape))
    )
    dtype = node.meta.get("dtype") or value.dtype
    return math.ceil(count * dtype_byte_size(dtype))


def _static_op_name(node):
    schema = getattr(node.target, "_schema", None)
    if (
        schema is None
        or schema.is_mutable
        or not schema.name.startswith("aten::")
    ):
        return None
    return schema.name.removeprefix("aten::")


def _constant_producer(node, model):
    """Retain a deferred generator's recipe without depending on dead attrs."""
    root, graph, remap = torch.nn.Module(), Graph(), {}
    for i, inp in enumerate(node.all_input_nodes):
        name = f"input_{i}"
        if (producer := inp.meta.get("producer")) is not None:
            root.add_module(name, producer)
            remap[inp] = graph.call_module(name)
        else:
            root.register_buffer(name, fetch_attr(model, inp.target))
            remap[inp] = graph.get_attr(name)
    graph.output(graph.node_copy(node, lambda n: remap[n]))
    return GraphModule(root, graph)


def _fold_constant(model, node, *, generated=False):
    graph = model.graph
    resolve = lambda n: fetch_attr(model, n.target)
    const = run_op(
        node.target, map_arg(node.args, resolve), map_arg(node.kwargs, resolve)
    )
    # Real parameter-derived values need no recipe holding the old weights.
    producer = (
        _constant_producer(node, model)
        if isinstance(const, FakeTensor)
        else None
    )
    src = next((n.target for n in node.all_input_nodes), "const")
    prefix = (
        node.meta.get("constant_name")
        or re.sub(r"_folded(_\d+)?$", "", str(src)) + "_folded"
    )
    # A compact contiguous buffer avoids retaining a slice's larger backing.
    const = const.contiguous()
    with graph.inserting_before(node):
        attr = create_getattr_from_value(model, graph, prefix, const, producer)
    for key in ("dtype", "physical_shape"):
        if key in node.meta:
            attr.meta[key] = node.meta[key]
    set_node_value(attr, const)
    if generated:
        attr.meta["constant_generator"] = True
    inputs = node.all_input_nodes
    node.replace_all_uses_with(attr)
    graph.erase_node(node)
    for inp in inputs:
        if inp.op != "get_attr" or inp.users:
            continue
        graph.erase_node(inp)
        if any(
            n.op == "get_attr" and n.target == inp.target for n in graph.nodes
        ):
            continue
        # An opaque module can still use an attribute internally.
        if any(
            n.op == "call_module" and inp.target.startswith(str(n.target) + ".")
            for n in graph.nodes
        ):
            continue
        path, _, name = str(inp.target).rpartition(".")
        owner = fetch_attr(model, path) if path else model
        delattr(owner, name)
    return attr


def fold_constant_generators(model: GraphModule, *, phase="normalize"):
    """Prepare static tensors without growing their stored representations.

    Retain existing deterministic generator setup (including transitive mask
    and position-table computation and shape-only ``full_like``). It creates
    constants; its final prepared bytes are reported separately. Then discover
    bounded, pure parameter-only transform components, and fold a component
    only when its final outputs
    cost no more bytes than the input attributes it makes entirely unused.
    Shared originals, duplicate get_attrs and every escaping output count.
    Intermediate transforms are not charged as stored final representations.

    Supported transforms cover shape/layout, subsetting, non-widening casts,
    elementwise arithmetic and reductions. Parameter preparation excludes
    expanding materialization and dequantization. Neither path bakes random,
    unrecognized custom or stateful functions, or reads from mutable buffers.
    Rejected operations remain ordinary graph operations for bufferization.
    This runs again after layout selection so target-requested transposes use
    the same policy. It is not a cost search over prepacking alternatives.
    """
    graph = model.graph
    # Exported tensor literals often have lift_fresh_copy -> detach_. For a
    # fresh, already non-grad value the latter changes no state; functionalize
    # that identity so it cannot block deterministic constant setup.
    for node in graph.nodes:
        if node.target is torch.ops.aten.detach_.default:
            source = node.args[0]
            if (
                source.target is torch.ops.aten.lift_fresh_copy.default
                and not getattr(source.value, "requires_grad", True)
                and not getattr(source.meta.get("val"), "requires_grad", False)
            ):
                node.target = torch.ops.aten.detach.default
    for node in list(graph.nodes):
        if node.target is torch.ops.aten.full_like.default:
            value = node.value
            with graph.inserting_before(node):
                full = graph.call_function(
                    torch.ops.aten.full.default,
                    (list(value.shape), node.args[1]),
                    {"dtype": value.dtype, "device": value.device},
                )
            propagate_shape(full, model)
            node.replace_all_uses_with(full)
            graph.erase_node(node)
            node = full

    mutated = written_values(graph)
    written = {n.target for n in mutated if n.op == "get_attr"}
    roots = {
        n
        for n in graph.nodes
        if n.op == "get_attr"
        and n.target not in written
        and isinstance(getattr(n, "value", None), torch.Tensor)
    }
    candidates, parent = set(), {}
    known = set(roots)
    generated = {n for n in roots if n.meta.get("constant_generator")}

    def find(n):
        parent.setdefault(n, n)
        if parent[n] is not n:
            parent[n] = find(parent[n])
        return parent[n]

    # Two get_attrs for the same attribute are one deployment allocation.
    by_target = {}
    for n in roots:
        if n.target in by_target:
            parent[find(n)] = find(by_target[n.target])
        by_target[n.target] = n
    for node in graph.nodes:
        name = _static_op_name(node) if node.op == "call_function" else None
        inputs = node.all_input_nodes
        if (
            node in mutated
            or any(n not in known for n in inputs)
            or not isinstance(getattr(node, "value", None), torch.Tensor)
        ):
            continue
        # Existing deterministic generator setup (causal masks, position
        # tables) is constant creation, not optional weight prepacking.
        # Preserve the entire setup chain, including broadcasts/matmuls.
        generator = (not inputs and name in _CONSTANT_GENERATORS) or (
            (
                name is not None
                or node.target is torch.ops.quantized_ops.quantize.default
            )
            and any(n in generated for n in inputs)
        )
        tags = getattr(node.target, "tags", ())
        if generator and not any(
            tag in tags
            for tag in (
                torch.Tag.nondeterministic_seeded,
                torch.Tag.nondeterministic_bitwise,
            )
        ):
            if name == "expand" and node.value.shape != inputs[0].value.shape:
                continue
            generated.add(node)
            known.add(node)
            continue
        if name not in _STATIC_TRANSFORMS or not inputs:
            continue
        # No per-op enlargement, including broadcast products and casts.
        if constant_storage_bytes(node) > max(
            constant_storage_bytes(n) for n in inputs
        ):
            continue
        if (
            name in _CASTS
            and node.value.element_size() > inputs[0].value.element_size()
        ):
            continue
        if name == "expand" and node.value.shape != inputs[0].value.shape:
            continue
        candidates.add(node)
        known.add(node)
        for inp in inputs:
            parent[find(inp)] = find(node)

    components = {}
    for node in graph.nodes:
        if node in candidates or node in roots and node in parent:
            components.setdefault(find(node), set()).add(node)
    accepted, decisions = set(), []
    for members in components.values():
        ops = members & candidates
        if not ops:
            continue
        outputs = [
            n
            for n in graph.nodes
            if n in ops and any(u not in ops for u in n.users)
        ]
        sources = {n.target for n in members & roots}
        released = {
            target
            for target in sources
            if all(
                u in ops for n in roots if n.target == target for u in n.users
            )
        }
        before = sum(constant_storage_bytes(by_target[t]) for t in released)
        after = sum(constant_storage_bytes(n) for n in outputs)
        permitted = after <= before
        decisions.append(
            dict(
                inputs=sorted(sources),
                outputs=[n.name for n in outputs],
                released_bytes=before,
                prepared_bytes=after,
                folded=permitted,
                reason=(
                    "no_storage_growth"
                    if permitted
                    else "retained_inputs_or_extra_outputs"
                ),
            )
        )
        if permitted:
            accepted.update(ops)

    prepared_generators = set()
    for node in list(graph.nodes):
        if node.op != "call_function" or any(
            n.op != "get_attr" for n in node.all_input_nodes
        ):
            continue
        if node in accepted or node in generated:
            attr = _fold_constant(model, node, generated=node in generated)
            if node in generated:
                prepared_generators.add(attr)
    graph.eliminate_dead_code()
    graph.lint()
    model.recompile()
    # Keep empty module namespaces: later parameter-based node naming uses
    # them to avoid collisions. Prepared attributes were retired above.
    model.meta.setdefault("constant_folding", []).append(
        dict(
            phase=phase,
            generator_bytes=sum(
                constant_storage_bytes(n)
                for n in graph.nodes
                if n in prepared_generators
            ),
            components=decisions,
        )
    )


def remove_zero_attention_mask(model: GraphModule, example_inputs):
    """Drop additive attention masks that are provably all-zero.

    Eager attention in transformers materializes ``add(scores, mask)`` where
    ``mask = where(valid, 0.0, min_value)``.  For bidirectional models with no
    padding (e.g. ViT) every position is valid, so the mask is identically
    zero and the add is a no-op.  The mask is evaluated on ``example_inputs``
    and an add is bypassed only when its mask operand is in fact all zero;
    dead-code elimination then removes the mask-building chain.  Masks that
    are not all zero (e.g. real causal masks) are left untouched.
    """
    masks = {}

    class _Capture(Interpreter):
        def run_node(self, n):
            out = super().run_node(n)
            if n.op == "call_function" and "where" in str(n.target):
                masks[n] = out
            return out

    _Capture(model).run(*example_inputs)

    removed = 0
    for mask_node, value in masks.items():
        # Guard: only drop the add when the mask is genuinely all zero.
        if not isinstance(value, torch.Tensor) or not bool((value == 0).all()):
            continue
        for add_node in list(mask_node.users):
            if add_node.target != torch.ops.aten.add.Tensor:
                continue
            others = [a for a in add_node.args if a is not mask_node]
            if len(others) != 1:
                continue
            add_node.replace_all_uses_with(others[0])
            model.graph.erase_node(add_node)
            removed += 1

    logger.info(f"Removed {removed} zero attention-mask add(s)")
    model.graph.eliminate_dead_code()
    model.graph.lint()
    model.recompile()
    return model


_SDPA = torch.ops.aten.scaled_dot_product_attention.default


def _repeated_heads(node: Node):
    """``(x, g)`` when ``node`` is ``repeat_kv`` of ``x`` -- ``unsqueeze(x,
    2)``, an ``expand`` of the new dim to ``g`` and a ``reshape`` of ``[b,
    h, s, d]`` to ``[b, h·g, s, d]`` -- else ``None``."""
    if node.target not in (
        torch.ops.aten.reshape.default,
        torch.ops.aten.view.default,
    ):
        return None
    expand = node.args[0]
    if expand.target is not torch.ops.aten.expand.default:
        return None
    unsqueeze = expand.args[0]
    if (
        unsqueeze.target is not torch.ops.aten.unsqueeze.default
        or unsqueeze.args[1] != 2
    ):
        return None
    x = unsqueeze.args[0]
    if x.value.ndim != 4:
        return None
    b, h, s, d = x.value.shape
    g = expand.value.shape[2]
    expanded, repeated = tuple(expand.value.shape), tuple(node.value.shape)
    if expanded != (b, h, g, s, d) or repeated != (b, h * g, s, d):
        return None
    return x, g


def normalize_attention_operands(model: GraphModule) -> GraphModule:
    """Hand each ``scaled_dot_product_attention`` its operands as the flash
    attention kernel takes them.

    A key and value whose heads a GQA repeat (``_repeated_heads``) multiplies
    by the same group are read unrepeated under ``enable_gqa``, so the kernel
    folds each group of query heads onto its KV head.  A bool ``attn_mask``
    becomes the additive one, zero where it keeps and the query dtype's
    minimum elsewhere.

    Args:
        model: The graph, rewritten in place.

    Returns:
        ``model``.
    """
    graph = model.graph
    changed = False
    for node in [n for n in graph.nodes if n.target is _SDPA]:
        repeats = [_repeated_heads(a) for a in node.args[1:3]]
        if (
            not node.kwargs.get("enable_gqa", False)
            and None not in repeats
            and repeats[0][1] == repeats[1][1]
        ):
            for position, (x, _) in zip((1, 2), repeats):
                repeat = node.args[position]
                node.update_arg(position, x)
                while not repeat.users and repeat is not x:
                    source = repeat.args[0]
                    graph.erase_node(repeat)
                    repeat = source
            node.update_kwarg("enable_gqa", True)
            changed = True
        mask = get_arg_value(node, 3, "attn_mask", None)
        if isinstance(mask, Node) and mask.value.dtype == torch.bool:
            dtype = node.args[0].value.dtype
            with graph.inserting_before(node):
                zero = create_getattr_from_value(
                    model, graph, "attention_zero", torch.tensor(0, dtype=dtype)
                )
                additive = graph.call_function(
                    torch.ops.aten.where.ScalarOther,
                    (mask, zero, torch.finfo(dtype).min),
                )
            propagate_shape(zero, model)
            propagate_shape(additive, model)
            node.replace_input_with(mask, additive)
            changed = True
    if changed:
        graph.lint()
        model.recompile()
    return model


# Ops that cast a tensor to another dtype.
_CASTS = (
    torch.ops.aten._to_copy.default,
    torch.ops.aten.to.dtype,
    torch.ops.aten.to.dtype_layout,
)

_HALF = (torch.bfloat16, torch.float16)


def remove_fp32_casts(model: torch.fx.GraphModule):
    """Compute in 16-bit float where the model casts up to float32.

    For an accelerator with no float32 datapath: a ``softmax`` asked for a
    float32 result gives it in its input's dtype, and a cast of a bfloat16
    or float16 tensor to float32 is dropped, so what reads it computes in
    the narrower dtype.  A float32 tensor that then meets a 16-bit one is
    made in 16-bit too, back through whatever built it from constants and
    integers -- a ``full``, a cast of integers -- or it would carry the
    computation back to float32: a loss's gradient starts as one, and an
    embedding's backward accumulates into one.  The casts back down then
    convert nothing, and ``remove_prunable_ops`` drops them.  Float32 that
    meets no 16-bit tensor -- a float32 buffer, or RoPE's frequencies made
    from integers -- stays float32, as does a 0-d tensor, which does not
    decide the dtype of what it meets.

    Args:
        model: The graph to rewrite in place.

    Returns:
        ``model``.
    """
    graph = model.graph
    # The dtype a node computes in once the casts are gone, where that is
    # not the one it was traced with.
    narrowed = {}

    def value(node):
        return getattr(node, "value", node.meta.get("val"))

    def dtype(node):
        return narrowed.get(node, getattr(value(node), "dtype", None))

    def floats(node):
        return [
            a
            for a in node.all_input_nodes
            if dtype(a) is not None and dtype(a).is_floating_point
        ]

    def narrow(node, half):
        """Build ``node``, a float32 tensor, in ``half``: a cast or a
        generator through its dtype argument, any other op from its
        operands, back to the graph's inputs."""
        stack = [node]
        while stack:
            node = stack.pop()
            if node.op != "call_function" or dtype(node) != torch.float32:
                continue
            narrowed[node] = half
            if "dtype" in node.kwargs:
                node.update_kwarg("dtype", half)
            elif node.target in _CASTS:
                node.update_arg(1, half)
            else:
                stack.extend(floats(node))

    for node in list(graph.nodes):
        if node.target == torch.ops.aten.softmax.int:
            node.args = node.args[:2]
            narrowed[node] = dtype(node.args[0])
            continue
        if (
            node.target in _CASTS
            and get_arg_value(node, 1, "dtype") == torch.float32
            and dtype(node.args[0]) in _HALF
        ):
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
            continue
        if dtype(node) != torch.float32 or "dtype" in node.kwargs:
            continue
        inputs = [a for a in floats(node) if value(a).ndim]
        half = {dtype(a) for a in inputs} & set(_HALF)
        if len(half) != 1:
            continue
        (half,) = half
        for a in inputs:
            narrow(a, half)
        if all(dtype(a) == half for a in inputs):
            narrowed[node] = half

    graph.lint()
    model.recompile()
    return model

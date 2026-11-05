"""Guarded frontend recognition and layout realization for Trainium regions."""

import copy
import math

import torch
from torch.fx import Node

from voyager_compiler.codegen.node_info import get_arg_value, swaps_last_two_dims
from voyager_compiler.shape_prop import ShapeProp, fake_like
from torch.utils._pytree import tree_flatten


def _shapes(prepared):
    args, _ = tree_flatten((prepared.args, prepared.kwargs))
    ShapeProp(prepared.graph).propagate(*map(fake_like, args))


def _attention_expression(module):
    """Expose unmasked FP32 attention to the existing streaming builder.

    Only consume private scores/probabilities. Explicit low-precision
    intermediate rounding, masks, non-last-axis softmax and observed scores
    are not equivalent to this SDPA contract and retain their original graph.
    The literal score scale is preserved; it is not replaced by 1/sqrt(D).
    """
    aten = torch.ops.aten
    for output in list(module.graph.nodes):
        if output.target != aten.matmul.default:
            continue
        probs, value = output.args[:2]
        if not isinstance(probs, Node) or probs.target != aten.softmax.int:
            continue
        if len(probs.users) != 1 or get_arg_value(probs, 2, 'dtype') is not None:
            continue
        if get_arg_value(probs, 1, 'dim') not in (-1, probs.value.ndim - 1):
            continue
        scaled = probs.args[0]
        if scaled.target not in (aten.div.Tensor, aten.mul.Tensor):
            continue
        scores, literal = scaled.args[:2]
        if (not isinstance(scores, Node) or scores.target != aten.matmul.default
                or type(literal) not in (int, float)
                or not math.isfinite(literal) or literal <= 0
                or len(scaled.users) != 1 or len(scores.users) != 1):
            continue
        query, key_t = scores.args[:2]
        if not swaps_last_two_dims(key_t):
            continue
        key = key_t.args[0]
        tensors = (query, key, value, scores, scaled, probs, output)
        if any(not isinstance(getattr(n, 'value', None), torch.Tensor)
               or n.value.dtype != torch.float32 for n in tensors):
            continue
        if (query.value.ndim < 3 or key.value.ndim != query.value.ndim
                or value.value.ndim != query.value.ndim
                or query.value.shape[:-2] != key.value.shape[:-2]
                or key.value.shape[:-2] != value.value.shape[:-2]
                or query.value.shape[-1] != key.value.shape[-1]
                or query.value.shape[-1] != value.value.shape[-1]
                or key.value.shape[-2] != value.value.shape[-2]):
            continue
        scale = 1.0 / literal if scaled.target == aten.div.Tensor else literal
        output.target = aten.scaled_dot_product_attention.default
        output.args = (query, key, value)
        output.kwargs = dict(attn_mask=None, dropout_p=0.0, is_causal=False,
                             scale=scale, enable_gqa=False)
    module.graph.eliminate_dead_code()
    module.graph.lint()
    module.recompile()


def legalize(prepared, config, options):
    """Canonicalize matrix layouts and guarded attention/RMS expressions.

    RMS recognition restricts scaling to a power-of-two last dimension:
    division before reduction is then exact for ordinary finite values. This is
    an accuracy-qualified RMS candidate, not a promise of bitwise equivalence
    across underflow/overflow or different reduction orders. Other axes, dtypes,
    nonzero epsilon expressions and mixed inputs retain their original graph.
    """
    _shapes(prepared)
    _attention_expression(prepared.graph)
    _shapes(prepared)
    graph = prepared.graph.graph
    aten = torch.ops.aten
    for node in list(graph.nodes):
        # torch.Tensor.T exports as numpy_T, which the common matrix-layout
        # recognizer does not classify as a last-two-dimension transpose.
        # For rank2 this is exactly the already supported transpose form.
        if node.target == aten.numpy_T.default and node.args[0].value.ndim == 2:
            node.target = aten.transpose.int
            node.args = (node.args[0], 0, 1)
            node.kwargs = {}
            continue
        if node.target != aten.div.Tensor or len(node.args) != 2:
            continue
        x, root = node.args
        if not isinstance(x, Node) or not isinstance(root, Node):
            continue
        if root.target != aten.sqrt.default:
            continue
        total = root.args[0]
        if not isinstance(total, Node) or total.target != aten.sum.dim_IntList:
            continue
        scaled = total.args[0]
        if not isinstance(scaled, Node) or scaled.target != aten.div.Tensor or len(scaled.args) != 2:
            continue
        square, divisor = scaled.args
        if not isinstance(square, Node) or square.target != aten.square.default:
            continue
        if square.args[0] is not x or node.kwargs or scaled.kwargs:
            continue
        shape = tuple(x.value.shape)
        if len(shape) < 2 or x.value.dtype != torch.float32:
            continue
        width = shape[-1]
        if not isinstance(width, int) or width <= 0 or width & (width - 1):
            continue
        if type(divisor) not in (int, float) or divisor != width:
            continue
        dims = get_arg_value(total, 1, 'dim')
        if dims not in ([-1], [len(shape) - 1], (-1,), (len(shape) - 1,)):
            continue
        if get_arg_value(total, 2, 'keepdim', False) is not True:
            continue
        if total.kwargs.get('dtype') not in (None, torch.float32):
            continue
        node.target = aten.rms_norm.default
        node.args = (x, [width], None, 0.0)
        node.kwargs = {}
    graph.eliminate_dead_code()
    graph.lint()
    prepared.graph.recompile()
    _shapes(prepared)


def lower(prepared, config, options):
    """Remove inverse input transposes and expose shared matrix/DAG candidates.

    Layout normalization can insert transpose(norm(x)) for a cx,ck product.
    Inside a fused region, use the equivalent xc,ck contraction instead: the
    existing GemmSteps prologue and transpose buffers realize this layout and
    its model prices their work. Cancel an external transpose only with its
    sole inverse consumer inside a region; preserve its other external uses.
    """
    # Cancel an external transpose with its sole inverse at a region's input.
    # Otherwise the host materializes one layout and the device immediately
    # transposes it back, even though the original input already has that form.
    for module in list(prepared.graph.modules()):
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for call in list(module.graph.nodes):
            sub = call.meta.get('submodule') if call.op == 'call_module' else None
            if not isinstance(sub, torch.fx.GraphModule):
                continue
            placeholders = [n for n in sub.graph.nodes if n.op == 'placeholder']
            args = list(call.args)
            for i, (ph, outer) in enumerate(zip(placeholders, args)):
                if not isinstance(outer, Node) or not swaps_last_two_dims(outer) or len(ph.users) != 1:
                    continue
                inner = next(iter(ph.users))
                if not swaps_last_two_dims(inner):
                    continue
                args[i] = outer.args[0]
                inner.replace_all_uses_with(ph)
                sub.graph.erase_node(inner)
            call.args = tuple(args)
            sub.graph.lint(); sub.recompile()
        module.graph.eliminate_dead_code(); module.graph.lint(); module.recompile()
    _shapes(prepared)
    for module in prepared.graph.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        changed = False
        for node in list(module.graph.nodes):
            if node.target != torch.ops.quantized_ops.matmul.default:
                continue
            if node.kwargs.get('layout') != 'cx,ck':
                continue
            transposed = node.args[0]
            if not swaps_last_two_dims(transposed):
                continue
            norm = transposed.args[0]
            if norm.target != torch.ops.aten.rms_norm.default:
                continue
            node.args = (norm, *node.args[1:])
            node.kwargs = dict(node.kwargs, layout='xc,ck')
            changed = True
        if changed:
            module.graph.eliminate_dead_code()
            module.graph.lint()
            module.recompile()
    from .scan import OP as SCAN_OP, ALGORITHM_TAG
    scan_algorithm = options.get(ALGORITHM_TAG, "direct")
    if scan_algorithm not in ("direct", "integrating_factor"):
        raise ValueError(f"unknown trainium_scan_algorithm: {scan_algorithm}")
    scan_execution = options.get('trainium_scan_execution','native')
    if scan_execution not in ('native','bufferized'):
        raise ValueError(f'unknown trainium_scan_execution: {scan_execution}')
    for module in prepared.graph.modules():
        if isinstance(module, torch.fx.GraphModule):
            for node in module.graph.nodes:
                if node.target is SCAN_OP:
                    node.meta[ALGORITHM_TAG] = scan_algorithm
                    node.meta['trainium_scan_execution'] = scan_execution
    _shapes(prepared)
    from .handoff import outline as outline_handoff
    outline_handoff(prepared.graph, options.get("trainium_resident_handoff", "off"))
    _shapes(prepared)
    from .dag import outline
    outline(prepared.graph, approximate_reciprocal=bool(
        options.get('trainium_approximate_reciprocal', False)))
    _shapes(prepared)
    if options.get('trainium_compensated_matmul', False):
        from voyager_compiler.codegen.node_info import get_anchor_node, is_gemm_op
        for module in prepared.graph.modules():
            if isinstance(module, torch.fx.GraphModule):
                for node in module.graph.nodes:
                    if is_gemm_op(get_anchor_node(node)):
                        node.meta['trainium_compensated_matmul'] = True


def materialized_implementations(components):
    """Keep separately priced unfused schedules for recognized matrix regions."""
    from voyager_compiler.codegen.node_info import (
        activation_prologue, is_gemm_op, SOFTMAX_OPS, tail_targets,
    )
    from voyager_compiler.compiler.bufferize.regions import (
        build_sequential_region, estimate_sequential_region,
    )
    from voyager_compiler.codegen.transform.operator_fusion import fuse_operator
    from voyager_compiler.extensions import ImplementationSpec
    from voyager_compiler import OpMatcher

    def rms_matches(node, anchor):
        return is_gemm_op(anchor) and any(op.target == torch.ops.aten.rms_norm.default
                   for op in activation_prologue(node))

    def build(node, context, parameters):
        region = copy.deepcopy(node.meta['submodule'])
        ShapeProp(region).propagate(*(fake_like(n.value) for n in node.all_input_nodes))
        # Keep add/RMS fused in either the prologue or epilogue, separated
        # from GEMM. The common region helper prices all component kernels.
        fuse_operator(region, [[OpMatcher('add'), OpMatcher('rms_norm')]])
        output = next(n for n in region.graph.nodes if n.op == 'output')
        if isinstance(output.args[0], Node):
            output.args = ((output.args[0],),)
            region.recompile()
        return build_sequential_region(region, context, components)

    return (
        ImplementationSpec(
            'trainium.rms_gemm.materialized', rms_matches, build,
            estimate=estimate_sequential_region,
            description='RMS prologue and GEMM using sequential shared bufferized builders',
        ),
        ImplementationSpec(
            'trainium.gemm_softmax.materialized',
            lambda node, anchor: is_gemm_op(anchor) and bool(set(tail_targets(node)) & SOFTMAX_OPS), build,
            estimate=estimate_sequential_region,
            description='GEMM and softmax using sequential shared bufferized builders',
        ),
        ImplementationSpec(
            'trainium.gemm_rms.materialized',
            lambda node, anchor: is_gemm_op(anchor) and
                torch.ops.aten.rms_norm.default in tail_targets(node), build,
            estimate=estimate_sequential_region,
            description='GEMM and RMS tail using sequential shared bufferized builders',
        ),
    )

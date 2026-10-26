"""Training with fake-quantized GEMMs, forward only or backward too.

``prepare_for_qat`` prepares QAT on an export, as ``prepare_qat_pt2e``
does: only the forward pass is quantized.  ``prepare_training`` captures the
forward and backward passes together, which quantizing the backward pass
and compiling a training step both need.  With an error spec its backward
GEMMs quantize their own operands; without one the forward alone is
quantized, and the backward GEMMs read either the forward's original
operands (``"high_precision"``) or its quantized ones (``"dequantized"``,
``_read_forward_quantized``).

``capture_training`` hands ``prepare_training`` AOTAutograd's joint graph of
the forward and backward passes before splitting it.  AOTAutograd traces it
with the ops broken into the ones the compiler lowers, and each GEMM is
renamed back to the op the model wrote -- a forward GEMM to its ``linear``
or ``matmul``, a backward one to ``matmul``.  Each GEMM operand is tagged
with its type: an error is computed from an incoming gradient, a weight
views a parameter, and anything else is an activation.  The quantizer
annotates the joint graph as it annotates an export, a tagged operand
taking its type's spec blocked along its GEMM's contraction axis, and
``prepare`` inserts the fake-quants.  Everything but the GEMM operands stays
in high precision, the bias gradient included.

A tensor several GEMMs read, as it is or through views, with the same spec
whose values do not depend on the axis -- per-tensor or a plain cast,
through any view, or blocks spanning both axes, through transposes -- is
quantized once: every such GEMM reads one fake-quant's output, which the
split saves when the backward reads it.

For the compiler, a training step is two graphs: ``gradient_program`` runs
the forward, the loss and the backward, and ``update_program`` the
optimizer's update.  Each writes the tensors it updates in place.
``TrainingStep`` runs the first on the model, eagerly or compiled by
``torch.compile``.
"""

import contextlib
import copy
import operator
import types
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from unittest.mock import patch

import torch
from torch import nn
from torch._decomp import core_aten_decompositions
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.backends.common import aot_autograd
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd.descriptors import (
    BufferAOTInput,
    GradAOTOutput,
    ParamAOTInput,
    PlainAOTInput,
    PlainAOTOutput,
    TangentAOTInput,
)
from torch._functorch.aot_autograd import (
    aot_module_simplified,
    make_boxed_func,
)
from torch._functorch.partitioners import default_partition
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.export._unlift import _check_inputs_match
from torch.fx import CodeGen, Graph, GraphModule, Node
from torch.fx.node import map_arg
from torch.utils import _pytree as pytree
from torchao.quantization.pt2e import FakeQuantizeBase
from torchao.quantization.pt2e import prepare as torchao_prepare
from torchao.quantization.pt2e.quantizer import QuantizationAnnotation

from voyager_compiler.export_utils import (
    create_getattr_from_value,
    export_model,
    merge_sibling_linears,
)
from voyager_compiler.ops import EAGER_NUMERICS, HADAMARD_SIZE
from voyager_compiler.quantization.fake_quantize import (
    SEED_BOUND,
    FusedAmaxObsFakeQuantize,
    MXFakeQuantize,
    _DerivedObserverOrFakeQuantize,
    calculate_global_scale,
)
from voyager_compiler.quantization.quantize_pt2e import (
    MX_OP_MAPPING,
    _get_obs_or_fq_map,
    _replace_observer_with_quantize_mx_node_decomposed,
    prepare_qat_pt2e,
    set_training,
)
from voyager_compiler.quantization.quantizer.quantizer import (
    QScheme,
    QuantizationSpec,
)
from voyager_compiler.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
)
from voyager_compiler.quantization.quantizer.xnnpack_quantizer_utils import (
    FUSED_SDPA,
    FUSED_SDPA_BACKWARD,
)

__all__ = [
    "DEFAULT_BACKWARD_OVERRIDE",
    "BACKWARD_OVERRIDES",
    "TrainingQuantizers",
    "TrainingStep",
    "capture_training",
    "disable_observers",
    "gemm_partners",
    "gradient_program",
    "prepare_for_qat",
    "prepare_training",
    "return_gradients",
    "update_program",
]

aten = torch.ops.aten

# The GEMMs AOTAutograd lowers linear and matmul to.
_GEMMS = (aten.mm.default, aten.addmm.default, aten.bmm.default)
# The ops the quantizer annotates as GEMMs.
_OPS = (aten.linear.default, aten.matmul.default)
_SDPA = aten.scaled_dot_product_attention.default
# What the backward GEMMs read without an error spec: the forward GEMMs'
# original operands, or their quantized ones.
BACKWARD_OVERRIDES = ("high_precision", "dequantized")
DEFAULT_BACKWARD_OVERRIDE = "dequantized"
# Ops that only view a tensor: an operand read through them is still the
# tensor they start from.
_VIEWS = {
    aten.view.default,
    aten._unsafe_view.default,
    aten.reshape.default,
    aten.t.default,
    aten.transpose.int,
    aten.permute.default,
    aten.expand.default,
    aten.alias.default,
    aten.unsqueeze.default,
    aten.squeeze.dim,
}
# Schemes whose scales belong to blocks along an axis.
_BLOCKS = (QScheme.MICROSCALING, QScheme.GROUP_WISE_AFFINE)


def _quantized_once(spec: QuantizationSpec) -> bool:
    """Whether ``spec`` gives a tensor the same values whatever the axis.

    A per-tensor scale, a plain cast, or blocks spanning both of a matrix's
    axes quantize a matrix and its transpose alike, so one quantization
    serves every GEMM reading the tensor -- unless it is rotated along an
    axis first.
    """
    # A bias's derived spec has no rotation.
    if getattr(spec, "rht_axis", None) is not None:
        return False
    if spec.qscheme in _BLOCKS:
        return isinstance(spec.ch_axis, tuple) and len(spec.ch_axis) > 1
    return spec.qscheme in (None, QScheme.PER_TENSOR_SYMMETRIC)


def _fake_quantize(
    fake_quant: nn.Module, x: torch.Tensor, **kwargs
) -> torch.Tensor:
    """Apply ``fake_quant`` to ``x``, passing it ``kwargs``.

    The graph node calling a fake-quant: AOTAutograd's split keeps a
    ``call_function``, where it drops a ``call_module``.  ``kwargs`` holds a
    global scale ``_share_global_scales`` shares, if any.
    """
    return fake_quant(x, **kwargs)


class TrainingQuantizers(nn.Module):
    """The fake-quants of a quantized training run.

    Held by the model, the fake-quants move with it and are saved in its
    state dict.  ``disable_observers`` leaves the ones named in
    ``gradients``, which quantize gradients, observing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gradients = set()


def _fold_permutes(joint: GraphModule) -> None:
    """Fold each permute of a permute into one permute of the source.

    Autograd transposes a transpose -- the backward reads a linear's weight
    through two permutes, and a weight gradient leaves through two -- so a
    pair that cancels becomes the tensor itself, and a GEMM operand reads
    the tensor it transposes.

    Args:
        joint: AOTAutograd's joint graph, rewritten in place.
    """
    graph = joint.graph
    for node in list(graph.nodes):
        if (
            node.target is not aten.permute.default
            or node.args[0].target is not aten.permute.default
        ):
            continue
        source, inner_dims = node.args[0].args
        ndim = len(inner_dims)
        dims = [inner_dims[d] % ndim for d in node.args[1]]
        if dims == list(range(ndim)):
            node.replace_all_uses_with(source)
            graph.erase_node(node)
        else:
            node.args = (source, dims)
    joint.recompile()


def _rename_gemms(joint: GraphModule, export: GraphModule) -> None:
    """Rename the joint graph's GEMMs back to the ops the model wrote.

    A forward GEMM becomes the ``linear`` or ``matmul`` of the export it was
    traced from; a backward one becomes ``matmul``.  A forward GEMM traced
    from any other op, such as an einsum, is left as it is.

    Args:
        joint: AOTAutograd's joint graph, rewritten in place.
        export: The graph AOTAutograd traced.
    """
    ops = {n.name: n.target for n in export.graph.nodes}
    graph = joint.graph
    for gemm in list(graph.nodes):
        if gemm.op != "call_function" or gemm.target not in _GEMMS:
            continue
        op = aten.matmul.default
        if gemm.meta.get("partitioner_tag") != "is_backward":
            op = ops.get(gemm.meta["from_node"][-1].name)
        if op not in _OPS:
            continue
        if gemm.target is aten.addmm.default:
            bias, a, b = gemm.args
        else:
            (a, b), bias = gemm.args, None
        if op is aten.linear.default and gemm.target is not aten.bmm.default:
            # A linear's lowering reads its weight transposed.
            args = (a, b.args[0]) if bias is None else (a, b.args[0], bias)
        else:
            op, args = aten.matmul.default, (a, b)
        with graph.inserting_before(gemm):
            node = graph.call_function(op, args)
        node.meta = gemm.meta
        gemm.replace_all_uses_with(node)
        graph.erase_node(gemm)
    joint.recompile()


def _unflatten_gemms(joint: GraphModule) -> None:
    """Give each GEMM the operands before AOTAutograd flattened them.

    AOTAutograd traces a ``linear`` or ``matmul`` of N-d tensors as a 2-d or
    3-d GEMM of views that flatten their leading dimensions, and views the
    result back.  Where the result is viewed back, the GEMM reads the
    tensors those views flatten -- transposed, if it read the view
    transposed, and past any ``expand`` that changes nothing -- and gives
    the N-d result, as in the model.  A GEMM contracting over the flattened
    dimensions, as a weight gradient does, keeps its views.

    Args:
        joint: AOTAutograd's joint graph, rewritten in place.
    """
    graph = joint.graph

    def unflattened(operand, gemm, lead):
        """The tensor whose ``lead`` dimensions ``operand`` flattens, or
        ``None``."""
        ndim = operand.meta["val"].ndim
        swap = [*range(ndim - 2), ndim - 1, ndim - 2]
        transposed = (
            operand.target is aten.permute.default
            and list(operand.args[1]) == swap
        )
        view = operand.args[0] if transposed else operand
        if view.target is not aten.view.default:
            return None
        tensor = view.args[0]
        while (
            tensor.target is aten.expand.default
            and tensor.meta["val"].shape == tensor.args[0].meta["val"].shape
        ):
            tensor = tensor.args[0]
        if tensor.meta["val"].shape != (*lead, *view.meta["val"].shape[1:]):
            return None
        if not transposed:
            return tensor
        n = len(tensor.meta["val"].shape)
        with graph.inserting_before(gemm):
            node = graph.call_function(
                aten.permute.default, (tensor, [*range(n - 2), n - 1, n - 2])
            )
        node.meta = {**operand.meta, "val": tensor.meta["val"].mT}
        return node

    for gemm in list(graph.nodes):
        if gemm.target not in _OPS or len(gemm.users) != 1:
            continue
        (view,) = gemm.users
        if view.target is not aten.view.default:
            continue
        _, *rest = gemm.meta["val"].shape
        lead = tuple(view.meta["val"].shape[: -len(rest)])
        if len(lead) < 2 or view.meta["val"].shape != (*lead, *rest):
            continue
        # A 3-d GEMM flattens both operands; a 2-d one, only the rows.
        count = 2 if len(rest) == 2 else 1
        operands = [unflattened(a, gemm, lead) for a in gemm.args[:count]]
        if None in operands:
            continue
        gemm.args = (*operands, *gemm.args[count:])
        gemm.meta["val"] = view.meta["val"]
        view.replace_all_uses_with(gemm)
        graph.erase_node(view)
    graph.eliminate_dead_code()
    joint.recompile()


def _tag_operands(joint: GraphModule) -> Set[Node]:
    """Tag each linear and matmul of the joint graph with its operand types
    and its kind.

    ``meta["operand_types"]`` holds a GEMM's two operand types: an error is
    computed from an incoming gradient (a tangent input), a weight views a
    parameter, and anything else is an activation.  ``meta["gemm_kind"]``
    is ``"fprop"`` for a forward GEMM, ``"wgrad"`` for a backward GEMM whose
    result reaches a parameter's gradient through views and sums only, and
    ``"dgrad"`` for any other backward GEMM.

    Args:
        joint: AOTAutograd's joint graph, tagged in place.

    Returns:
        The nodes computed from incoming gradients.
    """
    errors = set()
    for node in joint.graph.nodes:
        if isinstance(
            node.meta.get("desc"), TangentAOTInput
        ) or not errors.isdisjoint(node.all_input_nodes):
            errors.add(node)
    output = joint.graph.output_node()
    pending = [
        value
        for value, desc in zip(output.args[0], output.meta["desc"])
        if isinstance(desc, GradAOTOutput) and value is not None
    ]
    wgrads = set()
    while pending:
        node = pending.pop()
        if node.target in _OPS:
            wgrads.add(node)
        elif node.target in _VIEWS or node.target is aten.add.Tensor:
            pending.extend(node.all_input_nodes)
    for gemm in joint.graph.nodes:
        if gemm.target not in _OPS:
            continue
        if gemm.meta.get("partitioner_tag") != "is_backward":
            gemm.meta["gemm_kind"] = "fprop"
        elif gemm in wgrads:
            gemm.meta["gemm_kind"] = "wgrad"
        else:
            gemm.meta["gemm_kind"] = "dgrad"
        kinds = []
        for operand in gemm.args[:2]:
            source = operand
            while source.target in _VIEWS:
                source = source.args[0]
            if operand in errors:
                kinds.append("error")
            elif isinstance(source.meta.get("desc"), ParamAOTInput):
                kinds.append("weight")
            else:
                kinds.append("activation")
        gemm.meta["operand_types"] = tuple(kinds)
    return errors


def _is_transpose(node: Node) -> bool:
    """Whether ``node`` swaps the last two axes of its input."""
    if node.op != "call_function":
        return False
    if node.target is aten.t.default:
        return True
    if node.target is aten.permute.default:
        ndim = node.meta["val"].dim()
        dims = [d % ndim for d in node.args[1]]
        return dims == list(range(ndim - 2)) + [ndim - 1, ndim - 2]
    if node.target is not aten.transpose.int:
        return False
    ndim = node.meta["val"].dim()
    return {d % ndim for d in node.args[1:3]} == {ndim - 2, ndim - 1}


def _same_annotation(a: Node, b: Node) -> bool:
    """Whether the quantizer annotated two GEMMs with the same specs."""
    annotations = [n.meta.get("quantization_annotation") for n in (a, b)]
    if None in annotations:
        return annotations[0] is annotations[1]
    first, second = (
        (annotation.output_qspec, list(annotation.input_qspec_map.values()))
        for annotation in annotations
    )
    return first == second


def _merge_duplicate_views(joint: GraphModule) -> None:
    """Make the readers of one view of a tensor read one view node.

    The joint graph traces a view per GEMM that reads a tensor through it --
    q, k and v's weight gradients each reshape the layer's input -- and
    ``prepare`` gives a tensor's readers one fake-quant only when they read
    the same node.  Views on the same side of the forward/backward split
    merge.

    Args:
        joint: Joint graph, rewritten in place.
    """
    graph = joint.graph
    seen = {}
    for node in list(graph.nodes):
        if node.target not in _VIEWS:
            continue
        key = (
            node.target,
            str(node.args),
            str(node.kwargs),
            node.meta.get("partitioner_tag"),
        )
        if key in seen:
            node.replace_all_uses_with(seen[key])
            graph.erase_node(node)
        else:
            seen[key] = node


def _share_quantized_views(joint: GraphModule) -> None:
    """Quantize once a tensor that one GEMM reads and another views.

    ``prepare`` gives a tensor's readers with the same spec one fake-quant.
    A GEMM operand that views a tensor another GEMM reads, both with the
    same spec quantizing once, becomes such a reader: its views are rebuilt
    to take the tensor through that fake-quant.  A per-tensor scale or a
    plain cast quantizes each element on its own, so it is shared through
    any view; blocks spanning both axes only through transposes, which keep
    each block whole.

    Args:
        joint: Annotated joint graph, rewritten in place.
    """
    graph = joint.graph
    for gemm in list(graph.nodes):
        annotation = gemm.meta.get("quantization_annotation")
        if annotation is None:
            continue
        for operand, spec in list(annotation.input_qspec_map.items()):
            if spec is None or not _quantized_once(spec):
                continue
            chain, source, shared = [], operand, False
            while not shared and source.target in _VIEWS:
                if spec.qscheme in _BLOCKS and not _is_transpose(source):
                    break
                chain.append(source)
                source = source.args[0]
                shared = any(
                    user.meta["quantization_annotation"].input_qspec_map.get(
                        source
                    )
                    == spec
                    for user in source.users
                    if "quantization_annotation" in user.meta
                )
            if not shared:
                continue
            copies = {}
            with graph.inserting_before(gemm):
                for node in reversed(chain):
                    # A view's other inputs, such as a symbolic size, are
                    # read as they are.
                    copies[node] = graph.node_copy(
                        node, lambda arg: copies.get(arg, arg)
                    )
                    # The split places a node by its tag.
                    copies[node].meta["partitioner_tag"] = gemm.meta[
                        "partitioner_tag"
                    ]
            copies[chain[-1]].meta["quantization_annotation"] = (
                QuantizationAnnotation(
                    input_qspec_map={source: spec}, _annotated=True
                )
            )
            gemm.replace_input_with(operand, copies[operand])
            del annotation.input_qspec_map[operand]
    joint.recompile()


def _view_chain(node: Node) -> List[Node]:
    """``node``, then each tensor it views, down to the one it starts from."""
    chain = [node]
    while chain[-1].target in _VIEWS:
        chain.append(chain[-1].args[0])
    return chain


def _undo_view(graph: Graph, view: Node, x: Node, tag: str) -> Node:
    """The tensor ``view`` reads, computed back from ``x``, its output.

    A symbolic size is read off the tensor ``view`` reads.  Every node
    added carries ``tag``, the partitioner tag the split places it by.

    Raises:
        NotImplementedError: ``view`` expands, and cannot be undone.
    """
    source = view.args[0].meta["val"]
    added = []
    if view.target in (
        aten.view.default,
        aten._unsafe_view.default,
        aten.reshape.default,
    ):
        shape = []
        for dim, size in enumerate(source.shape):
            if not isinstance(size, int):
                added.append(
                    graph.call_function(aten.sym_size.int, (view.args[0], dim))
                )
                added[-1].meta["val"] = size
                size = added[-1]
            shape.append(size)
        target, args = aten.view.default, (x, shape)
    elif view.target in (aten.t.default, aten.alias.default):
        target, args = view.target, (x,)
    elif view.target is aten.transpose.int:
        target, args = view.target, (x, *view.args[1:])
    elif view.target is aten.permute.default:
        dims = [d % source.dim() for d in view.args[1]]
        target, args = view.target, (
            x,
            sorted(range(len(dims)), key=dims.__getitem__),
        )
    elif view.target is aten.unsqueeze.default:
        target, args = aten.squeeze.dim, (x, view.args[1])
    elif view.target is aten.squeeze.dim:
        dim = view.args[1] % source.dim()
        target, args = (
            (aten.unsqueeze.default, (x, dim))
            if source.shape[dim] == 1
            else (aten.alias.default, (x,))
        )
    else:
        raise NotImplementedError(f"{view.name} cannot be undone")
    added.append(graph.call_function(target, args))
    added[-1].meta["val"] = source
    for node in added:
        node.meta["partitioner_tag"] = tag
    return added[-1]


def _read_forward_quantized(joint: GraphModule, backward_override: str) -> None:
    """Make each backward GEMM without an error spec read its forward GEMM's
    operands as ``backward_override`` says.

    Such a GEMM -- one whose error operand the quantizer left unquantized --
    has its own operand specs dropped, and its incoming gradient stays in
    high precision.  Under ``"high_precision"`` its other operands stay the
    original tensors.  Under ``"dequantized"`` each reads the output of the
    forward GEMM's fake-quant (paired by ``seq_nr``) instead: back through
    the forward's views to where the two operands meet, then on through the
    backward's.

    Args:
        joint: Annotated, tagged joint graph, rewritten in place.
        backward_override: One of ``BACKWARD_OVERRIDES``.

    Raises:
        NotImplementedError: A backward operand views none of its forward
            GEMM's operands, or meets one only through a view that cannot be
            undone.
    """
    graph = joint.graph
    forward = {
        gemm.meta["seq_nr"]: gemm
        for gemm in graph.nodes
        if gemm.meta.get("gemm_kind") == "fprop"
        and "quantization_annotation" in gemm.meta
    }
    for gemm in list(graph.nodes):
        annotation = gemm.meta.get("quantization_annotation")
        if (
            gemm.target not in _OPS
            or annotation is None
            or gemm.meta["gemm_kind"] == "fprop"
        ):
            continue
        specs = annotation.input_qspec_map
        operands = list(zip(gemm.args[:2], gemm.meta["operand_types"]))
        if any(
            kind == "error" and specs.get(operand) is not None
            for operand, kind in operands
        ):
            continue
        fprop = forward.get(gemm.meta["seq_nr"])
        for operand, kind in operands:
            specs.pop(operand, None)
            if (
                backward_override == "high_precision"
                or kind == "error"
                or fprop is None
            ):
                continue
            below = _view_chain(operand)
            for tensor in fprop.args[:2]:
                above = _view_chain(tensor)
                common = next((n for n in below if n in above), None)
                if common is not None:
                    break
            else:
                raise NotImplementedError(
                    f"{gemm.name} reads {operand.name}, which views none of "
                    f"{fprop.name}'s operands"
                )
            spec = fprop.meta["quantization_annotation"].input_qspec_map.get(
                tensor
            )
            if spec is None:
                continue
            undo = above[: above.index(common)]
            redo = below[: below.index(common)]
            if not undo and not redo:
                specs[operand] = spec
                continue
            with graph.inserting_before(gemm):
                read = [tensor]
                for view in undo:
                    read.append(
                        _undo_view(
                            graph, view, read[-1], gemm.meta["partitioner_tag"]
                        )
                    )
                copies = {common: read[-1]}
                for view in reversed(redo):
                    copies[view] = graph.node_copy(
                        view, lambda arg: copies.get(arg, arg)
                    )
                    read.append(copies[view])
            for node in read[1:]:
                # The split places a node by its tag.
                node.meta["partitioner_tag"] = gemm.meta["partitioner_tag"]
            read[1].meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map={tensor: spec}, _annotated=True
            )
            gemm.replace_input_with(operand, read[-1])
    joint.recompile()


def capture_training(
    model: torch.nn.Module,
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]],
    dynamic_shapes: Optional[Dict[str, Any]],
    make_transform: Optional[
        Callable[[GraphModule], Callable[[GraphModule], GraphModule]]
    ] = None,
    hidden: Sequence[str] = (),
    merge_linears: bool = False,
) -> None:
    """Run ``model`` through the graphs AOTAutograd derives from its export.

    ``model``, in training mode, is exported once with the example inputs.
    AOTAutograd traces the export's forward and backward into one joint
    graph, with the ops broken into the ones the compiler lowers and the
    GEMMs named as the model wrote them.  ``make_transform(export)``, when
    given, rewrites the joint graph before AOTAutograd splits it into the
    forward graph and the backward graph that differentiates it.
    ``model.forward.graphs`` holds the ``"joint"``, ``"forward"`` and
    ``"backward"`` graphs, ``model.forward.in_spec`` and ``out_spec`` the
    pytree specs of the export's inputs and outputs, and ``model``'s forward
    is replaced to run the last two graphs.  Parameters stay the model's
    own, so optimizers, checkpoints and ``train()`` / ``eval()`` work as on
    the eager model: ``eval()`` switches the graphs' dropouts off, and under
    ``torch.no_grad()`` the outputs come back detached.  Moving or casting
    the model, and choosing which parameters require grad, must come before
    this call.  If capture fails, ``model`` is left as it was.  A deep copy
    of a captured model still runs the original's graphs, and pickling one
    fails.

    Args:
        model: Module to capture, in training mode.
        args: Positional example inputs.
        kwargs: Keyword example inputs.
        dynamic_shapes: Dynamic-shape spec forwarded to ``torch.export``.
        make_transform: Called with the export graph; returns the rewrite of
            the joint graph.  None leaves the joint graph as traced.
        hidden: Names of ``model``'s submodules to detach while exporting,
            so that state the forward never reads stays out of the graphs.
        merge_linears: Merge the bias-free linears that read one input into
            one (``merge_sibling_linears``).

    Raises:
        ValueError: ``model`` is in eval mode.
        NotImplementedError: The forward graph has a batch norm or a fused
            attention dropout, which cannot switch between training and
            eval in place.
    """
    if not model.training:
        raise ValueError("Capture the model in training mode")
    detached = {name: model._modules.pop(name) for name in hidden}
    try:
        gm = export_model(model, args, kwargs, dynamic_shapes=dynamic_shapes)
    finally:
        model._modules.update(detached)
    if merge_linears:
        merge_sibling_linears(gm, model)
    transform = None if make_transform is None else make_transform(gm)
    in_spec, out_spec = gm._in_spec, gm._out_spec
    inputs = [n.meta["val"] for n in gm.graph.nodes if n.op == "placeholder"]
    # AOTAutograd takes flat inputs and a tuple of outputs.
    gm.graph._codegen = CodeGen()
    output = next(n for n in gm.graph.nodes if n.op == "output")
    output.args = (tuple(output.args[0]),)
    gm.recompile()
    graphs = {}

    def partition(joint, joint_inputs, **kwargs):
        _rename_gemms(joint, gm)
        _unflatten_gemms(joint)
        _fold_permutes(joint)
        graphs["joint"] = joint if transform is None else transform(joint)
        return default_partition(graphs["joint"], joint_inputs, **kwargs)

    def compile_forward(graph, _):
        graphs["forward"] = graph
        return make_boxed_func(graph)

    def compile_backward(graph, _):
        # AOTAutograd hands over a deep copy; the backward calls the joint
        # graph's submodules, as the forward does.
        for name, _ in list(graph.named_children()):
            setattr(graph, name, getattr(graphs["joint"], name))
        # Drop the partitioner's dead copies of random forward ops, which
        # dead-code elimination otherwise keeps as impure.
        graph.graph.eliminate_dead_code(
            is_impure_node=lambda n: n.is_impure(impure_random=False)
        )
        graph.recompile()
        graphs["backward"] = graph
        return make_boxed_func(graph)

    # Backward ops break into the smaller ones the compiler lowers.
    decompositions = core_aten_decompositions()
    # Workaround for a PyTorch bug (2.13): the decomposition of CPU flash
    # attention returns the attention weights where its fused backward
    # expects the logsumexp, so the backward crashes in bf16 and computes
    # wrong gradients in fp32.  Keep the fused op undecomposed; drop this
    # once PyTorch fixes the decomposition.
    decompositions.pop(aten._scaled_dot_product_flash_attention_for_cpu.default)
    # The backward is compiled now, not at the first backward call.
    with (
        torch.enable_grad(),
        functorch_config.patch(force_non_lazy_backward_lowering=True),
    ):
        run = aot_module_simplified(
            gm,
            inputs,
            fw_compiler=compile_forward,
            bw_compiler=compile_backward,
            partition_fn=partition,
            decompositions=decompositions,
        )
    for node in graphs["forward"].graph.nodes:
        schema = getattr(node.target, "_schema", None)
        if schema is None:
            continue
        names = [a.name for a in schema.arguments]
        dropout_p = 0.0
        if "dropout_p" in names:
            index = names.index("dropout_p")
            dropout_p = (
                node.args[index]
                if index < len(node.args)
                else node.kwargs.get("dropout_p", 0.0)
            )
        if "batch_norm" in schema.name or dropout_p:
            raise NotImplementedError(
                f"{node.target} cannot switch between training and eval"
            )

    def visible(name):
        return name.split(".")[0] not in hidden

    requires_grad = [
        p.requires_grad for n, p in model.named_parameters() if visible(n)
    ]
    training = [True]

    def forward(*call_args, **call_kwargs):
        if [
            p.requires_grad for n, p in model.named_parameters() if visible(n)
        ] != requires_grad:
            raise RuntimeError(
                "Which parameters require grad changed after capture"
            )
        if model.training != training[0]:
            training[0] = model.training
            # Out of training a dropout passes its input through.
            for graph in graphs.values():
                for node in graph.graph.nodes:
                    if node.target == torch.ops.aten.native_dropout.default:
                        node.update_arg(2, model.training)
                graph.recompile()
        flat = _check_inputs_match(call_args, call_kwargs, in_spec)
        outputs = list(run(*(leaf for _, leaf in flat)))
        # AOTAutograd records the backward even under no_grad.
        if not torch.is_grad_enabled():
            outputs = [
                o.detach() if isinstance(o, torch.Tensor) else o
                for o in outputs
            ]
        return pytree.tree_unflatten(outputs, out_spec)

    forward.graphs = graphs
    forward.in_spec, forward.out_spec = in_spec, out_spec
    model.forward = forward


def _maybe_insert_input_observers_for_node(
    node: Node,
    qconfig,
    model: nn.Module,
    named_modules: Dict[str, nn.Module],
    obs_or_fq_map,
    is_qat: bool,
    model_device: Optional[torch.device] = None,
) -> None:
    """torchao's ``_maybe_insert_input_observers_for_node``, refusing only a
    keyword tensor the annotation gives a spec.

    torchao inserts fake-quants on positional arguments alone, and refuses
    any tensor passed by keyword so that none goes unquantized silently.  A
    tensor without a spec, such as the attention mask CPU flash attention
    takes by keyword, is left as it is.

    Raises:
        NotImplementedError: A keyword tensor has a spec.
    """
    node.args = tuple(
        torchao_prepare._maybe_insert_input_observer_for_arg_or_kwarg(
            node,
            arg,
            qconfig,
            model,
            named_modules,
            obs_or_fq_map,
            is_qat,
            model_device,
        )
        for arg in node.args
    )
    annotation = node.meta.get("quantization_annotation")
    specs = {} if annotation is None else annotation.input_qspec_map
    for name, value in node.kwargs.items():
        tensors = []
        map_arg(value, tensors.append)
        if any(specs.get(tensor) is not None for tensor in tensors):
            raise NotImplementedError(
                f"{node.name} passes its quantized {name} by keyword"
            )


def prepare_training(
    model: nn.Module,
    quantizer: XNNPACKQuantizer,
    example_args: Tuple[Any, ...],
    example_kwargs: Optional[Dict[str, Any]],
    dynamic_shapes,
    merge_linears: bool = False,
    backward_override: str = DEFAULT_BACKWARD_OVERRIDE,
) -> nn.Module:
    """Quantize ``model``'s GEMMs in the forward and the backward pass.

    A backward GEMM whose incoming gradient the quantizer leaves
    unquantized -- every one, without an error spec -- quantizes none of its
    operands, and reads its forward GEMM's as ``backward_override`` says.
    A quantized fused attention's backward reads the forward's quantized
    query, key and value either way: it recomputes the attention from them
    and the forward's saved statistics.  ``model`` keeps its parameters,
    names and ``train()`` / ``eval()``; its forward runs the graphs
    ``capture_training`` captures.  The fake-quants are held by
    ``model.training_quantizers``.

    Args:
        model: Float model, in training mode.
        quantizer: Annotates which GEMMs are quantized and how; its GEMM
            configs carry the error spec.
        example_args: Positional inputs to export with.
        example_kwargs: Keyword inputs to export with.
        dynamic_shapes: Dynamic dimensions of the inputs.
        merge_linears: Merge the bias-free linears that read one input with
            the same specs into one (``merge_sibling_linears``), whose operands
            then share their scales.
        backward_override: What a backward GEMM without an error spec reads:
            its forward GEMM's original operands (``"high_precision"``) or
            their quantized values (``"dequantized"``).

    Returns:
        ``model``.

    Raises:
        NotImplementedError: The quantizer would quantize an op other than
            a linear or matmul, or a GEMM output; or the model has a batch
            norm or a fused attention dropout.
        ValueError: A spec is per-channel, ``model`` is in eval mode, or
            ``backward_override`` is not one of ``BACKWARD_OVERRIDES``.
    """
    if backward_override not in BACKWARD_OVERRIDES:
        raise ValueError(
            f"backward_override is {backward_override!r}, not one of "
            f"{BACKWARD_OVERRIDES}"
        )
    model.training_quantizers = quantizers = TrainingQuantizers()

    def make_transform(export):
        quantizer.annotate(export)
        if merge_linears:
            merge_sibling_linears(export, model, _same_annotation)
        for node in export.graph.nodes:
            annotation = node.meta.pop("quantization_annotation", None)
            if (
                annotation is not None
                and node.target not in (*_OPS, _SDPA)
                and any(
                    spec is not None
                    for spec in annotation.input_qspec_map.values()
                )
            ):
                raise NotImplementedError(
                    f"Quantized training does not support {node.target} yet"
                )

        def transform(joint):
            _merge_duplicate_views(joint)
            errors = _tag_operands(joint)
            quantizer.annotate(joint)
            _read_forward_quantized(joint, backward_override)
            for node in joint.graph.nodes:
                annotation = node.meta.get("quantization_annotation")
                if annotation is None:
                    continue
                if annotation.output_qspec is not None:
                    raise NotImplementedError(
                        "GEMM outputs are not quantized in quantized training"
                    )
                if any(
                    getattr(spec, "qscheme", None)
                    == QScheme.PER_CHANNEL_SYMMETRIC
                    for spec in annotation.input_qspec_map.values()
                ):
                    raise ValueError(
                        "Per-channel specs have no contraction axis to "
                        "quantize along; use per-tensor or block specs"
                    )
            _share_quantized_views(joint)
            torchao_prepare._get_obs_or_fq_map = _get_obs_or_fq_map
            with (
                patch.object(
                    torchao_prepare,
                    "_maybe_insert_input_observers_for_node",
                    _maybe_insert_input_observers_for_node,
                ),
                unset_fake_temporarily(),
            ):
                prepared = torchao_prepare.prepare(joint, {}, True)
            graph = prepared.graph
            for node in list(graph.nodes):
                if node.op != "call_module":
                    continue
                (operand,) = node.args
                fake_quant = prepared.get_submodule(node.target)
                with unset_fake_temporarily():
                    fake_quant.to(operand.meta["val"].device)
                quantizers.add_module(node.target, fake_quant)
                if operand in errors:
                    quantizers.gradients.add(node.target)
                with graph.inserting_before(node):
                    call = graph.call_function(
                        _fake_quantize, (graph.get_attr(node.target), operand)
                    )
                call.meta["val"] = operand.meta["val"]
                node.replace_all_uses_with(call)
                graph.erase_node(node)
            prepared.recompile()
            return prepared

        return transform

    try:
        capture_training(
            model,
            example_args,
            example_kwargs,
            dynamic_shapes,
            make_transform,
            hidden=("training_quantizers",),
        )
    except Exception:
        del model.training_quantizers
        raise
    return model


def _parameter_name(target: str) -> str:
    """The graph name of the parameter at module path ``target``."""
    return target.replace(".", "_")


def _delayed_scale(
    graph: Graph,
    operand: Node,
    history: Node,
    scale: Node,
    fake_quant: FusedAmaxObsFakeQuantize,
) -> Node:
    """Build the scale update a delayed-scaling fake-quant makes per call.

    The new scale is the largest amax in ``history`` over ``quant_max``,
    or the old ``scale`` when that amax is zero or not finite; ``operand``'s
    own amax then joins the history in place of its oldest.  Both state
    tensors are written in place.

    Args:
        graph: Graph to build in, at its insertion point.
        operand: The tensor the fake-quant quantizes.
        history: ``fake_quant``'s amax history.
        scale: ``fake_quant``'s scale.
        fake_quant: The per-tensor fake-quant being lowered.

    Returns:
        The new scale.
    """
    call = graph.call_function
    magnitude = call(aten.abs.default, (operand,))
    newest = call(aten.amax.default, (magnitude, []))
    amax = call(aten.amax.default, (history, [0]))
    # The update rolls the history back by one and overwrites its first.
    length = fake_quant.amax_history_len
    entries = [call(aten.view.default, (newest, [1]))]
    if length > 1:
        entries.append(call(aten.slice.Tensor, (history, 0, 2, length)))
        entries.append(call(aten.slice.Tensor, (history, 0, 0, 1)))
    updated = call(aten.cat.default, (entries,))
    call(aten.copy_.default, (history, updated))

    new = call(aten.div.Tensor, (amax, fake_quant.quant_max))
    positive = call(aten.gt.Scalar, (amax, 0.0))
    new = call(aten.where.self, (positive, new, scale))
    # An amax is never negative, so below infinity is finite.
    finite = call(aten.lt.Scalar, (amax, float("inf")))
    new = call(aten.where.self, (finite, new, scale))
    if fake_quant.power_2_scale:
        exponent = call(aten.log2.default, (new,))
        exponent = call(aten.ceil.default, (exponent,))
        new = call(aten.pow.Scalar, (2.0, exponent))
    call(aten.copy_.default, (scale, new))
    return new


def _dequantize_gemms(
    graph: Graph, scales: Dict[Node, Node], biases: Set[Node]
) -> None:
    """Scale each GEMM of per-tensor quantized operands back after it.

    The GEMM reads the quantized values themselves, so its result is scaled
    by the product of its operands' scales.  A bias quantized by that
    product stays in the GEMM, added before the result is scaled back; any
    other bias is added after.

    Args:
        graph: Graph rewritten in place.
        scales: Each per-tensor ``quantize`` -> the scale it divides by.
        biases: The ``quantize`` of each bias, by its GEMM's scale product.

    Raises:
        NotImplementedError: A quantized tensor is read by an op other than
            a GEMM.
    """
    for quantized in scales:
        stack = list(quantized.users)
        while stack:
            user = stack.pop()
            if user.target in _VIEWS:
                stack.extend(user.users)
            elif user.target not in _OPS:
                raise NotImplementedError(
                    f"{user.target} reads the quantized {quantized.name}"
                )
    for gemm in list(graph.nodes):
        if gemm.target not in _OPS:
            continue
        found = []
        for operand in gemm.args[:2]:
            while operand.target in _VIEWS:
                operand = operand.args[0]
            if operand in scales:
                found.append(scales[operand])
        if not found:
            continue
        bias = gemm.args[2] if len(gemm.args) > 2 else None
        if bias not in biases:
            gemm.args = gemm.args[:2]
        users = list(gemm.users)
        with graph.inserting_before(gemm.next):
            product = found[0]
            for other in found[1:]:
                product = graph.call_function(aten.mul.Tensor, (product, other))
            result = graph.call_function(
                torch.ops.quantized_ops.dequantize.default, (gemm, product)
            )
            if bias is not None and bias not in biases:
                result = graph.call_function(aten.add.Tensor, (result, bias))
        for user in users:
            user.replace_input_with(gemm, result)


def _insert_rotation(
    graph: Graph, operand: Node, axis: int, matrix: Node
) -> Node:
    """Insert the ops ``hadamard_rotate`` runs, at the graph's insertion
    point.

    A length known at capture is padded only if it needs to be; a symbolic
    one is padded by a size computed in the graph, zero when it is whole.
    The matrix is read through a float32 cast of its own, so a pass that
    narrows the computation to 16 bits narrows it too.  Each node gets its
    fake value, as capture gives every node, for the passes that read it.

    Args:
        graph: The program's graph.
        operand: The tensor to rotate.
        axis: Its last or second-to-last axis.
        matrix: The ``get_attr`` of ``hadamard_matrix``, with its fake value.

    Returns:
        The rotated tensor, in ``operand``'s dtype, its axis padded.
    """

    def call(target, *args, **kwargs):
        node = graph.call_function(target, args, kwargs)
        # Symbolic sizes take the dispatcher tracing uses.
        with value.fake_mode, enable_python_dispatcher():
            node.meta["val"] = target(
                *map_arg(args, lambda arg: arg.meta["val"]), **kwargs
            )
        return node

    value = operand.meta["val"]
    axis %= value.ndim
    padding = -value.shape[axis] % HADAMARD_SIZE
    if not isinstance(padding, int):
        length = call(aten.sym_size.int, operand, axis)
        padding = call(operator.mod, call(operator.neg, length), HADAMARD_SIZE)
    if padding:
        trailing = [0, 0] * (value.ndim - 1 - axis)
        operand = call(
            aten.constant_pad_nd.default, operand, [*trailing, 0, padding]
        )
    x = call(aten._to_copy.default, operand, dtype=torch.float32)
    runs = call(aten.unflatten.int, x, axis, [-1, HADAMARD_SIZE])
    matrix = call(aten._to_copy.default, matrix, dtype=torch.float32)
    if axis == value.ndim - 1:
        rotated = call(aten.matmul.default, runs, matrix)
    else:
        rotated = call(aten.matmul.default, call(aten.t.default, matrix), runs)
    flat = call(aten.flatten.using_ints, rotated, axis, axis + 1)
    return call(aten._to_copy.default, flat, dtype=value.dtype)


def _hadamard_of(node: Node) -> Optional[Node]:
    """The ``get_attr`` of the Hadamard matrix ``node`` reads through the
    casts and the transpose ``_insert_rotation`` puts in front of it, or
    None."""
    while node.target in (aten._to_copy.default, aten.t.default):
        node = node.args[0]
    if node.op == "get_attr" and node.target.startswith("rht_matrix"):
        return node
    return None


def rotation_input(node: Node) -> Optional[Tuple[Node, int, Node]]:
    """What ``node`` rotates, if it ends a rotation ``_insert_rotation``
    inserted and that rotation pads nothing.

    Returns:
        ``(tensor, axis, matrix)``: the tensor rotated, the axis and the
        Hadamard matrix's ``get_attr``; or None.
    """
    if node.target is not aten._to_copy.default:
        return None
    flat = node.args[0]
    if flat.target is not aten.flatten.using_ints:
        return None
    product = flat.args[0]
    if product.target is not aten.matmul.default:
        return None
    matrices = [_hadamard_of(arg) for arg in product.args]
    if not any(matrices):
        return None
    side = 0 if matrices[0] else 1
    runs = product.args[1 - side]
    if runs.target is not aten.unflatten.int:
        return None
    upcast = runs.args[0]
    if upcast.target is not aten._to_copy.default:
        return None
    source = upcast.args[0]
    if source.target is aten.constant_pad_nd.default:
        return None
    return source, runs.args[1], matrices[side]


_CONVERT_ELEMENT_TYPE = torch.ops.prims.convert_element_type.default
_QUANTIZE_MX = torch.ops.quantized_ops.quantize_mx.default
_QUANTIZE_MX_TRANSPOSE = torch.ops.quantized_ops.quantize_mx_transpose.default
# The views that keep a tensor's elements in order.
_RESHAPES = (
    aten.view.default,
    aten._unsafe_view.default,
    aten.reshape.default,
    aten.alias.default,
)
# quantize_mx's arguments a pair shares: index and name.
_PAIR_SHARED = (
    (1, "qmap"),
    (3, "block_size"),
    (4, "quant_max"),
    (5, "force_scale_power_of_two"),
    (6, "scale_qmap"),
    (7, "output_code"),
)


def _argument(node: Node, index: int, name: str):
    """``node``'s argument at ``index``, given positionally or by ``name``."""
    if index < len(node.args):
        return node.args[index]
    return node.kwargs.get(name)


def _root(node: Node) -> Node:
    """The tensor ``node`` is, through the views that keep its elements in
    order."""
    while node.target in _RESHAPES:
        node = node.args[0]
    return node


def _layout(quantize: Node):
    """How a single-axis ``quantize_mx`` reads its tensor, for pairing.

    Returns:
        ``("rows", tensor, None)`` for blocks along ``tensor``'s last axis;
        ``("transpose", tensor, matrix)`` for blocks along its transpose's
        last axis -- read through a permute of the last two axes, or in
        place along the second-to-last, which rounds nearest -- rotated by
        ``matrix``'s Hadamard rotation if not None; None otherwise.
    """
    axes = quantize.args[2]
    if len(axes) != 1:
        return None
    value = quantize.args[0]
    ndim = value.meta["val"].ndim
    axis = axes[0] % ndim
    matrix = None
    rotation = rotation_input(value)
    if rotation is not None:
        value, rotated_axis, matrix = rotation
        if rotated_axis % ndim != axis:
            return None
    if axis == ndim - 1:
        swap = [*range(ndim - 2), ndim - 1, ndim - 2]
        permuted = value.target is aten.permute.default
        if permuted and list(value.args[1]) == swap:
            return "transpose", value.args[0], matrix
        return ("rows", value, None) if matrix is None else None
    if axis == ndim - 2 and _argument(quantize, 8, "seed") is None:
        return "transpose", value, matrix
    return None


def _same_settings(program: GraphModule, a: Node, b: Node) -> bool:
    """Whether two quantize_mx nodes quantize alike but for their axes,
    seeds and global scales."""
    for index, name in _PAIR_SHARED:
        x, y = _argument(a, index, name), _argument(b, index, name)
        if isinstance(x, Node) and isinstance(y, Node):
            if x.op != "get_attr" or y.op != "get_attr":
                if x is not y:
                    return False
                continue
            # Lookup tables hold NaNs, which equal nothing.
            tx, ty = program.get_buffer(x.target), program.get_buffer(y.target)
            if (
                tx.shape != ty.shape
                or not ((tx == ty) | (tx.isnan() & ty.isnan())).all()
            ):
                return False
        elif x != y:
            return False
    return True


def _hoist(graph: Graph, values: Sequence[Node], before: Node) -> bool:
    """Move what ``values`` are computed from, where it comes after
    ``before``, to just before it, in order.

    Returns:
        False, and moves nothing, if that would move a seed's draw past
        another's: torch's generator hands them out in graph order.
    """
    order = {node: i for i, node in enumerate(graph.nodes)}
    start = order[before]
    moving, pending = set(), [v for v in values if isinstance(v, Node)]
    while pending:
        node = pending.pop()
        if node in moving or order[node] < start:
            continue
        moving.add(node)
        pending.extend(node.all_input_nodes)
    draws = [n for n in moving if n.target is aten.randint.default]
    if draws:
        last = max(order[n] for n in draws)
        for node in graph.nodes:
            if (
                start <= order[node] < last
                and node.target is aten.randint.default
                and node not in moving
            ):
                return False
    for node in sorted(moving, key=order.__getitem__):
        before.prepend(node)
    return True


def pair_quantize_transposes(program: GraphModule) -> None:
    """Merge each pair of ``quantize_mx`` reading one tensor -- along its
    last axis and along its transpose's, that one maybe rotated -- into one
    ``quantize_mx_transpose``, as a backend that casts both layouts in one
    read wants them.  The program computes what it did, bit for bit, once
    the caller recompiles it.

    Args:
        program: A ``gradient_program``, rewritten in place.
    """
    graph = program.graph
    rows, transposes = {}, {}
    for node in graph.nodes:
        if node.target is not _QUANTIZE_MX:
            continue
        layout = _layout(node)
        if layout is None:
            continue
        kind, tensor, matrix = layout
        found = rows if kind == "rows" else transposes
        found.setdefault(_root(tensor), []).append((node, tensor, matrix))
    for root, row_quantizes in rows.items():
        for transpose, tensor, matrix in transposes.get(root, []):
            match = next(
                (
                    (quantize, value)
                    for quantize, value, _ in row_quantizes
                    if _same_settings(program, quantize, transpose)
                    and value.meta["val"].numel() == tensor.meta["val"].numel()
                    and value.meta["val"].shape[-1]
                    == tensor.meta["val"].shape[-1]
                ),
                None,
            )
            if match is None:
                continue
            quantize, value = match
            if _merge_pair(program, quantize, value, transpose, tensor, matrix):
                row_quantizes.remove((quantize, value, None))
    graph.eliminate_dead_code()


def _merge_pair(
    program: GraphModule,
    rows: Node,
    value: Node,
    transpose: Node,
    tensor: Node,
    matrix: Optional[Node],
) -> bool:
    """Replace ``rows`` (``quantize_mx`` of ``value`` along its last axis)
    and ``transpose`` (of ``tensor``'s transpose) by one
    ``quantize_mx_transpose`` of ``tensor``.

    Returns:
        Whether they were merged; not when that would reorder seed draws.
    """
    graph = program.graph
    order = {node: i for i, node in enumerate(graph.nodes)}
    first = min(rows, transpose, key=order.__getitem__)
    arguments = (
        tensor,
        *(_argument(rows, index, name) for index, name in _PAIR_SHARED),
        _argument(rows, 8, "seed"),
        _argument(rows, 9, "global_scale"),
        _argument(transpose, 8, "seed"),
        _argument(transpose, 9, "global_scale"),
        matrix,
    )
    if not _hoist(graph, arguments, first):
        return False
    with graph.inserting_before(first):
        pair = graph.call_function(_QUANTIZE_MX_TRANSPOSE, arguments)
        pair.meta["dtype"] = rows.meta.get("dtype")
        outputs = [
            graph.call_function(operator.getitem, (pair, i)) for i in range(4)
        ]
    # The row-wise outputs in the shapes ``rows`` gave them: the codes in
    # ``value``'s, the scales one per block of its last axis.
    shape = list(value.meta["val"].shape)
    block_size = _argument(rows, 3, "block_size")
    with graph.inserting_after(outputs[3]):
        row_outputs = [
            graph.call_function(
                aten.view.default,
                (outputs[0], [*shape[:-1], shape[-1] // block_size]),
            ),
            graph.call_function(aten.view.default, (outputs[1], shape)),
        ]
    transpose_outputs = outputs[2:]
    ndim = transpose.args[0].meta["val"].ndim
    if transpose.args[2][0] % ndim == ndim - 2:
        # Blocked in place along the second-to-last axis: the transpose's
        # outputs swapped back.
        swap = [*range(ndim - 2), ndim - 1, ndim - 2]
        with graph.inserting_after(row_outputs[-1]):
            transpose_outputs = [
                graph.call_function(aten.permute.default, (output, swap))
                for output in transpose_outputs
            ]
    for old, new in ((transpose, transpose_outputs), (rows, row_outputs)):
        for user in list(old.users):
            user.replace_all_uses_with(new[user.args[1]])
            graph.erase_node(user)
        graph.erase_node(old)
    return True


def _reads_mx(program: GraphModule, gemm: Node) -> bool:
    """Whether ``gemm`` takes both its operands as MX values: as an ``_mx``
    twin already, or reading two enabled microscaling fake-quants."""
    if gemm.target in MX_OP_MAPPING.values():
        return True
    if gemm.target not in _OPS:
        return False
    for operand in gemm.args[:2]:
        if operand.target is not _fake_quantize:
            return False
        fake_quant = program.get_submodule(operand.args[0].target)
        if not (
            isinstance(fake_quant, MXFakeQuantize)
            and fake_quant.fake_quant_enabled
        ):
            return False
    return True


def _lower_fake_quants(program: GraphModule, names: Dict[str, str]) -> None:
    """Replace each fake-quant of ``program`` with the ops that quantize.

    A microscaling fake-quant becomes a ``quantize_mx`` and the GEMMs
    reading it their ``_mx`` twins, as ``convert_pt2e`` lowers an
    activation's.  A weight is a program input that changes every step, so
    it is quantized in the graph like an activation rather than baked.
    Stochastic rounding draws a fresh seed for each ``quantize_mx`` from
    torch's generator, as the fake-quant does.  A random Hadamard transform
    rotates the operand before its ``quantize_mx`` (``_insert_rotation``);
    where the fake-quant rotates it back, the GEMM multiplies the rotations
    of its two operands, which cancel, over the padded contraction axis.
    A reader that cannot take MX values -- a view, or a GEMM whose other
    operand is not MX, as a QAT backward GEMM's -- reads them dequantized.

    A per-tensor fake-quant scales by its delayed-scaling state: its amax
    history and scale become program inputs, named after the fake-quant's
    buffers in the model, that the program updates as the fake-quant does
    (``_delayed_scale``).  Its GEMMs read the quantized values and are
    scaled back after (``_dequantize_gemms``).  A bias is quantized by the
    product of its GEMM's operand scales as they stand where its fake-quant
    runs, and the GEMM adds it before scaling back.

    Args:
        program: A gradient program, rewritten in place.
        names: Each fake-quant's attribute in ``program`` -> its module path
            in the model.

    Raises:
        NotImplementedError: A fake-quant's scheme has no lowering yet, a
            GEMM rotates one of its operands only, a per-tensor quantized
            tensor is read by an op other than a GEMM, or a fused attention
            reads a quantized tensor.
    """
    graph = program.graph
    calls = [n for n in graph.nodes if n.target is _fake_quantize]
    for node in calls:
        readers = list(node.users)
        while readers:
            reader = readers.pop()
            if reader.target in _VIEWS:
                readers.extend(reader.users)
            elif reader.target in (*FUSED_SDPA, *FUSED_SDPA_BACKWARD):
                raise NotImplementedError(
                    f"{reader.name}: a fused attention has no lowering in a "
                    "training step; quantize eager attention instead"
                )
    attributes = {id(m): name for name, m in program.named_children()}

    # The scale each per-tensor fake-quant holds where the graph now stands:
    # its state until it runs, the scale it computes after.
    current = {}
    states = {}
    # ``prepare`` puts a parameter's fake-quant right after it; moving the
    # inputs ahead of every op lets the state inputs follow them.
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    first = next(n for n in graph.nodes if n.op != "placeholder")
    for placeholder in placeholders:
        first.prepend(placeholder)
    last = placeholders[-1]
    for node in calls:
        target = node.args[0].target
        fake_quant = program.get_submodule(target)
        if not isinstance(fake_quant, FusedAmaxObsFakeQuantize):
            continue
        state = []
        for buffer in ("amax_history", "scale"):
            with graph.inserting_after(last):
                last = graph.placeholder(
                    _parameter_name(f"{names[target]}.{buffer}")
                )
            last.meta["val"] = torch.empty_like(getattr(fake_quant, buffer))
            state.append(last)
        states[target] = state
        current[target] = state[1]

    rotated = set()
    for node in calls:
        fake_quant = program.get_submodule(node.args[0].target)
        if (
            isinstance(fake_quant, MXFakeQuantize)
            and fake_quant.fake_quant_enabled
            and fake_quant.rht_axis is not None
        ):
            rotated.add(node)
    for node in rotated:
        for gemm in node.users:
            if gemm.target not in _OPS or not rotated.issuperset(gemm.args[:2]):
                raise NotImplementedError(
                    f"{gemm.name} reads a rotated operand without rotating "
                    "the other"
                )

    scales, biases = {}, set()
    for node in calls:
        fake_quant_attr, operand = node.args
        target = fake_quant_attr.target
        fake_quant = program.get_submodule(target)
        if not fake_quant.fake_quant_enabled:
            node.replace_all_uses_with(operand)
            graph.erase_node(node)
            continue
        if isinstance(fake_quant, MXFakeQuantize):
            with graph.inserting_before(node):
                # Each call draws a fresh seed, as the fake-quant does.
                seed = (
                    graph.call_function(
                        aten.randint.default,
                        (SEED_BOUND, []),
                        {"device": operand.meta["val"].device},
                    )
                    if fake_quant.stochastic_rounding
                    else None
                )
                if fake_quant.rht_axis is not None:
                    matrix = create_getattr_from_value(
                        program, graph, "rht_matrix", fake_quant.rht_matrix
                    )
                    fake_mode = operand.meta["val"].fake_mode
                    matrix.meta["val"] = fake_mode.from_tensor(
                        fake_quant.rht_matrix
                    )
                    operand = _insert_rotation(
                        graph, operand, fake_quant.rht_axis, matrix
                    )
                # The lowering replaces the call_module form ``prepare``
                # leaves.
                call = graph.call_module(target, (operand,))
            call.meta = node.meta
            # A reader that cannot take MX values -- a view, or a GEMM whose
            # other operand is not MX, as a QAT backward GEMM's gradient --
            # reads them dequantized.
            plain = [u for u in node.users if not _reads_mx(program, u)]
            node.replace_all_uses_with(call, lambda user: user not in plain)
            modules = dict(program.named_modules(remove_duplicate=False))
            quantized, scale, code, global_scale = (
                _replace_observer_with_quantize_mx_node_decomposed(
                    program, call, modules, seed
                )
            )
            if plain:
                if fake_quant.outlier_threshold is not None:
                    raise NotImplementedError(
                        f"{node.name}: dequantizing an outlier-filtered "
                        "tensor has no lowering yet"
                    )
                with graph.inserting_before(node):
                    value = graph.call_function(
                        torch.ops.quantized_ops.dequantize_mx.default,
                        (quantized, scale, fake_quant.block_size),
                        {"code": code, "global_scale": global_scale},
                    )
                value.meta["val"] = node.meta["val"]
                node.replace_all_uses_with(value)
            graph.erase_node(node)
            continue
        if isinstance(fake_quant, FusedAmaxObsFakeQuantize):
            if fake_quant.is_per_channel:
                raise NotImplementedError(
                    "Per-channel scales have no lowering for training"
                )
            with graph.inserting_before(node):
                if fake_quant.observer_enabled:
                    current[target] = _delayed_scale(
                        graph, operand, *states[target], fake_quant
                    )
        elif isinstance(fake_quant, _DerivedObserverOrFakeQuantize):
            act, weight = (
                current[attributes[id(m)]] for m in fake_quant.obs_or_fqs
            )
            with graph.inserting_before(node):
                weight = graph.call_function(aten.flatten.using_ints, (weight,))
                current[target] = graph.call_function(
                    aten.mul.Tensor, (act, weight)
                )
        else:
            raise NotImplementedError(
                f"{type(fake_quant).__name__} has no lowering for training"
            )
        with graph.inserting_before(node):
            # The fake-quant scales in its operand's dtype.
            scale = graph.call_function(
                aten._to_copy.default,
                (current[target],),
                {"dtype": operand.meta["val"].dtype},
            )
            qmap = (
                None
                if fake_quant.qmap is None
                else create_getattr_from_value(
                    program, graph, "qmap", fake_quant.qmap
                )
            )
            quantized = graph.call_function(
                torch.ops.quantized_ops.quantize.default,
                (operand, scale, None, None, None, qmap),
            )
            quantized.meta["dtype"] = fake_quant.dtype
            if isinstance(fake_quant, _DerivedObserverOrFakeQuantize):
                biases.add(quantized)
            else:
                scales[quantized] = scale
        node.replace_all_uses_with(quantized)
        graph.erase_node(node)
    _dequantize_gemms(graph, scales, biases)
    graph.eliminate_dead_code()
    program.delete_all_unused_submodules()


def gradient_program(
    model: nn.Module, lower_fake_quants: bool = True
) -> GraphModule:
    """The forward, the loss and the backward of a training step, as a graph.

    Built from the joint graph ``capture_training`` captured from ``model``,
    whose forward returns the loss, alone or as the ``loss`` of its output,
    such as a Hugging Face ``ModelOutput``.  The program returns the loss
    and starts the backward from a loss gradient of 1.  The other outputs,
    such as the logits, are dropped, and their gradient is zero: where it is
    added to another gradient, as the logits' is to the one the loss sends
    them, the sum is just that other gradient.  Each parameter's gradient
    is written in place into a buffer of its own.  A parameter is named
    after its module path (``fc1.weight`` becomes ``fc1_weight``), a tied
    one after its first, and its gradient buffer after it, with ``_grad``.
    The fake-quants of a model ``prepare_training`` prepared become the ops
    that quantize, or with ``lower_fake_quants`` off stay calls to the
    model's own fake-quants, so a change to them, such as a refit lookup
    table, reaches the program.  Which of them share a global scale
    (``_share_global_scales``) is decided once, when the program is built,
    from their element and scale maxima, global-scale flags and transforms.

    Args:
        model: Model captured by ``capture_training``.
        lower_fake_quants: Lower the fake-quants to the ops that quantize.

    Returns:
        A graph taking the parameters, the model's inputs and one gradient
        buffer per trainable parameter, in that order, and returning the loss.

    Raises:
        NotImplementedError: A dropped output's gradient is used other than
            added, unscaled, to a gradient of the sum's shape and dtype.
    """
    joint = model.forward.graphs["joint"]
    paths = {id(m): name for name, m in model.named_modules()}
    names = {}
    for node in joint.graph.find_nodes(
        op="call_function", target=_fake_quantize
    ):
        fake_quant_attr, operand = node.args
        module = joint.get_submodule(fake_quant_attr.target)
        names[fake_quant_attr.target] = paths[id(module)]
        if (
            isinstance(module, FusedAmaxObsFakeQuantize)
            and module.amax_history.numel() == 0
        ):
            # As the fake-quant's first call starts it.
            module.start_history((), operand.meta["val"].dtype)
    # Unlowered, the program calls the model's own fake-quants; lowering
    # rewrites copies of them.
    shared = {} if lower_fake_quants else {id(m): m for m in joint.children()}
    program = copy.deepcopy(joint, shared)
    graph = program.graph
    # The joint graph takes its inputs as two lists, primals and tangents.
    graph._codegen = CodeGen()
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    # A tensor under several names, such as tied embeddings, takes the first,
    # as ``named_parameters`` and so ``update_program`` name it.
    tensors = [
        *model.named_parameters(remove_duplicate=False),
        *model.named_buffers(remove_duplicate=False),
    ]
    first = {}
    for name, tensor in tensors:
        first.setdefault(tensor, name)
    canonical = {name: first[tensor] for name, tensor in tensors}
    named = {}
    for node in placeholders:
        desc = node.meta["desc"]
        if not isinstance(desc, (ParamAOTInput, BufferAOTInput)):
            continue
        name = _parameter_name(canonical[desc.target])
        if name in named:
            # The joint graph takes a tied tensor once per name.
            node.replace_all_uses_with(named[name])
            graph.erase_node(node)
        else:
            named[name] = node
            node._rename(name)
            node.target = node.name

    # The loss's position among the forward's flattened outputs: rebuilding
    # the output with each leaf replaced by its position, 0, 1, ..., puts
    # the loss's position where the loss was.
    out_spec = model.forward.out_spec
    structure = pytree.tree_unflatten(range(out_spec.num_leaves), out_spec)
    index = structure if isinstance(structure, int) else structure.loss
    output = graph.output_node()
    (loss,) = [
        value
        for value, desc in zip(output.args[0], output.meta["desc"])
        if isinstance(desc, PlainAOTOutput) and desc.idx == index
    ]
    for tangent in placeholders:
        desc = tangent.meta["desc"]
        if not isinstance(desc, TangentAOTInput):
            continue
        if desc.output.idx == index:
            with graph.inserting_after(placeholders[-1]):
                one = graph.call_function(
                    aten.full.default,
                    ([], 1.0),
                    {
                        "dtype": tangent.meta["val"].dtype,
                        "device": tangent.meta["val"].device,
                    },
                )
            one.meta["val"] = tangent.meta["val"]
            tangent.replace_all_uses_with(one)
        else:
            # A dropped output's gradient is zero, so a sum it joins is the
            # other term -- when the sum neither scales that term (alpha)
            # nor broadcasts or casts it.
            for add in list(tangent.users):
                others = [a for a in add.args if a is not tangent]
                if (
                    add.target is aten.add.Tensor
                    and not add.kwargs
                    and len(others) == 1
                    and isinstance(others[0], Node)
                ):
                    value, total = others[0].meta["val"], add.meta["val"]
                    if (value.shape, value.dtype) == (total.shape, total.dtype):
                        add.replace_all_uses_with(others[0])
                        graph.erase_node(add)
            if tangent.users:
                raise NotImplementedError(
                    "A dropped output's gradient may only be added, "
                    "unscaled, to a gradient of the sum's shape and dtype"
                )
        graph.erase_node(tangent)

    grads = {}
    for value, desc in zip(output.args[0], output.meta["desc"]):
        if value is not None and isinstance(desc, GradAOTOutput):
            name = _parameter_name(canonical[desc.grad_of.target])
            grads.setdefault(name, []).append(value)
    for name, values in grads.items():
        with graph.inserting_before(one):
            buffer = graph.placeholder(f"{name}_grad")
        # A new placeholder's name is snake-cased; keep the parameter's.
        buffer._rename(f"{name}_grad")
        buffer.target = buffer.name
        buffer.meta["val"] = values[0].meta["val"]
        # A tied tensor's gradient sums the ones through each of its names.
        total = values[0]
        with graph.inserting_before(output):
            for value in values[1:]:
                total = graph.call_function(aten.add.Tensor, (total, value))
                total.meta["val"] = values[0].meta["val"]
            graph.call_function(aten.copy_.default, (buffer, total))
    output.args = ((loss,),)
    graph.eliminate_dead_code()
    if lower_fake_quants:
        _lower_fake_quants(program, names)
    else:
        _share_global_scales(program)
    program.recompile()
    return program


def _share_global_scales(program: GraphModule) -> None:
    """Compute each tensor's global scale once for all its fake-quants.

    A tensor is fake-quantized once per orientation its GEMMs read it in,
    as it is or through views.  Microscaling fake-quants of one tensor that
    divide by a global scale (``calculate_global_scale``) with the same
    element and scale maxima, and that quantize the tensor itself, neither
    rotated nor with outliers zeroed, take one global scale computed from
    the tensor: a max is the same in any order.

    Args:
        program: A gradient program with its fake-quants unlowered,
            rewritten in place.
    """
    graph = program.graph
    groups = {}
    for node in graph.nodes:
        if node.target is not _fake_quantize:
            continue
        fake_quant_attr, source = node.args
        fake_quant = program.get_submodule(fake_quant_attr.target)
        if not (
            isinstance(fake_quant, MXFakeQuantize)
            and fake_quant.global_scale
            and fake_quant.rht_axis is None
            and fake_quant.outlier_threshold is None
        ):
            continue
        while source.target in _VIEWS:
            source = source.args[0]
        key = (source, fake_quant.quant_max, fake_quant.scale_max)
        groups.setdefault(key, []).append(node)
    for (source, quant_max, scale_max), calls in groups.items():
        if len(calls) < 2:
            continue
        with graph.inserting_before(calls[0]):
            scale = graph.call_function(
                calculate_global_scale, (source, quant_max, scale_max)
            )
        for node in calls:
            node.kwargs = {"global_scale": scale}


def return_gradients(program: GraphModule) -> List[str]:
    """Make a gradient program return its gradients after the loss instead
    of copying them into its gradient buffers, in place.  Each ``copy_``
    into a ``<name>_grad`` input becomes an output, in the input's dtype,
    and the input goes.

    Args:
        program: A program ``gradient_program`` built.

    Returns:
        Each output gradient's parameter name, as the program's inputs spell
        it, in output order.
    """
    graph = program.graph
    output = graph.output_node()
    names, gradients = [], []
    for write in graph.find_nodes(
        op="call_function", target=aten.copy_.default
    ):
        buffer, gradient = write.args
        if buffer.op != "placeholder" or not buffer.name.endswith("_grad"):
            continue
        # A conversion to the dtype it already has is no op to Inductor.
        with graph.inserting_before(write):
            gradient = graph.call_function(
                _CONVERT_ELEMENT_TYPE, (gradient, buffer.meta["val"].dtype)
            )
        gradient.meta["val"] = buffer.meta["val"]
        names.append(buffer.name.removesuffix("_grad"))
        gradients.append(gradient)
        graph.erase_node(write)
        graph.erase_node(buffer)
    output.args = ((*output.args[0], *gradients),)
    graph.lint()
    program.recompile()
    return names


def split_weight_quantization(program: GraphModule) -> GraphModule:
    """Move the quantization of ``program``'s weights into a program of its
    own.

    A weight's fake-quant -- a fake-quant call on a parameter or a view of
    one -- reads the parameters alone, so its output serves every step until
    they change.  Each such call becomes an input of ``program``, after its
    others, unless it rounds stochastically, which draws afresh on every
    call.  No gradient flows through a fake-quant here: the joint graph was
    differentiated before the fake-quants went in.

    Args:
        program: A gradient program with its fake-quants unlowered,
            rewritten in place.

    Returns:
        A program taking parameters, named as in ``program``, and returning
        ``program``'s new inputs, in order.
    """
    graph = program.graph
    calls = []
    for node in graph.find_nodes(op="call_function", target=_fake_quantize):
        fake_quant_attr, source = node.args
        while source.target in _VIEWS:
            source = source.args[0]
        fake_quant = program.get_submodule(fake_quant_attr.target)
        if isinstance(source.meta.get("desc"), ParamAOTInput) and not getattr(
            fake_quant, "stochastic_rounding", False
        ):
            calls.append(node)
    needed, pending = set(), list(calls)
    while pending:
        node = pending.pop()
        if node not in needed:
            needed.add(node)
            pending.extend(node.all_input_nodes)
    weights = Graph()
    copies = {}
    for node in graph.nodes:
        if node in needed:
            copies[node] = weights.node_copy(node, copies.__getitem__)
    weights.output(tuple(copies[node] for node in calls))
    last = graph.find_nodes(op="placeholder")[-1]
    for node in calls:
        with graph.inserting_after(last):
            last = graph.placeholder(node.name)
        last.meta["val"] = node.meta["val"]
        node.replace_all_uses_with(last)
        graph.erase_node(node)
    graph.eliminate_dead_code()
    program.recompile()
    return GraphModule(program, weights)


class TrainingStep:
    """The loss and the gradients of a training step of ``model``.

    A step runs eagerly -- the model's forward and ``loss.backward()`` -- or
    as the model's ``gradient_program`` under ``torch.compile``.  The
    compiled program keeps the fake-quants as calls to the model's own,
    unless ``lower`` rewrites its quantized GEMMs as real kernels.  On a GPU
    those fake-quants round as eager's do, but Inductor may sum reductions
    in another order, so results can differ from eager's in the last bits,
    which 4-bit rounding can amplify; on a CPU Inductor's kernels also round
    differently.  Each call reads the model's parameters, buffers and
    fake-quants afresh, so an optimizer's update, a refit lookup table or a
    loaded checkpoint takes effect on the next one -- except inside
    ``fixed_weights``, where the weights are quantized once for every
    compiled step, and in a lowered program, which holds its lookup tables
    as constants.  Compiled code runs no hooks: a step whose fake-quants
    are hooked, as in a codebook refit window, must run eagerly
    (``eager=True``).

    Args:
        model: Model to train, whose forward returns its loss as the
            ``loss`` of its output; to compile, one ``prepare_training``
            prepared.
        compile: Run the steps the caller does not ask to run eagerly as the
            compiled program; otherwise every step runs eagerly.
        lower: Rewrites the program's quantized GEMMs as real kernels in
            place; None keeps the
            fake-quants.
    """

    def __init__(
        self,
        model: nn.Module,
        compile: bool,
        lower: Optional[Callable[[GraphModule], Any]] = None,
    ) -> None:
        self.model = model
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.program = None
        # The quantized weights ``fixed_weights`` holds.
        self.quantized = None
        self.fixed = False
        if not compile:
            return
        program = gradient_program(model, lower_fake_quants=lower is not None)
        if lower is not None:
            lower(program)
        index = {name: i for i, name in enumerate(return_gradients(program))}
        self.order = [
            index[_parameter_name(name)]
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
        # Each input is a model input (None) or the (module, attribute) of a
        # tensor of the model, read on every call.
        sources = {}
        for name, _ in [
            *model.named_parameters(remove_duplicate=False),
            *model.named_buffers(remove_duplicate=False),
        ]:
            path, _, attr = name.rpartition(".")
            sources[_parameter_name(name)] = (model.get_submodule(path), attr)
        self.inputs = [
            sources.get(node.name)
            for node in program.graph.find_nodes(op="placeholder")
        ]
        weights = split_weight_quantization(program)
        self.weight_inputs = [
            sources[node.name]
            for node in weights.graph.find_nodes(op="placeholder")
        ]
        self.in_spec = model.forward.in_spec
        options = EAGER_NUMERICS if lower is None else None
        self.weights = torch.compile(weights, options=options)
        self.program = torch.compile(program, options=options)

    @contextlib.contextmanager
    def fixed_weights(self):
        """Quantize each weight once for the compiled steps run inside.

        The first compiled step quantizes the weights and the rest reuse
        them, so nothing a weight's fake-quant reads -- the parameters, a
        lookup table -- may change inside.  Outside, each step quantizes
        them afresh.  A weight that rounds stochastically draws afresh on
        every step either way (``split_weight_quantization``).
        """
        self.fixed = True
        try:
            yield
        finally:
            self.fixed, self.quantized = False, None

    def __call__(self, *args, eager: bool, **kwargs):
        """Run a step on a batch.

        Args:
            *args: The model's positional inputs.
            eager: Run this step eagerly, as a step whose fake-quants are
                hooked must.
            **kwargs: The model's keyword inputs.

        Returns:
            ``(loss, gradients)``: the loss, detached, and the gradients of
            ``model``'s trainable parameters, in ``model.parameters()``
            order.
        """
        if eager or self.program is None:
            with torch.enable_grad():
                loss = self.model(*args, **kwargs).loss
                loss.backward()
            grads = [p.grad for p in self.params]
            for param in self.params:
                param.grad = None
            return loss.detach(), grads
        flat = _check_inputs_match(args, kwargs, self.in_spec)
        batch = (leaf for _, leaf in flat)
        with torch.no_grad():
            quantized = self.quantized
            if quantized is None:
                quantized = self.weights(
                    *(getattr(*source) for source in self.weight_inputs)
                )
                if self.fixed:
                    self.quantized = quantized
            loss, *grads = self.program(
                *(
                    next(batch) if source is None else getattr(*source)
                    for source in self.inputs
                ),
                *quantized,
            )
        return loss, [grads[i] for i in self.order]


def update_program(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> GraphModule:
    """``optimizer``'s update of ``model``'s parameters, as a graph.

    Captured from ``optimizer.step`` itself, so it is whatever update the
    optimizer makes.  An optimizer of the same class and groups steps meta
    copies of the parameters twice under dynamo: the first step creates its
    state, and the second step's graph is the program.  It is capturable
    where the optimizer offers that, so its step count and other scalars
    stay tensors instead of Python numbers.  Every tensor that step reads
    is an input -- the parameters, their gradients, the optimizer's state
    and each group's learning rate, a 0-d tensor so a schedule can change
    it -- and every tensor it updates is written in place.

    Args:
        model: Model whose parameters ``optimizer`` updates.
        optimizer: The optimizer training ``model``.

    Returns:
        A graph whose inputs are named after the parameters as
        ``gradient_program`` names them: ``fc1_weight``, its gradient
        ``fc1_weight_grad`` and each state tensor under its key, such as
        ``fc1_weight_exp_avg``; each group's learning rate is ``lr_<i>``.
    """
    names = {p: _parameter_name(n) for n, p in model.named_parameters()}
    inputs = {}
    groups = []
    for i, group in enumerate(optimizer.param_groups):
        params = []
        for param in group["params"]:
            meta_param = nn.Parameter(
                torch.empty_like(param, device="meta"), param.requires_grad
            )
            inputs[meta_param] = names[param]
            if param.requires_grad:
                meta_param.grad = torch.empty_like(meta_param)
                inputs[meta_param.grad] = f"{names[param]}_grad"
            params.append(meta_param)
        lr = torch.tensor(float(group["lr"]))
        inputs[lr] = f"lr_{i}"
        groups.append({**group, "params": params, "lr": lr})
        if "capturable" in group:
            groups[-1]["capturable"] = True
    stand_in = type(optimizer)(groups)

    captured = []

    def backend(graph, example_inputs):
        attributes = dict(graph.named_parameters())
        attributes.update(graph.named_buffers())

        def keep(program, _):
            captured.append((program, attributes, example_inputs))
            return program

        return aot_autograd(
            fw_compiler=keep, keep_inference_input_mutations=True
        )(graph, example_inputs)

    step = torch.compile(stand_in.step, backend=backend)
    step()
    first = len(captured)
    step()
    # The first step creates the optimizer's state; the second retraces
    # unless that left the step unchanged.
    ((program, attributes, example_inputs),) = captured[first:] or captured
    # Named after the compiled step, which may replace a state tensor, such
    # as moving a step count to the parameters' device.
    for meta_param, state in stand_in.state.items():
        for key, value in state.items():
            inputs[value] = f"{inputs[meta_param]}_{key}"
    for node in program.graph.nodes:
        if node.op != "placeholder":
            continue
        desc = node.meta["desc"]
        tensor = (
            example_inputs[desc.idx]
            if isinstance(desc, PlainAOTInput)
            else attributes[desc.target]
        )
        node._rename(inputs[tensor])
        node.target = node.name
    # Dynamo writes an update as ``copy_(x, copy(x, value))``, where the
    # inner copy only gives ``value`` the shape and dtype it already has.
    graph = program.graph
    for node in list(graph.nodes):
        if node.target is not aten.copy.default:
            continue
        destination, value = (a.meta["val"] for a in node.args)
        if (destination.shape, destination.dtype) == (value.shape, value.dtype):
            node.replace_all_uses_with(node.args[1])
            graph.erase_node(node)
    # A tensor the step creates, such as ``where``'s scalar operand, goes on
    # the parameters' device, not the stand-in's.
    device = next(model.parameters()).device
    for node in graph.nodes:
        if node.kwargs.get("device") == torch.device("meta"):
            node.update_kwarg("device", device)
    program.recompile()
    return program


def gemm_partners(model: nn.Module) -> Dict[str, Set[str]]:
    """Each fake-quant of a prepared model -> the ones it is contracted with.

    Read off the joint graph ``capture_training`` keeps: a GEMM operand
    traced back past its views reaches the fake-quant that quantized it, and
    the GEMM's other operand traced the same way names its partner.  An
    operand no fake-quant quantized names none.

    Args:
        model: Model ``prepare_training`` prepared.

    Returns:
        ``name -> partner names``, by the names in
        ``model.training_quantizers``.
    """
    joint = model.forward.graphs["joint"]
    names = {
        id(module): name
        for name, module in model.training_quantizers.named_children()
    }
    partners = {name: set() for name in names.values()}
    for gemm in joint.graph.nodes:
        if gemm.target not in _OPS:
            continue
        quantized = []
        for operand in gemm.args[:2]:
            while operand.target in _VIEWS:
                operand = operand.args[0]
            quantized.append(
                names[id(joint.get_submodule(operand.args[0].target))]
                if operand.target is _fake_quantize
                else None
            )
        first, second = quantized
        if first is not None and second is not None:
            partners[first].add(second)
            partners[second].add(first)
    return partners


def forward_gemm_operands(
    model: nn.Module,
) -> Dict[str, List[Tuple[Node, int]]]:
    """Each fake-quant of a prepared model feeding forward GEMMs -> where.

    Read off the joint graph ``capture_training`` keeps, as
    ``gemm_partners`` reads it: a forward GEMM's operand traced back past
    its views reaches the fake-quant that quantized it.

    Args:
        model: Model ``prepare_training`` prepared.

    Returns:
        ``name -> [(GEMM, position)]``, by the names in
        ``model.training_quantizers``: the forward GEMMs each fake-quant
        feeds, in graph order, and which of their two operands it is.
    """
    joint = model.forward.graphs["joint"]
    names = {
        id(module): name
        for name, module in model.training_quantizers.named_children()
    }
    operands = {}
    for gemm in joint.graph.nodes:
        if gemm.meta.get("gemm_kind") != "fprop":
            continue
        for position, operand in enumerate(gemm.args[:2]):
            while operand.target in _VIEWS:
                operand = operand.args[0]
            if operand.target is _fake_quantize:
                fake_quant = joint.get_submodule(operand.args[0].target)
                operands.setdefault(names[id(fake_quant)], []).append(
                    (gemm, position)
                )
    return operands


def operand_gradient(gemm: Node, position: int) -> Optional[Node]:
    """Return the backward GEMM computing a forward GEMM operand's gradient.

    A forward GEMM's backward GEMMs carry its ``seq_nr``, and the dgrad of
    its operand ``position`` takes the incoming gradient at that same
    position.

    Args:
        gemm: A forward GEMM of the joint graph.
        position: Which of its two operands, 0 or 1.

    Returns:
        The dgrad node, or None when the operand takes no gradient.

    Raises:
        ValueError: Several backward GEMMs match.
    """
    matches = [
        node
        for node in gemm.graph.nodes
        if node.meta.get("gemm_kind") == "dgrad"
        and node.meta["seq_nr"] == gemm.meta["seq_nr"]
        and node.meta["operand_types"][position] == "error"
    ]
    if len(matches) > 1:
        raise ValueError(
            f"{len(matches)} backward GEMMs compute the gradient of "
            f"{gemm.name}'s operand {position}"
        )
    return matches[0] if matches else None


def disable_observers(model: nn.Module) -> None:
    """Fix every fake-quant's scale except the gradients', which keep moving.

    Args:
        model: Prepared graph, or a model prepared by ``prepare_training``.
    """
    for module in model.modules():
        if isinstance(module, FakeQuantizeBase):
            module.disable_observer()
    for module in model.modules():
        if isinstance(module, TrainingQuantizers):
            for name in module.gradients:
                getattr(module, name).enable_observer()


def prepare_for_qat(
    model: nn.Module,
    quantizer: XNNPACKQuantizer,
    example_args: Tuple[Any, ...],
    example_kwargs: Optional[Dict[str, Any]],
    dynamic_shapes,
    merge_linears: bool,
) -> GraphModule:
    """Prepare ``model`` for QAT: its forward GEMMs train through
    ``quantizer``'s fake-quants, and the backward pass stays in high
    precision.

    ``model`` is exported in training mode and prepared by
    ``prepare_qat_pt2e``: each conv-BN pair trains through the simulated
    fold, and ``train()`` and ``eval()`` switch the graph's dropouts and
    batch norms.  ``prepare_training`` instead quantizes the backward pass
    too, and captures the graph a training step compiles from.

    Args:
        model: Float model.
        quantizer: Picks the GEMM operands to quantize and their specs.
        example_args: Positional inputs to export with.
        example_kwargs: Keyword inputs to export with.
        dynamic_shapes: Dynamic dimensions of the inputs.
        merge_linears: Merge the bias-free linears that read one input into
            one GEMM (``merge_sibling_linears``).

    Returns:
        The prepared graph.
    """
    exported = export_model(
        model.train(),
        example_args,
        example_kwargs,
        dynamic_shapes=dynamic_shapes,
        merge_linears=merge_linears,
    )
    model = prepare_qat_pt2e(exported, quantizer)

    def train(self, mode=True):
        set_training(self, mode)
        return self

    model.train = types.MethodType(train, model)
    model.eval = types.MethodType(nn.Module.eval, model)
    return model

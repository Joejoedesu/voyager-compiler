"""Voyager reduction opcodes with explicit, schema-visible scratch writes.

These are tile bodies in the shared buffer pipeline, with the established
Voyager ABI. Other targets use with_scratch or their own registered bodies.
"""

from typing import Optional, Tuple, Union

import torch
from torch.library import Library, impl

from voyager_compiler.codegen.subgraph import copy_graph_module
from voyager_compiler.extensions import OperatorSemantics

quantized_ops_lib = Library("quantized_ops", "FRAGMENT")

# ``mean`` / ``variance`` / ``normalized`` are the on-chip regions the
# accelerator keeps between its passes over the data.  The graph never reads
# them, so eager ignores them; naming them on the op is what tells the backend
# which bytes are its to write (see ``reduction_scratch``).
quantized_ops_lib.define(
    "layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, "
    "Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True, "
    "Tensor(a!)? mean=None, Tensor(b!)? variance=None, "
    "Tensor(c!)? normalized=None) "
    "-> Tensor"
)


@impl(quantized_ops_lib, "layer_norm", "CompositeExplicitAutograd")
def layer_norm(
    input: torch.Tensor,
    normalized_shape: Union[int, Tuple[int]],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    cudnn_enable: bool = True,
    mean: Optional[torch.Tensor] = None,
    variance: Optional[torch.Tensor] = None,
    normalized: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    k = normalized_shape[-1]
    output = torch.ops.aten.layer_norm.default(
        input[..., :k],
        normalized_shape,
        None if weight is None else weight[..., :k],
        None if bias is None else bias[..., :k],
        eps,
        cudnn_enable,
    )
    # Pad the output back to the original input shape
    output = torch.nn.functional.pad(
        output, (0, input.shape[-1] - normalized_shape[-1])
    )
    return output


quantized_ops_lib.define(
    "softmax(Tensor input, int dim, ScalarType? dtype=None, "
    "Tensor(a!)? max=None, Tensor(b!)? sum=None) -> Tensor"
)


@impl(quantized_ops_lib, "softmax", "CompositeExplicitAutograd")
def softmax(
    input: torch.Tensor,
    dim: int,
    dtype: Optional[torch.dtype] = None,
    max: Optional[torch.Tensor] = None,
    sum: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``aten.softmax`` that also names its on-chip scratch.

    ``max`` and ``sum`` are the row statistics the accelerator holds between
    its passes; eager computes the softmax in one go and ignores them.
    """
    return torch.ops.aten.softmax.int(input, dim, dtype)


REDUCTIONS = {
    torch.ops.aten.layer_norm.default: torch.ops.quantized_ops.layer_norm.default,
    torch.ops.aten.softmax.int: torch.ops.quantized_ops.softmax.default,
}
OPERATORS = tuple((op, OperatorSemantics()) for op in REDUCTIONS.values())


def scratch_kernel(node, names, fallback):
    """Use the established Voyager instruction body where its ABI applies."""
    if not names:
        return fallback
    if node.op == "call_module":
        submod = node.meta.get("submodule")
        gm = copy_graph_module(submod)
        reduction = next(
            (n for n in gm.graph.nodes if n.target in REDUCTIONS), None
        )
        if reduction is None:
            return fallback
        placeholders = [p for p in gm.graph.nodes if p.op == "placeholder"]
        anchor = placeholders[0]
        for placeholder in placeholders[1:]:
            anchor.append(placeholder)
            anchor = placeholder
        scratch = []
        for name in names:
            with gm.graph.inserting_after(anchor):
                anchor = gm.graph.placeholder(f"scratch_{name}")
            scratch.append(anchor)
        reduction.target = REDUCTIONS[reduction.target]
        reduction.kwargs = {**reduction.kwargs, **dict(zip(names, scratch))}
        gm.graph.lint()
        gm.recompile()
        return gm
    if node.target not in REDUCTIONS:
        return fallback
    inputs = node.all_input_nodes
    order = {n: i for i, n in enumerate(inputs)}
    plain = lambda value: list(value) if isinstance(value, list) else value
    arg_slots = [
        order[a] if isinstance(a, torch.fx.Node) else None for a in node.args
    ]
    kw_slots = {
        k: order[v] if isinstance(v, torch.fx.Node) else None
        for k, v in node.kwargs.items()
    }
    op_args = [plain(a) for a in node.args]
    op_kwargs = {k: plain(v) for k, v in node.kwargs.items()}
    op = REDUCTIONS[node.target]
    n_scratch = len(names)

    def compute(*tiles):
        end = len(tiles) - n_scratch
        args = [
            tiles[i] if i is not None else a for i, a in zip(arg_slots, op_args)
        ]
        kwargs = {
            k: tiles[i] if i is not None else op_kwargs[k]
            for k, i in kw_slots.items()
        }
        kwargs.update(zip(names, tiles[end:]))
        return op(*args, **kwargs)

    return compute


def build_pointwise(node, context, parameters):
    from voyager_compiler.compiler.bufferize.pipeline import (
        build_pointwise as shared_build,
    )

    return shared_build(node, context, kernel_factory=scratch_kernel)

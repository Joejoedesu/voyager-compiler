"""Run a training step on a GPU with real FP8 and NVFP4 GEMMs.

``gradient_program`` writes a training step -- the forward, the loss and the
backward -- as one graph whose quantized GEMMs, forward (fprop), input
gradient (dgrad) and weight gradient (wgrad), read their operands' codes.  An
FP8 operand is a ``quantize`` against its delayed-scaling scale, which the
graph updates in place, read through views; its GEMM is scaled back by a
``dequantize``.  An NVFP4 operand is a ``quantize_mx`` blocked along its
GEMM's contraction axis, or in 16x16 blocks for a weight, after its random
Hadamard transform where the recipe has one; its GEMM is a ``linear_mx`` or
``matmul_mx`` scaled by ``alpha``.

``lower_training`` swaps each such GEMM for a real one, in place, as
``lower`` does for inference:

- each FP8 ``quantize`` becomes one cast to FP8 (``quantize_fp8``), and each
  GEMM with its dequantize one ``fp8_linear``.  That GEMM takes both operands
  K-major (TN), the only layout torch's FP8 GEMM takes, so an operand a dgrad
  or wgrad contracts along its rows is a transposed copy of its FP8 codes.  A
  matrix the program takes as input and a dgrad reads, a weight, is cast to
  both layouts by one Triton kernel (``quantize_fp8_transpose``), which also
  gives its delayed-scaling update the matrix's amax.  An activation's codes,
  and the transposed copy its wgrads read, are each stored once for all the
  GEMMs reading them, and its delayed-scaling amax is taken row by row;
- each NVFP4 ``quantize_mx`` becomes one cast to packed codes
  (``quantize_nvfp4``, with its stochastic rounding's seed), and each GEMM one
  ``nvfp4_linear``.  A weight in 16x16 blocks is cast once
  (``quantize_nvfp4_square``) and packed both ways, so the dgrad reads the
  forward's codes transposed.  The random Hadamard transform in front of a
  cast is folded into it, the cast reading the graph's Hadamard matrix, one
  Triton kernel reading the operand in place.  A tensor cast both ways, its
  rows and its transpose's, is first merged into one
  ``quantize_mx_transpose`` (``pair_quantize_transposes``), and becomes one
  cast to both layouts (``quantize_nvfp4_pair``).

The delayed-scaling updates and every other op run as written.
"""

import operator
from typing import Dict, List, Optional, Tuple, Union

import torch

# Registers torch.ops.prims.inductor_force_stride_order.
import torch._inductor.inductor_prims  # noqa: F401
from torch._prims_common import make_contiguous_strides_for
from torch.fx import Graph, GraphModule, Node

from voyager_compiler.codegen.node_info import get_arg_value
from voyager_compiler.quantization.training import (
    _VIEWS,
    pair_quantize_transposes,
    rotation_input,
)

from . import nvfp4
from .fp8 import (
    DEQUANTIZE,
    FP8_DTYPES,
    LINEAR,
    MATMUL,
    QUANTIZE,
    fp8_linear,
    quantize_fp8,
    quantize_fp8_transpose,
)
from .nvfp4 import (
    BLOCK_SIZE,
    FP4_MAX,
    LINEAR_MX,
    NVFP4_DTYPES,
    NVFP4_LINEAR,
    QUANTIZE_MX,
    quantize_nvfp4,
    quantize_nvfp4_pair,
    quantize_nvfp4_square,
)
from .rmsnorm import fuse_rmsnorm_backward

__all__ = ["lower_training"]

aten = torch.ops.aten

MATMUL_MX = torch.ops.quantized_ops.matmul_mx.default
QUANTIZE_MX_TRANSPOSE = torch.ops.quantized_ops.quantize_mx_transpose.default
# The indices of ``quantize_mx_transpose``'s row-wise scale and codes; the
# transpose's follow them.
PAIR_ROWS = (0, 1)
# Stores a tensor in one buffer of the given strides; Inductor gives each
# GEMM reading a tensor it has not stored a copy of its own.
FORCE_STRIDES = torch.ops.prims.inductor_force_stride_order.default
# The views that only reshape a tensor.
RESHAPES = (aten.view.default, aten._unsafe_view.default, aten.reshape.default)

# A cast's packed codes and block scales, by the axis a GEMM reads them
# along.
Packed = Dict[int, Tuple[Node, Node]]

# The axes of a ``quantize_mx`` in 16x16 blocks.
SQUARE_BLOCKS = (-2, -1)


def _fp8_quantize(operand: Node) -> Node:
    """The FP8 ``quantize`` whose codes a GEMM reads, as they are or through
    views, as ``operand``.

    Raises:
        NotImplementedError: ``operand`` is not cast to FP8.
    """
    quantize = operand
    while quantize.target in _VIEWS:
        quantize = quantize.args[0]
    if (
        quantize.target is not QUANTIZE
        or quantize.meta.get("dtype") not in FP8_DTYPES
    ):
        raise NotImplementedError(f"{operand.name} is not cast to FP8")
    return quantize


def _insert_fp8_gemm(
    graph: Graph, gemm: Node, transposes: Dict[Node, Node]
) -> Node:
    """Insert the real GEMM computing an FP8 ``gemm`` and the dequantize after
    it, before that dequantize.

    A linear reads its ``[N, K]`` weight as it is; a matmul reads its
    ``[K, N]`` second operand's transposed codes where its cast made them,
    and otherwise a transposed copy, so the GEMM reads it K-major.

    Args:
        graph: The program's graph, its FP8 casts in place.
        gemm: A linear, or a matmul whose second operand is 2-d, of two FP8
            operands; its only user scales it back.
        transposes: Each matrix's FP8 codes -> its transpose's, for the
            weights cast both ways and the activations wgrads read.

    Returns:
        The GEMM.
    """
    (output,) = gemm.users
    input, other = gemm.args
    with graph.inserting_before(output):
        scale = graph.call_function(
            aten._to_copy.default, (output.args[1],), {"dtype": torch.float32}
        )
        if gemm.target is MATMUL:
            other = transposes.get(other) or graph.call_function(
                aten.clone.default,
                (graph.call_function(aten.t.default, (other,)),),
                {"memory_format": torch.contiguous_format},
            )
        return graph.call_function(fp8_linear, (input, other, scale))


def _nvfp4_pair(codes: Node) -> Optional[Tuple[Node, str]]:
    """The ``quantize_mx_transpose`` whose codes a GEMM reads as ``codes``,
    through a view or a permute, and which of its layouts: ``"rows"`` or
    ``"transpose"``; None if ``codes`` are not a pair's.

    Raises:
        NotImplementedError: The pair's codes are not NVFP4 codes.
    """
    node = codes
    while node.target in (aten.view.default, aten.permute.default):
        node = node.args[0]
    if (
        node.target is not operator.getitem
        or node.args[0].target is not QUANTIZE_MX_TRANSPOSE
    ):
        return None
    pair = node.args[0]
    if (
        pair.meta.get("dtype") != NVFP4_DTYPES
        or pair.args[2:4] != (BLOCK_SIZE, FP4_MAX)
        or get_arg_value(pair, 8, "global_scale") is None
    ):
        raise NotImplementedError(f"{codes.name} is not cast to NVFP4")
    return pair, "rows" if node.args[1] in PAIR_ROWS else "transpose"


def _insert_pair_cast(graph: Graph, pair: Node) -> Dict[str, Packed]:
    """Insert the cast replacing an NVFP4 ``quantize_mx_transpose``
    (``quantize_nvfp4_pair``), before it, and make each ``alpha`` read its
    global scales.

    Args:
        graph: The program's graph.
        pair: The ``quantize_mx_transpose``.

    Returns:
        Each layout's packed codes and block scales, by the axis a GEMM
        reads them along: the row-wise ones along the last, their codes in
        the shape the pair's are viewed in; the transposed ones along the
        transpose's last, or in place along the second-to-last.
    """
    (codes,) = [user for user in pair.users if user.args[1] == PAIR_ROWS[1]]
    (view,) = codes.users
    with graph.inserting_before(pair):
        cast = graph.call_function(
            quantize_nvfp4_pair,
            (
                pair.args[0],
                get_arg_value(pair, 7, "seed"),
                get_arg_value(pair, 9, "transpose_seed"),
                get_arg_value(pair, 11, "hadamard"),
            ),
        )
        outputs = [
            graph.call_function(operator.getitem, (cast, i)) for i in range(6)
        ]
        outputs[0] = graph.call_function(
            aten.view.default, (outputs[0], [*view.args[1][:-1], -1])
        )
    get_arg_value(pair, 8, "global_scale").replace_all_uses_with(outputs[2])
    get_arg_value(pair, 10, "transpose_global_scale").replace_all_uses_with(
        outputs[5]
    )
    transpose = tuple(outputs[3:5])
    return {
        "rows": {-1: tuple(outputs[:2])},
        "transpose": {-1: transpose, -2: transpose},
    }


def _nvfp4_quantize(codes: Node) -> Node:
    """The NVFP4 ``quantize_mx`` whose codes a GEMM reads as ``codes``.

    Raises:
        NotImplementedError: ``codes`` are not NVFP4 codes.
    """
    quantize = codes.args[0] if codes.target is operator.getitem else codes
    if (
        quantize.target is not QUANTIZE_MX
        or quantize.meta.get("dtype") != NVFP4_DTYPES
        or quantize.args[3:5] != (BLOCK_SIZE, FP4_MAX)
        or get_arg_value(quantize, 9, "global_scale") is None
    ):
        raise NotImplementedError(f"{codes.name} is not cast to NVFP4")
    return quantize


def _insert_nvfp4_cast(graph: Graph, quantize: Node) -> Packed:
    """Insert the cast replacing an NVFP4 ``quantize``, before it, and make
    each ``alpha`` read the cast's global scale.  A rotation ending at the
    quantize's input, along its blocks' axis, is cast with it.

    Args:
        graph: The program's graph.
        quantize: An NVFP4 ``quantize_mx``.

    Returns:
        The cast's packed codes and block scales, by the axis of the tensor
        they are packed along: its blocks' axis, or both axes for 16x16
        blocks.

    Raises:
        NotImplementedError: ``quantize`` rounds 16x16 blocks stochastically,
            or blocks along two axes that are not the last two.
    """
    input, axes = quantize.args[0], tuple(quantize.args[2])
    seed = get_arg_value(quantize, 8, "seed")
    with graph.inserting_before(quantize):
        if len(axes) == 1:
            operand, hadamard = input, None
            rotation = rotation_input(input)
            ndim = input.meta["val"].ndim
            if rotation is not None and rotation[1] == axes[0] % ndim:
                operand, _, hadamard = rotation
            cast = graph.call_function(
                quantize_nvfp4, (operand, axes[0], seed, hadamard)
            )
            packed_along = axes
        elif axes == SQUARE_BLOCKS and seed is None:
            cast = graph.call_function(quantize_nvfp4_square, (input,))
            packed_along = (-1, -2)
        else:
            raise NotImplementedError(
                f"{quantize.name}: blocks along {axes} have no NVFP4 cast"
            )
        outputs = [
            graph.call_function(operator.getitem, (cast, i))
            for i in range(2 * len(packed_along) + 1)
        ]
    get_arg_value(quantize, 9, "global_scale").replace_all_uses_with(
        outputs[-1]
    )
    return {
        axis: tuple(outputs[2 * i : 2 * i + 2])
        for i, axis in enumerate(packed_along)
    }


def _insert_nvfp4_gemm(
    program: GraphModule,
    gemm: Node,
    casts: Dict[Node, Union[Packed, Dict[str, Packed]]],
) -> Node:
    """Insert the casts of an NVFP4 ``gemm``'s operands no earlier GEMM
    inserted, and its real GEMM before it.

    Args:
        program: The gradient program.
        gemm: A ``linear_mx``, or a ``matmul_mx`` whose second operand is
            2-d.
        casts: Each NVFP4 ``quantize_mx`` or ``quantize_mx_transpose`` cast
            so far -> its packed operands, as ``_insert_nvfp4_cast`` or
            ``_insert_pair_cast`` returns them; updated.

    Returns:
        The GEMM.

    Raises:
        NotImplementedError: An operand is not NVFP4 codes blocked along
            the GEMM's contraction axis.
    """
    graph = program.graph
    # The axis each operand is contracted along: a linear's weight is
    # [N, K], a matmul's second operand [K, N].
    contraction = (-1, -1 if gemm.target is LINEAR_MX else -2)
    operands = []
    for codes, axis in zip(gemm.args[:2], contraction):
        pair = _nvfp4_pair(codes)
        if pair is not None:
            source, layout = pair
            if source not in casts:
                casts[source] = _insert_pair_cast(graph, source)
            packed = casts[source][layout]
        else:
            quantize = _nvfp4_quantize(codes)
            if quantize not in casts:
                casts[quantize] = _insert_nvfp4_cast(graph, quantize)
            packed = casts[quantize]
        if axis not in packed:
            raise NotImplementedError(
                f"{codes.name} is not blocked along {gemm.name}'s "
                "contraction axis"
            )
        operands.extend(packed[axis])
    bias = get_arg_value(gemm, 2, "bias")
    with graph.inserting_before(gemm):
        alpha, one = nvfp4.insert_gemm_scalars(
            program, gemm.kwargs["alpha"], gemm.meta["val"].device
        )
        return graph.call_function(NVFP4_LINEAR, (*operands, alpha, one, bias))


def _whole_amaxes(tensor: Node) -> List[Node]:
    """The ``amax(abs(tensor))`` reductions over the whole of ``tensor``, as
    a delayed-scaling update takes them."""
    return [
        reduction
        for magnitude in tensor.users
        if magnitude.target is aten.abs.default
        for reduction in magnitude.users
        if reduction.target is aten.amax.default and reduction.args[1:] == ([],)
    ]


def _insert_activation_cast(
    graph: Graph,
    quantize: Node,
    dtype: torch.dtype,
    transposes: Dict[Node, Node],
) -> Node:
    """Cast an activation to FP8 once for every GEMM reading it.

    Inserts, before ``quantize``, a ``quantize_fp8`` whose codes are stored
    in one contiguous buffer (``FORCE_STRIDES``).  The codes as one
    ``[rows, K]`` matrix -- themselves, or one view standing for every
    reshape of ``quantize`` to it -- get one stored transposed copy for the
    wgrads, added to ``transposes``; dead-code elimination drops it if no
    wgrad reads it.  Each delayed-scaling ``amax(abs(input))`` over the
    whole input becomes the amax of its rows' amaxes, the same value, which
    Inductor computes in the kernel making the input.

    Args:
        graph: The program's graph.
        quantize: The FP8 ``quantize`` of an activation.
        dtype: Its torch FP8 dtype.
        transposes: Each matrix's FP8 codes -> its transpose's; updated in
            place.

    Returns:
        The codes.
    """
    input = quantize.args[0]
    shape = input.meta["val"].shape
    rows = shape[:-1].numel()
    with graph.inserting_before(quantize):
        cast = graph.call_function(quantize_fp8, (*quantize.args[:2], dtype))
        codes = graph.call_function(
            FORCE_STRIDES, (cast, make_contiguous_strides_for(shape))
        )
    matrix = codes
    if len(shape) > 2:
        with graph.inserting_before(quantize):
            matrix = graph.call_function(
                aten.reshape.default, (codes, [rows, shape[-1]])
            )
        views = [
            view
            for view in quantize.users
            if view.target in RESHAPES
            and view.meta["val"].shape == (rows, shape[-1])
        ]
        for view in views:
            view.replace_all_uses_with(matrix)
    with graph.inserting_before(quantize):
        transposes[matrix] = graph.call_function(
            FORCE_STRIDES,
            (graph.call_function(aten.t.default, (matrix,)), [rows, 1]),
        )
    for reduction in _whole_amaxes(input):
        with graph.inserting_before(reduction):
            row_amaxes = graph.call_function(
                aten.amax.default, (reduction.args[0], [-1])
            )
        reduction.args = (row_amaxes, [])
    return codes


def _insert_input_matrix_cast(
    graph: Graph, quantize: Node, dtype: torch.dtype
) -> Tuple[Node, Node]:
    """Cast a placeholder matrix to FP8 in both layouts in one op.

    Inserts a ``quantize_fp8_transpose`` replacing ``quantize``, before it,
    and makes every ``amax(abs(matrix))`` over the whole matrix, the
    delayed-scaling update's, read the cast's amax instead.  The ops reading
    that amax, the amax history's update, move after the cast: the scale the
    cast divides by comes from the history before that update.

    Args:
        graph: The program's graph.
        quantize: The FP8 ``quantize`` of a 2-d placeholder.
        dtype: Its torch FP8 dtype.

    Returns:
        The matrix's FP8 codes and its transpose's.
    """
    matrix = quantize.args[0]
    with graph.inserting_before(quantize):
        cast = graph.call_function(
            quantize_fp8_transpose, (*quantize.args[:2], dtype)
        )
        codes, transpose, amax = (
            graph.call_function(operator.getitem, (cast, i)) for i in range(3)
        )
    for reduction in _whole_amaxes(matrix):
        reduction.replace_all_uses_with(amax)
        moved, last = {amax}, amax
        node = reduction.next
        while node is not cast:
            following = node.next
            if moved.intersection(node.all_input_nodes):
                moved.add(node)
                last.append(node)
                last = node
            node = following
    return codes, transpose


def _lower_fp8_gemms(graph: Graph) -> List[str]:
    """Replace each FP8 ``quantize`` with one cast to FP8 that every GEMM it
    fed reads, and each FP8 GEMM with its dequantize by one ``fp8_linear``.
    A matrix the program takes as input and a matmul reads as its second
    operand, such as a weight in its dgrad, is cast to both layouts at once
    (``_insert_input_matrix_cast``).

    Args:
        graph: The program's graph, rewritten in place.

    Returns:
        The names of the GEMMs replaced.

    Raises:
        NotImplementedError: A GEMM scaled back by a dequantize reads an
            operand that is not cast to FP8.
    """
    gemms = [
        gemm
        for gemm in graph.nodes
        if gemm.target in (LINEAR, MATMUL)
        and any(user.target is DEQUANTIZE for user in gemm.users)
    ]
    quantizes = dict.fromkeys(
        _fp8_quantize(operand) for gemm in gemms for operand in gemm.args
    )
    transposes = {}
    for quantize in quantizes:
        dtype = FP8_DTYPES[quantize.meta["dtype"]]
        input = quantize.args[0]
        transposed = any(
            user.target is MATMUL and user.args[1] is quantize
            for user in quantize.users
        )
        if input.op == "placeholder" and transposed:
            cast, transpose = _insert_input_matrix_cast(graph, quantize, dtype)
            transposes[cast] = transpose
        else:
            cast = _insert_activation_cast(graph, quantize, dtype, transposes)
        quantize.replace_all_uses_with(cast)
        graph.erase_node(quantize)
    for gemm in gemms:
        (output,) = gemm.users
        output.replace_all_uses_with(_insert_fp8_gemm(graph, gemm, transposes))
        graph.erase_node(output)
        graph.erase_node(gemm)
    return [gemm.name for gemm in gemms]


def _lower_nvfp4_gemms(program: GraphModule) -> List[str]:
    """Replace each NVFP4 ``quantize_mx`` with one cast to packed codes, and
    each ``linear_mx`` and ``matmul_mx`` by one ``nvfp4_linear``.

    Args:
        program: The gradient program, rewritten in place.

    Returns:
        The names of the GEMMs replaced.

    Raises:
        NotImplementedError: A GEMM reads an operand that is not NVFP4 codes
            blocked along its contraction axis.
    """
    graph = program.graph
    gemms = [
        gemm for gemm in graph.nodes if gemm.target in (LINEAR_MX, MATMUL_MX)
    ]
    casts = {}
    for gemm in gemms:
        gemm.replace_all_uses_with(_insert_nvfp4_gemm(program, gemm, casts))
        graph.erase_node(gemm)
    return [gemm.name for gemm in gemms]


def lower_training(program: GraphModule) -> Dict[str, str]:
    """Swap each quantized GEMM of a gradient program for a real one, in
    place.

    Every quantized GEMM is a linear's: a forward linear, or a backward
    matmul whose second operand is 2-d.  Each RMSNorm's backward becomes one
    kernel (``fuse_rmsnorm_backward``).  The ops the old casts leave unused,
    such as the global scales' amaxes, are dropped.

    Args:
        program: A program ``gradient_program`` built from a bfloat16 model.

    Returns:
        Per GEMM lowered, by name: ``"fp8"`` or ``"nvfp4"``.

    Raises:
        NotImplementedError: A quantized GEMM reads an operand that is not
            cast to FP8 or NVFP4, or NVFP4 codes it cannot read; ``program``
            is then left partly rewritten.
    """
    graph = program.graph
    pair_quantize_transposes(program)
    lowered = dict.fromkeys(_lower_fp8_gemms(graph), "fp8")
    nvfp4_gemms = _lower_nvfp4_gemms(program)
    lowered.update(dict.fromkeys(nvfp4_gemms, "nvfp4"))
    fuse_rmsnorm_backward(graph)
    graph.eliminate_dead_code()
    graph.lint()
    program.recompile()
    return lowered

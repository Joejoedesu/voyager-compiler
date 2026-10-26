"""Trainium's instruction patterns: the NKI instructions that do the work of
several aten or ``quantized_ops`` ops.

Each is a named ``Pattern``, the ``OpMatcher`` of each op along a chain,
each checking that op's operands.  A Trainium builder fuses the ops it
built by them (``fuse_patterns(gm, INSTRUCTION_PATTERNS)``), so each match
becomes a fused node recording the pattern's name (``meta['pattern']``).
The emitter writes the name as the fused op's ``pattern``, and the
translator emits the group as that instruction
(``translate.PATTERN_HANDLERS``).
"""

from torch.fx import Node

from voyager_compiler import OpMatcher, Pattern
from voyager_compiler.codegen.node_info import (
    get_arg_value,
    is_matmul,
    reads_kmajor,
)

# The names of the patterns, each the NKI instruction form it is.
TENSOR_SCALAR = "tensor_scalar"
TENSOR_SCALAR_REDUCE = "tensor_scalar_reduce"
ACTIVATION_REDUCE = "activation_reduce"
MATMUL_ACCUMULATE = "matmul_accumulate"
SCALED_IDENTITY = "scaled_identity"

# The ops an engine's ALU operator computes, on their operands in order.
ALU_OPS = ("add", "sub", "mul", "maximum")


def _unscaled(node):
    """Whether ``node`` does not scale its second operand (an ``add`` or
    ``sub``'s ``alpha``)."""
    return node.kwargs.get("alpha", 1) == 1


def _by_column(node):
    """Whether ``node`` takes a tile by a column of one value per row of it:
    ``tile <op> column``."""
    tile, column = node.args[:2]
    return (
        _unscaled(node)
        and isinstance(tile, Node)
        and isinstance(column, Node)
        and tuple(column.shape) == (*tile.shape[:-1], 1)
    )


def _by_constant(node):
    """Whether ``node`` takes a tile by a constant: ``tile <op> constant``."""
    return _unscaled(node) and isinstance(node.args[1], (int, float))


def _row_reduction(node):
    """Whether ``node`` reduces each row of its operand: its last dim,
    kept."""
    last = len(node.args[0].shape) - 1
    dims = list(get_arg_value(node, 1, "dim", []))
    return dims in ([-1], [last]) and get_arg_value(node, 2, "keepdim", False)


def _without_bias(node):
    """Whether GEMM ``node`` adds no bias."""
    return is_matmul(node) or get_arg_value(node, 2, "bias") is None


def _kmajor_without_bias(node):
    """Whether GEMM ``node`` reads its activation K-major (``"cx"``) and adds
    no bias."""
    return reads_kmajor(node) and _without_bias(node)


def _of_tiles(node):
    """Whether ``node`` takes two tiles of one shape: ``tile <op> tile``."""
    first, second = node.args[:2]
    return (
        _unscaled(node)
        and isinstance(first, Node)
        and isinstance(second, Node)
        and tuple(first.shape) == tuple(second.shape)
    )


# Each row of the tile the instruction stores, reduced as it stores it.
ROW_MAX_OR_SUM = OpMatcher(
    "amax", "sum", predicate=_row_reduction, keeps_input=True
)

# The patterns, the earlier first where two match one op.
INSTRUCTION_PATTERNS = [
    # VectorE's ``tensor_scalar`` with two operators, e.g. a softmax's bias
    # ``-max(old max, new max)``: ``maximum`` by a column, then ``mul`` by -1.
    Pattern(
        OpMatcher(*ALU_OPS, predicate=_by_column),
        OpMatcher(*ALU_OPS, predicate=_by_constant),
        name=TENSOR_SCALAR,
    ),
    # ``tensor_scalar_reduce``: a tile by a constant, then each row of the
    # result as stored reduced, e.g. scores copied out of PSUM scaled, with
    # their row max.  A result stored in another type is cast first.
    Pattern(
        OpMatcher(*ALU_OPS, predicate=_by_constant),
        OpMatcher("to"),
        ROW_MAX_OR_SUM,
        name=TENSOR_SCALAR_REDUCE,
    ),
    Pattern(
        OpMatcher(*ALU_OPS, predicate=_by_constant),
        ROW_MAX_OR_SUM,
        name=TENSOR_SCALAR_REDUCE,
    ),
    # ScalarE's ``activation`` with a ``reduce_op``: ``exp(tile + column)``
    # and each row of it summed, e.g. a softmax's exponentials with their
    # row sums.
    Pattern(
        OpMatcher("add", predicate=_by_column),
        OpMatcher("exp"),
        OpMatcher("sum", predicate=_row_reduction, keeps_input=True),
        name=ACTIVATION_REDUCE,
    ),
    # ``nc_matmul`` with ``accumulate``: a K-major GEMM's product added to
    # the PSUM partial sum it writes, e.g. a split reduction's later steps,
    # or a mask added onto scores by a matmul through an identity.
    Pattern(
        OpMatcher("linear", "matmul", predicate=_kmajor_without_bias),
        OpMatcher("add", predicate=_of_tiles),
        name=MATMUL_ACCUMULATE,
    ),
    # ``nc_matmul_mx`` with ``accumulate``: an MX GEMM's product added to
    # the PSUM partial sum it writes, a split reduction's later steps.
    Pattern(
        OpMatcher("linear_mx", predicate=_without_bias),
        OpMatcher("add", predicate=_of_tiles),
        name=MATMUL_ACCUMULATE,
    ),
    # An identity scaled by a constant, made by GpSimd's ``affine_select``
    # keeping the diagonal of a ``memset`` of the constant, e.g. the one a
    # mask is added onto scores in PSUM through.
    Pattern(
        OpMatcher("eye"),
        OpMatcher("mul", predicate=_by_constant),
        name=SCALED_IDENTITY,
    ),
]

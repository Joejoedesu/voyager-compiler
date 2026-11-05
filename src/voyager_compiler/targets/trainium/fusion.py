"""Target-owned graph fusion policy; hardware configuration does not derive it."""

import torch

from voyager_compiler import OpMatcher
from voyager_compiler.codegen.node_info import get_arg_value, is_fully_connected, swaps_last_two_dims


def _can_fuse(node):
    # Keep unquantized fully-connected vector execution outside GEMM fusion.
    if hasattr(node, "value") and is_fully_connected(node):
        input_node = node.args[0]
        return input_node.meta.get("dtype") is not None
    return True


def _is_constant_div(node):
    if node.target != torch.ops.aten.div.Tensor:
        return True

    divisor = node.args[1]
    if isinstance(divisor, torch.fx.Node):
        return divisor.value.numel() == 1

    return True


def _has_rms_prologue(node):
    if not _can_fuse(node):
        return False
    activation = node.args[0]
    if swaps_last_two_dims(activation):
        activation = activation.args[0]
    return activation.target == torch.ops.aten.rms_norm.default


def _last_dim_softmax(node):
    dim = get_arg_value(node, 1, 'dim')
    return dim in (-1, node.value.ndim - 1)


MXU_OPS = [
    "conv2d",
    "linear",
    "matmul",
    "conv2d_mx",
    "linear_mx",
    "matmul_mx",
]
QUANT_OPS = [
    "quantize",
    "quantize_mx",
    "quantize_mx_outlier",
    "quantize_affine",
]
# Requantizations a GEMM's fused tail may end in.  ``quantize_mx_outlier`` is
# not one: an epilogue emitting an outlier CSR cuts its slices from the GEMM's
# own column tile, which pins every consumer's reduction tile to it.  The
# outlier quantize runs as its own row-swept nest, or fused onto a whole-row
# op, and its slice width follows the consumers.
GEMM_QUANT_OPS = [op for op in QUANT_OPS if op != "quantize_mx_outlier"]

FUSION_PATTERNS = [
    [
        OpMatcher(*MXU_OPS, predicate=_can_fuse),
        OpMatcher("dequantize"),
        OpMatcher("add", "sub", "mul", "div", predicate=_is_constant_div),
        OpMatcher("exp", "abs", "relu"),
        OpMatcher("add", "mul", "div", predicate=_is_constant_div),
        OpMatcher(*GEMM_QUANT_OPS),
    ],
    [
        OpMatcher(*MXU_OPS, predicate=_can_fuse),
        OpMatcher("dequantize"),
        OpMatcher("gelu", "sigmoid", "silu", "tanh", "hardtanh"),
        OpMatcher(*GEMM_QUANT_OPS),
    ],
    [
        OpMatcher("layer_norm", "rms_norm", "rms_norm_kmajor", "softmax"),
        OpMatcher(*QUANT_OPS),
    ],
]


RMS_PROLOGUE_PATTERN = [
        OpMatcher("add"),
        OpMatcher("rms_norm"),
        OpMatcher("permute", "transpose", predicate=swaps_last_two_dims),
        OpMatcher(*MXU_OPS, predicate=_has_rms_prologue),
]

FUSION_PATTERNS += [
    RMS_PROLOGUE_PATTERN,
    [OpMatcher(*MXU_OPS, predicate=_can_fuse),
     OpMatcher("softmax", predicate=_last_dim_softmax)],
    [OpMatcher("add"), OpMatcher("rms_norm")],
    [
        OpMatcher("add"),
        OpMatcher("rms_norm"),
        OpMatcher(*MXU_OPS, predicate=_can_fuse),
    ],
    [
        OpMatcher(*MXU_OPS, predicate=_can_fuse),
        OpMatcher("add"),
        OpMatcher("rms_norm"),
    ],
]

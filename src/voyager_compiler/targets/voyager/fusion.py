"""Target-owned graph fusion policy; hardware configuration does not derive it."""

import torch

from voyager_compiler import OpMatcher
from voyager_compiler.codegen.node_info import is_fully_connected


def _can_fuse(node):
    # A bf16 FC runs on the vector unit itself, so nothing chains after it.
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

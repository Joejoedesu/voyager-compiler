# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Hardware-pipeline matching for Voyager's generic native target.

This is the source-owned form of the historical ``test_codegen`` pipeline.
Keeping the exact matcher inventory here makes ``generic`` a real named
target and prevents workload drivers from silently imposing its datapath on
every concrete backend.
"""

import torch

from voyager_compiler.codegen.mapping_utils import is_fully_connected
from voyager_compiler.codegen.hardware import OpMatcher


def _is_bf16_fc(node):
    """Return whether a fully connected call runs on the generic vector unit."""

    if hasattr(node, "value") and is_fully_connected(node):
        input_node = node.args[0]
        return input_node.meta.get("dtype") is None
    return False


def _is_spmm(node):
    return node.kwargs.get("A_data") is not None


def _can_fuse(node):
    return not _is_spmm(node) and not _is_bf16_fc(node)


def _is_constant_div(node):
    if node.target != torch.ops.aten.div.Tensor:
        return True

    divisor = node.args[1]
    if isinstance(divisor, torch.fx.Node):
        return divisor.value.numel() == 1
    return True


GENERIC_MATRIX_OPERATIONS = (
    "conv2d",
    "linear",
    "matmul",
    "conv2d_mx",
    "conv2d_vsq",
    "linear_mx",
    "matmul_mx",
)
GENERIC_QUANTIZE_OPERATIONS = (
    "quantize",
    "quantize_mx",
    "quantize_mx_outlier",
    "quantize_vsq",
    "pvs_quantize",
)

# Preserve the historical stage and overload order exactly. Matcher order is
# observable because generic fusion greedily takes the first sequential
# candidate within a stage.
GENERIC_HARDWARE_PIPELINE = (
    (
        OpMatcher(*GENERIC_MATRIX_OPERATIONS, predicate=_can_fuse),
        OpMatcher("dequantize"),
        OpMatcher("add", "sub", "mul", "div", predicate=_is_constant_div),
        OpMatcher("exp", "abs", "relu"),
        OpMatcher("add", "mul", "div", predicate=_is_constant_div),
        OpMatcher(*GENERIC_QUANTIZE_OPERATIONS, "mul", "div"),
    ),
    (
        OpMatcher(*GENERIC_MATRIX_OPERATIONS, predicate=_can_fuse),
        OpMatcher("dequantize"),
        OpMatcher("gelu", "sigmoid", "silu", "tanh", "hardtanh"),
        OpMatcher(*GENERIC_QUANTIZE_OPERATIONS, "mul", "div"),
    ),
    (
        OpMatcher(*GENERIC_MATRIX_OPERATIONS, predicate=_is_spmm),
        OpMatcher("dequantize"),
        OpMatcher("exp", "abs", "relu"),
        OpMatcher("add", "mul", "div"),
        OpMatcher(*GENERIC_QUANTIZE_OPERATIONS),
    ),
    (
        OpMatcher("layer_norm", "softmax"),
        OpMatcher(*GENERIC_QUANTIZE_OPERATIONS),
    ),
)


__all__ = [
    "GENERIC_HARDWARE_PIPELINE",
    "GENERIC_MATRIX_OPERATIONS",
    "GENERIC_QUANTIZE_OPERATIONS",
]

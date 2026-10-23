# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""PyTorch-semantic hardware pipeline modeled for Gemmini.

The matcher inventory describes only linear paths that the pinned public
helpers can realize after target-specific legality checks. It deliberately
omits capture-only leading input quantization, the residual two-input DAG,
and unavailable normalization. The current Gemmini workload adapter keeps
generic fusion disabled while its target-owned grouping remains authoritative.
"""

import torch

from voyager_compiler.codegen.hardware import OpMatcher

_CONV = OpMatcher(
    torch.ops.aten.conv2d.default,
    torch.ops.quantized_ops.conv2d.default,
)
_DEQUANTIZE = OpMatcher(torch.ops.quantized_ops.dequantize.default)
_QUANTIZE = OpMatcher(torch.ops.quantized_ops.quantize.default)
_CODE_RELU = OpMatcher(torch.ops.aten.relu.default)
_MAX_POOL = OpMatcher(torch.ops.aten.max_pool2d.default)
_LINEAR = OpMatcher(torch.ops.aten.linear.default)
_GLOBAL_AVERAGE = OpMatcher(torch.ops.aten.adaptive_avg_pool2d.default)
_FLATTEN = OpMatcher(torch.ops.aten.flatten.using_ints)

_CONV_CORE = (_CONV, _DEQUANTIZE, _QUANTIZE)

GEMMINI_HARDWARE_PIPELINE = (
    (*_CONV_CORE, _CODE_RELU, _MAX_POOL),
    (*_CONV_CORE, _CODE_RELU),
    _CONV_CORE,
    (_LINEAR, _DEQUANTIZE, _QUANTIZE),
    (_GLOBAL_AVERAGE, _FLATTEN, _QUANTIZE),
)

# The public registration boundary includes input quantization. Derive its
# required final-FX grouping from the same Conv stages instead of maintaining
# another operation inventory.
GEMMINI_EXECUTION_FUSION_PATTERNS = (
    tuple(stage.targets for stage in (_QUANTIZE, *_CONV_CORE)),
)


__all__ = [
    "GEMMINI_EXECUTION_FUSION_PATTERNS",
    "GEMMINI_HARDWARE_PIPELINE",
]

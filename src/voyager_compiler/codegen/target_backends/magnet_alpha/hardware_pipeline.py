# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""OpMatcher hardware pipeline for Magnet Alpha target semantics.

Alpha's custom FX operators are already semantic hardware operations, so each
PE/PPU or depthwise/NMP execution path can be expressed directly. The same
inventory is projected into the required final-FX execution groups.
"""

from voyager_compiler.codegen.target_backends.magnet_alpha.operators import (
    INT8_CONV2D_ACCUMULATOR_OPERATOR,
    NMP_FIXED_REQUANTIZE_OPERATOR,
    PPU_FIXED_RELU_OPERATOR,
    PVS_QUANTIZE_OPERATOR,
    SATURATING_ADD_OPERATOR,
    VSQ_CONV2D_OPERATOR,
)
from voyager_compiler.codegen.hardware import OpMatcher

_INT8_ACCUMULATOR = OpMatcher(INT8_CONV2D_ACCUMULATOR_OPERATOR)
_PE_ACCUMULATOR = OpMatcher(
    INT8_CONV2D_ACCUMULATOR_OPERATOR,
    VSQ_CONV2D_OPERATOR,
)
_PPU = OpMatcher(PPU_FIXED_RELU_OPERATOR)
_EADD = OpMatcher(SATURATING_ADD_OPERATOR)
_NMP = OpMatcher(NMP_FIXED_REQUANTIZE_OPERATOR)
_PVS = OpMatcher(PVS_QUANTIZE_OPERATOR)


MAGNET_ALPHA_HARDWARE_PIPELINE = (
    (_INT8_ACCUMULATOR, _PPU, _EADD),
    (_PE_ACCUMULATOR, _PPU, _PVS),
    (_PE_ACCUMULATOR, _PPU),
    (_INT8_ACCUMULATOR, _NMP, _PVS),
    (_INT8_ACCUMULATOR, _NMP),
)


__all__ = ["MAGNET_ALPHA_HARDWARE_PIPELINE"]

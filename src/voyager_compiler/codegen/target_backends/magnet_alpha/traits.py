# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Immutable Magnet Alpha operator, mapping, and transfer traits."""

import torch

from voyager_compiler.codegen.hardware import TensorRole
from voyager_compiler.codegen.target_backends.magnet_alpha.graph import (
    is_channel_shuffle_cat,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.operators import (
    CHANNEL_SHUFFLE_OPERATOR,
    INT8_CONV2D_ACCUMULATOR_OPERATOR,
    NMP_FIXED_REQUANTIZE_OPERATOR,
    PPU_FIXED_RELU_OPERATOR,
    PVS_QUANTIZE_OPERATOR,
    SATURATING_ADD_OPERATOR,
    TILED_CONV2D_OPERATORS,
    VSQ_CONV2D_OPERATOR,
)
from voyager_compiler.codegen.target_backends.spec import TargetOperatorTraits
from voyager_compiler.quantization_contracts import (
    PreparedDomainInputFormat,
    PreparedDomainRelation,
    PreparedDomainTransferRule,
)

EXECUTION_FUSION_PREDECESSOR_INPUTS = {
    SATURATING_ADD_OPERATOR: 0,
}
NON_ALLOCATING_OPERAND_META = (
    "magnet_alpha_instruction_operand",
    "magnet_alpha_semantic_operand",
)
OPERAND_ROLE_OVERRIDES = {
    "accumulator_bias": TensorRole.BIAS,
    "bias_payload": TensorRole.BIAS,
    "code": TensorRole.SCALE,
    "coarse_scale": TensorRole.SCALE,
    "fine_scale": TensorRole.SCALE,
    "folded_coarse_scale": TensorRole.SCALE,
    "input_scale": TensorRole.SCALE,
    "logical_scale": TensorRole.SCALE,
    "primary_multiplier": TensorRole.SCALE,
    "qmap": TensorRole.SCALE,
    "semantic_accumulator_scale": TensorRole.SCALE,
    "secondary_multiplier": TensorRole.SCALE,
    "weight_coarse_scale": TensorRole.SCALE,
    "weight_scale": TensorRole.SCALE,
}
GEMM_OPERATORS = (
    INT8_CONV2D_ACCUMULATOR_OPERATOR,
    VSQ_CONV2D_OPERATOR,
)
CONV2D_OPERATORS = GEMM_OPERATORS
OUTPUT_PADDING_EPILOGUE_OPERATORS = (
    PPU_FIXED_RELU_OPERATOR,
    NMP_FIXED_REQUANTIZE_OPERATOR,
    PVS_QUANTIZE_OPERATOR,
)
# Tuple-valued physical epilogues keep their non-payload metadata intact.
# Software-visible channel slicing is applied only to the payload component.
OUTPUT_PADDING_COMPONENTS = {
    PVS_QUANTIZE_OPERATOR: (2,),
}
CONV2D_INPUT_CHANNEL_DIMS = {operator: -1 for operator in CONV2D_OPERATORS}
PARAMETER_OPERAND_INDEX = {operator: 1 for operator in CONV2D_OPERATORS}
FOLDED_SCALE_OPERATORS = (VSQ_CONV2D_OPERATOR,)
SCALE_BEARING_CONV2D_OPERATORS = (VSQ_CONV2D_OPERATOR,)
CONV2D_TILING_EPILOGUE_OPERATORS = (PPU_FIXED_RELU_OPERATOR,)
QUANTIZATION_OUTPUT_EPILOGUE_OPERATORS = (
    torch.ops.aten.relu.default,
    torch.ops.aten.relu_.default,
    torch.ops.aten.silu.default,
)
PREPARED_DOMAIN_TRANSFER_RULES = (
    *(
        PreparedDomainTransferRule(
            target=target,
            input_format=PreparedDomainInputFormat.PREFIX,
            input_count=2,
            relation=PreparedDomainRelation.INPUT_JOIN,
            region_root=True,
            forcing_root=True,
        )
        for target in (
            torch.ops.aten.add.Tensor,
            torch.ops.aten.add_.Tensor,
        )
    ),
    *(
        PreparedDomainTransferRule(
            target=target,
            input_format=PreparedDomainInputFormat.FIRST,
            relation=PreparedDomainRelation.ALIAS,
        )
        for target in (
            torch.ops.aten.relu.default,
            torch.ops.aten.relu_.default,
        )
    ),
    PreparedDomainTransferRule(
        target=torch.ops.aten.cat.default,
        input_format=PreparedDomainInputFormat.SEQUENCE,
        relation=PreparedDomainRelation.JOIN,
        region_root=True,
        forcing_root=True,
        predicate=is_channel_shuffle_cat,
    ),
    PreparedDomainTransferRule(
        target=CHANNEL_SHUFFLE_OPERATOR,
        input_format=PreparedDomainInputFormat.PREFIX,
        input_count=2,
        relation=PreparedDomainRelation.JOIN,
        region_root=True,
        forcing_root=True,
    ),
)
TRANSPOSED_OPERATORS = {operator: operator for operator in CONV2D_OPERATORS}


MAGNET_ALPHA_OPERATOR_TRAITS = TargetOperatorTraits(
    gemm_operators=GEMM_OPERATORS,
    conv2d_operators=CONV2D_OPERATORS,
    output_padding_epilogue_operators=(OUTPUT_PADDING_EPILOGUE_OPERATORS),
    output_padding_components=OUTPUT_PADDING_COMPONENTS,
    conv2d_input_channel_dims=CONV2D_INPUT_CHANNEL_DIMS,
    prepared_domain_transfer_rules=PREPARED_DOMAIN_TRANSFER_RULES,
    transposed_operators=TRANSPOSED_OPERATORS,
    parameter_operand_index=PARAMETER_OPERAND_INDEX,
    non_allocating_operand_meta=NON_ALLOCATING_OPERAND_META,
    folded_scale_operators=FOLDED_SCALE_OPERATORS,
    scale_bearing_conv2d_operators=(SCALE_BEARING_CONV2D_OPERATORS),
    tiled_conv2d_operators=TILED_CONV2D_OPERATORS,
    conv2d_tiling_epilogue_operators=(CONV2D_TILING_EPILOGUE_OPERATORS),
    quantization_output_epilogue_operators=(
        QUANTIZATION_OUTPUT_EPILOGUE_OPERATORS
    ),
    operand_role_overrides=OPERAND_ROLE_OVERRIDES,
    execution_fusion_predecessor_inputs=(EXECUTION_FUSION_PREDECESSOR_INPUTS),
)


__all__ = [
    "EXECUTION_FUSION_PREDECESSOR_INPUTS",
    "MAGNET_ALPHA_OPERATOR_TRAITS",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Voyager graph-to-scheduler integration.

This adapter is the only scheduler module that understands Voyager FX nodes.
It projects compiler-owned tiled shapes into the backend-neutral workload
consumed by dispatch and every local scheduling engine.
"""

from __future__ import annotations

import math

from scheduler.types import (
    GemmWorkload,
    OperationClassification,
    OperationOperand,
    OperationWorkload,
)


def _unique_operand_shape(
    operands: tuple[OperationOperand, ...],
    *ports: str,
) -> tuple[int, ...] | None:
    """Return the first unambiguous typed operand shape for ``ports``."""
    for port in ports:
        matches = tuple(
            operand.shape
            for operand in operands
            if operand.port == port and operand.shape is not None
        )
        if len(matches) > 1:
            raise ValueError(f"scheduler workload has multiple {port!r} shapes")
        if matches:
            return matches[0]
    return None


def extract_gemm_workload(
    first_node,
    operands,
    output,
    hardware_config=None,
    operator_traits=None,
) -> GemmWorkload | None:
    """Build a canonical workload from one tiled Voyager matrix operation.

    ``None`` means that local GEMM scheduling does not apply to this node.
    Compiler imports stay deferred so the standalone scheduler engines remain
    importable without initializing Voyager.
    """
    from voyager_compiler.codegen.mapping_utils import (
        FIRST_CONV_INPUT_CHANNELS,
        is_conv2d,
        is_depthwise_conv,
        is_matmul,
    )
    from voyager_compiler.codegen.passes.tiling import _conv2d_layout
    from voyager_compiler.codegen.passes.utils import _pair, get_arg_value

    operands = tuple(operands)
    if not isinstance(output, OperationOperand) or output.port != "output":
        raise TypeError("GEMM extraction requires a typed output operand")
    if is_depthwise_conv(first_node, operator_traits):
        return None

    input_shape = _unique_operand_shape(operands, "input")
    if input_shape is None:
        return None

    # Fully connected layers have no spatial tile to search.
    if math.prod(input_shape[:-1]) == 1:
        return None

    if is_conv2d(first_node, operator_traits):
        weight_shape = _unique_operand_shape(operands, "weight")
        if weight_shape is None or len(weight_shape) != 4:
            return None

        transposed = first_node.meta.get("transposed", False)
        (
            filter_height,
            filter_width,
            input_channels,
            output_channels,
        ) = _conv2d_layout(weight_shape, True, not transposed)
        _, height, width, _ = _conv2d_layout(
            output.shape,
            False,
            not transposed,
        )

        supports_partial_input = (
            hardware_config is not None
            and hardware_config.supports_partial_matrix_input(input_channels)
        )
        if (
            input_channels in FIRST_CONV_INPUT_CHANNELS
            and not supports_partial_input
        ):
            return None

        stride_height, stride_width = _pair(
            get_arg_value(first_node, 3, "stride", 1)
        )
    else:
        # Linear and matmul become a 1x1 filter over flattened positions.
        weight_shape = _unique_operand_shape(
            operands,
            "weight",
            "other",
        )
        if weight_shape is None or len(weight_shape) != 2:
            return None

        weight_transposed = is_matmul(first_node) ^ first_node.meta.get(
            "transposed", False
        )
        if weight_transposed:
            input_channels, output_channels = weight_shape
        else:
            output_channels, input_channels = weight_shape

        filter_height = filter_width = 1
        height = 1
        width = math.prod(output.shape[:-1])
        stride_height = stride_width = 1

    return GemmWorkload(
        nifm=input_channels,
        nofm=output_channels,
        wofm=width,
        hofm=height,
        wfil=filter_width,
        hfil=filter_height,
        wstd=stride_width,
        hstd=stride_height,
    )


def extract_operation_workload(
    operator,
    operands,
    output,
    hardware_config=None,
    operator_traits=None,
    *,
    operation: OperationClassification | None = None,
    has_high_precision_vector_input=False,
    has_sparse_op=False,
) -> OperationWorkload:
    """Freeze compiler-resolved facts for one scheduling decision.

    Target classification and operand discovery are deliberately inputs.  This
    adapter may project an FX operation into a canonical GEMM, but it never
    imports a target backend or reconstructs semantic operands from parallel
    shape/precision dictionaries.
    """
    operands = tuple(operands)

    return OperationWorkload(
        operands=operands,
        output=output,
        operation=operation,
        gemm=extract_gemm_workload(
            operator,
            operands,
            output,
            hardware_config,
            operator_traits,
        ),
        has_high_precision_vector_input=(has_high_precision_vector_input),
        has_sparse_op=has_sparse_op,
    )


__all__ = [
    "extract_gemm_workload",
    "extract_operation_workload",
]

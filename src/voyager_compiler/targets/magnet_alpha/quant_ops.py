# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Exact Alpha PTQ operators, with schemas distinct from generic MX ops."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.library import Library, impl

from voyager_compiler.quantization.mx_utils import (
    _reshape_to_blocks,
    _shared_exponents,
)

quantized_decomposed_lib = Library("alpha_quant", "DEF")


def expand(input, shape, block_size):
    while input.ndim < len(shape):
        input = input.unsqueeze(0)

    # Repeat the input along each dimension to match the target shape
    for dim in range(len(shape)):
        if input.shape[dim] != shape[dim]:
            input = torch.repeat_interleave(input, block_size, dim)

    # If the shape is not a multiple of block_size, we may need to slice
    if list(input.shape) != list(shape):
        slices = [slice(0, x) for x in shape]
        input = input[tuple(slices)]
    return input


quantized_decomposed_lib.define("vmap(Tensor self, Tensor other) -> Tensor")


@impl(quantized_decomposed_lib, "vmap", "CompositeExplicitAutograd")
def vmap(
    input: torch.Tensor, qmap: torch.Tensor, chunk_size=1024 * 1024
) -> torch.Tensor:
    input_dtype = input.dtype

    if input.dtype != torch.bfloat16:
        input = input.to(torch.bfloat16)

    indices = input.view(torch.int16)

    output = torch.empty(
        input.shape,
        dtype=qmap.dtype,
        device=input.device,
        memory_format=torch.contiguous_format,
    )
    indices_flat = indices.reshape(-1)
    output_flat = output.view(-1)

    for start in range(0, indices_flat.numel(), chunk_size):
        end = min(start + chunk_size, indices_flat.numel())
        indices_chunk = indices_flat[start:end].to(torch.int32) & 0xFFFF
        output_flat[start:end] = qmap[indices_chunk]

    return output.to(input_dtype)


quantized_decomposed_lib.define(
    "quantize(Tensor input, Tensor scale, Tensor? zero_point=None, SymInt[]? axes=None, "
    "int? block_size=None, Tensor? qmap=None, Tensor? output_code=None, "
    "str rounding='nearest', float? quant_min=None, "
    "float? quant_max=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize", "CompositeExplicitAutograd")
def quantize(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    axes: Optional[Tuple[int]] = None,
    block_size: Optional[int] = None,
    qmap: torch.Tensor = None,
    output_code: Optional[torch.Tensor] = None,
    rounding: str = "nearest",
    quant_min: Optional[float] = None,
    quant_max: Optional[float] = None,
) -> torch.Tensor:
    """Quantization for the Tensor using scales and zero points to map
    from floating point to quantized values

    Args:
        input (torch.Tensor): original float32 or bfloat16 Tensor
        scale (torch.Tensor): scale factors for quantization
        zero_point (torch.Tensor): zero point for quantization, default is None
        axes (Tuple[int]): axes for group-wise quantization, default is None
        block_size (int): block size for group-wise quantization, default is None
        qmap (torch.Tensor): quantization map for mapping from float to quantized values
        output_code (torch.Tensor): codebook for quantizing the output

    Returns:
        Tensor with requested dtype (e.g. int8), note the quantization parameters
        are not stored in the Tensor, we are storing them in function arguments instead
    """
    if rounding not in ("nearest", "floor"):
        raise ValueError("quantization rounding must be nearest or floor")
    if rounding == "nearest":
        assert qmap is not None, "qmap must be provided for quantization"
    elif quant_min is None or quant_max is None:
        raise ValueError("floor quantization requires quantization bounds")

    if block_size is not None:
        scale = expand(scale, input.shape, block_size)
        if zero_point is not None:
            zero_point = expand(zero_point, input.shape, block_size)

    if zero_point is None:
        input = input / scale
    else:
        input = input / scale + zero_point

    if rounding == "floor":
        return torch.clamp(torch.floor(input), quant_min, quant_max)
    return vmap(input, qmap)


quantized_decomposed_lib.define(
    "dequantize(Tensor input, Tensor scale, Tensor? zero_point=None, SymInt[]? axes=None, "
    "int? block_size=None, Tensor? input_qmap=None, Tensor? output_qmap=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "dequantize", "CompositeExplicitAutograd")
def dequantize(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
    axes: Optional[Tuple[int]] = None,
    block_size: Optional[int] = None,
    input_qmap: Optional[torch.Tensor] = None,
    output_qmap: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Dequantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values

    Args:
        input (torch.Tensor): original float32 or bfloat16 Tensor
        scale (torch.Tensor): scale factors for dequantization
        zero_point (torch.Tensor): zero point for quantization, default is None
        axes (Tuple[int]): axes for group-wise quantization, default is None
        block_size (int): block size for group-wise quantization, default is None
        input_qmap (torch.Tensor): quantization map used to quantize the input
        output_qmap (torch.Tensor): quantization map used to quantize the output

    Returns:
        Tensor with floating point types, note the quantization parameters
        are not stored in the Tensor, we are storing them in function arguments instead
    """

    if input_qmap is not None:
        input = vmap(input, input_qmap)

    if block_size is not None:
        scale = expand(scale, input.shape, block_size)
        if zero_point is not None:
            zero_point = expand(zero_point, input.shape, block_size)

    if zero_point is None:
        dequantized = input * scale
    else:
        dequantized = (input - zero_point) * scale

    if output_qmap is not None:
        dequantized = vmap(dequantized, output_qmap)

    return dequantized


quantized_decomposed_lib.define(
    "dequantize_vsq(Tensor input, Tensor fine_scale, "
    "Tensor coarse_scale, SymInt[] axes, int block_size, "
    "Tensor? input_qmap=None) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "dequantize_vsq",
    "CompositeExplicitAutograd",
)
def dequantize_vsq(
    input: torch.Tensor,
    fine_scale: torch.Tensor,
    coarse_scale: torch.Tensor,
    axes: Tuple[int],
    block_size: int,
    input_qmap: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reconstruct a VSQ value without collapsing its two scale levels."""
    if input_qmap is not None:
        input = vmap(input, input_qmap)
    fine_scale = expand(fine_scale, input.shape, block_size)
    return input * fine_scale * coarse_scale


quantized_decomposed_lib.define(
    "conv2d_mx(Tensor input, Tensor weight, Tensor? bias=None, "
    "SymInt[2] stride=1, "
    "SymInt[2] padding=0, SymInt[2] dilation=1, SymInt groups=1, *, "
    "Tensor? input_scale=None, Tensor? input_coarse_scale=None, "
    "Tensor? weight_scale=None, Tensor? weight_coarse_scale=None, "
    "int? block_size=None, Tensor? input_code=None, "
    "Tensor? weight_code=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "conv2d_mx", "CompositeExplicitAutograd")
def conv2d_mx(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1,
    *,
    input_scale: Optional[torch.Tensor] = None,
    input_coarse_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    weight_coarse_scale: Optional[torch.Tensor] = None,
    block_size: Optional[int] = None,
    input_code: Optional[torch.Tensor] = None,
    weight_code: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # For codebook quantization, decode input and weight into float values first
    if input_code is not None:
        input = input_code[input.to(torch.long)].to(input.dtype)
    if weight_code is not None:
        weight = weight_code[weight.to(torch.long)].to(weight.dtype)

    # Replicate scales to match input and weight shapes
    if input_scale is not None:
        input = input * expand(input_scale, input.shape, block_size)
    if input_coarse_scale is not None:
        input = input * input_coarse_scale
    if weight_scale is not None:
        weight = weight * expand(weight_scale, weight.shape, block_size)
    if weight_coarse_scale is not None:
        weight = weight * weight_coarse_scale

    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


quantized_decomposed_lib.define(
    "calculate_mx_qparam(Tensor self, SymInt[] axes, int block_size, float quant_max, "
    "bool force_scale_power_of_two=False, Tensor scale_qmap=None) -> Tensor"
)


@impl(
    quantized_decomposed_lib, "calculate_mx_qparam", "CompositeExplicitAutograd"
)
def calculate_mx_qparam(
    input: torch.Tensor,
    axes: Union[int, List[int]],
    block_size: int,
    quant_max: float,
    force_scale_power_of_two: bool = False,
    scale_qmap: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert block_size > 0

    # Make sure axes is a list of non-negative numbers
    axes = [axes] if type(axes) == int else axes
    axes = [x + input.ndim if x < 0 else x for x in axes]

    # Perform tiling to the hardware vector size
    input, axes, orig_shape, padded_shape = _reshape_to_blocks(
        input, axes, block_size
    )

    shared_exp_axes = [x + 1 for x in axes]

    if force_scale_power_of_two:
        # Get shared exponents
        shared_exp = _shared_exponents(
            input,
            method="max",
            axes=shared_exp_axes,
            ebits=0,
        )

        # Offset the max exponent by the largest representable exponent
        # in the element data format
        shared_exp = shared_exp - math.floor(math.log2(quant_max))

        for axis in reversed(axes):
            # Remove extra dimension
            shared_exp = torch.squeeze(shared_exp, dim=axis + 1)

        scale = 2**shared_exp
    else:
        # Use absolute maximum value to compute scaling factors
        amax = torch.amax(torch.abs(input), dim=shared_exp_axes)
        scale = amax / quant_max

        # Quantize the scale using the codebook
        if scale_qmap is not None:
            scale = vmap(scale, scale_qmap)

    scale = torch.where(scale > 0.0, scale, 1.0)
    return scale


quantized_decomposed_lib.define(
    "quantize_mx(Tensor self, Tensor qmap, SymInt[] axes, int block_size, float quant_max, "
    "bool force_scale_power_of_two=False, Tensor scale_qmap=None, Tensor output_code=None) -> (Tensor, Tensor)"
)


@impl(quantized_decomposed_lib, "quantize_mx", "CompositeExplicitAutograd")
def quantize_mx(
    input: torch.Tensor,
    qmap: torch.Tensor,
    axes: Tuple[int],
    block_size: int,
    quant_max: float,
    force_scale_power_of_two: bool = False,
    scale_qmap: Optional[torch.Tensor] = None,
    output_code: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor]:
    scale = calculate_mx_qparam(
        input,
        axes=axes,
        block_size=block_size,
        quant_max=quant_max,
        force_scale_power_of_two=force_scale_power_of_two,
        scale_qmap=scale_qmap,
    )
    input = quantize(input, scale, None, axes, block_size, qmap)
    return scale, input


quantized_decomposed_lib.define(
    "quantize_vsq(Tensor self, Tensor coarse_scale, Tensor coarse_qmap, "
    "Tensor fine_qmap, SymInt[] axes, int block_size, "
    "float fine_quant_max, bool force_scale_power_of_two=False, "
    "Tensor? fine_scale_qmap=None, Tensor? fine_output_code=None, "
    "float? coarse_scale_quant_max=None) "
    "-> (Tensor, Tensor, Tensor)"
)


@impl(quantized_decomposed_lib, "quantize_vsq", "CompositeExplicitAutograd")
def quantize_vsq(
    input: torch.Tensor,
    coarse_scale: torch.Tensor,
    coarse_qmap: torch.Tensor,
    fine_qmap: torch.Tensor,
    axes: Tuple[int],
    block_size: int,
    fine_quant_max: float,
    force_scale_power_of_two: bool = False,
    fine_scale_qmap: Optional[torch.Tensor] = None,
    fine_output_code: Optional[torch.Tensor] = None,
    coarse_scale_quant_max: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize into one VSQ value with coarse and fine scale outputs."""
    if coarse_scale_quant_max is not None and (
        not math.isfinite(coarse_scale_quant_max) or coarse_scale_quant_max <= 0
    ):
        raise ValueError("coarse_scale_quant_max must be finite and positive")
    coarse_data = quantize(input, coarse_scale, qmap=coarse_qmap)
    fine_scale, fine_data = quantize_mx(
        coarse_data,
        fine_qmap,
        axes,
        block_size,
        fine_quant_max,
        force_scale_power_of_two,
        fine_scale_qmap,
        fine_output_code,
    )
    return coarse_scale, fine_scale, fine_data

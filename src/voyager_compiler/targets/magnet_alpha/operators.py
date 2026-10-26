# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Magnet Alpha custom-operator registration and executable semantics."""

from collections.abc import Mapping
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.library import Library, impl

from voyager_compiler.quantization.fixed_point import (
    quantize_fixed_point,
    reconstruct_fixed_point,
)
from voyager_compiler.targets.magnet_alpha.fixed_point import (
    ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    ALPHA_VSQ_MANTISSA_MAX,
    alpha_int8_conv2d_accumulator,
    alpha_nmp_epilogue,
    alpha_pe_epilogue,
    alpha_pe_postprocess,
    alpha_pvs_quantize,
    alpha_q8_8_format,
    alpha_truncate_output,
    alpha_vsq_conv2d_accumulator,
    alpha_vsq_rebalance,
    encode_alpha_vsq_mantissa,
    encode_alpha_vsq_scales,
)
from voyager_compiler.targets.magnet_alpha.fx import OperationClassification
from voyager_compiler.targets.magnet_alpha.hardware import (
    ALPHA_VSQ_BLOCK_VALUES,
)
from voyager_compiler.targets.magnet_alpha.quant_ops import expand

OPERATION_CLASSIFICATION_META = "operation_classification"
OPERATION_COMPUTE_UNIT_META = "operation_compute_unit"
_TILED_VSQ_SEMANTIC_KWARGS = frozenset(
    {
        "input_fine_quant_max",
        "input_coarse_scale_quant_max",
        "weight_fine_quant_max",
        "weight_coarse_scale_quant_max",
    }
)


MAGNET_ALPHA_LIBRARY = Library("magnet_alpha", "DEF")

MAGNET_ALPHA_LIBRARY.define(
    "channel_shuffle(Tensor input1, Tensor input2, int groups) -> Tensor"
)
MAGNET_ALPHA_LIBRARY.define("channel_shuffle_copy(Tensor input) -> Tensor")
MAGNET_ALPHA_LIBRARY.define(
    "channel_shuffle_buffered(Tensor input1, Tensor input2, Tensor(a!) secondary, int groups) -> Tensor"
)
MAGNET_ALPHA_LIBRARY.define(
    "channel_shuffle_copy_buffered(Tensor input, Tensor secondary) -> Tensor"
)
MAGNET_ALPHA_LIBRARY.define(
    "upsample_nearest2d(Tensor input, SymInt[]? output_size, "
    "float[]? scale_factors) -> Tensor"
)
MAGNET_ALPHA_LIBRARY.define(
    "saturating_add(Tensor input1, Tensor(a!) input2) -> Tensor(a!)"
)


@impl(MAGNET_ALPHA_LIBRARY, "channel_shuffle", "CompositeImplicitAutograd")
def magnet_alpha_channel_shuffle(
    input1: torch.Tensor,
    input2: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    """Concatenate two branches and shuffle their channel groups."""
    output = torch.cat((input1, input2), dim=1)
    batch, channels, height, width = output.shape
    if groups <= 0 or channels % groups != 0:
        raise ValueError("Channel count must be divisible by positive groups")

    channels_per_group = channels // groups
    output = output.view(
        batch,
        groups,
        channels_per_group,
        height,
        width,
    )
    output = output.transpose(1, 2).contiguous()
    return output.view(batch, channels, height, width)


@impl(MAGNET_ALPHA_LIBRARY, "channel_shuffle_copy", "CompositeExplicitAutograd")
def magnet_alpha_channel_shuffle_copy(input: torch.Tensor) -> torch.Tensor:
    """Execute one explicit physical channel-shuffle copy."""
    return input.clone(memory_format=torch.preserve_format)


@torch.library.register_fake("magnet_alpha::channel_shuffle_copy")
def magnet_alpha_channel_shuffle_copy_fake(input: torch.Tensor) -> torch.Tensor:
    """Preserve the copied tensor's shape, dtype, device, and layout."""
    return torch.empty_like(input, memory_format=torch.preserve_format)


@impl(
    MAGNET_ALPHA_LIBRARY,
    "channel_shuffle_buffered",
    "CompositeImplicitAutograd",
)
def channel_shuffle_buffered(input1, input2, secondary, groups):
    """Logical shuffle semantics with its explicit physical secondary backing.

    As in the Alpha reference, tensors contain decoded logical values; the
    instruction's target_layout describes the primary/secondary bank encoding.
    Physical copy calls preserve that logical value while advancing the layout.
    """
    value = magnet_alpha_channel_shuffle(input1, input2, groups)
    secondary.copy_(value)
    return value


@impl(
    MAGNET_ALPHA_LIBRARY,
    "channel_shuffle_copy_buffered",
    "CompositeImplicitAutograd",
)
def channel_shuffle_copy_buffered(input, secondary):
    return input.clone(memory_format=torch.preserve_format)


@impl(MAGNET_ALPHA_LIBRARY, "upsample_nearest2d", "CompositeExplicitAutograd")
def magnet_alpha_upsample_nearest2d(
    input: torch.Tensor,
    output_size: Optional[list[int]],
    scale_factors: Optional[list[float]],
) -> torch.Tensor:
    """Replicate a physical INT8 payload without changing its bits.

    PyTorch's CPU nearest-neighbor kernel does not accept signed INT8.
    Alpha's NMP operation is pure replication, so the unsigned view is an
    exact executable reference for the same stored bytes.
    """
    if input.dtype != torch.int8:
        return torch.ops.aten.upsample_nearest2d.vec(
            input,
            output_size,
            scale_factors,
        )
    unsigned = input.to(torch.uint8)
    output = torch.ops.aten.upsample_nearest2d.vec(
        unsigned,
        output_size,
        scale_factors,
    )
    return output.to(torch.int8)


@torch.library.register_fake("magnet_alpha::upsample_nearest2d")
def magnet_alpha_upsample_nearest2d_fake(
    input: torch.Tensor,
    output_size: Optional[list[int]],
    scale_factors: Optional[list[float]],
) -> torch.Tensor:
    return magnet_alpha_upsample_nearest2d(input, output_size, scale_factors)


def _validate_saturating_add_inputs(input1, input2) -> None:
    supported_dtypes = (
        torch.int8,
        torch.float16,
        torch.float32,
        torch.float64,
    )
    if input1.dtype != input2.dtype or input1.dtype not in supported_dtypes:
        raise TypeError(
            "Alpha EADD requires matching signed-INT8 payload containers"
        )
    if input1.shape != input2.shape:
        raise ValueError("Alpha EADD inputs must have identical shapes")
    if input1.device != input2.device:
        raise ValueError("Alpha EADD inputs must be on the same device")


@impl(MAGNET_ALPHA_LIBRARY, "saturating_add", "CompositeExplicitAutograd")
def magnet_alpha_saturating_add(
    input1: torch.Tensor,
    input2: torch.Tensor,
) -> torch.Tensor:
    """Execute Alpha NMP EADD's lane-wise signed-INT8 saturation."""
    _validate_saturating_add_inputs(input1, input2)
    if input1.is_floating_point():
        for operand in (input1, input2):
            if not torch.isfinite(operand).all():
                raise ValueError("Alpha EADD payload codes must be finite")
            if not torch.equal(operand, operand.trunc()):
                raise ValueError("Alpha EADD payload codes must be integral")
            if (operand < -128).any() or (operand > 127).any():
                raise ValueError(
                    "Alpha EADD payload codes must fit signed INT8"
                )
    widened = input1.to(torch.int16) + input2.to(torch.int16)
    result = widened.clamp(
        torch.iinfo(torch.int8).min,
        torch.iinfo(torch.int8).max,
    ).to(input1.dtype)
    input2.copy_(result)
    return input2


@torch.library.register_fake("magnet_alpha::saturating_add")
def magnet_alpha_saturating_add_fake(
    input1: torch.Tensor,
    input2: torch.Tensor,
) -> torch.Tensor:
    """Return the validated mutable Alpha EADD destination."""
    _validate_saturating_add_inputs(input1, input2)
    return input2


MAGNET_ALPHA_LIBRARY.define(
    "conv2d_vsq(Tensor input, Tensor weight, Tensor? bias=None, "
    "SymInt[2] stride=1, SymInt[2] padding=0, "
    "SymInt[2] dilation=1, SymInt groups=1, *, "
    "Tensor input_scale, Tensor weight_scale, "
    "Tensor folded_coarse_scale, int block_size, "
    "Tensor? semantic_input_coarse_scale=None, "
    "Tensor? semantic_weight_coarse_scale=None, "
    "float input_fine_quant_max=7.0, "
    "float input_coarse_scale_quant_max=127.0, "
    "float weight_fine_quant_max=7.0, "
    "float weight_coarse_scale_quant_max=127.0) -> Tensor"
)

MAGNET_ALPHA_LIBRARY.define(
    "conv2d_int8_accumulator(Tensor input, Tensor weight, "
    "Tensor? bias=None, "
    "SymInt[2] stride=1, SymInt[2] padding=0, "
    "SymInt[2] dilation=1, SymInt groups=1) -> Tensor"
)

MAGNET_ALPHA_LIBRARY.define(
    "ppu_fixed_relu(Tensor input, Tensor primary_multiplier, "
    "Tensor? bias_payload, *, Tensor? accumulator_bias=None, "
    "float output_scale=1.0, bool integer_accumulator=False, "
    "float semantic_step, str rounding, "
    "bool saturate, int primary_shift, int secondary_multiplier, "
    "int secondary_shift, int output_precision, bool relu=True) -> Tensor"
)

MAGNET_ALPHA_LIBRARY.define(
    "nmp_fixed_requantize(Tensor accumulator, Tensor? accumulator_bias, "
    "Tensor semantic_accumulator_scale, *, "
    "int scale_factor, int binary_point, int bias_factor, "
    "float output_scale, int output_precision, bool relu) -> Tensor"
)

MAGNET_ALPHA_LIBRARY.define(
    "pvs_quantize(Tensor input, Tensor coarse_scale, *, "
    "int output_precision, int block_size) "
    "-> (Tensor, Tensor, Tensor)"
)

MAGNET_ALPHA_LIBRARY.define(
    "vsq_scale_encode(Tensor logical_scale, float fine_quant_max=7.0, "
    "float coarse_scale_quant_max=127.0) -> Tensor"
)


@impl(MAGNET_ALPHA_LIBRARY, "vsq_scale_encode", "CompositeExplicitAutograd")
def magnet_alpha_vsq_scale_encode(
    logical_scale: torch.Tensor,
    fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    coarse_scale_quant_max: float = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
) -> torch.Tensor:
    """Materialize the exact Alpha uint8 mantissa consumed by PE."""
    rebalance = alpha_vsq_rebalance(
        fine_quant_max,
        coarse_scale_quant_max,
        ALPHA_VSQ_MANTISSA_MAX,
    )
    return encode_alpha_vsq_mantissa(
        logical_scale,
        rebalance=rebalance,
        mantissa_max=ALPHA_VSQ_MANTISSA_MAX,
    )


@torch.library.register_fake("magnet_alpha::vsq_scale_encode")
def magnet_alpha_vsq_scale_encode_fake(
    logical_scale: torch.Tensor,
    fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    coarse_scale_quant_max: float = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
) -> torch.Tensor:
    if logical_scale.numel() == 0:
        raise ValueError("Alpha VSQ scale must be nonempty")
    return torch.empty_like(logical_scale, dtype=torch.uint8)


@impl(
    MAGNET_ALPHA_LIBRARY,
    "conv2d_int8_accumulator",
    "CompositeExplicitAutograd",
)
def magnet_alpha_conv2d_int8_accumulator(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """Execute the exact bias-free Alpha signed-int8 accumulator."""
    if bias is not None:
        raise ValueError("Alpha int8 accumulator bias belongs in the epilogue")
    return alpha_int8_conv2d_accumulator(
        input,
        weight,
        stride,
        padding,
        dilation,
        groups,
    )


@torch.library.register_fake("magnet_alpha::conv2d_int8_accumulator")
def magnet_alpha_conv2d_int8_accumulator_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """Infer the accumulator shape without executing integer arithmetic."""
    if bias is not None:
        raise ValueError("Alpha int8 accumulator bias belongs in the epilogue")
    output = F.conv2d(
        input.to(torch.float32),
        weight.to(torch.float32),
        None,
        stride,
        padding,
        dilation,
        groups,
    )
    return torch.empty_like(output, dtype=torch.int64)


@impl(MAGNET_ALPHA_LIBRARY, "ppu_fixed_relu", "CompositeExplicitAutograd")
def magnet_alpha_ppu_fixed_relu(
    input: torch.Tensor,
    primary_multiplier: torch.Tensor,
    bias_payload: Optional[torch.Tensor],
    *,
    accumulator_bias: Optional[torch.Tensor] = None,
    output_scale: float = 1.0,
    integer_accumulator: bool = False,
    semantic_step: float,
    rounding: str,
    saturate: bool,
    primary_shift: int,
    secondary_multiplier: int,
    secondary_shift: int,
    output_precision: int,
    relu: bool = True,
) -> torch.Tensor:
    """Execute Alpha PE arithmetic or its unlowered semantic reference."""
    if output_precision not in (4, 8):
        raise ValueError("Alpha PE output precision must be 4 or 8")
    if integer_accumulator:
        del accumulator_bias, output_scale, semantic_step, rounding, saturate
        channels = input.shape[1] if input.ndim == 4 else input.shape[-1]

        def channel_operand(
            value: Optional[torch.Tensor],
            name: str,
        ) -> torch.Tensor:
            if value is None:
                return torch.zeros((), dtype=torch.int64, device=input.device)
            flattened = value.reshape(-1)
            if flattened.numel() == 1:
                return flattened[0]
            if flattened.numel() != channels:
                raise ValueError(
                    f"Alpha PE {name} must be scalar or per-channel"
                )
            shape = [1] * input.ndim
            shape[1 if input.ndim == 4 else -1] = channels
            return flattened.reshape(shape)

        multiplier = channel_operand(
            primary_multiplier,
            "primary multiplier",
        )
        bias = channel_operand(bias_payload, "bias")
        arguments = {
            "accumulator": input,
            "primary_multiplier": multiplier,
            "primary_shift": primary_shift,
            "bias": bias,
            "secondary_multiplier": secondary_multiplier,
            "secondary_shift": secondary_shift,
            "relu": relu,
        }
        if output_precision == 4:
            return alpha_pe_postprocess(**arguments)
        return alpha_pe_epilogue(
            **arguments,
            output_precision=output_precision,
        )

    del (
        primary_multiplier,
        bias_payload,
        accumulator_bias,
        output_scale,
        primary_shift,
        secondary_multiplier,
        secondary_shift,
    )
    fixed_format = alpha_q8_8_format(
        float(semantic_step) * 256.0,
        rounding=rounding,
        saturate=saturate,
    )
    payload = quantize_fixed_point(input, fixed_format)
    reconstructed = reconstruct_fixed_point(
        payload,
        fixed_format,
        output_dtype=input.dtype,
    )
    return torch.relu(reconstructed) if relu else reconstructed


@torch.library.register_fake("magnet_alpha::ppu_fixed_relu")
def magnet_alpha_ppu_fixed_relu_fake(
    input: torch.Tensor,
    primary_multiplier: torch.Tensor,
    bias_payload: Optional[torch.Tensor],
    *,
    accumulator_bias: Optional[torch.Tensor] = None,
    output_scale: float = 1.0,
    integer_accumulator: bool = False,
    semantic_step: float,
    rounding: str,
    saturate: bool,
    primary_shift: int,
    secondary_multiplier: int,
    secondary_shift: int,
    output_precision: int,
    relu: bool = True,
) -> torch.Tensor:
    return magnet_alpha_ppu_fixed_relu(
        input,
        primary_multiplier,
        bias_payload,
        accumulator_bias=accumulator_bias,
        output_scale=output_scale,
        integer_accumulator=integer_accumulator,
        semantic_step=semantic_step,
        rounding=rounding,
        saturate=saturate,
        primary_shift=primary_shift,
        secondary_multiplier=secondary_multiplier,
        secondary_shift=secondary_shift,
        output_precision=output_precision,
        relu=relu,
    )


@impl(MAGNET_ALPHA_LIBRARY, "nmp_fixed_requantize", "CompositeExplicitAutograd")
def magnet_alpha_nmp_fixed_requantize(
    accumulator: torch.Tensor,
    accumulator_bias: Optional[torch.Tensor],
    semantic_accumulator_scale: torch.Tensor,
    *,
    scale_factor: int,
    binary_point: int,
    bias_factor: int,
    output_scale: float,
    output_precision: int,
    relu: bool,
) -> torch.Tensor:
    """Execute Alpha NMP scaling and scalar bias on a raw accumulator."""
    del accumulator_bias, semantic_accumulator_scale, output_scale
    if output_precision not in (4, 8):
        raise ValueError("Alpha NMP output precision must be 4 or 8")
    output = alpha_nmp_epilogue(
        accumulator,
        scale_factor=scale_factor,
        binary_point=binary_point,
        bias_factor=bias_factor,
        relu=relu,
    )
    if output_precision == 8:
        return alpha_truncate_output(output, output_precision)
    return output


@torch.library.register_fake("magnet_alpha::nmp_fixed_requantize")
def magnet_alpha_nmp_fixed_requantize_fake(
    accumulator: torch.Tensor,
    accumulator_bias: Optional[torch.Tensor],
    semantic_accumulator_scale: torch.Tensor,
    *,
    scale_factor: int,
    binary_point: int,
    bias_factor: int,
    output_scale: float,
    output_precision: int,
    relu: bool,
) -> torch.Tensor:
    return magnet_alpha_nmp_fixed_requantize(
        accumulator,
        accumulator_bias,
        semantic_accumulator_scale,
        scale_factor=scale_factor,
        binary_point=binary_point,
        bias_factor=bias_factor,
        output_scale=output_scale,
        output_precision=output_precision,
        relu=relu,
    )


@impl(MAGNET_ALPHA_LIBRARY, "pvs_quantize", "CompositeExplicitAutograd")
def magnet_alpha_pvs_quantize(
    input: torch.Tensor,
    coarse_scale: torch.Tensor,
    *,
    output_precision: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode one integer pre-PVS NCHW Alpha output."""
    if input.ndim != 4:
        raise ValueError("Alpha PVS requires one NCHW tensor")
    if block_size != ALPHA_VSQ_BLOCK_VALUES:
        raise ValueError("Alpha PVS requires 32-channel blocks")
    scale = coarse_scale.reshape(-1)
    if (
        scale.numel() != 1
        or not torch.isfinite(scale).all()
        or torch.any(scale <= 0)
    ):
        raise ValueError("Alpha PVS requires one positive finite coarse scale")

    encoded = alpha_pvs_quantize(
        input,
        output_precision=output_precision,
        vector_dim=1,
    )
    return coarse_scale, encoded.mantissa, encoded.payload


@torch.library.register_fake("magnet_alpha::pvs_quantize")
def magnet_alpha_pvs_quantize_fake(
    input: torch.Tensor,
    coarse_scale: torch.Tensor,
    *,
    output_precision: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target component shapes without reading FakeTensor data."""
    if input.ndim != 4:
        raise ValueError("Alpha PVS requires one NCHW tensor")
    if block_size != ALPHA_VSQ_BLOCK_VALUES:
        raise ValueError("Alpha PVS requires 32-channel blocks")
    if output_precision not in (4, 8):
        raise ValueError("Alpha PVS output precision must be 4 or 8")
    if coarse_scale.numel() != 1:
        raise ValueError("Alpha PVS requires one coarse scale")

    batch, channels, height, width = input.shape
    mantissa = torch.empty(
        (batch, (channels + block_size - 1) // block_size, height, width),
        dtype=torch.uint8,
        device=input.device,
    )
    payload = torch.empty_like(input, dtype=torch.int8)
    return coarse_scale, mantissa, payload


@impl(MAGNET_ALPHA_LIBRARY, "conv2d_vsq", "CompositeExplicitAutograd")
def magnet_alpha_conv2d_vsq(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1,
    *,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    folded_coarse_scale: torch.Tensor,
    block_size: int,
    semantic_input_coarse_scale: Optional[torch.Tensor] = None,
    semantic_weight_coarse_scale: Optional[torch.Tensor] = None,
    input_fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    input_coarse_scale_quant_max: float = (ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT),
    weight_fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    weight_coarse_scale_quant_max: float = (ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT),
) -> torch.Tensor:
    """Execute an Alpha accumulator or the reusable semantic VSQ reference.

    Alpha's physical accumulator is valid only for 32-value blocks after the
    bias has moved into the PPU epilogue. Smaller blocks and bias-bearing
    calls remain useful as target-independent reference configurations, so
    preserve their established floating-point semantics.
    """
    if input.ndim != 4 or weight.ndim != 4:
        raise ValueError("Magnet Alpha VSQ convolution expects NCHW and KCRS")

    declared_folded = folded_coarse_scale.reshape(-1)
    if declared_folded.numel() not in (1, weight.shape[0]):
        raise ValueError(
            "Magnet Alpha VSQ folded coarse scale must contain one value or "
            f"{weight.shape[0]} output-channel values"
        )

    alpha_accumulator = (
        block_size == ALPHA_VSQ_BLOCK_VALUES
        and bias is None
        and semantic_input_coarse_scale is not None
        and semantic_weight_coarse_scale is not None
    )
    if not alpha_accumulator:
        if input.shape[1] != weight.shape[1] * groups:
            raise ValueError(
                "Magnet Alpha VSQ convolution expects NCHW input channels to "
                "match KCRS weight channels and groups"
            )
        scaled_input = input * expand(input_scale, input.shape, block_size)
        scaled_weight = weight * expand(
            weight_scale,
            weight.shape,
            block_size,
        )
        scale_shape = (
            1 if declared_folded.numel() == 1 else weight.shape[0],
            1,
            1,
            1,
        )
        scaled_weight = scaled_weight * declared_folded.reshape(scale_shape)
        return F.conv2d(
            scaled_input,
            scaled_weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )

    input_encoding = encode_alpha_vsq_scales(
        semantic_input_coarse_scale.new_ones(1),
        semantic_input_coarse_scale,
        fine_quant_max=input_fine_quant_max,
        coarse_scale_quant_max=input_coarse_scale_quant_max,
    )
    weight_encoding = encode_alpha_vsq_scales(
        semantic_weight_coarse_scale.new_ones(1),
        semantic_weight_coarse_scale,
        fine_quant_max=weight_fine_quant_max,
        coarse_scale_quant_max=weight_coarse_scale_quant_max,
    )
    physical_folded = input_encoding.coarse_scale.reshape(-1)[
        0
    ] * weight_encoding.coarse_scale.reshape(-1)
    if not torch.allclose(
        declared_folded.to(torch.float64),
        physical_folded.to(torch.float64),
        rtol=1e-6,
        atol=0.0,
    ):
        raise ValueError(
            "Magnet Alpha VSQ folded coarse scale is not Alpha encoded"
        )
    return alpha_vsq_conv2d_accumulator(
        input,
        weight,
        input_scale,
        weight_scale,
        stride,
        padding,
        dilation,
        groups,
        block_size=block_size,
    )


@torch.library.register_fake("magnet_alpha::conv2d_vsq")
def magnet_alpha_conv2d_vsq_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int]] = 1,
    padding: Union[int, Tuple[int]] = 0,
    dilation: Union[int, Tuple[int]] = 1,
    groups: int = 1,
    *,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    folded_coarse_scale: torch.Tensor,
    block_size: int,
    semantic_input_coarse_scale: Optional[torch.Tensor] = None,
    semantic_weight_coarse_scale: Optional[torch.Tensor] = None,
    input_fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    input_coarse_scale_quant_max: float = (ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT),
    weight_fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    weight_coarse_scale_quant_max: float = (ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT),
) -> torch.Tensor:
    declared_folded = folded_coarse_scale.reshape(-1)
    if declared_folded.numel() not in (1, weight.shape[0]):
        raise ValueError(
            "Magnet Alpha VSQ folded coarse scale must contain one value or "
            f"{weight.shape[0]} output-channel values"
        )
    alpha_accumulator = (
        block_size == ALPHA_VSQ_BLOCK_VALUES
        and bias is None
        and semantic_input_coarse_scale is not None
        and semantic_weight_coarse_scale is not None
    )
    if not alpha_accumulator:
        return magnet_alpha_conv2d_vsq(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
            input_scale=input_scale,
            weight_scale=weight_scale,
            folded_coarse_scale=folded_coarse_scale,
            block_size=block_size,
            semantic_input_coarse_scale=semantic_input_coarse_scale,
            semantic_weight_coarse_scale=semantic_weight_coarse_scale,
            input_fine_quant_max=input_fine_quant_max,
            input_coarse_scale_quant_max=(input_coarse_scale_quant_max),
            weight_fine_quant_max=weight_fine_quant_max,
            weight_coarse_scale_quant_max=(weight_coarse_scale_quant_max),
        )
    output = F.conv2d(
        input,
        weight,
        None,
        stride,
        padding,
        dilation,
        groups,
    )
    return output.to(torch.int64)


OPERATOR_LIBRARY = torch.ops.magnet_alpha
CHANNEL_SHUFFLE_OPERATOR = OPERATOR_LIBRARY.channel_shuffle.default
CHANNEL_SHUFFLE_COPY_OPERATOR = OPERATOR_LIBRARY.channel_shuffle_copy.default
UPSAMPLE_NEAREST2D_OPERATOR = OPERATOR_LIBRARY.upsample_nearest2d.default
INT8_CONV2D_ACCUMULATOR_OPERATOR = (
    OPERATOR_LIBRARY.conv2d_int8_accumulator.default
)
NMP_FIXED_REQUANTIZE_OPERATOR = OPERATOR_LIBRARY.nmp_fixed_requantize.default
PPU_FIXED_RELU_OPERATOR = OPERATOR_LIBRARY.ppu_fixed_relu.default
PVS_QUANTIZE_OPERATOR = OPERATOR_LIBRARY.pvs_quantize.default
SATURATING_ADD_OPERATOR = OPERATOR_LIBRARY.saturating_add.default
VSQ_SCALE_ENCODE_OPERATOR = OPERATOR_LIBRARY.vsq_scale_encode.default
VSQ_CONV2D_OPERATOR = OPERATOR_LIBRARY.conv2d_vsq.default
TILED_CONV2D_OPERATORS = (
    INT8_CONV2D_ACCUMULATOR_OPERATOR,
    VSQ_CONV2D_OPERATOR,
)


_NMP_POOL_OPERATORS = frozenset(
    {
        torch.ops.aten.max_pool2d.default,
        torch.ops.quantized_ops.max_pool2d.default,
    }
)
_NMP_CONV2D_OPERATORS = frozenset(
    {
        torch.ops.aten.conv2d.default,
        torch.ops.quantized_ops.conv2d.default,
        torch.ops.alpha_quant.conv2d_mx.default,
        INT8_CONV2D_ACCUMULATOR_OPERATOR,
        VSQ_CONV2D_OPERATOR,
    }
)


def magnet_alpha_operation_classification(
    kind: str,
    attributes=(),
) -> OperationClassification:
    """Freeze one Magnet Alpha operation's scheduler classification."""
    return OperationClassification(
        kind=kind,
        attributes=attributes,
    )


def _scheduler_arg(node, index, name, default=None):
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(name, default)


def _scheduler_input_shape(operands):
    matches = tuple(
        operand.shape for operand in operands if operand.port == "input"
    )
    return matches[0] if len(matches) == 1 else None


def _node_input_shape(node, index):
    if len(node.args) <= index:
        return None
    input_node = node.args[index]
    if not isinstance(input_node, torch.fx.Node):
        return None
    candidates = (
        input_node,
        input_node.meta.get("source_node"),
    )
    for candidate in candidates:
        shape = getattr(candidate, "shape", None)
        if shape is not None:
            return shape
        for key in ("val", "tensor_meta"):
            shape = getattr(
                getattr(candidate, "meta", {}).get(key),
                "shape",
                None,
            )
            if shape is not None:
                return shape
    return None


def _activation_channels(shape, transposed):
    if shape is None or len(shape) != 4:
        return None
    return int(shape[-1] if transposed else shape[1])


def classify_scheduler_operation(
    node,
    operands=(),
    *,
    input_shape=None,
    use_bound=True,
) -> OperationClassification | None:
    """Freeze one Magnet Alpha NMP operation before scheduler dispatch.

    This is the sole Torch/FX recognition boundary shared by compiler memory
    planning and the graph-to-scheduler adapter. Returned facts are immutable
    and contain no graph objects.
    """
    magnet_alpha_meta = node.meta.get("magnet_alpha", {})
    bound = magnet_alpha_meta.get(OPERATION_CLASSIFICATION_META)
    if use_bound and bound is not None:
        if not isinstance(bound, OperationClassification):
            raise TypeError(
                f"{node.name}: invalid bound Magnet Alpha classification"
            )
        return bound

    if input_shape is None:
        input_shape = _scheduler_input_shape(operands)
    if input_shape is None:
        input_shape = _node_input_shape(node, 0)
    transposed = bool(node.meta.get("transposed", False))
    attributes = {"transposed": transposed}

    if node.target == CHANNEL_SHUFFLE_COPY_OPERATOR:
        block_channels = magnet_alpha_meta.get("block_channels")
        if (
            not isinstance(block_channels, int)
            or isinstance(block_channels, bool)
            or block_channels <= 0
        ):
            raise ValueError(
                f"Magnet Alpha shuffle copy {node.name} requires positive "
                "block_channels metadata"
            )
        attributes["block_channels"] = block_channels
        return magnet_alpha_operation_classification(
            "channel_shuffle_copy",
            attributes,
        )

    if node.target in _NMP_CONV2D_OPERATORS:
        input_channels = _activation_channels(
            input_shape,
            transposed,
        )
        if input_channels is None:
            return None
        groups = int(_scheduler_arg(node, 6, "groups", 1))
        if groups != input_channels or groups == 1:
            return None
        attributes.update(
            {
                "groups": groups,
                "stride": _scheduler_arg(node, 3, "stride", 1),
                "padding": _scheduler_arg(node, 4, "padding", 0),
                "dilation": _scheduler_arg(node, 5, "dilation", 1),
            }
        )
        return magnet_alpha_operation_classification(
            "depthwise_conv2d",
            attributes,
        )

    if node.target in _NMP_POOL_OPERATORS:
        attributes.update(
            {
                "kernel_size": _scheduler_arg(
                    node,
                    1,
                    "kernel_size",
                ),
                "stride": _scheduler_arg(node, 2, "stride", ()),
                "padding": _scheduler_arg(node, 3, "padding", 0),
                "dilation": _scheduler_arg(node, 4, "dilation", 1),
                "ceil_mode": _scheduler_arg(
                    node,
                    5,
                    "ceil_mode",
                    False,
                ),
            }
        )
        return magnet_alpha_operation_classification(
            "pool2d",
            attributes,
        )
    if node.target == CHANNEL_SHUFFLE_OPERATOR:
        groups = _scheduler_arg(node, 2, "groups")
        input_shapes = {
            operand.port: operand.shape
            for operand in operands
            if operand.port in {"input1", "input2"}
        }
        if not input_shapes:
            input_shapes = {
                "input1": _node_input_shape(node, 0),
                "input2": _node_input_shape(node, 1),
            }
        if groups != 2 or input_shapes.get("input1") != input_shapes.get(
            "input2"
        ):
            return None
        attributes["groups"] = groups
        return magnet_alpha_operation_classification(
            "channel_shuffle",
            attributes,
        )
    if node.target in (
        UPSAMPLE_NEAREST2D_OPERATOR,
        torch.ops.aten.upsample_nearest2d.default,
        torch.ops.aten.upsample_nearest2d.vec,
    ):
        return magnet_alpha_operation_classification(
            "upsample_nearest2d",
            attributes,
        )
    if node.target == torch.ops.aten.group_norm.default:
        return magnet_alpha_operation_classification(
            "group_norm",
            attributes,
        )
    if node.target in (
        torch.ops.aten.softmax.int,
        torch.ops.aten._softmax.default,
    ):
        attributes["dim"] = int(_scheduler_arg(node, 1, "dim", -1))
        return magnet_alpha_operation_classification(
            "softmax",
            attributes,
        )
    return None


def run_tiled_conv2d(
    operator_target,
    args,
    scales,
    folded_coarse_scale,
    block_size,
    semantic_coarse_scales=(None, None),
    operator_kwargs: Mapping[str, object] | None = None,
):
    """Execute one tiled Magnet Alpha convolution with explicit scale operands."""
    if operator_target not in TILED_CONV2D_OPERATORS:
        raise ValueError(
            f"Unsupported Magnet Alpha tiled convolution: {operator_target}"
        )
    if operator_kwargs is None:
        operator_kwargs = {}
    elif not isinstance(operator_kwargs, Mapping):
        raise TypeError("Alpha tiled operator_kwargs must be a mapping")
    else:
        operator_kwargs = dict(operator_kwargs)
    if operator_target == INT8_CONV2D_ACCUMULATOR_OPERATOR:
        if operator_kwargs:
            raise ValueError(
                "Alpha INT8 tiled convolution does not accept operator "
                f"kwargs: {sorted(operator_kwargs)}"
            )
        return operator_target(*args)
    unexpected = sorted(
        set(operator_kwargs).difference(_TILED_VSQ_SEMANTIC_KWARGS)
    )
    if unexpected:
        raise ValueError(
            "Alpha VSQ tiled convolution received unsupported operator "
            f"kwargs: {unexpected}"
        )
    return operator_target(
        *args,
        input_scale=scales[0],
        weight_scale=scales[1],
        folded_coarse_scale=folded_coarse_scale,
        block_size=block_size,
        semantic_input_coarse_scale=semantic_coarse_scales[0],
        semantic_weight_coarse_scale=semantic_coarse_scales[1],
        **operator_kwargs,
    )


__all__ = [
    "CHANNEL_SHUFFLE_COPY_OPERATOR",
    "CHANNEL_SHUFFLE_OPERATOR",
    "INT8_CONV2D_ACCUMULATOR_OPERATOR",
    "MAGNET_ALPHA_LIBRARY",
    "NMP_FIXED_REQUANTIZE_OPERATOR",
    "OPERATION_CLASSIFICATION_META",
    "OPERATION_COMPUTE_UNIT_META",
    "OPERATOR_LIBRARY",
    "PPU_FIXED_RELU_OPERATOR",
    "PVS_QUANTIZE_OPERATOR",
    "SATURATING_ADD_OPERATOR",
    "TILED_CONV2D_OPERATORS",
    "UPSAMPLE_NEAREST2D_OPERATOR",
    "VSQ_SCALE_ENCODE_OPERATOR",
    "VSQ_CONV2D_OPERATOR",
    "classify_scheduler_operation",
    "magnet_alpha_channel_shuffle_copy",
    "magnet_alpha_channel_shuffle_copy_fake",
    "magnet_alpha_channel_shuffle",
    "magnet_alpha_operation_classification",
    "magnet_alpha_conv2d_int8_accumulator",
    "magnet_alpha_conv2d_int8_accumulator_fake",
    "magnet_alpha_conv2d_vsq",
    "magnet_alpha_conv2d_vsq_fake",
    "magnet_alpha_nmp_fixed_requantize",
    "magnet_alpha_nmp_fixed_requantize_fake",
    "magnet_alpha_ppu_fixed_relu",
    "magnet_alpha_ppu_fixed_relu_fake",
    "magnet_alpha_pvs_quantize",
    "magnet_alpha_pvs_quantize_fake",
    "magnet_alpha_saturating_add",
    "magnet_alpha_saturating_add_fake",
    "magnet_alpha_upsample_nearest2d",
    "magnet_alpha_upsample_nearest2d_fake",
    "magnet_alpha_vsq_scale_encode",
    "magnet_alpha_vsq_scale_encode_fake",
    "run_tiled_conv2d",
]

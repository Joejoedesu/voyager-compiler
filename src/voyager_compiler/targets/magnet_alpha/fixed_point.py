# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Alpha fixed-point instruction encoding and integer reference math."""

import logging
import math
from dataclasses import dataclass

import torch
from torch._subclasses.fake_tensor import is_fake

from voyager_compiler.quantization.fixed_point import (
    FixedPointFormat,
    FixedPointRounding,
)
from voyager_compiler.targets.magnet_alpha.hardware import (
    ALPHA_VSQ_BLOCK_VALUES,
)

logger = logging.getLogger(__name__)

ALPHA_VSQ_MANTISSA_MAX = 255
ALPHA_DEFAULT_VSQ_FINE_ENDPOINT = 7.0
ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT = 127.0
ALPHA_VSQ_REBALANCE = (
    ALPHA_VSQ_MANTISSA_MAX
    * ALPHA_DEFAULT_VSQ_FINE_ENDPOINT
    / ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT
)


def alpha_vsq_rebalance(
    fine_quant_max: float,
    coarse_scale_quant_max: float,
    mantissa_max: int,
) -> float:
    for value, name in (
        (fine_quant_max, "fine_quant_max"),
        (coarse_scale_quant_max, "coarse_scale_quant_max"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 1 <= mantissa_max <= ALPHA_VSQ_MANTISSA_MAX:
        raise ValueError("mantissa_max must fit uint8 and be positive")
    return mantissa_max * fine_quant_max / coarse_scale_quant_max


def _integer_format(
    storage_dtype: torch.dtype,
    bit_width: int,
    *,
    saturate: bool,
) -> FixedPointFormat:
    """Describe one physical integer domain as zero-fraction fixed point."""
    return FixedPointFormat(
        storage_dtype=storage_dtype,
        bit_width=bit_width,
        fractional_bits=0,
        value_scale=1.0,
        signed=True,
        rounding=FixedPointRounding.TOWARD_ZERO,
        saturate=saturate,
    )


ALPHA_INT4_FORMAT = _integer_format(torch.int8, 4, saturate=True)
ALPHA_INT8_FORMAT = _integer_format(torch.int8, 8, saturate=True)


def alpha_q8_8_format(
    normalization_scale: float,
    *,
    rounding: FixedPointRounding | str = FixedPointRounding.NEAREST_EVEN,
    saturate: bool = True,
) -> FixedPointFormat:
    """Describe Alpha's signed normalized-Q8.8 semantic reference.

    ``normalization_scale`` is the real value represented by a raw Q8.8
    payload of 256, so the logical quantization step is
    ``normalization_scale / 256``.
    """
    return FixedPointFormat(
        storage_dtype=torch.int16,
        bit_width=16,
        fractional_bits=8,
        value_scale=normalization_scale,
        signed=True,
        rounding=rounding,
        saturate=saturate,
    )


ALPHA_Q8_8_REFERENCE_FORMAT = alpha_q8_8_format(16.0)
_ALPHA_INT16_WRAP_FORMAT = _integer_format(
    torch.int16,
    16,
    saturate=False,
)
_ALPHA_INT24_WRAP_FORMAT = _integer_format(
    torch.int32,
    24,
    saturate=False,
)
_ALPHA_INT32_WRAP_FORMAT = _integer_format(
    torch.int32,
    32,
    saturate=False,
)


def _instruction_operand(
    value,
    name: str,
    minimum: int,
    maximum: int,
) -> torch.Tensor:
    operand = torch.as_tensor(value)
    if operand.numel() == 0:
        raise ValueError(f"{name} must be nonempty")
    # FX shape propagation may carry target integer operands in floating
    # containers. Preserve the operator's established cast-at-execution
    # behavior; the pure encoders still produce integral payloads.
    operand = operand.to(torch.int64)
    if not is_fake(operand) and torch.any(
        (operand < minimum) | (operand > maximum)
    ):
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return operand


@dataclass(frozen=True)
class AlphaMultiplier:
    """One unsigned 16-bit multiplier and its arithmetic right shift."""

    values: torch.Tensor
    shift: int


@dataclass(frozen=True)
class AlphaPVS:
    """Alpha vector-scaled payload and its uint8 companion mantissa."""

    payload: torch.Tensor
    mantissa: torch.Tensor


@dataclass(frozen=True)
class AlphaVSQScaleEncoding:
    """One logical VSQ scale rebalanced into Alpha's physical domain."""

    mantissa: torch.Tensor
    coarse_scale: torch.Tensor
    rebalance: float


@dataclass(frozen=True)
class AlphaScaleStage:
    """One shared integer multiply and arithmetic-right-shift stage."""

    multiplier: torch.Tensor
    shift: int

    def __post_init__(self) -> None:
        if not isinstance(self.shift, int) or not 0 <= self.shift <= 255:
            raise ValueError("Alpha scale-stage shift must fit unsigned 8-bit")
        object.__setattr__(
            self,
            "multiplier",
            _instruction_operand(
                self.multiplier,
                "Alpha scale-stage multiplier",
                0,
                65535,
            ),
        )


@dataclass(frozen=True)
class AlphaBiasReluStage:
    """One signed-int8 bias followed by an optional ReLU."""

    bias: torch.Tensor | None
    relu: bool

    def __post_init__(self) -> None:
        if not isinstance(self.relu, bool):
            raise ValueError("Alpha ReLU flag must be a bool")
        if self.bias is None:
            return
        object.__setattr__(
            self,
            "bias",
            _instruction_operand(
                self.bias,
                "Alpha bias",
                -128,
                127,
            ),
        )


@dataclass(frozen=True)
class AlphaOutputDomain:
    """Compiler-supported Alpha output scale and integer format.

    Alpha hardware exposes output precision and PVS as independent controls.
    Voyager's current compiler policy deliberately lowers O4 through PVS and
    O8 without PVS; this domain describes that supported subset.
    """

    scale: torch.Tensor
    fixed_format: FixedPointFormat
    uses_pvs: bool

    def __post_init__(self) -> None:
        scale = torch.as_tensor(self.scale).detach().reshape(-1)
        if (
            scale.numel() != 1
            or not torch.is_floating_point(scale)
            or not torch.isfinite(scale).all()
            or scale[0] <= 0
        ):
            raise ValueError(
                "Alpha output scale must be one positive floating value"
            )
        fixed_format = self.fixed_format
        if (
            fixed_format.fractional_bits != 0
            or fixed_format.value_scale != 1.0
            or not fixed_format.signed
            or not fixed_format.saturate
            or fixed_format.bit_width not in (4, 8)
        ):
            raise ValueError("Alpha output must be saturating signed O4 or O8")
        if self.uses_pvs != (fixed_format.bit_width == 4):
            raise ValueError(
                "Voyager Alpha policy requires O4 output to use PVS and "
                "O8 output to omit PVS"
            )
        object.__setattr__(self, "scale", scale.clone())

    @property
    def precision(self) -> int:
        """Return the physical output payload width."""
        return self.fixed_format.bit_width


@dataclass(frozen=True)
class AlphaRequantizationPipeline:
    """Shared PE/NMP scale, bias/ReLU, and output-domain description."""

    primary: AlphaScaleStage
    bias_relu: AlphaBiasReluStage
    output: AlphaOutputDomain
    secondary: AlphaScaleStage | None = None
    accumulator_format: FixedPointFormat | None = None


@dataclass(frozen=True)
class AlphaPENormalization:
    """One deterministic semantic normalization selected for a PE output."""

    normalization_scale: float
    semantic_step: float
    direct_output: bool
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.normalization_scale)
            or self.normalization_scale <= 0
            or not math.isfinite(self.semantic_step)
            or self.semantic_step <= 0
        ):
            raise ValueError(
                "Alpha PE normalization values must be finite and positive"
            )
        if self.direct_output and self.fallback_reason is not None:
            raise ValueError(
                "A direct Alpha PE normalization cannot have a fallback reason"
            )


class AlphaPEDirectHeadroomError(ValueError):
    """The qparam-selected direct PE normalization cannot avoid int16 wrap."""


def encode_alpha_vsq_mantissa(
    logical_fine_scale: torch.Tensor,
    *,
    rebalance: float = ALPHA_VSQ_REBALANCE,
    mantissa_max: int = ALPHA_VSQ_MANTISSA_MAX,
) -> torch.Tensor:
    """Encode one logical VSQ fine scale as an Alpha uint8 mantissa."""
    if not math.isfinite(rebalance) or rebalance <= 0:
        raise ValueError("VSQ scale rebalance must be finite and positive")
    if not 1 <= mantissa_max <= 255:
        raise ValueError("mantissa_max must fit uint8 and be positive")
    fine = logical_fine_scale.detach()
    if fine.dtype in {torch.float16, torch.bfloat16}:
        raise ValueError(
            "Alpha VSQ encoding requires float32 or float64 logical scales"
        )
    if not fine.is_floating_point():
        fine = fine.to(torch.get_default_dtype())
    if fine.numel() == 0 or not torch.isfinite(fine).all():
        raise ValueError("logical fine scale must be nonempty and finite")
    if torch.any(fine < 0):
        raise ValueError("logical fine scale must be nonnegative")
    scaled = fine * rebalance
    # AoV's datapath truncates the normalized value.  Advance by one
    # representable value first so a codebook value rounded just below its
    # integer in the source dtype still recovers that code.  This does not
    # change genuinely off-codebook values.
    scaled = torch.nextafter(
        scaled,
        torch.full_like(scaled, float("inf")),
    )
    return torch.clamp(
        torch.floor(scaled),
        1,
        mantissa_max,
    ).to(torch.uint8)


def encode_alpha_vsq_scales(
    logical_fine_scale: torch.Tensor,
    logical_coarse_scale: torch.Tensor,
    *,
    fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    coarse_scale_quant_max: float = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    mantissa_max: int = ALPHA_VSQ_MANTISSA_MAX,
) -> AlphaVSQScaleEncoding:
    """Rebalance logical VSQ scales into Alpha uint8 mantissas.

    Voyager's logical fine scale can grow to
    ``coarse_quant_max / fine_quant_max``. Alpha instead stores that range in
    an unsigned mantissa. Multiplying the fine scale and dividing the coarse
    scale by the same factor preserves the represented value, apart from the
    final integer mantissa truncation.
    """
    coarse = logical_coarse_scale.detach()
    if not torch.is_floating_point(coarse):
        coarse = coarse.to(torch.float64)
    if coarse.numel() == 0 or not torch.isfinite(coarse).all():
        raise ValueError("logical coarse scale must be nonempty and finite")
    if torch.any(coarse < 0):
        raise ValueError("logical coarse scale must be nonnegative")

    rebalance = alpha_vsq_rebalance(
        fine_quant_max,
        coarse_scale_quant_max,
        mantissa_max,
    )
    mantissa = encode_alpha_vsq_mantissa(
        logical_fine_scale,
        rebalance=rebalance,
        mantissa_max=mantissa_max,
    )
    return AlphaVSQScaleEncoding(
        mantissa=mantissa,
        coarse_scale=coarse / rebalance,
        rebalance=rebalance,
    )


def encode_multiplier(values: torch.Tensor) -> AlphaMultiplier:
    """Encode positive factors using AoV's shared 16-bit binary point."""
    values = values.detach().to(dtype=torch.float64).reshape(-1)
    if values.numel() == 0:
        raise ValueError("Alpha multiplier requires at least one factor")
    if not torch.isfinite(values).all() or torch.any(values < 0):
        raise ValueError("Alpha multiplier factors must be finite and positive")

    maximum = float(torch.max(values).item())
    if maximum == 0.0:
        return AlphaMultiplier(
            values=torch.zeros_like(values, dtype=torch.int64),
            shift=0,
        )

    shift = 16 - int(math.ceil(math.log2(maximum)))
    scaled = torch.round(values * math.ldexp(1.0, shift))
    # Rounding at an exact binary boundary can produce 65536 even though the
    # same factor has a legal uint16 representation with one fewer shift bit.
    while torch.any(scaled > 65535):
        shift -= 1
        scaled = torch.round(values * math.ldexp(1.0, shift))
    return AlphaMultiplier(
        values=scaled.to(torch.int64),
        shift=shift,
    )


def encode_bias(
    bias: torch.Tensor,
    normalization_scale: float,
) -> torch.Tensor:
    """Encode Alpha PE's signed int8 bias in normalized Q8.8 bits."""
    if not math.isfinite(normalization_scale) or normalization_scale <= 0:
        raise ValueError("normalization_scale must be finite and positive")
    scaled = bias.detach().to(torch.float64)
    scaled = torch.round(scaled * (256.0 / normalization_scale))
    return torch.clamp(scaled, -128, 127).to(torch.int8)


def encode_nmp_multiplier(
    values: torch.Tensor,
) -> AlphaMultiplier:
    """Encode Alpha NMP's scalar multiplier and consumed right shift.

    NMP applies this pair directly as ``(accumulator * multiplier) >> shift``.
    Output precision affects the later PVS/truncation stage, not this shift.
    """
    encoded = encode_multiplier(values)
    if torch.any(encoded.values == 0):
        raise ValueError(
            "Alpha NMP multiplier must be nonzero because the encoding treats zero "
            "as an implicit factor of one"
        )
    if not 0 <= encoded.shift <= 255:
        raise ValueError(
            "Alpha NMP binary point "
            f"{encoded.shift} is not an unsigned 8-bit value"
        )
    return encoded


def encode_nmp_bias(
    bias: torch.Tensor,
    accumulator_to_output_scale: torch.Tensor,
) -> int:
    """Encode Alpha NMP's one signed int8 scalar bias.

    The schedule/configuration word carries 16 bits, but Alpha assigns that
    field into ``spec::BiasScalarType`` in the executable NMP datapath.
    ``BiasScalarType`` is signed 8-bit.  Saturate here, as for the PE bias
    encoder, so the compiler reference and serialized target consume the same
    value.
    """
    bias = bias.detach().to(torch.float64).reshape(-1)
    scale = accumulator_to_output_scale.detach().to(torch.float64).reshape(-1)
    if scale.numel() == 1:
        scale = scale.expand_as(bias)
    elif bias.numel() == 1:
        bias = bias.expand_as(scale)
    elif scale.numel() != bias.numel():
        raise ValueError(
            "NMP accumulator scale must be scalar or match the bias channels"
        )
    encoded = torch.round(bias * scale).to(torch.int64)
    if encoded.numel() == 0 or not torch.all(encoded == encoded[0]):
        raise ValueError(
            "Alpha NMP depthwise convolution requires one scalar bias"
        )
    return int(torch.clamp(encoded[0], -128, 127).item())


def reconstruct_accumulator_bias(
    bias: torch.Tensor,
    accumulator_scale: torch.Tensor,
) -> torch.Tensor:
    """Convert integer-convolution bias from accumulator to real units."""
    bias = bias.detach().to(torch.float64).reshape(-1)
    accumulator_scale = accumulator_scale.detach().to(torch.float64).reshape(-1)
    if accumulator_scale.numel() == 1:
        accumulator_scale = accumulator_scale.expand_as(bias)
    elif accumulator_scale.numel() != bias.numel():
        raise ValueError(
            "accumulator scale must be scalar or have one value per bias "
            "channel"
        )
    return bias * accumulator_scale


def primary_factors(
    accumulator_scale: torch.Tensor,
    normalization_scale: float,
    vsq_uint8_fine_scales: bool,
) -> torch.Tensor:
    """Convert Voyager accumulator scales to Alpha primary multipliers.

    Alpha's Q8.8 raw payload contributes a factor of 256. VSQ's two uint8
    fine-scale operands use a combined hardware convention that contributes
    one additional factor of 256 relative to W8 accumulator scales.
    """
    if not math.isfinite(normalization_scale) or normalization_scale <= 0:
        raise ValueError("normalization_scale must be finite and positive")
    storage_factor = 65536.0 if vsq_uint8_fine_scales else 256.0
    return accumulator_scale * (storage_factor / normalization_scale)


def alpha_output_domain(
    logical_scale: torch.Tensor | float,
    output_precision: int,
    uses_pvs: bool,
    *,
    fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    coarse_scale_quant_max: float = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    mantissa_max: int = ALPHA_VSQ_MANTISSA_MAX,
) -> AlphaOutputDomain:
    """Build the compiler-supported output domain shared by PE and NMP."""
    if output_precision == 4:
        fixed_format = ALPHA_INT4_FORMAT
    elif output_precision == 8:
        fixed_format = ALPHA_INT8_FORMAT
    else:
        raise ValueError("Alpha output precision must be 4 or 8")
    scale = torch.as_tensor(logical_scale)
    if not torch.is_floating_point(scale):
        scale = scale.to(torch.get_default_dtype())
    if uses_pvs:
        scale = scale / alpha_vsq_rebalance(
            fine_quant_max,
            coarse_scale_quant_max,
            mantissa_max,
        )
    return AlphaOutputDomain(
        scale=scale,
        fixed_format=fixed_format,
        uses_pvs=uses_pvs,
    )


def _scale_stage(
    encoded: AlphaMultiplier,
    *,
    shift_offset: int = 0,
) -> AlphaScaleStage:
    return AlphaScaleStage(
        multiplier=encoded.values,
        shift=encoded.shift + shift_offset,
    )


def encode_alpha_pe_pipeline(
    folded_scale: torch.Tensor,
    real_bias: torch.Tensor | None,
    output: AlphaOutputDomain,
    normalization_scale: float,
    *,
    vsq_uint8_fine_scales: bool,
    relu: bool,
) -> AlphaRequantizationPipeline:
    """Encode one PE epilogue without projecting its instruction schema."""
    folded_scale = folded_scale.detach()
    primary = encode_multiplier(
        primary_factors(
            folded_scale.reshape(-1),
            normalization_scale,
            vsq_uint8_fine_scales,
        )
    )
    secondary = encode_multiplier(
        folded_scale.new_tensor(
            [normalization_scale / float(output.scale[0].item())]
        )
    )
    bias = (
        None
        if real_bias is None
        else encode_bias(real_bias, normalization_scale)
    )
    return AlphaRequantizationPipeline(
        primary=_scale_stage(primary),
        bias_relu=AlphaBiasReluStage(bias=bias, relu=relu),
        output=output,
        secondary=_scale_stage(secondary, shift_offset=8),
    )


def select_alpha_pe_normalization(
    output: AlphaOutputDomain,
    *,
    vsq_uint8_fine_scales: bool,
    relu: bool,
) -> AlphaPENormalization:
    """Select a PE normalization from frozen output-domain facts only.

    A bias-free residual PE and an ordinary scalar-O8 terminal both benefit
    from making the normalized signed-int16 step equal to the final output
    step.  That makes Alpha's secondary multiplier an exact identity and
    avoids magnifying the first-stage floor loss.  The currently validated
    surface is W8, scalar O8, and no fused ReLU.  VSQ, O4/PVS, and ReLU paths
    retain the established fixed Q8.8 normalization.
    """
    direct_output = (
        not vsq_uint8_fine_scales
        and output.precision == 8
        and not output.uses_pvs
        and not relu
    )
    if direct_output:
        semantic_step = float(output.scale.reshape(-1)[0].item())
        normalization_scale = semantic_step * 256.0
    else:
        normalization_scale = ALPHA_Q8_8_REFERENCE_FORMAT.value_scale
        semantic_step = ALPHA_Q8_8_REFERENCE_FORMAT.scale
    return AlphaPENormalization(
        normalization_scale=normalization_scale,
        semantic_step=semantic_step,
        direct_output=direct_output,
    )


def _expand_pe_channel_values(
    values: torch.Tensor,
    channels: int,
    description: str,
) -> torch.Tensor:
    values = values.detach().to(torch.float64).reshape(-1)
    if values.numel() == 1:
        return values.expand(channels)
    if values.numel() != channels:
        raise ValueError(
            f"{description} must be scalar or match the PE output channels"
        )
    return values


def _validate_direct_alpha_pe_pipeline(
    folded_scale: torch.Tensor,
    real_bias: torch.Tensor | None,
    output: AlphaOutputDomain,
    pipeline: AlphaRequantizationPipeline,
    accumulator_real_amax: torch.Tensor | float | None,
) -> None:
    """Prove direct-output operands and signed-int16 accumulator headroom."""
    folded = folded_scale.detach().to(torch.float64).reshape(-1)
    if (
        folded.numel() == 0
        or not torch.isfinite(folded).all()
        or torch.any(folded <= 0)
    ):
        raise ValueError(
            "Alpha direct-output PE scale must be finite and positive"
        )
    channels = folded.numel()
    bias = (
        torch.zeros(channels, dtype=torch.float64, device=folded.device)
        if real_bias is None
        else _expand_pe_channel_values(real_bias, channels, "Alpha PE bias")
    )
    if not torch.isfinite(bias).all():
        raise ValueError("Alpha direct-output PE bias must be finite")

    encoded_bias = pipeline.bias_relu.bias
    if encoded_bias is None:
        bias_codes = torch.zeros(
            channels,
            dtype=torch.int64,
            device=folded.device,
        )
    else:
        bias_codes = _expand_pe_channel_values(
            encoded_bias,
            channels,
            "Alpha encoded PE bias",
        ).to(torch.int64)
    output_scale = float(output.scale.reshape(-1)[0].item())
    unbounded_bias_codes = torch.round(bias / output_scale)
    if torch.any((unbounded_bias_codes < -128) | (unbounded_bias_codes > 127)):
        raise ValueError(
            "Alpha direct-output PE bias exceeds signed-int8 range"
        )
    if not torch.equal(
        bias_codes.cpu(),
        unbounded_bias_codes.to(torch.int64).cpu(),
    ):
        raise ValueError("Alpha direct-output PE bias was not encoded exactly")

    primary = pipeline.primary.multiplier.reshape(-1)
    if primary.numel() != channels or torch.any(primary == 0):
        raise ValueError(
            "Alpha direct-output PE requires one nonzero uint16 primary "
            "multiplier per output channel"
        )
    secondary = pipeline.secondary
    if secondary is None or secondary.multiplier.numel() != 1:
        raise ValueError(
            "Alpha direct-output PE requires one secondary multiplier"
        )
    secondary_multiplier = int(secondary.multiplier.reshape(-1)[0].item())
    if secondary_multiplier != 1 << secondary.shift:
        raise ValueError(
            "Alpha direct-output PE secondary stage must be exact unity"
        )

    if accumulator_real_amax is None:
        raise ValueError(
            "Alpha direct-output PE requires a proved accumulator bound"
        )
    accumulator_bound = torch.as_tensor(
        accumulator_real_amax,
        dtype=torch.float64,
        device=folded.device,
    )
    accumulator_bound = _expand_pe_channel_values(
        accumulator_bound,
        channels,
        "Alpha PE accumulator bound",
    )
    if not torch.isfinite(accumulator_bound).all() or torch.any(
        accumulator_bound < 0
    ):
        raise ValueError(
            "Alpha PE accumulator bound must be finite and " "nonnegative"
        )

    primary_values = primary.to(torch.int64).cpu().tolist()
    folded_values = folded.cpu().tolist()
    bias_code_values = bias_codes.cpu().tolist()
    accumulator_bound_values = accumulator_bound.cpu().tolist()
    for channel, (
        channel_scale,
        channel_bias_code,
        accumulator_amax,
        multiplier,
    ) in enumerate(
        zip(
            folded_values,
            bias_code_values,
            accumulator_bound_values,
            primary_values,
        )
    ):
        raw_bound = math.ceil(
            math.nextafter(
                accumulator_amax / channel_scale,
                math.inf,
            )
        )
        positive = (raw_bound * multiplier) >> pipeline.primary.shift
        negative = (-raw_bound * multiplier) >> pipeline.primary.shift
        for stage_one in (negative, positive):
            if not -(1 << 15) <= stage_one <= (1 << 15) - 1:
                raise AlphaPEDirectHeadroomError(
                    "Alpha direct-output PE proved primary-stage bound "
                    f"exceeds signed-int16 range in channel {channel}"
                )
            biased = stage_one + channel_bias_code
            if not -(1 << 15) <= biased <= (1 << 15) - 1:
                raise AlphaPEDirectHeadroomError(
                    "Alpha direct-output PE proved post-bias bound exceeds "
                    f"signed-int16 range in channel {channel}"
                )


def encode_selected_alpha_pe_pipeline(
    folded_scale: torch.Tensor,
    real_bias: torch.Tensor | None,
    output: AlphaOutputDomain,
    *,
    vsq_uint8_fine_scales: bool,
    relu: bool,
    accumulator_real_amax: torch.Tensor | float | None = None,
) -> tuple[AlphaRequantizationPipeline, AlphaPENormalization]:
    """Select, encode, and validate one production Alpha PE pipeline.

    Alpha stores the primary and biased result in a signed 16-bit register
    whose nonlinear interpretation is Q8.8. Direct O8 normalization may use
    those raw bits only when its proved range fits that register. If range is
    the sole failure, native Q8.8 normalization is the legal narrower-scale
    fallback; malformed qparams or an unencodable multiplier remain errors.
    """
    normalization = select_alpha_pe_normalization(
        output,
        vsq_uint8_fine_scales=vsq_uint8_fine_scales,
        relu=relu,
    )
    pipeline = encode_alpha_pe_pipeline(
        folded_scale,
        real_bias,
        output,
        normalization.normalization_scale,
        vsq_uint8_fine_scales=vsq_uint8_fine_scales,
        relu=relu,
    )
    if normalization.direct_output:
        try:
            _validate_direct_alpha_pe_pipeline(
                folded_scale,
                real_bias,
                output,
                pipeline,
                accumulator_real_amax,
            )
        except AlphaPEDirectHeadroomError as error:
            normalization = AlphaPENormalization(
                normalization_scale=(ALPHA_Q8_8_REFERENCE_FORMAT.value_scale),
                semantic_step=ALPHA_Q8_8_REFERENCE_FORMAT.scale,
                direct_output=False,
                fallback_reason=str(error),
            )
            pipeline = encode_alpha_pe_pipeline(
                folded_scale,
                real_bias,
                output,
                normalization.normalization_scale,
                vsq_uint8_fine_scales=vsq_uint8_fine_scales,
                relu=relu,
            )
    return pipeline, normalization


def _uniform_positive_scalar(
    values: torch.Tensor,
    description: str,
) -> torch.Tensor:
    values = values.detach().reshape(-1)
    if (
        values.numel() == 0
        or not torch.is_floating_point(values)
        or not torch.isfinite(values).all()
        or torch.any(values <= 0)
        or not torch.all(values == values[0])
    ):
        raise ValueError(f"{description} must be one positive scalar value")
    return values[:1]


def encode_alpha_nmp_pipeline(
    accumulator_scale: torch.Tensor,
    accumulator_bias: torch.Tensor | None,
    output: AlphaOutputDomain,
    *,
    relu: bool,
) -> AlphaRequantizationPipeline:
    """Encode one NMP epilogue without projecting its instruction schema."""
    output_scale = output.scale.to(
        device=accumulator_scale.device,
        dtype=accumulator_scale.dtype,
    )
    accumulator_to_output = _uniform_positive_scalar(
        accumulator_scale / output_scale,
        "Alpha NMP accumulator-to-output scale",
    )
    multiplier = encode_nmp_multiplier(accumulator_to_output)
    bias = (
        0
        if accumulator_bias is None
        else encode_nmp_bias(
            accumulator_bias,
            accumulator_to_output,
        )
    )
    return AlphaRequantizationPipeline(
        primary=_scale_stage(multiplier),
        bias_relu=AlphaBiasReluStage(
            bias=torch.tensor(
                [bias],
                dtype=torch.int64,
                device=accumulator_scale.device,
            ),
            relu=relu,
        ),
        output=output,
        accumulator_format=_ALPHA_INT24_WRAP_FORMAT,
    )


def _arithmetic_shift(values: torch.Tensor, shift: int) -> torch.Tensor:
    values = values.to(torch.int64)
    if shift >= 0:
        return torch.bitwise_right_shift(values, shift)
    return torch.bitwise_left_shift(values, -shift)


def _apply_integer_format(
    values: torch.Tensor,
    fixed_format: FixedPointFormat,
) -> torch.Tensor:
    """Apply one zero-fraction physical domain to integral values."""
    if fixed_format.fractional_bits != 0 or fixed_format.value_scale != 1.0:
        raise ValueError("physical integer domains require zero fraction bits")
    values = values.to(torch.int64)
    if fixed_format.saturate:
        return torch.clamp(values, fixed_format.qmin, fixed_format.qmax)
    span = 1 << fixed_format.bit_width
    return torch.remainder(values - fixed_format.qmin, span) + fixed_format.qmin


def alpha_truncate_output(
    values: torch.Tensor,
    output_precision: int,
) -> torch.Tensor:
    """Apply Alpha's final saturating O4/O8 payload conversion."""
    if output_precision == 4:
        output_format = ALPHA_INT4_FORMAT
    elif output_precision == 8:
        output_format = ALPHA_INT8_FORMAT
    else:
        raise ValueError("Alpha output precision must be 4 or 8")
    return _apply_integer_format(values, output_format).to(torch.int8)


def _execute_alpha_stages(
    accumulator: torch.Tensor,
    primary: AlphaScaleStage,
    bias_relu: AlphaBiasReluStage,
    secondary: AlphaScaleStage | None,
    *,
    accumulator_format: FixedPointFormat | None = None,
) -> torch.Tensor:
    values = accumulator.to(torch.int64)
    if accumulator_format is not None:
        values = _apply_integer_format(values, accumulator_format)
    values = _arithmetic_shift(
        values * primary.multiplier,
        primary.shift,
    )
    values = _apply_integer_format(values, _ALPHA_INT16_WRAP_FORMAT)

    bias = bias_relu.bias
    if bias is not None:
        values = values + bias
    values = _apply_integer_format(values, _ALPHA_INT16_WRAP_FORMAT)
    if bias_relu.relu:
        values = torch.clamp_min(values, 0)

    if secondary is not None:
        values = _arithmetic_shift(
            values * secondary.multiplier,
            secondary.shift,
        )
    return values


def execute_alpha_requantization(
    accumulator: torch.Tensor,
    pipeline: AlphaRequantizationPipeline,
) -> torch.Tensor:
    """Execute the scale and bias/ReLU stages shared by PE and NMP."""
    return _execute_alpha_stages(
        accumulator,
        pipeline.primary,
        pipeline.bias_relu,
        pipeline.secondary,
        accumulator_format=pipeline.accumulator_format,
    )


def _pair(
    value: int | tuple[int, int] | list[int],
    name: str,
    *,
    allow_zero: bool,
) -> tuple[int, int]:
    if isinstance(value, int):
        pair = (value, value)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        pair = tuple(value)
    else:
        raise ValueError(f"{name} must be one integer or a pair")
    minimum = 0 if allow_zero else 1
    if any(not isinstance(item, int) or item < minimum for item in pair):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} values must be {qualifier} integers")
    return pair


def _integer_tensor(
    values: torch.Tensor,
    name: str,
    minimum: int,
    maximum: int,
) -> torch.Tensor:
    if values.numel() == 0:
        raise ValueError(f"{name} must be nonempty")
    if torch.is_floating_point(values):
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} must be finite")
        if not torch.all(values == torch.round(values)):
            raise ValueError(f"{name} must contain integer values")
    integers = values.to(torch.int64)
    if torch.any((integers < minimum) | (integers > maximum)):
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return integers


def alpha_w8_accumulator_real_bound(
    weight_payload: torch.Tensor,
    accumulator_scale: torch.Tensor,
    *,
    input_code_amax: int = 128,
) -> torch.Tensor:
    """Bound each signed-INT32 W8 accumulator in its real-value domain."""
    if weight_payload.ndim < 2 or weight_payload.numel() == 0:
        raise ValueError("Alpha W8 accumulator bound requires matrix weights")
    if not isinstance(input_code_amax, int) or input_code_amax <= 0:
        raise ValueError("Alpha W8 input-code maximum must be positive")
    weight = _integer_tensor(
        weight_payload,
        "Alpha W8 weight payload",
        -128,
        127,
    ).to(torch.float64)

    reduction_axes = tuple(range(1, weight.ndim))
    raw_bound = torch.sum(
        torch.abs(weight) * float(input_code_amax),
        dim=reduction_axes,
    )
    # Alpha exposes a signed-int32 accumulator to its PPU. If a larger dot
    # product wraps, no post-wrap code can exceed the signed-INT32 magnitude.
    raw_bound = torch.clamp_max(raw_bound, float(1 << 31))

    scale = accumulator_scale.detach().to(torch.float64).reshape(-1)
    if scale.numel() == 1:
        scale = scale.expand_as(raw_bound)
    if (
        scale.shape != raw_bound.shape
        or not torch.isfinite(scale).all()
        or torch.any(scale <= 0)
    ):
        raise ValueError(
            "Alpha W8 accumulator scale must be positive and match the "
            "weight outputs"
        )
    return raw_bound * scale


def alpha_int8_conv2d_accumulator(
    input_payload: torch.Tensor,
    weight_payload: torch.Tensor,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
) -> torch.Tensor:
    """Execute a bias-free signed-int8 Conv2d with int32 MAC semantics.

    Integer products are accumulated losslessly in int64 before the result is
    narrowed modulo 2**32. Returning the signed result in an int64 tensor
    avoids PyTorch backend-dependent integer-convolution behavior while
    preserving every physical int32 accumulator bit.
    """
    if input_payload.ndim != 4 or weight_payload.ndim != 4:
        raise ValueError("Alpha int8 Conv2d expects NCHW input and KCRS weight")
    if not isinstance(groups, int) or groups <= 0:
        raise ValueError("Alpha int8 Conv2d groups must be positive")

    stride_h, stride_w = _pair(stride, "stride", allow_zero=False)
    pad_h, pad_w = _pair(padding, "padding", allow_zero=True)
    dilation_h, dilation_w = _pair(
        dilation,
        "dilation",
        allow_zero=False,
    )
    batch, channels, input_h, input_w = input_payload.shape
    outputs, weight_channels, kernel_h, kernel_w = weight_payload.shape
    if channels != weight_channels * groups:
        raise ValueError("Alpha int8 Conv2d input and weight channels differ")
    if outputs % groups:
        raise ValueError(
            "Alpha int8 Conv2d output channels must be group aligned"
        )
    if input_payload.device != weight_payload.device:
        raise ValueError("Alpha int8 Conv2d operands must share one device")

    input_values = _integer_tensor(
        input_payload,
        "input payload",
        -128,
        127,
    )
    weight_values = _integer_tensor(
        weight_payload,
        "weight payload",
        -128,
        127,
    )
    output_h = (
        input_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    output_w = (
        input_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    if output_h <= 0 or output_w <= 0:
        raise ValueError("Alpha int8 Conv2d has an empty output")

    padded = torch.nn.functional.pad(
        input_values,
        (pad_w, pad_w, pad_h, pad_h),
    )
    channels_per_group = channels // groups
    outputs_per_group = outputs // groups
    grouped_input = padded.reshape(
        batch,
        groups,
        channels_per_group,
        padded.shape[2],
        padded.shape[3],
    )
    grouped_weight = weight_values.reshape(
        groups,
        outputs_per_group,
        channels_per_group,
        kernel_h,
        kernel_w,
    )
    accumulator = torch.zeros(
        (batch, groups, outputs_per_group, output_h, output_w),
        dtype=torch.int64,
        device=input_payload.device,
    )
    for kernel_row in range(kernel_h):
        row_start = kernel_row * dilation_h
        row_stop = row_start + output_h * stride_h
        for kernel_column in range(kernel_w):
            column_start = kernel_column * dilation_w
            column_stop = column_start + output_w * stride_w
            patch = grouped_input[
                :,
                :,
                :,
                row_start:row_stop:stride_h,
                column_start:column_stop:stride_w,
            ]
            kernel = grouped_weight[
                :,
                :,
                :,
                kernel_row,
                kernel_column,
            ]
            accumulator += torch.einsum(
                "ngchw,gkc->ngkhw",
                patch,
                kernel,
            )
    return _apply_integer_format(
        accumulator.reshape(batch, outputs, output_h, output_w),
        _ALPHA_INT32_WRAP_FORMAT,
    )


@dataclass(frozen=True)
class _AlphaVSQConv2dOperands:
    """Validated physical operands for one Alpha VSQ Conv2d."""

    input_values: torch.Tensor
    weight_values: torch.Tensor
    input_scales: torch.Tensor
    weight_scales: torch.Tensor
    stride_h: int
    stride_w: int
    pad_h: int
    pad_w: int
    dilation_h: int
    dilation_w: int
    output_h: int
    output_w: int
    block_size: int


def _prepare_alpha_vsq_conv2d(
    input_payload: torch.Tensor,
    weight_payload: torch.Tensor,
    input_mantissa: torch.Tensor,
    weight_mantissa: torch.Tensor,
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
    dilation: int | tuple[int, int],
    groups: int,
    *,
    block_size: int,
) -> _AlphaVSQConv2dOperands:
    """Validate and normalize one Alpha VSQ Conv2d invocation."""
    if input_payload.ndim != 4 or weight_payload.ndim != 4:
        raise ValueError("Alpha VSQ Conv2d expects NCHW input and KCRS weight")
    if groups != 1:
        raise ValueError("Alpha PE VSQ Conv2d currently requires groups=1")
    if block_size != ALPHA_VSQ_BLOCK_VALUES:
        raise ValueError("Alpha PE VSQ Conv2d requires 32-value blocks")

    stride_h, stride_w = _pair(stride, "stride", allow_zero=False)
    pad_h, pad_w = _pair(padding, "padding", allow_zero=True)
    dilation_h, dilation_w = _pair(
        dilation,
        "dilation",
        allow_zero=False,
    )
    batch, channels, input_h, input_w = input_payload.shape
    outputs, weight_channels, kernel_h, kernel_w = weight_payload.shape
    if channels != weight_channels:
        raise ValueError("Alpha VSQ Conv2d input and weight channels differ")
    if channels % block_size:
        raise ValueError("Alpha VSQ Conv2d channels must be block aligned")

    channel_blocks = channels // block_size
    expected_input_scales = (batch, channel_blocks, input_h, input_w)
    expected_weight_scales = (
        outputs,
        channel_blocks,
        kernel_h,
        kernel_w,
    )
    if tuple(input_mantissa.shape) != expected_input_scales:
        raise ValueError(
            "Alpha input mantissa shape must be "
            f"{expected_input_scales}, got {tuple(input_mantissa.shape)}"
        )
    if tuple(weight_mantissa.shape) != expected_weight_scales:
        raise ValueError(
            "Alpha weight mantissa shape must be "
            f"{expected_weight_scales}, got {tuple(weight_mantissa.shape)}"
        )

    tensors = (
        input_payload,
        weight_payload,
        input_mantissa,
        weight_mantissa,
    )
    if any(tensor.device != input_payload.device for tensor in tensors[1:]):
        raise ValueError("Alpha VSQ Conv2d operands must share one device")
    input_values = _integer_tensor(
        input_payload,
        "input payload",
        -8,
        7,
    )
    weight_values = _integer_tensor(
        weight_payload,
        "weight payload",
        # Voyager's Alpha compilation policy uses symmetric W4 weights.
        # The physical signed nibble can encode -8, but current calibrated
        # qspecs intentionally reserve that code and lower weights in [-7, 7].
        -7,
        7,
    )
    input_scales = _integer_tensor(
        input_mantissa,
        "input mantissa",
        0,
        255,
    )
    weight_scales = _integer_tensor(
        weight_mantissa,
        "weight mantissa",
        0,
        255,
    )

    output_h = (
        input_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1
    ) // stride_h + 1
    output_w = (
        input_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1
    ) // stride_w + 1
    if output_h <= 0 or output_w <= 0:
        raise ValueError("Alpha VSQ Conv2d has an empty output")

    return _AlphaVSQConv2dOperands(
        input_values=input_values,
        weight_values=weight_values,
        input_scales=input_scales,
        weight_scales=weight_scales,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        output_h=output_h,
        output_w=output_w,
        block_size=block_size,
    )


_ALPHA_VSQ_CONV_TARGET_ELEMENTS = 8 * 1024 * 1024
_ALPHA_VSQ_DOT_PRODUCT_MAX = ALPHA_VSQ_BLOCK_VALUES * 8 * 7
_ALPHA_VSQ_SCALE_PRODUCT_MAX = (
    ALPHA_VSQ_MANTISSA_MAX * ALPHA_VSQ_MANTISSA_MAX
) >> 8
_ALPHA_VSQ_MAX_EXACT_FLOAT_TERMS = (2**24 - 1) // (
    _ALPHA_VSQ_DOT_PRODUCT_MAX * _ALPHA_VSQ_SCALE_PRODUCT_MAX
)


def _alpha_vsq_conv2d_accumulator_vectorized(
    operands: _AlphaVSQConv2dOperands,
) -> torch.Tensor:
    """Execute VSQ MACs with grouped 1x1 convolutions over all positions."""
    input_values = operands.input_values
    weight_values = operands.weight_values
    input_scales = operands.input_scales
    weight_scales = operands.weight_scales
    batch, channels, _, _ = input_values.shape
    outputs, _, kernel_h, kernel_w = weight_values.shape
    channel_blocks = channels // operands.block_size
    stride_h = operands.stride_h
    stride_w = operands.stride_w

    padded_input = torch.nn.functional.pad(
        input_values.to(torch.float32),
        (
            operands.pad_w,
            operands.pad_w,
            operands.pad_h,
            operands.pad_h,
        ),
    )
    padded_scales = torch.nn.functional.pad(
        input_scales,
        (
            operands.pad_w,
            operands.pad_w,
            operands.pad_h,
            operands.pad_h,
        ),
    )
    blocked_weight = weight_values.reshape(
        outputs,
        channel_blocks,
        operands.block_size,
        kernel_h,
        kernel_w,
    ).to(torch.float32)
    accumulator = torch.zeros(
        (batch, outputs, operands.output_h, operands.output_w),
        dtype=(
            torch.float32
            if channel_blocks * kernel_h * kernel_w
            <= _ALPHA_VSQ_MAX_EXACT_FLOAT_TERMS
            else torch.int64
        ),
        device=input_values.device,
    )
    # Float32 represents every integer through 2**24 exactly. Each dot and
    # scaled term is below the bounds used above, so common Alpha channel
    # counts can use optimized floating convolution without changing one bit.
    exact_float_reduction = channel_blocks <= _ALPHA_VSQ_MAX_EXACT_FLOAT_TERMS
    exact_float_accumulation = accumulator.dtype == torch.float32
    if exact_float_reduction:
        padded_scales = padded_scales.to(torch.float32)
        weight_scales = weight_scales.to(torch.float32)

    elements_per_output = (
        batch * channel_blocks * operands.output_h * operands.output_w
    )
    outputs_per_chunk = max(
        1,
        _ALPHA_VSQ_CONV_TARGET_ELEMENTS // elements_per_output,
    )
    outputs_per_chunk = min(outputs, outputs_per_chunk)

    for kernel_row in range(kernel_h):
        row_start = kernel_row * operands.dilation_h
        row_stop = row_start + operands.output_h * stride_h
        for kernel_column in range(kernel_w):
            column_start = kernel_column * operands.dilation_w
            column_stop = column_start + operands.output_w * stride_w
            input_patch = padded_input[
                :,
                :,
                row_start:row_stop:stride_h,
                column_start:column_stop:stride_w,
            ]
            scale_patch = padded_scales[
                :,
                :,
                row_start:row_stop:stride_h,
                column_start:column_stop:stride_w,
            ]

            for output_start in range(0, outputs, outputs_per_chunk):
                output_stop = min(
                    outputs,
                    output_start + outputs_per_chunk,
                )
                output_count = output_stop - output_start
                kernel = (
                    blocked_weight[
                        output_start:output_stop,
                        :,
                        :,
                        kernel_row,
                        kernel_column,
                    ]
                    .permute(1, 0, 2)
                    .reshape(
                        channel_blocks * output_count,
                        operands.block_size,
                        1,
                        1,
                    )
                )
                # Each float32 dot contains 32 products of signed int4
                # values. Its magnitude is below 2**24, so the result is an
                # exact integer even when a backend changes summation order.
                dot = torch.nn.functional.conv2d(
                    input_patch,
                    kernel,
                    groups=channel_blocks,
                )
                kernel_scales = weight_scales[
                    output_start:output_stop,
                    :,
                    kernel_row,
                    kernel_column,
                ].transpose(0, 1)
                if exact_float_reduction:
                    if channel_blocks == 1:
                        dot = dot.reshape(
                            batch,
                            output_count,
                            operands.output_h,
                            operands.output_w,
                        )
                        scale_product = torch.mul(
                            scale_patch[:, 0, None, :, :],
                            kernel_scales[0, None, :, None, None],
                        )
                        scale_product.div_(256.0).floor_()
                        contribution = dot.mul_(scale_product)
                    else:
                        dot = dot.reshape(
                            batch,
                            channel_blocks,
                            output_count,
                            operands.output_h,
                            operands.output_w,
                        )
                        scale_product = torch.mul(
                            scale_patch[:, :, None, :, :],
                            kernel_scales[None, :, :, None, None],
                        )
                        scale_product.div_(256.0).floor_()
                        contribution = torch.sum(
                            dot.mul_(scale_product),
                            dim=1,
                        )
                else:
                    dot = dot.to(torch.int64).reshape(
                        batch,
                        channel_blocks,
                        output_count,
                        operands.output_h,
                        operands.output_w,
                    )
                    scale_product = torch.bitwise_right_shift(
                        scale_patch[:, :, None, :, :]
                        * kernel_scales[None, :, :, None, None],
                        8,
                    )
                    contribution = torch.sum(
                        dot * scale_product,
                        dim=1,
                    )

                current = accumulator[:, output_start:output_stop]
                if exact_float_accumulation:
                    current += contribution
                else:
                    # Grouping block contributions before each wrap is exact:
                    # addition modulo 2**32 is associative.
                    accumulator[:, output_start:output_stop] = (
                        _apply_integer_format(
                            current + contribution.to(torch.int64),
                            _ALPHA_INT32_WRAP_FORMAT,
                        )
                    )
    return _apply_integer_format(
        accumulator.to(torch.int64),
        _ALPHA_INT32_WRAP_FORMAT,
    )


def alpha_vsq_conv2d_accumulator(
    input_payload: torch.Tensor,
    weight_payload: torch.Tensor,
    input_mantissa: torch.Tensor,
    weight_mantissa: torch.Tensor,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
    groups: int = 1,
    *,
    block_size: int = ALPHA_VSQ_BLOCK_VALUES,
) -> torch.Tensor:
    """Execute Alpha's W4/A4 VSQ Conv2d accumulation.

    Each 32-value dot product is multiplied by the high eight bits of the two
    uint8 scale mantissas before being accumulated into a wrapping signed
    32-bit result. Coarse scales, bias, and PPU processing are intentionally
    outside this helper.
    """
    operands = _prepare_alpha_vsq_conv2d(
        input_payload,
        weight_payload,
        input_mantissa,
        weight_mantissa,
        stride,
        padding,
        dilation,
        groups,
        block_size=block_size,
    )
    return _alpha_vsq_conv2d_accumulator_vectorized(operands)


def alpha_pvs_quantize(
    values: torch.Tensor,
    output_precision: int,
    vector_dim: int = -1,
) -> AlphaPVS:
    """Apply Alpha PVS independently to each zero-padded 32-lane vector."""
    if output_precision not in (4, 8):
        raise ValueError("Alpha PVS output precision must be 4 or 8")
    if values.ndim == 0:
        raise ValueError("Alpha PVS requires a vector dimension")
    vector_dim %= values.ndim
    moved = torch.movedim(values.to(torch.int64), vector_dim, -1)
    lanes = moved.shape[-1]
    padding = (-lanes) % ALPHA_VSQ_BLOCK_VALUES
    if padding:
        moved = torch.nn.functional.pad(moved, (0, padding))
    packets = moved.reshape(
        *moved.shape[:-1],
        -1,
        ALPHA_VSQ_BLOCK_VALUES,
    )

    maximum = torch.amax(torch.abs(packets), dim=-1)
    raw_mantissa = torch.where(
        maximum <= 7,
        torch.ones_like(maximum),
        torch.bitwise_right_shift(maximum, 3),
    )
    mantissa = torch.bitwise_and(raw_mantissa, 0xFF)
    mantissa = torch.where(mantissa == 0, 1, mantissa)
    reciprocal = torch.div(8192, mantissa, rounding_mode="floor")

    clipped = torch.clamp(packets, -8191, 8191)
    payload = _arithmetic_shift(clipped * reciprocal.unsqueeze(-1), 13)
    payload = alpha_truncate_output(payload, output_precision).reshape(
        moved.shape
    )
    payload = torch.movedim(payload[..., :lanes], -1, vector_dim)
    mantissa = torch.movedim(mantissa, -1, vector_dim)
    return AlphaPVS(payload, mantissa.to(torch.uint8))


def alpha_nmp_epilogue(
    accumulator: torch.Tensor,
    scale_factor: int,
    binary_point: int,
    bias_factor: int,
    relu: bool = True,
) -> torch.Tensor:
    """Execute Alpha depthwise scaling and scalar bias ordering."""
    # Alpha NMP's VMAC accumulator is signed 24-bit.  The shared exact Conv
    # operator returns a signed-int32 PE accumulator, so narrow at the NMP
    # boundary before applying the scalar post-processing multiplier.
    return _execute_alpha_stages(
        accumulator,
        AlphaScaleStage(
            multiplier=torch.tensor(
                scale_factor,
                dtype=torch.int64,
                device=accumulator.device,
            ),
            shift=binary_point,
        ),
        AlphaBiasReluStage(
            bias=torch.tensor(
                bias_factor,
                dtype=torch.int64,
                device=accumulator.device,
            ),
            relu=relu,
        ),
        None,
        accumulator_format=_ALPHA_INT24_WRAP_FORMAT,
    )


def alpha_depthwise_conv2d_reference(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    stride: tuple[int, int],
    dilation: tuple[int, int],
    padding: tuple[int, int, int, int],
    scale_factor: int,
    binary_point: int,
    bias_factor: int,
    output_precision: int,
    relu: bool,
) -> AlphaPVS:
    """Run one biasless Alpha depthwise convolution through NMP and PVS.

    ``padding`` is ordered as top, bottom, left, right. Floating-point
    convolution is exact here because both payloads contain small integers and
    float64 exactly represents every possible depthwise accumulator.
    """
    input = torch.as_tensor(input)
    weight = torch.as_tensor(weight)
    if input.ndim != 4 or weight.ndim != 4:
        raise ValueError("Alpha depthwise input and weight must be rank 4")
    channels = input.shape[1]
    if weight.shape[:2] != (channels, 1):
        raise ValueError(
            "Alpha depthwise weight must have shape [channels, 1, r, s]"
        )
    if input.numel() == 0 or weight.numel() == 0:
        raise ValueError("Alpha depthwise payloads must be nonempty")
    if not torch.isfinite(input).all() or not torch.isfinite(weight).all():
        raise ValueError("Alpha depthwise payloads must be finite")
    if not torch.equal(input, torch.round(input)):
        raise ValueError("Alpha depthwise input payload must be integral")
    if not torch.equal(weight, torch.round(weight)):
        raise ValueError("Alpha depthwise weight payload must be integral")

    top, bottom, left, right = padding
    padded = torch.nn.functional.pad(
        input.to(torch.float64),
        (left, right, top, bottom),
    )
    accumulator = torch.nn.functional.conv2d(
        padded,
        weight.to(torch.float64),
        stride=stride,
        dilation=dilation,
        groups=channels,
    ).to(torch.int64)
    pre_pvs = alpha_nmp_epilogue(
        accumulator,
        scale_factor=scale_factor,
        binary_point=binary_point,
        bias_factor=bias_factor,
        relu=relu,
    )
    return alpha_pvs_quantize(
        pre_pvs,
        output_precision=output_precision,
        vector_dim=1,
    )


def alpha_pe_postprocess(
    accumulator: torch.Tensor,
    primary_multiplier: torch.Tensor,
    primary_shift: int,
    bias: torch.Tensor,
    secondary_multiplier: int,
    secondary_shift: int,
    relu: bool = True,
) -> torch.Tensor:
    """Execute Alpha's PE integer ordering and return its pre-PVS value.

    Inputs must already be broadcast-compatible. The Q8.8 transition is a
    bit reinterpretation, so signed-int16 narrowing happens before activation.
    The returned signed integer is consumed by either scalar output clipping or
    ``alpha_pvs_quantize``.
    """
    return _execute_alpha_stages(
        accumulator,
        AlphaScaleStage(
            multiplier=primary_multiplier,
            shift=primary_shift,
        ),
        AlphaBiasReluStage(
            bias=bias,
            relu=relu,
        ),
        AlphaScaleStage(
            multiplier=torch.tensor(
                secondary_multiplier,
                dtype=torch.int64,
                device=accumulator.device,
            ),
            shift=secondary_shift,
        ),
    )


def alpha_pe_epilogue(
    accumulator: torch.Tensor,
    primary_multiplier: torch.Tensor,
    primary_shift: int,
    bias: torch.Tensor,
    secondary_multiplier: int,
    secondary_shift: int,
    output_precision: int,
    relu: bool = True,
) -> torch.Tensor:
    """Execute Alpha's PE integer ordering through scalar O4/O8 clipping."""
    if output_precision not in (4, 8):
        raise ValueError("Alpha PE output precision must be 4 or 8")

    output = alpha_pe_postprocess(
        accumulator=accumulator,
        primary_multiplier=primary_multiplier,
        primary_shift=primary_shift,
        bias=bias,
        secondary_multiplier=secondary_multiplier,
        secondary_shift=secondary_shift,
        relu=relu,
    )
    return alpha_truncate_output(output, output_precision)


def ideal_int8_qdq_expected_codes(
    source_scale: float,
    output_scale: float,
) -> torch.Tensor:
    """Evaluate Alpha's target-native scalar INT8 QDQ contract."""
    codes = torch.arange(-128, 128, dtype=torch.int64)
    ratio = source_scale / output_scale
    return torch.clamp(
        torch.floor(codes.to(torch.float64) * ratio),
        -128,
        127,
    ).to(torch.int64)


def prove_exact_identity_dwc_encoding(
    source_scale: float,
    output_scale: float,
    *,
    relu: bool,
    context: str,
    expected_codes: torch.Tensor | None = None,
) -> tuple[int, int]:
    """Prove one NMP encoding against every possible signed-INT8 code."""
    ratio = source_scale / output_scale
    try:
        encoded = encode_nmp_multiplier(
            torch.tensor([ratio], dtype=torch.float64)
        )
    except ValueError as error:
        raise ValueError(f"{context}: {error}") from error
    binary_point = int(encoded.shift)
    codes = torch.arange(-128, 128, dtype=torch.int64)
    if expected_codes is None:
        expected = torch.floor(codes.to(torch.float64) * ratio).to(torch.int64)
        if relu:
            expected = torch.clamp_min(expected, 0)
        expected = torch.clamp(expected, -128, 127)
    else:
        expected = expected_codes.detach().cpu().reshape(-1)
        if expected.dtype != torch.int64 or expected.numel() != codes.numel():
            raise ValueError(
                f"{context}: exact code reference must contain all 256 "
                "signed-INT8 results as int64"
            )

    def matches(scale_factor: int) -> bool:
        actual = alpha_truncate_output(
            alpha_nmp_epilogue(
                codes,
                scale_factor,
                binary_point,
                0,
                relu=relu,
            ),
            8,
        ).to(torch.int64)
        return torch.equal(actual, expected)

    canonical = int(encoded.values.reshape(-1)[0].item())
    if matches(canonical):
        return canonical, binary_point
    for candidate in (canonical - 1, canonical + 1):
        if 1 <= candidate <= 65535 and matches(candidate):
            logger.info(
                "%s: exact identity DWC selected multiplier %d/%d after "
                "canonical %d/%d failed the all-code proof",
                context,
                candidate,
                binary_point,
                canonical,
                binary_point,
            )
            return candidate, binary_point
    raise ValueError(
        f"{context}: neither the canonical nor an adjacent nonzero uint16 "
        f"Alpha multiplier at binary point {binary_point} exactly implements "
        f"all signed-INT8 codes for {source_scale} -> {output_scale} "
        f"(ratio={ratio}, canonical multiplier={canonical})"
    )


def prove_exact_adaptive_average_dwc_encoding(
    source_scale: float,
    output_scale: float,
    kernel_area: int,
    *,
    context: str,
) -> tuple[int, int]:
    """Prove one NMP encoding for every nonnegative INT8 pool sum."""
    if kernel_area <= 0:
        raise ValueError(f"{context}: average-pool kernel must be nonempty")
    ratio = source_scale / (kernel_area * output_scale)
    try:
        encoded = encode_nmp_multiplier(
            torch.tensor([ratio], dtype=torch.float64)
        )
    except ValueError as error:
        raise ValueError(f"{context}: {error}") from error

    scale_factor = int(encoded.values.reshape(-1)[0].item())
    binary_point = int(encoded.shift)
    sums = torch.arange(0, 127 * kernel_area + 1, dtype=torch.int64)
    expected = torch.floor(sums.to(torch.float64) * ratio).to(torch.int64)
    expected = torch.clamp(expected, -128, 127)
    actual = alpha_truncate_output(
        alpha_nmp_epilogue(
            sums,
            scale_factor,
            binary_point,
            0,
            relu=False,
        ),
        8,
    ).to(torch.int64)
    if not torch.equal(actual, expected):
        mismatch = int(torch.nonzero(actual != expected)[0].item())
        raise ValueError(
            f"{context}: Alpha average DWC encoding is not exact for all "
            f"attainable nonnegative INT8 sums (first mismatch at "
            f"{mismatch}: hardware={int(actual[mismatch])}, "
            f"reference={int(expected[mismatch])}, ratio={ratio}, "
            f"multiplier={scale_factor}, binary_point={binary_point})"
        )
    return scale_factor, binary_point


__all__ = [
    "ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT",
    "ALPHA_DEFAULT_VSQ_FINE_ENDPOINT",
    "ALPHA_INT4_FORMAT",
    "ALPHA_INT8_FORMAT",
    "ALPHA_Q8_8_REFERENCE_FORMAT",
    "ALPHA_VSQ_MANTISSA_MAX",
    "ALPHA_VSQ_REBALANCE",
    "AlphaBiasReluStage",
    "alpha_depthwise_conv2d_reference",
    "AlphaMultiplier",
    "AlphaOutputDomain",
    "AlphaPEDirectHeadroomError",
    "AlphaPENormalization",
    "AlphaPVS",
    "AlphaRequantizationPipeline",
    "AlphaScaleStage",
    "AlphaVSQScaleEncoding",
    "alpha_int8_conv2d_accumulator",
    "alpha_nmp_epilogue",
    "alpha_output_domain",
    "alpha_pe_epilogue",
    "alpha_pe_postprocess",
    "alpha_pvs_quantize",
    "alpha_q8_8_format",
    "alpha_truncate_output",
    "alpha_vsq_conv2d_accumulator",
    "alpha_vsq_rebalance",
    "alpha_w8_accumulator_real_bound",
    "encode_alpha_nmp_pipeline",
    "encode_alpha_pe_pipeline",
    "encode_selected_alpha_pe_pipeline",
    "encode_alpha_vsq_scales",
    "encode_alpha_vsq_mantissa",
    "encode_bias",
    "encode_multiplier",
    "encode_nmp_bias",
    "encode_nmp_multiplier",
    "execute_alpha_requantization",
    "ideal_int8_qdq_expected_codes",
    "primary_factors",
    "prove_exact_adaptive_average_dwc_encoding",
    "prove_exact_identity_dwc_encoding",
    "reconstruct_accumulator_bias",
    "select_alpha_pe_normalization",
]

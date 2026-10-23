# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Target-neutral fixed-point reference and QAT arithmetic."""

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Union

import torch

__all__ = [
    "FixedPointFakeRequantize",
    "FixedPointFormat",
    "FixedPointRounding",
    "FixedPointScalePolicy",
    "quantize_fixed_point",
    "reconstruct_fixed_point",
]


class FixedPointRounding(str, Enum):
    """Rounding applied before storing a fixed-point integer payload."""

    NEAREST_EVEN = "nearest_even"
    NEAREST_AWAY_FROM_ZERO = "nearest_away_from_zero"
    NEAREST_TOWARD_POSITIVE_INFINITY = "nearest_toward_positive_infinity"
    TOWARD_ZERO = "toward_zero"


class FixedPointScalePolicy(str, Enum):
    """Policy that supplies the binary point during fake requantization."""

    FIXED_BINARY_POINT = "fixed_binary_point"
    LEARNED_ROUNDED_BINARY_POINT = "learned_rounded_binary_point"


@dataclass(frozen=True)
class FixedPointFormat:
    """Logical fixed-point format independent of a compilation target.

    ``bit_width`` defines the logical integer range while ``storage_dtype``
    defines the PyTorch container used for the integer payload. Logical values
    are divided by ``value_scale`` before applying the binary point. Thus, the
    logical quantization step is ``value_scale * 2**-fractional_bits``. When
    ``saturate`` is false, overflow wraps modulo the logical bit width.
    """

    storage_dtype: Union[torch.dtype, str]
    bit_width: int
    fractional_bits: int
    value_scale: float
    signed: bool
    rounding: Union[FixedPointRounding, str]
    saturate: bool

    def __post_init__(self) -> None:
        storage_dtype = self.storage_dtype
        if isinstance(storage_dtype, str):
            storage_dtype = getattr(torch, storage_dtype, None)
        try:
            storage_info = torch.iinfo(storage_dtype)
        except TypeError as error:
            raise ValueError(
                "storage_dtype must be an integer torch dtype"
            ) from error

        if not isinstance(self.bit_width, int) or self.bit_width <= 0:
            raise ValueError("bit_width must be a positive integer")
        if not isinstance(self.fractional_bits, int):
            raise ValueError("fractional_bits must be an integer")
        if self.fractional_bits < 0:
            raise ValueError("fractional_bits must be non-negative")
        if (
            not isinstance(self.value_scale, Real)
            or isinstance(self.value_scale, bool)
            or not math.isfinite(self.value_scale)
            or self.value_scale <= 0.0
        ):
            raise ValueError("value_scale must be finite and positive")
        if not isinstance(self.signed, bool):
            raise ValueError("signed must be a bool")
        if not isinstance(self.saturate, bool):
            raise ValueError("saturate must be a bool")

        try:
            rounding = FixedPointRounding(self.rounding)
        except ValueError as error:
            raise ValueError(
                f"unsupported fixed-point rounding: {self.rounding}"
            ) from error

        qmin, qmax = _integer_range(self.bit_width, self.signed)
        if qmin < storage_info.min or qmax > storage_info.max:
            raise ValueError(
                f"{storage_dtype} cannot store the logical "
                f"[{qmin}, {qmax}] range"
            )
        logical_scale = self.value_scale * math.ldexp(
            1.0, -self.fractional_bits
        )
        if logical_scale == 0.0:
            raise ValueError(
                "fractional_bits produces an unrepresentable scale"
            )

        object.__setattr__(self, "storage_dtype", storage_dtype)
        object.__setattr__(self, "rounding", rounding)
        object.__setattr__(self, "value_scale", float(self.value_scale))

    @property
    def qmin(self) -> int:
        return _integer_range(self.bit_width, self.signed)[0]

    @property
    def qmax(self) -> int:
        return _integer_range(self.bit_width, self.signed)[1]

    @property
    def scale(self) -> float:
        return self.value_scale * math.ldexp(1.0, -self.fractional_bits)


def _integer_range(bit_width: int, signed: bool) -> tuple[int, int]:
    if signed:
        return -(2 ** (bit_width - 1)), 2 ** (bit_width - 1) - 1
    return 0, 2**bit_width - 1


def _working_dtype(tensor: torch.Tensor) -> torch.dtype:
    if tensor.dtype == torch.float64:
        return torch.float64
    return torch.float32


def _round(values: torch.Tensor, mode: FixedPointRounding) -> torch.Tensor:
    if mode == FixedPointRounding.NEAREST_EVEN:
        return torch.round(values)
    if mode == FixedPointRounding.NEAREST_AWAY_FROM_ZERO:
        return torch.sign(values) * torch.floor(torch.abs(values) + 0.5)
    if mode == FixedPointRounding.NEAREST_TOWARD_POSITIVE_INFINITY:
        return torch.floor(values + 0.5)
    if mode == FixedPointRounding.TOWARD_ZERO:
        return torch.trunc(values)
    raise AssertionError(f"unhandled fixed-point rounding mode: {mode}")


def _bound_payload(
    rounded: torch.Tensor,
    fixed_format: FixedPointFormat,
) -> torch.Tensor:
    if fixed_format.saturate:
        return torch.clamp(rounded, fixed_format.qmin, fixed_format.qmax)

    span = 2**fixed_format.bit_width
    return (
        torch.remainder(rounded - fixed_format.qmin, span) + fixed_format.qmin
    )


def _quantize_with_scale(
    values: torch.Tensor,
    scale: torch.Tensor,
    fixed_format: FixedPointFormat,
) -> tuple[torch.Tensor, torch.Tensor]:
    work_values = values.to(_working_dtype(values))
    work_scale = scale.to(device=values.device, dtype=work_values.dtype)
    scaled = work_values / work_scale
    rounded = _round(scaled, fixed_format.rounding)
    return _bound_payload(rounded, fixed_format), scaled


def quantize_fixed_point(
    values: torch.Tensor,
    fixed_format: FixedPointFormat,
) -> torch.Tensor:
    """Quantize real values into the format's integer storage payload."""

    if not values.is_floating_point():
        raise ValueError("fixed-point quantization expects a floating tensor")
    scale = values.new_tensor(fixed_format.scale, dtype=_working_dtype(values))
    payload, _ = _quantize_with_scale(values, scale, fixed_format)
    return payload.to(fixed_format.storage_dtype)


def reconstruct_fixed_point(
    payload: torch.Tensor,
    fixed_format: FixedPointFormat,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reconstruct real values from a fixed-point integer payload."""

    if payload.dtype != fixed_format.storage_dtype:
        raise ValueError(
            "payload dtype does not match the fixed-point storage dtype"
        )
    if not torch.empty((), dtype=output_dtype).is_floating_point():
        raise ValueError("output_dtype must be floating point")
    return payload.to(output_dtype) * fixed_format.scale


class _FixedPointFakeRequantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: torch.Tensor,
        scale: torch.Tensor,
        fixed_format: FixedPointFormat,
    ) -> torch.Tensor:
        payload, scaled = _quantize_with_scale(values, scale, fixed_format)
        ctx.save_for_backward(payload, scaled, scale)
        ctx.saturate = fixed_format.saturate
        ctx.qmin = fixed_format.qmin
        ctx.qmax = fixed_format.qmax
        return (payload * scale).to(values.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        payload, scaled, scale = ctx.saved_tensors
        work_grad = grad_output.to(payload.dtype)

        if ctx.saturate:
            scale_gradient = torch.where(
                scaled < ctx.qmin,
                torch.full_like(scaled, ctx.qmin),
                torch.where(
                    scaled > ctx.qmax,
                    torch.full_like(scaled, ctx.qmax),
                    payload - scaled,
                ),
            )
        else:
            scale_gradient = payload - scaled

        normalizer = max(scaled.numel() * max(abs(ctx.qmin), ctx.qmax), 1)
        scale_gradient = (work_grad * scale_gradient).sum()
        scale_gradient = scale_gradient / math.sqrt(normalizer)
        return grad_output, scale_gradient.reshape_as(scale), None


class FixedPointFakeRequantize(torch.nn.Module):
    """Executable fixed-point fake requantization for QAT.

    Rounding and overflow exactly follow ``fixed_format``. The input gradient
    uses a straight-through estimator. A learned policy keeps the effective
    binary point integral and uses an STE through its nearest-even rounding;
    it never introduces a generic PT2E quantization scheme.
    """

    def __init__(
        self,
        fixed_format: FixedPointFormat,
        scale_policy: Union[FixedPointScalePolicy, str],
    ) -> None:
        super().__init__()
        self.fixed_format = fixed_format
        try:
            self.scale_policy = FixedPointScalePolicy(scale_policy)
        except ValueError as error:
            raise ValueError(
                f"unsupported fixed-point scale policy: {scale_policy}"
            ) from error

        binary_point = torch.tensor(float(fixed_format.fractional_bits))
        if (
            self.scale_policy
            == FixedPointScalePolicy.LEARNED_ROUNDED_BINARY_POINT
        ):
            self.binary_point = torch.nn.Parameter(binary_point)
        else:
            self.register_buffer("binary_point", binary_point)

    def effective_binary_point(self) -> torch.Tensor:
        if self.scale_policy == FixedPointScalePolicy.FIXED_BINARY_POINT:
            return self.binary_point
        rounded = torch.round(self.binary_point)
        return self.binary_point + (rounded - self.binary_point).detach()

    def effective_scale(self) -> torch.Tensor:
        return self.fixed_format.value_scale * torch.exp2(
            -self.effective_binary_point()
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not values.is_floating_point():
            raise ValueError("fixed-point fake requantization expects floats")
        return _FixedPointFakeRequantizeSTE.apply(
            values,
            self.effective_scale(),
            self.fixed_format,
        )

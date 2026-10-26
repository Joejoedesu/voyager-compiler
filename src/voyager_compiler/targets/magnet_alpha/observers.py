# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
import math
import re
from typing import List, Optional, Union

import torch
from torchao.quantization.pt2e import FakeQuantizeBase

from voyager_compiler.quantization.mx_utils import _reshape_to_blocks
from voyager_compiler.quantization.qspec import QScheme
from voyager_compiler.targets.magnet_alpha.quant_ops import (
    expand,
    quantize_mx,
    vmap,
)

__all__ = [
    "FusedAmaxObsFakeQuantize",
    "get_vsq_scale_quantization_map",
    "_DerivedObserverOrFakeQuantize",
]


logger = logging.getLogger(__name__)

DEFAULT_QUANTIZATION_ROUNDING = "nearest"
SUPPORTED_QUANTIZATION_ROUNDING = frozenset(
    (DEFAULT_QUANTIZATION_ROUNDING, "floor")
)


def _scale_from_amax(
    amax: torch.Tensor,
    quant_max: float,
    fallback: torch.Tensor,
    force_scale_power_of_two: bool,
) -> torch.Tensor:
    scale = amax / quant_max
    scale = torch.where(amax > 0.0, scale, fallback)
    scale = torch.where(torch.isfinite(amax), scale, fallback)
    if force_scale_power_of_two:
        scale = torch.pow(2, torch.ceil(torch.log2(scale)))
    return scale


def get_quantization_map(
    dtype,
    device=None,
    quant_min=None,
    quant_max=None,
    rounding=DEFAULT_QUANTIZATION_ROUNDING,
):
    """Return a quantization map for the given dtype."""
    if rounding not in SUPPORTED_QUANTIZATION_ROUNDING:
        raise ValueError(
            f"rounding must be one of {sorted(SUPPORTED_QUANTIZATION_ROUNDING)}"
        )
    round_values = torch.floor if rounding == "floor" else torch.round
    values = torch.arange(2**16, dtype=torch.int16, device=device).view(
        torch.bfloat16
    )
    if dtype is None:
        return values

    # PyTorch native dtypes
    if dtype in ["float32", "float16", "bfloat16"]:
        torch_dtype = getattr(torch, dtype)
        return values.to(torch_dtype).to(torch.bfloat16)

    # Integer data types
    if match := re.fullmatch(r"int(\d+)", dtype, re.IGNORECASE):
        nbits = int(match.group(1))
        storage_min, storage_max = -(2 ** (nbits - 1)), 2 ** (nbits - 1) - 1
        quant_min = storage_min if quant_min is None else quant_min
        quant_max = storage_max if quant_max is None else quant_max
        _validate_integer_bounds(
            dtype,
            quant_min,
            quant_max,
            storage_min,
            storage_max,
        )
        return torch.clamp(round_values(values), quant_min, quant_max)

    # Unsigned integer
    if match := re.fullmatch(r"uint(\d+)", dtype, re.IGNORECASE):
        nbits = int(match.group(1))
        storage_min, storage_max = 0, 2**nbits - 1
        quant_min = storage_min if quant_min is None else quant_min
        quant_max = storage_max if quant_max is None else quant_max
        _validate_integer_bounds(
            dtype,
            quant_min,
            quant_max,
            storage_min,
            storage_max,
        )
        return torch.clamp(round_values(values), quant_min, quant_max)

    from voyager_compiler.quantization.fake_quantize import (
        get_quantization_map as shared_map,
    )

    if rounding != "nearest" or quant_min is not None or quant_max is not None:
        raise ValueError(
            "custom Alpha rounding and bounds require integer storage"
        )
    return shared_map(dtype, device)


def get_vsq_scale_quantization_map(
    dtype,
    device,
    fine_quant_max,
    coarse_scale_quant_max,
):
    """Return a normalized VSQ fine-scale codebook.

    Integer fine-scale storage represents a fraction of the full logical
    fine-scale range, rather than an integer-valued logical scale.
    """
    match = re.fullmatch(r"uint(\d+)", dtype or "", re.IGNORECASE)
    if match is None:
        return get_quantization_map(dtype, device)

    storage_max = 2 ** int(match.group(1)) - 1
    for value, name in (
        (fine_quant_max, "fine_quant_max"),
        (coarse_scale_quant_max, "coarse_scale_quant_max"),
    ):
        if value is None or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    step = coarse_scale_quant_max / (fine_quant_max * storage_max)
    values = torch.arange(
        2**16,
        dtype=torch.int16,
        device=device,
    ).view(torch.bfloat16)
    codes = torch.floor(values.to(torch.float32) / step)
    codes = torch.clamp(codes, 1, storage_max)
    return codes * step


def _validate_integer_bounds(
    dtype,
    quant_min,
    quant_max,
    storage_min,
    storage_max,
):
    bounds = (quant_min, quant_max)
    if any(
        not math.isfinite(bound) or bound != round(bound) for bound in bounds
    ):
        raise ValueError(f"{dtype} quantization bounds must be finite integers")
    if quant_min > quant_max:
        raise ValueError(f"{dtype} quant_min must not exceed quant_max")
    if quant_min < storage_min or quant_max > storage_max:
        raise ValueError(
            f"{dtype} quantization bounds [{quant_min}, {quant_max}] "
            f"exceed storage range [{storage_min}, {storage_max}]"
        )


class MXFakeQuantFunction(torch.autograd.Function):
    """This function performs MX quantization by calculating the scaling
    factor using absolute maximum values in the input.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        fake_quant_enabled: torch.Tensor,
        scale: torch.Tensor,
        qmap: torch.Tensor,
        axes: Union[int, List[int]],
        block_size: Union[int, List[int]],
        quant_max: float,
        force_scale_power_of_two=False,
        scale_qmap=None,
    ):
        if fake_quant_enabled[0] == 0:
            return input
        sf, input = quantize_mx(
            input,
            qmap,
            axes,
            block_size,
            quant_max,
            force_scale_power_of_two,
            scale_qmap=scale_qmap,
        )
        scale.resize_(sf.shape).copy_(sf)
        input = input * expand(sf, input.shape, block_size)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None, None, None


class GroupWiseAffineFakeQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        fake_quant_enabled: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        axes: Union[int, List[int]],
        block_size: Union[int, List[int]],
        quant_min: float,
        quant_max: float,
        scale_qmap=None,
    ):
        if fake_quant_enabled[0] == 0:
            return input

        assert block_size > 0

        # Make sure axes is a list of non-negative numbers
        axes = [axes] if type(axes) == int else axes
        axes = [x + input.ndim if x < 0 else x for x in axes]

        # Perform tiling to the hardware vector size
        reshaped_input, axes, orig_shape, padded_shape = _reshape_to_blocks(
            input, axes, block_size
        )

        shared_exp_axes = [x + 1 for x in axes]

        min = torch.amin(reshaped_input, dim=shared_exp_axes)
        max = torch.amax(reshaped_input, dim=shared_exp_axes)

        sf = (max - min) / (quant_max - quant_min)
        sf = torch.where(sf > 0.0, sf, 1.0)
        zp = -min / sf + quant_min

        # Quantize the scale using the codebook
        if scale_qmap is not None:
            sf = vmap(sf, scale_qmap)
            zp = vmap(zp, scale_qmap)

        scale.resize_(sf.shape).copy_(sf)
        zero_point.resize_(zp.shape).copy_(zp)

        sf = expand(sf, input.shape, block_size)
        zp = expand(zp, input.shape, block_size)

        # Quantize
        q = torch.clamp(torch.round(input / sf + zp), quant_min, quant_max)

        # Dequantize
        input = (q - zp) * sf

        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None, None, None, None


class FusedAmaxObsFakeQuantFunction(torch.autograd.Function):
    """This function observes the amax statistics of inputs and
    quantize the inputs based on the observed amax values.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        observer_enabled: torch.Tensor,
        fake_quant_enabled: torch.Tensor,
        qmap: torch.Tensor,
        amax_history: torch.Tensor,
        scale: torch.Tensor,
        amax_history_len: int,
        quant_max: float,
        ch_axis: Optional[int] = None,
        per_row_fake_quant=False,
        force_scale_power_of_two=False,
        rounding=DEFAULT_QUANTIZATION_ROUNDING,
        quant_min=None,
        storage_quant_max=None,
    ) -> torch.Tensor:
        if observer_enabled[0] == 1:
            if per_row_fake_quant:
                ch_axis = ch_axis + input.ndim if ch_axis < 0 else ch_axis
                dim = tuple(i for i in range(input.ndim) if i != ch_axis)
                amax_cur = torch.amax(torch.abs(input), dim=dim, keepdim=True)
            else:
                amax_cur = torch.amax(torch.abs(input))

            if amax_history.numel() == 0:
                size = amax_cur.shape
                amax_history.resize_((amax_history_len,) + size).fill_(0.0)
                scale.resize_(size).fill_(1.0)

            amax = torch.amax(amax_history, dim=0)

            if amax_history.shape[0] > 1:
                new_amax_history = torch.roll(amax_history, -1, 0)
                amax_history.copy_(new_amax_history)
            amax_history[0] = amax_cur

            sf = _scale_from_amax(
                amax,
                quant_max,
                scale,
                force_scale_power_of_two,
            )
            scale.copy_(sf)

        if fake_quant_enabled[0] == 1:
            scale = scale.to(input.dtype)
            normalized = input / scale
            if rounding == "floor":
                if quant_min is None or storage_quant_max is None:
                    raise ValueError(
                        "floor quantization requires storage bounds"
                    )
                normalized = torch.clamp(
                    torch.floor(normalized),
                    quant_min,
                    storage_quant_max,
                )
            else:
                normalized = vmap(normalized, qmap)
            input = normalized * scale

        return input

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FusedAmaxObsFakeQuantize(FakeQuantizeBase):
    r"""Simulate the quantize and dequantize operations in training time.

    Observer module for computing the quantization parameters based on the
    historical amax values.

    This observer uses the tensor amax statistics to compute the quantization
    parameters. The module records the maximums of absolute values of output
    tensors, and uses this statistic to compute the quantization parameters.
    """

    qmap: torch.Tensor
    scale_qmap: Optional[torch.Tensor]
    amax_history: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(
        self,
        dtype: str,
        qscheme: Optional[str] = None,
        quant_min: Optional[float] = None,
        quant_max: Optional[float] = None,
        scale_quant_max: Optional[float] = None,
        rounding: str = DEFAULT_QUANTIZATION_ROUNDING,
        amax_history_len: int = None,
        ch_axis: Optional[int] = None,
        block_size: Optional[int] = None,
        scale_dtype: Optional[str] = None,
        force_scale_power_of_two: bool = False,
        record_histogram: bool = False,
        outlier_threshold: Optional[float] = None,
        outlier_pct: Optional[float] = None,
        coarse_dtype: Optional[str] = None,
        coarse_quant_min: Optional[float] = None,
        coarse_quant_max: Optional[float] = None,
        coarse_scale_quant_max: Optional[float] = None,
        coarse_qscheme=None,
        coarse_ch_axis: Optional[int] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.scale_quant_max = (
            quant_max if scale_quant_max is None else scale_quant_max
        )
        self.rounding = rounding
        self.amax_history_len = amax_history_len
        self.ch_axis = ch_axis
        self.block_size = block_size
        self.scale_dtype = scale_dtype
        self.force_scale_power_of_two = force_scale_power_of_two
        self.record_histogram = record_histogram
        self.outlier_threshold = outlier_threshold
        self.outlier_pct = outlier_pct
        self.coarse_dtype = coarse_dtype
        self.coarse_quant_min = coarse_quant_min
        self.coarse_quant_max = coarse_quant_max
        self.coarse_scale_quant_max = (
            coarse_quant_max
            if coarse_scale_quant_max is None
            else coarse_scale_quant_max
        )
        self.coarse_qscheme = coarse_qscheme
        self.coarse_ch_axis = coarse_ch_axis
        assert (
            outlier_pct is None or outlier_threshold is None
        ), "Only one of outlier_pct and outlier_threshold can be set."
        self.outlier_max_pct = 0.0

        factory_kwargs = {"device": device, "dtype": torch.float}

        # Generate quantization map buffers
        quant_map = get_quantization_map(
            dtype,
            device,
            quant_min,
            quant_max,
            rounding,
        )
        if isinstance(quant_map, tuple):
            indices, values = quant_map
            quant_map = values[indices]
        self.register_buffer("qmap", quant_map, persistent=False)

        coarse_quant_map = (
            get_quantization_map(
                coarse_dtype,
                device,
                coarse_quant_min,
                coarse_quant_max,
            )
            if coarse_dtype is not None
            else None
        )
        if isinstance(coarse_quant_map, tuple):
            indices, values = coarse_quant_map
            coarse_quant_map = values[indices]
        self.register_buffer("coarse_qmap", coarse_quant_map, persistent=False)

        if self.scale_dtype is None:
            scale_map = None
        elif self.qscheme == QScheme.VSQ:
            scale_map = get_vsq_scale_quantization_map(
                self.scale_dtype,
                device,
                self.scale_quant_max,
                self.coarse_scale_quant_max,
            )
        else:
            scale_map = get_quantization_map(self.scale_dtype, device)
        self.register_buffer("scale_qmap", scale_map, persistent=False)

        # Create amax history and scale buffers
        self.register_buffer("amax_history", torch.tensor([], **factory_kwargs))
        self.register_buffer("scale", torch.tensor([1.0], **factory_kwargs))
        fine_scale = (
            torch.tensor([1.0], **factory_kwargs)
            if self.qscheme == QScheme.VSQ
            else None
        )
        self.register_buffer("fine_scale", fine_scale)
        self.register_buffer(
            "zero_point", torch.tensor([1.0], **factory_kwargs)
        )
        self.register_buffer(
            "static_vsq_payload",
            torch.empty(0, **factory_kwargs),
        )
        self.register_buffer(
            "static_vsq_fine_scale",
            torch.empty(0, **factory_kwargs),
        )
        self.register_buffer(
            "static_vsq_coarse_scale",
            torch.empty(0, **factory_kwargs),
        )

        # Create histogram buffer
        if self.record_histogram:
            self.register_buffer(
                "histogram",
                torch.zeros(254, **factory_kwargs),
                persistent=False,
            )

        self.observer_enabled[0] = self.qscheme is not None
        self.is_per_channel = self.qscheme == QScheme.PER_CHANNEL_SYMMETRIC
        self.coarse_is_per_channel = (
            self.coarse_qscheme == QScheme.PER_CHANNEL_SYMMETRIC
        )

        if outlier_pct is not None:
            self.outlier_ema_decay = kwargs.get("outlier_ema_decay", 0.9)
            self.register_buffer(
                "outlier_threshold_history", torch.tensor(0.0, **factory_kwargs)
            )
            self.register_buffer(
                "outlier_initialized",
                torch.tensor(False, device=device),
            )

    @torch.jit.export
    def calculate_qparams(self):
        if self.static_vsq_coarse_scale.numel() > 0:
            return self.static_vsq_coarse_scale
        if self.qscheme == QScheme.GROUP_WISE_AFFINE:
            return self.scale, self.zero_point

        if self.amax_history.numel() > 0:
            quant_max = (
                self.coarse_scale_quant_max
                if self.qscheme == QScheme.VSQ
                else self.scale_quant_max
            )
            amax = torch.amax(self.amax_history, dim=0)
            sf = _scale_from_amax(
                amax,
                quant_max,
                self.scale,
                self.force_scale_power_of_two,
            )
            self.scale.copy_(sf)

        return self.scale

    def reset_observer_statistics(self) -> None:
        """Clear dynamic amax state without changing observer enablement."""
        if self.static_vsq_payload.numel() > 0:
            raise ValueError(
                "Static VSQ components cannot be reset for live observation"
            )
        self.amax_history.resize_(0)
        self.scale.resize_(1).fill_(1.0)

    def _update_qparams_from(self, values: torch.Tensor) -> None:
        """Observe live values once and restore the prior observer state."""
        if self.static_vsq_payload.numel() > 0:
            raise ValueError("Static VSQ components cannot observe live values")
        observer_enabled = bool(self.observer_enabled[0].item())
        try:
            self.enable_observer()
            with torch.no_grad():
                self(values.detach())
                self.calculate_qparams()
        finally:
            if not observer_enabled:
                self.disable_observer()

    def rebase_qparams_from(self, values: torch.Tensor) -> None:
        """Update qparams from live values, including an empty-history seed."""
        needs_seed = self.amax_history.numel() == 0
        # The fused observer first seeds an empty history, then derives the
        # scale from that history on its next invocation.
        self._update_qparams_from(values)
        if needs_seed:
            self._update_qparams_from(values)

    def set_static_vsq_components(
        self,
        payload: torch.Tensor,
        fine_scale: torch.Tensor,
        coarse_scale: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
    ) -> None:
        """Freeze a finalized VSQ value instead of re-observing a parameter."""
        if self.qscheme != QScheme.VSQ:
            raise ValueError("Static VSQ components require a VSQ observer")
        if payload.numel() == 0 or payload.ndim == 0:
            raise ValueError("Static VSQ payload must be a nonempty tensor")
        if reference is not None and payload.shape != reference.shape:
            raise ValueError("Static VSQ payload and reference shapes differ")
        if not torch.isfinite(payload).all():
            raise ValueError("Static VSQ payload must be finite")
        if not torch.equal(payload, torch.round(payload)):
            raise ValueError("Static VSQ payload must contain integers")
        if torch.any(payload < self.quant_min) or torch.any(
            payload > self.quant_max
        ):
            raise ValueError("Static VSQ payload exceeds quantization bounds")
        for value, name in (
            (fine_scale, "fine scale"),
            (coarse_scale, "coarse scale"),
        ):
            if value.numel() == 0 or not torch.isfinite(value).all():
                raise ValueError(f"Static VSQ {name} must be finite")
            if torch.any(value <= 0):
                raise ValueError(f"Static VSQ {name} must be positive")

        reconstructed = (
            payload
            * expand(
                fine_scale,
                payload.shape,
                self.block_size,
            )
            * coarse_scale
        )
        if reference is not None and not torch.allclose(
            reconstructed.to(torch.float64),
            reference.detach().to(torch.float64),
            rtol=1e-5,
            atol=1e-6,
        ):
            maximum_error = torch.amax(
                torch.abs(
                    reconstructed.to(torch.float64)
                    - reference.detach().to(torch.float64)
                )
            )
            raise ValueError(
                "Static VSQ components do not reconstruct the reference; "
                f"maximum error is {float(maximum_error)}"
            )

        self.static_vsq_payload = payload.detach().clone()
        self.static_vsq_fine_scale = fine_scale.detach().clone()
        self.static_vsq_coarse_scale = coarse_scale.detach().clone()
        self.scale.resize_(coarse_scale.shape).copy_(coarse_scale)
        self.disable_observer()

    def quantize_vsq_components(
        self,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return finalized static components or perform ordinary VSQ."""
        if self.qscheme != QScheme.VSQ:
            raise ValueError("VSQ components require a VSQ observer")
        if self.static_vsq_payload.numel() > 0:
            if tuple(values.shape) != tuple(self.static_vsq_payload.shape):
                raise ValueError(
                    "Static VSQ payload shape does not match the input"
                )
            return (
                self.static_vsq_coarse_scale.to(
                    device=values.device,
                    dtype=values.dtype,
                ),
                self.static_vsq_fine_scale.to(
                    device=values.device,
                    dtype=values.dtype,
                ),
                self.static_vsq_payload.to(
                    device=values.device,
                    dtype=values.dtype,
                ),
            )

        axes = self.ch_axis
        axes = (axes,) if isinstance(axes, int) else tuple(axes)
        return torch.ops.alpha_quant.quantize_vsq.default(
            values,
            self.calculate_qparams().to(
                device=values.device,
                dtype=values.dtype,
            ),
            self.coarse_qmap.to(values.device),
            self.qmap.to(values.device),
            axes,
            self.block_size,
            self.scale_quant_max,
            self.force_scale_power_of_two,
            (
                self.scale_qmap.to(values.device)
                if self.scale_qmap is not None
                else None
            ),
            None,
            self.coarse_scale_quant_max,
        )

    @torch.jit.export
    def extra_repr(self):
        return (
            "fake_quant_enabled={}, observer_enabled={}, dtype={}, amax_history_len={}, "
            "quant_max={}, qscheme={}, ch_axis={}, block_size={}, force_scale_power_of_two={}, "
            "scale={}".format(
                self.fake_quant_enabled,
                self.observer_enabled,
                self.dtype,
                self.amax_history_len,
                self.quant_max,
                self.qscheme,
                self.ch_axis,
                self.block_size,
                self.force_scale_power_of_two,
                self.scale,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.record_histogram:
            exp = torch.floor(torch.log2(torch.abs(x.detach().float())))
            self.histogram += torch.histc(exp, 254, min=-126, max=127)

        if self.outlier_pct is not None:
            q = 1.0 - self.outlier_pct
            value = torch.quantile(x.abs().float(), q)

            if not self.outlier_initialized:
                self.outlier_threshold_history.copy_(value)
                self.outlier_initialized.fill_(True)
            else:
                self.outlier_threshold_history.mul_(
                    self.outlier_ema_decay
                ).add_(value * (1.0 - self.outlier_ema_decay))

            self.outlier_threshold = self.outlier_threshold_history.item()

        # Remove outliers from the activation.
        if self.outlier_threshold is not None:
            orig_x = x.clone()
            mask = torch.abs(x) < self.outlier_threshold
            x = torch.where(mask, x, torch.zeros_like(x))

            outlier_pct = mask.bitwise_not().sum().item() / x.numel()
            self.outlier_max_pct = max(outlier_pct, self.outlier_max_pct)

        if self.qscheme == QScheme.VSQ and self.static_vsq_payload.numel() > 0:
            if self.fake_quant_enabled[0] == 1:
                coarse, fine, payload = self.quantize_vsq_components(x)
                x = (
                    payload
                    * expand(
                        fine,
                        payload.shape,
                        self.block_size,
                    )
                    * coarse
                )
        elif self.qscheme == QScheme.VSQ:
            coarse_dequantized = FusedAmaxObsFakeQuantFunction.apply(
                x,
                self.observer_enabled,
                self.fake_quant_enabled,
                self.coarse_qmap,
                self.amax_history,
                self.scale,
                self.amax_history_len,
                self.coarse_scale_quant_max,
                self.coarse_ch_axis,
                self.coarse_is_per_channel,
                self.force_scale_power_of_two,
                self.rounding,
                self.coarse_quant_min,
                self.coarse_quant_max,
            )
            coarse_data = coarse_dequantized / self.scale.to(x.dtype)
            fine_dequantized = MXFakeQuantFunction.apply(
                coarse_data,
                self.fake_quant_enabled,
                self.fine_scale,
                self.qmap,
                self.ch_axis,
                self.block_size,
                self.scale_quant_max,
                self.force_scale_power_of_two,
                self.scale_qmap,
            )
            x = fine_dequantized * self.scale.to(x.dtype)
        elif self.qscheme == QScheme.MICROSCALING:
            x = MXFakeQuantFunction.apply(
                x,
                self.fake_quant_enabled,
                self.scale,
                self.qmap,
                self.ch_axis,
                self.block_size,
                self.scale_quant_max,
                self.force_scale_power_of_two,
                self.scale_qmap,
            )
        elif self.qscheme == QScheme.GROUP_WISE_AFFINE:
            x = GroupWiseAffineFakeQuantFunction.apply(
                x,
                self.fake_quant_enabled,
                self.scale,
                self.zero_point,
                self.ch_axis,
                self.block_size,
                self.quant_min,
                self.quant_max,
                self.scale_qmap,
            )
        else:
            x = FusedAmaxObsFakeQuantFunction.apply(
                x,
                self.observer_enabled,
                self.fake_quant_enabled,
                self.qmap,
                self.amax_history,
                self.scale,
                self.amax_history_len,
                self.scale_quant_max,
                self.ch_axis,
                self.is_per_channel,
                self.force_scale_power_of_two,
                self.rounding,
                self.quant_min,
                self.quant_max,
            )

        # Restore outliers
        if self.outlier_threshold is not None:
            x = torch.where(mask, x, orig_x)

        return x

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Removing this function throws an error that the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = [
            "scale",
            "amax_history",
            "static_vsq_payload",
            "static_vsq_fine_scale",
            "static_vsq_coarse_scale",
        ]
        if self.fine_scale is not None:
            local_state.append("fine_scale")
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and amax_history
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                buffer = getattr(self, name)
                buffer.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():
                    buffer.copy_(val)
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


from voyager_compiler.quantization.fake_quantize import (
    _DerivedObserverOrFakeQuantize,
)

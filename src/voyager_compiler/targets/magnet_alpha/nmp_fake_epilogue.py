# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Calibration-time exact simulation of the Magnet Alpha NMP epilogue."""

import torch

from voyager_compiler.targets.magnet_alpha.fixed_point import (
    ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    alpha_int8_conv2d_accumulator,
    alpha_nmp_epilogue,
    alpha_output_domain,
    alpha_pvs_quantize,
    alpha_truncate_output,
    encode_alpha_nmp_pipeline,
)
from voyager_compiler.targets.magnet_alpha.observers import (
    FusedAmaxObsFakeQuantize,
)
from voyager_compiler.targets.magnet_alpha.quant_ops import expand


class MagnetAlphaNMPFakeEpilogue(torch.nn.Module):
    """Two-stage calibration reference for Alpha depthwise execution."""

    def __init__(
        self,
        *,
        lane_width: int,
        stride,
        padding,
        dilation,
        groups: int,
    ) -> None:
        super().__init__()
        if lane_width <= 0:
            raise ValueError("Magnet Alpha NMP lane width must be positive")
        self.lane_width = int(lane_width)
        self.stride = self._pair(stride)
        self.padding = self._pair(padding)
        self.dilation = self._pair(dilation)
        self.groups = int(groups)
        self.register_buffer("input_scale", torch.ones(1))
        self.register_buffer("weight_scale", torch.ones(1))
        self.register_buffer("input_qmap", torch.empty(0))
        self.register_buffer("weight_qmap", torch.empty(0))
        self.register_buffer("output_scale", torch.ones(1))
        self.exact_enabled = False
        self.output_precision = 8
        self.uses_pvs = False
        self.primary_multiplier = 1
        self.primary_shift = 0
        self.bias_payload = 0
        self.relu = False

    @staticmethod
    def _pair(value) -> tuple[int, int]:
        if isinstance(value, int):
            return (value, value)
        pair = tuple(int(item) for item in value)
        if len(pair) != 2:
            raise ValueError(
                "Magnet Alpha NMP Conv2d arguments must have two values"
            )
        return pair

    def configure_exact(
        self,
        input_observer: FusedAmaxObsFakeQuantize,
        weight_observer: FusedAmaxObsFakeQuantize,
        real_bias: torch.Tensor,
        output_scale: torch.Tensor,
        *,
        output_precision: int,
        uses_pvs: bool,
        relu: bool,
        fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
        coarse_scale_quant_max: float = (ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT),
    ) -> None:
        """Configure finalized W8 DWC operands for a second pass."""
        input_scale = input_observer.calculate_qparams().detach().reshape(-1)
        weight_scale = weight_observer.calculate_qparams().detach().reshape(-1)
        if input_scale.numel() != 1 or weight_scale.numel() != 1:
            raise ValueError(
                "Magnet Alpha NMP requires per-tensor input and weight scales"
            )
        if (
            not torch.isfinite(input_scale).all()
            or not torch.isfinite(weight_scale).all()
            or input_scale[0] <= 0
            or weight_scale[0] <= 0
        ):
            raise ValueError(
                "Magnet Alpha NMP scales must be finite and positive"
            )
        output = alpha_output_domain(
            output_scale,
            output_precision,
            uses_pvs,
            fine_quant_max=fine_quant_max,
            coarse_scale_quant_max=coarse_scale_quant_max,
        )
        accumulator_scale = input_scale[0] * weight_scale[0]
        accumulator_bias = torch.round(
            real_bias.detach().reshape(-1) / accumulator_scale
        )
        pipeline = encode_alpha_nmp_pipeline(
            accumulator_scale.reshape(1),
            accumulator_bias,
            output,
            relu=relu,
        )
        assert pipeline.bias_relu.bias is not None

        self.input_scale = input_scale.clone()
        self.weight_scale = weight_scale.clone()
        self.input_qmap = input_observer.qmap.detach().clone()
        self.weight_qmap = weight_observer.qmap.detach().clone()
        self.output_scale = pipeline.output.scale.clone()
        self.primary_multiplier = int(pipeline.primary.multiplier[0].item())
        self.primary_shift = pipeline.primary.shift
        self.bias_payload = int(pipeline.bias_relu.bias[0].item())
        self.output_precision = pipeline.output.precision
        self.uses_pvs = pipeline.output.uses_pvs
        self.relu = pipeline.bias_relu.relu
        self.exact_enabled = True

    @staticmethod
    def _quantize_w8(
        values: torch.Tensor,
        scale: torch.Tensor,
        qmap: torch.Tensor,
    ) -> torch.Tensor:
        return torch.ops.alpha_quant.quantize.default(
            values,
            scale.to(device=values.device, dtype=values.dtype),
            None,
            None,
            None,
            qmap.to(values.device),
            None,
        )

    def exact_components(
        self,
        input_values: torch.Tensor,
        weight_values: torch.Tensor,
    ):
        """Return exact logical O8 payload or O4 encoded components."""
        if not self.exact_enabled:
            raise RuntimeError(
                "Magnet Alpha NMP fake epilogue is not configured"
            )
        input_payload = self._quantize_w8(
            input_values,
            self.input_scale,
            self.input_qmap,
        )
        weight_payload = self._quantize_w8(
            weight_values,
            self.weight_scale,
            self.weight_qmap,
        )
        if input_payload.ndim != 4 or weight_payload.ndim != 4:
            raise ValueError(
                "Magnet Alpha NMP expects NCHW input and KCRS weight"
            )
        logical_channels = int(input_payload.shape[1])
        if (
            logical_channels != self.groups
            or int(weight_payload.shape[0]) != logical_channels
            or int(weight_payload.shape[1]) != 1
        ):
            raise ValueError(
                "Magnet Alpha NMP requires multiplier-one depthwise convolution"
            )
        physical_channels = (
            (logical_channels + self.lane_width - 1) // self.lane_width
        ) * self.lane_width
        channel_padding = physical_channels - logical_channels
        if channel_padding:
            input_payload = torch.nn.functional.pad(
                input_payload,
                (0, 0, 0, 0, 0, channel_padding),
            )
            weight_payload = torch.nn.functional.pad(
                weight_payload,
                (0, 0, 0, 0, 0, 0, 0, channel_padding),
            )

        accumulator = alpha_int8_conv2d_accumulator(
            input_payload,
            weight_payload,
            self.stride,
            self.padding,
            self.dilation,
            physical_channels,
        )
        pre_output = alpha_nmp_epilogue(
            accumulator,
            scale_factor=self.primary_multiplier,
            binary_point=self.primary_shift,
            bias_factor=self.bias_payload,
            relu=self.relu,
        )
        if self.uses_pvs:
            encoded = alpha_pvs_quantize(
                pre_output,
                output_precision=4,
                vector_dim=1,
            )
            return (
                self.output_scale.to(
                    device=input_values.device,
                    dtype=input_values.dtype,
                ),
                encoded.mantissa,
                encoded.payload[:, :logical_channels],
            )
        payload = alpha_truncate_output(pre_output, output_precision=8)
        return payload[:, :logical_channels]

    def forward(
        self,
        semantic_output: torch.Tensor,
        input_values: torch.Tensor,
        weight_values: torch.Tensor,
        *,
        return_components: bool = False,
    ):
        if not self.exact_enabled:
            if return_components:
                raise RuntimeError(
                    "Magnet Alpha PVS components require exact calibration"
                )
            return semantic_output
        components = self.exact_components(input_values, weight_values)
        if return_components:
            if not isinstance(components, tuple):
                raise RuntimeError(
                    "Magnet Alpha encoded calibration output requires O4/PVS"
                )
            return components
        if isinstance(components, tuple):
            coarse_scale, fine_scale, payload = components
            exact = (
                payload.to(semantic_output.dtype)
                * expand(fine_scale, payload.shape, self.lane_width).to(
                    semantic_output.dtype
                )
                * coarse_scale
            )
        else:
            exact = components.to(semantic_output.dtype) * self.output_scale.to(
                device=semantic_output.device,
                dtype=semantic_output.dtype,
            )
        return exact.detach() + (semantic_output - semantic_output.detach())


__all__ = ["MagnetAlphaNMPFakeEpilogue"]

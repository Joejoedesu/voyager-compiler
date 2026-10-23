# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Calibration-time exact simulation of the Magnet Alpha PE epilogue."""

import math

import torch

from voyager_compiler.codegen.target_backends.magnet_alpha.fixed_point import (
    ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT,
    ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
    ALPHA_Q8_8_REFERENCE_FORMAT,
    alpha_output_domain,
    alpha_pe_epilogue,
    alpha_pe_postprocess,
    alpha_pvs_quantize,
    alpha_vsq_conv2d_accumulator,
    encode_alpha_vsq_scales,
    encode_selected_alpha_pe_pipeline,
    select_alpha_pe_normalization,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.hardware import (
    ALPHA_VSQ_BLOCK_VALUES,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.operators import (
    VSQ_SCALE_ENCODE_OPERATOR,
)
from voyager_compiler.decomposed import expand
from voyager_compiler.fake_quantize import FusedAmaxObsFakeQuantize
from voyager_compiler.fixed_point import (
    FixedPointFakeRequantize,
    FixedPointScalePolicy,
)

_ALPHA_PE_BINARY_POINT = 8
_ALPHA_PE_INTERMEDIATE_MIN = -(1 << 15)
_ALPHA_PE_INTERMEDIATE_MAX = (1 << 15) - 1
_ALPHA_PE_BIAS_MIN = -(1 << 7)
_ALPHA_PE_BIAS_MAX = (1 << 7) - 1


def _observer_scale_endpoints(
    observer: FusedAmaxObsFakeQuantize,
) -> tuple[float, float]:
    fine_endpoint = getattr(observer, "scale_quant_max", None)
    if fine_endpoint is None:
        fine_endpoint = getattr(observer, "quant_max", None)
    if fine_endpoint is None:
        fine_endpoint = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT

    coarse_endpoint = getattr(
        observer,
        "coarse_scale_quant_max",
        None,
    )
    if coarse_endpoint is None:
        coarse_endpoint = getattr(observer, "coarse_quant_max", None)
    if coarse_endpoint is None:
        coarse_endpoint = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT
    return float(fine_endpoint), float(coarse_endpoint)


class MagnetAlphaPEFakeEpilogue(FixedPointFakeRequantize):
    """Two-stage calibration reference for Alpha's exact PE epilogue.

    The first calibration pass retains the target-neutral fixed-point
    reference inherited from :class:`FixedPointFakeRequantize`. After
    quantization parameters are finalized, the ``configure_exact`` methods
    switch the module to Alpha's integer instruction ordering. VSQ requires
    the prepared Conv input and weight because its block-mantissa product is
    rounded before accumulation and cannot be recovered from the float Conv
    result. This module is calibration-only; converted execution remains
    owned by ``conv2d_vsq``, ``ppu_fixed_relu``, and ``pvs_quantize``.
    """

    def __init__(self) -> None:
        super().__init__(
            ALPHA_Q8_8_REFERENCE_FORMAT,
            FixedPointScalePolicy.FIXED_BINARY_POINT,
        )
        self.register_buffer(
            "accumulator_scale",
            torch.empty(0, dtype=torch.float64),
        )
        self.register_buffer(
            "real_bias",
            torch.empty(0, dtype=torch.float64),
        )
        self.register_buffer(
            "primary_multiplier",
            torch.empty(0, dtype=torch.int64),
        )
        self.register_buffer(
            "bias_payload",
            torch.empty(0, dtype=torch.int8),
        )
        self.register_buffer(
            "exact_output_scale",
            torch.tensor(1.0, dtype=torch.float64),
        )
        self.register_buffer(
            "exact_semantic_step",
            torch.tensor(
                ALPHA_Q8_8_REFERENCE_FORMAT.scale,
                dtype=torch.float64,
            ),
        )
        self.register_buffer(
            "semantic_conv_output_amax",
            torch.tensor(0.0, dtype=torch.float64),
        )
        self.register_buffer(
            "input_coarse_scale",
            torch.empty(0),
        )
        self.register_buffer(
            "input_coarse_qmap",
            torch.empty(0),
        )
        self.register_buffer(
            "input_fine_qmap",
            torch.empty(0),
        )
        self.register_buffer(
            "input_scale_qmap",
            torch.empty(0),
        )
        self.register_buffer(
            "weight_coarse_scale",
            torch.empty(0),
        )
        self.register_buffer(
            "weight_coarse_qmap",
            torch.empty(0),
        )
        self.register_buffer(
            "weight_fine_qmap",
            torch.empty(0),
        )
        self.register_buffer(
            "weight_scale_qmap",
            torch.empty(0),
        )
        self.register_buffer(
            "input_static_payload",
            torch.empty(0),
        )
        self.register_buffer(
            "input_static_fine_scale",
            torch.empty(0),
        )
        self.register_buffer(
            "weight_static_payload",
            torch.empty(0),
        )
        self.register_buffer(
            "weight_static_fine_scale",
            torch.empty(0),
        )
        self.exact_enabled = False
        self.exact_mode = "none"
        self.dynamic_qat_bypass = False
        self.primary_shift = 0
        self.secondary_multiplier = 0
        self.secondary_shift = 0
        self.output_precision = 8
        self.uses_pvs = False
        self.relu = False
        self.direct_output_normalization = False
        self.normalization_fallback_reason = None
        self.input_axes = (1,)
        self.weight_axes = (1,)
        self.input_block_size = ALPHA_VSQ_BLOCK_VALUES
        self.weight_block_size = ALPHA_VSQ_BLOCK_VALUES
        self.input_fine_quant_max = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT
        self.weight_fine_quant_max = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT
        self.input_coarse_scale_quant_max = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT
        self.weight_coarse_scale_quant_max = ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT
        self.input_force_scale_power_of_two = False
        self.weight_force_scale_power_of_two = False
        self.input_has_scale_qmap = False
        self.weight_has_scale_qmap = False
        self.stride = (1, 1)
        self.padding = (0, 0)
        self.dilation = (1, 1)
        self.groups = 1

    def _copy_buffer(self, name: str, value: torch.Tensor) -> None:
        setattr(self, name, value.detach().clone())

    @staticmethod
    def _pair(value) -> tuple[int, int]:
        if isinstance(value, int):
            return (value, value)
        pair = tuple(int(item) for item in value)
        if len(pair) != 2:
            raise ValueError(
                "Magnet Alpha PE Conv2d arguments must have two values"
            )
        return pair

    @staticmethod
    def _channel_operand(
        value: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        value = value.to(device=inputs.device).reshape(-1)
        if value.numel() == 1:
            return value[0]
        channels = inputs.shape[1] if inputs.ndim == 4 else inputs.shape[-1]
        if value.numel() != channels:
            raise ValueError(
                "Magnet Alpha PE calibration operand must be scalar or "
                "per-channel"
            )
        shape = [1] * inputs.ndim
        shape[1 if inputs.ndim == 4 else -1] = channels
        return value.reshape(shape)

    def _validate_fixed_q8_8_range(
        self,
        real_bias: torch.Tensor,
    ) -> None:
        """Preserve the established fixed-Q8.8 calibration proof."""
        output_amax = float(self.semantic_conv_output_amax.item())
        bias = real_bias.detach().to(torch.float64)
        bias_amax = (
            float(torch.amax(torch.abs(bias)).item()) if bias.numel() else 0.0
        )
        if (
            not math.isfinite(output_amax)
            or output_amax < 0.0
            or not math.isfinite(bias_amax)
        ):
            raise ValueError(
                "Magnet Alpha PE calibration headroom must be finite"
            )

        if int(self.effective_binary_point().item()) != (
            _ALPHA_PE_BINARY_POINT
        ):
            raise ValueError("Alpha PE requires fixed Q8.8 normalization")

        semantic_step = float(self.effective_scale().item())
        minimum = _ALPHA_PE_INTERMEDIATE_MIN * semantic_step
        maximum = _ALPHA_PE_INTERMEDIATE_MAX * semantic_step
        # The observed Conv output already includes bias. Adding its maximum
        # magnitude conservatively bounds a cancelling pre-bias accumulator.
        intermediate_amax = output_amax + bias_amax
        if intermediate_amax > min(abs(minimum), maximum):
            raise ValueError(
                "Alpha PE Q8.8 intermediate exceeds signed-int16 range"
            )

        normalization_scale = semantic_step * 256.0
        encoded_bias = torch.round(bias * (256.0 / normalization_scale))
        if torch.any(
            (encoded_bias < _ALPHA_PE_BIAS_MIN)
            | (encoded_bias > _ALPHA_PE_BIAS_MAX)
        ):
            raise ValueError("Alpha PE Q8.8 bias exceeds signed-int8 range")

    def _configure_epilogue(
        self,
        folded_scale: torch.Tensor,
        real_bias: torch.Tensor,
        logical_output_scale: torch.Tensor,
        *,
        accumulator_real_amax: torch.Tensor | None,
        vsq_uint8_fine_scales: bool,
        output_precision: int,
        uses_pvs: bool,
        relu: bool,
        fine_quant_max: float,
        coarse_scale_quant_max: float,
    ) -> None:
        folded_scale = folded_scale.detach()
        stored_folded_scale = folded_scale.to(torch.float64)
        real_bias = real_bias.detach().to(torch.float64)
        output = alpha_output_domain(
            logical_output_scale,
            output_precision,
            uses_pvs,
            fine_quant_max=fine_quant_max,
            coarse_scale_quant_max=coarse_scale_quant_max,
        )

        selected_normalization = select_alpha_pe_normalization(
            output,
            vsq_uint8_fine_scales=vsq_uint8_fine_scales,
            relu=relu,
        )
        encoding_scale = (
            folded_scale
            if selected_normalization.direct_output
            else stored_folded_scale
        )
        pipeline, normalization = encode_selected_alpha_pe_pipeline(
            encoding_scale,
            real_bias,
            output,
            vsq_uint8_fine_scales=vsq_uint8_fine_scales,
            relu=relu,
            accumulator_real_amax=accumulator_real_amax,
        )
        if not normalization.direct_output:
            self._validate_fixed_q8_8_range(real_bias)
        if normalization != selected_normalization and not (
            selected_normalization.direct_output
            and normalization.fallback_reason is not None
        ):
            raise AssertionError("Alpha PE normalization selection drifted")
        assert pipeline.secondary is not None
        assert pipeline.bias_relu.bias is not None

        self.accumulator_scale.resize_(stored_folded_scale.numel()).copy_(
            stored_folded_scale.reshape(-1)
        )
        self.real_bias.resize_(real_bias.numel()).copy_(real_bias.reshape(-1))
        self.primary_multiplier.resize_(
            pipeline.primary.multiplier.numel()
        ).copy_(pipeline.primary.multiplier)
        self.bias_payload.resize_(pipeline.bias_relu.bias.numel()).copy_(
            pipeline.bias_relu.bias.reshape(-1)
        )
        self.exact_output_scale.copy_(pipeline.output.scale[0])
        self.exact_semantic_step.fill_(normalization.semantic_step)
        self.primary_shift = pipeline.primary.shift
        self.secondary_multiplier = int(
            pipeline.secondary.multiplier.reshape(-1)[0].item()
        )
        self.secondary_shift = pipeline.secondary.shift
        self.output_precision = pipeline.output.precision
        self.uses_pvs = pipeline.output.uses_pvs
        self.relu = pipeline.bias_relu.relu
        self.direct_output_normalization = normalization.direct_output
        self.normalization_fallback_reason = normalization.fallback_reason
        self.exact_enabled = True

    def configure_exact(
        self,
        accumulator_scale: torch.Tensor,
        real_bias: torch.Tensor,
        output_scale: torch.Tensor,
        *,
        accumulator_real_amax: torch.Tensor,
        output_precision: int = 8,
        uses_pvs: bool = False,
        relu: bool,
        fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
        coarse_scale_quant_max: float = (ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT),
    ) -> None:
        """Configure finalized W8 Alpha operands for a second pass."""
        self._configure_epilogue(
            accumulator_scale,
            real_bias,
            output_scale,
            accumulator_real_amax=accumulator_real_amax,
            vsq_uint8_fine_scales=False,
            output_precision=output_precision,
            uses_pvs=uses_pvs,
            relu=relu,
            fine_quant_max=fine_quant_max,
            coarse_scale_quant_max=coarse_scale_quant_max,
        )
        self.exact_mode = "w8_o4" if uses_pvs else "w8_o8"

    @staticmethod
    def _axes(observer: FusedAmaxObsFakeQuantize) -> tuple[int, ...]:
        axes = observer.ch_axis
        return (axes,) if isinstance(axes, int) else tuple(axes)

    def _configure_vsq_quantizer(
        self,
        role: str,
        observer: FusedAmaxObsFakeQuantize,
    ) -> None:
        self._copy_buffer(
            f"{role}_coarse_scale",
            observer.calculate_qparams(),
        )
        self._copy_buffer(f"{role}_coarse_qmap", observer.coarse_qmap)
        self._copy_buffer(f"{role}_fine_qmap", observer.qmap)
        scale_qmap = observer.scale_qmap
        setattr(self, f"{role}_has_scale_qmap", scale_qmap is not None)
        if scale_qmap is not None:
            self._copy_buffer(f"{role}_scale_qmap", scale_qmap)
        if observer.static_vsq_payload.numel() > 0:
            self._copy_buffer(
                f"{role}_static_payload",
                observer.static_vsq_payload,
            )
            self._copy_buffer(
                f"{role}_static_fine_scale",
                observer.static_vsq_fine_scale,
            )
        setattr(self, f"{role}_axes", self._axes(observer))
        setattr(self, f"{role}_block_size", int(observer.block_size))
        fine_endpoint, coarse_endpoint = _observer_scale_endpoints(observer)
        setattr(
            self,
            f"{role}_fine_quant_max",
            fine_endpoint,
        )
        setattr(
            self,
            f"{role}_coarse_scale_quant_max",
            coarse_endpoint,
        )
        setattr(
            self,
            f"{role}_force_scale_power_of_two",
            bool(observer.force_scale_power_of_two),
        )

    def configure_exact_vsq(
        self,
        input_observer: FusedAmaxObsFakeQuantize,
        weight_observer: FusedAmaxObsFakeQuantize,
        real_bias: torch.Tensor,
        output_scale: torch.Tensor,
        *,
        output_precision: int,
        uses_pvs: bool,
        stride,
        padding,
        dilation,
        groups: int,
        relu: bool,
        fine_quant_max: float = ALPHA_DEFAULT_VSQ_FINE_ENDPOINT,
        coarse_scale_quant_max: float = (ALPHA_DEFAULT_VSQ_COARSE_ENDPOINT),
    ) -> None:
        """Configure finalized W4 VSQ operands for a second pass."""
        self._configure_vsq_quantizer("input", input_observer)
        self._configure_vsq_quantizer("weight", weight_observer)
        input_encoding = encode_alpha_vsq_scales(
            self.input_coarse_scale.new_ones(1),
            self.input_coarse_scale,
            fine_quant_max=self.input_fine_quant_max,
            coarse_scale_quant_max=(self.input_coarse_scale_quant_max),
        )
        weight_encoding = encode_alpha_vsq_scales(
            self.weight_coarse_scale.new_ones(1),
            self.weight_coarse_scale,
            fine_quant_max=self.weight_fine_quant_max,
            coarse_scale_quant_max=(self.weight_coarse_scale_quant_max),
        )
        folded_scale = input_encoding.coarse_scale.reshape(-1)[
            0
        ] * weight_encoding.coarse_scale.reshape(-1)
        self._configure_epilogue(
            folded_scale,
            real_bias,
            output_scale,
            accumulator_real_amax=None,
            vsq_uint8_fine_scales=True,
            output_precision=output_precision,
            uses_pvs=uses_pvs,
            relu=relu,
            fine_quant_max=fine_quant_max,
            coarse_scale_quant_max=coarse_scale_quant_max,
        )
        self.stride = self._pair(stride)
        self.padding = self._pair(padding)
        self.dilation = self._pair(dilation)
        self.groups = int(groups)
        self.exact_mode = "vsq_o4" if uses_pvs else "vsq_o8"

    def _quantize_vsq(
        self,
        values: torch.Tensor,
        role: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coarse_scale = getattr(self, f"{role}_coarse_scale").to(
            device=values.device,
            dtype=values.dtype,
        )
        coarse_qmap = getattr(self, f"{role}_coarse_qmap").to(values.device)
        fine_qmap = getattr(self, f"{role}_fine_qmap").to(values.device)
        scale_qmap = (
            getattr(self, f"{role}_scale_qmap").to(values.device)
            if getattr(self, f"{role}_has_scale_qmap")
            else None
        )
        static_payload = getattr(self, f"{role}_static_payload")
        if static_payload.numel() > 0:
            if tuple(static_payload.shape) != tuple(values.shape):
                raise ValueError(
                    "Magnet Alpha static VSQ payload shape does not match "
                    "operand"
                )
            payload = static_payload.to(
                device=values.device,
                dtype=values.dtype,
            )
            logical_fine_scale = getattr(
                self,
                f"{role}_static_fine_scale",
            ).to(device=values.device, dtype=values.dtype)
        else:
            _, logical_fine_scale, payload = (
                torch.ops.quantized_ops.quantize_vsq.default(
                    values,
                    coarse_scale,
                    coarse_qmap,
                    fine_qmap,
                    getattr(self, f"{role}_axes"),
                    getattr(self, f"{role}_block_size"),
                    getattr(self, f"{role}_fine_quant_max"),
                    getattr(self, f"{role}_force_scale_power_of_two"),
                    scale_qmap,
                    None,
                    getattr(self, f"{role}_coarse_scale_quant_max"),
                )
            )
        encoding = encode_alpha_vsq_scales(
            logical_fine_scale,
            coarse_scale,
            fine_quant_max=getattr(
                self,
                f"{role}_fine_quant_max",
            ),
            coarse_scale_quant_max=getattr(
                self,
                f"{role}_coarse_scale_quant_max",
            ),
        )
        mantissa = VSQ_SCALE_ENCODE_OPERATOR(
            logical_fine_scale,
            getattr(self, f"{role}_fine_quant_max"),
            getattr(self, f"{role}_coarse_scale_quant_max"),
        )
        return payload, mantissa, encoding.coarse_scale

    def exact_components(
        self,
        values: torch.Tensor,
        input_values: torch.Tensor | None = None,
        weight_values: torch.Tensor | None = None,
        input_components: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None,
    ):
        """Return exact Alpha O8 payload or O4 encoded components."""
        if not self.exact_enabled:
            raise RuntimeError(
                "Magnet Alpha PE fake epilogue is not configured"
            )
        if self.exact_mode.startswith("w8_"):
            accumulator_scale = self._channel_operand(
                self.accumulator_scale,
                values,
            )
            real_bias = self._channel_operand(self.real_bias, values)
            accumulator = torch.round(
                (values.to(torch.float64) - real_bias) / accumulator_scale
            ).to(torch.int64)
        else:
            if input_values is None or weight_values is None:
                raise ValueError(
                    "Magnet Alpha VSQ fake epilogue requires input and weight"
                )
            if input_components is None:
                input_payload, input_mantissa, _ = self._quantize_vsq(
                    input_values,
                    "input",
                )
            else:
                (
                    _input_coarse_scale,
                    input_mantissa,
                    input_payload,
                ) = input_components
            weight_payload, weight_mantissa, _ = self._quantize_vsq(
                weight_values,
                "weight",
            )
            accumulator = alpha_vsq_conv2d_accumulator(
                input_payload,
                weight_payload,
                input_mantissa,
                weight_mantissa,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                block_size=self.input_block_size,
            )

        primary_multiplier = self._channel_operand(
            self.primary_multiplier,
            values,
        )
        bias_payload = self._channel_operand(self.bias_payload, values)
        if self.uses_pvs:
            pre_pvs = alpha_pe_postprocess(
                accumulator,
                primary_multiplier,
                self.primary_shift,
                bias_payload,
                self.secondary_multiplier,
                self.secondary_shift,
                relu=self.relu,
            )
            pvs = alpha_pvs_quantize(
                pre_pvs,
                output_precision=self.output_precision,
                vector_dim=1,
            )
            return (
                self.exact_output_scale.to(
                    device=values.device,
                    dtype=values.dtype,
                ),
                pvs.mantissa,
                pvs.payload,
            )
        return alpha_pe_epilogue(
            accumulator,
            primary_multiplier,
            self.primary_shift,
            bias_payload,
            self.secondary_multiplier,
            self.secondary_shift,
            self.output_precision,
            relu=self.relu,
        )

    def exact_payload(self, values: torch.Tensor) -> torch.Tensor:
        """Return the exact Alpha O8 payload for prepared Conv output."""
        components = self.exact_components(values)
        if isinstance(components, tuple):
            return components[-1]
        return components

    def forward(
        self,
        values: torch.Tensor,
        input_values: torch.Tensor | None = None,
        weight_values: torch.Tensor | None = None,
        *,
        input_components: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None,
        return_components: bool = False,
    ):
        if not self.exact_enabled:
            if return_components:
                raise RuntimeError(
                    "Magnet Alpha PVS components require exact calibration"
                )
            current_amax = torch.amax(torch.abs(values.detach()))
            current_amax = current_amax.to(
                device=self.semantic_conv_output_amax.device,
                dtype=self.semantic_conv_output_amax.dtype,
            )
            self.semantic_conv_output_amax.copy_(
                torch.maximum(
                    self.semantic_conv_output_amax,
                    current_amax,
                )
            )
            if self.dynamic_qat_bypass:
                return values
            return super().forward(values)
        components = self.exact_components(
            values,
            input_values,
            weight_values,
            input_components,
        )
        if return_components:
            if not isinstance(components, tuple):
                raise RuntimeError(
                    "Magnet Alpha encoded calibration output requires O4/PVS"
                )
            return components
        if isinstance(components, tuple):
            coarse_scale, fine_scale, payload = components
            exact = (
                payload.to(values.dtype)
                * expand(
                    fine_scale,
                    payload.shape,
                    ALPHA_VSQ_BLOCK_VALUES,
                ).to(values.dtype)
                * coarse_scale.to(
                    device=values.device,
                    dtype=values.dtype,
                )
            )
        else:
            payload = components
            exact = payload.to(values.dtype) * self.exact_output_scale.to(
                device=values.device,
                dtype=values.dtype,
            )
        return exact.detach() + (values - values.detach())


__all__ = ["MagnetAlphaPEFakeEpilogue"]

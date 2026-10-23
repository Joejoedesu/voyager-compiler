# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import unittest
from dataclasses import asdict

import torch
from torch.ao.quantization import FakeQuantizeBase

import voyager_compiler as voyager
from voyager_compiler.codegen.mapping_utils import SOURCE_OP_NAME
from voyager_compiler.codegen.target_backends import TargetOperatorTraits
from voyager_compiler.fake_quantize import (
    FusedAmaxObsFakeQuantize,
    get_quantization_map,
)
from voyager_compiler.pt2e_utils import COMPILER_ONLY
from voyager_compiler.quantization_regions import (
    _move_output_observers_after_epilogues,
)
from voyager_compiler.quantizer.quantizer import (
    QScheme,
    QuantizationSpec,
)

VSQ_SPEC = "int4,qs=vsq,coarse=int8,bs=4,scale=uint8"
VSQ_ACTIVATION_SPEC = (
    "int4,qs=vsq,coarse=int8,cqs=per_tensor_symmetric," "bs=4,scale=uint8"
)
VSQ_WEIGHT_SPEC = (
    "int4,qs=vsq,coarse=int8,cqs=per_channel_symmetric,cax=0,"
    "bs=4,scale=uint8"
)
ENDPOINT_VSQ_SPEC = "int4,qs=vsq,coarse=int8,ax=1,bs=4,sqmax=8,csqmax=126"
_RELU_EPILOGUE_TRAITS = TargetOperatorTraits(
    quantization_output_epilogue_operators=(torch.ops.aten.relu.default,),
)


class TinyConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 3, 1, bias=True)

    def forward(self, x):
        return self.conv(x)


class TinyConvRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 3, 1, bias=True)

    def forward(self, x):
        return torch.relu(self.conv(x))


class TinyConvClamp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 3, 1, bias=True)

    def forward(self, x):
        return torch.clamp(self.conv(x), 0.0, 6.0)


class TinyReshapedConvRelu(torch.nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 1, bias=True)
        self.output_shape = output_shape

    def forward(self, x):
        return torch.relu(self.conv(x).reshape(self.output_shape))


class TinyPermutedConvRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 1, bias=True)

    def forward(self, x):
        return torch.relu(self.conv(x).permute(0, 1, 3, 2))


class TinyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3, bias=True)

    def forward(self, x):
        return self.linear(x)


class TinyMatmul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 3))

    def forward(self, x):
        return torch.ops.aten.matmul.default(x, self.weight)


def reconstruct_vsq(outputs, shape, block_size):
    coarse_scale, fine_scale, data = outputs
    expanded_fine_scale = voyager.expand(fine_scale, shape, block_size)
    return data * expanded_fine_scale * coarse_scale


class TestVSQQuantization(unittest.TestCase):
    def test_output_observer_does_not_assume_undeclared_epilogue(self):
        model = TinyConvRelu().eval()
        x = torch.randn(1, 4, 3, 3)
        exported = torch.export.export(model, (x,)).module()
        quantizer = voyager.get_default_quantizer(
            input_activation="int8,qs=per_tensor_symmetric",
            output_activation="int8,qs=per_tensor_symmetric",
            weight="int8,qs=per_channel_symmetric,ax=0",
            bias="int32",
        )
        prepared = voyager.prepare_pt2e(exported, quantizer)

        _move_output_observers_after_epilogues(
            prepared,
            TargetOperatorTraits(),
        )

        conv = next(
            node
            for node in prepared.graph.nodes
            if node.target == torch.ops.aten.conv2d.default
        )
        observer = next(iter(conv.users))
        relu = next(iter(observer.users))
        self.assertIsInstance(
            prepared.get_submodule(str(observer.target)),
            FusedAmaxObsFakeQuantize,
        )
        self.assertEqual(relu.target, torch.ops.aten.relu.default)

    def test_output_observer_uses_target_declared_epilogue(self):
        model = TinyConvClamp().eval()
        x = torch.randn(1, 4, 3, 3)
        exported = torch.export.export(model, (x,)).module()
        quantizer = voyager.get_default_quantizer(
            input_activation="int8,qs=per_tensor_symmetric",
            output_activation="int8,qs=per_tensor_symmetric",
            weight="int8,qs=per_channel_symmetric,ax=0",
            bias="int32",
        )
        prepared = voyager.prepare_pt2e(exported, quantizer)
        traits = TargetOperatorTraits(
            quantization_output_epilogue_operators=(
                torch.ops.aten.clamp.default,
            ),
        )

        _move_output_observers_after_epilogues(prepared, traits)

        conv = next(
            node
            for node in prepared.graph.nodes
            if node.target == torch.ops.aten.conv2d.default
        )
        clamp = next(iter(conv.users))
        observer = next(iter(clamp.users))
        self.assertEqual(clamp.target, torch.ops.aten.clamp.default)
        self.assertIsInstance(
            prepared.get_submodule(str(observer.target)),
            FusedAmaxObsFakeQuantize,
        )

    def test_output_observer_crosses_address_only_view_per_tensor(self):
        model = TinyReshapedConvRelu((1, 4, 4, 4)).eval()
        with torch.no_grad():
            model.conv.weight.zero_()
            model.conv.bias.copy_(torch.tensor([-10.0, 1.0, 2.0, 3.0]))
        x = torch.zeros(1, 4, 1, 16)
        exported = torch.export.export(model, (x,)).module()
        quantizer = voyager.get_default_quantizer(
            input_activation="int8,qs=per_tensor_symmetric",
            output_activation="int8,qs=per_tensor_symmetric",
            weight="int8,qs=per_channel_symmetric,ax=0",
            bias="int32",
        )
        prepared = voyager.prepare_pt2e(exported, quantizer)

        _move_output_observers_after_epilogues(
            prepared,
            _RELU_EPILOGUE_TRAITS,
        )

        conv = next(
            node
            for node in prepared.graph.nodes
            if node.target == torch.ops.aten.conv2d.default
        )
        reshape = next(iter(conv.users))
        relu = next(iter(reshape.users))
        observer = next(iter(relu.users))
        self.assertEqual(reshape.target, torch.ops.aten.reshape.default)
        self.assertEqual(relu.target, torch.ops.aten.relu.default)
        self.assertIsInstance(
            prepared.get_submodule(str(observer.target)),
            FusedAmaxObsFakeQuantize,
        )
        self.assertEqual(tuple(observer.meta["val"].shape), (1, 4, 4, 4))
        prepared(x)
        observer_module = prepared.get_submodule(str(observer.target))
        self.assertEqual(torch.amax(observer_module.amax_history).item(), 3.0)

    def test_output_observer_does_not_cross_view_per_channel(self):
        model = TinyReshapedConvRelu((1, 1, 4, 16)).eval()
        x = torch.randn(1, 4, 1, 16)
        exported = torch.export.export(model, (x,)).module()
        quantizer = voyager.get_default_quantizer(
            input_activation="int8,qs=per_tensor_symmetric",
            output_activation="int8,qs=per_channel_symmetric,ax=1",
            weight="int8,qs=per_channel_symmetric,ax=0",
            bias="int32",
        )
        prepared = voyager.prepare_pt2e(exported, quantizer)

        _move_output_observers_after_epilogues(
            prepared,
            _RELU_EPILOGUE_TRAITS,
        )

        conv = next(
            node
            for node in prepared.graph.nodes
            if node.target == torch.ops.aten.conv2d.default
        )
        observer = next(iter(conv.users))
        reshape = next(iter(observer.users))
        relu = next(iter(reshape.users))
        observer_module = prepared.get_submodule(str(observer.target))
        self.assertIsInstance(observer_module, FusedAmaxObsFakeQuantize)
        self.assertEqual(
            observer_module.qscheme,
            QScheme.PER_CHANNEL_SYMMETRIC,
        )
        self.assertEqual(reshape.target, torch.ops.aten.reshape.default)
        self.assertEqual(relu.target, torch.ops.aten.relu.default)
        prepared(x)
        observer_module.calculate_qparams()
        self.assertEqual(tuple(observer_module.scale.shape), (1, 4, 1, 1))

    def test_output_observer_does_not_cross_non_address_only_permute(self):
        model = TinyPermutedConvRelu().eval()
        x = torch.randn(1, 4, 1, 16)
        exported = torch.export.export(model, (x,)).module()
        quantizer = voyager.get_default_quantizer(
            input_activation="int8,qs=per_tensor_symmetric",
            output_activation="int8,qs=per_tensor_symmetric",
            weight="int8,qs=per_channel_symmetric,ax=0",
            bias="int32",
        )
        prepared = voyager.prepare_pt2e(exported, quantizer)

        _move_output_observers_after_epilogues(
            prepared,
            _RELU_EPILOGUE_TRAITS,
        )

        conv = next(
            node
            for node in prepared.graph.nodes
            if node.target == torch.ops.aten.conv2d.default
        )
        observer = next(iter(conv.users))
        permute = next(iter(observer.users))
        relu = next(iter(permute.users))
        self.assertIsInstance(
            prepared.get_submodule(str(observer.target)),
            FusedAmaxObsFakeQuantize,
        )
        self.assertEqual(permute.target, torch.ops.aten.permute.default)
        self.assertEqual(relu.target, torch.ops.aten.relu.default)

    def test_zero_step_calibration_preserves_identity_dequantize(self):
        def prepare_unobserved_conv():
            model = TinyConv().eval()
            x = torch.randn(1, 4, 3, 3)
            exported = torch.export.export(model, (x,)).module()
            quantizer = voyager.get_default_quantizer(
                input_activation="int8,qs=per_tensor_symmetric",
                weight="int8,qs=per_channel_symmetric,ax=0",
                bias="int32",
            )
            return voyager.prepare_pt2e(exported, quantizer)

        preserved = voyager.convert_pt2e(
            prepare_unobserved_conv(),
            calibration_steps=0,
        )
        input_quantize = next(
            node
            for node in preserved.graph.nodes
            if node.target == torch.ops.quantized_ops.quantize.default
        )
        input_scale = preserved.get_buffer(input_quantize.args[1].target)
        output_dequantize = next(
            node
            for node in preserved.graph.nodes
            if node.target == torch.ops.quantized_ops.dequantize.default
        )
        output_scale = preserved.get_buffer(output_dequantize.args[1].target)

        self.assertEqual(tuple(input_scale.shape), (1,))
        self.assertEqual(tuple(output_scale.shape), (3, 1, 1))
        torch.testing.assert_close(input_scale, torch.ones(1))
        torch.testing.assert_close(output_scale, torch.ones(3, 1, 1))

        eliminated = voyager.convert_pt2e(
            prepare_unobserved_conv(),
            calibration_steps=1,
        )

        dequantize = torch.ops.quantized_ops.dequantize.default
        self.assertIn(
            dequantize,
            (node.target for node in preserved.graph.nodes),
        )
        self.assertNotIn(
            dequantize,
            (node.target for node in eliminated.graph.nodes),
        )

    def test_calculate_qparams_includes_latest_amax(self):
        spec = QuantizationSpec.from_str("int8,qs=per_tensor_symmetric,ahl=3")
        kwargs = asdict(spec)
        kwargs.pop("observer_or_fake_quant_ctr")
        fake_quant = FusedAmaxObsFakeQuantize(**kwargs)

        for amax in (2.0, 5.0, 10.0):
            fake_quant(torch.tensor([amax]))

        torch.testing.assert_close(
            fake_quant.scale,
            torch.tensor(5.0 / 127.0),
        )
        torch.testing.assert_close(
            fake_quant.calculate_qparams(),
            torch.tensor(10.0 / 127.0),
        )

    def test_fake_quant_rebase_encapsulates_empty_history_seed(self):
        spec = QuantizationSpec.from_str("int8,qs=per_tensor_symmetric,ahl=3")
        kwargs = asdict(spec)
        kwargs.pop("observer_or_fake_quant_ctr")
        fake_quant = FusedAmaxObsFakeQuantize(**kwargs)
        fake_quant.disable_observer()

        fake_quant.rebase_qparams_from(torch.tensor([-254.0, 127.0]))
        self.assertFalse(bool(fake_quant.observer_enabled[0].item()))
        torch.testing.assert_close(
            fake_quant.calculate_qparams(),
            torch.tensor(2.0),
        )

        fake_quant.reset_observer_statistics()
        self.assertEqual(fake_quant.amax_history.numel(), 0)
        torch.testing.assert_close(fake_quant.scale, torch.ones(1))
        self.assertFalse(bool(fake_quant.observer_enabled[0].item()))

    def test_static_vsq_components_round_trip_without_dynamic_observation(
        self,
    ):
        spec = QuantizationSpec.from_str(
            "int4,qs=vsq,coarse=int8,cqs=per_channel_symmetric,"
            "cax=0,ax=1,bs=4,scale=uint8,qmin=-7,qmax=7"
        )
        kwargs = asdict(spec)
        kwargs.pop("observer_or_fake_quant_ctr")
        fake_quant = FusedAmaxObsFakeQuantize(**kwargs)
        self.assertEqual(fake_quant.static_vsq_payload.numel(), 0)

        payload = torch.tensor(
            [
                [[[-7.0]], [[-2.0]], [[3.0]], [[7.0]]],
                [[[-6.0]], [[-1.0]], [[2.0]], [[6.0]]],
            ]
        )
        mantissa = torch.tensor([255.0, 127.0]).reshape(2, 1, 1, 1)
        fine = mantissa * 127.0 / (7.0 * 255.0)
        coarse = torch.tensor([0.01, 0.02]).reshape(2, 1, 1, 1)
        reference = payload * fine * coarse

        fake_quant.set_static_vsq_components(
            payload,
            fine,
            coarse,
            reference=reference,
        )

        torch.testing.assert_close(fake_quant(reference), reference)
        components = fake_quant.quantize_vsq_components(reference)
        torch.testing.assert_close(components[0], coarse)
        torch.testing.assert_close(components[1], fine)
        torch.testing.assert_close(components[2], payload)
        torch.testing.assert_close(fake_quant.calculate_qparams(), coarse)

        reloaded = FusedAmaxObsFakeQuantize(**kwargs)
        reloaded.load_state_dict(fake_quant.state_dict())
        torch.testing.assert_close(reloaded(reference), reference)
        torch.testing.assert_close(
            reloaded.static_vsq_payload,
            payload,
        )

    def test_vsq_spec_parses_both_stages(self):
        spec = QuantizationSpec.from_str(VSQ_SPEC)

        self.assertEqual(spec.qscheme, QScheme.VSQ)
        self.assertEqual(spec.dtype, "int4")
        self.assertEqual(spec.quant_min, -8)
        self.assertEqual(spec.quant_max, 7)
        self.assertEqual(spec.coarse_dtype, "int8")
        self.assertEqual(spec.coarse_quant_min, -128)
        self.assertEqual(spec.coarse_quant_max, 127)
        self.assertEqual(spec.block_size, 4)
        self.assertEqual(spec.scale_dtype, "uint8")
        self.assertEqual(
            spec.coarse_qscheme,
            QScheme.PER_TENSOR_SYMMETRIC,
        )
        self.assertIsNone(spec.coarse_ch_axis)

        weight_spec = QuantizationSpec.from_str(VSQ_WEIGHT_SPEC)
        self.assertEqual(
            weight_spec.coarse_qscheme,
            QScheme.PER_CHANNEL_SYMMETRIC,
        )
        self.assertEqual(weight_spec.coarse_ch_axis, 0)

        with self.assertRaisesRegex(ValueError, "coarse_dtype"):
            QuantizationSpec.from_str("int4,qs=vsq,bs=4")

        with self.assertRaisesRegex(ValueError, "bias must be None"):
            voyager.get_default_quantizer(
                input_activation=VSQ_ACTIVATION_SPEC,
                weight=VSQ_WEIGHT_SPEC,
                bias="int24",
            )

        voyager.get_default_quantizer(output_activation=VSQ_SPEC)

    def test_scale_endpoints_are_independent_of_storage_bounds(self):
        spec = QuantizationSpec.from_str(ENDPOINT_VSQ_SPEC)
        self.assertEqual(spec.quant_min, -8)
        self.assertEqual(spec.quant_max, 7)
        self.assertEqual(spec.scale_quant_max, 8)
        self.assertEqual(spec.coarse_quant_min, -128)
        self.assertEqual(spec.coarse_quant_max, 127)
        self.assertEqual(spec.coarse_scale_quant_max, 126)

        fake_quant = FusedAmaxObsFakeQuantize(**asdict(spec))
        fake_quant.amax_history.resize_((16,)).fill_(252.0)
        fake_quant.calculate_qparams()
        fake_quant.disable_observer()

        values = torch.tensor([[16.0, 14.0, -16.0, -14.0]])
        actual = fake_quant(values)
        self.assertEqual(fake_quant.scale.item(), 2.0)
        self.assertEqual(fake_quant.fine_scale.item(), 1.0)
        torch.testing.assert_close(
            actual,
            torch.tensor([[14.0, 14.0, -16.0, -14.0]]),
            rtol=0.0,
            atol=0.0,
        )

        fine_map = fake_quant.qmap[~torch.isnan(fake_quant.qmap)]
        coarse_map = fake_quant.coarse_qmap[
            ~torch.isnan(fake_quant.coarse_qmap)
        ]
        self.assertEqual(fine_map.min().item(), -8.0)
        self.assertEqual(fine_map.max().item(), 7.0)
        self.assertEqual(coarse_map.min().item(), -128.0)
        self.assertEqual(coarse_map.max().item(), 127.0)

    def test_per_tensor_scale_endpoint_preserves_int8_storage_range(self):
        spec = QuantizationSpec.from_str(
            "int8,qs=per_tensor_symmetric,sqmax=126"
        )
        fake_quant = FusedAmaxObsFakeQuantize(**asdict(spec))
        fake_quant.amax_history.resize_((16,)).fill_(252.0)
        fake_quant.calculate_qparams()
        fake_quant.disable_observer()

        values = torch.tensor([-256.0, 252.0, 254.0])
        actual = fake_quant(values)
        self.assertEqual(fake_quant.scale.item(), 2.0)
        torch.testing.assert_close(
            actual,
            torch.tensor([-256.0, 252.0, 254.0]),
            rtol=0.0,
            atol=0.0,
        )
        quant_map = fake_quant.qmap[~torch.isnan(fake_quant.qmap)]
        self.assertEqual(quant_map.min().item(), -128.0)
        self.assertEqual(quant_map.max().item(), 127.0)

    def test_floor_rounding_is_explicit_and_preserves_storage_bounds(self):
        spec = QuantizationSpec.from_str(
            "int8,qs=per_tensor_symmetric,sqmax=126,rnd=floor"
        )
        fake_quant = FusedAmaxObsFakeQuantize(**asdict(spec))
        fake_quant.amax_history.resize_((16,)).fill_(252.0)
        fake_quant.calculate_qparams()
        fake_quant.disable_observer()

        actual = fake_quant(torch.tensor([-255.0, 253.0, 255.0]))
        torch.testing.assert_close(
            actual,
            torch.tensor([-256.0, 252.0, 254.0]),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(spec.quant_min, -128)
        self.assertEqual(spec.quant_max, 127)

        boundary_values = torch.tensor([-1.001, 1.999])
        fake_quant.scale.fill_(1.0)
        torch.testing.assert_close(
            fake_quant(boundary_values),
            torch.tensor([-2.0, 1.0]),
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            torch.ops.quantized_ops.quantize.default(
                boundary_values,
                torch.ones(1),
                None,
                None,
                None,
                None,
                None,
                "floor",
                -128.0,
                127.0,
            ),
            torch.tensor([-2.0, 1.0]),
            rtol=0.0,
            atol=0.0,
        )

        with self.assertRaisesRegex(ValueError, "rounding must be"):
            QuantizationSpec.from_str(
                "int8,qs=per_tensor_symmetric,rnd=stochastic"
            )

    def test_vsq_conversion_keeps_scale_endpoint(self):
        model = TinyConv().eval()
        values = torch.tensor(
            [
                [
                    [[16.0]],
                    [[14.0]],
                    [[-16.0]],
                    [[-14.0]],
                ]
            ]
        )
        exported = torch.export.export(model, (values,)).module()
        quantizer = voyager.get_default_quantizer(
            input_activation=ENDPOINT_VSQ_SPEC,
            weight=VSQ_WEIGHT_SPEC,
        )
        prepared = voyager.prepare_pt2e(exported, quantizer)
        prepared(values)
        prepared(values)
        converted = voyager.convert_pt2e(prepared)

        activation_quantizer = next(
            node
            for node in converted.graph.nodes
            if (
                node.target == torch.ops.quantized_ops.quantize_vsq.default
                and node.args[0].op == "placeholder"
            )
        )
        self.assertEqual(activation_quantizer.args[6], 8.0)
        self.assertEqual(activation_quantizer.args[10], 126.0)
        coarse_scale = converted.get_buffer(activation_quantizer.args[1].target)
        torch.testing.assert_close(
            coarse_scale,
            torch.tensor(16.0 / 126.0),
        )

    def test_explicit_int4_bounds_constrain_fake_and_decomposed_codes(self):
        weight_spec = QuantizationSpec.from_str(
            VSQ_WEIGHT_SPEC + ",qmin=-7,qmax=7"
        )
        weight_kwargs = asdict(weight_spec)
        weight_kwargs.pop("observer_or_fake_quant_ctr")
        weight_fake_quant = FusedAmaxObsFakeQuantize(**weight_kwargs)

        fake_spec = QuantizationSpec.from_str(
            "int4,qs=per_tensor_symmetric,qmin=-7,qmax=7"
        )
        kwargs = asdict(fake_spec)
        kwargs.pop("observer_or_fake_quant_ctr")
        fake_quant = FusedAmaxObsFakeQuantize(**kwargs)
        fake_quant.disable_observer()
        fake_quant.scale.fill_(1.0)
        values = torch.tensor([-8.0, -7.0, 0.0, 7.0])

        fake_output = fake_quant(values)
        materialized = torch.ops.quantized_ops.quantize.default(
            values,
            torch.tensor(1.0),
            qmap=weight_fake_quant.qmap,
        )

        expected = torch.tensor([-7.0, -7.0, 0.0, 7.0])
        weight_codes = weight_fake_quant.qmap[
            torch.isfinite(weight_fake_quant.qmap)
        ]
        fake_codes = fake_quant.qmap[torch.isfinite(fake_quant.qmap)]
        self.assertEqual(torch.amin(weight_codes).item(), -7.0)
        self.assertEqual(torch.amin(fake_codes).item(), -7.0)
        torch.testing.assert_close(fake_output, expected)
        torch.testing.assert_close(materialized, expected)
        self.assertFalse(torch.any(materialized == -8))

        vsq_values = torch.tensor([[-15.0, 0.0, 0.0, 0.0]])
        vsq_args = (
            vsq_values,
            torch.tensor(1.0),
            weight_fake_quant.coarse_qmap,
        )
        bounded_vsq = torch.ops.quantized_ops.quantize_vsq.default(
            *vsq_args,
            weight_fake_quant.qmap,
            [-1],
            4,
            7.0,
            False,
            weight_fake_quant.scale_qmap,
            None,
        )
        unbounded_codes = get_quantization_map("int4")
        unbounded_codes = unbounded_codes[torch.isfinite(unbounded_codes)]
        self.assertEqual(torch.amin(unbounded_codes).item(), -8.0)
        self.assertFalse(torch.any(bounded_vsq[2] == -8))
        self.assertGreaterEqual(torch.amin(bounded_vsq[2]).item(), -7.0)

    def test_default_int4_activation_retains_negative_eight(self):
        default_spec = QuantizationSpec.from_str("int4,qs=per_tensor_symmetric")
        kwargs = asdict(default_spec)
        kwargs.pop("observer_or_fake_quant_ctr")
        fake_quant = FusedAmaxObsFakeQuantize(**kwargs)
        fake_quant.disable_observer()
        fake_quant.scale.fill_(1.0)

        actual = fake_quant(torch.tensor([-8.0, 7.0]))

        codes = fake_quant.qmap[torch.isfinite(fake_quant.qmap)]
        self.assertEqual(torch.amin(codes).item(), -8.0)
        torch.testing.assert_close(actual, torch.tensor([-8.0, 7.0]))

    def test_atomic_vsq_node_reconstructs_two_stage_quantization(self):
        x = torch.tensor([[-6.0, -2.0, 2.0, 6.0]])
        coarse_scale = torch.tensor(0.5)

        outputs = torch.ops.quantized_ops.quantize_vsq.default(
            x,
            coarse_scale,
            get_quantization_map("int8"),
            get_quantization_map("int4"),
            [-1],
            4,
            7.0,
        )

        coarse_output, fine_scale, data = outputs
        torch.testing.assert_close(coarse_output, coarse_scale)
        self.assertEqual(fine_scale.shape, (1, 1))
        self.assertTrue(torch.all(data >= -8))
        self.assertTrue(torch.all(data <= 7))
        torch.testing.assert_close(
            reconstruct_vsq(outputs, x.shape, 4),
            torch.tensor([[-6.0, -12.0 / 7.0, 12.0 / 7.0, 6.0]]),
        )

    def test_block_aligned_vsq_concat_preserves_encoded_components(self):
        torch.manual_seed(0)
        left = 3 * torch.randn(1, 4, 2, 2)
        right = 7 * torch.randn(1, 4, 2, 2)
        coarse_scale = torch.tensor(0.125)
        args = (
            coarse_scale,
            get_quantization_map("int8"),
            get_quantization_map("int4"),
            [1],
            4,
            7.0,
            False,
            get_quantization_map("uint8"),
            None,
        )

        left_vsq = torch.ops.quantized_ops.quantize_vsq.default(
            left,
            *args,
        )
        right_vsq = torch.ops.quantized_ops.quantize_vsq.default(
            right,
            *args,
        )
        left_float = torch.ops.quantized_ops.dequantize_vsq.default(
            left_vsq[2],
            left_vsq[1],
            left_vsq[0],
            [1],
            4,
        )
        right_float = torch.ops.quantized_ops.dequantize_vsq.default(
            right_vsq[2],
            right_vsq[1],
            right_vsq[0],
            [1],
            4,
        )
        joined_scale = torch.cat((left_vsq[1], right_vsq[1]), dim=1)
        joined_payload = torch.cat(
            (left_vsq[2], right_vsq[2]),
            dim=1,
        )
        joined_float = reconstruct_vsq(
            (coarse_scale, joined_scale, joined_payload),
            (1, 8, 2, 2),
            4,
        )
        torch.testing.assert_close(
            joined_float,
            torch.cat((left_float, right_float), dim=1),
        )

    def test_vsq_output_activation_keeps_explicit_scale_and_payload(self):
        torch.manual_seed(0)
        model = TinyConv().eval()
        x = torch.randn(1, 4, 3, 3)
        exported = torch.export.export(model, (x,)).module()
        quantizer = voyager.get_default_quantizer(
            output_activation=VSQ_SPEC,
        )
        prepared = voyager.prepare_pt2e(exported, quantizer)

        prepared(x)
        expected = prepared(x)
        converted = voyager.convert_pt2e(prepared)
        actual = converted(x)

        targets = [node.target for node in converted.graph.nodes]
        self.assertEqual(
            targets.count(torch.ops.quantized_ops.quantize_vsq.default),
            1,
        )
        self.assertEqual(
            targets.count(torch.ops.quantized_ops.dequantize_vsq.default),
            1,
        )
        self.assertNotIn(torch.ops.aten.mul.Tensor, targets)
        torch.testing.assert_close(actual, expected)

    def test_vsq_fake_quant_matches_atomic_node_and_has_ste_gradient(self):
        spec = QuantizationSpec.from_str(VSQ_SPEC)
        kwargs = asdict(spec)
        kwargs.pop("observer_or_fake_quant_ctr")
        kwargs["ch_axis"] = -1
        fake_quant = FusedAmaxObsFakeQuantize(**kwargs)
        fake_quant.disable_observer()
        fake_quant.scale.fill_(0.5)

        x = torch.tensor([[-6.0, -2.0, 2.0, 6.0]], requires_grad=True)
        actual = fake_quant(x)
        atomic_outputs = torch.ops.quantized_ops.quantize_vsq.default(
            x.detach(),
            fake_quant.scale.detach(),
            fake_quant.coarse_qmap,
            fake_quant.qmap,
            [-1],
            4,
            7.0,
            False,
            fake_quant.scale_qmap,
        )
        expected = reconstruct_vsq(atomic_outputs, x.shape, 4)

        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_vsq_state_dict_round_trip_with_resized_fine_scale(self):
        spec = QuantizationSpec.from_str(VSQ_SPEC)
        kwargs = asdict(spec)
        kwargs.pop("observer_or_fake_quant_ctr")
        kwargs["ch_axis"] = -1
        fake_quant = FusedAmaxObsFakeQuantize(**kwargs)
        fake_quant(torch.randn(2, 8))

        self.assertEqual(tuple(fake_quant.fine_scale.shape), (2, 2))

        reloaded = FusedAmaxObsFakeQuantize(**kwargs)
        self.assertEqual(tuple(reloaded.fine_scale.shape), (1,))
        reloaded.load_state_dict(fake_quant.state_dict())

        self.assertEqual(tuple(reloaded.fine_scale.shape), (2, 2))
        torch.testing.assert_close(
            reloaded.fine_scale,
            fake_quant.fine_scale,
        )
        torch.testing.assert_close(reloaded.scale, fake_quant.scale)
        torch.testing.assert_close(
            reloaded.amax_history,
            fake_quant.amax_history,
        )

    def test_vsq_pt2e_conversion_is_atomic_and_keeps_float_bias(self):
        torch.manual_seed(0)
        model = TinyConv().eval()
        x = torch.randn(1, 4, 3, 3)
        exported = torch.export.export(model, (x,)).module()
        quantizer = voyager.get_default_quantizer(
            input_activation=VSQ_ACTIVATION_SPEC,
            weight=VSQ_WEIGHT_SPEC,
        )
        prepared = voyager.prepare_pt2e(exported, quantizer)

        prepared(x)
        prepared(x)
        for module in prepared.modules():
            if isinstance(module, FakeQuantizeBase):
                module.disable_observer()

        expected = prepared(x)
        converted = voyager.convert_pt2e(prepared)
        actual = converted(x)

        targets = [node.target for node in converted.graph.nodes]
        self.assertEqual(
            targets.count(torch.ops.quantized_ops.quantize_vsq.default),
            1,
        )
        self.assertIn(torch.ops.quantized_ops.conv2d_mx.default, targets)
        self.assertNotIn(torch.ops.quantized_ops.dequantize.default, targets)
        self.assertNotIn(torch.ops.aten.mul.Tensor, targets)

        bias_nodes = [
            node
            for node in converted.graph.nodes
            if node.op == "get_attr" and "bias" in str(node.target)
        ]
        self.assertEqual(len(bias_nodes), 1)
        self.assertIsNone(bias_nodes[0].meta.get("dtype"))
        mx_node = next(
            node
            for node in converted.graph.nodes
            if node.target == torch.ops.quantized_ops.conv2d_mx.default
        )
        self.assertEqual(mx_node.meta.get(SOURCE_OP_NAME), "conv")
        self.assertIn("input_scale", mx_node.kwargs)
        self.assertIn("input_coarse_scale", mx_node.kwargs)
        self.assertIn("weight_scale", mx_node.kwargs)
        self.assertIn("weight_coarse_scale", mx_node.kwargs)
        vsq_node = next(
            node
            for node in converted.graph.nodes
            if node.target == torch.ops.quantized_ops.quantize_vsq.default
        )
        for lookup in (
            vsq_node.args[2],
            vsq_node.args[3],
            vsq_node.args[8],
            vsq_node.args[9],
        ):
            if isinstance(lookup, torch.fx.Node):
                self.assertTrue(lookup.meta.get(COMPILER_ONLY, False))
        self.assertFalse(vsq_node.args[1].meta.get(COMPILER_ONLY, False))
        input_coarse = mx_node.kwargs["input_coarse_scale"]
        weight_coarse = mx_node.kwargs["weight_coarse_scale"]
        input_coarse_attr = input_coarse.args[0].args[1]
        self.assertEqual(weight_coarse.op, "get_attr")
        weight_coarse_attr = weight_coarse
        input_coarse_buffer = converted.get_buffer(input_coarse_attr.target)
        self.assertEqual(input_coarse_buffer.numel(), 1)
        self.assertEqual(
            tuple(converted.get_buffer(weight_coarse_attr.target).shape),
            (3, 1, 1, 1),
        )
        torch.testing.assert_close(actual, expected)

    def test_vsq_fusion_preserves_logical_compute_name(self):
        torch.manual_seed(0)
        model = TinyConvRelu().eval()
        x = torch.randn(1, 4, 3, 3)
        exported = torch.export.export(model, (x,)).module()
        quantizer = voyager.get_default_quantizer(
            input_activation=VSQ_ACTIVATION_SPEC,
            weight=VSQ_WEIGHT_SPEC,
        )
        prepared = voyager.prepare_pt2e(exported, quantizer)

        prepared(x)
        prepared(x)
        for module in prepared.modules():
            if isinstance(module, FakeQuantizeBase):
                module.disable_observer()

        expected = prepared(x)
        converted = voyager.convert_pt2e(prepared)
        patterns = [
            [
                voyager.OpMatcher("conv2d_mx"),
                voyager.OpMatcher("relu"),
            ]
        ]
        original_linear = torch.nn.functional.linear
        original_matmul = torch.matmul
        self.addCleanup(setattr, torch.nn.functional, "linear", original_linear)
        self.addCleanup(setattr, torch, "matmul", original_matmul)
        voyager.transform(converted, (x,), patterns=patterns)
        actual = converted(x)

        fused_node = next(
            node for node in converted.graph.nodes if node.op == "call_module"
        )
        fused_module = converted.get_submodule(fused_node.target)
        mx_node = next(
            node
            for node in fused_module.graph.nodes
            if node.target == torch.ops.quantized_ops.conv2d_mx.default
        )

        self.assertEqual(fused_node.name, "conv_fused")
        self.assertEqual(mx_node.name, "conv")
        torch.testing.assert_close(actual, expected)

    def test_vsq_pt2e_converts_linear_and_matmul(self):
        torch.manual_seed(0)
        cases = [
            (
                TinyLinear().eval(),
                torch.randn(2, 4),
                torch.ops.quantized_ops.linear_mx.default,
            ),
            (
                TinyMatmul().eval(),
                torch.randn(2, 4),
                torch.ops.quantized_ops.matmul_mx.default,
            ),
        ]

        for model, x, expected_target in cases:
            with self.subTest(target=expected_target):
                exported = torch.export.export(model, (x,)).module()
                quantizer = voyager.get_default_quantizer(
                    input_activation=VSQ_ACTIVATION_SPEC,
                    weight=VSQ_WEIGHT_SPEC,
                )
                prepared = voyager.prepare_pt2e(exported, quantizer)
                prepared(x)
                prepared(x)
                for module in prepared.modules():
                    if isinstance(module, FakeQuantizeBase):
                        module.disable_observer()

                expected = prepared(x)
                converted = voyager.convert_pt2e(prepared)
                actual = converted(x)
                targets = [node.target for node in converted.graph.nodes]

                self.assertIn(expected_target, targets)
                self.assertEqual(
                    targets.count(torch.ops.quantized_ops.quantize_vsq.default),
                    1,
                )
                torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()

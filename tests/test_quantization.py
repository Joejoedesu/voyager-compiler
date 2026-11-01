"""Shared PT2E boundaries and target-specific conversion handoffs."""

import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from voyager_compiler.compiler.quantize import quantize
from voyager_compiler.extensions import (
    QuantizationPolicy,
    WorkloadSpec,
    describe_target,
)
from voyager_compiler.ir.graph import PreparedWorkload
from voyager_compiler.pipeline import compile_workload
from voyager_compiler.quantization.calibration import finalize_amax_observers
from voyager_compiler.quantization.fake_quantize import FusedAmaxObsFakeQuantize
from voyager_compiler.quantization.qspec import QScheme
from voyager_compiler.targets.gemmini.quantization import default_quantizer
from voyager_compiler.targets.gemmini.target import TARGET


class QuantizationTests(unittest.TestCase):
    def test_mx_outlier_product_rounds_after_addition(self):
        # 257 rounds to 256 in BF16. Adding the -256 outlier first must
        # retain 1, instead of rounding the dense and sparse paths apart.
        a = torch.tensor([[256, 1]], dtype=torch.bfloat16)
        b = torch.ones(2, 1, dtype=torch.bfloat16)
        result = torch.ops.quantized_ops.matmul_mx(
            a,
            b,
            A_data=torch.tensor([-256], dtype=torch.bfloat16),
            A_indices=torch.tensor([0]),
            A_indptr=torch.tensor([0, 1]),
        )
        self.assertEqual(result.item(), 1)
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_layout_spellings_and_convolution_output_precision(self):
        from voyager_compiler.ops.layout import GemmLayout

        a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        b = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        for old, new in (("xc,ck", "xc,ck->xk"), ("cx,ck,kx", "cx,ck->kx")):
            with self.subTest(layout=new):
                self.assertEqual(GemmLayout.parse(old), GemmLayout.parse(new))
                x = a.T if old.startswith("cx") else a
                result = torch.ops.quantized_ops.matmul(x, b, layout=new)
                expected = (a @ b).T if new.endswith("kx") else a @ b
                torch.testing.assert_close(result, expected)

        x = torch.tensor([256, 1], dtype=torch.bfloat16).reshape(1, 2, 1, 1)
        w = torch.ones(1, 2, 1, 1, dtype=torch.bfloat16)
        for layout in ("nchw", "nhwc"):
            with self.subTest(layout=layout):
                result = torch.ops.quantized_ops.conv2d(
                    x if layout == "nchw" else x.permute(0, 2, 3, 1),
                    w if layout == "nchw" else w.permute(2, 3, 1, 0),
                    layout=layout,
                    out_dtype=torch.float32,
                )
                self.assertEqual(result.dtype, torch.float32)
                self.assertEqual(result.item(), 257)
        # The formerly generated conv twin remains NHWC when unspecified.
        legacy = torch.ops.quantized_ops.conv2d(
            x.permute(0, 2, 3, 1), w.permute(2, 3, 1, 0)
        )
        self.assertEqual(legacy.item(), 256)

    def test_shared_stage_order_and_explicit_quantizer_override(self):
        events = []

        class Graph:
            def __call__(self, x):
                events.append("execute")
                return x

        g = Graph()
        policy = QuantizationPolicy(
            lambda p, o: self.fail("explicit quantizer must win"),
            prepare=lambda g, o: events.append("prepared"),
            finalize=lambda g, o: events.append("finalized"),
            bias="int32",
        )
        p = PreparedWorkload(g, (torch.ones(1),))
        with (
            patch(
                "voyager_compiler.compiler.quantize.prepare_pt2e",
                side_effect=lambda g, q: events.append("prepare") or g,
            ),
            patch(
                "voyager_compiler.compiler.quantize.convert_pt2e",
                side_effect=lambda *a, **k: events.append("convert"),
            ),
        ):
            quantize(
                p,
                SimpleNamespace(quantization=policy),
                dict(quantizer=object(), calibration=[(p.args, {})]),
            )
        self.assertEqual(
            events,
            [
                "prepare",
                "prepared",
                "execute",
                "finalized",
                "convert",
                "execute",
            ],
        )
        events.clear()
        quantize(p, SimpleNamespace(quantization=policy), {"quantizer": None})
        self.assertEqual(events, [])

    def test_static_finalize_includes_last_sample(self):
        observer = FusedAmaxObsFakeQuantize(
            "int8",
            quant_max=127,
            amax_history_len=64,
            qscheme=QScheme.PER_TENSOR_SYMMETRIC,
            power_2_scale=True,
        )
        observer.disable_fake_quant()
        observer(torch.tensor([1.0]))
        observer(torch.tensor([64.0]))
        finalize_amax_observers(observer, {})
        self.assertEqual(observer.scale.item(), 1.0)

    def test_integer_rounding_does_not_round_through_bfloat16(self):
        x = torch.tensor([2.501, 2.5, 3.5, -2.501, 300.0])
        actual = torch.ops.quantized_ops.quantize_symmetric(
            x, torch.tensor(1.0), -128, 127
        )
        self.assertEqual(actual.tolist(), [3.0, 2.0, 4.0, -3.0, 127.0])

    def test_per_module_override_survives_shared_conversion_and_lowering(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(5, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)
        ).eval()
        x = torch.tensor([[4.0, 1.0, -2.0, 0.0, 3.0]])
        # The default is a configurable shared quantizer, not an alternate frontend.
        q = default_quantizer(
            PreparedWorkload(torch.fx.symbolic_trace(model), (x,)), {}
        )
        config = q.global_config
        q.set_module_name(
            "0",
            replace(
                config,
                input_activation=replace(config.input_activation, quant_max=63),
            ),
        )
        with tempfile.TemporaryDirectory() as out:
            result = compile_workload(
                WorkloadSpec(
                    "scoped", lambda _: model, lambda m, o: ((x,), {})
                ),
                TARGET,
                out,
                options={"quantizer": q},
                dump_tensors=False,
            )
            self.assertEqual(result.args[0][0, 0].item(), 32)
            self.assertEqual(result.args[0].dtype, torch.int8)
        bindings = describe_target(TARGET)
        self.assertIn("quantization", bindings)
        self.assertTrue(
            bindings["bindings"]["lower"].endswith("lower_operations")
        )
        self.assertNotIn("numerics", bindings["bindings"])

    def test_non_gemm_output_override_is_preserved(self):
        from voyager_compiler.export_utils import export_model
        from voyager_compiler.quantization import prepare_pt2e

        model = torch.nn.Sequential(
            torch.nn.Linear(5, 4), torch.nn.ReLU()
        ).eval()
        args = (torch.tensor([[4.0, 1.0, -2.0, 0.0, 3.0]]),)
        graph = export_model(model, args, {})
        q = default_quantizer(PreparedWorkload(graph, args), {})
        config = q.global_config
        q.set_module_name(
            "1",
            replace(
                config,
                output_activation=replace(
                    config.output_activation, quant_max=31
                ),
            ),
        )
        prepared = prepare_pt2e(graph, q)
        relu = next(
            n
            for n in prepared.graph.nodes
            if n.target == torch.ops.aten.relu.default
        )
        observer = prepared.get_submodule(next(iter(relu.users)).target)
        self.assertEqual(observer.quant_max, 31)
        input_observer = prepared.get_submodule(relu.args[0].target)
        self.assertEqual(input_observer.quant_max, 127)
        self.assertIsNot(observer, input_observer)

    def test_unsupported_precision_is_rejected_not_silently_int8(self):
        model = torch.nn.Linear(5, 4).eval()
        x = torch.ones(1, 5)
        from voyager_compiler.quantization import get_default_quantizer

        q = get_default_quantizer(
            input_activation="int4,qs=per_tensor_symmetric,pot=1",
            output_activation="int4,qs=per_tensor_symmetric,pot=1",
            weight="int4,qs=per_tensor_symmetric,pot=1",
            bias="int32",
        )
        with (
            tempfile.TemporaryDirectory() as out,
            self.assertRaisesRegex(NotImplementedError, "INT8"),
        ):
            compile_workload(
                WorkloadSpec("int4", lambda _: model, lambda m, o: ((x,), {})),
                TARGET,
                out,
                options={"quantizer": q},
                dump_tensors=False,
            )


if __name__ == "__main__":
    unittest.main()

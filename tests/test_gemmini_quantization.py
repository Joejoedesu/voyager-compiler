"""Gemmini quantization domains, boundary folding and bounded activations."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from voyager_compiler.codegen.transform.rewrites import (
    replace_conv2d_with_im2col,
)
from voyager_compiler.export_utils import export_model
from voyager_compiler.extensions import WorkloadSpec
from voyager_compiler.ir.graph import PreparedWorkload
from voyager_compiler.pipeline import compile_workload
from voyager_compiler.quantization import get_default_quantizer, prepare_pt2e
from voyager_compiler.runtime import run
from voyager_compiler.targets.gemmini.quantization import Int8Quantizer
from voyager_compiler.targets.gemmini.target import TARGET


def quantizer():
    spec = "int8,qs=per_tensor_symmetric,qmax=127,pot=0"
    common = get_default_quantizer(
        input_activation=spec, output_activation=spec, weight=spec, bias="int32"
    )
    return Int8Quantizer(local_domains=True).set_global(
        common.object_type_config[torch.ops.aten.conv2d.default]
    )


class Residual(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Conv2d(3, 4, 1)
        self.first = torch.nn.Conv2d(4, 4, 1)
        self.second = torch.nn.Conv2d(4, 4, 1)

    def forward(self, x):
        x = self.stem(x).relu()
        x = (self.first(x) + x).relu()
        return (self.second(x) + x).relu()


class QuantizationTests(unittest.TestCase):
    def test_local_residual_domains_do_not_merge_through_sibling_consumers(
        self,
    ):
        model = Residual().eval()
        graph = prepare_pt2e(
            export_model(model, (torch.randn(1, 3, 4, 4),)), quantizer()
        )
        domains = []
        for node in graph.graph.nodes:
            if node.target != torch.ops.aten.relu.default:
                continue
            source = node.args[0]
            if source.op == "call_module":
                source = source.args[0]
            if source.target != torch.ops.aten.add.Tensor:
                continue
            observer = next(iter(node.users))
            self.assertEqual(observer.op, "call_module")
            domains.append(graph.get_submodule(observer.target))
        self.assertEqual(len(domains), 2)
        self.assertIsNot(domains[0], domains[1])

    def test_disabled_module_remains_unquantized(self):
        graph = export_model(Residual().eval(), (torch.randn(1, 3, 4, 4),))
        q = quantizer().set_module_name("second", None)
        q.annotate(graph)
        second = next(
            n
            for n in graph.graph.nodes
            if n.target == torch.ops.aten.conv2d.default
            and "second" in str(n.args[1])
        )
        annotation = second.meta["quantization_annotation"]
        self.assertIsNone(annotation.output_qspec)
        self.assertTrue(
            all(s is None for s in annotation.input_qspec_map.values())
        )

    def test_packed_input_flows_through_shared_quantization_and_native_matmul(
        self,
    ):
        torch.manual_seed(12)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 4),
        ).eval()
        args = (torch.randn(1, 3, 5, 5),)

        def prepare(model, inputs, kwargs, options):
            expected = model(*inputs)
            graph = export_model(model, inputs)
            replace_conv2d_with_im2col(graph)
            torch.testing.assert_close(graph(*inputs), expected)
            return PreparedWorkload(
                graph, inputs, kwargs, "packed-stem", expected, {}
            )

        with tempfile.TemporaryDirectory() as out:
            result = compile_workload(
                WorkloadSpec(
                    "packed-stem",
                    lambda _: model,
                    lambda m, o: (args, {}),
                    prepare=prepare,
                ),
                TARGET,
                out,
                options={"quantizer": quantizer(), "allow_input_packing": True},
                dump_tensors=False,
            )
            run(result, reference=result.reference, rtol=0, atol=0)
            self.assertEqual(tuple(result.args[0].shape), (1, 25, 27))
            self.assertEqual(result.args[0].dtype, torch.int8)
            choices = list(result.program.plan.choices.values())
            self.assertFalse(any("conv2d" in c.implementation for c in choices))
            self.assertTrue(
                choices[0].implementation.startswith("gemmini.matmul.")
            )
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaisesRegex(
                NotImplementedError, "allow_input_packing"
            ):
                compile_workload(
                    WorkloadSpec(
                        "packing-not-permitted",
                        lambda _: model,
                        lambda m, o: (args, {}),
                        prepare=prepare,
                    ),
                    TARGET,
                    out,
                    options={"quantizer": quantizer()},
                    dump_tensors=False,
                )

    def test_average_scales_sum_before_single_rounding(self):
        x = torch.tensor([[[[1]], [[2]], [[4]]]], dtype=torch.int8)
        got = torch.ops.gemmini.average.default(x, 0.6)
        self.assertEqual(got.item(), 1)  # round((1+2+4) * .6/3)
        # Rounding the average first would instead produce round(2 * .6)=1;
        # this second multiplier distinguishes the two boundaries.
        self.assertEqual(torch.ops.gemmini.average.default(x, 1.2).item(), 3)

    def test_residual_requantization_is_realized_in_loads(self):
        class Network(Residual):
            def forward(self, x):
                y = super().forward(x)
                return torch.nn.functional.adaptive_avg_pool2d(y, 1).flatten(1)

        torch.manual_seed(7)
        model = Network().eval()
        args = (torch.randn(1, 3, 4, 4),)
        with tempfile.TemporaryDirectory() as out:
            result = compile_workload(
                WorkloadSpec(
                    "residual-domains", lambda _: model, lambda m, o: (args, {})
                ),
                TARGET,
                out,
                options={"quantizer": quantizer()},
                dump_tensors=False,
            )
            run(result, reference=result.reference, rtol=0, atol=0)
            nodes = list(result.program.graph.graph.nodes)
            self.assertFalse(
                any(
                    n.target == torch.ops.gemmini.requantize.default
                    for n in nodes
                )
            )
            adds = [
                n for n in nodes if n.target == torch.ops.gemmini.add.default
            ]
            self.assertEqual(len(adds), 2)
            self.assertTrue(any(n.args[2] != 1 or n.args[3] != 1 for n in adds))

    def test_relu_fusion_does_not_modify_shared_unclamped_output(self):
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 4)

            def forward(self, x):
                y = self.fc(x)
                return y, y.relu()

        torch.manual_seed(9)
        model = Network().eval()
        args = (torch.randn(2, 4),)
        with tempfile.TemporaryDirectory() as out:
            result = compile_workload(
                WorkloadSpec(
                    "shared-output", lambda _: model, lambda m, o: (args, {})
                ),
                TARGET,
                out,
                options={"quantizer": quantizer()},
                dump_tensors=False,
            )
            outputs, _ = run(result, reference=result.reference, rtol=0, atol=0)
            self.assertTrue((outputs[0] < 0).any())
            self.assertTrue((outputs[1] >= 0).all())


class GemminiRelu6Tests(unittest.TestCase):
    def test_relu6_uses_accumulator_scale_without_pre_activation_clip(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, 1, bias=False),
            torch.nn.ReLU6(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        ).eval()
        with torch.no_grad():
            model[0].weight.fill_(10)
        x = torch.tensor([-2.0, 0.0, 0.3, 0.6, 1.0, 4.0]).reshape(1, 1, 2, 3)
        with tempfile.TemporaryDirectory() as out:
            result = compile_workload(
                WorkloadSpec("relu6", lambda _: model, lambda m, o: ((x,), {})),
                TARGET,
                out,
                dump_tensors=False,
            )
            run(result, reference=result.reference, rtol=0, atol=0)
            stream = json.loads(
                (Path(out) / "gemmini/commands.json").read_text()
            )
            self.assertTrue(
                any(
                    b["implementation"] == "gemmini.conv2d.hardware_loop"
                    for b in stream["bodies"]
                )
            )
            # A following domain conversion is allowed; the convolution itself
            # must clamp directly from the accumulator, with fused ReLU.
            graph = result.program.graph
            conv = next(
                n
                for n in graph.graph.nodes
                if n.target == torch.ops.gemmini.conv2d.default
            )
            self.assertTrue(conv.kwargs.get("relu"))


if __name__ == "__main__":
    unittest.main()

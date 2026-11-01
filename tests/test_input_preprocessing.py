"""Selected input representations, calibration and selective im2col rewrites."""

import tempfile
import unittest
from dataclasses import replace

import torch

from voyager_compiler.codegen.transform.rewrites import (
    replace_conv2d_with_im2col,
)
from voyager_compiler.compiler.preprocess import preprocess_inputs
from voyager_compiler.export_utils import export_model
from voyager_compiler.extensions import (
    QuantizationPolicy,
    WorkloadSpec,
    describe_target,
    registry,
)
from voyager_compiler.ir.graph import PreparedWorkload
from voyager_compiler.pipeline import compile_workload
from voyager_compiler.runtime import run


class PackedRelu(torch.nn.Module):
    def forward(self, packed):
        return torch.relu(packed).T


class InputPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.target = registry.target("voyager")
        self.x = torch.arange(-16.0, 16.0).reshape(4, 8)
        self.config = self.target.hardware(
            pe_array_size=(4, 4),
            scratchpad_size=4096,
            num_banks=4,
            bank_width=4,
            vector_unit_width=4,
        )

    def test_preprocessing_precedes_quantization_and_adapts_calibration(self):
        events = []
        samples = [(self.x, self.x + 2)]

        def calibration(prepared, options):
            events.append("samples")
            torch.testing.assert_close(prepared.args[0], self.x)
            return [((x,), {}) for x in samples[0]]

        def rewrite(prepared, config, options):
            events.append("preprocessing")
            self.assertIs(config, self.config)
            packed = prepared.args[0].T.contiguous()
            # A replacement PreparedWorkload need not copy asset provenance.
            options["calibration"] = [
                ((args[0].T.contiguous(),), kwargs)
                for args, kwargs in options["calibration"]
            ]
            return PreparedWorkload(
                export_model(PackedRelu(), (packed,)),
                (packed,),
                provenance={"input_preprocessing": {"name": "packed-input"}},
            )

        def quantization(prepared, config, options):
            events.append("quantization")
            for (args, kwargs), raw in zip(options["calibration"], samples[0]):
                torch.testing.assert_close(
                    prepared.graph(*args, **kwargs), torch.relu(raw)
                )
            # Mutating the working reference must not destroy the original.
            prepared.reference.zero_()
            prepared.provenance["quantization"] = {"flow": "identity-test"}
            return prepared

        target = replace(
            self.target,
            input_preprocessing=rewrite,
            quantization=QuantizationPolicy(run=quantization),
        )
        spec = WorkloadSpec(
            "packed",
            lambda _: torch.nn.ReLU(),
            lambda m, o: ((self.x,), {}),
            calibration=calibration,
        )
        caller_options = {}
        with tempfile.TemporaryDirectory() as directory:
            result = compile_workload(
                spec,
                target,
                directory,
                config=self.config,
                options=caller_options,
                dump_tensors=False,
            )
            run(result, reference=torch.relu(self.x), rtol=0, atol=0)
        self.assertEqual(events, ["samples", "preprocessing", "quantization"])
        self.assertEqual(caller_options, {})
        torch.testing.assert_close(result.args[0], self.x.T)
        torch.testing.assert_close(
            result.original_reference, torch.relu(self.x)
        )
        self.assertIsNotNone(
            describe_target(target)["bindings"]["input_preprocessing"]
        )

    def test_caller_override_disable_and_provenance(self):
        def forbidden(*args):
            self.fail("target default must not run")

        target = replace(self.target, input_preprocessing=forbidden)
        prepared = PreparedWorkload(
            torch.nn.ReLU(), (self.x,), provenance={"asset": "checksum"}
        )
        self.assertIs(
            preprocess_inputs(
                prepared, target, self.config, {"input_preprocessing": None}
            ),
            prepared,
        )

        def alternate(p, config, options):
            return PreparedWorkload(
                p.graph,
                p.args,
                provenance={"input_preprocessing": {"name": "unchanged-input"}},
            )

        result = preprocess_inputs(
            prepared, target, self.config, {"input_preprocessing": alternate}
        )
        self.assertEqual(result.provenance["asset"], "checksum")
        torch.testing.assert_close(result.reference, torch.relu(self.x))

    def test_invalid_callback_results_reject(self):
        prepared = PreparedWorkload(torch.nn.ReLU(), (self.x,))
        for callback, error, message in [
            (False, TypeError, "callable or None"),
            (lambda *a: None, TypeError, "return PreparedWorkload"),
            (lambda *a: prepared, ValueError, "record its provenance"),
        ]:
            with (
                self.subTest(callback=callback),
                self.assertRaisesRegex(error, message),
            ):
                preprocess_inputs(
                    prepared,
                    self.target,
                    self.config,
                    {"input_preprocessing": callback},
                )


class TwoConvolutions(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.other = torch.nn.Conv2d(3, 5, 1)

    def forward(self, x):
        return self.first(x), self.other(x)


class InputRewriteTests(unittest.TestCase):
    def make_graph(self):
        torch.manual_seed(1)
        x = torch.randn(1, 3, 7, 7)
        model = TwoConvolutions().eval()
        graph = export_model(model, (x,))
        expected = model(x)
        return graph, x, expected

    def convolutions(self, graph):
        return [
            n
            for n in graph.graph.nodes
            if n.target == torch.ops.aten.conv2d.default
        ]

    def test_binding_selects_one_convolution(self):
        graph, x, expected = self.make_graph()
        first, other = self.convolutions(graph)
        replace_conv2d_with_im2col(graph, predicate=lambda n: n is first)
        self.assertEqual(self.convolutions(graph), [other])
        self.assertTrue(
            any(
                n.target == torch.ops.aten.im2col.default
                for n in graph.graph.nodes
            )
        )
        torch.testing.assert_close(graph(x), expected)

    def test_declined_rewrite_preserves_graph_and_weights(self):
        graph, x, expected = self.make_graph()
        before = str(graph.graph)
        weights = {k: v.detach().clone() for k, v in graph.state_dict().items()}
        replace_conv2d_with_im2col(graph, predicate=lambda n: False)
        self.assertEqual(str(graph.graph), before)
        torch.testing.assert_close(graph.state_dict(), weights, rtol=0, atol=0)
        torch.testing.assert_close(graph(x), expected)

    def test_default_still_rewrites_all_eligible_convolutions(self):
        graph, x, expected = self.make_graph()
        replace_conv2d_with_im2col(graph)
        self.assertEqual(self.convolutions(graph), [])
        torch.testing.assert_close(graph(x), expected)

    def test_explicit_reduction_padding_preserves_values_and_materializes_weight(
        self,
    ):
        graph, x, expected = self.make_graph()
        replace_conv2d_with_im2col(graph, reduction_multiple=16)
        torch.testing.assert_close(graph(x), expected)
        linears = [
            n
            for n in graph.graph.nodes
            if n.target == torch.ops.aten.linear.default
        ]
        self.assertEqual(len(linears), 2)
        for node in linears:
            weight = getattr(graph, node.args[1].target)
            self.assertEqual(weight.shape[-1] % 16, 0)
            self.assertEqual(node.args[0].target, torch.ops.aten.pad.default)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            replace_conv2d_with_im2col(graph, reduction_multiple=0)


if __name__ == "__main__":
    unittest.main()

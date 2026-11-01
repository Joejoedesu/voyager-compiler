"""Convolution connectivity is shared; execution policy belongs to a target."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from voyager_compiler.codegen.node_info import (
    is_depthwise_conv,
    is_grouped_conv,
)
from voyager_compiler.compiler.search.interstellar import _prepare_search
from voyager_compiler.export_utils import export_model
from voyager_compiler.shape_prop import ShapeProp


class DepthwiseDispatchTests(unittest.TestCase):
    def node(self, channels, outputs, groups):
        args = (torch.randn(1, channels, 5, 5),)
        gm = export_model(
            torch.nn.Conv2d(channels, outputs, 3, groups=groups).eval(), args
        )
        ShapeProp(gm).propagate(*args)
        return next(
            n
            for n in gm.graph.nodes
            if n.target == torch.ops.aten.conv2d.default
        )

    def test_connectivity_and_layout_are_not_hardware_policy(self):
        for channels, outputs, groups, grouped, depthwise in (
            (4, 8, 1, False, False),
            (4, 8, 2, True, False),
            (4, 4, 4, True, True),
            (4, 8, 4, True, True),
        ):
            with self.subTest(groups=groups, outputs=outputs):
                node = self.node(channels, outputs, groups)
                self.assertEqual(is_grouped_conv(node), grouped)
                self.assertEqual(is_depthwise_conv(node), depthwise)
                # NHWC activations still have grouped OIHW weights.
                node.args[0].value = node.args[0].value.permute(0, 2, 3, 1)
                node.args[0].shape = node.args[0].value.shape
                node.meta["transposed"] = True
                self.assertEqual(is_depthwise_conv(node), depthwise)

    def test_shared_search_consults_target_for_depthwise(self):
        node = self.node(4, 4, 4)
        operation = SimpleNamespace(space=object())
        model = SimpleNamespace(prepare=Mock(return_value=operation))
        self.assertEqual(
            _prepare_search(node, SimpleNamespace(model=model), None),
            (operation, None),
        )
        model.prepare.assert_called_once_with(node)

    def test_voyager_has_explicit_disjoint_recipes(self):
        from voyager_compiler.targets.voyager.target import TARGET

        for groups, expected in (
            (1, "voyager.conv2d"),
            (2, "voyager.grouped"),
            (4, "voyager.depthwise"),
        ):
            node = self.node(4, 4, groups)
            matches = [
                i.name for i in TARGET.implementations if i.matches(node, node)
            ]
            self.assertEqual(matches, [expected])

    def test_padding_preserves_group_membership_and_multiplier(self):
        from voyager_compiler.codegen.transform.padding import (
            pad_matrix_op_dimensions,
        )

        for channels, outputs, groups in ((6, 10, 2), (5, 10, 5)):
            with self.subTest(groups=groups):
                model = torch.nn.Conv2d(
                    channels, outputs, 3, groups=groups
                ).eval()
                args = (torch.randn(1, channels, 5, 5),)
                gm = export_model(model, args)
                ShapeProp(gm).propagate(*args)
                expected = gm(*args)
                pad_matrix_op_dimensions(gm, (4, 4))
                torch.testing.assert_close(gm(*args), expected)

    def test_voyager_whole_recipe_preserves_fused_depthwise_epilogue(self):
        from voyager_compiler.codegen.subgraph import create_and_insert_subgraph
        from voyager_compiler.hardware import AcceleratorConfig
        from voyager_compiler.pipeline import compile_bufferized
        from voyager_compiler.runtime import run
        from voyager_compiler.targets.voyager.target import TARGET

        model = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 3, groups=4), torch.nn.ReLU()
        ).eval()
        args = (torch.randn(1, 4, 5, 5),)
        gm = export_model(model, args)
        ShapeProp(gm).propagate(*args)
        create_and_insert_subgraph(
            [n for n in gm.graph.nodes if n.op == "call_function"], gm
        )
        config = AcceleratorConfig(
            pe_array_size=(4, 4),
            scratchpad_size=16384,
            num_banks=4,
            bank_width=4,
            vector_unit_width=4,
        )
        with torch.no_grad():
            expected = gm(*args)
        with tempfile.TemporaryDirectory() as out:
            result = compile_bufferized(
                gm,
                args,
                config=config,
                output_dir=out,
                target=TARGET,
                dump_tensors=False,
            )
            run(result, args, reference=expected)
            self.assertIn(
                "voyager.depthwise",
                [
                    c.implementation
                    for c in result.program.plan.choices.values()
                ],
            )


if __name__ == "__main__":
    unittest.main()

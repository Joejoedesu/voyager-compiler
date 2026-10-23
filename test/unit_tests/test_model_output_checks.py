# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import unittest

import torch

from test.verify_help import (
    assert_output_close,
    compare_output_statistics,
)
from voyager_compiler import export_model
from voyager_compiler.pt2e_utils import convert_scalars_to_attrs
from voyager_compiler.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
)


class _ScalarArithmetic(torch.nn.Module):
    def forward(self, value):
        return ((value + 2.0) * 3.0) / 4.0


class ModelOutputCheckTest(unittest.TestCase):
    def test_scalar_arithmetic_operands_become_get_attrs(self):
        inputs = (torch.ones(2),)
        model = export_model(_ScalarArithmetic(), inputs)
        expected = model(*inputs)

        XNNPACKQuantizer().transform_for_annotation(model)
        arithmetic = tuple(
            node
            for node in model.graph.nodes
            if node.target
            in {
                torch.ops.aten.add.Tensor,
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.div.Tensor,
            }
        )
        constants = tuple(
            node for node in model.graph.nodes if node.op == "get_attr"
        )

        self.assertEqual(len(arithmetic), 3)
        self.assertEqual(len(constants), 3)
        self.assertEqual(
            tuple(node.target for node in constants),
            (
                "_tensor_constant_0",
                "_tensor_constant_1",
                "_tensor_constant_2",
            ),
        )
        self.assertTrue(
            all(
                all(isinstance(arg, torch.fx.Node) for arg in node.args)
                for node in arithmetic
            )
        )
        torch.testing.assert_close(model(*inputs), expected)

    def test_scalar_attr_names_skip_existing_buffers(self):
        inputs = (torch.ones(2),)
        model = export_model(_ScalarArithmetic(), inputs)
        expected = model(*inputs)
        model.register_buffer(
            "_tensor_constant_0",
            torch.tensor(-1.0),
        )
        model.register_buffer(
            "_tensor_constant_2",
            torch.tensor(-2.0),
        )

        convert_scalars_to_attrs(model)
        constants = tuple(
            node.target for node in model.graph.nodes if node.op == "get_attr"
        )
        buffers = tuple(name for name, _ in model.named_buffers())

        self.assertEqual(
            constants,
            (
                "_tensor_constant_1",
                "_tensor_constant_3",
                "_tensor_constant_4",
            ),
        )

        convert_scalars_to_attrs(model)
        self.assertEqual(
            tuple(
                node.target
                for node in model.graph.nodes
                if node.op == "get_attr"
            ),
            constants,
        )
        self.assertEqual(
            tuple(name for name, _ in model.named_buffers()),
            buffers,
        )
        torch.testing.assert_close(model(*inputs), expected)

    def test_statistics_cover_every_output(self):
        reference = (
            torch.tensor([1.0, 2.0, 3.0]),
            [torch.tensor([4.0, 5.0])],
        )
        close = (
            reference[0] + 0.01,
            [reference[1][0] - 0.01],
        )
        statistics = compare_output_statistics(
            reference,
            close,
            label="unit-test quantization",
            minimum_cosine=0.99,
            minimum_sqnr_db=20.0,
        )
        self.assertEqual(len(statistics), 2)

        bad_second_output = (reference[0], [torch.zeros(2)])
        with self.assertRaisesRegex(AssertionError, r"output\[1\]"):
            compare_output_statistics(
                reference,
                bad_second_output,
                label="unit-test quantization",
                minimum_cosine=0.99,
                minimum_sqnr_db=20.0,
            )

    def test_strict_comparison_covers_every_output(self):
        reference = [torch.ones(2), torch.ones(2)]
        actual = [torch.ones(2), torch.tensor([1.0, 1.1])]
        with self.assertRaises(AssertionError) as context:
            assert_output_close(
                reference,
                actual,
                label="unit-test rewrite",
                rtol=1e-5,
                atol=1e-6,
            )
        message = str(context.exception)
        self.assertIn("output[1]", message)
        self.assertIn("maximum absolute deviation=", message)
        self.assertIn("RMSE=", message)


if __name__ == "__main__":
    unittest.main()

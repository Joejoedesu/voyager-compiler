"""The selected FP8 recipe must affect emitted NKI, with unchanged semantics."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from voyager_compiler.extensions import WorkloadSpec
from voyager_compiler.pipeline import compile_workload
from voyager_compiler.runtime import run


class TrainiumAlternativeTests(unittest.TestCase):
    def test_fp8_single_and_double_row(self):
        from voyager_compiler.targets.trainium.target import TARGET

        class Matrix(torch.nn.Module):
            def forward(self, a, b):
                return torch.ops.quantized_ops.matmul.default(
                    a, b, out_dtype=torch.float32
                )

        torch.manual_seed(7)
        args = tuple(
            torch.randint(-3, 4, shape).to(torch.float8_e4m3fn)
            for shape in ((128, 256), (256, 128))
        )
        spec = WorkloadSpec(
            "fp8-matrix", lambda _: Matrix(), lambda m, o: (args, {})
        )
        for recipe in TARGET.implementations[:2]:
            with (
                self.subTest(recipe=recipe.name),
                tempfile.TemporaryDirectory() as directory,
            ):
                result = compile_workload(
                    spec,
                    replace(TARGET, implementations=(recipe,)),
                    directory,
                    dump_tensors=False,
                )
                run(result, reference=result.reference, rtol=0, atol=0)
                kernel = (Path(directory) / "nki/kernels.py").read_text()
                self.assertEqual(
                    "perf_mode=nisa.matmul_perf_mode.double_row" in kernel,
                    recipe.name.endswith("double_row"),
                )
                self.assertTrue(
                    any(
                        c.implementation == recipe.name
                        for c in result.program.plan.choices.values()
                        if c
                    )
                )

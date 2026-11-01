"""Saved constants and selected FP8 recipes must survive independent NKI emission."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_pipeline import assert_saved_emission

from voyager_compiler.extensions import WorkloadSpec
from voyager_compiler.pipeline import compile_workload
from voyager_compiler.runtime import run


class TrainiumAlternativeTests(unittest.TestCase):
    def test_vector_partition_spreading_and_broadcast_legality(self):
        from voyager_compiler.targets.trainium.vector import (
            TrainiumVector,
            on_reading_partitions,
        )

        # Exercise the placement rule with multi-row tiles and broadcast
        # operands; a row reduction must not become a flat elementwise op.
        for inputs, row_op, expected in (
            ([(2, 128)], None, (128, True)),
            ([(2, 128), (1, 128)], None, (2, False)),
            ([(2, 128)], torch.ops.aten.rms_norm.default, (2, False)),
        ):
            model = SimpleNamespace(
                flat=True, inputs=inputs, group=(None, row_op)
            )
            self.assertEqual(
                TrainiumVector.row_parts(model, (2, 128)), expected
            )
        self.assertTrue(on_reading_partitions((1, 2, 16), (4, 2, 16)))
        self.assertFalse(on_reading_partitions((2, 1, 16), (2, 4, 16)))

    def test_partial_output_checker_and_older_manifests(self):
        import importlib.util
        import sys

        from voyager_compiler.targets.trainium import translate

        # Check comparison semantics without claiming NKI simulation:
        # inject a kernel result containing deliberately unwritten garbage.
        path = Path(translate.__code__.co_filename).parent / "runtime/check.py"
        spec = importlib.util.spec_from_file_location(
            "trainium_check_test", path
        )
        checker = importlib.util.module_from_spec(spec)
        unused_dtypes = SimpleNamespace(
            float8_e4m3fn=None,
            float8_e4m3=None,
            float8_e5m2=None,
            bfloat16=None,
        )
        with patch.dict(
            sys.modules, {"nki": SimpleNamespace(), "ml_dtypes": unused_dtypes}
        ):
            spec.loader.exec_module(checker)
        call = dict(kernel="partial", inputs=[], outputs=["y"], ops=["op"])
        manifest = dict(steps=[call], dtypes={"y": "float32"})
        expected = {"y": torch.ones(4)}
        with patch.object(
            checker,
            "simulate",
            return_value=[torch.tensor([1.0, 9.0, 1.0, 9.0])],
        ):
            call["partial"] = {"y": [[0, 1], [2, 3]]}
            self.assertEqual(
                checker.check_calls(
                    None, None, manifest, expected, None, False
                ),
                0,
            )
            del call["partial"]
            self.assertEqual(
                checker.check_calls(
                    None, None, manifest, expected, None, False
                ),
                1,
            )

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
                assert_saved_emission(self, directory)
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

    def test_scalar_constants_survive_saved_emission(self):
        from voyager_compiler.artifacts import SavedBundle
        from voyager_compiler.targets.trainium.target import TARGET

        class AddConstant(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.register_buffer("constant", torch.tensor(value))

            def forward(self, x):
                return x + self.constant

        for value in (1.25, float("-inf")):
            with (
                self.subTest(value=value),
                tempfile.TemporaryDirectory() as directory,
            ):
                workload = WorkloadSpec(
                    "constant",
                    lambda _: AddConstant(value),
                    lambda m, o: ((torch.ones(128, 16),), {}),
                )
                result = compile_workload(
                    workload, TARGET, directory, dump_tensors=False
                )
                run(result, reference=result.reference, rtol=0, atol=0)
                saved = SavedBundle.load(directory)
                constants = saved.target_data["immediates"]
                self.assertTrue(constants)
                self.assertIn(
                    value,
                    [
                        saved.read_payload(key).item()
                        for key in constants.values()
                    ],
                )
                assert_saved_emission(self, directory)
                if value == float("-inf"):
                    self.assertIn(
                        "float('-inf')",
                        (Path(directory) / "nki/kernels.py").read_text(),
                    )

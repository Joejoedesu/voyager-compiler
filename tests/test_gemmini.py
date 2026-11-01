"""Both real ISA families must replay and encode their selected ownership."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from test_pipeline import assert_saved_emission

from voyager_compiler.extensions import WorkloadSpec
from voyager_compiler.pipeline import compile_workload
from voyager_compiler.runtime import run


class GemminiTests(unittest.TestCase):
    def test_refinement_capacity_failure_selects_next_recipe(self):
        from voyager_compiler.compiler.allocate import MemoryCapacityError
        from voyager_compiler.ir.plan import Cost
        from voyager_compiler.targets.gemmini.target import TARGET

        class Matrix(torch.nn.Module):
            def forward(self, a, b):
                return torch._int_mm(a, b)

        args = (
            torch.ones((16, 16), dtype=torch.int8),
            torch.eye(16, dtype=torch.int8),
        )
        workload = WorkloadSpec(
            "replan-matrix", lambda _: Matrix(), lambda m, o: (args, {})
        )
        first, second = TARGET.implementations[:2]
        recipes = (
            replace(first, estimate=lambda *a: Cost(1)),
            replace(second, estimate=lambda *a: Cost(10)),
        )

        def refine(program, config):
            site, choice = next(iter(program.plan.choices.items()))
            if choice.implementation == first.name:
                raise MemoryCapacityError(
                    "refined fixture exceeds capacity", sites=(site,)
                )
            TARGET.refine(program, config)

        target = replace(TARGET, implementations=recipes, refine=refine)
        with tempfile.TemporaryDirectory() as directory:
            result = compile_workload(
                workload, target, directory, max_replans=1, dump_tensors=False
            )
            run(result, reference=torch._int_mm(*args), rtol=0, atol=0)
            self.assertEqual(len(result.program.validation["replans"]), 1)
            self.assertEqual(
                next(iter(result.program.plan.choices.values())).implementation,
                second.name,
            )
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(MemoryCapacityError),
        ):
            compile_workload(
                workload, target, directory, max_replans=0, dump_tensors=False
            )

    def test_explicit_and_loop_realizations(self):
        from voyager_compiler.targets.gemmini.target import TARGET

        class Matrix(torch.nn.Module):
            def forward(self, a, b):
                return torch._int_mm(a, b)

        args = (
            torch.arange(256, dtype=torch.int8).reshape(16, 16),
            torch.eye(16, dtype=torch.int8),
        )
        spec = WorkloadSpec(
            "integer-matrix", lambda _: Matrix(), lambda m, o: (args, {})
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
                run(result, reference=torch._int_mm(*args), rtol=0, atol=0)
                assert_saved_emission(self, directory)
                code = json.loads(
                    (Path(directory) / "gemmini/commands.json").read_text()
                )
                functs = [
                    int(c["instruction"], 16) >> 25 for c in code["commands"]
                ]
                if recipe.name.endswith("hardware_loop"):
                    self.assertEqual(functs[-6:], [9, 10, 11, 12, 13, 8])
                    self.assertEqual(
                        int(code["commands"][-1]["rs1"], 16) & 2, 2
                    )
                    self.assertTrue(
                        any(
                            b.owner == "native"
                            for b in result.program.plan.allocation.buffers
                        )
                    )
                else:
                    self.assertIn(6, functs)
                    self.assertIn(4, functs)
                    self.assertNotIn(8, functs)
                    self.assertEqual(
                        {
                            name
                            for name, _ in result.program.plan.allocation.arenas
                        },
                        {"scratchpad", "accumulator"},
                    )

"""Model calibration must hold out workload cases and preserve hardware fit."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from voyager_compiler.artifacts import hardware_digest
from voyager_compiler.hardware import AcceleratorConfig
from voyager_compiler.ir.plan import ExecutionOptions
from voyager_compiler.performance import Measurement, fit_profile


class PerformanceTests(unittest.TestCase):
    def sample(self, case, cycles):
        return Measurement(
            case,
            "matrix",
            "example",
            "physical-hash",
            cycles,
            (2 * cycles + 3) / 1e9,
            1,
            "device",
            "synthetic-test-fixture",
        )

    def test_fit_and_disjoint_holdout(self):
        profile, evidence = fit_profile(
            "example",
            [self.sample("train-a", 10), self.sample("train-b", 100)],
            [self.sample("test", 50)],
        )
        self.assertAlmostEqual(profile.correct("matrix", 50), 103)
        self.assertLess(evidence["max_relative_error"], 1e-12)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_profile(
                "leak", [self.sample("same", 10)], [self.sample("same", 30)]
            )

    def test_host_replay_is_not_accelerator_timing(self):
        with self.assertRaisesRegex(ValueError, "requires device"):
            replace(self.sample("case", 10), evidence_kind="host_replay")

    def test_software_buffering_does_not_change_physical_identity(self):
        hardware = AcceleratorConfig(
            scratchpad_size=4096, num_banks=4, bank_width=16
        )
        serial = replace(hardware, execution=ExecutionOptions(1, False))
        self.assertEqual(hardware_digest(hardware), hardware_digest(serial))
        memory = hardware.memory_instance("scratchpad")
        self.assertEqual(memory.buffering, 1)
        self.assertEqual((hardware.num_slots, serial.num_slots), (2, 1))

    def test_characterization_uses_compiled_cases_and_holdout(self):
        from voyager_compiler.extensions import WorkloadSpec
        from voyager_compiler.performance.characterize import (
            CharacterizationCase,
            characterize,
        )
        from voyager_compiler.targets.gemmini.target import TARGET

        class Matrix(torch.nn.Module):
            def forward(self, a, b):
                return torch._int_mm(a, b)

        workload = WorkloadSpec(
            "characterization-matrix",
            lambda _: Matrix(),
            lambda m, o: (
                (
                    torch.ones((o["rows"], 16), dtype=torch.int8),
                    torch.eye(16, dtype=torch.int8),
                ),
                {},
            ),
        )
        target = replace(TARGET, implementations=(TARGET.implementations[1],))
        family = target.implementations[0].name
        cases = [
            CharacterizationCase(
                str(rows),
                workload,
                family,
                (("rows", rows),),
                held_out=rows == 48,
            )
            for rows in (16, 32, 48)
        ]

        # This is a deterministic adapter fixture, not measured device evidence.
        def measure(result, case, config):
            self.assertTrue(result.program.plan.frozen)
            cycles = sum(
                c.cost.latency for c in result.program.plan.choices.values()
            )
            return Measurement(
                case.name,
                family,
                target.name,
                hardware_digest(config),
                cycles,
                (2 * cycles + 3) / 1e9,
                1,
                "cycle_accurate_simulator",
                "unit-test synthetic timing fixture",
            )

        with tempfile.TemporaryDirectory() as directory:
            profile, evidence = characterize(
                target, cases, Path(directory) / "measurements", measure=measure
            )
            self.assertEqual(profile.held_out_cases, ("48",))
            self.assertLess(evidence["max_relative_error"], 1e-12)

    def test_recipe_profile_supersedes_op_family_correction(self):
        from types import SimpleNamespace

        from voyager_compiler.compiler.bufferize.builders import BuildContext
        from voyager_compiler.compiler.search.planner import (
            select_implementation,
        )
        from voyager_compiler.compiler.search.tiler import Tiler
        from voyager_compiler.extensions import ImplementationSpec, RuntimeModel
        from voyager_compiler.ir.plan import Cost
        from voyager_compiler.performance.profile import (
            CalibratedModel,
            PerformanceProfile,
        )

        class Model(RuntimeModel):
            def prepare(self, node):
                return None

        config = AcceleratorConfig()
        base = Model(config, False)
        profile = PerformanceProfile(
            "test",
            "voyager",
            hardware_digest(config),
            (("recipe", 2, 3), ("Op", 7, 0)),
            ("a", "b"),
            ("c",),
            "device",
        )
        context = BuildContext(
            Tiler(CalibratedModel(base, profile), 0.02),
            1,
            False,
            False,
            True,
            False,
            True,
        )

        def build(node, trial, parameters):
            self.assertIs(trial.tiler.model, base)
            return "graph"

        recipe = ImplementationSpec(
            "recipe", lambda *a: True, build, estimate=lambda *a: Cost(100)
        )
        _, choice, _, _ = select_implementation(
            SimpleNamespace(name="n", meta={}),
            SimpleNamespace(meta={}),
            context,
            (recipe,),
        )
        self.assertEqual(choice.cost.latency, 203)


if __name__ == "__main__":
    unittest.main()

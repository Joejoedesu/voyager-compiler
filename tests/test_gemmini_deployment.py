"""Deployment coverage, numerical rules and selected command realization."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from voyager_compiler.extensions import WorkloadSpec
from voyager_compiler.pipeline import compile_workload
from voyager_compiler.runtime import run
from voyager_compiler.targets.gemmini.emission.replay import write_replay
from voyager_compiler.targets.gemmini.semantics import quant
from voyager_compiler.targets.gemmini.target import TARGET


class DeploymentTests(unittest.TestCase):
    def test_residual_rounds_and_saturates_before_addition(self):
        a = torch.tensor([[100, -100, 3, 5]], dtype=torch.int8)
        b = torch.tensor([[-100, 100, -3, -5]], dtype=torch.int8)
        got = torch.ops.gemmini.add.default(a, b, 2.0, 1.0)
        self.assertEqual(got.tolist(), [[27, -28, 3, 5]])
        self.assertEqual(
            quant(torch.tensor([0.5, 1.5, 2.5, -0.5, -1.5])).tolist(),
            [0, 2, 2, 0, -2],
        )

    def test_both_matmul_modes_use_selected_split_and_tail(self):
        class Matrix(torch.nn.Module):
            def forward(self, a, b, d):
                return torch.ops.gemmini.matmul.default(a, b, d, 0.125)

        torch.manual_seed(9)
        args = (
            torch.randint(-8, 8, (19, 35), dtype=torch.int8),
            torch.randint(-8, 8, (35, 21), dtype=torch.int8),
            torch.arange(-10, 11, dtype=torch.int32),
        )
        expected = (
            ((args[0].double() @ args[1].double() + args[2]).float() * 0.125)
            .round()
            .clamp(-128, 127)
            .to(torch.int8)
        )
        for mode in ("explicit", "hardware_loop"):
            recipe = next(
                r
                for r in TARGET.implementations
                if r.name == f"gemmini.matmul.{mode}"
            )
            recipe = replace(
                recipe,
                candidates=lambda n, c, mode=mode: (
                    {"tile": (16, 16, 16), "mode": mode},
                ),
            )
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as out:
                result = compile_workload(
                    WorkloadSpec(
                        "split-tail",
                        lambda _: Matrix(),
                        lambda m, o: (args, {}),
                    ),
                    replace(TARGET, implementations=(recipe,)),
                    out,
                    dump_tensors=False,
                )
                run(result, reference=expected, rtol=0, atol=0)
                code = json.loads(
                    (Path(out) / "gemmini/commands.json").read_text()
                )
                self.assertEqual(
                    code["bodies"][0]["parameters"]["tile"], [16, 16, 16]
                )
                functs = [
                    int(c["instruction"], 16) >> 25
                    for c in code["commands"]
                    if c["type"] == "command"
                ]
                self.assertEqual(8 in functs, mode == "hardware_loop")
                self.assertEqual(14 in functs, mode == "explicit")
                self.assertEqual(code["commands"][-1]["type"], "drain")
                write_replay(
                    result,
                    Path(out) / "replay",
                    Path(out) / "gemmini/commands.json",
                )
                self.assertNotEqual(
                    (Path(out) / "replay/initial.bin").read_bytes(),
                    (Path(out) / "replay/expected.bin").read_bytes(),
                )

    def test_connected_vision_uses_native_ops_and_no_float_fallback(self):
        class Vision(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 17, 3, padding=1)
                self.bn = torch.nn.BatchNorm2d(17)
                self.fc = torch.nn.Linear(17, 9)

            def forward(self, x):
                y = torch.relu(self.bn(self.conv(x)))
                y = y + y
                y = torch.nn.functional.max_pool2d(torch.relu(y), 3, 2, 1)
                y = torch.nn.functional.adaptive_avg_pool2d(y, 1)
                return self.fc(torch.flatten(y, 1))

        torch.manual_seed(8)
        model = Vision().eval()
        args = (torch.randn(1, 3, 8, 8),)
        with tempfile.TemporaryDirectory() as out:
            result = compile_workload(
                WorkloadSpec(
                    "vision", lambda _: model, lambda m, o: (args, {})
                ),
                TARGET,
                out,
                dump_tensors=False,
            )
            run(result, reference=result.reference, rtol=0, atol=0)
            self.assertEqual(result.args[0].dtype, torch.int8)
            self.assertEqual(tuple(result.args[0].shape), (1, 8, 8, 3))
            names = {
                c.implementation.split(".")[1]
                for c in result.program.plan.choices.values()
            }
            self.assertEqual(
                names,
                {"conv2d", "requantize", "add", "maxpool", "average", "matmul"},
            )
            code = json.loads((Path(out) / "gemmini/commands.json").read_text())
            self.assertEqual(
                len(code["bodies"]), len(result.program.plan.choices)
            )
            for body in code["bodies"]:
                self.assertEqual(
                    code["commands"][body["command_end"] - 1]["type"], "drain"
                )

    def test_unsupported_pool_is_rejected_before_emission(self):
        model = torch.nn.MaxPool2d(3, 2, 1).eval()
        args = (torch.randn(1, 3, 8, 8),)
        with tempfile.TemporaryDirectory() as out:
            with self.assertRaisesRegex(NotImplementedError, "nonnegative"):
                compile_workload(
                    WorkloadSpec(
                        "bad-pool", lambda _: model, lambda m, o: (args, {})
                    ),
                    TARGET,
                    out,
                    dump_tensors=False,
                )
            self.assertFalse((Path(out) / "gemmini").exists())

    def test_capacity_rejects_oversized_selected_tile(self):
        from voyager_compiler.extensions import CandidateRejected
        from voyager_compiler.targets.gemmini.scheduling import footprint

        with self.assertRaisesRegex(CandidateRejected, "capacity"):
            footprint(
                "matmul",
                {},
                (1024, 1024, 1024),
                "hardware_loop",
                TARGET.hardware(),
            )

    def test_search_compares_modes_for_non_aligned_matmul(self):
        class Matrix(torch.nn.Module):
            def forward(self, a, b, d):
                return torch.ops.gemmini.matmul.default(a, b, d, 1.0)

        args = (
            torch.ones(17, 31, dtype=torch.int8),
            torch.ones(31, 19, dtype=torch.int8),
            torch.zeros(19, dtype=torch.int32),
        )
        with tempfile.TemporaryDirectory() as out:
            result = compile_workload(
                WorkloadSpec(
                    "alternatives", lambda _: Matrix(), lambda m, o: (args, {})
                ),
                TARGET,
                out,
                dump_tensors=False,
            )
            choices = next(iter(result.program.plan.alternatives.values()))
            self.assertEqual(
                {c.implementation for c in choices},
                {"gemmini.matmul.explicit", "gemmini.matmul.hardware_loop"},
            )
            self.assertTrue(all(c.cost.latency > 0 for c in choices))


if __name__ == "__main__":
    unittest.main()

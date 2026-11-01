"""Deployment coverage and selected command realization."""

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
from voyager_compiler.targets.gemmini.emission.replay import write_replay
from voyager_compiler.targets.gemmini.semantics import quant
from voyager_compiler.targets.gemmini.target import TARGET


class DeploymentTests(unittest.TestCase):
    def test_interstellar_multirow_mapping_reaches_native_commands(self):
        from voyager_compiler.compiler.search.interstellar import search_matrix
        from voyager_compiler.targets.gemmini.emission._commands import conv
        from voyager_compiler.targets.gemmini.scheduling import (
            NativeMatrixModel,
            cost,
            selected_tile,
        )

        p = dict(
            batch=1,
            h=7,
            w=7,
            oh=7,
            ow=7,
            ci=512,
            co=512,
            kernel=3,
            stride=1,
            pad=1,
        )
        model = NativeMatrixModel(
            "conv2d", p, "hardware_loop", TARGET.hardware()
        )
        result = search_matrix(model, model.layer)
        tile = selected_tile("conv2d", p, result.tiling)
        self.assertEqual(tile, (7, 7, 128, 64))
        commands = conv(
            dict(p, tile=tile),
            dict(
                x=0x80000000, weight=0x80100000, bias=0x80400000, y=0x80500000
            ),
        )
        launch = [c for c in commands if int(c["instruction"], 16) >> 25 == 15]
        geometry = [
            int(c["rs2"], 16)
            for c in commands
            if int(c["instruction"], 16) >> 25 == 17
        ]
        self.assertEqual(len(launch), 32)
        self.assertTrue(
            all(
                ((g >> 32) & 65535, (g >> 16) & 65535, g & 65535) == (7, 7, 128)
                for g in geometry
            )
        )
        self.assertLess(
            result.cost.latency,
            cost("conv2d", p, (1, 7, 32, 256), "hardware_loop").latency,
        )

    def test_prime_matrix_extent_can_use_aligned_interior_tiles(self):
        from voyager_compiler.compiler.search.interstellar import search_matrix
        from voyager_compiler.targets.gemmini.scheduling import (
            NativeMatrixModel,
        )
        from voyager_compiler.targets.gemmini.semantics import selected_tile

        p = dict(m=1021, n=64, k=64)
        model = NativeMatrixModel(
            "matmul", p, "hardware_loop", TARGET.hardware()
        )
        result = search_matrix(model, model.layer)
        tile = selected_tile("matmul", p, result.tiling)
        self.assertGreater(tile[0], 1)
        self.assertEqual(tile[0] % 16, 0)
        self.assertLess(tile[0], p["m"])
        self.assertIsNotNone(model.fit(result.tiling))

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
                        lambda m, o: (args[:2], {"d": args[2]}),
                    ),
                    replace(TARGET, implementations=(recipe,)),
                    out,
                    dump_tensors=False,
                )
                run(result, reference=expected, rtol=0, atol=0)
                assert_saved_emission(self, out)
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
                {
                    "conv2d",
                    "add",
                    "maxpool",
                    "average",
                    "matmul",
                },
            )
            code = json.loads((Path(out) / "gemmini/commands.json").read_text())
            self.assertTrue(
                any(
                    n.kwargs.get("relu")
                    for n in result.program.graph.graph.nodes
                )
            )
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
            self.assertTrue(all(c.tile is not None for c in choices))
            for choice in choices:
                self.assertEqual(len(choice.tile.tiling.loop_blockings[0]), 3)
            result.program.assert_frozen()


class Depthwise(torch.nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x, w, b):
        return torch.ops.gemmini.depthwise_conv2d.default(
            x, w, b, self.stride, 1, 0.125
        )


class GemminiDepthwiseTests(unittest.TestCase):
    def test_native_split_spatial_tiles_preserve_groups_and_halos(self):
        torch.manual_seed(9)
        args = (
            torch.randint(-8, 8, (2, 9, 11, 3), dtype=torch.int8),
            torch.randint(-4, 4, (3, 1, 3, 3), dtype=torch.int8),
            torch.tensor([-11, 0, 9], dtype=torch.int32),
        )
        recipe = next(
            r
            for r in TARGET.implementations
            if r.name == "gemmini.depthwise_conv2d.hardware_loop"
        )
        target = replace(
            TARGET,
            implementations=(
                replace(
                    recipe,
                    candidates=lambda n, c: (
                        {"tile": (2, 3), "mode": "hardware_loop"},
                    ),
                ),
            ),
        )
        for stride in (1, 2):
            with (
                self.subTest(stride=stride),
                tempfile.TemporaryDirectory() as out,
            ):
                expected = torch.nn.functional.conv2d(
                    args[0].permute(0, 3, 1, 2).double(),
                    args[1].double(),
                    args[2].double(),
                    stride,
                    1,
                    groups=3,
                )
                expected = (
                    (expected.float() * 0.125)
                    .round()
                    .clamp(-128, 127)
                    .to(torch.int8)
                    .permute(0, 2, 3, 1)
                )
                result = compile_workload(
                    WorkloadSpec(
                        "dw",
                        lambda _: Depthwise(stride),
                        lambda m, o: (args, {}),
                    ),
                    target,
                    out,
                    dump_tensors=False,
                )
                run(result, reference=expected, rtol=0, atol=0)
                stream = json.loads(
                    (Path(out) / "gemmini/commands.json").read_text()
                )
                launches = [
                    c
                    for c in stream["commands"]
                    if c["type"] == "command"
                    and int(c["instruction"], 16) >> 25 == 15
                ]
                self.assertEqual(
                    len(launches),
                    2
                    * 3
                    * ((expected.shape[1] + 1) // 2)
                    * ((expected.shape[2] + 2) // 3),
                )
                self.assertTrue(all(int(c["rs1"], 16) & 64 for c in launches))


if __name__ == "__main__":
    unittest.main()

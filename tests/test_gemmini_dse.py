"""Family propagation and native service regression, separate from RTL timing."""

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
from test_pipeline import assert_saved_emission

from voyager_compiler.extensions import CandidateRejected, WorkloadSpec
from voyager_compiler.pipeline import compile_workload
from voyager_compiler.runtime import run
from voyager_compiler.targets.gemmini.emission._commands import (
    conv,
    explicit_matmul,
    matmul,
)
from voyager_compiler.targets.gemmini.emission.replay import write_replay
from voyager_compiler.targets.gemmini.hardware import hardware, parameters
from voyager_compiler.targets.gemmini.native_service import NativeService
from voyager_compiler.targets.gemmini.scheduling import (
    NativeMatrixModel,
    cost,
    footprint,
)
from voyager_compiler.targets.gemmini.semantics import tile_storage
from voyager_compiler.targets.gemmini.target import TARGET
from voyager_compiler.targets.gemmini.traffic import matrix_records, row_bytes


class GemminiDSETests(unittest.TestCase):
    def test_hardware_roundtrip_and_invalid_geometry(self):
        config = hardware(dim=32, scratchpad_kib=512, accumulator_kib=128)
        self.assertEqual(config.pe_array_size, (32, 32))
        self.assertEqual(config.memory_instance("scratchpad").word_bytes, 32)
        self.assertEqual(config.memory_instance("accumulator").word_bytes, 128)
        self.assertEqual(
            parameters(hardware(**parameters(config))), parameters(config)
        )
        for kwargs in (
            dict(dim=24),
            dict(dim=0),
            dict(scratchpad_kib=0),
            dict(dma_bytes_per_cycle=float("nan")),
            dict(accumulator_banks=3),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                hardware(**kwargs)

    def test_native_tail_encoding_and_dma_block_stride(self):
        p = dict(m=19, n=21, k=35, tile=(19, 21, 35))
        addr = dict(a=0x80000000, b=0x80001000, bias=0x80002000, y=0x80003000)
        for dim, bounds, pads in (
            (16, (2, 2, 3), (13, 11, 13)),
            (32, (1, 1, 2), (13, 11, 29)),
        ):
            commands = matmul(p, addr, hardware(dim=dim))
            bounds_word = next(
                c for c in commands if int(c["instruction"], 16) >> 25 == 9
            )
            for field, expected in (("rs2", bounds), ("rs1", pads)):
                value = int(bounds_word[field], 16)
                self.assertEqual(
                    tuple((value >> s) & 65535 for s in (0, 16, 32)), expected
                )
            load_configs = [
                int(c["rs1"], 16)
                for c in commands
                if int(c["instruction"], 16) >> 25 == 0
                and int(c["rs1"], 16) & 3 == 1
            ]
            self.assertEqual(
                [(v >> 16) & 65535 for v in load_configs], [dim] * 3
            )

    def test_capacity_matches_command_expansion_and_dense_padding(self):
        p = dict(
            batch=1,
            h=7,
            w=7,
            oh=7,
            ow=7,
            ci=64,
            co=128,
            kernel=3,
            stride=1,
            pad=1,
            tile=(7, 7, 128, 64),
        )
        addr = dict(
            x=0x80000000, weight=0x80100000, bias=0x80200000, y=0x80300000
        )
        for dim in (16, 32):
            config = hardware(dim=dim)
            sp, acc = tile_storage("conv2d", p, p["tile"], "hardware_loop", dim)
            self.assertEqual((sp, acc), (78912, 32768))
            footprint("conv2d", p, p["tile"], "hardware_loop", config)
            self.assertEqual(
                sum(
                    int(c["instruction"], 16) >> 25 == 15
                    for c in conv(p, addr, config=config)
                ),
                1,
            )
            small = hardware(dim=dim, accumulator_kib=32)
            with self.assertRaises(CandidateRejected):
                footprint("conv2d", p, p["tile"], "hardware_loop", small)
            with self.assertRaises(ValueError):
                conv(p, addr, config=small)

    def test_dma_phases_match_each_launch_including_k_tails(self):
        m, n, k, tile, dim = 71, 37, 41, (32, 32, 32), 32
        records = matrix_records(m, n, k, tile, dim)
        actual = sum((a + b) * count for _, _, _, a, b, count in records)
        expected = 0
        for r in range(0, m, tile[0]):
            mr = min(tile[0], m - r)
            for c in range(0, n, tile[1]):
                nr = min(tile[1], n - c)
                expected += ((mr + dim - 1) // dim) * row_bytes(
                    c * 4, 1, 0, nr * 4, chunk=dim * 4
                )
                expected += mr * nr
                for t in range(0, k, tile[2]):
                    kr = min(tile[2], k - t)
                    expected += row_bytes(r * k + t, mr, k, kr)
                    expected += row_bytes(
                        t * n + c, ((kr + dim - 1) // dim) * dim, n, nr
                    )
        self.assertEqual(actual, expected)
        self.assertEqual(row_bytes(0, 3, 32, 1), 48)
        self.assertEqual(row_bytes(15, 1, 0, 2), 32)

    def test_predictor_uses_geometry_bandwidth_and_service_floor(self):
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
        tile = (7, 7, 128, 64)
        service = NativeService()
        estimates = []
        for dim in (16, 32):
            config = hardware(dim=dim)
            predicted = cost("conv2d", p, tile, "hardware_loop", config)
            self.assertGreaterEqual(
                predicted.latency, service.convolution(dim, 3, tile, 32)
            )
            estimates.append(predicted.latency)
            model = NativeMatrixModel("conv2d", p, "hardware_loop", config)
            self.assertEqual(model.layer.nifm, 512)
        self.assertLess(estimates[1], estimates[0])
        slow = cost(
            "conv2d", p, tile, "hardware_loop", hardware(dma_bytes_per_cycle=1)
        )
        self.assertGreater(slow.latency, estimates[0])

    def test_dim32_raw_modes_and_relocated_saved_emission(self):
        class Matrix(torch.nn.Module):
            def forward(self, a, b):
                return torch._int_mm(a, b)

        args = (
            torch.arange(2048, dtype=torch.int8).reshape(32, 64),
            torch.eye(64, dtype=torch.int8),
        )
        spec = WorkloadSpec(
            "dim32-raw", lambda _: Matrix(), lambda m, o: (args, {})
        )
        for recipe in TARGET.implementations[:2]:
            with (
                self.subTest(recipe=recipe.name),
                tempfile.TemporaryDirectory() as directory,
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = compile_workload(
                        spec,
                        replace(TARGET, implementations=(recipe,)),
                        directory,
                        config=hardware(dim=32),
                        dump_tensors=False,
                    )
                run(result, reference=torch._int_mm(*args), rtol=0, atol=0)
                assert_saved_emission(self, directory)
                commands = json.loads(
                    (Path(directory) / "gemmini/commands.json").read_text()
                )
                self.assertEqual(commands["hardware_parameters"]["dim"], 32)
                if recipe.name.endswith("explicit"):
                    computes = [
                        c
                        for c in commands["commands"]
                        if int(c["instruction"], 16) >> 25 == 4
                    ]
                    self.assertEqual(len(computes), 4)
                    self.assertEqual(int(computes[0]["rs1"], 16) >> 48, 32)

    def test_dim32_native_deployment_replay_aperture(self):
        class Matrix(torch.nn.Module):
            def forward(self, a, b, bias):
                return torch.ops.gemmini.matmul.default(a, b, bias, 0.125)

        args = (
            torch.ones((19, 35), dtype=torch.int8),
            torch.ones((35, 21), dtype=torch.int8),
            torch.zeros(21, dtype=torch.int32),
        )
        spec = WorkloadSpec(
            "dim32-deploy", lambda _: Matrix(), lambda m, o: (args, {})
        )
        config = hardware(dim=32, dram_bytes=32 << 20)
        for mode in ("hardware_loop", "explicit"):
            recipe = next(
                r
                for r in TARGET.implementations
                if r.name == f"gemmini.matmul.{mode}"
            )
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory() as temporary,
            ):
                directory = Path(temporary) / "compiler"
                with contextlib.redirect_stdout(io.StringIO()):
                    result = compile_workload(
                        spec,
                        replace(TARGET, implementations=(recipe,)),
                        directory,
                        config=config,
                        dump_tensors=False,
                    )
                run(result, reference=Matrix()(*args), rtol=0, atol=0)
                assert_saved_emission(self, directory)
                write_replay(
                    result,
                    Path(temporary) / "replay",
                    directory / "gemmini/commands.json",
                )
                manifest = json.loads(
                    (Path(temporary) / "replay/memory.json").read_text()
                )
                self.assertEqual(manifest["size"], 32 << 20)
                # An explicit software tile has a whole arena, not a half.
                if mode == "explicit":
                    explicit_matmul(
                        dict(m=19, n=21, k=35, tile=(19, 21, 35)),
                        dict(
                            a=0x80000000,
                            b=0x80001000,
                            bias=0x80002000,
                            y=0x80003000,
                        ),
                        config,
                    )

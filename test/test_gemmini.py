"""Integration contracts for Gemmini's reuse of the shared bufferized compiler."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from google.protobuf import text_format

import voyager_compiler as vc
from voyager_compiler.codegen import voyager_ir_pb2 as ir
from voyager_compiler.gemmini.collateral import convert
from voyager_compiler.gemmini.hardware import lean_config
from voyager_compiler.quantization.fake_quantize import get_quantization_map
from voyager_compiler.targets import get_backend


class GemminiFlowTest(unittest.TestCase):
    def test_cost_model_prices_partial_rows_and_shared_loop_reuse(self):
        from interstellar import Layer, MappingPoint
        from interstellar import loop_enum as le
        from voyager_compiler.codegen.transform.tiling.tiler import (
            RuntimeCalculator,
        )
        from voyager_compiler.gemmini.execution import GemminiCostModel

        shared = RuntimeCalculator.__new__(RuntimeCalculator)
        shared.batch = shared.weight_batch = 1
        shared.dram_bandwidth = 16
        shared.dram_access_latency_cycles = 8
        cost = GemminiCostModel(shared, batch=4)
        layer = Layer(128, 128, 28, 28, 3, 3)

        def mapping(x, y, channel_innermost=False):
            blocks = [[1, 1, 1, 1] for _ in range(le.NUM)]
            parts = [[1, 1, 1, 1] for _ in range(le.NUM)]
            orders = [[6, 6, 6, 6] for _ in range(le.NUM)]
            for d in (le.FX, le.FY):
                blocks[d][1] = 3
            for d, size in ((le.OX, x), (le.OY, y)):
                blocks[d][1], blocks[d][3] = size, 28 // size
            parts[le.IC][0] = parts[le.OC][0] = 16
            blocks[le.IC][1], blocks[le.OC][3] = 8, 8
            sequence = (
                (le.OC, le.OX, le.OY)
                if channel_innermost
                else (le.OX, le.OY, le.OC)
            )
            for rank, d in enumerate(sequence):
                orders[d][3] = rank
            return MappingPoint(orders, blocks, parts)

        narrow = cost.calculate_runtime(None, layer, mapping(4, 14))
        wide = cost.calculate_runtime(None, layer, mapping(28, 2))
        self.assertLess(
            wide, narrow
        )  # equal output area, fewer short instructions
        held_weights = cost.dram_bytes.copy()
        cost.calculate_runtime(None, layer, mapping(28, 2, True))
        self.assertLess(cost.dram_bytes["input"], held_weights["input"])
        self.assertGreater(cost.dram_bytes["weight"], held_weights["weight"])
        self.assertLess(
            cost.calculate_memory_cost(None, layer, mapping(28, 2, True)),
            cost.calculate_memory_cost(None, layer, mapping(28, 2)),
        )

    def test_native_rounding_differs_from_bfloat_lookup(self):
        values = torch.tensor(
            [-128.7, -120.51, -1.5, -0.5, 0.5, 1.5, 126.51, 200.0]
        )
        scale = torch.tensor(1.0)
        native = torch.ops.quantized_ops.quantize(
            values, scale, rounding="nearest_even_int8"
        )
        torch.testing.assert_close(
            native,
            torch.tensor([-128.0, -121.0, -2.0, 0.0, 0.0, 2.0, 127.0, 127.0]),
        )
        lookup = torch.ops.quantized_ops.quantize(
            values, scale, qmap=get_quantization_map("int8")
        )
        self.assertNotEqual(float(native[1]), float(lookup[1]))

    def test_global_pool_rewrite_preserves_mean(self):
        from voyager_compiler.codegen.transform.rewrites import (
            normalize_global_average_pool,
        )
        from voyager_compiler.shape_prop import ShapeProp

        class Pool(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))

        x = torch.arange(2 * 16 * 7 * 7).reshape(2, 16, 7, 7).float()
        graph = vc.export_model(Pool(), (x,))
        ShapeProp(graph).propagate(x)
        expected = graph(x)
        normalize_global_average_pool(graph)
        torch.testing.assert_close(graph(x), expected, rtol=0, atol=0)
        self.assertTrue(
            any(
                n.target == torch.ops.aten.avg_pool2d.default
                for n in graph.graph.nodes
            )
        )

    def test_default_passes_and_standard_collaterals(self):
        class Kernel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.tensor(2.0))
                self.register_buffer("qmap", get_quantization_map("int8"))

            def forward(self, a, b):
                return torch.ops.quantized_ops.quantize(
                    a @ b,
                    self.scale,
                    qmap=self.qmap,
                    rounding="nearest_even_int8",
                )

        torch.manual_seed(4)
        torch.set_num_threads(2)
        a = torch.randint(-4, 5, (512, 64)).float()
        b = torch.randint(-4, 5, (64, 32)).float()
        m = Kernel()
        expected = m(a, b)
        g = vc.export_model(m, (a, b))
        for n in g.graph.nodes:
            if (
                n.op == "placeholder"
                or n.target is torch.ops.quantized_ops.quantize.default
            ):
                n.meta["dtype"] = "int8"
        config = lean_config()
        backend = get_backend("gemmini")
        with (
            tempfile.TemporaryDirectory() as td,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            names = [
                "_transform_voyager",
                "build_interstellar_tiler",
                "bufferize_graph",
                "plan_memory",
                "gen_code_bufferized",
            ]
            with contextlib.ExitStack() as stack:
                spies = {
                    n: stack.enter_context(
                        patch.object(vc, n, wraps=getattr(vc, n))
                    )
                    for n in names
                }
                vc.transform(
                    g,
                    (a, b),
                    config=config,
                    patterns=backend.fusion_patterns(config),
                    layout_policy="systolic",
                )
                vc.compile(g, (a, b), config=config, output_dir=td)
                for name, spy in spies.items():
                    self.assertGreater(spy.call_count, 0, name)
            torch.testing.assert_close(g(a, b), expected, rtol=0, atol=0)
            path = Path(td)
            text = (path / "model.txt").read_text()
            self.assertIn("for_loop", text)
            self.assertIn("voyager::async_copy", text)
            self.assertIn("tiling {", text)
            self.assertNotIn("gemmini.", text)
            proto = text_format.Parse(text, ir.Model())
            banks = []
            for op in proto.ops:
                for output in op.outputs:
                    if output.WhichOneof("result_type") != "tensor_box":
                        continue
                    box = output.tensor_box
                    if (
                        box.memory.level == ir.MEMORY_LEVEL_SCRATCHPAD
                        and box.dtype == "int8"
                        and box.memory.address < 262144
                    ):
                        slots = box.bank_count or 1
                        if slots > 1:
                            self.assertGreaterEqual(
                                box.bank_stride_bytes, 65536
                            )
                        used = {
                            (box.memory.address + i * box.bank_stride_bytes)
                            // 65536
                            for i in range(slots)
                        }
                        self.assertFalse(used.intersection(banks))
                        banks.extend(used)
            self.assertGreaterEqual(len(banks), 3)
            self.assertTrue(all(0 <= b < 4 for b in banks))
            replay = convert(path)
            commands = [
                json.loads(s)
                for s in (replay / "commands.jsonl").read_text().splitlines()
            ]
            functs = {
                int(c["instruction"], 16) >> 25
                for c in commands
                if c["type"] == "command"
            }
            self.assertTrue({2, 3, 4, 6}.issubset(functs))
            self.assertFalse(functs.intersection(range(8, 14)))
            manifest = json.loads((replay / "memory.json").read_text())
            for region in manifest["regions"]:
                if "expected" in region:
                    self.assertNotIn("input", region)
                    self.assertEqual(region["fill"], 165)

    def test_old_whole_tensor_format_is_not_a_conversion_path(self):
        with tempfile.TemporaryDirectory() as td:
            m = ir.Model()
            m.ops.add(name="old").prim.target = "gemmini.matmul"
            (Path(td) / "model.txt").write_text(text_format.MessageToString(m))
            with self.assertRaises(NotImplementedError):
                from voyager_compiler.compilation import CompilerContext

                convert(td, context=CompilerContext.resolve(lean_config()))

    def test_physical_config_and_no_native_model_driver(self):
        config = lean_config()
        backend = get_backend("gemmini")
        self.assertFalse(hasattr(backend, "run_model_case"))
        self.assertEqual(
            config.memory_instance("scratchpad").size.value, 262144
        )
        self.assertEqual(
            config.memory_instance("accumulator").size.value, 65536
        )
        self.assertEqual(
            config.memory_instance("instruction_queue").size.value, 2
        )
        self.assertTrue(backend.uses_bufferized_flow)


if __name__ == "__main__":
    unittest.main()

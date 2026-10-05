"""Integration contracts: shared bufferization, target legality and IR consumption."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import voyager_compiler as vc
from voyager_compiler.codegen import voyager_ir_pb2 as ir
from voyager_compiler.trainium.converter import Converter, convert
from voyager_compiler.trainium.hardware import neuron_core, validate_tile


class Matmul(torch.nn.Module):
    def forward(self, a, b):
        return a @ b


class TrainiumFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(29)
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.inputs = (torch.randn(128, 256), torch.randn(256, 128))
        graph = vc.export_model(Matmul(), cls.inputs)
        cls.config = neuron_core(2)
        cls.calls = {}
        with (
            contextlib.ExitStack() as stack,
            contextlib.redirect_stdout(io.StringIO()),
            torch.no_grad(),
        ):
            for name in (
                "build_interstellar_tiler",
                "bufferize_graph",
                "plan_memory",
                "gen_code_bufferized",
            ):
                cls.calls[name] = stack.enter_context(
                    patch.object(vc, name, wraps=getattr(vc, name))
                )
            vc.transform(graph, cls.inputs, config=cls.config)
            cls.model = vc.compile(
                graph, cls.inputs, config=cls.config, output_dir=cls.root
            )
            torch.testing.assert_close(
                graph(*cls.inputs),
                cls.inputs[0] @ cls.inputs[1],
                rtol=3e-4,
                atol=3e-4,
            )
        cls.plan = json.loads((cls.root / "nki/plan.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_shared_passes_are_executed(self):
        for name, spy in self.calls.items():
            self.assertEqual(spy.call_count, 1, name)
        self.assertTrue((self.root / "layers.txt").is_file())
        self.assertTrue((self.root / "tensor_files").is_dir())

    def test_schedule_controls_dma_slots_and_reduction(self):
        self.assertEqual(self.plan["stats"]["tensor_instructions"], 2)
        self.assertEqual(self.plan["stats"]["dma_copies"], 5)
        self.assertGreater(self.plan["stats"]["waits"], 0)
        self.assertGreater(self.plan["stats"]["loop"], 0)
        self.assertTrue(any(b["slots"] == 2 for b in self.plan["buffers"]))
        self.assertTrue(any(e["kind"] == "async" for e in self.plan["events"]))
        self.assertNotIn(
            "kernels.gemm", (self.root / "nki/program.py").read_text()
        )

    def test_converter_rejects_semantic_export(self):
        with self.assertRaisesRegex(ValueError, "bufferized"):
            Converter(ir.Model(), "trainium-v2").source()

    def test_converter_rejects_unbalanced_wait(self):
        c = Converter(self.model, "trainium-v2")
        registers = [
            b
            for b in c.boxes.values()
            if b.memory.level == ir.MEMORY_LEVEL_REGISTER
        ]
        r = c.ref(ir.TensorBoxRef(box=registers[0]))
        with self.assertRaisesRegex(ValueError, "Unbalanced"):
            c.wait(r)

    def test_converter_reads_serialized_program(self):
        destination = self.root / "converted-again"
        plan = convert(self.root, destination, "trainium-v3")
        self.assertEqual(plan["source_sha256"], self.plan["source_sha256"])
        self.assertEqual(plan["events"], self.plan["events"])
        self.assertEqual(
            (destination / "program.py").read_bytes(),
            (self.root / "nki/program.py").read_bytes(),
        )

    def test_physical_hardware_and_instruction_limits(self):
        for version, size in ((2, 24), (3, 28)):
            c = neuron_core(version)
            self.assertEqual(c.memory_instance("SBUF").size.value, size << 20)
            self.assertEqual(c.memory_instance("SBUF").partitions, 128)
            self.assertEqual(c.memory_instance("PSUM").banks, 8)
            self.assertEqual(c.memory_instance("PSUM").size.value, 2 << 20)
            self.assertFalse(
                any(m.name == "scratchpad" for m in c.memory.instances)
            )
        for shape in ((129, 128, 128), (128, 513, 128), (128, 128, 129)):
            with self.assertRaises(ValueError):
                validate_tile(*shape)
        for b in self.plan["buffers"]:
            if b["level"] == ir.MEMORY_LEVEL_SCRATCHPAD:
                self.assertEqual(b["address"] % (16 * 128), 0)
                self.assertEqual(b["stride"] % (16 * 128), 0)


if __name__ == "__main__":
    unittest.main()

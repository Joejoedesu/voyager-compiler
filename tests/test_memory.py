"""Allocation lifetimes, physical bank constraints and shared residency."""

import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch.fx import Graph, GraphModule

from voyager_compiler.compiler.allocate import (
    MemoryCapacityError,
    _Buf,
    plan_memory,
)
from voyager_compiler.compiler.banked_allocation import place_bank_entries
from voyager_compiler.compiler.bufferize import annotate_tensor_spaces
from voyager_compiler.export_utils import export_model
from voyager_compiler.hardware import AcceleratorConfig
from voyager_compiler.ir.plan import BankLayout, BufferComponent
from voyager_compiler.pipeline import compile_bufferized
from voyager_compiler.shape_prop import ShapeProp, fake_like
from voyager_compiler.targets.magnet_alpha.hardware import (
    MagnetAlphaHardwareConfig,
)
from voyager_compiler.targets.magnet_alpha.target import TARGET


class AsyncAllocationTests(unittest.TestCase):
    def graph(self, drain=True):
        graph, ops = Graph(), torch.ops.voyager
        source = graph.placeholder("source")

        def alloc(name, space):
            return graph.create_node(
                "call_function",
                ops.alloc.default,
                ([128], torch.float32, space),
                {},
                name,
            )

        def copy(source, dest):
            sem = graph.call_function(ops.zeros.default, ([], torch.int64))
            graph.call_function(
                ops.async_copy.default, (source, dest, [0], [128], sem)
            )
            return sem

        first = alloc("first", 2)
        semaphore = copy(source, first)
        graph.call_function(ops.async_wait.default, (semaphore,))
        output = alloc("output_buffer", 3)
        pending = copy(first, output)
        second = alloc("second", 2)
        semaphore = copy(source, second)
        graph.call_function(ops.async_wait.default, (semaphore,))
        if drain:
            graph.call_function(ops.async_wait.default, (pending,))
        graph.output((output, second))
        module = GraphModule(torch.nn.Module(), graph)
        ShapeProp(module).propagate(fake_like(torch.ones(128)))
        annotate_tensor_spaces(module)
        return module, first, second

    def test_pending_copy_prevents_address_reuse(self):
        module, first, second = self.graph()
        plan_memory(
            module, AcceleratorConfig(scratchpad_size=4096, num_banks=None)
        )
        self.assertNotEqual(
            first.meta["scratchpad"].start, second.meta["scratchpad"].start
        )

    def test_missing_completion_is_rejected(self):
        module, _, _ = self.graph(False)
        with self.assertRaisesRegex(ValueError, "no completion consumer"):
            plan_memory(
                module, AcceleratorConfig(scratchpad_size=4096, num_banks=None)
            )


class BankAllocationTests(unittest.TestCase):
    def test_cluster_modes_cannot_overlap_physical_cells(self):
        graph = Graph()
        a, b, c = [graph.placeholder(name) for name in ("a", "b", "c")]
        a.meta["bank_layout"] = BankLayout(1, (0,))
        b.meta["bank_layout"] = BankLayout(2, (0,))
        c.meta["bank_layout"] = BankLayout(1, (0,))
        memory = SimpleNamespace(
            name="test", banks=4, word_bytes=16, size=SimpleNamespace(value=256)
        )
        placed, peak = place_bank_entries(
            {
                a: _Buf(48, 0, 4, [a]),
                b: _Buf(64, 1, 4, [b]),
                c: _Buf(48, 5, 8, [c]),
            },
            memory,
        )

        def cells(ranges):
            return {
                (bank, entry)
                for bank, start, stop in ranges
                for entry in range(start, stop)
            }

        self.assertFalse(cells(placed[a][2]) & cells(placed[b][2]))
        self.assertEqual(placed[a], placed[c])
        self.assertEqual(peak, 112)

    def test_bank_conflicts_are_stronger_than_byte_overlap(self):
        graph = Graph()
        a, b = graph.placeholder("a"), graph.placeholder("b")
        a.meta["bank_layout"] = BankLayout(1)
        b.meta["bank_layout"] = BankLayout(1, avoid_buffers=("a",))
        memory = SimpleNamespace(
            name="test", banks=1, word_bytes=16, size=SimpleNamespace(value=256)
        )
        with self.assertRaises(MemoryCapacityError):
            place_bank_entries(
                {a: _Buf(16, 0, 3, [a]), b: _Buf(16, 1, 2, [b])}, memory
            )

    def test_packed_components_reserve_backing_once(self):
        graph = Graph()
        payload = graph.create_node(
            "call_function",
            torch.ops.voyager.alloc.default,
            ([128], torch.int8, 2),
            {},
            "payload",
        )
        scale = graph.create_node(
            "call_function",
            torch.ops.voyager.alloc.default,
            ([8], torch.uint8, 2),
            {},
            "scale",
        )
        graph.output((payload, scale))
        module = GraphModule(torch.nn.Module(), graph)
        ShapeProp(module).propagate()
        annotate_tensor_spaces(module)
        for n in (payload, scale):
            n.meta.update(
                memory_instance="global_activation",
                bank_layout=BankLayout(1, (0, 1)),
            )
        scale.meta["physical_component"] = BufferComponent(
            "/payload", "test.packed", "scale"
        )
        allocation = plan_memory(module, MagnetAlphaHardwareConfig())
        self.assertEqual(
            payload.meta["scratchpad"].start, scale.meta["scratchpad"].start
        )
        self.assertEqual(dict(allocation.arenas)["global_activation"], 128)
        scale.meta["physical_component"] = BufferComponent(
            "/missing", "test.packed", "scale"
        )
        with self.assertRaisesRegex(ValueError, "concrete SRAM backing"):
            plan_memory(module, MagnetAlphaHardwareConfig())


class ResidencyTests(unittest.TestCase):
    def test_shared_whole_kernels_replay_after_residency(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                a = torch.relu(x)
                return (a + 2) * a

        args = (torch.randn(8, 32),)
        graph = export_model(Model(), args)
        reference = graph(*args)
        with tempfile.TemporaryDirectory() as output:
            result = compile_bufferized(
                graph,
                args,
                config=MagnetAlphaHardwareConfig(),
                target=replace(TARGET, encode=None, emit=None),
                output_dir=output,
                dump_tensors=False,
                reference_outputs=reference,
            )
        self.assertGreater(result.program.validation["resident_merges"], 0)
        self.assertEqual(
            result.program.validation["numerical_replay"], "passed"
        )
        self.assertTrue(result.program.plan.allocation.arenas)
        from voyager_compiler.compiler.allocate import _buffer_identity

        identities = _buffer_identity(result.program.graph)

        def cells(node):
            node = identities.get(node, node)
            return {
                (bank, entry)
                for bank, start, stop in node.meta.get("bank_ranges", ())
                for entry in range(start, stop)
            }

        checked = 0
        for node in result.program.graph.graph.nodes:
            outputs = [
                u.args[1]
                for u in node.users
                if u.target == torch.ops.voyager.insert.default
            ]
            for destination in outputs:
                for source in node.all_input_nodes:
                    if cells(source) and cells(destination):
                        self.assertFalse(cells(source) & cells(destination))
                        checked += 1
        self.assertGreater(checked, 0)
        result.program.assert_frozen()

    def test_residency_retains_writes_into_concat_views(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.cat((torch.relu(x), torch.relu(x + 3)), dim=1) * 2

        args = (torch.arange(-16, 16, dtype=torch.float32).reshape(1, 8, 2, 2),)
        graph = export_model(Model(), args)
        reference = graph(*args)
        with tempfile.TemporaryDirectory() as output:
            compile_bufferized(
                graph,
                args,
                config=MagnetAlphaHardwareConfig(),
                target=replace(TARGET, encode=None, emit=None),
                output_dir=output,
                dump_tensors=False,
                reference_outputs=reference,
            )


if __name__ == "__main__":
    unittest.main()

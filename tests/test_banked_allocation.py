"""Cluster modes share physical cells; packed components share lifetimes."""

import unittest
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
from voyager_compiler.ir.plan import BankLayout, BufferComponent
from voyager_compiler.shape_prop import ShapeProp
from voyager_compiler.targets.magnet_alpha.hardware import (
    MagnetAlphaHardwareConfig,
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

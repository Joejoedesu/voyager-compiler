"""Pending DMA reads keep their source alive until the completion wait."""

import unittest

import torch
from torch.fx import Graph, GraphModule

from voyager_compiler.compiler.allocate import plan_memory
from voyager_compiler.compiler.bufferize import annotate_tensor_spaces
from voyager_compiler.hardware import AcceleratorConfig
from voyager_compiler.shape_prop import ShapeProp, fake_like


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

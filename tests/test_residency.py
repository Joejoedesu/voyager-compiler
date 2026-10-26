"""Residency must preserve values and independent physical memory arenas."""

import tempfile
import unittest
from dataclasses import replace

import torch

from voyager_compiler.export_utils import export_model
from voyager_compiler.pipeline import compile_bufferized
from voyager_compiler.targets.magnet_alpha.hardware import (
    MagnetAlphaHardwareConfig,
)
from voyager_compiler.targets.magnet_alpha.target import TARGET


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
                target=replace(TARGET, emit=None),
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
                target=replace(TARGET, emit=None),
                output_dir=output,
                dump_tensors=False,
                reference_outputs=reference,
            )


if __name__ == "__main__":
    unittest.main()

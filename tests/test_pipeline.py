"""The real shared pipeline must freeze a runnable program and its placements."""

import tempfile
import unittest

import torch

from voyager_compiler.extensions import WorkloadSpec, registry
from voyager_compiler.hardware import AcceleratorConfig
from voyager_compiler.pipeline import compile_workload
from voyager_compiler.runtime import run


class PipelineTests(unittest.TestCase):
    def test_buffer_program_replay_and_frozen_plan(self):
        class Tiny(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x + 1)

        x = torch.arange(-12, 12, dtype=torch.float32).reshape(4, 6)
        workload = WorkloadSpec(
            "tiny", lambda _: Tiny(), lambda m, o: ((x,), {})
        )
        config = AcceleratorConfig(
            pe_array_size=(4, 4),
            scratchpad_size=4096,
            num_banks=4,
            bank_width=4,
            vector_unit_width=4,
        )
        with tempfile.TemporaryDirectory(prefix="voyager-pipeline-") as output:
            result = compile_workload(
                workload,
                registry.target("voyager"),
                output,
                config=config,
                dump_tensors=False,
            )
            _, evidence = run(result, (x,), reference=result.reference)
            self.assertEqual(evidence["kind"], "host_replay")
            self.assertTrue(result.program.bindings)
            self.assertTrue(result.program.plan.allocation.buffers)
            with self.assertRaisesRegex(RuntimeError, "frozen"):
                result.program.plan.target = "changed"
            # A post-freeze graph edit must also be rejected before execution.
            node = next(
                n
                for n in result.program.graph.graph.nodes
                if n.op == "call_function"
            )
            node.name += "_changed"
            with self.assertRaisesRegex(ValueError, "unchanged frozen"):
                run(result, (x,))

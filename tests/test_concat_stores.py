"""An equal-value concat must still emit writes to both distinct windows."""

import tempfile
import unittest

import torch

from voyager_compiler.extensions import WorkloadSpec, registry
from voyager_compiler.hardware import AcceleratorConfig
from voyager_compiler.pipeline import compile_workload
from voyager_compiler.runtime import run


class ConcatStoreTests(unittest.TestCase):
    def test_repeated_source_writes_both_halves(self):
        class Repeat(torch.nn.Module):
            def forward(self, x):
                return torch.cat((x, x), dim=-1)

        x = torch.arange(64, dtype=torch.float32).reshape(1, 1, 64)
        workload = WorkloadSpec(
            "repeat", lambda _: Repeat(), lambda m, o: ((x,), {})
        )
        with tempfile.TemporaryDirectory() as output:
            result = compile_workload(
                workload,
                registry.target("voyager"),
                output,
                config=AcceleratorConfig(),
                dump_tensors=False,
            )
            run(result, reference=torch.cat((x, x), dim=-1), rtol=0, atol=0)
            copies = [
                op
                for op in result.model.ops
                if op.HasField("prim") and op.prim.target == "aten::clone"
            ]
            self.assertEqual(len(copies), 2)
            destinations = [op.outputs[0].destination for op in copies]
            self.assertEqual(destinations[0].box.node, destinations[1].box.node)
            self.assertEqual(
                {d.offsets[-1].int_value for d in destinations}, {0, 64}
            )
            self.assertTrue(
                all(tuple(d.sizes) == (1, 1, 64) for d in destinations)
            )

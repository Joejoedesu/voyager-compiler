"""Shared pipeline freezing, complete stores and replay-image boundaries."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import torch

from voyager_compiler.extensions import WorkloadSpec, registry
from voyager_compiler.hardware import AcceleratorConfig
from voyager_compiler.pipeline import compile_workload
from voyager_compiler.replay import ReplayImages
from voyager_compiler.runtime import run


def assert_saved_emission(test, directory):
    """A new process sees only relocated, manifest-listed inputs, never live FX."""
    source = Path(directory)
    manifest = json.loads((source / "bundle.json").read_text())
    expected = json.loads((source / "emission.json").read_text())
    with tempfile.TemporaryDirectory() as temporary:
        bundle, output = Path(temporary) / "bundle", Path(temporary) / "emitted"
        for name in ("bundle.json", *manifest["files"]):
            destination = bundle / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, destination)
        env = dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from voyager_compiler.compiler.emit import emit_bundle; import sys; emit_bundle(*sys.argv[1:])",
                str(bundle),
                str(output),
            ],
            env=env,
            cwd=temporary,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        actual = json.loads((output / "emission.json").read_text())
        test.assertEqual(actual, expected)
        for name in actual["files"]:
            test.assertEqual(
                (output / name).read_bytes(), (source / name).read_bytes()
            )


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
        emitter = Mock()
        target = replace(registry.target("voyager"), emit=emitter)
        with tempfile.TemporaryDirectory(prefix="voyager-pipeline-") as output:
            result = compile_workload(
                workload,
                target,
                output,
                config=config,
                dump_tensors=False,
                emit_target=False,
            )
            from voyager_compiler.artifacts import SavedBundle
            from voyager_compiler.compiler.emit import emit_bundle

            emitter.assert_not_called()
            self.assertFalse((Path(output) / "emission.json").exists())
            emit_bundle(output, output)  # Default Voyager consumes IR directly.
            assert_saved_emission(self, output)
            saved = SavedBundle.load(output)
            with self.assertRaisesRegex(ValueError, "not a bundle member"):
                saved.path("../outside")
            with self.assertRaisesRegex(ValueError, "does not match"):
                emit_bundle(
                    output, output, target=replace(target, name="wrong")
                )
            original = (Path(output) / "target_ir.json").read_text()
            (Path(output) / "target_ir.json").write_text(original + " ")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                emit_bundle(output, output, target=target)
            emitter.assert_not_called()
            (Path(output) / "target_ir.json").write_text(original)
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


class ReplayTests(unittest.TestCase):
    def test_input_output_separation_and_complete_bounds(self):
        images = ReplayImages.create(16)
        images.store(0, 4, b"abcd", is_input=True)
        images.store(4, 8, b"wxyz")
        self.assertEqual(images.initial[:4], b"abcd")
        self.assertEqual(images.expected[:8], b"abcdwxyz")
        self.assertEqual(images.initial[4:], bytes([0xA5]) * 12)
        for start, end, payload in [
            (-1, 4, b"x"),
            (14, 18, b"x"),
            (4, 5, b"xx"),
        ]:
            with (
                self.subTest(start=start, end=end),
                self.assertRaises(ValueError),
            ):
                images.store(start, end, payload)
        self.assertEqual(len(images.initial), 16)
        self.assertEqual(len(images.expected), 16)


if __name__ == "__main__":
    unittest.main()

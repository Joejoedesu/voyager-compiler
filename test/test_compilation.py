"""Target selection, recipe precedence, and stateful verification contracts."""

import contextlib
import io
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_ci
import torch
from compilation.context import parse_args
from compilation.pipeline import PreparedModel, compile_prepared
from test_codegen import build_parser

from voyager_compiler import compile, transform
from voyager_compiler.hardware_config import VOYAGER
from voyager_compiler.quantization.recipes import (
    FamilyPolicy,
    Recipe,
    register_family,
)
from voyager_compiler.targets import (
    Target,
    register_backend,
    register_target,
)


class CompilationTest(unittest.TestCase):
    def args(self, *extra, model="resnet18", output="/tmp/unused-voyager-test"):
        return parse_args(
            build_parser(), [model, "--model_output_dir", output, *extra]
        )

    def test_recipe_cli_overrides(self):
        args = self.args(
            "--quantization_recipe", "MXNF4", "--pe_array_size", "64,64"
        )
        self.assertTrue(args.bf16)
        args = self.args(
            "--quantization_recipe",
            "MXNF4",
            "--activation",
            "int8",
            "--no-bf16",
            "--no-conv2d_im2col",
            "--no-quantize_fc",
            "--pe_array_size",
            "64,64",
        )
        self.assertEqual(args.bank_width, 64)
        self.assertFalse(args.bf16)
        self.assertFalse(args.conv2d_im2col)
        self.assertFalse(args.quantize_fc)

    def test_parser_reuse_does_not_leak_recipe_defaults(self):
        parser = build_parser()
        parse_args(
            parser,
            [
                "resnet18",
                "--model_output_dir",
                "/tmp/unused",
                "--quantization_recipe",
                "INT8",
            ],
        )
        plain = parse_args(
            parser, ["resnet18", "--model_output_dir", "/tmp/unused"]
        )
        self.assertIsNone(plain.activation)
        self.assertEqual(plain.calibration_steps, 0)

    def test_new_family_and_backend_do_not_require_voyager_lowering(self):
        class Backend:
            def validate(self, config):
                self.seen = config

            def transform(self, model, args, kwargs, **options):
                return ("transformed", options["config"])

            def compile(self, model, args, kwargs, **options):
                return ("compiled", options["config"])

        from voyager_compiler import targets
        from voyager_compiler.quantization import recipes

        backend = Backend()
        with (
            patch.dict(targets._TARGETS),
            patch.dict(targets._BACKENDS),
            patch.dict(recipes._FAMILIES),
        ):
            register_backend("unit-backend", backend)
            register_family(
                "unit-family",
                FamilyPolicy(
                    {
                        "base": Recipe(
                            {"activation": "int8"},
                            models={"bert": {"bias": "int24"}},
                        )
                    },
                    {"per-op": {}},
                    lambda *a, **kw: None,
                ),
            )
            register_target(
                Target(
                    "unit-target",
                    "unit-family",
                    "unit-backend",
                    lambda args: replace(
                        VOYAGER, name="unit-target", backend="unit-backend"
                    ),
                )
            )
            args = self.args(
                "--target_hardware",
                "unit-target",
                "--quantization_recipe",
                "base",
                "--qconfig",
                "per-op",
                model="bert",
            )
            self.assertEqual(args.bias, "int24")
            self.assertIs(backend.seen, args.compilation_context.hardware)
            self.assertEqual(
                transform(None, (), config=backend.seen),
                ("transformed", backend.seen),
            )
            self.assertEqual(
                compile(None, (), config=backend.seen),
                ("compiled", backend.seen),
            )
            register_target(
                Target(
                    "unit-variant",
                    "unit-family",
                    "unit-backend",
                    lambda args: backend.seen,
                    recipes={"base": Recipe({"activation": "int4"})},
                )
            )
            self.assertEqual(
                self.args(
                    "--target_hardware",
                    "unit-variant",
                    "--quantization_recipe",
                    "base",
                ).activation,
                "int4",
            )

    def test_unknown_target_or_recipe_fails_before_model_loading(self):
        for flags in (
            ("--target_hardware", "gemmini"),
            ("--quantization_recipe", "missing"),
            ("--qconfig", "missing"),
        ):
            with (
                self.subTest(flags=flags),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                self.args(*flags)

    def test_ci_parallel_processes_keep_artifacts_and_report_order(self):
        # Each process waits for the other to start, so serial execution fails.
        worker = """import sys, time
from pathlib import Path
root, dest, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
(root / ('ready.' + name)).touch()
deadline = time.monotonic() + 5
while len(list(root.glob('ready.*'))) != 2:
    if time.monotonic() > deadline:
        raise RuntimeError('cases did not run concurrently')
    time.sleep(0.01)
(dest / 'model.txt').write_text(name)
print('Results match')
"""
        commands = [
            run_ci.Command(name, "test") for name in ("first", "second")
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before, after = root / "before", root / "after"
            for command in commands:
                path = before / run_ci._label(command) / "model.txt"
                path.parent.mkdir(parents=True)
                path.write_text(command.model)

            def build(command, run_dir, threads):
                label = run_ci._label(command)
                dest = run_dir / label
                return (
                    label,
                    dest,
                    [
                        sys.executable,
                        "-c",
                        worker,
                        str(root),
                        str(dest),
                        command.model,
                    ],
                )

            with (
                patch("run_ci._build", side_effect=build),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                results = run_ci._run_cases(commands, after, before, jobs=2)
            self.assertEqual(
                [r[0] for r in results], [run_ci._label(c) for c in commands]
            )
            self.assertEqual([r[2] for r in results], ["MATCH", "MATCH"])

    def test_verification_restores_state_and_uses_inputs_before_emission(self):
        class Stateful(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("cache", torch.tensor(2.0))
                self.graph = SimpleNamespace(print_tabular=lambda: None)

            def forward(self, x):
                cache = (
                    self.lowered_cache
                    if hasattr(self, "lowered_cache")
                    else self.cache
                )
                cache.add_(1)
                return x + cache

        graph = Stateful()
        inputs = (torch.tensor(4.0),)
        reference = graph(*inputs)

        def restore(gm):
            gm.cache.fill_(2)

        def emit(gm, inputs, kwargs, **options):
            gm.register_buffer("lowered_cache", gm.cache.clone())
            del gm.cache
            options["before_emit"](gm)
            # Tensor dumping may execute a graph and mutate both state and input.
            gm(*inputs)
            inputs[0].fill_(100)

        def capture(gm):
            saved = gm.lowered_cache.clone()
            return lambda graph: graph.lowered_cache.copy_(saved)

        prepared = PreparedModel(
            graph,
            inputs,
            reference,
            restore_state=restore,
            capture_state=capture,
        )
        with (
            patch("compilation.pipeline.transform"),
            patch("compilation.pipeline.compile", side_effect=emit),
        ):
            _, before, after = compile_prepared(
                prepared, self.args("--debug"), []
            )
        torch.testing.assert_close(before, after)
        self.assertEqual(graph.lowered_cache.item(), 2)


if __name__ == "__main__":
    unittest.main()

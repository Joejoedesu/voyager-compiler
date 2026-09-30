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


class QuantizationRuleTest(unittest.TestCase):
    class Concat(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(12, 2)
            self.other = torch.nn.Linear(4, 2)

        def forward(self, a, b, c):
            joined = torch.cat((a, b), -1).view(1, 8)
            return self.fc(torch.cat((joined, c), -1)), self.other(a)

    def prepare(self, rules=True, conflict=False):
        from voyager_compiler import get_default_quantizer, prepare_pt2e
        from voyager_compiler.quantization.rules import CONCAT_INT8

        args = tuple(torch.ones(1, 4) * v for v in (1, 10, 100))
        quantizer = get_default_quantizer(
            input_activation="int8,qs=per_tensor_symmetric,ahl=1",
            weight="int8,qs=per_tensor_symmetric",
            bias="int32",
        )
        if conflict:
            from voyager_compiler import QuantizationConfig, QuantizationSpec

            quantizer.set_module_name(
                "other",
                QuantizationConfig(
                    QuantizationSpec.from_str("int4,qs=per_tensor_symmetric"),
                    None,
                    None,
                    None,
                ),
            )
        if rules:
            quantizer.set_quantization_rules((CONCAT_INT8,))
        return prepare_pt2e(self.Concat().eval(), quantizer, args), args

    def test_concat_observers_share_before_folding_and_keep_fanout_separate(
        self,
    ):
        from voyager_compiler import convert_pt2e
        from voyager_compiler.quantization.fake_quantize import (
            SharedAmaxObsFakeQuantize,
        )

        graph, args = self.prepare()
        group = [
            m
            for m in graph.modules()
            if isinstance(m, SharedAmaxObsFakeQuantize)
        ]
        self.assertEqual(len(group), 1)
        self.assertEqual(len(graph.meta["quantization_rule_groups"]), 1)
        calls = [
            n
            for n in graph.graph.nodes
            if n.op == "call_module"
            and graph.get_submodule(n.target) is group[0]
        ]
        self.assertGreaterEqual(len(calls), 5)
        graph(*args)
        self.assertAlmostEqual(group[0].scale.item(), 100 / 127, places=6)
        # A short history and later small batches cannot erase a branch's range.
        graph(*(torch.ones_like(a) for a in args))
        self.assertAlmostEqual(group[0].scale.item(), 100 / 127, places=6)
        a = next(
            n
            for n in graph.graph.nodes
            if n.op == "placeholder" and n.target == "a"
        )
        observers = [
            graph.get_submodule(u.target)
            for u in a.users
            if u.op == "call_module"
        ]
        self.assertEqual(len({id(m) for m in observers}), 2)
        # No transform/folding pass has run: conversion alone preserves common scales.
        convert_pt2e(graph)
        graph(*args)
        scales = [
            getattr(graph, n.args[1].target)
            for n in graph.graph.nodes
            if n.target is torch.ops.quantized_ops.quantize.default
        ]
        self.assertGreaterEqual(
            sum(torch.allclose(s, torch.tensor([100 / 127])) for s in scales), 3
        )

    def test_rule_selection_and_unrelated_precision(self):
        from voyager_compiler.quantization.fake_quantize import (
            SharedAmaxObsFakeQuantize,
        )

        plain, _ = self.prepare(rules=False)
        self.assertFalse(
            any(
                isinstance(m, SharedAmaxObsFakeQuantize)
                for m in plain.modules()
            )
        )
        graph, args = self.prepare(conflict=True)
        graph(*args)  # Other fan-out consumer may independently use INT4.
        self.assertEqual(len(graph.meta["quantization_rule_groups"]), 1)

    def test_branch_order_and_conflicting_concat_specs(self):
        from voyager_compiler import (
            QuantizationConfig,
            QuantizationSpec,
            get_default_quantizer,
            prepare_pt2e,
        )
        from voyager_compiler.quantization.fake_quantize import (
            SharedAmaxObsFakeQuantize,
        )
        from voyager_compiler.quantization.rules import CONCAT_INT8

        graph, args = self.prepare()
        observer = next(
            m
            for m in graph.modules()
            if isinstance(m, SharedAmaxObsFakeQuantize)
        )
        graph(*reversed(args))
        self.assertAlmostEqual(observer.scale.item(), 100 / 127, places=6)

        class Conflict(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a, self.b = torch.nn.Linear(8, 2), torch.nn.Linear(8, 2)

            def forward(self, x, y):
                z = torch.cat((x, y), -1)
                return self.a(z), self.b(z)

        q = get_default_quantizer(
            input_activation="int8,qs=per_tensor_symmetric", bias="int32"
        )
        q.set_module_name(
            "b",
            QuantizationConfig(
                QuantizationSpec.from_str("int4,qs=per_tensor_symmetric"),
                None,
                None,
                None,
            ),
        ).set_quantization_rules((CONCAT_INT8,))
        with self.assertRaisesRegex(
            ValueError, "concat_int8_shared_scale: incompatible"
        ):
            prepare_pt2e(Conflict(), q, args[:2])


class ResidencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        cls.grad_enabled = torch.is_grad_enabled()
        torch.set_num_threads(2)
        torch.set_grad_enabled(False)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)
        torch.set_grad_enabled(cls.grad_enabled)

    class Chain(torch.nn.Module):
        def forward(self, x):
            return (torch.relu(x) * 2) + 1

    class Parameters(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("w1", torch.randn(4, 16))
            self.register_buffer("w2", torch.randn(4, 16))

        def forward(self, x):
            a = torch.relu(x)
            return a * self.w1 * self.w2 + a

    def config(self, **kwargs):
        from voyager_compiler.hardware_config import voyager_config

        options = dict(
            pe_array_size=(16, 16),
            scratchpad_size=65536,
            num_banks=16,
            bank_width=16,
        )
        options.update(kwargs)
        return voyager_config(**options)

    def graph(self, model=None):
        from voyager_compiler.export_utils import export_model
        from voyager_compiler.shape_prop import ShapeProp

        x = torch.randn(4, 16)
        model = model or self.Chain()
        graph = export_model(model, (x,))
        ShapeProp(graph).propagate(x)
        return graph, x, model(x)

    def lower(self, graph, config, flow="resident", strategy="on_demand"):
        from voyager_compiler import lower_to_buffers
        from voyager_compiler.codegen.transform.bufferize import (
            BufferizationOptions,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            lower_to_buffers(
                graph,
                config,
                options=BufferizationOptions(
                    flow=flow, parameter_loading=strategy
                ),
            )

    def test_resident_chain_has_only_boundary_dma_and_svg_cluster(self):
        from voyager_compiler.codegen.reporting import estimate_schedule
        from voyager_compiler.codegen.transform.bufferize import (
            BufferizationOptions,
        )

        graph, x, expected = self.graph()
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            compile(
                graph,
                (x,),
                config=self.config(),
                output_dir=tmp,
                dump_tensors=False,
                bufferization_options=BufferizationOptions(flow="resident"),
            )
            svg = (Path(tmp) / "compute_graph.svg").read_text()
            self.assertIn("cluster_sram_0", svg)
            self.assertIn("SRAM resident", svg)
            self.assertTrue((Path(tmp) / "sram_regions.json").exists())
            self.assertTrue((Path(tmp) / "model.txt").exists())
        torch.testing.assert_close(graph(x), expected)
        copies = [
            n
            for n in graph.graph.nodes
            if n.target is torch.ops.voyager.async_copy.default
        ]
        self.assertEqual(len(copies), 2)
        self.assertEqual(len(graph.meta["sram_regions"]), 1)
        result = estimate_schedule(graph, self.config())
        self.assertEqual(result.dram_read_bytes, x.numel() * x.element_size())
        self.assertEqual(result.dram_write_bytes, x.numel() * x.element_size())
        # The compute destination cannot alias an input still read by that ISA.
        for n in graph.graph.nodes:
            if n.target is torch.ops.voyager.insert.default:
                compute, dest = n.args
                out = dest.meta["scratchpad"]
                for source in compute.all_input_nodes:
                    segment = source.meta.get("scratchpad")
                    if segment:
                        self.assertTrue(
                            segment.end <= out.start or out.end <= segment.start
                        )

    def test_parameter_strategies_preserve_branches_and_load_each_weight_once(
        self,
    ):
        for strategy in ("preload", "on_demand"):
            with self.subTest(strategy=strategy):
                graph, x, expected = self.graph(self.Parameters())
                self.lower(graph, self.config(), strategy=strategy)
                torch.testing.assert_close(graph(x), expected)
                self.assertEqual(len(graph.meta["sram_regions"]), 1)
                nodes = list(graph.graph.nodes)
                loads = [
                    n
                    for n in nodes
                    if n.target is torch.ops.voyager.async_copy.default
                    and n.args[0].op == "get_attr"
                ]
                self.assertEqual(len(loads), 2)
                first_compute = next(
                    n for n in nodes if n.target is torch.ops.aten.relu.default
                )
                if strategy == "preload":
                    self.assertTrue(
                        all(
                            nodes.index(n) < nodes.index(first_compute)
                            for n in loads
                        )
                    )
                else:
                    self.assertTrue(
                        all(
                            nodes.index(n) > nodes.index(first_compute)
                            for n in loads
                        )
                    )

    def test_capacity_and_bank_failures_are_distinguished(self):
        from voyager_compiler.codegen.transform.bufferize.residency import (
            _build_region,
            _fit,
        )

        graph, _, _ = self.graph(self.Parameters())
        nodes = [n for n in graph.graph.nodes if n.op == "call_function"]
        region, _, _ = _build_region(graph, nodes, "probe", "preload")
        # Use the public factory so all compatibility views stay consistent.
        from voyager_compiler.hardware_config import voyager_config

        capacity = voyager_config(
            pe_array_size=(16, 16), scratchpad_size=128, num_banks=16
        )
        banks = voyager_config(
            pe_array_size=(16, 16), scratchpad_size=65536, num_banks=2
        )
        self.assertEqual(_fit(region, capacity)[1], "SRAM capacity")
        self.assertEqual(_fit(region, banks)[1], "bank allocation")

    def test_scalar_inputs_and_results_keep_original_scalar_handling(self):
        from voyager_compiler.export_utils import export_model
        from voyager_compiler.shape_prop import ShapeProp

        class Scalar(torch.nn.Module):
            def forward(self, x, scalar):
                viewed = scalar.unsqueeze(0)
                return torch.relu(x) * viewed, viewed

        x, scalar = torch.randn(4, 16), torch.tensor(2.0)
        model = Scalar()
        graph = export_model(model, (x, scalar))
        ShapeProp(graph).propagate(x, scalar)
        self.lower(graph, self.config())
        torch.testing.assert_close(graph(x, scalar), model(x, scalar))
        self.assertTrue(
            any(
                "scalar result" in r["reason"]
                for r in graph.meta["sram_fallbacks"]
            )
        )
        for node in graph.graph.nodes:
            if node.target is torch.ops.voyager.async_copy.default:
                self.assertGreater(node.args[0].value.numel(), 1)
                self.assertGreater(node.args[1].value.numel(), 1)

    def test_multiple_outputs_emit_one_contiguous_region_layer(self):
        from voyager_compiler.codegen.transform.bufferize import (
            BufferizationOptions,
        )

        class Branch(torch.nn.Module):
            def forward(self, x):
                return x.cos() * 2, x.sin() * 3

        graph, x, expected = self.graph(Branch())
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            compile(
                graph,
                (x,),
                config=self.config(),
                output_dir=tmp,
                dump_tensors=False,
                bufferization_options=BufferizationOptions(flow="resident"),
            )
            self.assertTrue((Path(tmp) / "layers.txt").exists())
        self.assertEqual(len(graph.meta["sram_regions"]), 1)
        torch.testing.assert_close(graph(x), expected)

    def test_view_of_mutable_input_stays_on_original_path(self):
        from voyager_compiler import build_interstellar_tiler
        from voyager_compiler.codegen.transform.bufferize.residency import (
            plan_resident_regions,
        )
        from voyager_compiler.export_utils import export_model
        from voyager_compiler.shape_prop import ShapeProp

        class Mutate(torch.nn.Module):
            def forward(self, x):
                x.view(-1).add_(1)
                return x

        x = torch.zeros(4, 16)
        graph = export_model(Mutate(), (x,))
        ShapeProp(graph).propagate(x)
        plan_resident_regions(graph, build_interstellar_tiler(self.config()))
        self.assertEqual(graph.meta["sram_regions"], [])
        self.assertTrue(
            any(
                "view aliases" in r["reason"]
                for r in graph.meta["sram_fallbacks"]
            )
        )
        x.zero_()
        torch.testing.assert_close(graph(x), torch.ones_like(x))

        class View(torch.nn.Module):
            def forward(self, x):
                return x.reshape(-1)

        graph = export_model(View(), (x,))
        ShapeProp(graph).propagate(x)
        plan_resident_regions(graph, build_interstellar_tiler(self.config()))
        self.assertEqual(graph.meta["sram_regions"], [])
        self.assertFalse(
            any(
                n.target is torch.ops.voyager.async_copy.default
                for n in graph.graph.nodes
            )
        )

    def test_original_flow_and_final_placement_fallback(self):
        from voyager_compiler.codegen.transform.bufferize import memory_planning

        graph, x, expected = self.graph()
        self.lower(graph, self.config(), flow="per_kernel")
        torch.testing.assert_close(
            torch.utils._pytree.tree_leaves(graph(x))[0], expected
        )

        self.assertNotIn("sram_regions", graph.meta)
        graph, x, expected = self.graph()
        real = memory_planning.plan_memory
        failed = False

        def fail_final(model, config):
            nonlocal failed
            if model is graph and not failed:
                failed = True
                raise memory_planning.MemoryPlanningError(
                    "forced final placement conflict"
                )
            return real(model, config)

        with patch("voyager_compiler.plan_memory", side_effect=fail_final):
            self.lower(graph, self.config())
        self.assertTrue(failed)
        self.assertEqual(graph.meta["sram_regions"], [])
        torch.testing.assert_close(
            torch.utils._pytree.tree_leaves(graph(x))[0], expected
        )

    def test_bank_failure_splits_region_and_preserves_live_branch(self):
        graph, x, expected = self.graph(self.Parameters())
        self.lower(
            graph,
            self.config(num_banks=3, scratchpad_size=3 * 16384),
            strategy="preload",
        )
        self.assertGreater(len(graph.meta["sram_regions"]), 1)
        self.assertTrue(
            any(
                r["reason"] == "bank allocation"
                for r in graph.meta["sram_fallbacks"]
            )
        )
        torch.testing.assert_close(
            torch.utils._pytree.tree_leaves(graph(x))[0], expected
        )

    def test_resident_boundary_quantization_overlaps_dma_and_compute(self):
        from voyager_compiler import export_model
        from voyager_compiler.codegen.reporting import estimate_schedule
        from voyager_compiler.quantization.fake_quantize import (
            get_quantization_map,
        )
        from voyager_compiler.shape_prop import ShapeProp

        class Quant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("scale", torch.tensor(0.1))
                self.register_buffer("qmap", get_quantization_map("int8"))

            def forward(self, x):
                return torch.ops.quantized_ops.quantize.default(
                    x, self.scale, qmap=self.qmap
                )

        model = Quant()
        x = torch.randn(1, 3, 320, 320)
        config = self.config(scratchpad_size=2 * 1024**2)
        schedules = {}
        for flow in ("per_kernel", "resident"):
            graph = export_model(model, (x,))
            ShapeProp(graph).propagate(x)
            for n in graph.graph.nodes:
                if n.target is torch.ops.quantized_ops.quantize.default:
                    n.meta["dtype"] = "int8"
            self.lower(graph, config, flow=flow)
            torch.testing.assert_close(
                torch.utils._pytree.tree_leaves(graph(x))[0], model(x)
            )
            schedules[flow] = estimate_schedule(graph, config, full_walk=True)
        original, resident = schedules["per_kernel"], schedules["resident"]
        self.assertEqual(len(graph.meta["sram_regions"]), 1)
        self.assertEqual(original.dram_read_bytes, resident.dram_read_bytes)
        self.assertEqual(original.dram_write_bytes, resident.dram_write_bytes)
        self.assertLessEqual(resident.total_latency, original.total_latency)
        self.assertLess(
            resident.total_latency, resident.busy_compute + resident.busy_dram
        )
        first_compute = next(r for r in resident.records if r.kind == "compute")
        loads = [r for r in resident.records if r.kind == "load"]
        self.assertGreater(len(loads), 1)
        self.assertLess(first_compute.start, loads[-1].end)
        computes = [r for r in resident.records if r.kind == "compute"]
        stores = [r for r in resident.records if r.kind == "store"]
        self.assertLess(stores[0].start, computes[-1].end)
        folded = estimate_schedule(graph, config)
        self.assertEqual(folded.total_latency, resident.total_latency)

    def test_resident_matrix_search_prices_boundary_stores(self):
        from voyager_compiler import (
            convert_pt2e,
            get_default_quantizer,
            prepare_pt2e,
        )
        from voyager_compiler.codegen.reporting import estimate_schedule
        from voyager_compiler.targets import get_backend

        torch.manual_seed(21)
        x = torch.randn(1, 32, 16, 16)
        model = torch.nn.Conv2d(32, 128, 1).eval()
        quantizer = get_default_quantizer(
            input_activation="int8,qs=per_tensor_symmetric",
            weight="int8,qs=per_tensor_symmetric",
            bias="int24",
        )
        graph = prepare_pt2e(model, quantizer, (x,))
        graph(x)
        convert_pt2e(graph, "int24")
        expected = graph(x)
        config = self.config(scratchpad_size=1024 * 1024)
        transform(
            graph,
            (x,),
            config=config,
            patterns=get_backend("voyager").fusion_patterns(config),
        )
        self.lower(graph, config)
        torch.testing.assert_close(
            torch.utils._pytree.tree_leaves(graph(x))[0], expected
        )
        self.assertEqual(len(graph.meta["sram_regions"]), 1)
        result = estimate_schedule(graph, config, full_walk=True)
        conv_keys = {o.key for o in result.ops if o.op_type == "conv"}
        computes = [r for r in result.records if r.op_key in conv_keys]
        region_ids = {r["id"] for r in graph.meta["sram_regions"]}
        self.assertTrue(all(r.kernel in region_ids for r in computes))
        # No forced tile metadata: charging the boundary store must make the
        # search choose multiple compute tiles, although the full output fits.
        self.assertGreater(len(computes), 1)
        stores = [r for r in result.records if r.kind == "store"]
        self.assertGreater(len(stores), 1)
        self.assertLess(stores[0].start, computes[-1].end)

    def test_resident_native_padding_keeps_convolution_chain_on_chip(self):
        from voyager_compiler import (
            convert_pt2e,
            export_model,
            get_default_quantizer,
            prepare_pt2e,
        )
        from voyager_compiler.codegen.node_info import (
            get_anchor_node,
            is_conv2d,
        )
        from voyager_compiler.codegen.reporting import estimate_schedule
        from voyager_compiler.targets import get_backend

        class ConvChain(torch.nn.Module):
            def __init__(self, depthwise, stride):
                super().__init__()
                self.a = torch.nn.Conv2d(32, 32, 3, padding=1)
                self.b = torch.nn.Conv2d(
                    32,
                    32,
                    3,
                    padding=1,
                    stride=stride,
                    groups=32 if depthwise else 1,
                )

            def forward(self, x):
                return torch.relu(self.b(torch.relu(self.a(x)) + x))

        for quantized, depthwise, stride, strategy in (
            (False, False, 1, "on_demand"),
            (False, True, 2, "on_demand"),
            (True, False, 2, "on_demand"),
            (False, False, 1, "preload"),
        ):
            with self.subTest(
                quantized=quantized,
                depthwise=depthwise,
                stride=stride,
                strategy=strategy,
            ):
                torch.manual_seed(32)
                x = torch.randn(1, 32, 8, 8)
                model = ConvChain(depthwise, stride).eval()
                if quantized:
                    q = get_default_quantizer(
                        input_activation="int8,qs=per_tensor_symmetric",
                        weight="int8,qs=per_tensor_symmetric",
                        bias="int24",
                    )
                    graph = prepare_pt2e(model, q, (x,))
                    graph(x)
                    convert_pt2e(graph, "int24")
                else:
                    graph = export_model(model, (x,))
                expected = graph(x)
                config = self.config(scratchpad_size=1024 * 1024)
                transform(
                    graph,
                    (x,),
                    config=config,
                    patterns=get_backend("voyager").fusion_patterns(config),
                )
                if not quantized and not depthwise:
                    # Split IC and OC while retaining the full spatial image:
                    # hardware padding must survive every reduction round.
                    for n in graph.graph.nodes:
                        anchor = get_anchor_node(n)
                        if anchor is not None and is_conv2d(anchor):
                            anchor.meta["l2_tiling"] = (1, 2, 1, 1, 2)
                self.lower(graph, config, strategy=strategy)
                torch.testing.assert_close(
                    torch.utils._pytree.tree_leaves(graph(x))[0],
                    expected,
                    atol=1e-4,
                    rtol=5e-2,
                )
                self.assertEqual(len(graph.meta["sram_regions"]), 1)
                self.assertFalse(graph.meta["sram_fallbacks"])
                convs = [
                    n
                    for sub in graph.modules()
                    if isinstance(sub, torch.fx.GraphModule)
                    for n in sub.graph.nodes
                    if n.op == "call_function" and is_conv2d(n)
                ]
                self.assertTrue(convs)
                self.assertTrue(all(tuple(n.args[4]) == (1, 1) for n in convs))
                copies = [
                    n
                    for sub in graph.modules()
                    if isinstance(sub, torch.fx.GraphModule)
                    for n in sub.graph.nodes
                    if n.target is torch.ops.voyager.async_copy.default
                ]
                self.assertTrue(
                    all(not any(n.kwargs.get("pad") or ()) for n in copies)
                )
                result = estimate_schedule(graph, config)
                if not quantized:
                    self.assertEqual(
                        result.dram_write_bytes, expected.numel() * 4
                    )
                    self.assertEqual(
                        result.dram_read_bytes,
                        (x.numel() + sum(p.numel() for p in model.parameters()))
                        * 4,
                    )

    def test_resident_padding_rejects_explicit_spatial_tiles(self):
        from voyager_compiler import export_model
        from voyager_compiler.codegen.node_info import (
            get_anchor_node,
            is_conv2d,
        )
        from voyager_compiler.targets import get_backend

        x = torch.randn(1, 16, 8, 8)
        graph = export_model(torch.nn.Conv2d(16, 16, 3, padding=1).eval(), (x,))
        expected = graph(x)
        config = self.config(scratchpad_size=1024 * 1024)
        transform(
            graph,
            (x,),
            config=config,
            patterns=get_backend("voyager").fusion_patterns(config),
        )
        for n in graph.graph.nodes:
            anchor = get_anchor_node(n)
            if anchor is not None and is_conv2d(anchor):
                anchor.meta["l2_tiling"] = (1, 1, 2, 2)
        self.lower(graph, config)
        torch.testing.assert_close(
            torch.utils._pytree.tree_leaves(graph(x))[0], expected
        )
        self.assertFalse(graph.meta["sram_regions"])
        self.assertTrue(graph.meta["sram_fallbacks"])

    def test_pipelined_boundary_retains_shared_input_and_exported_intermediate(
        self,
    ):
        from voyager_compiler import compile, export_model
        from voyager_compiler.codegen.reporting import estimate_schedule
        from voyager_compiler.codegen.transform.bufferize import (
            BufferizationOptions,
        )

        class Branch(torch.nn.Module):
            def forward(self, x):
                a = torch.relu(x)
                return a, a * 2 + x

        x = torch.randn(64, 256)
        model = Branch()
        graph = export_model(model, (x,))
        config = self.config(scratchpad_size=512 * 1024)
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            compile(
                graph,
                (x,),
                config=config,
                output_dir=tmp,
                dump_tensors=False,
                bufferization_options=BufferizationOptions(flow="resident"),
            )
            self.assertTrue((Path(tmp) / "model.txt").exists())
        torch.testing.assert_close(graph(x), model(x))
        self.assertEqual(len(graph.meta["sram_regions"]), 1)
        result = estimate_schedule(graph, config, full_walk=True)
        self.assertEqual(result.dram_read_bytes, x.numel() * 4)
        self.assertEqual(result.dram_write_bytes, 2 * x.numel() * 4)
        self.assertLess(
            result.total_latency, result.busy_compute + result.busy_dram
        )

    def test_resident_activations_with_weights_larger_than_sram(self):
        from voyager_compiler import (
            convert_pt2e,
            get_default_quantizer,
            prepare_pt2e,
        )
        from voyager_compiler.codegen.reporting import estimate_schedule
        from voyager_compiler.codegen.transform.bufferize import (
            BufferizationOptions,
        )
        from voyager_compiler.export_utils import export_model
        from voyager_compiler.targets import get_backend

        for quantized in (False, True):
            with self.subTest(int8=quantized):
                torch.manual_seed(17)
                model = torch.nn.Sequential(
                    torch.nn.Linear(128, 128, bias=False),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128, bias=False),
                    torch.nn.ReLU(),
                ).eval()
                x = torch.randn(4, 128)
                if quantized:
                    q = get_default_quantizer(
                        input_activation="int8,qs=per_tensor_symmetric",
                        weight="int8,qs=per_tensor_symmetric",
                        bias="int24",
                    )
                    graph = prepare_pt2e(model, q, (x,))
                    graph(x)
                    convert_pt2e(graph, "int24")
                else:
                    graph = export_model(model, (x,))
                expected = graph(x)
                config = self.config(
                    scratchpad_size=16384 if quantized else 65536
                )
                transform(
                    graph,
                    (x,),
                    config=config,
                    patterns=get_backend("voyager").fusion_patterns(config),
                )
                with (
                    tempfile.TemporaryDirectory() as tmp,
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertNoLogs(
                        "voyager_compiler.codegen.transform.bufferize.memory_planning",
                        level="WARNING",
                    ),
                ):
                    compile(
                        graph,
                        (x,),
                        config=config,
                        output_dir=tmp,
                        dump_tensors=False,
                        bufferization_options=BufferizationOptions(
                            flow="resident"
                        ),
                    )
                torch.testing.assert_close(
                    torch.utils._pytree.tree_leaves(graph(x))[0], expected
                )
                self.assertEqual(len(graph.meta["sram_regions"]), 1)
                self.assertEqual(
                    len(graph.meta["sram_regions"][0]["streamed_parameters"]), 2
                )
                result = estimate_schedule(graph, config)
                self.assertEqual(
                    result.dram_activation_bytes,
                    (x.numel() + expected.numel()) * 4,
                )
                self.assertGreaterEqual(
                    result.dram_weight_bytes,
                    2 * 128 * 128 * (1 if quantized else 4),
                )

    def test_resident_matrix_tiles_complete_reduction_before_relu(self):
        from voyager_compiler.codegen.reporting import estimate_schedule
        from voyager_compiler.codegen.transform.bufferize import (
            BufferizationOptions,
        )
        from voyager_compiler.codegen.transform.tiling import tiler as tiling
        from voyager_compiler.targets import get_backend

        searches = []
        run_search = tiling._run_search

        def constrained_search(search):
            self.assertIsNotNone(search.tiler.resident_bytes)
            self.assertTrue(search.rc.resident)
            found = run_search(search)
            if not any(search.rc.resident_boundary):
                self.assertEqual(sum(search.rc.dram_bytes.values()), 0)
            else:
                self.assertGreater(sum(search.rc.dram_bytes.values()), 0)
            searches.append(search)
            return found

        class Matrix(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a, self.b = (
                    torch.nn.Linear(16, 16, bias=False),
                    torch.nn.Linear(16, 16, bias=False),
                )

            def forward(self, x):
                # The in-place tail owns the matrix intermediate, not x.
                return torch.relu_(self.b(torch.relu_(self.a(x))))

        for outer in (False, True):
            with self.subTest(outer=outer):
                torch.manual_seed(13)
                graph, x, expected = self.graph(Matrix())
                config = self.config()
                transform(
                    graph,
                    (x,),
                    config=config,
                    patterns=get_backend("voyager").fusion_patterns(config),
                )
                first = next(
                    n for n in graph.graph.nodes if n.op == "call_module"
                )
                from voyager_compiler.codegen.node_info import get_anchor_node

                if outer:
                    get_anchor_node(first).meta["l2_tiling"] = (2, 2, 2)
                    second = [
                        n for n in graph.graph.nodes if n.op == "call_module"
                    ][1]
                    get_anchor_node(second).meta["l2_tiling"] = (1, 2, 2)
                with (
                    tempfile.TemporaryDirectory() as tmp,
                    contextlib.redirect_stdout(io.StringIO()),
                    patch.object(
                        tiling, "_run_search", side_effect=constrained_search
                    ),
                ):
                    compile(
                        graph,
                        (x,),
                        config=config,
                        output_dir=tmp,
                        dump_tensors=False,
                        bufferization_options=BufferizationOptions(
                            flow="resident"
                        ),
                    )
                    self.assertTrue((Path(tmp) / "model.txt").exists())
                torch.testing.assert_close(
                    torch.utils._pytree.tree_leaves(graph(x))[0], expected
                )
                self.assertEqual(len(graph.meta["sram_regions"]), 1)
                self.assertEqual(len(graph.meta["sram_regions"][0]["nodes"]), 2)
                result = estimate_schedule(graph, config)
                self.assertEqual(
                    result.dram_read_bytes, 2 * 16 * 16 * 4 + x.numel() * 4
                )
                self.assertEqual(result.dram_write_bytes, expected.numel() * 4)
                # Boundary DMA is now inside the tile loops. Repeated input
                # windows still load once, and stores follow the final reduction.
                unfolded = estimate_schedule(graph, config, full_walk=True)
                self.assertEqual(result.total_latency, unfolded.total_latency)
                self.assertEqual(
                    result.dram_read_bytes, unfolded.dram_read_bytes
                )
                self.assertEqual(
                    result.dram_write_bytes, unfolded.dram_write_bytes
                )
        self.assertTrue(searches)


if __name__ == "__main__":
    unittest.main()

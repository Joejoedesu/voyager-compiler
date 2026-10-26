"""Target quantization dispatch, replayable calibration and target fusion wiring."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from voyager_compiler.compiler.quantize import quantize
from voyager_compiler.extensions import (
    QuantizationPolicy,
    WorkloadSpec,
    describe_target,
    registry,
)
from voyager_compiler.ir.graph import PreparedWorkload
from voyager_compiler.pipeline import compile_workload
from voyager_compiler.runtime import run


class QuantizationFlowTests(unittest.TestCase):
    def test_override_receives_compile_config_and_is_not_converted_twice(self):
        target = registry.target("voyager")
        config = target.hardware(
            pe_array_size=(4, 4),
            scratchpad_size=4096,
            num_banks=4,
            bank_width=4,
            vector_unit_width=4,
        )
        model = torch.nn.ReLU()
        x = torch.randn(4, 4)
        caller_quantizer = object()
        seen = []

        def override(prepared, actual_config, options):
            self.assertIs(actual_config, config)
            self.assertIs(options["quantizer"], caller_quantizer)
            seen.append(actual_config)
            prepared.provenance["quantization"] = {"flow": "test override"}
            return prepared

        target = replace(target, quantization=QuantizationPolicy(run=override))
        workload = WorkloadSpec(
            "override", lambda _: model, lambda m, o: ((x,), {})
        )
        with (
            tempfile.TemporaryDirectory() as output,
            patch(
                "voyager_compiler.compiler.quantize.convert_pt2e",
                side_effect=AssertionError(
                    "override must not enter shared conversion"
                ),
            ),
        ):
            result = compile_workload(
                workload,
                target,
                output,
                config=config,
                options={"quantizer": caller_quantizer},
                dump_tensors=False,
            )
            run(result, reference=model(x), rtol=0, atol=0)
        self.assertEqual(len(seen), 1)
        p = PreparedWorkload(model, (x,))
        self.assertIs(
            quantize(p, target, {"quantizer": None}, config=config), p
        )
        self.assertEqual(len(seen), 1)

    def test_alpha_two_pass_matches_explicit_hardware_aware_recipe(self):
        from voyager_compiler.targets.magnet_alpha import (
            quantization_flow as flow,
        )
        from voyager_compiler.targets.magnet_alpha.hardware import (
            MagnetAlphaHardwareConfig,
        )
        from voyager_compiler.targets.magnet_alpha.observers import (
            FusedAmaxObsFakeQuantize,
        )
        from voyager_compiler.targets.magnet_alpha.ptq import (
            get_default_quantizer,
        )
        from voyager_compiler.targets.magnet_alpha.qspec import QuantizationSpec

        torch.manual_seed(23)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 1),
            torch.nn.ReLU(),
        ).eval()
        x = torch.randn(1, 32, 4, 4)
        sample_tensors = [x, x * 2]
        config = MagnetAlphaHardwareConfig(frequency=123456789)
        q = get_default_quantizer(
            input_activation="int4,qs=vsq,coarse=int8,cqs=per_tensor_symmetric,bs=32,scale=uint8,sqmax=8,csqmax=126",
            weight="int4,qs=vsq,coarse=int8,cqs=per_channel_symmetric,cax=0,bs=32,scale=uint8,qmin=-7,qmax=7",
        )
        region = QuantizationSpec.from_str(
            "int8,qs=per_tensor_symmetric,sqmax=126,rnd=floor"
        )
        region.observer_or_fake_quant_ctr = FusedAmaxObsFakeQuantize.with_args(
            force_scale_power_of_two=False
        )
        recipe = flow.AlphaQuantizationRecipe(
            q,
            lambda: iter([((t,), {}) for t in sample_tensors]),
            steps=2,
            region_qspec=region,
        )
        # Independent explicit ordering of the established recipe, before refactor.
        baseline = flow.prepare_pt2e(
            copy.deepcopy(model),
            copy.deepcopy(q),
            (x,),
            hardware_config=config,
            region_qspec=recipe.region_qspec,
        )
        with torch.no_grad():
            for tensor in sample_tensors:
                baseline(tensor)
            flow.finalize_pt2e_calibration(baseline, config)
            configured = flow.prepare_quantization_recalibration(
                baseline, reset_observers=True
            )
            self.assertGreater(configured, 0)
            for tensor in sample_tensors:
                flow.prepare_quantization_recalibration(
                    baseline, reset_observers=False
                )
                baseline(tensor)
            flow.convert_pt2e(
                baseline,
                recipe.bias,
                calibration_steps=2,
                hardware_config=config,
            )
        calls = []

        def samples():
            calls.append(1)
            return iter([((t,), {}) for t in sample_tensors])

        recipe = replace(recipe, samples=samples, quantizer=copy.deepcopy(q))
        prepared = PreparedWorkload(copy.deepcopy(model), (x,))
        with patch.object(
            flow, "prepare_pt2e", wraps=flow.prepare_pt2e
        ) as prepare:
            result = quantize(
                prepared,
                registry.target("magnet-alpha"),
                {"alpha_quantization": recipe},
                config=config,
            )
        self.assertIs(prepare.call_args.kwargs["hardware_config"], config)
        self.assertEqual(len(calls), 2)
        self.assertGreater(
            result.provenance["quantization"]["configured_exact_boundaries"], 0
        )
        with torch.no_grad():
            for tensor in [*sample_tensors, x * 0.7]:
                torch.testing.assert_close(
                    result.graph(tensor), baseline(tensor), rtol=0, atol=0
                )
        # A one-shot source must not silently turn the second pass into zero samples.
        stream = iter([((t,), {}) for t in sample_tensors])
        with self.assertRaisesRegex(ValueError, "sample count 0 != 2"):
            flow.calibrate_and_convert(
                copy.deepcopy(model),
                (x,),
                {},
                config,
                replace(recipe, samples=lambda: stream),
                copy.deepcopy(q),
            )

    def test_alpha_saved_profile_skips_fresh_calibration(self):
        from voyager_compiler.targets.magnet_alpha import (
            quantization_flow as flow,
        )
        from voyager_compiler.targets.magnet_alpha.hardware import (
            MagnetAlphaHardwareConfig,
        )

        graph = torch.fx.symbolic_trace(torch.nn.ReLU())
        events = []
        recipe = flow.AlphaQuantizationRecipe(
            quantizer=object(),
            samples=lambda: self.fail("saved state must not recalibrate"),
            restore_profile=lambda g, c: (
                events.append("restore") or {"asset": "fixture"}
            ),
            expected_exact_boundaries=3,
            profile="saved",
        )
        with (
            patch.object(flow, "prepare_pt2e", return_value=graph),
            patch.object(
                flow,
                "finalize_pt2e_calibration",
                side_effect=lambda *a: events.append("finalize"),
            ),
            patch.object(
                flow,
                "prepare_quantization_recalibration",
                side_effect=lambda g, **kw: events.append(kw) or 3,
            ),
            patch.object(
                flow,
                "convert_pt2e",
                side_effect=lambda *a, **kw: events.append("convert"),
            ),
        ):
            _, report = flow.calibrate_and_convert(
                graph,
                (torch.ones(1),),
                {},
                MagnetAlphaHardwareConfig(),
                recipe,
                recipe.quantizer,
            )
        self.assertEqual(
            events,
            ["restore", "finalize", {"reset_observers": False}, "convert"],
        )
        self.assertEqual(report["asset"], "fixture")

    def test_trainium_fused_relu_emits_and_replays(self):
        target = registry.target("trainium")
        torch.manual_seed(17)
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 32), torch.nn.ReLU()
        ).eval()
        x = torch.randn(32, 32)
        with tempfile.TemporaryDirectory() as output, torch.no_grad():
            result = compile_workload(
                WorkloadSpec(
                    "linear-relu", lambda _: model, lambda m, o: ((x,), {})
                ),
                target,
                output,
                dump_tensors=False,
            )
            run(result, reference=model(x))
            emitted = (Path(output) / "nki/kernels.py").read_text()
            self.assertIn("op0=nl.maximum, operand0=0.0", emitted)
            compile(emitted, "kernels.py", "exec")

    def test_registered_fusion_matches_target_table(self):
        from voyager_compiler import export_model, transform
        from voyager_compiler.targets.trainium.fusion import (
            FUSION_PATTERNS as trainium_patterns,
        )
        from voyager_compiler.targets.voyager.fusion import (
            FUSION_PATTERNS as voyager_patterns,
        )

        self.assertIsNot(voyager_patterns, trainium_patterns)
        for name, patterns in [
            ("voyager", voyager_patterns),
            ("trainium", trainium_patterns),
        ]:
            with self.subTest(target=name), torch.no_grad():
                target = registry.target(name)
                discovery = json.loads(json.dumps(describe_target(target)))
                self.assertEqual(
                    len(discovery["normalization"]["patterns"]), len(patterns)
                )
                self.assertIs(dict(target.normalization)["patterns"], patterns)
                torch.manual_seed(5)
                model = torch.nn.Sequential(
                    torch.nn.Linear(32, 32), torch.nn.ReLU()
                ).eval()
                x = torch.randn(32, 32)
                graph = export_model(model, (x,))
                transform(
                    graph,
                    (x,),
                    config=target.hardware(),
                    **dict(target.normalization),
                )
                self.assertNotIn(
                    torch.ops.aten.relu.default,
                    [n.target for n in graph.graph.nodes],
                )
                self.assertTrue(
                    any(n.op == "call_module" for n in graph.graph.nodes)
                )
                torch.testing.assert_close(graph(x), model(x))


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()

"""Cross-stage contracts required for adding a target without editing searches."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from interstellar import Layer
from voyager_compiler import export_model
from voyager_compiler.compilation import CompilerContext
from voyager_compiler.hardware_config import VOYAGER
from voyager_compiler.gemmini.hardware import lean_config
from voyager_compiler.gemmini.mapping import GemminiMappingPolicy, GemminiTuning
from voyager_compiler.gemmini.scheduling import SubmissionPolicy
from voyager_compiler.gemmini.execution import GemminiCostModel
from voyager_compiler.codegen.transform.tiling.contracts import (
    StorageRequirement,
    resources_fit,
    NonMatrixFootprint,
)
from voyager_compiler.codegen.transform.tiling.policy import (
    VoyagerMappingPolicy,
)
from voyager_compiler.codegen.transform.tiling.search import _search_tiling
from voyager_compiler.codegen.transform.tiling.traversal import MappingTraversal
from test_mapping_policy import point


class ExtensibilityTest(unittest.TestCase):
    def test_resolved_policy_roundtrip_preserves_nondefault_options(self):
        hardware = lean_config()
        policy = GemminiMappingPolicy(
            hardware, GemminiTuning(False, 3, SubmissionPolicy(8, 2, 2))
        )
        for context in (
            CompilerContext.resolve(hardware, policy),
            CompilerContext.resolve(VOYAGER),
        ):
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "compilation.json"
                path.write_text(json.dumps({"existing_field": "preserve"}))
                context.write(td)
                restored = CompilerContext.from_artifacts(td, context.hardware)
                self.assertEqual(restored.record(), context.record())
                self.assertEqual(
                    json.loads(path.read_text())["existing_field"], "preserve"
                )
        self.assertFalse(CompilerContext(hardware, policy).cost_tradeoff)

    def test_mismatched_conversion_context_is_rejected(self):
        hardware = lean_config()
        original = CompilerContext.resolve(hardware)
        changed = CompilerContext.resolve(
            hardware,
            GemminiMappingPolicy(
                hardware, GemminiTuning(submission=SubmissionPolicy(8, 2, 2))
            ),
        )
        with tempfile.TemporaryDirectory() as td:
            original.write(td)
            with self.assertRaisesRegex(ValueError, "differs from the policy"):
                changed.check_artifacts(td)
            record = json.loads((Path(td) / "compilation.json").read_text())
            record["compiler"]["hardware_sha256"] = "wrong"
            (Path(td) / "compilation.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "hardware/backend"):
                CompilerContext.from_artifacts(td, hardware)

    def test_legacy_artifact_requires_explicit_policy(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "explicit context"):
                CompilerContext.from_artifacts(td, lean_config())
            # Explicit policy acknowledges that old artifacts do not record it.
            CompilerContext.resolve(lean_config()).check_artifacts(td)

    def test_storage_checks_sum_aligned_generations_per_physical_memory(self):
        hardware = lean_config()
        fits = (
            StorageRequirement("scratchpad", 65537, 2, 65536),
            StorageRequirement("accumulator", 16385, 2, 32768),
        )
        self.assertTrue(resources_fit(hardware, fits))
        self.assertFalse(
            resources_fit(
                hardware, fits + (StorageRequirement("scratchpad", 1),)
            )
        )
        self.assertFalse(
            resources_fit(
                hardware, fits + (StorageRequirement("accumulator", 1),)
            )
        )
        with self.assertRaises(ValueError):
            StorageRequirement("scratchpad", 1, 0)

    def test_candidate_plan_does_not_depend_on_last_estimator_call(self):
        estimator = GemminiCostModel(MappingTraversal())
        layer = Layer(256, 64, 12544, 1, 1, 1)
        first = estimator.evaluate(None, layer, point(128))
        different = estimator.evaluate(None, layer, point(16))
        self.assertNotEqual(first.cycles, different.cycles)
        self.assertEqual(first, estimator.evaluate(None, layer, point(128)))
        self.assertTrue(resources_fit(lean_config(), first.storage))
        self.assertFalse(hasattr(estimator, "plan"))
        self.assertEqual(first.buffer_plan.accumulator_copies, 2)

    def test_nonmatrix_search_accepts_target_footprint_groups_and_score(self):
        class Policy(VoyagerMappingPolicy):
            def nonmatrix_slot_size(self, node, slots):
                return 16

            def nonmatrix_footprint(self, node, shapes, sharing, default):
                return NonMatrixFootprint(
                    shapes[node][0], ("target-storage-group",)
                )

            def nonmatrix_cost(self, kind, node, default):
                return lambda node, sizes, shapes, counts: (
                    abs(sizes[0] - 8),
                    0,
                )

        class Model(torch.nn.Module):
            def forward(self, x):
                return x.relu()

        model = export_model(Model(), (torch.zeros(32),))
        node = next(n for n in model.graph.nodes if n.op == "call_function")
        result = _search_tiling(
            node,
            (32,),
            lambda node, tile, counts: {node: tile},
            VOYAGER,
            policy=Policy(VOYAGER),
        )
        self.assertEqual(result, ((8,), ("target-storage-group",), 0))

    def test_policy_hardware_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "hardware differs"):
            CompilerContext.resolve(
                VOYAGER, GemminiMappingPolicy(lean_config())
            )


if __name__ == "__main__":
    unittest.main()

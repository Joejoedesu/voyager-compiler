"""Compare complete algorithms at one fixed input/output boundary."""

import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from voyager_compiler.compiler.allocate import MemoryCapacityError, plan_memory
from voyager_compiler.compiler.bufferize.builders import BuildContext
from voyager_compiler.compiler.bufferize.regions import build_sequential_region
from voyager_compiler.compiler.normalize import transform
from voyager_compiler.compiler.search.tiler import Tiler
from voyager_compiler.export_utils import export_model
from voyager_compiler.extensions import CandidateRejected, WorkloadSpec
from voyager_compiler.hardware import AcceleratorConfig
from voyager_compiler.ir.plan import Cost
from voyager_compiler.pipeline import compile_workload
from voyager_compiler.runtime import run
from voyager_compiler.targets.voyager.target import (
    TARGET,
    with_attention_algorithms,
)


class RegionAlgorithmTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = AcceleratorConfig(
            pe_array_size=(4, 4),
            scratchpad_size=65536,
            num_banks=8,
            bank_width=16,
            vector_unit_width=4,
            input_buffer_size=16,
            weight_buffer_size=16,
            accum_buffer_size=16,
        )

    def compile_attention(self, target, lengths=(8, 8), scale=None):
        class Attention(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, scale=scale
                )

        q, k = lengths
        args = (
            torch.randn(1, q, 8),
            torch.randn(1, k, 8),
            torch.randn(1, k, 8),
        )
        model = Attention().eval()
        workload = WorkloadSpec(
            "attention", lambda _: model, lambda m, o: (args, {})
        )
        with tempfile.TemporaryDirectory() as output:
            result = compile_workload(
                workload, target, output, config=self.config, dump_tensors=False
            )
            run(result, reference=model(*args), rtol=1e-4, atol=1e-5)
        choices = [
            c
            for c in result.program.plan.choices.values()
            if c.implementation.startswith("voyager.attention.")
        ]
        self.assertEqual(len(choices), 1)
        self.assertTrue(result.program.plan.frozen)
        return result, choices[0]

    def test_both_algorithms_replay_including_tails_and_custom_scale(self):
        target = with_attention_algorithms()
        for algorithm in ("materialized", "streaming"):
            selected = replace(
                target,
                implementations=tuple(
                    s
                    for s in target.implementations
                    if not s.name.startswith("voyager.attention.")
                    or s.name.endswith("." + algorithm)
                ),
            )
            for lengths in (
                ((8, 8), (5, 7), (32, 64))
                if algorithm == "materialized"
                else ((8, 8), (8, 12), (32, 64))
            ):
                with self.subTest(algorithm=algorithm, lengths=lengths):
                    _, choice = self.compile_attention(
                        selected, lengths, scale=0.25
                    )
                    self.assertEqual(
                        choice.implementation, "voyager.attention." + algorithm
                    )
                    self.assertGreater(choice.cost.latency, 0)
                    if algorithm == "materialized":
                        self.assertGreater(len(choice.components), 1)
                        self.assertTrue(
                            all(
                                c.tile is not None for _, c in choice.components
                            )
                        )
                        self.assertAlmostEqual(
                            choice.cost.latency,
                            sum(c.cost.latency for _, c in choice.components),
                        )
                        self.assertGreater(choice.cost.dram_bytes, 0)
                    else:
                        self.assertEqual(choice.components, ())

    def test_selection_compares_both_regions_and_preserves_default(self):
        default = TARGET.implementations
        target = with_attention_algorithms()
        result, _ = self.compile_attention(target)
        alternatives = next(
            v
            for v in result.program.plan.alternatives.values()
            if any(c.implementation.endswith(".materialized") for c in v)
        )
        self.assertEqual(
            {c.implementation for c in alternatives},
            {"voyager.attention.materialized", "voyager.attention.streaming"},
        )
        self.assertIs(TARGET.implementations, default)
        # The materialized recipe can pad individual GEMMs without adding keys
        # to softmax. Streaming currently needs a legal unpadded product mapping.
        tail_result, tail_choice = self.compile_attention(target, (5, 7))
        self.assertEqual(
            tail_choice.implementation, "voyager.attention.materialized"
        )
        self.assertTrue(
            any(
                "no tiling" in reason
                for reasons in tail_result.program.plan.rejected.values()
                for reason in reasons
            )
        )
        # An attributable model change must change selection, not registration order.
        for winner in ("materialized", "streaming"):
            with self.subTest(winner=winner):
                priced = replace(
                    target,
                    implementations=tuple(
                        (
                            replace(s, estimate=lambda *args: Cost(0.001, 1))
                            if s.name.endswith("." + winner)
                            else s
                        )
                        for s in reversed(target.implementations)
                    ),
                )
                _, choice = self.compile_attention(priced)
                self.assertEqual(
                    choice.implementation, "voyager.attention." + winner
                )

    def test_unsupported_numerics_are_rejected(self):
        class Dropout(torch.nn.Module):
            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, dropout_p=0.2
                )

        args = tuple(torch.randn(1, 8, 8) for _ in range(3))
        workload = WorkloadSpec(
            "dropout", lambda _: Dropout(), lambda m, o: (args, {})
        )
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(
                CandidateRejected, "excludes masks, dropout"
            ):
                compile_workload(
                    workload,
                    with_attention_algorithms(),
                    output,
                    config=self.config,
                    dump_tensors=False,
                )

    def test_streaming_replays_multiple_query_and_key_blocks(self):
        from voyager_compiler.targets.voyager.implementations import (
            VoyagerModel,
        )

        class SmallTilesModel(VoyagerModel):
            def prepare_attention(self, node, acc_dtype, bool_mask):
                op = super().prepare_attention(node, acc_dtype, bool_mask)
                fit = op.fit
                op.fit = lambda tiling: (
                    fit(tiling) if tiling.tile_sizes == (4, 4) else None
                )
                return op

        target = with_attention_algorithms()
        target = replace(
            target,
            implementations=tuple(
                (
                    replace(
                        s,
                        model=lambda config, accumulate, parameters: SmallTilesModel(
                            config, accumulate
                        ),
                    )
                    if s.name.endswith(".streaming")
                    else s
                )
                for s in target.implementations
                if not s.name.endswith(".materialized")
            ),
        )
        _, choice = self.compile_attention(target, (8, 12))
        self.assertEqual(choice.tile.tiling.grid[-2:], (2, 3))

    def test_sequential_region_is_private_and_requires_complete_costs(self):
        class Pair(torch.nn.Module):
            def forward(self, x):
                return (torch.relu(x) + 1,)

        args = (torch.randn(4, 4),)
        graph = export_model(Pair(), args)
        transform(graph, args, config=self.config, skip_op_fusion=True)
        original = graph.code
        context = BuildContext(
            Tiler(TARGET.model(self.config, False), 0),
            2,
            True,
            False,
            False,
            True,
            True,
        )
        built = build_sequential_region(graph, context, TARGET.implementations)
        self.assertEqual(graph.code, original)
        self.assertIsNot(built, graph)
        self.assertEqual(len(built.meta["region_choices"]), 2)
        missing = tuple(
            replace(s, estimate=lambda *args: None)
            for s in TARGET.implementations
        )
        with self.assertRaisesRegex(
            CandidateRejected, "every region component"
        ):
            build_sequential_region(graph, context, missing)

    def test_child_allocation_failure_replans_its_own_region(self):
        class Pair(torch.nn.Module):
            def forward(self, q, k, v):
                return (
                    torch.nn.functional.scaled_dot_product_attention(q, k, v),
                    torch.nn.functional.scaled_dot_product_attention(
                        q, k, v * 2
                    ),
                )

        args = tuple(torch.randn(1, 8, 8) for _ in range(3))
        target = with_attention_algorithms()
        target = replace(
            target,
            implementations=tuple(
                (
                    replace(s, estimate=lambda *args: Cost(100))
                    if s.name.endswith(".materialized")
                    else (
                        replace(s, estimate=lambda *args: Cost(200))
                        if s.name.endswith(".streaming")
                        else s
                    )
                )
                for s in target.implementations
            ),
        )
        calls = []

        def allocate(graph, config, dram_layout):
            calls.append(1)
            if len(calls) == 1:
                raise MemoryCapacityError(
                    "fixture: second region needs another algorithm",
                    sites=("scaled_dot_product_attention_1/matmul_default",),
                )
            return plan_memory(graph, config, dram_layout)

        workload = WorkloadSpec(
            "pair", lambda _: Pair(), lambda m, o: (args, {})
        )
        with (
            tempfile.TemporaryDirectory() as output,
            patch(
                "voyager_compiler.pipeline.plan_memory", side_effect=allocate
            ),
        ):
            result = compile_workload(
                workload,
                target,
                output,
                config=self.config,
                dump_tensors=False,
                max_replans=1,
            )
            run(result, reference=Pair()(*args), rtol=1e-4, atol=1e-5)
        choices = result.program.plan.choices
        self.assertTrue(
            choices["scaled_dot_product_attention"].implementation.endswith(
                ".materialized"
            )
        )
        self.assertTrue(
            choices["scaled_dot_product_attention_1"].implementation.endswith(
                ".streaming"
            )
        )
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()

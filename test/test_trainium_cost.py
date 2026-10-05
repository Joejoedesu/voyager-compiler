"""Trainium search contracts: instructions, DMA, buffering and speed objective."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from interstellar import Layer, MappingPoint
from interstellar import loop_enum as le
from interstellar import mapping_point_generator as mpg
from voyager_compiler.codegen.transform.tiling.policy import (
    VoyagerMappingPolicy,
)
from voyager_compiler.codegen.transform.tiling.tiler import (
    RuntimeCalculator,
    build_interstellar_tiler,
)
from voyager_compiler.hardware_config import VOYAGER
from voyager_compiler.trainium.cost import (
    TrainiumCostModel,
    dma_service,
    matmul_ns,
)
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.mapping import (
    TrainiumMappingPolicy,
    make_plan,
    slot_bytes,
)


def point(m=128, n=128, k=128, gm=1, gn=1, gk=1):
    blocks = [[1] * 4 for _ in range(le.NUM)]
    parts = [[1] * 4 for _ in range(le.NUM)]
    orders = [[6] * 4 for _ in range(le.NUM)]
    for d, tile, grid, rank in (
        (le.OX, m, gm, 2),
        (le.OC, n, gn, 1),
        (le.IC, k, gk, 0),
    ):
        blocks[d][1], blocks[d][3], orders[d][3] = (
            tile,
            grid,
            rank if grid > 1 else 6,
        )
    return MappingPoint(orders, blocks, parts)


def cost(config, batch=1, bias=0):
    rc = RuntimeCalculator.__new__(RuntimeCalculator)
    rc.batch = rc.weight_batch = batch
    rc.input_dtype_width = rc.weight_dtype_width = rc.output_dtype_width = 32
    rc.weight_transposed = rc.has_tail = False
    rc.weight_hbm_ck = True
    rc.bias_width = bias
    return TrainiumCostModel(rc, config)


class TrainiumCostTest(unittest.TestCase):
    def test_target_sizes_bypass_voyager_banking_and_keep_semantic_constraints(
        self,
    ):
        policy = TrainiumMappingPolicy(neuron_core(2))
        rc = cost(policy.config).shared
        node = SimpleNamespace(target=torch.ops.aten.matmul.default, meta={})

        def forbidden(**kwargs):
            raise AssertionError(
                "Voyager size policy must not run for Trainium"
            )

        sizes, _ = policy.prepare(forbidden, rc, 1, node=node)
        p = point(256, 256, 256)
        footprint = sizes((65536, 65536, 65536), p, 2, None, None, None)
        self.assertLess(footprint[0], policy.config.usable_scratchpad_size)
        self.assertEqual(footprint[1:], (0, 0))
        blocked, _ = policy.prepare(
            forbidden,
            rc,
            1,
            node=node,
            constraint=SimpleNamespace(allows=lambda extent: False),
        )
        self.assertEqual(
            blocked((65536,) * 3, p, 2, None, None, None), (float("inf"),) * 3
        )

    def test_policy_preserves_default_and_uses_speed_only(self):
        c = neuron_core(2)
        self.assertIn(
            "level2",
            VoyagerMappingPolicy(VOYAGER).schedule()["schedule_hint"]["IC"],
        )
        policy = TrainiumMappingPolicy(c)
        self.assertNotIn("level2", policy.schedule()["schedule_hint"]["IC"])
        self.assertEqual(
            policy.schedule()["schedule_hint"]["IC"]["level3"]["order"], 0
        )
        self.assertFalse(build_interstellar_tiler(c).interstellar_cost_tradeoff)
        self.assertTrue(
            build_interstellar_tiler(VOYAGER).interstellar_cost_tradeoff
        )

    def test_short_reduction_does_not_reduce_instruction_time(self):
        for version in (2, 3):
            c = neuron_core(version)
            self.assertEqual(
                matmul_ns(c, 512, 128, 16, 32), matmul_ns(c, 512, 128, 128, 32)
            )
            self.assertAlmostEqual(
                matmul_ns(c, 512, 128, 128, 32), 4 * 512 / c.frequency
            )
            self.assertEqual(
                matmul_ns(c, 128, 128, 128, 32),
                4 * matmul_ns(c, 128, 128, 128, 16),
            )

    def test_dma_partition_underfill_and_fragmentation(self):
        c = neuron_core(2)
        full, narrow = dma_service(c, 128, 128, 32), dma_service(c, 8, 128, 32)
        self.assertEqual(full.payload_ns, narrow.payload_ns)
        self.assertEqual(full.bytes, 16 * narrow.bytes)
        fragmented = dma_service(c, 128, 128, 32, row_block=8)
        self.assertEqual(fragmented.bytes, full.bytes)
        self.assertEqual(fragmented.commands, 16 * full.commands)
        self.assertEqual(fragmented.startup_ns, 16 * full.startup_ns)
        faster = replace(
            c,
            connections=tuple(
                replace(
                    x,
                    bandwidth=replace(x.bandwidth, value=x.bandwidth.value * 2),
                )
                if x.name == "HBM_DMA"
                else x
                for x in c.connections
            ),
        )
        self.assertEqual(
            dma_service(faster, 128, 128, 32).payload_ns, full.payload_ns / 2
        )

    def test_partition_pitch_and_batch_slots(self):
        self.assertEqual(slot_bytes(128, 128, 32), 2048)
        self.assertEqual(make_plan(point()).output_slots, 1)
        plan = make_plan(point(), batch_count=2)
        self.assertEqual(
            (plan.input_slots, plan.weight_slots, plan.output_slots), (2, 2, 2)
        )
        # GEMM's batch loop already exists; batch_tiles only adds conv batches.
        self.assertEqual(plan.batch_tiles, 1)

    def test_retaining_input_reduces_emitted_traffic_and_latency(self):
        c = neuron_core(2)
        model = cost(c)
        layer = Layer(256, 256, 256, 1, 1, 1)
        old = model.calculate_runtime(
            None, layer, point(128, 128, 128, 2, 2, 2)
        )
        old_bytes = model.estimate["hbm_bytes"]
        new = model.calculate_runtime(
            None, layer, point(256, 128, 256, 1, 2, 1)
        )
        self.assertEqual(old_bytes, 1310720)
        self.assertEqual(model.estimate["hbm_bytes"], 786432)
        self.assertEqual(model.estimate["tensor_instructions"], 4)
        self.assertEqual(model.estimate["dma_commands"], 12)
        self.assertLess(new, old)
        self.assertLess(
            model.estimate["zero_startup_sensitivity_ns"],
            model.estimate["predicted_ns"],
        )

    def test_bias_is_loaded_once_when_retained(self):
        p, layer = point(gk=2), Layer(256, 128, 128, 1, 1, 1)
        plain, biased = cost(neuron_core(2)), cost(neuron_core(2), bias=32)
        plain.calculate_runtime(None, layer, p)
        biased.calculate_runtime(None, layer, p)
        self.assertEqual(
            biased.estimate["hbm_bytes"] - plain.estimate["hbm_bytes"], 512
        )
        self.assertEqual(
            biased.estimate["dma_commands"] - plain.estimate["dma_commands"], 1
        )

    def test_runtime_only_ignores_traffic_and_tolerance(self):
        resource = SimpleNamespace(para_index=[])
        candidates = [
            SimpleNamespace(runtime=100, traffic=1000),
            SimpleNamespace(runtime=101, traffic=1),
        ]
        generator_name = "opt_mapping_point_generator_function"
        # Use the actual selector with a controlled candidate generator.
        for candidate in candidates:
            candidate.loop_partitionings = []
            candidate.para_loop_dim = None
        with (
            patch.object(
                mpg.cost_model, "get_ideal_performance", return_value=1
            ),
            patch.object(
                mpg,
                "opt_get_loop_order_generator",
                return_value=iter(candidates),
            ),
            patch.object(
                mpg,
                "blocking_partitioning_generator_function",
                return_value=[([[]], [[]], None)],
            ),
            patch.object(mpg, "partitioned_loop_string", return_value=("", 1)),
            patch.object(mpg, "get_utilization", return_value=1),
        ):
            result = getattr(mpg, generator_name)(
                resource,
                None,
                runtime_calc_func=lambda a, b, p: p.runtime,
                cost_calc_func=lambda a, b, p: p.traffic,
                runtime_tolerance=0.1,
                cost_tradeoff=False,
            )
        self.assertIs(result[3], candidates[0])


if __name__ == "__main__":
    unittest.main()

"""Target contracts: independent constraints, graph-derived facts and WS ranking."""

import unittest
from dataclasses import replace
from interstellar import Layer, MappingPoint, loop_enum as le
from voyager_compiler.codegen.transform.tiling.tiler import (
    RuntimeCalculator,
    build_interstellar_tiler,
)
from voyager_compiler.gemmini.hardware import lean_config, interstellar_memory
from voyager_compiler.hardware_config import VOYAGER
from voyager_compiler.gemmini.execution import GemminiCostModel
from voyager_compiler.gemmini.execution import array_work, instruction_rows
from voyager_compiler.gemmini.scheduling import QueueGeometry
from voyager_compiler.targets import get_backend


def point(m=128, n=64, k=256, retain=True):
    blocks = [[1] * 4 for _ in range(le.NUM)]
    parts = [[1] * 4 for _ in range(le.NUM)]
    orders = [[6] * 4 for _ in range(le.NUM)]
    parts[le.IC][0] = parts[le.OC][0] = 16
    for d, v in ((le.OX, m), (le.OC, n // 16), (le.IC, k // 16)):
        blocks[d][1] = v
    blocks[le.OX][3] = 12544 // m
    seq = (le.OX, le.IC, le.OC) if retain else (le.IC, le.OC, le.OX)
    for rank, d in enumerate(seq):
        orders[d][1] = rank
    orders[le.OX][3] = 0
    return MappingPoint(orders, blocks, parts)


class MappingPolicyTest(unittest.TestCase):
    def test_batch_candidates_depend_only_on_problem_extent(self):
        from voyager_compiler.codegen.transform.bufferize.plan import (
            batch_candidates,
        )

        self.assertEqual(batch_candidates(4), (1, 2, 4))
        self.assertEqual(batch_candidates(7), (1, 7))
        self.assertEqual(batch_candidates(4, split=False), (4,))
        with self.assertRaises(ValueError):
            batch_candidates(0)

    def test_pointwise_budget_scales_with_hardware_and_explicit_policy(self):
        from voyager_compiler.gemmini.mapping import (
            GemminiMappingPolicy,
            GemminiTuning,
        )

        config = lean_config()
        policy = GemminiMappingPolicy(config)
        original = policy.nonmatrix_slot_size(None, 2)
        self.assertEqual(
            original, 4096
        )  # preserved baseline, no literal in search
        changed = replace(
            config,
            memory=replace(
                config.memory,
                levels=tuple(
                    replace(
                        level,
                        instances=tuple(
                            (
                                replace(
                                    mem,
                                    size=replace(
                                        mem.size, value=mem.size.value * 2
                                    ),
                                )
                                if mem.name == "accumulator"
                                else mem
                            )
                            for mem in level.instances
                        ),
                    )
                    for level in config.memory.levels
                ),
            ),
        )
        self.assertEqual(
            GemminiMappingPolicy(changed).nonmatrix_slot_size(None, 2),
            original * 2,
        )
        tuned = GemminiMappingPolicy(
            config, GemminiTuning(pointwise_wide_working_sets=1)
        )
        self.assertEqual(tuned.nonmatrix_slot_size(None, 2), original * 2)
        self.assertIs(
            build_interstellar_tiler(
                config, mapping_policy=tuned
            ).mapping_policy,
            tuned,
        )
        with self.assertRaisesRegex(ValueError, "hardware differs"):
            build_interstellar_tiler(
                config, mapping_policy=GemminiMappingPolicy(changed)
            )

    def test_invalid_buffer_plan_is_rejected_before_building_loops(self):
        from voyager_compiler.codegen.transform.bufferize.plan import (
            KernelBufferPlan,
        )

        for values in (
            dict(batch_tiles=0),
            dict(batch_tiles=1.5),
            dict(accumulator_copies=-1),
        ):
            with self.assertRaises(ValueError):
                KernelBufferPlan(**values)

    def test_independent_target_restrictions_and_objective(self):
        v = get_backend("voyager").mapping_policy(VOYAGER).schedule()
        g = get_backend("gemmini").mapping_policy(lean_config()).schedule()
        self.assertEqual(v["schedule_hint"]["IC"]["level2"]["order"], 0)
        self.assertNotIn("level2", g["schedule_hint"]["IC"])
        self.assertTrue(
            build_interstellar_tiler(VOYAGER).interstellar_cost_tradeoff
        )
        self.assertFalse(
            build_interstellar_tiler(lean_config()).interstellar_cost_tradeoff
        )

    def test_storage_and_queue_views_derive_from_graph(self):
        c = lean_config()
        levels = tuple(
            replace(
                l,
                instances=tuple(
                    (
                        replace(m, size=replace(m.size, value=m.size.value * 2))
                        if m.name in ("accumulator", "reservation_execute")
                        else m
                    )
                    for m in l.instances
                ),
            )
            for l in c.memory.levels
        )
        changed = replace(c, memory=replace(c.memory, levels=levels))
        self.assertEqual(changed.accum_buffer_size, c.accum_buffer_size * 2)
        self.assertEqual(
            interstellar_memory(changed)["buf_capacity_list"][2][1], 131072
        )
        self.assertEqual(
            QueueGeometry.from_hardware(changed).reservation_execute, 32
        )
        # Analytical mutation must not silently claim the fixed RTL has changed.
        with self.assertRaises(ValueError):
            get_backend("gemmini").validate(changed)

    def test_weight_reuse_depends_on_order_not_dram_bytes(self):
        a = array_work(point(), 1, 16, 4)
        b = array_work(point(retain=False), 1, 16, 4)
        self.assertEqual(a.microtiles, b.microtiles)
        self.assertEqual(a.retained_weights, 7 * a.fresh_preloads)
        self.assertEqual(b.retained_weights, 0)
        shared = RuntimeCalculator.__new__(RuntimeCalculator)
        shared.batch = shared.weight_batch = 1
        model = GemminiCostModel(shared)
        layer = Layer(256, 64, 12544, 1, 1, 1)
        faster = model.calculate_runtime(None, layer, point())
        traffic = dict(model.dram_bytes)
        slower = model.calculate_runtime(None, layer, point(retain=False))
        self.assertEqual(traffic, model.dram_bytes)
        # Full-row preload service pipelines with compute. Retention alone
        # must not create an artificial throughput bonus with separate banks.
        self.assertEqual(faster, slower)

    def test_original_gemm2_shape_ranks_above_previous_shape(self):
        shared = RuntimeCalculator.__new__(RuntimeCalculator)
        shared.batch = shared.weight_batch = 1
        model = GemminiCostModel(shared)
        layer = Layer(256, 64, 12544, 1, 1, 1)
        self.assertLess(
            model.calculate_runtime(None, layer, point(128)),
            model.calculate_runtime(None, layer, point(16)),
        )

    def test_fresh_preload_prevents_short_row_execution(self):
        self.assertEqual(instruction_rows(7, True, 16, 4), 16)
        self.assertEqual(instruction_rows(7, False, 16, 4), 7)
        self.assertEqual(instruction_rows(1, False, 16, 4), 4)


if __name__ == "__main__":
    unittest.main()

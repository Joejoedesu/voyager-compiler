"""Dependency-safe submission and Gemmini tile-cost tests."""

import unittest
from collections import Counter

from interstellar import Layer, MappingPoint
from interstellar import loop_enum as le
from voyager_compiler.codegen.transform.tiling.tiler import RuntimeCalculator
from voyager_compiler.gemmini.execution import GemminiCostModel
from voyager_compiler.gemmini.isa import ACC, command, ex, ld, st, tile
from voyager_compiler.gemmini.scheduling import (
    AsyncSchedule,
    funct,
)


def signature(commands):
    return Counter(tuple(sorted(c.items())) for c in commands)


class AsyncLoweringTest(unittest.TestCase):
    def make_schedule(self):
        commands = [dict(type="defaults"), ex()]
        scheduler = AsyncSchedule(commands)
        with scheduler.task("input0"):
            commands += [ld(16), command(2, 0x80000000, tile(0, 16, 16))]
            scheduler.signal("input", 1)
        with scheduler.task("compute0"):
            scheduler.wait("input")
            for i in range(32):
                commands += [
                    command(6, tile(8192, 16, 16), tile(ACC, 16, 16)),
                    command(4, tile(0, 16, 16), tile(0xFFFFFFFF, 16, 16)),
                ]
            scheduler.signal("done", 1)
        return commands, scheduler

    def test_submission_tuning_changes_interleaving_not_instructions(self):
        from voyager_compiler.gemmini.scheduling import SubmissionPolicy

        results = []
        for quantum in (8, 32):
            commands, scheduler = self.make_schedule()
            scheduler.policy = SubmissionPolicy(quantum, 1, 1)
            prefetch = command(2, 0x80001000, tile(4096, 16, 16))
            with scheduler.task("prefetch"):
                commands += [ld(16), prefetch]
            ordered, stats = scheduler.reorder()
            self.assertEqual(signature(commands), signature(ordered))
            self.assertEqual(stats["execute_quantum"], quantum)
            results.append(ordered.index(prefetch))
        self.assertLess(results[0], results[1])
        with self.assertRaises(ValueError):
            SubmissionPolicy(3, 1, 1)

    def test_independent_prefetch_passes_compute_without_changing_instructions(
        self,
    ):
        commands, scheduler = self.make_schedule()
        prefetch = command(2, 0x80001000, tile(4096, 16, 16))
        with scheduler.task("input1"):
            commands += [ld(16), prefetch]
        original = commands.copy()
        ordered, _ = scheduler.reorder()
        self.assertEqual(signature(original), signature(ordered))
        pos = ordered.index(prefetch)
        self.assertTrue(any(funct(c) == 4 for c in ordered[1:pos]))
        self.assertTrue(any(funct(c) == 4 for c in ordered[pos + 1 :]))

    def test_large_prefetch_is_split_and_keeps_its_configuration(self):
        commands, scheduler = self.make_schedule()
        # More execute work than two quanta, and a DMA burst exceeding LD RS.
        with scheduler.task("input1"):
            for i in range(12):
                commands += [
                    ld(16 + i),
                    command(
                        2, 0x80001000 + i * 256, tile(4096 + i * 16, 16, 16)
                    ),
                ]
        original = commands.copy()
        ordered, _ = scheduler.reorder()
        self.assertEqual(signature(original), signature(ordered))
        loads = [i for i, c in enumerate(ordered[1:], 1) if funct(c) == 2][1:]
        self.assertTrue(
            any(funct(c) in (4, 5) for c in ordered[loads[3] + 1 : loads[4]])
        )
        for i, pos in enumerate(loads):
            self.assertEqual(ordered[pos - 1], ld(16 + i))

    def test_overwrite_cannot_pass_last_reader(self):
        commands, scheduler = self.make_schedule()
        overwrite = command(2, 0x80001000, tile(0, 16, 16))
        with scheduler.task("reuse_input0"):
            commands += [ld(16), overwrite]
        ordered, _ = scheduler.reorder()
        last_compute = max(
            i
            for i, c in enumerate(ordered)
            if c.get("instruction") and funct(c) == 4
        )
        self.assertGreater(ordered.index(overwrite), last_compute)

    def test_explicit_wait_blocks_even_disjoint_prefetch(self):
        commands, scheduler = self.make_schedule()
        scheduler.wait("done")
        prefetch = command(2, 0x80001000, tile(4096, 16, 16))
        with scheduler.task("input1"):
            commands += [ld(16), prefetch]
        ordered, _ = scheduler.reorder()
        last_compute = max(
            i
            for i, c in enumerate(ordered)
            if c.get("instruction") and funct(c) == 4
        )
        self.assertGreater(ordered.index(prefetch), last_compute)

    def test_output_cannot_store_before_producer_is_submitted(self):
        commands, scheduler = self.make_schedule()
        store = command(3, 0x80002000, tile(ACC, 16, 16))
        with scheduler.task("output0"):
            commands += [st(16), store]
        ordered, _ = scheduler.reorder()
        self.assertEqual(ordered[-1], store)

    def test_partial_alias_overlap_is_a_dependency(self):
        commands, scheduler = self.make_schedule()
        overwrite = command(2, 0x80001000, tile(8, 16, 16))
        with scheduler.task("partially_overlapping_view"):
            commands += [ld(16), overwrite]
        ordered, _ = scheduler.reorder()
        self.assertEqual(ordered[-1], overwrite)


class GemminiCostTest(unittest.TestCase):
    @staticmethod
    def example():
        shared = RuntimeCalculator.__new__(RuntimeCalculator)
        shared.batch = shared.weight_batch = 1
        shared.dram_bandwidth = 16
        shared.dram_access_latency_cycles = 8
        blocks = [[1, 1, 1, 1] for _ in range(le.NUM)]
        parts = [[1, 1, 1, 1] for _ in range(le.NUM)]
        orders = [[6, 6, 6, 6] for _ in range(le.NUM)]
        parts[le.IC][0] = parts[le.OC][0] = 16
        blocks[le.IC][1] = 4
        blocks[le.OX][1], blocks[le.OX][3] = 32, 128
        orders[le.OX][3] = 0
        return (
            shared,
            Layer(64, 16, 1, 4096, 1, 1),
            MappingPoint(orders, blocks, parts),
        )

    def test_streamed_input_uses_pingpong_and_resident_weight_uses_one_slot(
        self,
    ):
        shared, layer, mapping = self.example()
        model = GemminiCostModel(shared)
        model.calculate_runtime(None, layer, mapping)
        self.assertEqual(model.buffer_slots, dict(input=2, weight=1))

    def test_dma_latency_remains_in_runtime_estimate(self):
        shared, layer, mapping = self.example()
        model = GemminiCostModel(shared)
        fast = model.calculate_runtime(None, layer, mapping)
        from dataclasses import replace

        hw = model.config
        model.config = replace(
            hw,
            connections=tuple(
                replace(c, latency_ns=1024) if c.name == "dram_dma" else c
                for c in hw.connections
            ),
        )
        slow = model.calculate_runtime(None, layer, mapping)
        self.assertGreater(slow, fast)


if __name__ == "__main__":
    unittest.main()

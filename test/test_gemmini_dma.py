"""DMA calibration and resource accounting for the target search model."""

import unittest
import tempfile
from pathlib import Path
from dataclasses import replace

from voyager_compiler.codegen.transform.tiling.cost import _sweep_cycles
from voyager_compiler.codegen.transform.tiling.transfers import (
    concurrent_dma_cycles,
    dma_service,
    dma_sweep_cycles,
)
from voyager_compiler.gemmini.hardware import lean_config
from voyager_compiler.gemmini.constraints import accumulator_slot_bytes
from voyager_compiler.gemmini.collateral import Converter, Ref, SRAM
from voyager_compiler.hardware_config import Bandwidth, BandwidthUnit


class GemminiDMATest(unittest.TestCase):
    def test_converter_realizes_disjoint_banks_without_retiling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            converter = Converter(root, [], root / "replay")
            converter.matrix_accumulators.add("output")
            first = Ref("output", SRAM, 0, (96, 64), (64, 1), "int8", slot=0)
            second = replace(first, slot=1)
            self.assertEqual(converter.accumulator(first), 0)
            self.assertEqual(converter.accumulator(second), 512)
            self.assertEqual(converter.accumulator(first), 0)
            self.assertEqual(first.shape, (96, 64))
            self.assertEqual(second.shape, first.shape)

    def test_accumulator_overlap_plan_separates_read_ports(self):
        config = lean_config()
        memory = config.memory_instance("accumulator")
        self.assertEqual(
            accumulator_slot_bytes(config, 128 * memory.row_bytes),
            memory.size.value // memory.banks,
        )
        # An endpoint with two independent read ports need not reserve a bank
        # for each buffer solely to overlap accumulation and writeback.
        levels = tuple(
            replace(
                level,
                instances=tuple(
                    (
                        replace(
                            mem,
                            ports=tuple(
                                (
                                    replace(port, count=2)
                                    if port.name == "read"
                                    else port
                                )
                                for port in mem.ports
                            ),
                        )
                        if mem.name == "accumulator"
                        else mem
                    )
                    for mem in level.instances
                ),
            )
            for level in config.memory.levels
        )
        changed = replace(config, memory=replace(config.memory, levels=levels))
        self.assertEqual(
            accumulator_slot_bytes(changed, 128 * memory.row_bytes),
            128 * memory.row_bytes,
        )

    def test_store_service_matches_isolated_size_sweep(self):
        # VCS: 256 commands, four/eight/sixteen rows of sixteen INT8 values.
        # Check measured controller service, excluding stream fill/drain.
        for rows, measured in ((4, 1639), (8, 3275), (16, 6553)):
            work = dma_service(
                lean_config(), "acc_to_dram", 256 * rows * 16, 256
            )
            self.assertLess(abs(work.endpoint_cycles - measured), 3)
            self.assertEqual(work.payload_cycles, 256 * rows)

    def test_command_overhead_is_distinct_from_stream_latency(self):
        config = lean_config()
        config = replace(
            config,
            connections=tuple(
                replace(edge, startup_ns=3) if edge.name == "dram_dma" else edge
                for edge in config.connections
            ),
        )
        one = dma_service(config, "dram_dma", 1024, 1)
        four = dma_service(config, "dram_dma", 1024, 4)
        self.assertEqual(four.cycles - one.cycles, 9)
        self.assertEqual(four.latency_cycles, one.latency_cycles)

    def test_store_endpoint_does_not_reserve_idle_bus_cycles(self):
        config = lean_config()
        load = dma_service(config, "dram_dma", 256, 1)
        store = dma_service(config, "acc_to_dram", 256, 1)
        both = concurrent_dma_cycles([load], store)
        self.assertGreaterEqual(
            both, load.payload_cycles + store.payload_cycles
        )
        self.assertGreaterEqual(both, store.cycles)
        self.assertLess(both, load.cycles + store.cycles)

    def test_without_endpoint_limit_matches_shared_sweep(self):
        config = lean_config()
        config = replace(
            config,
            connections=tuple(
                replace(edge, service_bandwidth=None)
                for edge in config.connections
            ),
        )
        a = dma_service(config, "dram_dma", 1024, 4)
        b = dma_service(config, "dram_dma", 512, 2)
        c = dma_service(config, "acc_to_dram", 2048, 8)
        for steps in (1, 2, 8):
            for repeat in (1, steps):
                for compute in (20, 200, 400):
                    actual = dma_sweep_cycles(
                        [(a, steps), (b, repeat)], c, steps, steps, compute
                    )
                    expected = _sweep_cycles(
                        [
                            (c.cycles, steps),
                            (a.cycles, steps),
                            (b.cycles, repeat),
                        ],
                        steps,
                        compute,
                    )
                    self.assertEqual(actual, expected)

    def test_service_limit_is_independent_of_shared_bandwidth(self):
        config = lean_config()
        changed = replace(
            config,
            connections=tuple(
                (
                    replace(
                        edge,
                        service_bandwidth=Bandwidth(
                            8, BandwidthUnit.BYTES_PER_CYCLE
                        ),
                    )
                    if edge.name == "acc_to_dram"
                    else edge
                )
                for edge in config.connections
            ),
        )
        self.assertEqual(changed.connection_bytes_per_cycle("acc_to_dram"), 16)
        self.assertGreater(
            dma_service(changed, "acc_to_dram", 256, 1).cycles,
            dma_service(config, "acc_to_dram", 256, 1).cycles,
        )
        with self.assertRaises(TypeError):
            replace(config.connection("acc_to_dram"), service_bandwidth=8)


class TransferIRTest(unittest.TestCase):
    def test_rectangular_command_limits(self):
        from voyager_compiler.hardware_config import TransferGeometry

        geometry = TransferGeometry(16, 64)
        self.assertEqual(geometry.command_count(33, 80), 6)
        self.assertEqual(geometry.command_count(0, 64), 0)
        for args in ((0, 64), (16, -1), (1.5, 16), (True, 16)):
            with self.assertRaises(ValueError):
                TransferGeometry(*args)

    def test_unrelated_dma_resources_can_overlap(self):
        config = lean_config()
        load_edge = config.connection("dram_dma")
        independent = replace(
            load_edge,
            name="independent_dma",
            service_resource="controller_store",
        )
        config = replace(
            config, connections=config.connections + (independent,)
        )
        first = dma_service(config, "dram_dma", 1024, 1)
        second = dma_service(config, "independent_dma", 1024, 1)
        self.assertEqual(concurrent_dma_cycles([first, second]), first.cycles)
        # A different link with the SAME controller still serializes service.
        shared_controller = replace(
            independent, service_resource="controller_load"
        )
        config = replace(
            config, connections=config.connections[:-1] + (shared_controller,)
        )
        second = dma_service(config, "independent_dma", 1024, 1)
        self.assertEqual(
            concurrent_dma_cycles([first, second]), first.cycles + second.cycles
        )

    def test_latency_inheritance_and_service_reference_validation(self):
        config = lean_config()
        inherited = dma_service(config, "acc_to_dram", 256, 1)
        self.assertEqual(inherited.latency_cycles, 8)
        config = replace(
            config,
            connections=tuple(
                replace(e, latency_ns=3) if e.name == "acc_to_dram" else e
                for e in config.connections
            ),
        )
        self.assertEqual(
            dma_service(config, "acc_to_dram", 256, 1).latency_cycles, 3
        )
        with self.assertRaisesRegex(ValueError, "service resource"):
            replace(
                config,
                connections=tuple(
                    (
                        replace(e, service_resource="missing")
                        if e.name == "acc_to_dram"
                        else e
                    )
                    for e in config.connections
                ),
            )

    def test_packed_bank_policy_is_shared_by_estimate_and_converter(self):
        from voyager_compiler.gemmini.mapping import (
            GemminiMappingPolicy,
            GemminiTuning,
        )

        config = lean_config()
        policy = GemminiMappingPolicy(
            config, GemminiTuning(separate_accumulator_banks=False)
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            converter = Converter(root, [], root / "replay", policy=policy)
            converter.matrix_accumulators.add("output")
            first = Ref("output", SRAM, 0, (96, 64), (64, 1), "int8", slot=0)
            self.assertEqual(converter.accumulator(first), 0)
            self.assertEqual(converter.accumulator(replace(first, slot=1)), 384)
        self.assertEqual(
            accumulator_slot_bytes(config, 96 * 64 * 4, separate_banks=False),
            384 * 64,
        )


if __name__ == "__main__":
    unittest.main()

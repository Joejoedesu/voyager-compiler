"""Fixed Voyager configuration, resource accounting and ablation boundaries."""

import importlib.util
import unittest
from dataclasses import replace

from voyager_compiler.hardware_config import VOYAGER, AcceleratorConfig
from voyager_compiler.voyager_adapter import (
    fusion_patterns,
    interstellar_memory,
)


class VoyagerConfigTest(unittest.TestCase):
    def test_sizes_and_bandwidth_still_drive_resource_accounting(self):
        config = AcceleratorConfig(
            pe_array_size=(8, 48),
            scratchpad_size=1 << 20,
            num_banks=16,
            bank_width=24,
            scratchpad_offset=65536,
            frequency=2,
        )
        self.assertEqual(config.usable_banks, 15)
        self.assertEqual(config.usable_scratchpad_size, 983040)
        self.assertEqual(config.bytes_per_cycle, 32)
        self.assertEqual(config.access_latency_cycles, 200)
        self.assertEqual(config.sram_bandwidth_bits(4), 192)
        resource = interstellar_memory(config)
        self.assertEqual(
            resource["buf_capacity_list"],
            [[1, 1, 1], [8192, 49152, 49152], [983040], [16 * 1024**3]],
        )
        self.assertEqual(resource["para_count_list"], [384, 1, 1, 1])
        self.assertEqual(
            resource["memory_partitions"],
            [[0, 1, 2], [0, 1, 2], [0, 0, 0], [0, 0, 0]],
        )

    def test_new_backends_and_graph_overrides_are_not_configurable(self):
        for override in (
            dict(backend="other"),
            dict(name="other"),
            dict(memory=VOYAGER.memory),
            dict(connections=VOYAGER.connections),
            dict(computation_units=VOYAGER.computation_units),
            dict(isa_pipelines=()),
            dict(provenance=()),
        ):
            with self.subTest(override=override), self.assertRaises(TypeError):
                AcceleratorConfig(**override)
        with self.assertRaises(TypeError):
            replace(VOYAGER, backend="other")
        self.assertIsNone(importlib.util.find_spec("voyager_compiler.targets"))
        self.assertIsNone(
            importlib.util.find_spec("voyager_compiler.quantization.recipes")
        )

    def test_fusion_remains_the_existing_fixed_voyager_pipeline(self):
        narrow = AcceleratorConfig(pe_array_size=(4, 8))
        wide = AcceleratorConfig(pe_array_size=(32, 64))

        def signatures(config):
            return [
                [matcher.targets for matcher in stage]
                for stage in fusion_patterns(config)
            ]

        self.assertEqual(signatures(narrow), signatures(wide))
        self.assertEqual(len(signatures(narrow)), 3)

    def test_bad_reserved_memory_still_fails(self):
        for offset in (-1, 1, 1 << 20):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                AcceleratorConfig(
                    scratchpad_size=1 << 20,
                    num_banks=16,
                    scratchpad_offset=offset,
                )


if __name__ == "__main__":
    unittest.main()

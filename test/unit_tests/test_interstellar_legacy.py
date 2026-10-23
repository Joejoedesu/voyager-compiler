# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

from scheduler import interstellar
from scheduler.dispatch import (
    INTERSTELLAR_SCHEDULER,
    schedule_operation,
)
from scheduler.interstellar.voyager import (
    InterstellarHardwareConfig,
    RuntimeCalculator,
    RuntimeCalculatorWithDRAM,
    build_architecture_and_schedule,
    build_architecture_and_schedule_from_hardware,
    build_architecture_and_schedule_with_dram,
    run_interstellar_scheduler,
)
from scheduler.types import (
    GemmWorkload,
    OperationOperand,
    OperationWorkload,
)
from voyager_compiler.codegen.hardware import HardwareConfig, TensorRole
from voyager_compiler.codegen.targets import (
    DEFAULT_COMPILATION_TARGET,
)

_NON_DEFAULT_TARGET = "unit_test_target"


class InterstellarRuntimeTest(unittest.TestCase):
    @staticmethod
    def _mapping(num_levels, ic_partition=16):
        blockings = [
            tuple(1 for _ in range(num_levels))
            for _ in range(interstellar.le.NUM)
        ]
        blockings[interstellar.le.OX] = (
            1,
            4,
            2,
            *([1] * (num_levels - 3)),
        )
        partitionings = [
            tuple(1 for _ in range(num_levels))
            for _ in range(interstellar.le.NUM)
        ]
        partitionings[interstellar.le.IC] = (
            ic_partition,
            *([1] * (num_levels - 1)),
        )
        return SimpleNamespace(
            loop_orders=[
                tuple(0 for _ in range(num_levels))
                for _ in range(interstellar.le.NUM)
            ],
            loop_blockings=blockings,
            loop_partitionings=partitionings,
        )

    @staticmethod
    def _runtime_cases():
        return (
            ("low_unbuffered", 8, False, False, 160),
            ("low_buffered", 8, True, False, 164),
            ("high_unbuffered", 16, False, False, 164),
            ("high_buffered", 16, True, False, 168),
            ("wide_unbuffered", 32, False, False, 164),
            ("wide_buffered", 32, True, False, 176),
            ("vector_high_unbuffered", 8, False, True, 164),
            ("vector_high_buffered", 8, True, True, 164),
        )

    def test_runtime_uses_original_accumulator_pipeline(self):
        mapping = self._mapping(3)

        for (
            name,
            output_bits,
            double_buffered,
            vector_high,
            expected,
        ) in self._runtime_cases():
            with self.subTest(name=name):
                calculator = RuntimeCalculator(
                    input_dtype_width=8,
                    output_dtype_width=output_bits,
                    double_buffered_accum_buffer=double_buffered,
                    has_high_precision_vector_input=vector_high,
                )

                self.assertEqual(
                    calculator.calculate_runtime(None, None, mapping),
                    expected,
                )

    def test_dram_runtime_uses_original_accumulator_pipeline(self):
        mapping = self._mapping(4)

        for (
            name,
            output_bits,
            double_buffered,
            vector_high,
            expected,
        ) in self._runtime_cases():
            with self.subTest(name=name):
                calculator = RuntimeCalculatorWithDRAM(
                    input_dtype_width=8,
                    weight_dtype_width=8,
                    output_dtype_width=output_bits,
                    double_buffered_accum_buffer=double_buffered,
                    dram_bandwidth=float("inf"),
                    has_high_precision_vector_input=vector_high,
                )

                self.assertEqual(
                    calculator.calculate_runtime(None, None, mapping),
                    expected,
                )

    def test_wide_output_sets_double_buffered_steady_state(self):
        calculators = (
            (
                "generic",
                RuntimeCalculator(
                    input_dtype_width=8,
                    output_dtype_width=32,
                    double_buffered_accum_buffer=True,
                ),
                self._mapping(3, ic_partition=1),
            ),
            (
                "dram",
                RuntimeCalculatorWithDRAM(
                    input_dtype_width=8,
                    weight_dtype_width=8,
                    output_dtype_width=32,
                    double_buffered_accum_buffer=True,
                    dram_bandwidth=float("inf"),
                ),
                self._mapping(4, ic_partition=1),
            ),
        )

        for name, calculator, mapping in calculators:
            with self.subTest(name=name):
                self.assertEqual(
                    calculator.calculate_runtime(None, None, mapping),
                    52,
                )

    def test_typed_dram_dispatch_uses_matrix_rhs_width(self):
        hardware = HardwareConfig.from_legacy(
            unroll_dims=(16, 16),
            cache_size=4096,
            input_buffer_size=32,
            weight_buffer_size=32,
            accum_buffer_size=32,
            dram_size=16384,
        )
        workload = OperationWorkload(
            operands=(
                OperationOperand("input", (1, 16, 1, 8), 8),
                OperationOperand("other", (16, 16), 4),
            ),
            output=OperationOperand("output", (1, 16, 1, 8), 16),
            gemm=GemmWorkload(16, 16, 8, 1, 1, 1),
        )
        native_mapping = self._mapping(4)
        provider_cache = {}

        with (
            patch(
                "scheduler.interstellar.voyager." "_build_cached_architecture",
                return_value=(object(), object()),
            ) as build_architecture,
            patch(
                "scheduler.interstellar.voyager." "RuntimeCalculatorWithDRAM"
            ) as runtime_type,
            patch.object(
                interstellar.optimizer,
                "opt_optimizer",
                return_value=(None, 1, native_mapping, None),
            ),
            patch.object(
                interstellar.cost_model,
                "get_cost",
                return_value=(
                    None,
                    None,
                    ((0, 0, 0),) * 4,
                ),
            ),
        ):
            run_interstellar_scheduler(
                workload,
                hardware,
                provider_cache,
                None,
            )

        runtime_type.assert_called_once_with(
            8,
            4,
            16,
            False,
            hardware_config=hardware,
            has_high_precision_vector_input=False,
            has_sparse_op=False,
        )
        build_architecture.assert_called_once_with(
            hardware,
            provider_cache,
            8,
            4,
            16,
        )

    def test_typed_dram_bandwidth_uses_weight_width(self):
        hardware = HardwareConfig.from_legacy(
            unroll_dims=(16, 16),
            cache_size=4096,
            input_buffer_size=32,
            weight_buffer_size=32,
            accum_buffer_size=32,
            dram_size=16384,
        )
        mapping = self._mapping(4, ic_partition=1)
        narrow_weights = RuntimeCalculatorWithDRAM(
            input_dtype_width=8,
            weight_dtype_width=4,
            output_dtype_width=8,
            double_buffered_accum_buffer=False,
            hardware_config=hardware,
        )
        wide_weights = RuntimeCalculatorWithDRAM(
            input_dtype_width=8,
            weight_dtype_width=8,
            output_dtype_width=8,
            double_buffered_accum_buffer=False,
            hardware_config=hardware,
        )

        narrow_runtime = narrow_weights.calculate_runtime(
            None,
            None,
            mapping,
        )
        wide_runtime = wide_weights.calculate_runtime(
            None,
            None,
            mapping,
        )
        weight_route = hardware.hierarchy_bandwidth(
            hardware.external_memory.name,
            hardware.global_memory.name,
            TensorRole.WEIGHT,
        )

        self.assertAlmostEqual(
            wide_runtime - narrow_runtime,
            weight_route.cycles_for(4),
        )

    def test_default_provider_runs_interstellar(self):
        hardware = HardwareConfig.from_legacy(
            unroll_dims=(16, 16),
            cache_size=4096,
            input_buffer_size=32,
            weight_buffer_size=32,
            accum_buffer_size=32,
        )
        workload = OperationWorkload(
            operands=(
                OperationOperand("input", (1, 16, 2, 2), 8),
                OperationOperand("weight", (16, 16, 1, 1), 8),
            ),
            output=OperationOperand("output", (1, 16, 2, 2), 8),
            gemm=GemmWorkload(16, 16, 2, 2, 1, 1),
        )

        result = schedule_operation(workload, hardware, {})

        self.assertEqual(result.backend, INTERSTELLAR_SCHEDULER)
        self.assertEqual(result.mapping.num_levels, 3)
        self.assertGreater(result.runtime, 0)

    def test_typed_dram_provider_runs_mixed_width(self):
        hardware = HardwareConfig.from_legacy(
            unroll_dims=(16, 16),
            cache_size=4096,
            input_buffer_size=32,
            weight_buffer_size=32,
            accum_buffer_size=32,
            dram_size=16384,
        )
        workload = OperationWorkload(
            operands=(
                OperationOperand("input", (1, 16, 2, 2), 8),
                OperationOperand("weight", (16, 16, 1, 1), 4),
            ),
            output=OperationOperand("output", (1, 16, 2, 2), 16),
            gemm=GemmWorkload(16, 16, 2, 2, 1, 1),
        )
        provider_cache = {}

        result = schedule_operation(
            workload,
            hardware,
            provider_cache,
        )

        self.assertEqual(result.backend, INTERSTELLAR_SCHEDULER)
        self.assertEqual(result.mapping.num_levels, 4)
        self.assertGreater(result.runtime, 0)
        self.assertIn((8, 4, 16), provider_cache)


class LegacyInterstellarBuilderTest(unittest.TestCase):
    def test_weight_width_controls_physical_weight_capacity(self):
        hardware = HardwareConfig.from_legacy(
            unroll_dims=(16, 16),
            cache_size=4096,
            input_buffer_size=32,
            weight_buffer_size=32,
            accum_buffer_size=32,
        )
        local_memory = replace(
            hardware.local_memory,
            instances=tuple(
                (
                    replace(memory, logical_capacity=None)
                    if memory.name == "weight_buffer"
                    else memory
                )
                for memory in hardware.local_memory.instances
            ),
        )
        hardware = replace(hardware, local_memory=local_memory)

        int4_weights, _ = build_architecture_and_schedule_from_hardware(
            hardware,
            input_dtype_width=8,
            output_dtype_width=8,
            weight_dtype_width=4,
        )
        int8_weights, _ = build_architecture_and_schedule_from_hardware(
            hardware,
            input_dtype_width=8,
            output_dtype_width=8,
            weight_dtype_width=8,
        )

        self.assertEqual(
            int4_weights.bufs[1].capacity[2],
            2 * int8_weights.bufs[1].capacity[2],
        )

    def test_generic_builder_preserves_scalar_capacities(self):
        architecture, _ = build_architecture_and_schedule(
            ic_dim=16,
            oc_dim=8,
            l2_cache_size=1024,
            input_buffer_size=32,
            weight_buffer_size=64,
            accum_buffer_size=128,
            target=DEFAULT_COMPILATION_TARGET,
            input_dtype_width=4,
        )

        self.assertEqual(architecture.num_levels, 3)
        self.assertEqual(architecture.bufs[1].capacity, [512, 1024, 512])
        self.assertEqual(architecture.bufs[2].capacity, [1024])
        self.assertEqual(architecture.para_count_list, [128, 1, 1])

    def test_generic_dram_builder_preserves_scalar_capacity(self):
        architecture, _ = build_architecture_and_schedule_with_dram(
            ic_dim=16,
            oc_dim=8,
            l2_cache_size=1024,
            input_buffer_size=32,
            weight_buffer_size=64,
            accum_buffer_size=128,
            dram_size=4096,
            target=DEFAULT_COMPILATION_TARGET,
            input_dtype_width=4,
        )

        self.assertEqual(architecture.num_levels, 4)
        self.assertEqual(architecture.bufs[2].capacity, [1024])
        self.assertEqual(architecture.bufs[3].capacity, [4096])

    def test_target_specific_scalar_builder_is_rejected(self):
        config = InterstellarHardwareConfig(
            ic_dim=16,
            oc_dim=8,
            l2_cache_size=1024,
            input_buffer_size=32,
            weight_buffer_size=64,
            accum_buffer_size=128,
            target=_NON_DEFAULT_TARGET,
        )

        with self.assertRaisesRegex(
            ValueError,
            "support only target 'generic'.*typed HardwareConfig",
        ):
            config.build(input_dtype_width=8)

    def test_target_specific_dram_builder_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "support only target 'generic'.*typed HardwareConfig",
        ):
            build_architecture_and_schedule_with_dram(
                ic_dim=16,
                oc_dim=8,
                l2_cache_size=1024,
                input_buffer_size=32,
                weight_buffer_size=64,
                accum_buffer_size=128,
                dram_size=4096,
                target=_NON_DEFAULT_TARGET,
            )


if __name__ == "__main__":
    unittest.main()

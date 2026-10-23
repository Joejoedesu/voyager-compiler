# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import operator
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from google.protobuf import text_format
from torch.fx import Graph, GraphModule

from scheduler.types import LocalScheduleResult, Loop, MappingPoint
from voyager_compiler.codegen.banking import (
    _get_scope,
    get_banking_strategies_for_op,
)
from voyager_compiler.codegen.hardware import (
    CollectorConfig,
    HardwareConfig,
    MemoryInstance,
    MemoryLevelConfig,
    PEConfig,
    TensorRole,
)
from voyager_compiler.codegen.mapping import (
    find_sequential_nodes_,
    gen_code,
    get_allocation_role,
    get_node_to_key_map,
    run_memory_mapping,
)
from voyager_compiler.codegen.mapping_utils import (
    TARGET_CODEGEN_KWARGS,
    bind_local_schedule,
)
from voyager_compiler.codegen.memory import (
    END_TO_END_SCRATCHPAD_MAPPING,
    PER_OPERATION_SCRATCHPAD_MAPPING,
    MemoryAllocator,
    ScratchpadAllocator,
    Segment,
)
from voyager_compiler.codegen.operands import require_allocation
from voyager_compiler.codegen.param_pb2 import Memory
from voyager_compiler.codegen.shape_prop import ShapeProp
from voyager_compiler.codegen.storage import ValueRef
from voyager_compiler.codegen.target_backends import (
    create_scratchpad_allocator,
)
from voyager_compiler.pt2e_utils import COMPILER_ONLY


def _global_memory():
    return MemoryLevelConfig(
        name="global",
        instances=(
            MemoryInstance(
                name="activation",
                bank_count=4,
                depth_per_bank=64,
                bits_per_bank_entry=128,
                accepted_accesses=frozenset(
                    {
                        (TensorRole.ACTIVATION, None),
                        (TensorRole.OUTPUT, None),
                        (TensorRole.SCALE, None),
                    }
                ),
                access_cost=100,
                access_granularity_bits=512,
            ),
            MemoryInstance(
                name="weight",
                bank_count=2,
                depth_per_bank=64,
                bits_per_bank_entry=128,
                accepted_accesses=frozenset(
                    {
                        (TensorRole.WEIGHT, None),
                        (TensorRole.BIAS, None),
                    }
                ),
                access_cost=100,
                access_granularity_bits=512,
            ),
        ),
        input_memory="activation",
        output_memory="activation",
        weight_memory="weight",
    )


def _conv_graph():
    root = torch.nn.Module()
    root.register_parameter(
        "kernel",
        torch.nn.Parameter(torch.randn(2, 2, 1, 1)),
    )
    graph = Graph()
    input_node = graph.placeholder("input")
    weight = graph.get_attr("kernel")
    conv = graph.call_function(
        torch.ops.aten.conv2d.default,
        (input_node, weight, None, [1, 1], [0, 0], [1, 1], 1),
    )
    relu = graph.call_function(torch.ops.aten.relu.default, (conv,))
    graph.output(relu)
    model = GraphModule(root, graph)
    ShapeProp(model).propagate(torch.randn(1, 2, 4, 4))
    return input_node, weight, conv, relu


def _relu_graph():
    graph = Graph()
    input_node = graph.placeholder("input")
    first = graph.call_function(torch.ops.aten.relu.default, (input_node,))
    second = graph.call_function(torch.ops.aten.relu.default, (first,))
    graph.output(second)
    model = GraphModule(torch.nn.Module(), graph)
    ShapeProp(model).propagate(torch.randn(1, 2, 4, 4))
    return model, input_node, first, second


def _hardware():
    return HardwareConfig(
        pe=PEConfig(
            input_lanes=2,
            output_lanes=2,
            input_word_bits=16,
            scalar_bits=8,
        ),
        local_memory=MemoryLevelConfig(
            name="local",
            instances=(
                MemoryInstance(
                    name="input",
                    bank_count=1,
                    depth_per_bank=64,
                    bits_per_bank_entry=16,
                    accepted_accesses=frozenset(
                        {(TensorRole.ACTIVATION, None)}
                    ),
                    access_cost=10,
                ),
                MemoryInstance(
                    name="output",
                    bank_count=1,
                    depth_per_bank=64,
                    bits_per_bank_entry=64,
                    accepted_accesses=frozenset(
                        {
                            (TensorRole.OUTPUT, None),
                            (TensorRole.PARTIAL_SUM, None),
                        }
                    ),
                    access_cost=10,
                ),
                MemoryInstance(
                    name="weight",
                    bank_count=2,
                    depth_per_bank=64,
                    bits_per_bank_entry=16,
                    accepted_accesses=frozenset({(TensorRole.WEIGHT, None)}),
                    access_cost=10,
                ),
            ),
            input_memory="input",
            output_memory="output",
            weight_memory="weight",
        ),
        global_memory=_global_memory(),
        output_collector=CollectorConfig(
            name="output",
            depth_words=64,
            lanes_per_word=2,
            accumulator_bits=32,
        ),
    )


def _local_schedule():
    outer_factors = {
        Loop.FX: 1,
        Loop.FY: 1,
        Loop.OX: 4,
        Loop.OY: 4,
        Loop.OC: 2,
        Loop.IC: 2,
        Loop.ON: 1,
    }
    mapping = MappingPoint(
        loop_orders=tuple((int(loop), int(loop)) for loop in Loop),
        loop_blockings=tuple((1, outer_factors[loop]) for loop in Loop),
        loop_partitionings=tuple((1, 1) for _loop in Loop),
    )
    return LocalScheduleResult(
        mapping=mapping,
        backend="test",
        named_factors={"tile_rows": 2},
    )


class GlobalMemoryPlanningTest(unittest.TestCase):
    @staticmethod
    def _generic_test_hardware():
        return HardwareConfig.from_legacy(
            unroll_dims=(4, 4),
            cache_size=4096,
            input_buffer_size=64,
            weight_buffer_size=64,
            accum_buffer_size=64,
            num_banks=4,
            bank_width=16,
        )

    def test_every_executable_operation_reaches_scheduler_dispatch(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        relu = graph.call_function(
            torch.ops.aten.relu.default,
            (input_node,),
        )
        sigmoid = graph.call_function(
            torch.ops.aten.sigmoid.default,
            (relu,),
        )
        graph.output(sigmoid)
        model = GraphModule(torch.nn.Module(), graph)
        inputs = (torch.randn(1, 4, 4, 4),)
        ShapeProp(model).propagate(*inputs)
        for node in model.graph.nodes:
            value = getattr(node, "value", None)
            if isinstance(value, torch.Tensor):
                node.meta["dtype"] = str(value.dtype).split(".")[-1]

        with patch(
            "voyager_compiler.codegen.mapping.schedule_operation",
            return_value=None,
        ) as dispatch:
            run_memory_mapping(
                model,
                MemoryAllocator(1 << 20),
                hardware_config=self._generic_test_hardware(),
            )

        self.assertEqual(dispatch.call_count, 2)
        self.assertEqual(
            tuple(
                call.args[0].output.shape for call in dispatch.call_args_list
            ),
            ((1, 4, 4, 4), (1, 4, 4, 4)),
        )

    def test_multi_output_scheduler_selects_primary_output(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        maximum = graph.call_function(
            torch.ops.aten.max.dim,
            (input_node, 1, False),
        )
        graph.output(maximum)
        model = GraphModule(torch.nn.Module(), graph)
        ShapeProp(model).propagate(torch.randn(1, 4, 4, 4))
        input_node.meta["dtype"] = "float32"
        maximum.meta["dtype"] = ("float32", "int64")
        maximum.meta[TARGET_CODEGEN_KWARGS] = {
            "output_payload": ValueRef(maximum, (0,))
        }

        with patch(
            "voyager_compiler.codegen.mapping.schedule_operation",
            return_value=None,
        ) as dispatch:
            run_memory_mapping(
                model,
                MemoryAllocator(1 << 20),
                hardware_config=self._generic_test_hardware(),
            )

        dispatch.assert_called_once()
        workload = dispatch.call_args.args[0]
        self.assertEqual(workload.output.occurrence, (0,))
        self.assertEqual(workload.output.shape, (1, 4, 4))
        self.assertEqual(workload.output.bits, 32)

    def test_unscheduled_multi_output_does_not_guess_an_output(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        maximum = graph.call_function(
            torch.ops.aten.max.dim,
            (input_node, 1, False),
        )
        graph.output(maximum)
        model = GraphModule(torch.nn.Module(), graph)
        ShapeProp(model).propagate(torch.randn(1, 4, 4, 4))
        input_node.meta["dtype"] = "float32"
        maximum.meta["dtype"] = ("float32", "int64")

        with patch(
            "voyager_compiler.codegen.mapping.schedule_operation",
            return_value=None,
        ) as dispatch:
            run_memory_mapping(
                model,
                MemoryAllocator(1 << 20),
                hardware_config=self._generic_test_hardware(),
            )
        dispatch.assert_not_called()

    def test_hardware_profile_rejects_conflicting_legacy_mapping_dims(self):
        model, *_ = _relu_graph()

        with self.assertRaisesRegex(
            ValueError,
            "conflicts with legacy mapping arguments: unroll_dims",
        ):
            run_memory_mapping(
                model,
                MemoryAllocator(1 << 20),
                unroll_dims=(1, 1),
                hardware_config=self._generic_test_hardware(),
            )

    def test_required_execution_fusion_rejects_suffix_and_partial_chains(self):
        class Matcher:
            def __init__(self, target):
                self.targets = (target,)

            def matches(self, node):
                return node.target in self.targets

        pattern = [
            Matcher(operator.add),
            Matcher(operator.neg),
            Matcher(torch.relu),
        ]

        def candidates(include_first, include_last):
            graph = Graph()
            value = graph.placeholder("value")
            first = (
                graph.call_function(operator.add, (value, value))
                if include_first
                else None
            )
            second = graph.call_function(
                operator.neg,
                (first if first is not None else value,),
            )
            last = (
                graph.call_function(torch.relu, (second,))
                if include_last
                else None
            )
            graph.output(last if last is not None else second)
            nodes = list(graph.nodes)
            by_source = {
                operator.add: [first] if first is not None else [],
                operator.neg: [second],
                torch.relu: [last] if last is not None else [],
            }
            return find_sequential_nodes_(
                pattern,
                {node: index for index, node in enumerate(nodes)},
                by_source,
                require_all_pattern_stages=True,
            )

        self.assertEqual(candidates(False, True), [])
        self.assertEqual(candidates(True, False), [])
        complete = candidates(True, True)
        self.assertEqual(len(complete), 1)
        self.assertEqual(len(complete[0]), 3)

    def test_default_target_keeps_flat_scratchpad_allocator(self):
        self.assertIsInstance(
            create_scratchpad_allocator(_hardware()),
            ScratchpadAllocator,
        )

    def test_shared_allocator_rejects_negative_view_offset(self):
        allocator = MemoryAllocator(64)
        backing = allocator.segments[0]

        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            allocator.create_view(backing, -1, 8)

    def test_quantizer_lookup_tables_are_not_runtime_allocations(self):
        graph = Graph()
        compiler_lookup = graph.get_attr("opaque_table")
        runtime_tensor = graph.get_attr("qmap")
        graph.call_function(
            torch.ops.aten.clone.default,
            (compiler_lookup,),
        )
        graph.call_function(
            torch.ops.aten.clone.default,
            (runtime_tensor,),
        )
        compiler_lookup.value = torch.arange(16)
        runtime_tensor.value = torch.arange(16)
        compiler_lookup.meta[COMPILER_ONLY] = True

        self.assertFalse(require_allocation(compiler_lookup))
        self.assertTrue(require_allocation(runtime_tensor))

    def test_fused_output_serializes_target_codegen_kwargs(self):
        inner_graph = Graph()
        inner_input = inner_graph.placeholder("input")
        relu = inner_graph.call_function(
            torch.ops.aten.relu.default,
            (inner_input,),
        )
        inner_graph.output(relu)
        inner = GraphModule(torch.nn.Module(), inner_graph)

        root = torch.nn.Module()
        root.add_module("fused", inner)
        graph = Graph()
        input_node = graph.placeholder("input")
        fused = graph.call_module("fused", (input_node,))
        graph.output(fused)
        model = GraphModule(root, graph)
        inputs = (torch.randn(1, 32, 2, 2),)
        ShapeProp(model).propagate(*inputs)
        fused.meta[TARGET_CODEGEN_KWARGS] = {
            "concat_idx_0": 1,
            "is_to_concat": True,
        }

        params = gen_code(model, inputs)

        operation = params.ops[0].fused_op.op_list[0]
        self.assertEqual(
            operation.kwargs["concat_idx_0"].int_value,
            1,
        )
        self.assertTrue(operation.kwargs["is_to_concat"].bool_value)
        self.assertNotIn(TARGET_CODEGEN_KWARGS, relu.meta)

    def test_fused_codegen_propagates_physical_scratchpad_override(self):
        inner_graph = Graph()
        inner_input = inner_graph.placeholder("input")
        relu = inner_graph.call_function(
            torch.ops.aten.relu.default,
            (inner_input,),
        )
        inner_graph.output(relu)
        inner = GraphModule(torch.nn.Module(), inner_graph)

        root = torch.nn.Module()
        root.add_module("fused", inner)
        graph = Graph()
        input_node = graph.placeholder("input")
        fused = graph.call_module("fused", (input_node,))
        graph.output(fused)
        model = GraphModule(root, graph)
        inputs = (torch.randn(1, 32, 2, 2),)
        ShapeProp(model).propagate(*inputs)
        inner_input.meta["source_node"] = input_node
        fused.meta["scratchpad_map"] = {
            input_node: Segment(128, 256, 0),
        }
        fused.meta["physical_scratchpad_map"] = {
            input_node: Segment(192, 256, 0),
        }

        params = gen_code(model, inputs)
        operation = params.ops[0].fused_op.op_list[0]
        self.assertEqual(
            operation.kwargs["input"].tensor.scratchpad.address,
            192,
        )

        fused.meta.pop("physical_scratchpad_map")
        params = gen_code(model, inputs)
        operation = params.ops[0].fused_op.op_list[0]
        self.assertEqual(
            operation.kwargs["input"].tensor.scratchpad.address,
            128,
        )

    def test_dump_tiling_uses_schedule_bound_to_fused_output(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        pool = graph.call_function(
            torch.ops.aten.max_pool2d.default,
            (input_node, (2, 2)),
        )
        fused_output = graph.call_function(
            torch.ops.aten.relu.default,
            (pool,),
        )
        graph.output(fused_output)
        model = GraphModule(torch.nn.Module(), graph)
        inputs = (torch.randn(1, 32, 8, 8),)
        ShapeProp(model).propagate(*inputs)
        schedule = _local_schedule()
        bind_local_schedule(fused_output, schedule)

        params = gen_code(model, inputs)
        operation = next(
            operation
            for operation in params.ops
            if operation.HasField("tiling")
        )
        factors = {
            Loop(bound.loop).name: bound.bound
            for bound in operation.tiling.level_tilings[0].loop_bounds
        }
        self.assertEqual(
            factors,
            {
                "FX": 1,
                "FY": 1,
                "OX": 4,
                "OY": 4,
                "OC": 2,
                "IC": 2,
                "ON": 1,
            },
        )
        self.assertEqual(
            {
                name.removeprefix("schedule_factor_"): argument.int_value
                for name, argument in operation.op.kwargs.items()
                if name.startswith("schedule_factor_")
            },
            dict(schedule.named_factors),
        )

    def test_schedule_factor_binding_rejects_conflicting_fact(self):
        node = SimpleNamespace(
            name="pool",
            meta={
                TARGET_CODEGEN_KWARGS: {
                    "schedule_factor_tile_rows": 99,
                }
            },
        )
        schedule = _local_schedule()

        with self.assertRaisesRegex(
            ValueError,
            "conflicting named schedule factor",
        ):
            bind_local_schedule(node, schedule)

    def test_memory_text_preserves_explicit_zero_fields(self):
        memory = Memory()
        memory.partition = 0
        memory.address = 0

        self.assertEqual(
            text_format.MessageToString(memory),
            "partition: 0\naddress: 0\n",
        )

    def test_liveness_reuse_is_isolated_by_physical_instance(self):
        input_node, weight, conv, relu = _conv_graph()
        allocator = ScratchpadAllocator(_global_memory())

        input_memory = allocator.allocate_memory(
            input_node,
            get_allocation_role(input_node),
        )
        weight_memory = allocator.allocate_memory(
            weight,
            get_allocation_role(weight),
        )
        conv_memory = allocator.allocate_memory(
            conv,
            get_allocation_role(conv),
        )
        allocator.free_memory(input_node)
        relu_memory = allocator.allocate_memory(
            relu,
            get_allocation_role(relu),
        )

        self.assertEqual(input_memory.memory_space, 0)
        self.assertEqual(conv_memory.memory_space, 0)
        self.assertEqual(weight_memory.memory_space, 1)
        self.assertEqual(relu_memory.start, input_memory.start)
        self.assertEqual(input_memory.start % 64, 0)
        self.assertEqual(weight_memory.start % 64, 0)

    def test_scratchpad_capacity_failure_does_not_spill_to_dram(self):
        memory = MemoryLevelConfig(
            name="tiny_global",
            instances=(
                MemoryInstance(
                    name="shared",
                    bank_count=1,
                    depth_per_bank=1,
                    bits_per_bank_entry=8,
                    accepted_accesses=frozenset(
                        {
                            (TensorRole.ACTIVATION, None),
                            (TensorRole.OUTPUT, None),
                            (TensorRole.WEIGHT, None),
                        }
                    ),
                    access_cost=100,
                ),
            ),
            input_memory="shared",
            output_memory="shared",
            weight_memory="shared",
        )
        _model, input_node, _first, _second = _relu_graph()

        with self.assertRaisesRegex(RuntimeError, "Memory allocation failed"):
            ScratchpadAllocator(memory).allocate_memory(
                input_node,
                TensorRole.ACTIVATION,
            )

    def test_shared_allocator_respects_compute_unit_access(self):
        memory = MemoryLevelConfig(
            name="split_global",
            instances=(
                MemoryInstance(
                    name="matrix_activation",
                    bank_count=1,
                    depth_per_bank=64,
                    bits_per_bank_entry=128,
                    accepted_accesses=frozenset(
                        {
                            (TensorRole.ACTIVATION, "matrix"),
                            (TensorRole.OUTPUT, "matrix"),
                            (TensorRole.WEIGHT, "matrix"),
                        }
                    ),
                    access_cost=1,
                ),
                MemoryInstance(
                    name="vector_activation",
                    bank_count=1,
                    depth_per_bank=64,
                    bits_per_bank_entry=128,
                    accepted_accesses=frozenset(
                        {(TensorRole.ACTIVATION, "vector")}
                    ),
                    access_cost=1,
                ),
            ),
            input_memory="matrix_activation",
            output_memory="matrix_activation",
            weight_memory="matrix_activation",
        )
        _model, input_node, _first, _second = _relu_graph()

        segment = ScratchpadAllocator(memory).allocate_memory(
            input_node,
            TensorRole.ACTIVATION,
            access_requirements={
                (TensorRole.ACTIVATION, "vector"),
            },
        )

        self.assertEqual(segment.memory_space, 1)

    def test_shared_allocator_uses_exact_physical_access_pairs(self):
        memory = MemoryLevelConfig(
            name="split_global",
            instances=(
                MemoryInstance(
                    name="matrix",
                    bank_count=1,
                    depth_per_bank=64,
                    bits_per_bank_entry=128,
                    accepted_accesses=frozenset(
                        {
                            (TensorRole.ACTIVATION, "matrix"),
                            (TensorRole.OUTPUT, "matrix"),
                            (TensorRole.WEIGHT, "matrix"),
                        }
                    ),
                    access_cost=1,
                ),
                MemoryInstance(
                    name="matrix_to_vector",
                    bank_count=1,
                    depth_per_bank=64,
                    bits_per_bank_entry=128,
                    accepted_accesses=frozenset(
                        {
                            (TensorRole.OUTPUT, "matrix"),
                            (TensorRole.ACTIVATION, "vector"),
                        }
                    ),
                    access_cost=1,
                ),
                MemoryInstance(
                    name="vector",
                    bank_count=1,
                    depth_per_bank=64,
                    bits_per_bank_entry=128,
                    accepted_accesses=frozenset(
                        {
                            (TensorRole.OUTPUT, "vector"),
                        }
                    ),
                    access_cost=1,
                ),
            ),
            input_memory="matrix",
            output_memory="matrix",
            weight_memory="matrix",
        )
        hardware = _hardware()
        vector = replace(hardware.pe, name="vector")
        hardware = replace(
            hardware,
            global_memory=memory,
            compute_units=(hardware.pe, vector),
        )
        model, input_node, first, second = _relu_graph()
        first.meta["compute_unit"] = "matrix"
        second.meta["compute_unit"] = "vector"

        run_memory_mapping(
            model,
            MemoryAllocator(4096),
            unroll_dims=hardware.unroll_dims,
            hardware_config=hardware,
            scratchpad_allocator=ScratchpadAllocator(memory),
            scratchpad_mapping=END_TO_END_SCRATCHPAD_MAPPING,
        )

        self.assertEqual(input_node.meta["scratchpad"].memory_space, 0)
        self.assertEqual(first.meta["scratchpad"].memory_space, 1)
        self.assertEqual(second.meta["scratchpad"].memory_space, 2)

    def test_operation_banking_uses_role_specific_bank_geometry(self):
        input_node, weight, conv, _relu = _conv_graph()
        memory = _global_memory()
        node_to_key = get_node_to_key_map(conv)
        scope = _get_scope(conv.target)
        key_to_node = {
            f"{scope}::{key}": node for node, key in node_to_key.items()
        }
        tiled_shapes = {
            input_node: input_node.shape,
            weight: weight.shape,
            conv: conv.shape,
        }

        strategy = get_banking_strategies_for_op(conv.target)[0]
        _total_size, placements = strategy.evaluate(
            key_to_node,
            conv,
            tiled_shapes,
            bank_width=None,
            bank_size=None,
            unroll_dim=2,
            memory_level=memory,
        )

        activation = memory.memory("activation")
        self.assertEqual(placements[input_node].memory_space, 0)
        self.assertEqual(placements[conv].memory_space, 0)
        self.assertEqual(placements[weight].memory_space, 1)
        self.assertEqual(placements[input_node].start, 0)
        self.assertEqual(
            placements[conv].start,
            activation.bank_capacity_bytes,
        )
        self.assertEqual(placements[weight].start, 0)
        self.assertTrue(memory.fits_segments(placements.values()))

    def test_end_to_end_mapping_uses_model_wide_scratchpad_addresses(self):
        model, input_node, first, second = _relu_graph()
        hardware = _hardware()
        example = torch.randn(1, 2, 4, 4)

        run_memory_mapping(
            model,
            MemoryAllocator(4096),
            unroll_dims=hardware.unroll_dims,
            hardware_config=hardware,
            scratchpad_mapping=END_TO_END_SCRATCHPAD_MAPPING,
        )

        self.assertIn("memory", input_node.meta)
        self.assertIn("scratchpad", input_node.meta)
        self.assertNotIn("memory", first.meta)
        self.assertNotIn("memory", second.meta)
        self.assertNotIn("scratchpad_map", first.meta)
        self.assertNotIn("scratchpad_map", second.meta)
        self.assertEqual(
            second.meta["scratchpad"].start,
            input_node.meta["scratchpad"].start,
        )
        storage_plan = model.meta["storage_plan"]
        input_dram = storage_plan.physical_id(
            input_node,
            "memory",
        )
        input_scratchpad = storage_plan.physical_id(
            input_node,
            "scratchpad",
        )
        second_scratchpad = storage_plan.physical_id(
            second,
            "scratchpad",
        )
        self.assertNotEqual(input_dram, input_scratchpad)
        self.assertNotEqual(input_scratchpad, second_scratchpad)
        self.assertLessEqual(
            storage_plan.intervals["scratchpad"][input_scratchpad].death,
            storage_plan.intervals["scratchpad"][second_scratchpad].birth,
        )

        params = gen_code(model, (example,))
        first_output = params.ops[0].output
        second_input = params.ops[1].op.kwargs["input"].tensor
        self.assertFalse(first_output.HasField("memory"))
        self.assertEqual(
            first_output.scratchpad.address,
            first.meta["scratchpad"].start,
        )
        self.assertEqual(
            second_input.scratchpad.address,
            first_output.scratchpad.address,
        )

    def test_per_operation_mapping_preserves_dram_backing(self):
        model, _input_node, first, second = _relu_graph()

        run_memory_mapping(
            model,
            MemoryAllocator(4096, bank_width=64),
            cache_size=4096,
            num_banks=4,
            bank_width=64,
            unroll_dims=(2, 2),
            scratchpad_mapping=PER_OPERATION_SCRATCHPAD_MAPPING,
        )

        self.assertIn("memory", first.meta)
        self.assertIn("memory", second.meta)
        self.assertNotIn("scratchpad", first.meta)
        self.assertIsNotNone(first.meta.get("scratchpad_map"))

    def test_per_operation_profile_without_scoped_allocator_uses_dram(self):
        model, _input_node, first, second = _relu_graph()
        hardware = _hardware()

        with patch(
            "voyager_compiler.codegen.mapping.create_scratchpad_allocator"
        ) as create_allocator:
            run_memory_mapping(
                model,
                MemoryAllocator(4096),
                hardware_config=hardware,
                scratchpad_mapping=PER_OPERATION_SCRATCHPAD_MAPPING,
            )

        create_allocator.assert_not_called()
        self.assertIn("memory", first.meta)
        self.assertIn("memory", second.meta)
        self.assertNotIn("scratchpad", first.meta)
        self.assertNotIn("scratchpad", second.meta)


if __name__ == "__main__":
    unittest.main()

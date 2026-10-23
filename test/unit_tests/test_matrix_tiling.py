# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import operator
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import torch
from torch.fx import Graph, GraphModule

from scheduler.dispatch import (
    GENERIC_SCHEDULER_PROVIDERS,
    INTERSTELLAR_SCHEDULER,
    resolve_local_scheduler,
    schedule_operation,
)
from scheduler.magnet_alpha_scheduler.providers import (
    MAGNET_ALPHA_SCHEDULER_PROVIDERS,
)
from scheduler.types import (
    GemmWorkload,
    LocalScheduleResult,
    Loop,
    MappingPoint,
    OperationClassification,
    OperationOperand,
    OperationWorkload,
    SchedulerProviderSet,
)
from scheduler.voyager import extract_gemm_workload
from voyager_compiler.codegen import (
    ShapeProp,
    convert_cat_and_stack_as_stack_on_dim0,
)
from voyager_compiler.codegen.hardware import (
    DTypeMode,
    HardwareConfig,
    UnrollDimension,
)
from voyager_compiler.codegen.mapping import (
    _get_scheduler_operands,
    _resolve_scheduler_output,
    build_scheduler_workload,
)
from voyager_compiler.codegen.mapping_utils import (
    TARGET_CODEGEN_KWARGS,
    TARGET_INSTRUCTION_NOP,
    is_declared_unsupported_operation,
    is_depthwise_conv,
)
from voyager_compiler.codegen.passes.padding import pad_matrix_op_dimensions
from voyager_compiler.codegen.passes.tiling import (
    _build_conv2d_shape_map,
    compute_output_tiled_shapes,
    run_matrix_op_l2_tiling,
    search_conv2d_tiling,
    split_conv2d_node,
)
from voyager_compiler.codegen.passes.utils import (
    get_hardware_logical_shape,
    inherit_hardware_logical_shape,
    preserve_hardware_logical_shape,
)
from voyager_compiler.codegen.storage import ValueRef
from voyager_compiler.codegen.target_backends import TargetOperatorTraits
from voyager_compiler.codegen.targets import DEFAULT_COMPILATION_TARGET
from voyager_compiler.pt2e_utils import COMPILER_ONLY
from voyager_compiler.quantization_contracts import (
    PhysicalQuantizationContract,
    set_physical_quantization_contract,
)

CONV_TARGETS = (
    torch.ops.aten.conv2d.default,
    torch.ops.quantized_ops.conv2d.default,
    torch.ops.quantized_ops.conv2d_mx.default,
)


def _make_tiling_hardware_config():
    base = HardwareConfig.from_legacy(
        unroll_dims=(32, 32),
        cache_size=4096,
        input_buffer_size=256,
        weight_buffer_size=64,
        accum_buffer_size=32,
        num_banks=4,
        bank_width=16,
    )
    matrix = replace(
        base.pe,
        unroll_dimensions=(
            UnrollDimension("IC", 32, dtype_extents=((8, 16),)),
            UnrollDimension("OC", 32),
        ),
        accepted_input_bits=frozenset({4, 8}),
    )
    return replace(
        base,
        pe=matrix,
        compute_units=(matrix,),
        dtype_modes=(
            DTypeMode(4, output_channel_alignment=32),
            DTypeMode(
                8,
                banks_per_value=2,
                output_channel_alignment=16,
            ),
        ),
        partial_matrix_input_channels=frozenset({1, 3}),
    )


TILING_HARDWARE_CONFIG = _make_tiling_hardware_config()


def _make_conv_graph(target):
    graph = Graph()
    input_node = graph.placeholder("input")
    depthwise_weight = graph.placeholder("depthwise_weight")
    regular_weight = graph.placeholder("regular_weight")
    input_node.shape = (1, 4, 8, 8)
    depthwise_weight.shape = (4, 1, 3, 3)
    regular_weight.shape = (8, 4, 1, 1)
    depthwise = graph.call_function(
        target,
        (
            input_node,
            depthwise_weight,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        ),
    )
    regular = graph.call_function(
        target,
        (
            depthwise,
            regular_weight,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        ),
    )
    graph.output(regular)
    return GraphModule(torch.nn.Module(), graph), depthwise, regular


def _make_grouped_conv_node(target):
    graph = Graph()
    input_node = graph.placeholder("input")
    weight = graph.placeholder("weight")
    input_node.shape = (1, 4, 8, 8)
    weight.shape = (8, 2, 3, 3)
    grouped = graph.call_function(
        target,
        (
            input_node,
            weight,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        ),
    )
    graph.output(grouped)
    return GraphModule(torch.nn.Module(), graph), grouped


def _make_rewritten_fused_conv():
    graph = Graph()
    original_input = graph.placeholder("original_input")
    current_input = graph.call_function(
        torch.ops.aten.add.Scalar,
        (original_input, 0),
    )
    weight = graph.placeholder("weight")

    subgraph = Graph()
    inner_input = subgraph.placeholder("input")
    inner_weight = subgraph.placeholder("weight")
    inner_input.meta["source_node"] = original_input
    inner_weight.meta["source_node"] = weight
    conv = subgraph.call_function(
        torch.ops.aten.conv2d.default,
        (
            inner_input,
            inner_weight,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        ),
    )
    subgraph.output(conv)
    fused = GraphModule(torch.nn.Module(), subgraph)

    root = torch.nn.Module()
    root.add_module("fused", fused)
    output = graph.call_module("fused", (current_input, weight))
    graph.output(output)
    model = GraphModule(root, graph)
    return model, current_input, weight, output, conv


def _make_mixed_precision_conv_graph():
    root = torch.nn.Module()
    root.register_buffer("int8_weight", torch.randn(32, 16, 1, 1))
    root.register_buffer("int4_weight", torch.randn(32, 16, 1, 1))

    graph = Graph()
    int8_input = graph.placeholder("int8_input")
    int4_input = graph.placeholder("int4_input")
    int8_weight = graph.get_attr("int8_weight")
    int4_weight = graph.get_attr("int4_weight")
    int8_conv = graph.call_function(
        torch.ops.aten.conv2d.default,
        (int8_input, int8_weight),
    )
    int4_conv = graph.call_function(
        torch.ops.aten.conv2d.default,
        (int4_input, int4_weight),
    )
    graph.output((int8_conv, int4_conv))
    model = GraphModule(root, graph)
    inputs = (
        torch.randn(1, 16, 4, 4),
        torch.randn(1, 16, 4, 4),
    )
    ShapeProp(model).propagate(*inputs)
    int8_input.meta["dtype"] = "int8"
    int4_input.meta["dtype"] = "int4"
    return model, inputs, int8_conv, int4_conv


def _make_int8_output_channels_graph(
    output_channels,
    input_channels=16,
    batch_size=1,
):
    root = torch.nn.Module()
    root.register_buffer(
        "weight",
        torch.randn(output_channels, input_channels, 1, 1),
    )

    graph = Graph()
    input_node = graph.placeholder("input")
    weight = graph.get_attr("weight")
    conv = graph.call_function(
        torch.ops.aten.conv2d.default,
        (input_node, weight),
    )
    graph.output(conv)
    model = GraphModule(root, graph)
    inputs = (torch.randn(batch_size, input_channels, 4, 4),)
    ShapeProp(model).propagate(*inputs)
    input_node.meta["dtype"] = "int8"
    return model, inputs, conv


def _make_depthwise_multiplier_graph():
    graph = Graph()
    input_node = graph.placeholder("input")
    weight = graph.placeholder("weight")
    depthwise = graph.call_function(
        torch.ops.aten.conv2d.default,
        (
            input_node,
            weight,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            5,
        ),
    )
    graph.output(depthwise)
    model = GraphModule(torch.nn.Module(), graph)
    inputs = (
        torch.randn(1, 5, 8, 8),
        torch.randn(10, 1, 3, 3),
    )
    ShapeProp(model).propagate(*inputs)
    return model, inputs, depthwise


def _make_quantize_mx_fanout_graph():
    root = torch.nn.Module()
    root.register_buffer("qmap", torch.zeros(65536))
    root.register_buffer("weight1", torch.randn(8, 5, 1, 1))
    root.register_buffer("weight2", torch.randn(8, 5, 1, 1))

    graph = Graph()
    input_node = graph.placeholder("input")
    qmap = graph.get_attr("qmap")
    weight1 = graph.get_attr("weight1")
    weight2 = graph.get_attr("weight2")
    quantize_mx = graph.call_function(
        torch.ops.quantized_ops.quantize_mx.default,
        (input_node, qmap, [1], 4, 7.0),
    )
    scale = graph.call_function(operator.getitem, (quantize_mx, 0))
    payload = graph.call_function(operator.getitem, (quantize_mx, 1))
    conv1 = graph.call_function(
        torch.ops.quantized_ops.conv2d_mx.default,
        (payload, weight1),
        {"input_scale": scale, "block_size": 4},
    )
    conv2 = graph.call_function(
        torch.ops.quantized_ops.conv2d_mx.default,
        (payload, weight2),
        {"input_scale": scale, "block_size": 4},
    )
    graph.output((conv1, conv2, payload))
    model = GraphModule(root, graph)
    inputs = (torch.randn(1, 5, 4, 4),)
    ShapeProp(model).propagate(*inputs)
    qmap.meta[COMPILER_ONLY] = True
    payload.meta["dtype"] = "int4"
    scale.meta["dtype"] = "float32"
    quantize_mx.meta["dtype"] = ("float32", "int4")
    return model, inputs, quantize_mx


def _make_per_channel_dequant_conv_graph():
    root = torch.nn.Module()
    root.register_buffer("weight", torch.randn(8, 4, 1, 1))
    root.register_buffer("scale", torch.ones(8, 1, 1))

    graph = Graph()
    input_node = graph.placeholder("input")
    weight = graph.get_attr("weight")
    scale = graph.get_attr("scale")
    conv = graph.call_function(
        torch.ops.aten.conv2d.default,
        (input_node, weight),
    )
    dequantize = graph.call_function(
        torch.ops.quantized_ops.dequantize.default,
        (conv, scale),
    )
    graph.output(dequantize)
    model = GraphModule(root, graph)
    inputs = (torch.randn(1, 4, 4, 4),)
    ShapeProp(model).propagate(*inputs)
    return model, inputs, conv


class SchedulerWorkloadTest(unittest.TestCase):
    @staticmethod
    def _unit_mapping():
        unit_factors = tuple((1,) for _loop in Loop)
        return MappingPoint(
            tuple((0,) for _loop in Loop),
            unit_factors,
            unit_factors,
        )

    def test_local_schedule_named_factors_are_validated_and_immutable(self):
        source = {"q1": 2, "p1": 1}
        result = LocalScheduleResult(
            mapping=self._unit_mapping(),
            backend="test",
            named_factors=source,
        )
        source["q1"] = 8

        self.assertEqual(
            result.named_factors,
            (("p1", 1), ("q1", 2)),
        )
        self.assertEqual(result.factor("q1"), 2)
        self.assertIsNone(result.factor("missing"))
        self.assertIsInstance(hash(result), int)

    def test_local_schedule_rejects_invalid_named_factors(self):
        invalid_factors = (
            {"Q1": 1},
            {"q-1": 1},
            {"q1": 0},
            {"q1": True},
            (("q1", 1), ("q1", 2)),
        )
        for factors in invalid_factors:
            with self.subTest(factors=factors):
                with self.assertRaises(ValueError):
                    LocalScheduleResult(
                        mapping=self._unit_mapping(),
                        backend="test",
                        named_factors=factors,
                    )

    def test_operand_order_does_not_change_schedule_cache_identity(self):
        output = OperationOperand("output", (1, 8, 4, 4), 8)
        input_operand = OperationOperand("input", (1, 4, 4, 4), 8)
        weight_operand = OperationOperand("weight", (8, 4, 1, 1), 8)
        first = OperationWorkload(
            operands=(weight_operand, input_operand),
            output=output,
        )
        second = OperationWorkload(
            operands=(input_operand, weight_operand),
            output=output,
        )

        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))

    def test_repeated_ports_keep_typed_occurrence_identity(self):
        operands = (
            OperationOperand(
                "tensors",
                (1, 4),
                8,
                occurrence=(0,),
            ),
            OperationOperand(
                "tensors",
                (1, 4),
                8,
                occurrence=(1,),
            ),
        )
        workload = OperationWorkload(
            operands=operands,
            output=OperationOperand("output", (1, 8), 8),
            operation=OperationClassification(
                "concatenate",
                {"dim": 1},
            ),
        )

        self.assertFalse(hasattr(workload, "operator"))
        self.assertEqual(
            workload.operand("tensors", (1,)),
            operands[1],
        )
        self.assertEqual(workload.operation.attribute("dim"), 1)
        with self.assertRaisesRegex(ValueError, "multiple 'tensors'"):
            workload.operand("tensors")

    def test_operation_provider_precedes_generic_gemm_provider(self):
        expected = LocalScheduleResult(
            mapping=self._unit_mapping(),
            backend="unit_operation",
        )
        operation_provider = Mock(return_value=expected)
        gemm_provider = Mock()
        providers = SchedulerProviderSet(
            default_backend="unit_gemm",
            gemm_providers={"unit_gemm": gemm_provider},
            operation_provider=operation_provider,
        )
        workload = OperationWorkload(
            operands=(OperationOperand("input", (1, 4), 8),),
            output=OperationOperand("output", (1, 4), 8),
            operation=OperationClassification("unit_operation"),
            gemm=GemmWorkload(1, 1, 4, 1, 1, 1),
        )
        hardware = object()

        result = schedule_operation(
            workload,
            hardware,
            {},
            providers=providers,
        )

        self.assertIs(result, expected)
        operation_provider.assert_called_once_with(workload, hardware)
        gemm_provider.assert_not_called()

    def test_scheduler_output_uses_selected_component_dtype(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        output_node = graph.call_function(operator.neg, (input_node,))
        output_node.value = (
            torch.empty((), dtype=torch.float32),
            torch.empty(2, 2, dtype=torch.uint8),
            torch.empty(2, 8, dtype=torch.int8),
        )
        output_node.meta["dtype"] = (
            "float32",
            "uint8",
            "int8",
        )
        set_physical_quantization_contract(
            output_node,
            PhysicalQuantizationContract(
                "int4",
                ("uint8",),
                32,
            ),
        )
        output_node.meta[TARGET_CODEGEN_KWARGS] = {
            "output_payload": ValueRef(output_node, (2,))
        }

        output = _resolve_scheduler_output(
            output_node,
            output_node,
            {
                output_node: (
                    (),
                    (2, 2),
                    (2, 8),
                )
            },
        )

        self.assertEqual(output.occurrence, (2,))
        self.assertEqual(output.shape, (2, 8))
        self.assertEqual(output.bits, 8)

    def test_sparse_gemm_output_falls_back_to_last_result(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        weight = graph.placeholder("weight")
        data = graph.placeholder("data")
        indices = graph.placeholder("indices")
        indptr = graph.placeholder("indptr")
        anchor = graph.call_function(
            torch.ops.quantized_ops.linear_mx.default,
            (input_node, weight),
            {
                "A_data": data,
                "A_indices": indices,
                "A_indptr": indptr,
            },
        )
        output_node = graph.call_function(operator.neg, (input_node,))
        output_node.value = (
            torch.empty(10, dtype=torch.int8),
            torch.empty(10, dtype=torch.int32),
            torch.empty(3, dtype=torch.int32),
            torch.empty(2, 2),
            torch.empty(2, 8, dtype=torch.int8),
        )
        output_node.meta["dtype"] = (
            "int4",
            "int32",
            "int32",
            "float32",
            "int4",
        )
        input_node.meta["dtype"] = "int4"
        weight.meta["dtype"] = "int4"
        data.meta["dtype"] = "int4"
        indices.meta["dtype"] = "int32"
        indptr.meta["dtype"] = "int32"
        tiled_shapes = {
            input_node: (2, 8),
            weight: (8, 8),
            data: (10,),
            indices: (10,),
            indptr: (3,),
            output_node: (
                (10,),
                (10,),
                (3,),
                (2, 2),
                (2, 8),
            ),
        }

        workload = build_scheduler_workload(
            output_node,
            anchor,
            tiled_shapes,
            None,
            {},
            DEFAULT_COMPILATION_TARGET,
        )

        self.assertEqual(workload.output.occurrence, (4,))
        self.assertEqual(workload.output.shape, (2, 8))
        self.assertEqual(workload.output.bits, 4)
        self.assertTrue(workload.has_sparse_op)
        self.assertIsNotNone(workload.gemm)
        self.assertEqual(workload.gemm.wofm, 2)

    def test_ambiguous_multi_output_requires_value_ref(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        output_node = graph.call_function(operator.neg, (input_node,))
        output_node.value = (
            torch.empty(2, 4),
            torch.empty(2, 4),
        )

        with self.assertRaisesRegex(
            ValueError,
            "requires an explicit output_payload ValueRef",
        ):
            _resolve_scheduler_output(
                output_node,
                output_node,
                {output_node: ((2, 4), (2, 4))},
            )

    def test_unscheduled_quantize_mx_does_not_guess_an_output(self):
        model, _, quantize_mx = _make_quantize_mx_fanout_graph()
        tiled_shapes = {
            node: node.shape for node in quantize_mx.all_input_nodes
        }
        tiled_shapes[quantize_mx] = tuple(
            tuple(value.shape) for value in quantize_mx.value
        )

        workload = build_scheduler_workload(
            quantize_mx,
            quantize_mx,
            tiled_shapes,
            None,
            dict(model.named_modules(remove_duplicate=False)),
            DEFAULT_COMPILATION_TARGET,
        )

        self.assertIsNone(workload)


class MatrixTilingTest(unittest.TestCase):
    def test_stride_two_tile_budgets_materialized_padding(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        weight = graph.placeholder("weight")
        input_node.shape = (1, 64, 56, 56)
        weight.shape = (128, 64, 3, 3)
        conv = graph.call_function(
            torch.ops.aten.conv2d.default,
            (
                input_node,
                weight,
                None,
                (2, 2),
                (1, 1),
                (1, 1),
                1,
            ),
        )
        conv.shape = (1, 128, 28, 28)

        tiled = _build_conv2d_shape_map(
            conv,
            (4, 14, 64, 16),
        )
        untiled = _build_conv2d_shape_map(
            conv,
            (28, 28, 64, 16),
        )
        one_axis_tiled = _build_conv2d_shape_map(
            conv,
            (28, 14, 64, 16),
        )

        self.assertEqual(tiled["input"], (1, 64, 9, 29))
        self.assertEqual(tiled["output"], (1, 16, 4, 14))
        self.assertEqual(untiled["input"], (1, 64, 56, 56))
        self.assertEqual(one_axis_tiled["input"], (1, 64, 57, 29))

    def test_generic_scheduler_default_is_interstellar(self):
        self.assertEqual(
            resolve_local_scheduler(
                None,
                providers=GENERIC_SCHEDULER_PROVIDERS,
            ),
            INTERSTELLAR_SCHEDULER,
        )

    def test_explicit_scheduler_overrides_provider_default(self):
        providers = SchedulerProviderSet(
            default_backend="default",
            gemm_providers={
                "default": lambda *_args: None,
                "alternate": lambda *_args: None,
            },
        )
        self.assertEqual(
            resolve_local_scheduler(
                "alternate",
                providers=providers,
            ),
            "alternate",
        )

    def test_output_tiling_distinguishes_vsq_from_sparse_tuple(self):
        class Output:
            value = (
                torch.empty(()),
                torch.empty(1, 1, 24, 24),
                torch.empty(1, 32, 24, 24),
            )

        shapes = compute_output_tiled_shapes(
            Output(),
            (1, 1, 2, 2),
            has_sparse_outputs=False,
        )

        self.assertEqual(
            shapes,
            ((), (1, 1, 12, 12), (1, 32, 12, 12)),
        )

    def test_output_tiling_preserves_explicit_sparse_layout(self):
        class Output:
            value = (
                torch.empty(1, 10),
                torch.empty(1, 10),
                torch.empty(1, 7),
            )

        shapes = compute_output_tiled_shapes(
            Output(),
            (2, 1),
            has_sparse_outputs=True,
        )

        self.assertEqual(shapes, ((1, 5), (1, 5), (1, 4)))

    def test_hardware_padding_preserves_original_logical_shape(self):
        graph = Graph()
        value = graph.placeholder("value")
        value.shape = (8, 3, 3, 3)

        self.assertEqual(
            preserve_hardware_logical_shape(value),
            (8, 3, 3, 3),
        )
        value.shape = (16, 4, 3, 3)
        self.assertEqual(
            preserve_hardware_logical_shape(value),
            (8, 3, 3, 3),
        )
        self.assertEqual(
            get_hardware_logical_shape(value),
            (8, 3, 3, 3),
        )

    def test_padded_replacement_inherits_semantic_shape(self):
        graph = Graph()
        source = graph.placeholder("source")
        padded = graph.placeholder("padded")
        source.shape = (1, 3, 8, 8)
        padded.shape = (1, 4, 8, 8)
        preserve_hardware_logical_shape(source)

        self.assertEqual(
            inherit_hardware_logical_shape(padded, source),
            (1, 3, 8, 8),
        )
        self.assertEqual(
            get_hardware_logical_shape(padded),
            (1, 3, 8, 8),
        )

    def test_hardware_padding_uses_per_node_dtype_unroll(self):
        model, inputs, int8_conv, int4_conv = _make_mixed_precision_conv_graph()
        expected = model(*inputs)

        pad_matrix_op_dimensions(
            model,
            hardware_config=TILING_HARDWARE_CONFIG,
        )
        actual = model(*inputs)

        self.assertEqual(tuple(int8_conv.args[0].shape), (1, 16, 4, 4))
        self.assertEqual(tuple(int8_conv.args[1].shape), (32, 16, 1, 1))
        self.assertEqual(tuple(int4_conv.args[0].shape), (1, 32, 4, 4))
        self.assertEqual(tuple(int4_conv.args[1].shape), (32, 32, 1, 1))
        torch.testing.assert_close(actual, expected)

    def test_int8_software_output_padding_uses_half_vector_alignment(self):
        expected_resident_channels = {
            8: 16,
            16: 16,
            32: 32,
            48: 64,
        }
        for (
            logical_channels,
            resident_channels,
        ) in expected_resident_channels.items():
            with self.subTest(logical_channels=logical_channels):
                model, inputs, conv = _make_int8_output_channels_graph(
                    logical_channels
                )
                expected = model(*inputs)

                pad_matrix_op_dimensions(
                    model,
                    hardware_config=TILING_HARDWARE_CONFIG,
                )
                actual = model(*inputs)

                self.assertEqual(
                    tuple(conv.args[1].shape),
                    (resident_channels, 16, 1, 1),
                )
                self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                torch.testing.assert_close(actual, expected)

    def test_output_alignment_is_dtype_declarative(self):
        hardware = TILING_HARDWARE_CONFIG

        self.assertEqual(hardware.matrix_output_alignment(4), 32)
        self.assertEqual(hardware.matrix_output_alignment(8), 16)
        self.assertTrue(hardware.supports_partial_matrix_output(16, 8))
        self.assertFalse(hardware.supports_partial_matrix_output(8, 8))
        self.assertFalse(hardware.supports_partial_matrix_output(32, 8))
        self.assertFalse(hardware.supports_partial_matrix_output(48, 8))
        self.assertEqual(
            hardware.matrix_output_resident_channels(48, 8),
            64,
        )

    def test_partial_image_input_still_legalizes_output_channels(self):
        model, inputs, conv = _make_int8_output_channels_graph(
            8,
            input_channels=1,
        )
        expected = model(*inputs)

        pad_matrix_op_dimensions(
            model,
            hardware_config=TILING_HARDWARE_CONFIG,
        )
        actual = model(*inputs)

        self.assertEqual(tuple(conv.args[0].shape), (1, 1, 4, 4))
        self.assertEqual(tuple(conv.args[1].shape), (16, 1, 1, 1))
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        torch.testing.assert_close(actual, expected)

    def test_legacy_padding_keeps_fixed_unroll(self):
        model, inputs, int8_conv, int4_conv = _make_mixed_precision_conv_graph()
        expected = model(*inputs)

        pad_matrix_op_dimensions(model, 32, 32)
        actual = model(*inputs)

        self.assertEqual(tuple(int8_conv.args[0].shape), (1, 32, 4, 4))
        self.assertEqual(tuple(int4_conv.args[0].shape), (1, 32, 4, 4))
        torch.testing.assert_close(actual, expected)

    def test_depthwise_padding_preserves_channel_multiplier(self):
        model, inputs, depthwise = _make_depthwise_multiplier_graph()
        expected = model(*inputs)

        pad_matrix_op_dimensions(model, 4, 8)
        actual = model(*inputs)

        padded_depthwise = next(
            node
            for node in model.graph.nodes
            if node.target == torch.ops.aten.conv2d.default
        )
        self.assertEqual(padded_depthwise.args[6], 8)
        self.assertEqual(tuple(padded_depthwise.args[0].shape), (1, 8, 8, 8))
        self.assertEqual(tuple(padded_depthwise.args[1].shape), (16, 1, 3, 3))
        self.assertEqual(tuple(depthwise.shape), (1, 16, 8, 8))
        torch.testing.assert_close(actual, expected)

    def test_typed_profile_pads_depthwise_execution_channels(self):
        model, inputs, depthwise = _make_depthwise_multiplier_graph()
        input_node = next(
            node for node in model.graph.nodes if node.op == "placeholder"
        )
        input_node.meta["dtype"] = "int8"
        expected = model(*inputs)

        self.assertTrue(TILING_HARDWARE_CONFIG.pad_depthwise_channels)
        pad_matrix_op_dimensions(
            model,
            hardware_config=TILING_HARDWARE_CONFIG,
        )
        actual = model(*inputs)

        padded_depthwise = next(
            node
            for node in model.graph.nodes
            if node.target == torch.ops.aten.conv2d.default
        )
        self.assertEqual(padded_depthwise.args[6], 16)
        self.assertEqual(
            tuple(padded_depthwise.args[0].shape),
            (1, 16, 8, 8),
        )
        self.assertEqual(
            tuple(padded_depthwise.args[1].shape),
            (32, 1, 3, 3),
        )
        self.assertEqual(tuple(depthwise.shape), (1, 32, 8, 8))
        logical_slice = next(iter(depthwise.users))
        self.assertEqual(
            logical_slice.target,
            torch.ops.aten.slice.Tensor,
        )
        self.assertTrue(logical_slice.meta["hardware_padding_slice"])
        self.assertEqual(tuple(logical_slice.shape), (1, 10, 8, 8))
        torch.testing.assert_close(actual, expected)

    def test_padding_extends_per_channel_output_scale(self):
        model, inputs, conv = _make_per_channel_dequant_conv_graph()
        expected = model(*inputs)

        pad_matrix_op_dimensions(model, 4, 32)
        actual = model(*inputs)

        self.assertEqual(tuple(conv.args[1].shape), (32, 4, 1, 1))
        self.assertEqual(tuple(model.get_buffer("scale").shape), (32, 1, 1))
        torch.testing.assert_close(
            model.get_buffer("scale")[:8],
            torch.ones(8, 1, 1),
        )
        torch.testing.assert_close(
            model.get_buffer("scale")[8:],
            torch.zeros(24, 1, 1),
        )
        torch.testing.assert_close(actual, expected)

    def test_quantize_mx_padding_clones_each_fanout_edge(self):
        model, inputs, original_quantize = _make_quantize_mx_fanout_graph()
        expected = model(*inputs)

        pad_matrix_op_dimensions(model, 8, 8)
        actual = model(*inputs)

        quantize_nodes = [
            node
            for node in model.graph.nodes
            if node.target == torch.ops.quantized_ops.quantize_mx.default
        ]
        self.assertEqual(len(quantize_nodes), 3)
        self.assertIs(quantize_nodes[0], original_quantize)
        self.assertTrue(
            all(
                node.args[1].meta.get(COMPILER_ONLY, False)
                for node in quantize_nodes
            )
        )
        self.assertEqual(tuple(original_quantize.args[0].shape), (1, 5, 4, 4))
        self.assertTrue(
            all(
                tuple(node.args[0].shape) == (1, 8, 4, 4)
                for node in quantize_nodes[1:]
            )
        )
        self.assertEqual(tuple(actual[2].shape), (1, 5, 4, 4))
        torch.testing.assert_close(actual, expected)

    def test_l2_tiling_uses_per_node_dtype_unroll(self):
        model, _, int8_conv, int4_conv = _make_mixed_precision_conv_graph()

        with patch(
            "voyager_compiler.codegen.passes.tiling.search_conv2d_tiling",
            return_value=(None, None),
        ) as search:
            run_matrix_op_l2_tiling(
                model,
                cache_size=1024,
                hardware_config=TILING_HARDWARE_CONFIG,
            )

        self.assertEqual(search.call_count, 2)
        calls = {call.args[0]: call.args[1] for call in search.call_args_list}
        self.assertEqual(calls[int8_conv], (16, 32))
        self.assertEqual(calls[int4_conv], (32, 32))

    def test_target_rejected_conv_tile_becomes_unsupported_boundary(self):
        model, _, int8_conv, int4_conv = _make_mixed_precision_conv_graph()
        providers = replace(
            GENERIC_SCHEDULER_PROVIDERS,
            gemm_tiling_candidate_fits=Mock(return_value=False),
        )

        with (
            patch(
                "voyager_compiler.codegen.passes.tiling."
                "get_target_scheduler_providers",
                return_value=providers,
            ),
            patch(
                "voyager_compiler.codegen.passes.tiling."
                "search_conv2d_tiling",
                return_value=(None, None),
            ),
        ):
            run_matrix_op_l2_tiling(
                model,
                cache_size=1024,
                hardware_config=TILING_HARDWARE_CONFIG,
            )

        for convolution in (int8_conv, int4_conv):
            self.assertTrue(is_declared_unsupported_operation(convolution))
            declaration = convolution.meta["unsupported_operation"]
            self.assertEqual(
                declaration["reason_code"],
                "no_endpoint_legal_conv2d_call_tile",
            )

    def test_scheduler_candidate_receives_full_batch_workload(self):
        _model, _, convolution = _make_int8_output_channels_graph(
            32,
            batch_size=2,
        )
        candidate_fits = Mock(return_value=False)
        providers = replace(
            GENERIC_SCHEDULER_PROVIDERS,
            gemm_tiling_candidate_fits=candidate_fits,
        )

        with patch(
            "voyager_compiler.codegen.passes.tiling."
            "get_target_scheduler_providers",
            return_value=providers,
        ):
            search_conv2d_tiling(
                convolution,
                (16, 32),
                1 << 30,
                None,
                None,
                None,
                hardware_config=TILING_HARDWARE_CONFIG,
            )

        self.assertTrue(candidate_fits.called)
        for call in candidate_fits.call_args_list:
            candidate, full, hardware, input_bits, can_partition = call.args
            self.assertEqual(candidate.nimg, 2)
            self.assertEqual(full.nimg, 2)
            self.assertEqual(full.nofm, 32)
            self.assertIs(hardware, TILING_HARDWARE_CONFIG)
            self.assertEqual(input_bits, 8)
            self.assertIsInstance(can_partition, bool)

    def test_shared_k_reconstruction_stack_is_not_a_target_nop(self):
        root = torch.nn.Module()
        root.register_buffer("weight", torch.randn(4, 2, 1, 1))
        graph = Graph()
        input_node = graph.placeholder("input")
        weight = graph.get_attr("weight")
        convolution = graph.call_function(
            torch.ops.aten.conv2d.default,
            (input_node, weight),
        )
        relu = graph.call_function(torch.ops.aten.relu.default, (convolution,))
        graph.output(relu)
        model = GraphModule(root, graph)
        ShapeProp(model).propagate(torch.randn(1, 2, 2, 2))
        input_node.meta["dtype"] = "float32"
        weight.meta["dtype"] = "float32"
        convolution.meta["dtype"] = "float32"
        relu.meta["dtype"] = "float32"
        traits = TargetOperatorTraits(
            conv2d_tiling_epilogue_operators=(torch.ops.aten.relu.default,),
        )

        split_conv2d_node(
            model,
            convolution,
            (2, 2, 2, 2),
            DEFAULT_COMPILATION_TARGET,
            traits,
        )

        stacks = [
            node
            for node in model.graph.nodes
            if node.target == torch.ops.aten.stack.default
        ]
        self.assertEqual(len(stacks), 1)
        self.assertFalse(stacks[0].meta.get(TARGET_INSTRUCTION_NOP, False))

    def test_scheduler_uses_current_fused_arguments_after_rewrite(self):
        model, current_input, weight, output, conv = (
            _make_rewritten_fused_conv()
        )
        set_physical_quantization_contract(
            current_input,
            PhysicalQuantizationContract("int4"),
        )
        weight.meta["dtype"] = "int8"
        tiled_shapes = {
            current_input: (1, 16, 12, 12),
            weight: (32, 16, 1, 1),
            output: (1, 32, 12, 12),
        }

        operands = _get_scheduler_operands(
            output,
            conv,
            tiled_shapes,
            dict(model.named_modules()),
        )

        shapes = {operand.port: operand.shape for operand in operands}
        widths = {operand.port: operand.bits for operand in operands}
        self.assertEqual(shapes["input"], (1, 16, 12, 12))
        self.assertEqual(shapes["weight"], (32, 16, 1, 1))
        self.assertEqual(widths["input"], 4)
        self.assertEqual(widths["weight"], 8)

    def test_interstellar_gates_rgb_and_grayscale_first_convs(self):
        for input_channels in (1, 3):
            with self.subTest(input_channels=input_channels):
                _, _, regular = _make_conv_graph(torch.ops.aten.conv2d.default)
                layer = extract_gemm_workload(
                    regular,
                    (
                        OperationOperand(
                            "input",
                            (1, input_channels, 192, 192),
                            8,
                        ),
                        OperationOperand(
                            "weight",
                            (16, input_channels, 3, 3),
                            8,
                        ),
                    ),
                    OperationOperand(
                        "output",
                        (1, 16, 96, 96),
                        8,
                    ),
                )
                self.assertIsNone(layer)

    def test_depthwise_convolutions_are_gated(self):
        for target in CONV_TARGETS:
            with self.subTest(target=target):
                model, depthwise, regular = _make_conv_graph(target)
                _, grouped = _make_grouped_conv_node(target)

                self.assertTrue(is_depthwise_conv(depthwise))
                self.assertFalse(is_depthwise_conv(regular))
                self.assertFalse(is_depthwise_conv(grouped))

                with patch(
                    "voyager_compiler.codegen.passes.tiling."
                    "search_conv2d_tiling",
                    return_value=(None, None),
                ) as search:
                    with self.assertLogs(
                        "voyager_compiler.codegen.passes.tiling",
                        level="INFO",
                    ) as logs:
                        run_matrix_op_l2_tiling(
                            model,
                            unroll=(4, 4),
                            cache_size=1024,
                        )

                search.assert_called_once()
                self.assertIs(search.call_args.args[0], regular)
                self.assertIn("depthwise convolution", " ".join(logs.output))

    def test_provider_sets_are_immutable_and_validate_defaults(self):
        self.assertIsNone(
            GENERIC_SCHEDULER_PROVIDERS.conv2d_tiling_min_spatial_extent
        )
        self.assertIsNone(
            MAGNET_ALPHA_SCHEDULER_PROVIDERS.conv2d_tiling_min_spatial_extent
        )
        with self.assertRaises(TypeError):
            GENERIC_SCHEDULER_PROVIDERS.gemm_providers["future"] = object()
        with self.assertRaisesRegex(
            ValueError,
            "default scheduler backend",
        ):
            SchedulerProviderSet(
                default_backend="missing",
                gemm_providers={"present": lambda *_args: None},
            )
        for invalid in (0, -1, True, 1.5):
            with (
                self.subTest(invalid=invalid),
                self.assertRaisesRegex(
                    ValueError,
                    "minimum spatial extent must be a positive integer",
                ),
            ):
                SchedulerProviderSet(
                    default_backend="present",
                    gemm_providers={"present": lambda *_args: None},
                    conv2d_tiling_min_spatial_extent=invalid,
                )

    def test_unregistered_scheduler_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "not registered in the selected provider set",
        ):
            resolve_local_scheduler(
                "unregistered",
                providers=GENERIC_SCHEDULER_PROVIDERS,
            )


def _make_cat_graph():
    graph = Graph()
    input1 = graph.placeholder("input1")
    input2 = graph.placeholder("input2")
    cat = graph.call_function(
        torch.ops.aten.cat.default,
        ((input1, input2), 1),
    )
    graph.output(cat)
    return GraphModule(torch.nn.Module(), graph)


class CatLoweringTest(unittest.TestCase):
    def _lower(self, batch_size):
        model = _make_cat_graph()
        input1 = torch.randn(batch_size, 4, 3, 2)
        input2 = torch.randn(batch_size, 4, 3, 2)
        expected = model(input1, input2)

        ShapeProp(model).propagate(input1, input2)
        convert_cat_and_stack_as_stack_on_dim0(model)
        actual = model(input1, input2)
        torch.testing.assert_close(actual, expected)
        return model

    def test_singleton_prefix_elides_permute(self):
        model = self._lower(batch_size=1)
        targets = [node.target for node in model.graph.nodes]

        self.assertIn(torch.ops.aten.stack.default, targets)
        self.assertIn(torch.ops.aten.reshape.default, targets)
        self.assertNotIn(torch.ops.aten.cat.default, targets)
        self.assertNotIn(torch.ops.aten.permute.default, targets)
        self.assertNotIn(torch.ops.aten.flatten.using_ints, targets)

    def test_non_singleton_prefix_retains_permute(self):
        model = self._lower(batch_size=2)
        targets = [node.target for node in model.graph.nodes]

        self.assertIn(torch.ops.aten.stack.default, targets)
        self.assertIn(torch.ops.aten.permute.default, targets)
        self.assertIn(torch.ops.aten.flatten.using_ints, targets)
        self.assertNotIn(torch.ops.aten.cat.default, targets)


if __name__ == "__main__":
    unittest.main()

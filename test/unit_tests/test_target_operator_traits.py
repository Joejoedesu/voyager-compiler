# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import torch
from torch.fx import Graph, GraphModule

import voyager_compiler.codegen.target_backends as target_backends
import voyager_compiler.codegen.targets as compilation_targets
from scheduler.dispatch import GENERIC_SCHEDULER_PROVIDERS
from scheduler.types import SchedulerProviderSet
from voyager_compiler import transform
from voyager_compiler.codegen.hardware import HardwareConfig
from voyager_compiler.codegen.mapping_utils import is_conv2d, is_gemm_op
from voyager_compiler.codegen.operands import require_allocation
from voyager_compiler.codegen.passes.tiling import (
    _decompose_conv2d_node,
    _make_conv2d_tiled_module,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.operators import (
    INT8_CONV2D_ACCUMULATOR_OPERATOR,
    NMP_FIXED_REQUANTIZE_OPERATOR,
    PPU_FIXED_RELU_OPERATOR,
    PVS_QUANTIZE_OPERATOR,
    SATURATING_ADD_OPERATOR,
    VSQ_CONV2D_OPERATOR,
)
from voyager_compiler.codegen.target_backends.magnet_alpha.hardware_pipeline import (
    MAGNET_ALPHA_HARDWARE_PIPELINE,
)
from voyager_compiler.codegen.target_backends import (
    TargetBackendSpec,
    TargetOperatorTraits,
    allocate_target_operation_scope,
    apply_target_graph_rewrites,
    apply_target_memory_rewrites,
    get_target_execution_fusion_patterns,
    get_target_hardware_pipeline,
    get_target_operator_traits,
    run_target_tiled_conv2d,
    target_has_operation_scope,
    target_supports_operation_scoped_scratchpad,
    target_uses_generic_aggregate_copies,
    validate_target_end_to_end_scratchpad_operations,
    validate_target_operation_scopes,
)
from voyager_compiler.codegen.targets import (
    COMPILATION_TARGET_META,
    DEFAULT_COMPILATION_TARGET,
    MAGNET_ALPHA_COMPILATION_TARGET,
)
from voyager_compiler.fake_quantize import get_quantization_map
from voyager_compiler.llm_utils import fuse_dequantize_quantize
from voyager_compiler.codegen.hardware import OpMatcher
from voyager_compiler.quantization_contracts import (
    PreparedDomainInputFormat,
    PreparedDomainRelation,
    PreparedDomainTransferRule,
    QuantizationContractTransferGraph,
)
from voyager_compiler.quantize_pt2e import (
    _bind_prepared_domain_target_rules,
    _effective_prepared_domain_rule_set,
    _prepared_domain_data_inputs,
    finalize_pt2e_calibration,
)

_TEST_TARGET = "unit_test_target"
_SECOND_TEST_TARGET = "second_unit_test_target"
_TEST_OPERATOR = torch.ops.aten.add.Tensor
_NON_ALLOCATING_META = "unit_test_non_allocating"


class _TestHardwareConfig(HardwareConfig):
    pass


def _identity_hook(model, *args, **kwargs):
    return model


def _create_allocator(hardware):
    return object()


def _run_tiled_conv2d(*args, **kwargs):
    return torch.ones(1)


def _test_traits():
    return TargetOperatorTraits(
        gemm_operators=(_TEST_OPERATOR,),
        conv2d_operators=(_TEST_OPERATOR,),
        prepared_domain_transfer_rules=(
            PreparedDomainTransferRule(
                target=_TEST_OPERATOR,
                input_format=PreparedDomainInputFormat.PREFIX,
                input_count=2,
                relation=PreparedDomainRelation.JOIN,
                region_root=True,
                forcing_root=True,
            ),
        ),
        non_allocating_operand_meta=(_NON_ALLOCATING_META,),
        tiled_conv2d_operators=(_TEST_OPERATOR,),
        operand_role_overrides={"code": "scale"},
    )


def _test_backend(target=_TEST_TARGET, **overrides):
    backend = TargetBackendSpec(
        target=target,
        operator_traits=_test_traits(),
        hardware_config_type=_TestHardwareConfig,
        scheduler_providers=GENERIC_SCHEDULER_PROVIDERS,
        apply_graph_rewrites=_identity_hook,
        apply_quantization_rewrites=_identity_hook,
        apply_memory_rewrites=_identity_hook,
        prepare_quantization=_identity_hook,
        finalize_quantization_calibration=_identity_hook,
        create_scratchpad_allocator=_create_allocator,
        allow_qdq_fusion=True,
        supports_bufferized=False,
        uses_generic_aggregate_copies=False,
        run_tiled_conv2d=_run_tiled_conv2d,
    )
    return replace(backend, **overrides)


def _test_hardware(target=_TEST_TARGET):
    return _TestHardwareConfig.from_legacy(
        unroll_dims=(1, 1),
        cache_size=64,
        input_buffer_size=1,
        weight_buffer_size=1,
        accum_buffer_size=1,
        target=target,
    )


def _untyped_test_hardware():
    return HardwareConfig.from_legacy(
        unroll_dims=(1, 1),
        cache_size=64,
        input_buffer_size=1,
        weight_buffer_size=1,
        accum_buffer_size=1,
        target=_TEST_TARGET,
    )


def _registry_patch(*backends):
    registry = target_backends._build_backend_registry(*backends)
    return patch.object(target_backends, "_TARGET_BACKENDS", registry)


def _call_node(target):
    graph = Graph()
    input_node = graph.placeholder("input")
    weight = graph.placeholder("weight")
    operation = graph.call_function(target, (input_node, weight))
    graph.output(operation)
    return operation


def _identity_model():
    graph = Graph()
    input_node = graph.placeholder("input")
    graph.output(input_node)
    return GraphModule(torch.nn.Module(), graph)


def _make_static_qdq_graph():
    root = torch.nn.Module()
    root.register_buffer(
        "payload",
        torch.arange(8, dtype=torch.float32).reshape(4, 2),
    )
    root.register_buffer("dq_scale", torch.full((2, 2), 0.5))
    root.register_buffer("q_scale", torch.full((2, 2), 0.25))
    root.register_buffer("qmap", get_quantization_map("int8"))

    graph = Graph()
    payload = graph.get_attr("payload")
    dq_scale = graph.get_attr("dq_scale")
    dequantize = graph.call_function(
        torch.ops.quantized_ops.dequantize.default,
        (payload, dq_scale, None, (0,), 2, None, None),
    )
    layout = graph.call_function(
        torch.ops.aten.transpose.int,
        (dequantize, 0, 1),
    )
    q_scale = graph.get_attr("q_scale")
    qmap = graph.get_attr("qmap")
    quantize = graph.call_function(
        torch.ops.quantized_ops.quantize.default,
        (layout, q_scale, None, (1,), 2, qmap, None),
    )
    graph.output(quantize)
    return GraphModule(root, graph)


class MagnetAlphaHardwarePipelineTest(unittest.TestCase):
    def test_hardware_pipeline_is_target_owned_op_matcher_inventory(self):
        pipeline = get_target_hardware_pipeline(MAGNET_ALPHA_COMPILATION_TARGET)
        self.assertEqual(pipeline, MAGNET_ALPHA_HARDWARE_PIPELINE)
        self.assertEqual(
            tuple(
                tuple(matcher.targets for matcher in pattern)
                for pattern in pipeline
            ),
            (
                (
                    (INT8_CONV2D_ACCUMULATOR_OPERATOR,),
                    (PPU_FIXED_RELU_OPERATOR,),
                    (SATURATING_ADD_OPERATOR,),
                ),
                (
                    (
                        INT8_CONV2D_ACCUMULATOR_OPERATOR,
                        VSQ_CONV2D_OPERATOR,
                    ),
                    (PPU_FIXED_RELU_OPERATOR,),
                    (PVS_QUANTIZE_OPERATOR,),
                ),
                (
                    (
                        INT8_CONV2D_ACCUMULATOR_OPERATOR,
                        VSQ_CONV2D_OPERATOR,
                    ),
                    (PPU_FIXED_RELU_OPERATOR,),
                ),
                (
                    (INT8_CONV2D_ACCUMULATOR_OPERATOR,),
                    (NMP_FIXED_REQUANTIZE_OPERATOR,),
                    (PVS_QUANTIZE_OPERATOR,),
                ),
                (
                    (INT8_CONV2D_ACCUMULATOR_OPERATOR,),
                    (NMP_FIXED_REQUANTIZE_OPERATOR,),
                ),
            ),
        )
        vsq = _call_node(VSQ_CONV2D_OPERATOR)
        self.assertFalse(pipeline[0][0].matches(vsq))
        self.assertTrue(pipeline[1][0].matches(vsq))

    def test_generic_fusion_patterns_are_derived_exactly(self):
        expected = tuple(
            tuple(matcher.targets for matcher in pattern)
            for pattern in MAGNET_ALPHA_HARDWARE_PIPELINE
        )

        self.assertEqual(
            get_target_execution_fusion_patterns(
                MAGNET_ALPHA_COMPILATION_TARGET
            ),
            expected,
        )


class TargetOperatorTraitSelectionTest(unittest.TestCase):
    def setUp(self):
        self.backend = _test_backend()
        self.generic_traits = get_target_operator_traits(
            DEFAULT_COMPILATION_TARGET
        )

    def test_mapping_recognition_requires_selected_traits(self):
        operation = _call_node(_TEST_OPERATOR)

        self.assertFalse(is_gemm_op(operation, self.generic_traits))
        self.assertFalse(is_conv2d(operation, self.generic_traits))
        with _registry_patch(self.backend):
            selected = get_target_operator_traits(_TEST_TARGET)

        self.assertTrue(is_gemm_op(operation, selected))
        self.assertTrue(is_conv2d(operation, selected))

    def test_operation_scope_hooks_are_an_all_or_none_contract(self):
        with self.assertRaisesRegex(
            ValueError,
            "requires has, allocate, and validate hooks",
        ):
            replace(
                self.backend,
                scratchpad_has_operation_scope=Mock(return_value=True),
            )

    def test_hardware_pipeline_and_required_execution_patterns_are_distinct(
        self,
    ):
        pipeline = ((OpMatcher(_TEST_OPERATOR),),)
        required_patterns = (((_TEST_OPERATOR,),),)
        backend = replace(
            self.backend,
            hardware_pipeline=pipeline,
            execution_fusion_patterns=required_patterns,
        )

        with _registry_patch(backend):
            self.assertEqual(
                get_target_hardware_pipeline(_TEST_TARGET),
                pipeline,
            )
            self.assertEqual(
                get_target_execution_fusion_patterns(_TEST_TARGET),
                required_patterns,
            )

    def test_operation_scope_hooks_dispatch_through_backend_spec(self):
        allocator = object()
        operation = _call_node(_TEST_OPERATOR)
        model = _identity_model()
        validate_end_to_end = Mock()
        has_scope = Mock(return_value=True)
        segment = object()
        allocate_scope = Mock(return_value={operation: segment})
        validate_scopes = Mock()
        backend = replace(
            self.backend,
            scratchpad_validate_end_to_end_operations=validate_end_to_end,
            scratchpad_has_operation_scope=has_scope,
            scratchpad_allocate_operation_scope=allocate_scope,
            scratchpad_validate_operation_scopes=validate_scopes,
        )

        with _registry_patch(backend):
            self.assertTrue(
                target_supports_operation_scoped_scratchpad(_TEST_TARGET)
            )
            self.assertTrue(
                target_has_operation_scope(
                    _TEST_TARGET,
                    allocator,
                    operation,
                )
            )
            allocation = allocate_target_operation_scope(
                _TEST_TARGET,
                allocator,
                operation,
            )
            validate_target_operation_scopes(
                _TEST_TARGET,
                allocator,
                model,
            )
            validate_target_end_to_end_scratchpad_operations(
                _TEST_TARGET,
                allocator,
                model,
            )

        self.assertEqual(allocation, {operation: segment})
        has_scope.assert_called_once_with(allocator, operation)
        allocate_scope.assert_called_once_with(allocator, operation)
        validate_scopes.assert_called_once_with(allocator, model)
        validate_end_to_end.assert_called_once_with(allocator, model)

    def test_tiled_convolution_dispatch_requires_selected_target(self):
        with self.assertRaisesRegex(
            ValueError,
            "does not support tiled convolution",
        ):
            run_target_tiled_conv2d(
                DEFAULT_COMPILATION_TARGET,
                _TEST_OPERATOR,
                (),
                (),
                None,
                32,
            )

    def test_tiled_convolution_dispatches_selected_callback(self):
        callback = Mock(return_value=torch.ones(1))
        backend = replace(self.backend, run_tiled_conv2d=callback)

        with _registry_patch(backend):
            result = run_target_tiled_conv2d(
                _TEST_TARGET,
                _TEST_OPERATOR,
                ("args",),
                ("scales",),
                "folded",
                32,
            )

        torch.testing.assert_close(result, torch.ones(1))
        callback.assert_called_once_with(
            _TEST_OPERATOR,
            ("args",),
            ("scales",),
            "folded",
            32,
            (None, None),
            {},
        )

    def test_tiled_convolution_dispatch_preserves_operator_kwargs(self):
        callback = Mock(return_value=torch.ones(1))
        backend = replace(self.backend, run_tiled_conv2d=callback)
        endpoint_kwargs = {"semantic_endpoint": 8.0}

        with _registry_patch(backend):
            result = run_target_tiled_conv2d(
                _TEST_TARGET,
                _TEST_OPERATOR,
                ("args",),
                ("scales",),
                "folded",
                32,
                operator_kwargs=endpoint_kwargs,
            )

        torch.testing.assert_close(result, torch.ones(1))
        callback.assert_called_once_with(
            _TEST_OPERATOR,
            ("args",),
            ("scales",),
            "folded",
            32,
            (None, None),
            endpoint_kwargs,
        )
        dispatched_kwargs = callback.call_args.args[-1]
        self.assertIsNot(dispatched_kwargs, endpoint_kwargs)
        with self.assertRaises(TypeError):
            dispatched_kwargs["semantic_endpoint"] = 7.0

    def test_tiled_convolution_closes_over_selected_target(self):
        tiled_module = _make_conv2d_tiled_module(
            1,
            1,
            1,
            1,
            0,
            False,
            False,
            True,
            _TEST_OPERATOR,
            _TEST_TARGET,
            (),
        )()
        expected = torch.ones(1, 1, 1, 1)

        with patch(
            "voyager_compiler.codegen.passes.tiling." "run_target_tiled_conv2d",
            return_value=expected,
        ) as dispatch:
            actual = tiled_module.run_conv(
                torch.ones(1, 1, 1, 1),
                torch.ones(1, 1, 1, 1),
                None,
                {"conv_padding": (0, 0)},
                (torch.ones(1), torch.ones(1)),
                (None, None),
                None,
                (None, None),
            )

        self.assertIs(actual, expected)
        self.assertEqual(
            dispatch.call_args.args[:2],
            (_TEST_TARGET, _TEST_OPERATOR),
        )

    def test_tiled_convolution_forwards_operator_kwargs(self):
        configs = (
            {
                "c_tile": (0, 1),
                "k_tile": (0, 1),
                "y_tile": (0, 1),
                "x_tile": (0, 1),
                "input_padding": (0, 0, 0, 0),
                "conv_padding": (0, 0),
                "keep_dims_and_padding": False,
            },
            {
                "c_tile": (0, 1),
                "k_tile": (1, 2),
                "y_tile": (0, 1),
                "x_tile": (0, 1),
                "input_padding": (0, 0, 0, 0),
                "conv_padding": (0, 0),
                "keep_dims_and_padding": False,
            },
        )
        tiled_module = _make_conv2d_tiled_module(
            1,
            1,
            1,
            1,
            0,
            False,
            False,
            True,
            _TEST_OPERATOR,
            _TEST_TARGET,
            configs,
            K=2,
            tile_k=1,
        )()
        expected = torch.ones(1, 1, 1, 1)

        with patch(
            "voyager_compiler.codegen.passes.tiling." "run_target_tiled_conv2d",
            return_value=expected,
        ) as dispatch:
            actual = tiled_module(
                torch.ones(1, 1, 1, 1),
                torch.ones(2, 1, 1, 1),
                semantic_endpoint=8.0,
            )

        torch.testing.assert_close(actual, torch.ones(1, 2, 1, 1))
        self.assertEqual(dispatch.call_count, 2)
        for call in dispatch.call_args_list:
            self.assertEqual(
                call.kwargs["operator_kwargs"],
                {"semantic_endpoint": 8.0},
            )

    def test_tiled_convolution_prunes_unused_export_placeholders(self):
        graph = Graph()
        input_node = graph.placeholder("input")
        input_node.shape = (1, 1, 1, 1)
        input_node.value = torch.ones(input_node.shape)
        weight = graph.placeholder("weight")
        weight.shape = (1, 1, 1, 1)
        weight.value = torch.ones(weight.shape)
        scale = graph.placeholder("scale")
        scale.shape = (1,)
        scale.value = torch.ones(scale.shape)
        convolution = graph.call_function(
            _TEST_OPERATOR,
            (input_node, weight, None),
            {"scale": scale, "semantic_endpoint": 8.0},
        )
        convolution.shape = (1, 1, 1, 1)
        graph.output(convolution)
        model = GraphModule(torch.nn.Module(), graph)

        replacement_graph = Graph()
        replacement_input = replacement_graph.placeholder("input")
        replacement_weight = replacement_graph.placeholder("weight")
        replacement_graph.placeholder("bias")
        replacement_scale = replacement_graph.placeholder("scale")
        replacement_graph.placeholder("semantic_endpoint")
        replacement_graph.output(
            (replacement_input, replacement_weight, replacement_scale)
        )
        replacement = GraphModule(torch.nn.Module(), replacement_graph)

        def stop_after_placeholder_check(
            _model,
            _source,
            replacement_module,
            _value_remap,
        ):
            placeholders = [
                node.target
                for node in replacement_module.graph.nodes
                if node.op == "placeholder"
            ]
            self.assertEqual(placeholders, ["input", "weight", "scale"])
            raise RuntimeError("placeholder check complete")

        config = {
            "c_tile": (0, 1),
            "k_tile": (0, 1),
            "y_tile": (0, 1),
            "x_tile": (0, 1),
        }
        with (
            patch(
                "voyager_compiler.codegen.passes.tiling.export_model",
                return_value=replacement,
            ),
            patch(
                "voyager_compiler.codegen.passes.tiling."
                "replace_node_with_graph_module",
                side_effect=stop_after_placeholder_check,
            ),
            patch(
                "voyager_compiler.codegen.passes.tiling.is_depthwise_conv",
                return_value=False,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "placeholder check complete",
            ),
        ):
            _decompose_conv2d_node(
                model,
                convolution,
                (1, 1, 1, 1),
                [],
                (config,),
                _TEST_TARGET,
                _test_traits(),
            )

    def test_graph_rewrite_dispatch_runs_each_idempotent_call(self):
        model = _identity_model()
        rewrite = Mock(side_effect=lambda value: value)
        backend = replace(self.backend, apply_graph_rewrites=rewrite)
        hardware = _test_hardware()

        with _registry_patch(backend):
            apply_target_graph_rewrites(model, _TEST_TARGET, hardware)
            apply_target_graph_rewrites(model, _TEST_TARGET, hardware)

        self.assertEqual(rewrite.call_count, 2)
        rewrite.assert_called_with(model)

    def test_memory_rewrite_dispatch_receives_selected_hardware(self):
        model = _identity_model()
        rewrite = Mock(return_value=model)
        backend = replace(self.backend, apply_memory_rewrites=rewrite)
        hardware = _test_hardware()

        with _registry_patch(backend):
            result = apply_target_memory_rewrites(
                model,
                _TEST_TARGET,
                hardware,
            )

        self.assertIs(result, model)
        rewrite.assert_called_once_with(model, hardware=hardware)

    def test_allocation_metadata_requires_selected_traits(self):
        graph = Graph()
        value = graph.placeholder("value")
        value.meta[_NON_ALLOCATING_META] = True

        self.assertTrue(require_allocation(value, self.generic_traits))
        self.assertFalse(
            require_allocation(value, self.backend.operator_traits)
        )

    def test_prepared_domain_rules_require_selected_traits(self):
        graph = Graph()
        input1 = graph.placeholder("input1")
        input2 = graph.placeholder("input2")
        operation = graph.call_function(
            _TEST_OPERATOR,
            (input1, input2),
        )
        graph.output(operation)
        model = GraphModule(torch.nn.Module(), graph)

        _bind_prepared_domain_target_rules(model, self.generic_traits)
        self.assertEqual(_prepared_domain_data_inputs(operation), ())

        _bind_prepared_domain_target_rules(
            model,
            self.backend.operator_traits,
        )
        self.assertEqual(
            _prepared_domain_data_inputs(operation),
            (input1, input2),
        )

    def test_selected_rules_compose_with_shared_domain_rules(self):
        graph = Graph()
        input1 = graph.placeholder("input1")
        input2 = graph.placeholder("input2")
        operation = graph.call_function(
            _TEST_OPERATOR,
            (input1, input2),
        )
        reshape = graph.call_function(
            torch.ops.aten.reshape.default,
            (operation, (1,)),
        )
        graph.output(reshape)

        generic_graph = QuantizationContractTransferGraph(
            graph.nodes,
            _effective_prepared_domain_rule_set(self.generic_traits),
        )
        selected_graph = QuantizationContractTransferGraph(
            graph.nodes,
            _effective_prepared_domain_rule_set(self.backend.operator_traits),
        )

        self.assertFalse(generic_graph.is_node(operation))
        self.assertEqual(
            selected_graph.connected_component(reshape),
            frozenset((operation, reshape)),
        )

    def test_ordinary_cat_uses_shared_non_forcing_rule(self):
        graph = Graph()
        input1 = graph.placeholder("input1")
        input2 = graph.placeholder("input2")
        cat = graph.call_function(
            torch.ops.aten.cat.default,
            ((input1, input2), 1),
        )
        graph.output(cat)

        domain_graph = QuantizationContractTransferGraph(
            graph.nodes,
            _effective_prepared_domain_rule_set(self.backend.operator_traits),
        )
        transfer = domain_graph.transfer(cat)

        self.assertIsNotNone(transfer)
        self.assertEqual(transfer.inputs, (input1, input2))
        self.assertEqual(
            transfer.rule.relation,
            PreparedDomainRelation.JOIN,
        )
        self.assertTrue(transfer.rule.region_root)
        self.assertFalse(transfer.rule.forcing_root)

    def test_explicit_none_zero_point_is_preserved(self):
        model = _make_static_qdq_graph()
        expected = model().clone()

        fuse_dequantize_quantize(model)

        torch.testing.assert_close(model(), expected)
        targets = [
            node.target
            for node in model.graph.nodes
            if node.op == "call_function"
        ]
        self.assertNotIn(
            torch.ops.quantized_ops.quantize.default,
            targets,
        )
        dequantize = next(
            node
            for node in model.graph.nodes
            if node.target == torch.ops.quantized_ops.dequantize.default
        )
        self.assertIsNone(dequantize.args[2])

    def test_operator_claims_are_qualified_by_selected_target(self):
        first = self.backend
        second = _test_backend(target=_SECOND_TEST_TARGET)
        backends = target_backends._build_backend_registry(first, second)

        self.assertIn(
            _TEST_OPERATOR,
            backends[_TEST_TARGET].operator_traits.gemm_operators,
        )
        self.assertIn(
            _TEST_OPERATOR,
            backends[_SECOND_TEST_TARGET].operator_traits.gemm_operators,
        )

    def test_scheduler_providers_are_selected_by_registered_target(self):
        providers = SchedulerProviderSet(
            default_backend="unit_scheduler",
            gemm_providers={"unit_scheduler": _identity_hook},
        )
        backend = _test_backend(scheduler_providers=providers)

        with _registry_patch(backend):
            self.assertIsNone(
                target_backends.get_target_backend_spec(
                    DEFAULT_COMPILATION_TARGET
                )
            )
            self.assertIs(
                target_backends.get_target_backend_spec(_TEST_TARGET),
                backend,
            )
            self.assertIs(
                target_backends.get_target_scheduler_providers(
                    DEFAULT_COMPILATION_TARGET
                ),
                GENERIC_SCHEDULER_PROVIDERS,
            )
            self.assertIs(
                target_backends.get_target_scheduler_providers(_TEST_TARGET),
                providers,
            )

    def test_scheduler_classification_uses_only_selected_target(self):
        classification = object()
        classifier = Mock(return_value=classification)
        backend = _test_backend(
            classify_scheduler_operation=classifier,
        )
        operands = (object(), object())

        with _registry_patch(backend):
            self.assertIsNone(
                target_backends.classify_target_scheduler_operation(
                    _TEST_OPERATOR,
                    DEFAULT_COMPILATION_TARGET,
                    operands,
                )
            )
            self.assertIs(
                target_backends.classify_target_scheduler_operation(
                    _TEST_OPERATOR,
                    _TEST_TARGET,
                    operands,
                ),
                classification,
            )

        classifier.assert_called_once_with(_TEST_OPERATOR, operands)

    def test_backend_registry_is_immutable_and_rejects_collisions(self):
        registry = target_backends._build_backend_registry(self.backend)
        with self.assertRaises(TypeError):
            registry[_SECOND_TEST_TARGET] = self.backend

        duplicate = replace(
            self.backend,
            operator_traits=TargetOperatorTraits(),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "Duplicate compilation target backend",
        ):
            target_backends._build_backend_registry(
                self.backend,
                duplicate,
            )

    def test_registered_operator_traits_are_deeply_immutable(self):
        with self.assertRaises(TypeError):
            self.backend.operator_traits.operand_role_overrides["new_role"] = (
                "input"
            )

    def test_backend_spec_rejects_missing_required_callback(self):
        with self.assertRaisesRegex(
            TypeError,
            "apply_graph_rewrites must be callable",
        ):
            replace(self.backend, apply_graph_rewrites=None)

    def test_aggregate_copy_policy_is_registered_per_target(self):
        self.assertTrue(
            target_uses_generic_aggregate_copies(DEFAULT_COMPILATION_TARGET)
        )
        with _registry_patch(self.backend):
            self.assertFalse(target_uses_generic_aggregate_copies(_TEST_TARGET))

    def test_low_level_dispatch_requires_matching_typed_hardware(self):
        model = _identity_model()
        with _registry_patch(self.backend):
            with self.assertRaisesRegex(
                TypeError,
                "_TestHardwareConfig",
            ):
                apply_target_graph_rewrites(model, _TEST_TARGET)
            with self.assertRaisesRegex(
                TypeError,
                "_TestHardwareConfig",
            ):
                apply_target_graph_rewrites(
                    model,
                    _TEST_TARGET,
                    _untyped_test_hardware(),
                )
            with self.assertRaisesRegex(
                ValueError,
                "does not match",
            ):
                apply_target_graph_rewrites(
                    model,
                    DEFAULT_COMPILATION_TARGET,
                    _test_hardware(),
                )

    def test_public_stages_reject_target_changes(self):
        model = _identity_model()
        with (
            _registry_patch(self.backend),
            patch.object(
                compilation_targets,
                "COMPILATION_TARGETS",
                (DEFAULT_COMPILATION_TARGET, _TEST_TARGET),
            ),
        ):
            finalize_pt2e_calibration(model, _test_hardware())

        self.assertEqual(
            model.meta[COMPILATION_TARGET_META],
            _TEST_TARGET,
        )
        with self.assertRaisesRegex(
            ValueError,
            "model is bound to compilation target",
        ):
            transform(
                model,
                (torch.ones(1),),
                hardware_config=None,
            )

    def test_invalid_hardware_does_not_bind_model_provenance(self):
        model = _identity_model()
        with (
            _registry_patch(self.backend),
            patch.object(
                compilation_targets,
                "COMPILATION_TARGETS",
                (DEFAULT_COMPILATION_TARGET, _TEST_TARGET),
            ),
        ):
            with self.assertRaisesRegex(
                TypeError,
                "_TestHardwareConfig",
            ):
                finalize_pt2e_calibration(
                    model,
                    _untyped_test_hardware(),
                )

        self.assertNotIn(COMPILATION_TARGET_META, model.meta)


if __name__ == "__main__":
    unittest.main()

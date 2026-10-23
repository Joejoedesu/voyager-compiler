# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Focused semantic-lowering checks for the Gemmini target."""

from __future__ import annotations

from collections import Counter

import torch
from torch.fx import Graph, GraphModule, Node
from torchvision.models import resnet18

from test.utils.models import resnet18_gemmini as adapter
from voyager_compiler import (
    convert_pt2e,
    export_model,
    finalize_pt2e_calibration,
    get_default_quantizer,
    prepare_pt2e,
    transform,
)
from voyager_compiler.codegen import GEMMINI_HARDWARE_CONFIG
from voyager_compiler.codegen.target_backends import (
    get_target_execution_fusion_patterns,
    get_target_hardware_pipeline,
)
from voyager_compiler.codegen.target_backends.gemmini.hardware_pipeline import (
    GEMMINI_HARDWARE_PIPELINE,
)
from voyager_compiler.codegen.target_backends.gemmini.lowering import (
    GemminiInjectionError,
    gemmini_workload_add_regions,
    gemmini_workload_average_regions,
    gemmini_workload_linear_regions,
    gemmini_workload_regions,
)
from voyager_compiler.codegen.target_backends.gemmini.quantization import (
    _fuse_final_residual_boundary,
    _legalize_quantized_value_relu,
)
from voyager_compiler.codegen.target_backends.gemmini.serialization import (
    GemminiSerializationError,
    validate_gemmini_pipeline_qmaps,
)
from voyager_compiler.fake_quantize import get_quantization_map

_QUANTIZE = torch.ops.quantized_ops.quantize.default
_DEQUANTIZE = torch.ops.quantized_ops.dequantize.default
_FLOAT_RELU = torch.ops.aten.relu_.default
_CODE_RELU = torch.ops.aten.relu.default


def _relu_qdq_graph(
    *,
    intermediate_scale: float = 0.25,
    dequantize_scale: float = 0.25,
    final_scale: float = 0.25,
    dequantize_input_qmap: bool = False,
    intermediate_rounding: str = "nearest",
    intermediate_output_code: bool = False,
) -> GraphModule:
    root = torch.nn.Module()
    root.register_buffer(
        "intermediate_scale",
        torch.tensor([intermediate_scale], dtype=torch.float32),
    )
    root.register_buffer(
        "dequantize_scale",
        torch.tensor([dequantize_scale], dtype=torch.float32),
    )
    root.register_buffer(
        "final_scale",
        torch.tensor([final_scale], dtype=torch.float32),
    )
    root.register_buffer("qmap", get_quantization_map("int8"))

    graph = Graph()
    left = graph.placeholder("left")
    right = graph.placeholder("right")
    producer = graph.call_function(torch.ops.aten.add_.Tensor, (left, right))
    qmap = graph.get_attr("qmap")
    first_scale = graph.get_attr("intermediate_scale")
    first_quantize = graph.call_function(
        _QUANTIZE,
        (
            producer,
            first_scale,
            None,
            None,
            None,
            qmap,
            qmap if intermediate_output_code else None,
            intermediate_rounding,
            -128.0,
            127.0,
        ),
    )
    dq_scale = graph.get_attr("dequantize_scale")
    input_qmap: Node | None = qmap if dequantize_input_qmap else None
    dequantize = graph.call_function(
        _DEQUANTIZE,
        (
            first_quantize,
            dq_scale,
            None,
            None,
            None,
            input_qmap,
            None,
        ),
    )
    activation = graph.call_function(_FLOAT_RELU, (dequantize,))
    output_scale = graph.get_attr("final_scale")
    output_quantize = graph.call_function(
        _QUANTIZE,
        (
            activation,
            output_scale,
            None,
            None,
            None,
            qmap,
            None,
            "nearest",
            -128.0,
            127.0,
        ),
    )
    graph.output(output_quantize)
    return GraphModule(root, graph)


def _terminal_residual_graph(
    *,
    dequantize_scale: float = 0.25,
    zero_point: float | None = None,
    dequantize_input_qmap: bool = False,
    quantize_output_code: bool = False,
    preexisting_code_relu: bool = False,
) -> GraphModule:
    root = torch.nn.Module()
    root.register_buffer(
        "quantize_scale",
        torch.tensor([0.25], dtype=torch.float32),
    )
    root.register_buffer(
        "dequantize_scale",
        torch.tensor([dequantize_scale], dtype=torch.float32),
    )
    root.register_buffer("qmap", get_quantization_map("int8"))
    if zero_point is not None:
        root.register_buffer(
            "zero_point",
            torch.tensor([zero_point], dtype=torch.float32),
        )

    graph = Graph()
    left = graph.placeholder("left")
    right = graph.placeholder("right")
    addition = graph.call_function(torch.ops.aten.add_.Tensor, (left, right))
    quantize_scale = graph.get_attr("quantize_scale")
    qmap = graph.get_attr("qmap")
    quantize_zero_point = (
        graph.get_attr("zero_point") if zero_point is not None else None
    )
    output_quantize = graph.call_function(
        _QUANTIZE,
        (
            addition,
            quantize_scale,
            quantize_zero_point,
            None,
            None,
            qmap,
            qmap if quantize_output_code else None,
            "nearest",
            -128.0,
            127.0,
        ),
    )
    if preexisting_code_relu:
        activation = graph.call_function(_CODE_RELU, (output_quantize,))
    else:
        dq_scale = graph.get_attr("dequantize_scale")
        input_qmap: Node | None = qmap if dequantize_input_qmap else None
        output_dequantize = graph.call_function(
            _DEQUANTIZE,
            (
                output_quantize,
                dq_scale,
                quantize_zero_point,
                None,
                None,
                input_qmap,
                None,
            ),
        )
        activation = graph.call_function(_FLOAT_RELU, (output_dequantize,))
    average = graph.call_function(
        torch.ops.aten.adaptive_avg_pool2d.default,
        (activation, [1, 1]),
    )
    graph.output(average)
    return GraphModule(root, graph)


def _call_targets(model: GraphModule) -> tuple[object, ...]:
    return tuple(
        node.target for node in model.graph.nodes if node.op == "call_function"
    )


def _malformed_fused_residual_graph() -> GraphModule:
    subgraph = Graph()
    left = subgraph.placeholder("left")
    right = subgraph.placeholder("right")
    wrong_input = subgraph.call_function(_QUANTIZE, (left,))
    right_dequantize = subgraph.call_function(_DEQUANTIZE, (right,))
    addition = subgraph.call_function(
        torch.ops.aten.add_.Tensor,
        (wrong_input, right_dequantize),
    )
    output_quantize = subgraph.call_function(_QUANTIZE, (addition,))
    activation = subgraph.call_function(_CODE_RELU, (output_quantize,))
    subgraph.output(activation)

    root = torch.nn.Module()
    root.add_module("fused", GraphModule(torch.nn.Module(), subgraph))
    graph = Graph()
    outer_left = graph.placeholder("left")
    outer_right = graph.placeholder("right")
    fused = graph.call_module("fused", (outer_left, outer_right))
    graph.output(fused)
    return GraphModule(root, graph)


def _node_input(node: Node, index: int, name: str):
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(name)


def _bound_region_attr(
    model: GraphModule,
    region,
    node: Node,
    index: int,
    name: str,
) -> torch.Tensor:
    operand = _node_input(node, index, name)
    if isinstance(operand, Node) and operand.op == "placeholder":
        placeholders = tuple(
            candidate
            for candidate in region.submodule.graph.nodes
            if candidate.op == "placeholder"
        )
        operand = region.outer.args[placeholders.index(operand)]
    assert isinstance(operand, Node) and operand.op == "get_attr"
    value: object = model
    for field in str(operand.target).split("."):
        value = getattr(value, field)
    assert isinstance(value, torch.Tensor)
    return value


def _assert_corrupt_qmap_fails_closed(
    model: GraphModule,
    region,
    node: Node,
    *,
    index: int = 5,
    name: str = "qmap",
) -> None:
    qmap = _bound_region_attr(model, region, node, index, name)
    bits = qmap.view(torch.int16)
    original = int(bits[0].item())
    bits[0] = original ^ 1
    try:
        try:
            validate_gemmini_pipeline_qmaps(model)
        except GemminiSerializationError as error:
            assert "canonical" in str(error)
        else:
            raise AssertionError("noncanonical qmap was accepted")
    finally:
        bits[0] = original
    validate_gemmini_pipeline_qmaps(model)


def _assert_unrepresentable_dq_qmap_fails_closed(
    model: GraphModule,
    region,
    dequantize: Node,
) -> None:
    qmap = _node_input(region.output_quantize, 5, "qmap")
    assert isinstance(qmap, Node)
    original_args = dequantize.args
    original_kwargs = dequantize.kwargs
    if len(dequantize.args) > 5:
        args = list(dequantize.args)
        args[5] = qmap
        dequantize.args = tuple(args)
    else:
        kwargs = dict(dequantize.kwargs)
        kwargs["input_qmap"] = qmap
        dequantize.kwargs = kwargs
    try:
        try:
            validate_gemmini_pipeline_qmaps(model)
        except GemminiSerializationError as error:
            assert "qmaps are not supported" in str(error)
        else:
            raise AssertionError("unrepresentable DQ qmap was accepted")
    finally:
        dequantize.args = original_args
        dequantize.kwargs = original_kwargs
    validate_gemmini_pipeline_qmaps(model)


def test_required_execution_group_is_derived_from_conv_pipeline() -> None:
    # Required execution grouping retains only the pre-existing R anchor and
    # cannot accidentally treat a bare Linear or an extended W epilogue as a
    # mandatory fused record.
    assert get_target_execution_fusion_patterns("gemmini") == (
        (
            (_QUANTIZE,),
            (
                torch.ops.aten.conv2d.default,
                torch.ops.quantized_ops.conv2d.default,
            ),
            (_DEQUANTIZE,),
            (_QUANTIZE,),
        ),
    )


def test_hardware_pipeline_models_only_linear_helper_paths() -> None:
    assert get_target_hardware_pipeline("gemmini") == GEMMINI_HARDWARE_PIPELINE
    assert tuple(len(pattern) for pattern in GEMMINI_HARDWARE_PIPELINE) == (
        5,
        4,
        3,
        3,
        3,
    )

    graph = Graph()
    input_node = graph.placeholder("input")
    weight = graph.placeholder("weight")
    conv = graph.call_function(
        torch.ops.quantized_ops.conv2d.default,
        (input_node, weight),
    )
    assert GEMMINI_HARDWARE_PIPELINE[0][0].matches(conv)

    starts = tuple(pattern[0].targets for pattern in GEMMINI_HARDWARE_PIPELINE)
    assert all(_QUANTIZE not in targets for targets in starts)
    assert starts[-1] == (torch.ops.aten.adaptive_avg_pool2d.default,)
    flattened_targets = {
        target
        for pattern in GEMMINI_HARDWARE_PIPELINE
        for matcher in pattern
        for target in matcher.targets
    }
    assert torch.ops.aten.add_.Tensor not in flattened_targets
    assert torch.ops.aten.layer_norm.default not in flattened_targets


def test_residual_group_rejects_a_non_dequantized_input() -> None:
    model = _malformed_fused_residual_graph()
    try:
        gemmini_workload_add_regions(model)
    except GemminiInjectionError as error:
        assert "residual inputs must both be dequantize" in str(error)
    else:
        raise AssertionError("malformed residual input was accepted")


def test_equal_relu_qdq_contract_legalizes_exactly() -> None:
    model = _relu_qdq_graph()
    left = torch.tensor([-100.0, -0.375, 0.125, 0.375, 100.0])
    right = torch.tensor([0.0, 0.125, 0.125, 0.125, 0.0])
    reference = model(left.clone(), right.clone())

    assert _legalize_quantized_value_relu(model) == 1
    result = model(left.clone(), right.clone())

    assert torch.equal(result, reference)
    targets = _call_targets(model)
    assert targets.count(_QUANTIZE) == 1
    assert _DEQUANTIZE not in targets
    assert _FLOAT_RELU not in targets
    assert targets.count(_CODE_RELU) == 1


def test_terminal_residual_boundary_legalizes_exactly() -> None:
    model = _terminal_residual_graph()
    left = torch.tensor(
        [[[[-100.0, -0.375], [0.125, 100.0]]]], dtype=torch.float32
    )
    right = torch.tensor([[[[0.0, 0.125], [0.125, 0.0]]]], dtype=torch.float32)
    reference = model(left.clone(), right.clone())

    _fuse_final_residual_boundary(model)
    result = model(left.clone(), right.clone())

    assert torch.equal(result, reference)
    regions = gemmini_workload_add_regions(model)
    assert len(regions) == 1
    assert regions[0].member_roles == (
        "add",
        "output_quantize",
        "code_relu",
    )


def test_connected_residual_rejects_noncanonical_qmap() -> None:
    model = _terminal_residual_graph(preexisting_code_relu=True)
    _fuse_final_residual_boundary(model)
    (region,) = gemmini_workload_add_regions(model)
    assert region.member_roles == (
        "add",
        "output_quantize",
        "code_relu",
    )
    validate_gemmini_pipeline_qmaps(model)

    _assert_corrupt_qmap_fails_closed(
        model,
        region,
        region.output_quantize,
    )


def test_terminal_residual_boundary_rejects_nonzero_zero_point() -> None:
    model = _terminal_residual_graph(zero_point=10.0)
    original_graph = str(model.graph)

    _fuse_final_residual_boundary(model)

    assert str(model.graph) == original_graph
    assert not gemmini_workload_add_regions(model)


def test_terminal_residual_boundary_rejects_mismatched_dq_scale() -> None:
    model = _terminal_residual_graph(dequantize_scale=0.5)
    original_graph = str(model.graph)

    _fuse_final_residual_boundary(model)

    assert str(model.graph) == original_graph
    assert not gemmini_workload_add_regions(model)


def test_terminal_residual_boundary_rejects_noncanonical_qdq_maps() -> None:
    for model in (
        _terminal_residual_graph(dequantize_input_qmap=True),
        _terminal_residual_graph(quantize_output_code=True),
        _terminal_residual_graph(
            quantize_output_code=True,
            preexisting_code_relu=True,
        ),
    ):
        original_graph = str(model.graph)

        _fuse_final_residual_boundary(model)

        assert str(model.graph) == original_graph
        assert not gemmini_workload_add_regions(model)


def test_mismatched_final_relu_scale_fails_closed() -> None:
    model = _relu_qdq_graph(final_scale=0.5)
    original_graph = str(model.graph)

    assert _legalize_quantized_value_relu(model) == 0
    assert str(model.graph) == original_graph
    assert _DEQUANTIZE in _call_targets(model)


def test_mismatched_dequantize_scale_fails_closed() -> None:
    model = _relu_qdq_graph(dequantize_scale=0.5)
    original_graph = str(model.graph)

    assert _legalize_quantized_value_relu(model) == 0
    assert str(model.graph) == original_graph
    assert _DEQUANTIZE in _call_targets(model)


def test_noncanonical_dequantize_contract_fails_closed() -> None:
    model = _relu_qdq_graph(dequantize_input_qmap=True)
    original_graph = str(model.graph)

    assert _legalize_quantized_value_relu(model) == 0
    assert str(model.graph) == original_graph
    assert _DEQUANTIZE in _call_targets(model)


def test_noncanonical_intermediate_quantize_fails_closed() -> None:
    model = _relu_qdq_graph(intermediate_rounding="floor")
    original_graph = str(model.graph)

    assert _legalize_quantized_value_relu(model) == 0
    assert str(model.graph) == original_graph
    assert _DEQUANTIZE in _call_targets(model)


def test_intermediate_output_code_fails_closed() -> None:
    model = _relu_qdq_graph(intermediate_output_code=True)
    original_graph = str(model.graph)

    assert _legalize_quantized_value_relu(model) == 0
    assert str(model.graph) == original_graph
    assert _DEQUANTIZE in _call_targets(model)


def test_frozen_resnet18_alignment_and_region_partition() -> None:
    torch.manual_seed(adapter.SEED)
    source = resnet18(weights=None).eval()
    example = torch.empty(adapter.INPUT_SHAPE).uniform_(-1.0, 1.0)
    graph = export_model(source, (example,), strict=False)
    quantizer = get_default_quantizer(
        input_activation=adapter.QSPEC,
        output_activation=adapter.QSPEC,
        weight=adapter.QSPEC,
        bias=adapter.BIAS_QSPEC,
    )
    graph = prepare_pt2e(
        graph,
        quantizer,
        hardware_config=GEMMINI_HARDWARE_CONFIG,
        fold_conv_bn=True,
    )
    with torch.inference_mode():
        for _ in range(adapter.CALIBRATION_STEPS):
            graph(example)
    finalize_pt2e_calibration(
        graph,
        hardware_config=GEMMINI_HARDWARE_CONFIG,
    )
    assert (
        graph.meta["gemmini_relu_domain_alignment_count"]
        == adapter.RELU_DOMAIN_ALIGNMENT_COUNT
    )

    graph = convert_pt2e(
        graph,
        calibration_steps=adapter.CALIBRATION_STEPS,
        hardware_config=GEMMINI_HARDWARE_CONFIG,
    )
    with torch.inference_mode():
        reference = graph(example)
    graph = transform(
        graph,
        (example,),
        hardware_config=GEMMINI_HARDWARE_CONFIG,
        transform_layout=False,
    )
    with torch.inference_mode():
        transformed = graph(example)

    assert torch.equal(transformed, reference)
    conv_regions = gemmini_workload_regions(graph)
    add_regions = gemmini_workload_add_regions(graph)
    average_regions = gemmini_workload_average_regions(graph)
    linear_regions = gemmini_workload_linear_regions(graph)
    assert {
        "conv": len(conv_regions),
        "add": len(add_regions),
        "average": len(average_regions),
        "linear": len(linear_regions),
    } == {
        "conv": adapter.CONV_REGION_COUNT,
        "add": adapter.ADD_REGION_COUNT,
        "average": adapter.AVERAGE_REGION_COUNT,
        "linear": adapter.LINEAR_REGION_COUNT,
    }
    assert Counter(region.member_roles for region in conv_regions) == {
        (
            "input_quantize",
            "conv2d",
            "accumulator_dequantize",
            "output_quantize",
            "code_relu",
            "max_pool2d",
        ): 1,
        (
            "conv2d",
            "accumulator_dequantize",
            "output_quantize",
            "code_relu",
        ): 8,
        (
            "conv2d",
            "accumulator_dequantize",
            "output_quantize",
        ): 11,
    }
    assert Counter(region.member_roles for region in add_regions) == {
        (
            "input_a_dequantize",
            "input_b_dequantize",
            "add",
            "output_quantize",
            "code_relu",
        ): 8,
    }
    assert [region.member_roles for region in average_regions] == [
        (
            "average_input_dequantize",
            "global_average",
            "flatten",
            "output_quantize",
        )
    ]
    assert [region.member_roles for region in linear_regions] == [
        (
            "linear",
            "accumulator_dequantize",
            "output_quantize",
        )
    ]
    validate_gemmini_pipeline_qmaps(graph)

    # Qmaps disappear during generic protobuf serialization, so retain one
    # negative check for every helper family that owns a quantize stage.
    for region, quantize in (
        (conv_regions[0], conv_regions[0].output_quantize),
        (add_regions[0], add_regions[0].output_quantize),
        (average_regions[0], average_regions[0].output_quantize),
        (linear_regions[0], linear_regions[0].output_quantize),
    ):
        _assert_corrupt_qmap_fails_closed(graph, region, quantize)

    # Conv and Linear helper ABIs cannot express arbitrary accumulator maps.
    for region, dequantize in (
        (conv_regions[0], conv_regions[0].core_members[-2]),
        (linear_regions[0], linear_regions[0].accumulator_dequantize),
    ):
        _assert_corrupt_qmap_fails_closed(
            graph,
            region,
            dequantize,
            name="input_qmap",
        )

    # Residual-input and global-average DQs have no helper qmap operands.
    _assert_unrepresentable_dq_qmap_fails_closed(
        graph,
        add_regions[0],
        add_regions[0].input_boundaries[0],
    )
    _assert_unrepresentable_dq_qmap_fails_closed(
        graph,
        average_regions[0],
        average_regions[0].input_dequantize,
    )

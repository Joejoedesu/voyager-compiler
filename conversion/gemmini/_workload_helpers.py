#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Internal projections for non-Conv public Gemmini workload helpers."""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from google.protobuf import text_format

from conversion.gemmini import convert as anchor
from conversion.gemmini import workload

ADD_OPERATION_ID = "aten::add"
LINEAR_OPERATION_ID = "aten::linear"
AVERAGE_OPERATION_ID = "aten::adaptive_avg_pool2d"
NORM_OPERATION_IDS = frozenset({"aten::layer_norm", "aten::softmax"})
NORM_TARGETS = frozenset({"layer_norm", "softmax"})
ADD_HELPER = "tiled_resadd_auto"
LINEAR_HELPER = "tiled_matmul_auto"
AVERAGE_HELPER = "tiled_global_average_auto"
NORM_HELPER = "tiled_norm_auto"


@dataclass(frozen=True)
class ResidualFacts:
    """Canonical facts consumed by one residual helper call."""

    fused_name: str
    member_names: tuple[str, ...]
    input_a: anchor.TensorFact
    input_b: anchor.TensorFact
    output: anchor.TensorFact
    input_a_scale_tensor: anchor.TensorFact
    input_b_scale_tensor: anchor.TensorFact
    intermediate_q_scale_tensor: anchor.TensorFact | None
    intermediate_dq_scale_tensor: anchor.TensorFact | None
    output_scale_tensor: anchor.TensorFact
    input_a_scale: float
    input_b_scale: float
    intermediate_q_scale: float | None
    intermediate_dq_scale: float | None
    output_scale: float


@dataclass(frozen=True)
class LinearFacts:
    """Canonical facts consumed by one matrix-multiplication helper call."""

    fused_name: str
    member_names: tuple[str, ...]
    model_input: anchor.TensorFact | None
    input_codes: anchor.TensorFact
    weight: anchor.TensorFact
    bias: anchor.TensorFact
    accumulator: anchor.TensorFact
    output: anchor.TensorFact
    input_scale_tensor: anchor.TensorFact
    accumulator_scale_tensor: anchor.TensorFact
    output_scale_tensor: anchor.TensorFact
    input_scale: float
    accumulator_scale: float
    output_scale: float


@dataclass(frozen=True)
class AverageFacts:
    """Canonical facts consumed by one public global-average call."""

    fused_name: str
    member_names: tuple[str, ...]
    input_codes: anchor.TensorFact
    output: anchor.TensorFact
    input_scale_tensor: anchor.TensorFact
    output_scale_tensor: anchor.TensorFact
    input_scale: float
    output_scale: float
    batches: int
    channels: int
    dimension: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise anchor.ConversionError(message)


def _quantize_contract(operation: Any, label: str) -> None:
    _require(
        anchor._string_argument(operation.kwargs["rounding"], label)
        == "nearest",
        f"{label} is not nearest",
    )
    bounds = (
        anchor._float_argument(operation.kwargs["quant_min"], label),
        anchor._float_argument(operation.kwargs["quant_max"], label),
    )
    _require(bounds == (-128.0, 127.0), f"{label} is not signed INT8")


def _load_residual_facts(model_dir: Path, fused_name: str) -> ResidualFacts:
    proto = anchor._load_proto()
    model = proto.Model()
    text_format.Parse((model_dir / "model.txt").read_text(), model)
    matches = [
        record
        for record in model.ops
        if record.WhichOneof("op_type") == "fused_op"
        and record.fused_op.name == fused_name
    ]
    _require(len(matches) == 1, "requested residual region is missing")
    record = matches[0]
    fused = record.fused_op
    members = tuple(fused.op_list)
    _require(len(members) == 5, "residual region must have five members")
    dq_a, dq_b, addition, q_output, activation = members
    workload._operation(
        dq_a,
        target="dequantize",
        kwargs=anchor.DEQUANTIZE_KWARGS,
        label="residual input-A DQ",
    )
    workload._operation(
        dq_b,
        target="dequantize",
        kwargs=anchor.DEQUANTIZE_KWARGS,
        label="residual input-B DQ",
    )
    workload._operation(
        addition,
        target="add_",
        kwargs=frozenset({"input", "other", "alpha"}),
        label="shared residual Add",
    )
    workload._operation(
        q_output,
        target="quantize",
        kwargs=anchor.QUANTIZE_KWARGS,
        label="residual output Q",
    )
    workload._operation(
        activation,
        target="relu",
        kwargs=frozenset({"input"}),
        label="quantized-value residual ReLU",
    )
    _quantize_contract(q_output, "residual output Q")
    _require(
        anchor._integer_argument(addition.kwargs["alpha"], "residual alpha")
        == 1,
        "residual Add alpha differs",
    )

    input_a = workload._tensor_argument(
        dq_a.kwargs["input"],
        "residual input A",
        require_memory=True,
        dtypes=frozenset({"int8"}),
    )
    input_b = workload._tensor_argument(
        dq_b.kwargs["input"],
        "residual input B",
        require_memory=True,
        dtypes=frozenset({"int8"}),
    )
    scale_a = workload._tensor_argument(dq_a.kwargs["scale"], "input-A scale")
    scale_b = workload._tensor_argument(dq_b.kwargs["scale"], "input-B scale")
    addition_a = workload._tensor_argument(
        addition.kwargs["input"], "residual Add input"
    )
    addition_b = workload._tensor_argument(
        addition.kwargs["other"], "residual Add other"
    )
    dq_bindings = {
        dq_a.name: (input_a, scale_a),
        dq_b.name: (input_b, scale_b),
    }
    _require(
        {addition_a.node, addition_b.node} == set(dq_bindings),
        "Add does not consume both declared input DQs",
    )
    input_a, scale_a = dq_bindings[addition_a.node]
    input_b, scale_b = dq_bindings[addition_b.node]
    q_output_input = workload._tensor_argument(
        q_output.kwargs["input"], "residual output Q input"
    )
    _require(q_output_input.node == addition.name, "output Q misses Add")
    activation_input = workload._tensor_argument(
        activation.kwargs["input"], "residual ReLU input"
    )
    _require(
        activation_input.node == q_output.name,
        "quantized-value ReLU misses output Q",
    )
    output_scale = workload._tensor_argument(
        q_output.kwargs["scale"], "residual output scale"
    )
    output = workload._tensor_fact(
        record.output,
        "residual fused output",
        require_memory=True,
        dtypes=frozenset({"int8"}),
    )
    _require(output.node == fused.name, "residual output wire differs")
    _require(
        input_a.shape == input_b.shape == output.shape,
        "residual tensor shapes differ",
    )
    _require(
        len(output.shape) == 4 and output.shape[0] == 1, "bad residual shape"
    )
    scales = (
        workload._scale(model_dir, scale_a, "input-A scale"),
        workload._scale(model_dir, scale_b, "input-B scale"),
        workload._scale(model_dir, output_scale, "output scale"),
    )
    return ResidualFacts(
        fused_name=fused.name,
        member_names=tuple(member.name for member in members),
        input_a=input_a,
        input_b=input_b,
        output=output,
        input_a_scale_tensor=scale_a,
        input_b_scale_tensor=scale_b,
        intermediate_q_scale_tensor=None,
        intermediate_dq_scale_tensor=None,
        output_scale_tensor=output_scale,
        input_a_scale=scales[0],
        input_b_scale=scales[1],
        intermediate_q_scale=None,
        intermediate_dq_scale=None,
        output_scale=scales[2],
    )


def _ordinary_operation(model: Any, node: str, label: str) -> Any:
    matches = [
        record.op
        for record in model.ops
        if record.WhichOneof("op_type") == "op" and record.op.name == node
    ]
    _require(len(matches) == 1, f"{label} producer is missing")
    return matches[0]


def _dram_tensor_argument_allow_schedule(
    argument: Any, label: str
) -> anchor.TensorFact:
    kind, tensor = anchor._argument(argument, label)
    _require(kind == "tensor", f"{label} must be a tensor")
    _require(not tensor.is_none, f"{label} cannot be null")
    _require(bool(tensor.node), f"{label} has no node")
    _require(
        anchor.TENSOR_NODE_RE.fullmatch(tensor.node) is not None,
        f"{label} node is unsafe",
    )
    _require(bool(tensor.shape), f"{label} has no shape")
    _require(tensor.dtype == "int8", f"{label} is not INT8")
    _require(tensor.HasField("memory"), f"{label} has no DRAM fact")
    _require(
        tensor.memory.partition == 0
        and tensor.memory.HasField("address")
        and tensor.memory.instance == "DRAM",
        f"{label} is not in DRAM partition zero",
    )
    _require(not tensor.HasField("dequant"), f"{label} has dequant metadata")
    _require(not tensor.HasField("reshape"), f"{label} has reshape metadata")
    return anchor.TensorFact(
        role=label,
        node=tensor.node,
        shape=tuple(int(value) for value in tensor.shape),
        dtype=tensor.dtype,
        source=f"tensor_files/{tensor.node}.bin",
        memory_partition=int(tensor.memory.partition),
        memory_address=int(tensor.memory.address),
        memory_instance=tensor.memory.instance,
    )


def _dequantized_residual_input(
    model: Any,
    argument: Any,
    label: str,
) -> tuple[anchor.TensorFact, anchor.TensorFact]:
    semantic = workload._tensor_argument(
        argument,
        f"{label} semantic input",
        require_memory=True,
        dtypes=frozenset({"float32"}),
    )
    dequantize = _ordinary_operation(model, semantic.node, label)
    workload._operation(
        dequantize,
        target="dequantize",
        kwargs=anchor.DEQUANTIZE_KWARGS | frozenset({"l2_tiling"}),
        label=f"{label} DQ",
    )
    codes = _dram_tensor_argument_allow_schedule(
        dequantize.kwargs["input"],
        f"{label} codes",
    )
    scale = workload._tensor_argument(
        dequantize.kwargs["scale"], f"{label} scale"
    )
    return codes, scale


def _load_terminal_residual_facts(
    model_dir: Path, fused_name: str
) -> ResidualFacts:
    proto = anchor._load_proto()
    model = proto.Model()
    text_format.Parse((model_dir / "model.txt").read_text(), model)
    matches = [
        record
        for record in model.ops
        if record.WhichOneof("op_type") == "fused_op"
        and record.fused_op.name == fused_name
    ]
    _require(len(matches) == 1, "requested terminal residual is missing")
    record = matches[0]
    fused = record.fused_op
    members = tuple(fused.op_list)
    _require(len(members) == 3, "terminal residual must have three members")
    addition, q_output, activation = members
    workload._operation(
        addition,
        target="add_",
        kwargs=frozenset({"input", "other", "alpha"}),
        label="terminal residual Add",
    )
    workload._operation(
        q_output,
        target="quantize",
        kwargs=anchor.QUANTIZE_KWARGS,
        label="terminal residual output Q",
    )
    workload._operation(
        activation,
        target="relu",
        kwargs=frozenset({"input"}),
        label="quantized-value terminal residual ReLU",
    )
    _quantize_contract(q_output, "terminal residual output Q")
    _require(
        anchor._integer_argument(addition.kwargs["alpha"], "residual alpha")
        == 1,
        "terminal residual Add alpha differs",
    )
    input_a, scale_a = _dequantized_residual_input(
        model, addition.kwargs["input"], "terminal residual input A"
    )
    input_b, scale_b = _dequantized_residual_input(
        model, addition.kwargs["other"], "terminal residual input B"
    )
    q_output_input = workload._tensor_argument(
        q_output.kwargs["input"], "terminal output Q input"
    )
    _require(q_output_input.node == addition.name, "terminal Q misses Add")
    activation_input = workload._tensor_argument(
        activation.kwargs["input"], "terminal ReLU input"
    )
    _require(
        activation_input.node == q_output.name,
        "terminal quantized-value ReLU misses output Q",
    )
    output_scale = workload._tensor_argument(
        q_output.kwargs["scale"], "terminal residual output scale"
    )
    output = workload._tensor_fact(
        record.output,
        "terminal residual output",
        require_memory=True,
        dtypes=frozenset({"int8"}),
    )
    _require(output.node == fused.name, "terminal residual wire differs")
    _require(
        input_a.shape == input_b.shape == output.shape,
        "terminal residual tensor shapes differ",
    )
    scales = (
        workload._scale(model_dir, scale_a, "terminal input-A scale"),
        workload._scale(model_dir, scale_b, "terminal input-B scale"),
        workload._scale(model_dir, output_scale, "terminal output scale"),
    )
    return ResidualFacts(
        fused_name=fused.name,
        member_names=tuple(member.name for member in members),
        input_a=input_a,
        input_b=input_b,
        output=output,
        input_a_scale_tensor=scale_a,
        input_b_scale_tensor=scale_b,
        intermediate_q_scale_tensor=None,
        intermediate_dq_scale_tensor=None,
        output_scale_tensor=output_scale,
        input_a_scale=scales[0],
        input_b_scale=scales[1],
        intermediate_q_scale=None,
        intermediate_dq_scale=None,
        output_scale=scales[2],
    )


def _fused_output_scale(model: Any, producer_name: str) -> anchor.TensorFact:
    matches = [
        record
        for record in model.ops
        if record.WhichOneof("op_type") == "fused_op"
        and record.fused_op.name == producer_name
    ]
    _require(len(matches) == 1, "quantized producer region is missing")
    producer = matches[0]
    _require(
        producer.output.node == producer_name,
        "quantized producer output wire differs",
    )
    members = tuple(producer.fused_op.op_list)
    _require(bool(members), "quantized producer has no semantic members")
    quantizes = tuple(
        member for member in members if member.target == "quantize"
    )
    _require(bool(quantizes), "quantized producer has no output Q")
    quantize = quantizes[-1]
    workload._operation(
        quantize,
        target="quantize",
        kwargs=anchor.QUANTIZE_KWARGS,
        label="quantized producer output Q",
    )
    _quantize_contract(quantize, "quantized producer output Q")
    return workload._tensor_argument(
        quantize.kwargs["scale"], "quantized producer output scale"
    )


def _load_linear_facts(model_dir: Path, fused_name: str) -> LinearFacts:
    proto = anchor._load_proto()
    model = proto.Model()
    text_format.Parse((model_dir / "model.txt").read_text(), model)
    matches = [
        record
        for record in model.ops
        if record.WhichOneof("op_type") == "fused_op"
        and record.fused_op.name == fused_name
    ]
    _require(len(matches) == 1, "requested Linear region is missing")
    record = matches[0]
    fused = record.fused_op
    members = tuple(fused.op_list)
    _require(len(members) in {3, 4}, "Linear region has wrong member count")
    if len(members) == 4:
        input_quantize = members[0]
        linear, accumulator_dequantize, output_quantize = members[1:]
        workload._operation(
            input_quantize,
            target="quantize",
            kwargs=anchor.QUANTIZE_KWARGS,
            label="Linear input Q",
        )
        _quantize_contract(input_quantize, "Linear input Q")
    else:
        input_quantize = None
        linear, accumulator_dequantize, output_quantize = members
    workload._operation(
        linear,
        target="linear",
        kwargs=frozenset({"input", "weight", "bias"}),
        label="shared Linear",
    )
    workload._operation(
        accumulator_dequantize,
        target="dequantize",
        kwargs=anchor.DEQUANTIZE_KWARGS,
        label="Linear accumulator DQ",
    )
    workload._operation(
        output_quantize,
        target="quantize",
        kwargs=anchor.QUANTIZE_KWARGS,
        label="Linear output Q",
    )
    _quantize_contract(output_quantize, "Linear output Q")

    input_codes = workload._tensor_argument(
        linear.kwargs["input"],
        "Linear input codes",
        require_memory=True,
        dtypes=frozenset({"int8"}),
    )
    if input_quantize is not None:
        model_input = workload._tensor_argument(
            input_quantize.kwargs["input"],
            "Linear FLOAT32 input",
            require_memory=True,
            dtypes=frozenset({"float32"}),
        )
        input_scale = workload._tensor_argument(
            input_quantize.kwargs["scale"], "Linear input scale"
        )
        _require(
            input_codes.node == input_quantize.name,
            "Linear does not consume its input Q",
        )
    else:
        model_input = None
        input_scale = _fused_output_scale(model, input_codes.node)
    weight = workload._tensor_argument(
        linear.kwargs["weight"],
        "Linear weight",
        require_memory=True,
        dtypes=frozenset({"int8"}),
    )
    bias = workload._tensor_argument(
        linear.kwargs["bias"],
        "Linear bias",
        require_memory=True,
        dtypes=frozenset({"int32"}),
    )
    accumulator = workload._tensor_argument(
        accumulator_dequantize.kwargs["input"],
        "Linear accumulator",
        dtypes=frozenset({"float32", "int32"}),
    )
    _require(accumulator.node == linear.name, "Linear misses accumulator DQ")
    accumulator_scale = workload._tensor_argument(
        accumulator_dequantize.kwargs["scale"],
        "Linear accumulator scale",
    )
    output_quantize_input = workload._tensor_argument(
        output_quantize.kwargs["input"], "Linear output Q input"
    )
    _require(
        output_quantize_input.node == accumulator_dequantize.name,
        "Linear output Q misses accumulator DQ",
    )
    output_scale = workload._tensor_argument(
        output_quantize.kwargs["scale"], "Linear output scale"
    )
    output = workload._tensor_fact(
        record.output,
        "Linear fused output",
        require_memory=True,
        dtypes=frozenset({"int8"}),
    )
    _require(output.node == fused.name, "Linear output wire differs")
    _require(
        len(input_codes.shape) == 2 and input_codes.shape[0] == 1,
        "Linear input must be one matrix row",
    )
    _require(
        len(weight.shape) == 2 and weight.shape[1] == input_codes.shape[1],
        "Linear weight/input shapes differ",
    )
    _require(bias.shape == (weight.shape[0],), "Linear bias shape differs")
    _require(
        output.shape == (input_codes.shape[0], weight.shape[0]),
        "Linear output shape differs",
    )
    scales = (
        workload._scale(model_dir, input_scale, "Linear input scale"),
        workload._scale(
            model_dir, accumulator_scale, "Linear accumulator scale"
        ),
        workload._scale(model_dir, output_scale, "Linear output scale"),
    )
    return LinearFacts(
        fused_name=fused.name,
        member_names=tuple(member.name for member in members),
        model_input=model_input,
        input_codes=input_codes,
        weight=weight,
        bias=bias,
        accumulator=accumulator,
        output=output,
        input_scale_tensor=input_scale,
        accumulator_scale_tensor=accumulator_scale,
        output_scale_tensor=output_scale,
        input_scale=scales[0],
        accumulator_scale=scales[1],
        output_scale=scales[2],
    )


def _load_average_facts(model_dir: Path, fused_name: str) -> AverageFacts:
    proto = anchor._load_proto()
    model = proto.Model()
    text_format.Parse((model_dir / "model.txt").read_text(), model)
    matches = [
        record
        for record in model.ops
        if record.WhichOneof("op_type") == "fused_op"
        and record.fused_op.name == fused_name
    ]
    _require(len(matches) == 1, "requested global-average region is missing")
    record = matches[0]
    fused = record.fused_op
    members = tuple(fused.op_list)
    _require(len(members) == 3, "global-average region needs three members")
    dequantize, average, quantize = members
    workload._operation(
        dequantize,
        target="dequantize",
        kwargs=anchor.DEQUANTIZE_KWARGS,
        label="global-average input DQ",
    )
    workload._operation(
        average,
        target="adaptive_avg_pool2d",
        kwargs=frozenset({"input", "output_size"}),
        label="shared global average",
    )
    workload._operation(
        quantize,
        target="quantize",
        kwargs=anchor.QUANTIZE_KWARGS,
        label="global-average output Q",
    )
    _quantize_contract(quantize, "global-average output Q")
    output_size = anchor._pair_argument(
        average.kwargs["output_size"], "global-average output size"
    )
    _require(output_size == (1, 1), "average is not global 1x1 pooling")
    input_codes = workload._tensor_argument(
        dequantize.kwargs["input"],
        "global-average input codes",
        require_memory=True,
        dtypes=frozenset({"int8"}),
    )
    input_scale = workload._tensor_argument(
        dequantize.kwargs["scale"], "global-average input scale"
    )
    semantic_input = workload._tensor_argument(
        average.kwargs["input"], "global-average semantic input"
    )
    _require(
        semantic_input.node == dequantize.name,
        "global average does not consume its input DQ",
    )
    quantize_input = workload._tensor_argument(
        quantize.kwargs["input"], "global-average output-Q input"
    )
    _require(
        quantize_input.node in {average.name, "flatten"},
        "global-average output Q misses the average/flatten identity",
    )
    output_scale = workload._tensor_argument(
        quantize.kwargs["scale"], "global-average output scale"
    )
    output = workload._tensor_fact(
        record.output,
        "global-average fused output",
        require_memory=True,
        dtypes=frozenset({"int8"}),
    )
    _require(output.node == fused.name, "global-average output wire differs")
    _require(
        len(input_codes.shape) == 4
        and input_codes.shape[0] == 1
        and input_codes.shape[2] == input_codes.shape[3],
        "global-average input must be square NCHW",
    )
    batches, channels, dimension, _ = input_codes.shape
    _require(
        output.shape == (batches, channels),
        "global-average output shape differs",
    )
    _require(
        quantize_input.shape == output.shape,
        "eliminated Flatten does not preserve the global-average shape",
    )
    scales = (
        workload._scale(model_dir, input_scale, "global-average input scale"),
        workload._scale(model_dir, output_scale, "global-average output scale"),
    )
    _require(
        np.float32(scales[0]).tobytes() == np.float32(scales[1]).tobytes(),
        "global-average input and output domains differ",
    )
    return AverageFacts(
        fused_name=fused.name,
        member_names=tuple(member.name for member in members),
        input_codes=input_codes,
        output=output,
        input_scale_tensor=input_scale,
        output_scale_tensor=output_scale,
        input_scale=scales[0],
        output_scale=scales[1],
        batches=int(batches),
        channels=int(channels),
        dimension=int(dimension),
    )


def _load_facts(
    model_dir: Path, fused_name: str
) -> ResidualFacts | LinearFacts | AverageFacts:
    proto = anchor._load_proto()
    model = proto.Model()
    text_format.Parse((model_dir / "model.txt").read_text(), model)
    matches = [
        tuple(record.fused_op.op_list)
        for record in model.ops
        if record.WhichOneof("op_type") == "fused_op"
        and record.fused_op.name == fused_name
    ]
    _require(len(matches) == 1, "requested helper region is missing")
    targets = tuple(member.target for member in matches[0])
    if targets == (
        "dequantize",
        "dequantize",
        "add_",
        "quantize",
        "relu",
    ):
        return _load_residual_facts(model_dir, fused_name)
    if targets == ("add_", "quantize", "relu"):
        return _load_terminal_residual_facts(model_dir, fused_name)
    if len(targets) in {3, 4} and "linear" in targets:
        return _load_linear_facts(model_dir, fused_name)
    if targets == ("dequantize", "adaptive_avg_pool2d", "quantize"):
        return _load_average_facts(model_dir, fused_name)
    if any(target in NORM_TARGETS for target in targets):
        raise anchor.ConversionError(
            "tiled_norm_auto binding unavailable: pinned target lacks "
            "HAS_NORMALIZATIONS"
        )
    raise anchor.ConversionError("requested region has no supported helper")


def _read_codes(
    model_dir: Path, fact: anchor.TensorFact, label: str
) -> np.ndarray:
    return anchor._read_integer_container(
        anchor._safe_artifact_path(model_dir, fact.source, label),
        fact.shape,
        np.dtype(np.int8),
    )


def _render_payloads(payloads: dict[str, np.ndarray]) -> str:
    bias = workload._initializer("voyager_bias", payloads["bias"], "acc_t")
    bias = bias.replace("aligned(16)", "aligned(16), used")
    weight = workload._initializer(
        "voyager_weights", payloads["weight"], "elem_t"
    )
    weight = weight.replace("aligned(16)", "aligned(16), used")
    return "\n".join(
        (
            "#ifndef VOYAGER_GEMMINI_W_RESADD_PAYLOADS_H",
            "#define VOYAGER_GEMMINI_W_RESADD_PAYLOADS_H",
            '#include "include/gemmini.h"',
            workload._initializer("voyager_input", payloads["input"], "elem_t"),
            weight,
            bias,
            workload._initializer(
                "voyager_expected", payloads["oracle"], "elem_t"
            ),
            workload._initializer(
                "voyager_output", payloads["output"], "elem_t"
            ),
            "#endif",
            "",
        )
    )


def _render_residual_runner(
    projection: dict[str, Any], element_count: int
) -> str:
    fields = projection["fields"]
    return f"""#include <stdint.h>
#include <stdio.h>
#include "include/gemmini.h"
#include "payloads.h"

__attribute__((noinline, noclone, used))
void voyager_gemmini_operation(void) {{
  tiled_resadd_auto(
      {fields["rows"]}, {fields["channels"]}, {fields["input_a_scale"]},
      {fields["input_b_scale"]}, {fields["output_scale"]}, voyager_input,
      voyager_weights, voyager_output, true, WS);
}}

int main(void) {{
  int exact = 0;
  int tolerance = 0;
  int maximum = 0;
  for (int index = 0; index < {element_count}; ++index) {{
    voyager_output[index] = 0;
  }}
  gemmini_flush(0);
  voyager_gemmini_operation();
  gemmini_fence();
  for (int index = 0; index < {element_count}; ++index) {{
    int difference = voyager_output[index] - voyager_expected[index];
    int absolute = difference < 0 ? -difference : difference;
    printf("OUTPUT %d %d\\n", index, voyager_output[index]);
    if (difference != 0) ++exact;
    if (absolute > 1) ++tolerance;
    if (absolute > maximum) maximum = absolute;
  }}
  printf("SUMMARY count={element_count} exact_mismatches=%d "
         "tolerance_mismatches=%d max_abs=%d\\n", exact, tolerance, maximum);
  return 0;
}}
"""


def _render_linear_runner(
    projection: dict[str, Any], element_count: int
) -> str:
    fields = projection["fields"]
    return f"""#include <stdint.h>
#include <stdio.h>
#include "include/gemmini.h"
#include "payloads.h"

__attribute__((noinline, noclone, used))
void voyager_gemmini_operation(void) {{
  tiled_matmul_auto(
      {fields["rows"]}, {fields["columns"]}, {fields["reduction"]},
      voyager_input, voyager_weights, voyager_bias, voyager_output,
      {fields["input_stride"]}, {fields["weight_stride"]},
      {fields["bias_stride"]}, {fields["output_stride"]},
      MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
      NO_ACTIVATION, {fields["output_scale"]}, 0, true, false, false,
      false, false, 0, WS);
}}

int main(void) {{
  int exact = 0;
  int tolerance = 0;
  int maximum = 0;
  for (int index = 0; index < {element_count}; ++index) {{
    voyager_output[index] = 0;
  }}
  gemmini_flush(0);
  voyager_gemmini_operation();
  gemmini_fence();
  for (int index = 0; index < {element_count}; ++index) {{
    int difference = voyager_output[index] - voyager_expected[index];
    int absolute = difference < 0 ? -difference : difference;
    printf("OUTPUT %d %d\\n", index, voyager_output[index]);
    if (difference != 0) ++exact;
    if (absolute > 1) ++tolerance;
    if (absolute > maximum) maximum = absolute;
  }}
  printf("SUMMARY count={element_count} exact_mismatches=%d "
         "tolerance_mismatches=%d max_abs=%d\\n", exact, tolerance, maximum);
  return 0;
}}
"""


def _render_average_runner(
    projection: dict[str, Any], element_count: int
) -> str:
    fields = projection["fields"]
    return f"""#include <stdint.h>
#include <stdio.h>
#include "include/gemmini.h"
#include "payloads.h"

__attribute__((noinline, noclone, used))
void voyager_gemmini_operation(void) {{
  tiled_global_average_auto(
      voyager_input, voyager_output, {fields["batches"]},
      {fields["channels"]}, {fields["dimension"]}, WS);
}}

int main(void) {{
  int exact = 0;
  int tolerance = 0;
  int maximum = 0;
  for (int index = 0; index < {element_count}; ++index) {{
    voyager_output[index] = 0;
  }}
  gemmini_flush(0);
  voyager_gemmini_operation();
  gemmini_fence();
  for (int index = 0; index < {element_count}; ++index) {{
    int difference = voyager_output[index] - voyager_expected[index];
    int absolute = difference < 0 ? -difference : difference;
    printf("OUTPUT %d %d\\n", index, voyager_output[index]);
    if (difference != 0) ++exact;
    if (absolute > 1) ++tolerance;
    if (absolute > maximum) maximum = absolute;
  }}
  printf("SUMMARY count={element_count} exact_mismatches=%d "
         "tolerance_mismatches=%d max_abs=%d\\n", exact, tolerance, maximum);
  return 0;
}}
"""


def convert_helper_region(
    model_dir: Path,
    output_dir: Path,
    contract: Path,
    contract_sha256: str,
    fused_name: str,
) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    output_dir = output_dir.resolve()
    contract = contract.resolve()
    _require(model_dir.is_dir(), "canonical model directory is missing")
    _require(not output_dir.exists(), "conversion directory must be fresh")
    _require(contract.is_file(), "W injection contract is missing")
    _require(
        contract_sha256 == workload.WORKLOAD_CONTRACT_SHA256,
        "wrong W contract hash",
    )
    _require(
        anchor.sha256_file(contract) == workload.WORKLOAD_CONTRACT_SHA256,
        "W contract changed",
    )
    facts = _load_facts(model_dir, fused_name)
    if isinstance(facts, ResidualFacts):
        input_a = _read_codes(model_dir, facts.input_a, "input-A path")
        input_b = _read_codes(model_dir, facts.input_b, "input-B path")
        output = _read_codes(model_dir, facts.output, "output path")
        packed_output = anchor.pack_activation(output)
        payloads = {
            "input": anchor.pack_activation(input_a),
            "weight": anchor.pack_activation(input_b),
            "bias": np.zeros((1,), dtype=np.int32),
            "oracle": packed_output,
            "output": np.zeros_like(packed_output),
        }
        common_scale = max(facts.input_a_scale, facts.input_b_scale)
        projection = {
            "helper": ADD_HELPER,
            "fields": {
                "rows": int(
                    output.shape[0] * output.shape[2] * output.shape[3]
                ),
                "channels": int(output.shape[1]),
                "input_a_scale": (
                    float.hex(facts.input_a_scale / common_scale) + "f"
                ),
                "input_b_scale": (
                    float.hex(facts.input_b_scale / common_scale) + "f"
                ),
                "output_scale": (
                    float.hex(common_scale / facts.output_scale) + "f"
                ),
                "relu": True,
                "dataflow": "WS",
            },
        }
        operation_id = ADD_OPERATION_ID
        accepted_scope = "fixed-static-residual-helper-smoke"
        runner = _render_residual_runner(projection, int(output.size))
    elif isinstance(facts, AverageFacts):
        input_codes = _read_codes(
            model_dir, facts.input_codes, "global-average input path"
        )
        output = _read_codes(
            model_dir, facts.output, "global-average output path"
        )
        payloads = {
            "input": anchor.pack_activation(input_codes),
            "weight": np.zeros((1,), dtype=np.int8),
            "bias": np.zeros((1,), dtype=np.int32),
            "oracle": np.ascontiguousarray(output),
            "output": np.zeros_like(output),
        }
        projection = {
            "helper": AVERAGE_HELPER,
            "fields": {
                "batches": facts.batches,
                "channels": facts.channels,
                "dimension": facts.dimension,
                "domain_scale": float.hex(facts.input_scale) + "f",
                "native_divisor": facts.dimension * facts.dimension,
                "dataflow": "WS",
            },
        }
        operation_id = AVERAGE_OPERATION_ID
        accepted_scope = "fixed-static-global-average-helper-smoke"
        runner = _render_average_runner(projection, int(output.size))
    else:
        if facts.model_input is None:
            input_codes = _read_codes(
                model_dir, facts.input_codes, "Linear input-code path"
            )
        else:
            source = anchor._read_float_container(
                anchor._safe_artifact_path(
                    model_dir, facts.model_input.source, "Linear input path"
                ),
                facts.model_input.shape,
            )
            input_codes = np.clip(
                np.rint(source / np.float32(facts.input_scale)), -128, 127
            ).astype(np.int8)
        weight = anchor._read_integer_container(
            anchor._safe_artifact_path(
                model_dir, facts.weight.source, "Linear weight path"
            ),
            facts.weight.shape,
            np.dtype(np.int8),
        )
        bias = anchor._read_integer_container(
            anchor._safe_artifact_path(
                model_dir, facts.bias.source, "Linear bias path"
            ),
            facts.bias.shape,
            np.dtype(np.int32),
        )
        output = _read_codes(model_dir, facts.output, "Linear output path")
        payloads = {
            "input": np.ascontiguousarray(input_codes),
            "weight": np.ascontiguousarray(weight.T),
            "bias": np.ascontiguousarray(bias),
            "oracle": np.ascontiguousarray(output),
            "output": np.zeros_like(output),
        }
        rows, reduction = facts.input_codes.shape
        columns = facts.weight.shape[0]
        projection = {
            "helper": LINEAR_HELPER,
            "fields": {
                "rows": int(rows),
                "columns": int(columns),
                "reduction": int(reduction),
                "input_stride": int(reduction),
                "weight_stride": int(columns),
                "bias_stride": int(columns),
                "output_stride": int(columns),
                "output_scale": (
                    float.hex(
                        float(
                            np.float32(facts.accumulator_scale)
                            / np.float32(facts.output_scale)
                        )
                    )
                    + "f"
                ),
                "activation": "NO_ACTIVATION",
                "repeating_bias": True,
                "transpose_input": False,
                "transpose_weight": False,
                "dataflow": "WS",
            },
        }
        operation_id = LINEAR_OPERATION_ID
        accepted_scope = "fixed-static-linear-helper-smoke"
        runner = _render_linear_runner(projection, int(output.size))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
        )
    )
    try:
        payload_dir = staging / "payloads"
        payload_dir.mkdir()
        for role, values in payloads.items():
            (payload_dir / f"{role}.bin").write_bytes(values.tobytes(order="C"))
        (staging / "payloads.h").write_text(_render_payloads(payloads))
        (staging / "runner.c").write_text(runner)
        semantic_source = {
            "path": "model.txt",
            "sha256": anchor.sha256_file(model_dir / "model.txt"),
        }
        emitted = {
            "schema_version": 1,
            "semantic_source": semantic_source,
            "operation_id": operation_id,
            "emitted_operation_helper_count": 1,
            "emitted_backend_calls": [
                {
                    "ordinal": 0,
                    "helper": projection["helper"],
                    "model_operation_name": facts.fused_name,
                    "fused_member_names": list(facts.member_names),
                }
            ],
            "lifecycle_calls": ["gemmini_flush", "gemmini_fence"],
        }
        anchor._write_json(staging / "emitted-calls.json", emitted)
        artifacts = [
            workload._artifact(path, staging)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        manifest = {
            "schema_version": 1,
            "status": "converted",
            "operation_id": operation_id,
            "contract": {"path": str(contract), "sha256": contract_sha256},
            "semantic_source": semantic_source,
            "facts": asdict(facts),
            "projection": projection,
            "compiler_layout_output": {
                "path": facts.output.source,
                "sha256": anchor.sha256_file(
                    anchor._safe_artifact_path(
                        model_dir, facts.output.source, "output path"
                    )
                ),
                "logical_dtype": "int8",
                "shape": list(facts.output.shape),
            },
            "accepted_scope": accepted_scope,
            "accuracy_required": False,
            "general_bit_exact_equivalence": False,
            "artifacts": artifacts,
        }
        anchor._write_json(staging / "conversion_manifest.json", manifest)
        staging.rename(output_dir)
        return manifest
    except BaseException:
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        staging.rmdir()
        raise


def validate_helper_conversion(
    converted: Path, contract: Path
) -> dict[str, Any]:
    manifest = anchor._load_json(
        converted / "conversion_manifest.json", "residual manifest"
    )
    _require(manifest.get("status") == "converted", "conversion did not finish")
    operation_id = manifest.get("operation_id")
    _require(
        operation_id
        in {ADD_OPERATION_ID, LINEAR_OPERATION_ID, AVERAGE_OPERATION_ID},
        "wrong operation",
    )
    _require(
        manifest.get("contract")
        == {
            "path": str(contract),
            "sha256": workload.WORKLOAD_CONTRACT_SHA256,
        },
        "wrong conversion contract",
    )
    projection = manifest.get("projection")
    _require(isinstance(projection, dict), "projection is missing")
    helpers = {
        ADD_OPERATION_ID: ADD_HELPER,
        LINEAR_OPERATION_ID: LINEAR_HELPER,
        AVERAGE_OPERATION_ID: AVERAGE_HELPER,
    }
    expected_helper = helpers[operation_id]
    _require(projection.get("helper") == expected_helper, "wrong native helper")
    fields = projection.get("fields")
    _require(isinstance(fields, dict), "helper fields are missing")
    if operation_id == ADD_OPERATION_ID:
        _require(fields.get("relu") is True, "residual ReLU is missing")
    elif operation_id == LINEAR_OPERATION_ID:
        _require(
            fields.get("activation") == "NO_ACTIVATION",
            "Linear activation differs",
        )
        _require(fields.get("repeating_bias") is True, "Linear bias differs")
    else:
        _require(fields.get("dataflow") == "WS", "average dataflow differs")
        _require(
            isinstance(fields.get("domain_scale"), str),
            "average domain scale is missing",
        )
        _require(
            fields.get("native_divisor")
            == fields.get("dimension") * fields.get("dimension"),
            "average native divisor differs",
        )
    for name in ("payloads.h", "runner.c"):
        _require((converted / name).is_file(), f"missing generated {name}")
    for role in ("input", "weight", "bias", "oracle", "output"):
        _require(
            (converted / "payloads" / f"{role}.bin").is_file(),
            f"missing {role} payload",
        )
    source = (converted / "runner.c").read_text()
    _require(source.count(f"{expected_helper}(") == 1, "wrong helper count")
    if operation_id == AVERAGE_OPERATION_ID:
        _require(
            "gemmini_scaled_average.h" not in source
            and "tiled_global_average_scaled_auto" not in source,
            "private scaled-average helper reached generated collateral",
        )
    return manifest


__all__ = [
    "ADD_OPERATION_ID",
    "AVERAGE_OPERATION_ID",
    "LINEAR_OPERATION_ID",
    "convert_helper_region",
    "validate_helper_conversion",
]

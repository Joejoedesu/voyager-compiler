#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Convert one canonical Voyager Gemmini Conv2d into native collateral."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from google.protobuf import text_format

CONTRACT_ID = "gemmini-R-default-s8-operations-20260816"
INJECTION_CONTRACT_SHA256 = (
    "a36b7a95d5d47bc8531abff4b72fa463731957f92ae5f34705c8cd7b5876d612"
)
OPERATION_ID = "aten::conv2d"
HELPER = "tiled_conv_auto"
PAYLOAD_ALIGNMENT_BYTES = 16
ORACLE_ALIGNMENT_BYTES = 16

FUSED_SERIALIZED_TARGETS = (
    "quantize",
    "conv2d",
    "dequantize",
    "quantize",
)
FUSED_SEMANTIC_MEMBERS = (
    "quantized_ops::quantize",
    "aten::conv2d",
    "quantized_ops::dequantize",
    "quantized_ops::quantize",
)
QUANTIZE_KWARGS = frozenset(
    {"input", "scale", "rounding", "quant_min", "quant_max"}
)
CONV2D_KWARGS = frozenset(
    {"input", "weight", "stride", "padding", "dilation", "groups"}
)
DEQUANTIZE_KWARGS = frozenset({"input", "scale"})
TENSOR_NODE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ORACLE_MANIFEST = "oracle-manifest.json"
ORACLE_ARTIFACT = "pt2e_reference.bin"
ORACLE_TERMINAL_ARTIFACT = "pt2e_terminal_reference.bin"
ORACLE_PRODUCER = "natural-pytorch-pt2e-qdq"
SCHEME_ID = "gemmini.s8-symmetric-per-tensor"
QDQ_FORM = "gemmini.s8s8-s32-to-s8-f32-rne-sat"
PINNED_NATIVE_FILES = {
    "gemmini.h": (
        "gemmini/software/gemmini-rocc-tests/include/gemmini.h",
        "18801f4eab0cd2e81e25a6c2aa97c7ed644b693e8d2c24cdeb41ab3a7208336d",
    ),
    "gemmini_params.h": (
        "gemmini/software/gemmini-rocc-tests/include/gemmini_params.h",
        "3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924",
    ),
}
GENERATED_ARTIFACT_PATHS = frozenset(
    {
        "emitted-calls.json",
        "layout-map.csv",
        "payloads.h",
        "runner.c",
        "payloads/input.bin",
        "payloads/weight.bin",
        "payloads/output.bin",
        "payloads/output_oracle.bin",
    }
)


class ConversionError(RuntimeError):
    """Canonical input cannot be represented by the frozen Gemmini ABI."""


@dataclass(frozen=True)
class TensorFact:
    """One canonical logical tensor and its compiler dump."""

    role: str
    node: str
    shape: tuple[int, ...]
    dtype: str
    source: str
    memory_partition: int | None
    memory_address: int | None
    memory_instance: str | None


@dataclass(frozen=True)
class OperationFacts:
    """Canonical semantic facts consumed from ``model.txt``."""

    fused_name: str
    member_names: tuple[str, str, str, str]
    terminal_name: str
    model_input: TensorFact
    input: TensorFact
    weight: TensorFact
    accumulator: TensorFact
    accumulator_dequantized: TensorFact
    output: TensorFact
    terminal_output: TensorFact
    input_scale_tensor: TensorFact
    accumulator_scale_tensor: TensorFact
    output_scale_tensor: TensorFact
    terminal_scale_tensor: TensorFact
    bias: TensorFact | None
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int
    input_scale: float
    input_zero_point: int
    accumulator_scale: float
    weight_scale: float
    weight_zero_point: int
    output_scale: float
    output_zero_point: int
    quant_min: float
    quant_max: float
    rounding: str


@dataclass(frozen=True)
class ProjectionField:
    """One native helper field with a single explicit provenance."""

    value: Any
    provenance: str


@dataclass(frozen=True)
class NativeProjection:
    """Target-owned projection of one canonical operation into Gemmini."""

    helper: str
    fields: dict[str, ProjectionField]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConversionError(message)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _native_dependencies() -> dict[str, dict[str, str]]:
    """Describe native headers without requiring a target checkout here."""
    evidence: dict[str, dict[str, str]] = {}
    for name, (relative, expected_sha256) in PINNED_NATIVE_FILES.items():
        evidence[name] = {
            "path": relative,
            "sha256": expected_sha256,
        }
    return evidence


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ConversionError(f"cannot load {label}: {error}") from error
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def _safe_artifact_path(root: Path, value: Any, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label} must be a path")
    candidate = value.replace("\\", "/")
    relative = PurePosixPath(candidate)
    _require(not relative.is_absolute(), f"{label} must be relative")
    _require(
        candidate == relative.as_posix()
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"{label} is not normalized",
    )
    path = (root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ConversionError(
            f"{label} escapes the canonical directory"
        ) from error
    return path


def _load_proto() -> Any:
    path = _repo_root() / "src/voyager_compiler/codegen/param_pb2.py"
    specification = importlib.util.spec_from_file_location(
        "gemmini_param_pb2", path
    )
    if specification is None or specification.loader is None:
        raise ConversionError(f"cannot load canonical protobuf schema: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _tensor_fact(
    tensor: Any, role: str, *, require_memory: bool = False
) -> TensorFact:
    _require(bool(tensor.node), f"{role} tensor has no node identity")
    _require(
        TENSOR_NODE_RE.fullmatch(tensor.node) is not None,
        f"{role} tensor node is not a safe file identity",
    )
    _require(bool(tensor.shape), f"{role} tensor has no shape")
    _require(tensor.dtype in {"float32", "int8"}, f"{role} has wrong dtype")
    has_memory = tensor.HasField("memory")
    _require(
        has_memory or not require_memory, f"{role} has no DRAM memory fact"
    )
    if has_memory:
        memory = tensor.memory
        _require(
            memory.HasField("partition") and memory.partition == 0,
            f"{role} memory is not DRAM partition zero",
        )
        _require(memory.HasField("address"), f"{role} DRAM address is missing")
        _require(
            memory.HasField("instance") and memory.instance == "DRAM",
            f"{role} memory instance is not DRAM",
        )
    _require(not tensor.tiled_shape, f"{role} carries a tiled shape")
    _require(not tensor.tile_strides, f"{role} carries tile strides")
    _require(
        not tensor.HasField("scratchpad"), f"{role} carries scratchpad control"
    )
    _require(not tensor.HasField("dequant"), f"{role} carries fused dequant")
    _require(not tensor.HasField("reshape"), f"{role} carries fused reshape")
    _require(not tensor.is_none, f"{role} tensor cannot be null")
    return TensorFact(
        role=role,
        node=tensor.node,
        shape=tuple(int(value) for value in tensor.shape),
        dtype=tensor.dtype,
        source=f"tensor_files/{tensor.node}.bin",
        memory_partition=int(tensor.memory.partition) if has_memory else None,
        memory_address=int(tensor.memory.address) if has_memory else None,
        memory_instance=tensor.memory.instance if has_memory else None,
    )


def _argument(argument: Any, name: str) -> tuple[str | None, Any]:
    kind = argument.WhichOneof("arg_type")
    return kind, None if kind is None else getattr(argument, kind)


def _tensor_argument(
    argument: Any, name: str, *, require_memory: bool = False
) -> TensorFact:
    kind, value = _argument(argument, name)
    _require(kind == "tensor", f"{name} must be a tensor argument")
    _require(not value.is_none, f"{name} cannot be null")
    return _tensor_fact(value, name, require_memory=require_memory)


def _pair_argument(argument: Any, name: str) -> tuple[int, int]:
    kind, value = _argument(argument, name)
    _require(kind == "int_list", f"{name} must be an integer list")
    pair = tuple(int(item) for item in value.values)
    _require(len(pair) == 2, f"{name} must contain two integers")
    return pair


def _float_argument(argument: Any, name: str) -> float:
    kind, value = _argument(argument, name)
    _require(kind == "float_value", f"{name} must be a protobuf float")
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    return result


def _integer_argument(argument: Any, name: str) -> int:
    kind, value = _argument(argument, name)
    _require(kind == "int_value", f"{name} must be a protobuf integer")
    return int(value)


def _string_argument(argument: Any, name: str) -> str:
    kind, value = _argument(argument, name)
    _require(kind == "str_value", f"{name} must be a protobuf string")
    return str(value)


def _validate_operation(
    operation: Any,
    *,
    target: str,
    kwargs: frozenset[str],
    label: str,
) -> None:
    _require(bool(operation.name), f"{label} has no node identity")
    _require(
        TENSOR_NODE_RE.fullmatch(operation.name) is not None,
        f"{label} name is not a safe file identity",
    )
    _require(operation.op == "call_function", f"{label} has wrong kind")
    _require(operation.target == target, f"{label} has wrong target")
    _require(not operation.args, f"{label} must have no arguments")
    actual = set(operation.kwargs)
    _require(
        actual == kwargs,
        f"{label} has wrong kwargs; missing={sorted(kwargs - actual)}, "
        f"extra={sorted(actual - kwargs)}",
    )


def _scale_value(model_dir: Path, fact: TensorFact, label: str) -> float:
    _require(fact.shape == (1,), f"{label} must be scalar")
    _require(fact.dtype == "float32", f"{label} must be FLOAT32")
    path = _safe_artifact_path(model_dir, fact.source, f"{label} path")
    value = _read_float_container(path, fact.shape).reshape(-1)[0]
    _require(value > 0, f"{label} must be positive")
    return float(np.float32(value))


def _wire_facts(fact: TensorFact) -> tuple[Any, ...]:
    return (
        fact.node,
        fact.shape,
        fact.dtype,
        fact.memory_partition,
        fact.memory_address,
        fact.memory_instance,
    )


def _validate_model(model_dir: Path) -> OperationFacts:
    path = _safe_artifact_path(model_dir, "model.txt", "canonical model path")
    _require(path.is_file(), f"canonical model is missing: {path}")
    model = _load_proto().Model()
    try:
        text_format.Parse(path.read_text(), model)
    except (OSError, text_format.ParseError) as error:
        raise ConversionError(f"invalid canonical model: {error}") from error

    _require(len(model.inputs) == 1, "model must have exactly one input")
    _require(len(model.parameters) == 1, "model must have exactly one weight")
    _require(len(model.ops) == 2, "model must have fused and terminal ops")

    fused_record, terminal_record = model.ops
    _require(
        fused_record.WhichOneof("op_type") == "fused_op",
        "first operation must be fused",
    )
    _require(
        terminal_record.WhichOneof("op_type") == "op",
        "second operation must be ordinary terminal dequantize",
    )
    for label, record in (
        ("fused operation", fused_record),
        ("terminal operation", terminal_record),
    ):
        _require(
            record.WhichOneof("return_type") == "output",
            f"{label} has wrong return type",
        )
        _require(
            not record.HasField("tiling"),
            f"{label} carries forbidden tiling",
        )

    fused = fused_record.fused_op
    _require(bool(fused.name), "fused operation has no node identity")
    _require(
        TENSOR_NODE_RE.fullmatch(fused.name) is not None,
        "fused operation name is not a safe file identity",
    )
    _require(
        len(fused.op_list) == 4,
        "fused operation must have exactly four members",
    )
    input_quantize, convolution, accumulator_dequantize, output_quantize = (
        fused.op_list
    )
    for operation, target, kwargs, label in zip(
        fused.op_list,
        FUSED_SERIALIZED_TARGETS,
        (
            QUANTIZE_KWARGS,
            CONV2D_KWARGS,
            DEQUANTIZE_KWARGS,
            QUANTIZE_KWARGS,
        ),
        (
            "input quantize",
            "shared conv2d",
            "accumulator dequantize",
            "output quantize",
        ),
        strict=True,
    ):
        _validate_operation(
            operation, target=target, kwargs=kwargs, label=label
        )
    terminal = terminal_record.op
    _validate_operation(
        terminal,
        target="dequantize",
        kwargs=DEQUANTIZE_KWARGS,
        label="terminal dequantize",
    )

    model_input = _tensor_fact(
        model.inputs[0], "model input", require_memory=True
    )
    parameter = _tensor_fact(
        model.parameters[0], "model weight", require_memory=True
    )
    quantize_input = _tensor_argument(
        input_quantize.kwargs["input"],
        "input quantize input",
        require_memory=True,
    )
    _require(
        _wire_facts(quantize_input) == _wire_facts(model_input),
        "input quantize differs from model input",
    )
    input_scale_tensor = _tensor_argument(
        input_quantize.kwargs["scale"], "input scale"
    )
    rounding = _string_argument(
        input_quantize.kwargs["rounding"], "input rounding"
    )
    quant_min = _float_argument(
        input_quantize.kwargs["quant_min"], "input quant_min"
    )
    quant_max = _float_argument(
        input_quantize.kwargs["quant_max"], "input quant_max"
    )
    _require(rounding == "nearest", "input quantize must use nearest rounding")
    _require(
        (quant_min, quant_max) == (-128.0, 127.0),
        "input quantize must use signed INT8 bounds",
    )

    input_codes = _tensor_argument(
        convolution.kwargs["input"], "conv2d input codes"
    )
    _require(
        input_codes.node == input_quantize.name,
        "conv2d input does not consume input quantize",
    )
    weight = _tensor_argument(
        convolution.kwargs["weight"], "conv2d weight", require_memory=True
    )
    _require(
        _wire_facts(weight) == _wire_facts(parameter),
        "conv2d weight differs from model parameter",
    )
    stride = _pair_argument(convolution.kwargs["stride"], "stride")
    padding = _pair_argument(convolution.kwargs["padding"], "padding")
    dilation = _pair_argument(convolution.kwargs["dilation"], "dilation")
    groups = _integer_argument(convolution.kwargs["groups"], "groups")

    accumulator = _tensor_argument(
        accumulator_dequantize.kwargs["input"], "accumulator codes"
    )
    _require(
        accumulator.node == convolution.name,
        "accumulator dequantize does not consume conv2d",
    )
    accumulator_scale_tensor = _tensor_argument(
        accumulator_dequantize.kwargs["scale"], "accumulator scale"
    )
    accumulator_dequantized = _tensor_argument(
        output_quantize.kwargs["input"], "dequantized accumulator"
    )
    _require(
        accumulator_dequantized.node == accumulator_dequantize.name,
        "output quantize does not consume accumulator dequantize",
    )
    output_scale_tensor = _tensor_argument(
        output_quantize.kwargs["scale"], "output quantize scale"
    )
    output_rounding = _string_argument(
        output_quantize.kwargs["rounding"], "output rounding"
    )
    output_min = _float_argument(
        output_quantize.kwargs["quant_min"], "output quant_min"
    )
    output_max = _float_argument(
        output_quantize.kwargs["quant_max"], "output quant_max"
    )
    _require(
        (output_rounding, output_min, output_max)
        == (rounding, quant_min, quant_max),
        "input and output quantize policies differ",
    )

    output = _tensor_fact(
        fused_record.output, "fused code output", require_memory=True
    )
    _require(output.node == fused.name, "fused output identity differs")
    terminal_input = _tensor_argument(
        terminal.kwargs["input"],
        "terminal dequantize input",
        require_memory=True,
    )
    _require(
        _wire_facts(terminal_input) == _wire_facts(output),
        "terminal dequantize does not consume fused output",
    )
    terminal_scale_tensor = _tensor_argument(
        terminal.kwargs["scale"], "terminal scale"
    )
    terminal_output = _tensor_fact(
        terminal_record.output, "terminal output", require_memory=True
    )
    _require(
        terminal_output.node == terminal.name,
        "terminal output identity differs from operation",
    )

    member_names = tuple(operation.name for operation in fused.op_list)
    _require(
        len(set((*member_names, fused.name, terminal.name))) == 6,
        "fused and terminal operation identities must be distinct",
    )
    _require(model_input.shape == (1, 16, 4, 4), "wrong model input shape")
    _require(model_input.dtype == "float32", "model input is not FLOAT32")
    _require(input_codes.shape == model_input.shape, "wrong input code shape")
    _require(input_codes.dtype == "int8", "input codes are not signed INT8")
    _require(weight.shape == (16, 16, 3, 3), "wrong weight shape")
    _require(weight.dtype == "int8", "weight is not signed INT8")
    for fact, label, dtype in (
        (accumulator, "accumulator", "int8"),
        (accumulator_dequantized, "dequantized accumulator", "float32"),
        (output, "fused output", "int8"),
        (terminal_output, "terminal output", "float32"),
    ):
        _require(fact.shape == (1, 16, 4, 4), f"wrong {label} shape")
        _require(fact.dtype == dtype, f"wrong {label} dtype")
    _require(stride == (1, 1), "registration anchor requires stride [1, 1]")
    _require(padding == (1, 1), "registration anchor requires padding [1, 1]")
    _require(dilation == (1, 1), "registration anchor requires dilation [1, 1]")
    _require(groups == 1, "registration anchor requires groups=1")

    input_scale = _scale_value(model_dir, input_scale_tensor, "input scale")
    accumulator_scale = _scale_value(
        model_dir, accumulator_scale_tensor, "accumulator scale"
    )
    output_scale = _scale_value(
        model_dir, output_scale_tensor, "output quantize scale"
    )
    terminal_scale = _scale_value(
        model_dir, terminal_scale_tensor, "terminal scale"
    )
    _require(
        np.float32(output_scale).tobytes()
        == np.float32(terminal_scale).tobytes(),
        "output quantize and terminal dequantize scales differ",
    )
    weight_scale = float(
        np.float32(np.float32(accumulator_scale) / np.float32(input_scale))
    )
    _require(
        math.isfinite(weight_scale) and weight_scale > 0,
        "derived weight scale is invalid",
    )
    return OperationFacts(
        fused_name=fused.name,
        member_names=member_names,
        terminal_name=terminal.name,
        model_input=model_input,
        input=input_codes,
        weight=weight,
        accumulator=accumulator,
        accumulator_dequantized=accumulator_dequantized,
        output=output,
        terminal_output=terminal_output,
        input_scale_tensor=input_scale_tensor,
        accumulator_scale_tensor=accumulator_scale_tensor,
        output_scale_tensor=output_scale_tensor,
        terminal_scale_tensor=terminal_scale_tensor,
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        input_scale=input_scale,
        input_zero_point=0,
        accumulator_scale=accumulator_scale,
        weight_scale=weight_scale,
        weight_zero_point=0,
        output_scale=output_scale,
        output_zero_point=0,
        quant_min=quant_min,
        quant_max=quant_max,
        rounding=rounding,
    )


def _native_projection(facts: OperationFacts) -> NativeProjection:
    batch, channels, rows, columns = facts.input.shape
    outputs, weight_channels, kernel_rows, kernel_columns = facts.weight.shape
    out_batch, out_channels, out_rows, out_columns = facts.output.shape
    _require(batch == out_batch, "input and output batches differ")
    _require(channels == weight_channels, "input and weight channels differ")
    _require(outputs == out_channels, "weight and output channels differ")
    _require(kernel_rows == kernel_columns, "helper requires square kernel")
    _require(
        facts.stride[0] == facts.stride[1], "helper requires square stride"
    )
    _require(
        facts.padding[0] == facts.padding[1], "helper requires square padding"
    )
    _require(facts.dilation == (1, 1), "helper requires unit dilation")
    compiler = "canonical:model.txt"
    contract = "native:injection-contract"
    multiplier = np.float32(
        np.float32(facts.accumulator_scale) / np.float32(facts.output_scale)
    )
    _require(
        bool(np.isfinite(multiplier)) and multiplier > 0,
        "native accumulator scale is invalid",
    )
    multiplier_literal = float.hex(float(multiplier)) + "f"
    values = {
        "batch_size": (batch, f"{compiler}:input.shape[0]"),
        "in_row_dim": (rows, f"{compiler}:input.shape[2]"),
        "in_col_dim": (columns, f"{compiler}:input.shape[3]"),
        "in_channels": (channels, f"{compiler}:input.shape[1]"),
        "out_channels": (outputs, f"{compiler}:weight.shape[0]"),
        "out_row_dim": (out_rows, f"{compiler}:output.shape[2]"),
        "out_col_dim": (out_columns, f"{compiler}:output.shape[3]"),
        "stride": (facts.stride[0], f"{compiler}:stride"),
        "input_dilation": (1, f"{contract}:input_dilation"),
        "kernel_dilation": (facts.dilation[0], f"{compiler}:dilation"),
        "padding": (facts.padding[0], f"{compiler}:padding"),
        "kernel_dim": (kernel_rows, f"{compiler}:weight.shape[2:4]"),
        "wrot180": (False, f"{contract}:wrot180"),
        "trans_output_1203": (False, f"{contract}:trans_output_1203"),
        "trans_input_3120": (False, f"{contract}:trans_input_3120"),
        "trans_weight_1203": (False, f"{contract}:trans_weight_1203"),
        "trans_weight_0132": (False, f"{contract}:trans_weight_0132"),
        "input": ("voyager_input", "converter:packed-NHWC-input"),
        "weights": ("voyager_weights", "converter:packed-HWIO-weight"),
        "bias": ("NULL", f"{compiler}:bias-omitted"),
        "output": ("voyager_output", "converter:packed-NHWC-output"),
        "activation": ("NO_ACTIVATION", f"{contract}:activation"),
        "acc_scale": (
            multiplier_literal,
            f"{compiler}:accumulator_dequant_scale/output_quantize_scale",
        ),
        "pool_size": (0, f"{contract}:pooling"),
        "pool_stride": (0, f"{contract}:pooling"),
        "pool_padding": (0, f"{contract}:pooling"),
        "dataflow": ("WS", f"{contract}:dataflow"),
    }
    return NativeProjection(
        helper=HELPER,
        fields={
            name: ProjectionField(value=value, provenance=provenance)
            for name, (value, provenance) in values.items()
        },
    )


def _read_integer_container(
    path: Path, shape: tuple[int, ...], dtype: np.dtype
) -> np.ndarray:
    _require(path.is_file(), f"logical tensor dump is missing: {path}")
    payload = path.read_bytes()
    count = math.prod(shape)
    _require(len(payload) == count * 4, f"wrong FLOAT32 byte count: {path}")
    values = np.frombuffer(payload, dtype="<f4")
    _require(np.isfinite(values).all(), f"non-finite FLOAT32 value: {path}")
    _require(
        np.equal(values, np.rint(values)).all(), f"fractional value: {path}"
    )
    info = np.iinfo(dtype)
    _require(
        ((values >= info.min) & (values <= info.max)).all(),
        f"out-of-range value: {path}",
    )
    return np.ascontiguousarray(values.astype(dtype).reshape(shape))


def _read_float_container(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    _require(path.is_file(), f"logical tensor dump is missing: {path}")
    payload = path.read_bytes()
    count = math.prod(shape)
    _require(len(payload) == count * 4, f"wrong FLOAT32 byte count: {path}")
    values = np.frombuffer(payload, dtype="<f4")
    _require(np.isfinite(values).all(), f"non-finite FLOAT32 value: {path}")
    return np.ascontiguousarray(values.reshape(shape))


def _materialize_input(
    path: Path,
    shape: tuple[int, ...],
    scale: float,
    zero_point: int,
    rounding: str = "nearest",
    quant_min: float = -128.0,
    quant_max: float = 127.0,
) -> np.ndarray:
    _require(rounding == "nearest", "unsupported input quantize rounding")
    _require(zero_point == 0, "input quantize zero point must be zero")
    _require(
        (quant_min, quant_max) == (-128.0, 127.0),
        "input quantize bounds must be signed INT8",
    )
    source = _read_float_container(path, shape)
    scaled = np.divide(source, np.float32(scale), dtype=np.float32)
    _require(np.isfinite(scaled).all(), "input quantize normalization overflow")
    codes = np.rint(scaled)
    return np.ascontiguousarray(
        np.clip(codes, quant_min, quant_max).astype(np.int8)
    )


def _qparams(facts: OperationFacts) -> dict[str, float | int]:
    return {
        "input_scale": facts.input_scale,
        "input_zero_point": facts.input_zero_point,
        "weight_scale": facts.weight_scale,
        "weight_zero_point": facts.weight_zero_point,
        "output_scale": facts.output_scale,
        "output_zero_point": facts.output_zero_point,
    }


def _validate_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _load_oracle(
    model_dir: Path, facts: OperationFacts
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest_path = _safe_artifact_path(
        model_dir, ORACLE_MANIFEST, "oracle manifest path"
    )
    _require(
        manifest_path.is_file(),
        f"independent oracle manifest is missing: {manifest_path}",
    )
    manifest = _load_json(manifest_path, "independent oracle manifest")
    expected_top = {
        "schema_version",
        "producer",
        "operation_id",
        "scheme_id",
        "qdq_form",
        "code_reference",
        "terminal_reference",
        "canonical_code_output",
        "canonical_terminal_output",
        "qparams",
    }
    _require(set(manifest) == expected_top, "wrong oracle manifest fields")
    _require(manifest["schema_version"] == 1, "wrong oracle schema version")
    _require(manifest["producer"] == ORACLE_PRODUCER, "wrong oracle producer")
    _require(manifest["operation_id"] == OPERATION_ID, "wrong oracle operation")
    _require(manifest["scheme_id"] == SCHEME_ID, "wrong oracle scheme")
    _require(manifest["qdq_form"] == QDQ_FORM, "wrong oracle Q/DQ form")

    expected_qparams = _qparams(facts)
    qparams = manifest["qparams"]
    _require(isinstance(qparams, dict), "oracle qparams must be an object")
    _require(
        set(qparams) == set(expected_qparams), "wrong oracle qparam fields"
    )
    for name, expected in expected_qparams.items():
        value = qparams[name]
        required_type = float if name.endswith("_scale") else int
        _require(
            type(value) is required_type, f"wrong oracle qparam type: {name}"
        )
        if required_type is float:
            _require(
                math.isfinite(value) and value > 0,
                f"invalid oracle qparam: {name}",
            )
            _require(
                float(np.float32(value)) == value,
                f"oracle qparam is not exactly FLOAT32: {name}",
            )
            equal = (
                np.float32(value).tobytes() == np.float32(expected).tobytes()
            )
        else:
            equal = value == expected
        _require(equal, f"oracle qparam mismatch: {name}")
    _require(
        np.float32(
            np.float32(qparams["input_scale"])
            * np.float32(qparams["weight_scale"])
        ).tobytes()
        == np.float32(facts.accumulator_scale).tobytes(),
        "oracle weight scale does not produce accumulator scale",
    )

    artifact = manifest["code_reference"]
    _require(isinstance(artifact, dict), "code reference must be an object")
    _require(
        set(artifact)
        == {"path", "sha256", "container_dtype", "logical_dtype", "shape"},
        "wrong code reference fields",
    )
    _require(artifact["path"] == ORACLE_ARTIFACT, "wrong code reference path")
    _require(
        artifact["container_dtype"] == "float32",
        "code reference container must be FLOAT32",
    )
    _require(
        artifact["logical_dtype"] == "int8",
        "code reference must be logical INT8",
    )
    _require(
        artifact["shape"] == list(facts.output.shape),
        "wrong code reference shape",
    )
    reference_path = _safe_artifact_path(
        model_dir, artifact["path"], "code reference path"
    )
    reference_sha256 = _validate_sha256(
        artifact["sha256"], "code reference hash"
    )
    _require(
        reference_path.is_file(),
        f"code reference is missing: {reference_path}",
    )
    _require(
        sha256_file(reference_path) == reference_sha256,
        "code reference hash mismatch",
    )

    terminal_reference = manifest["terminal_reference"]
    _require(
        isinstance(terminal_reference, dict),
        "terminal reference must be an object",
    )
    _require(
        set(terminal_reference) == {"path", "sha256", "dtype", "shape"},
        "wrong terminal reference fields",
    )
    _require(
        terminal_reference["path"] == ORACLE_TERMINAL_ARTIFACT,
        "wrong terminal reference path",
    )
    _require(
        terminal_reference["dtype"] == "float32",
        "terminal reference must be FLOAT32",
    )
    _require(
        terminal_reference["shape"] == list(facts.terminal_output.shape),
        "wrong terminal reference shape",
    )
    terminal_reference_path = _safe_artifact_path(
        model_dir,
        terminal_reference["path"],
        "terminal reference path",
    )
    terminal_reference_sha256 = _validate_sha256(
        terminal_reference["sha256"], "terminal reference hash"
    )
    _require(
        terminal_reference_path.is_file(),
        f"terminal reference is missing: {terminal_reference_path}",
    )
    _require(
        sha256_file(terminal_reference_path) == terminal_reference_sha256,
        "terminal reference hash mismatch",
    )

    canonical = manifest["canonical_code_output"]
    _require(
        isinstance(canonical, dict), "canonical code output must be an object"
    )
    _require(
        set(canonical) == {"path", "sha256", "equal_to_reference"},
        "wrong canonical code output binding fields",
    )
    _require(
        canonical["path"] == facts.output.source,
        "oracle binds the wrong canonical code output",
    )
    _require(
        canonical["equal_to_reference"] is True,
        "oracle manifest does not declare code equality",
    )
    canonical_path = _safe_artifact_path(
        model_dir, canonical["path"], "canonical code output path"
    )
    canonical_sha256 = _validate_sha256(
        canonical["sha256"], "canonical code output hash"
    )
    _require(
        canonical_path.is_file(),
        f"canonical code output is missing: {canonical_path}",
    )
    _require(
        sha256_file(canonical_path) == canonical_sha256,
        "canonical code output hash mismatch",
    )

    canonical_terminal = manifest["canonical_terminal_output"]
    _require(
        isinstance(canonical_terminal, dict),
        "canonical terminal output must be an object",
    )
    _require(
        set(canonical_terminal) == {"path", "sha256", "equal_to_reference"},
        "wrong canonical terminal output binding fields",
    )
    _require(
        canonical_terminal["path"] == facts.terminal_output.source,
        "oracle binds the wrong canonical terminal output",
    )
    _require(
        canonical_terminal["equal_to_reference"] is True,
        "oracle manifest does not declare terminal equality",
    )
    canonical_terminal_path = _safe_artifact_path(
        model_dir,
        canonical_terminal["path"],
        "canonical terminal output path",
    )
    canonical_terminal_sha256 = _validate_sha256(
        canonical_terminal["sha256"], "canonical terminal output hash"
    )
    _require(
        canonical_terminal_path.is_file(),
        f"canonical terminal output is missing: {canonical_terminal_path}",
    )
    _require(
        sha256_file(canonical_terminal_path) == canonical_terminal_sha256,
        "canonical terminal output hash mismatch",
    )

    reference = _read_integer_container(
        reference_path, facts.output.shape, np.dtype(np.int8)
    )
    canonical_values = _read_integer_container(
        canonical_path, facts.output.shape, np.dtype(np.int8)
    )
    _require(
        np.array_equal(reference, canonical_values),
        "independent code reference differs from canonical fused output",
    )
    terminal_values = _read_float_container(
        terminal_reference_path, facts.terminal_output.shape
    )
    canonical_terminal_values = _read_float_container(
        canonical_terminal_path, facts.terminal_output.shape
    )
    _require(
        np.array_equal(terminal_values, canonical_terminal_values),
        "independent terminal reference differs from canonical terminal output",
    )
    expected_terminal = np.multiply(
        reference.astype(np.float32),
        np.float32(facts.output_scale),
        dtype=np.float32,
    )
    _require(
        np.array_equal(terminal_values, expected_terminal),
        "terminal reference is not dequantized code output",
    )
    binding = {
        "manifest": {
            "path": ORACLE_MANIFEST,
            "sha256": sha256_file(manifest_path),
        },
        "code_reference": {
            "path": ORACLE_ARTIFACT,
            "sha256": reference_sha256,
            "producer": ORACLE_PRODUCER,
            "scheme_id": SCHEME_ID,
            "qdq_form": QDQ_FORM,
            "container_dtype": "float32",
            "logical_dtype": "int8",
            "shape": list(facts.output.shape),
        },
        "terminal_reference": {
            "path": ORACLE_TERMINAL_ARTIFACT,
            "sha256": terminal_reference_sha256,
            "producer": ORACLE_PRODUCER,
            "scheme_id": SCHEME_ID,
            "qdq_form": QDQ_FORM,
            "dtype": "float32",
            "shape": list(facts.terminal_output.shape),
        },
        "canonical_code_output": {
            "path": facts.output.source,
            "sha256": canonical_sha256,
            "verified_equal": True,
        },
        "canonical_terminal_output": {
            "path": facts.terminal_output.source,
            "sha256": canonical_terminal_sha256,
            "verified_equal": True,
        },
        "qparams": expected_qparams,
    }
    return reference, terminal_values, binding


def pack_activation(logical: np.ndarray) -> np.ndarray:
    _require(logical.ndim == 4, "activation must be four-dimensional")
    return np.ascontiguousarray(logical.transpose(0, 2, 3, 1))


def unpack_activation(physical: np.ndarray) -> np.ndarray:
    _require(physical.ndim == 4, "physical activation must be four-dimensional")
    return np.ascontiguousarray(physical.transpose(0, 3, 1, 2))


def pack_weight(logical: np.ndarray) -> np.ndarray:
    _require(logical.ndim == 4, "weight must be four-dimensional")
    outputs, inputs, kernel_rows, kernel_columns = logical.shape
    physical = logical.transpose(2, 3, 1, 0)
    return np.ascontiguousarray(
        physical.reshape(kernel_rows * kernel_columns * inputs, outputs)
    )


def unpack_weight(
    physical: np.ndarray, logical_shape: tuple[int, int, int, int]
) -> np.ndarray:
    outputs, inputs, kernel_rows, kernel_columns = logical_shape
    expected = (kernel_rows * kernel_columns * inputs, outputs)
    _require(physical.shape == expected, "physical weight has wrong shape")
    return np.ascontiguousarray(
        physical.reshape(
            kernel_rows, kernel_columns, inputs, outputs
        ).transpose(3, 2, 0, 1)
    )


def _mapping_rows(role: str, logical: np.ndarray) -> list[dict[str, int | str]]:
    physical = (
        pack_weight(logical) if role == "weight" else pack_activation(logical)
    )
    rows = []
    for logical_index in range(logical.size):
        if role == "weight":
            output, channel, kernel_row, kernel_column = np.unravel_index(
                logical_index, logical.shape
            )
            physical_index = (
                (kernel_row * logical.shape[3] + kernel_column)
                * logical.shape[1]
                + channel
            ) * logical.shape[0] + output
        else:
            batch, channel, row, column = np.unravel_index(
                logical_index, logical.shape
            )
            physical_index = (
                (batch * logical.shape[2] + row) * logical.shape[3] + column
            ) * logical.shape[1] + channel
        rows.append(
            {
                "role": role,
                "logical_linear_index": logical_index,
                "physical_linear_index": physical_index,
                "value": int(physical.reshape(-1)[physical_index]),
            }
        )
    return rows


def _validate_mapping_rows(
    rows: list[dict[str, int | str]],
    logical: dict[str, np.ndarray],
    physical: dict[str, np.ndarray],
) -> None:
    expected_roles = {"input", "weight", "output", "output_oracle"}
    _require(set(logical) == expected_roles, "wrong logical layout roles")
    _require(set(physical) == expected_roles, "wrong physical layout roles")
    _require(
        len(rows) == sum(value.size for value in logical.values()),
        "layout map has the wrong total row count",
    )
    for role in sorted(expected_roles):
        selected = [row for row in rows if row["role"] == role]
        size = logical[role].size
        _require(len(selected) == size, f"wrong layout row count: {role}")
        logical_indices = [int(row["logical_linear_index"]) for row in selected]
        physical_indices = [
            int(row["physical_linear_index"]) for row in selected
        ]
        _require(
            sorted(logical_indices) == list(range(size)),
            f"logical layout map is not bijective: {role}",
        )
        _require(
            sorted(physical_indices) == list(range(size)),
            f"physical layout map is not bijective: {role}",
        )
        logical_flat = logical[role].reshape(-1)
        physical_flat = physical[role].reshape(-1)
        for row in selected:
            logical_index = int(row["logical_linear_index"])
            physical_index = int(row["physical_linear_index"])
            value = int(row["value"])
            _require(
                value == int(logical_flat[logical_index]),
                f"logical layout value mismatch: {role}",
            )
            _require(
                value == int(physical_flat[physical_index]),
                f"physical layout value mismatch: {role}",
            )


def _array_initializer(
    name: str,
    values: np.ndarray,
    *,
    mutable: bool = False,
    accelerator_payload: bool = True,
) -> str:
    flattened = values.reshape(-1)
    lines = []
    for offset in range(0, flattened.size, 16):
        lines.append(
            "    "
            + ", ".join(
                str(int(value)) for value in flattened[offset : offset + 16]
            )
        )
    qualifier = "" if mutable else "const "
    alignment = "row_align(1)" if accelerator_payload else "row_align(1)"
    return (
        f"{qualifier}elem_t {name}[{flattened.size}] {alignment} = {{\n"
        + ",\n".join(lines)
        + "\n};\n"
    )


def _render_payloads(payloads: dict[str, np.ndarray]) -> str:
    return """/* Generated from canonical Voyager logical tensor dumps. */
#ifndef VOYAGER_GEMMINI_PAYLOADS_H
#define VOYAGER_GEMMINI_PAYLOADS_H
#include "include/gemmini.h"

""" + "\n".join(
        [
            _array_initializer("voyager_input", payloads["input"]),
            _array_initializer("voyager_weights", payloads["weight"]),
            _array_initializer(
                "voyager_expected",
                payloads["output_oracle"],
                accelerator_payload=False,
            ),
            _array_initializer(
                "voyager_output",
                payloads["output"],
                mutable=True,
            ),
            "#endif\n",
        ]
    )


def _c_value(value: Any) -> str:
    if type(value) is bool:
        return "true" if value else "false"
    return str(value)


def _render_runner(projection: NativeProjection, element_count: int) -> str:
    field = {name: item.value for name, item in projection.fields.items()}
    arguments = [
        field[name]
        for name in (
            "batch_size",
            "in_row_dim",
            "in_col_dim",
            "in_channels",
            "out_channels",
            "out_row_dim",
            "out_col_dim",
            "stride",
            "input_dilation",
            "kernel_dilation",
            "padding",
            "kernel_dim",
            "wrot180",
            "trans_output_1203",
            "trans_input_3120",
            "trans_weight_1203",
            "trans_weight_0132",
        )
    ]
    prefix = ", ".join(_c_value(value) for value in arguments)
    return f"""/* Generated CPU-launch-only Gemmini bare-metal runner. */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "include/gemmini.h"
#include "payloads.h"

__attribute__((noinline, noclone, used))
void voyager_gemmini_operation(void) {{
    {projection.helper}(
        {prefix},
        {field['input']}, {field['weights']}, {field['bias']},
        {field['output']},
        {field['activation']}, {field['acc_scale']},
        {field['pool_size']}, {field['pool_stride']},
        {field['pool_padding']}, {field['dataflow']});
}}

int main(void) {{
    int exact_mismatches = 0;
    int tolerance_mismatches = 0;
    int first_mismatch = -1;
    int max_abs_error = 0;
    uint64_t squared_error = 0;

    for (int index = 0; index < {element_count}; ++index) {{
        voyager_output[index] = 0;
    }}
    gemmini_flush(0);
    voyager_gemmini_operation();
    gemmini_fence();

    for (int index = 0; index < {element_count}; ++index) {{
        int actual = voyager_output[index];
        int expected = voyager_expected[index];
        int difference = actual - expected;
        int absolute = difference < 0 ? -difference : difference;
        printf("OUTPUT %d %d\\n", index, actual);
        if (difference != 0) {{
            if (first_mismatch < 0) first_mismatch = index;
            ++exact_mismatches;
        }}
        if (absolute > 1) ++tolerance_mismatches;
        if (absolute > max_abs_error) max_abs_error = absolute;
        squared_error += (uint64_t)(difference * difference);
    }}
    if (first_mismatch < 0) {{
        printf("SUMMARY count={element_count} exact_mismatches=%d "
               "tolerance_mismatches=%d first=none max_abs=%d "
               "squared_error=%llu\\n",
               exact_mismatches, tolerance_mismatches, max_abs_error,
               (unsigned long long)squared_error);
    }} else {{
        printf("SUMMARY count={element_count} exact_mismatches=%d "
               "tolerance_mismatches=%d first=%d max_abs=%d "
               "squared_error=%llu\\n",
               exact_mismatches, tolerance_mismatches, first_mismatch,
               max_abs_error,
               (unsigned long long)squared_error);
    }}
    return tolerance_mismatches == 0 ? 0 : 1;
}}
"""


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_generated_inventory(
    root: Path, generated: list[Path], *, manifest_written: bool
) -> None:
    relative = [path.relative_to(root).as_posix() for path in generated]
    _require(
        len(relative) == len(set(relative)), "duplicate generated artifact"
    )
    _require(
        set(relative) == GENERATED_ARTIFACT_PATHS,
        "wrong generated artifact inventory",
    )
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected = set(GENERATED_ARTIFACT_PATHS)
    if manifest_written:
        expected.add("conversion_manifest.json")
    _require(actual == expected, "staging directory has undeclared artifacts")


def convert(
    model_dir: Path | str,
    output_dir: Path | str,
    contract_path: Path | str,
    contract_sha256: str,
) -> dict[str, Any]:
    """Validate and atomically convert one canonical registration anchor."""
    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir).resolve()
    contract_path = Path(contract_path).resolve()
    _require(model_dir.is_dir(), f"canonical directory is missing: {model_dir}")
    _require(not output_dir.exists(), f"output must be fresh: {output_dir}")
    _require(contract_path.is_file(), f"contract is missing: {contract_path}")
    _validate_sha256(contract_sha256, "injection contract hash")
    _require(
        contract_sha256 == INJECTION_CONTRACT_SHA256,
        "wrong finalized injection contract hash",
    )
    _require(
        sha256_file(contract_path) == contract_sha256,
        "injection contract hash mismatch",
    )
    native_dependencies = _native_dependencies()
    facts = _validate_model(model_dir)
    projection = _native_projection(facts)
    oracle, terminal_oracle, oracle_binding = _load_oracle(model_dir, facts)
    del terminal_oracle
    input_path = _safe_artifact_path(
        model_dir, facts.model_input.source, "canonical input path"
    )
    weight_path = _safe_artifact_path(
        model_dir, facts.weight.source, "canonical weight path"
    )
    logical = {
        "input": _materialize_input(
            input_path,
            facts.input.shape,
            facts.input_scale,
            facts.input_zero_point,
            facts.rounding,
            facts.quant_min,
            facts.quant_max,
        ),
        "weight": _read_integer_container(
            weight_path,
            facts.weight.shape,
            np.dtype(np.int8),
        ),
        "output": np.zeros(facts.output.shape, dtype=np.int8),
        "output_oracle": oracle,
    }
    physical = {
        "input": pack_activation(logical["input"]),
        "weight": pack_weight(logical["weight"]),
        "output": pack_activation(logical["output"]),
        "output_oracle": pack_activation(logical["output_oracle"]),
    }
    _require(
        np.array_equal(unpack_activation(physical["input"]), logical["input"]),
        "input layout round trip failed",
    )
    _require(
        np.array_equal(
            unpack_weight(physical["weight"], facts.weight.shape),
            logical["weight"],
        ),
        "weight layout round trip failed",
    )
    _require(
        np.array_equal(
            unpack_activation(physical["output"]), logical["output"]
        ),
        "output layout round trip failed",
    )
    _require(
        np.array_equal(
            unpack_activation(physical["output_oracle"]),
            logical["output_oracle"],
        ),
        "output layout round trip failed",
    )
    mapping_rows = []
    for role in ("input", "weight", "output", "output_oracle"):
        mapping_rows.extend(_mapping_rows(role, logical[role]))
    _validate_mapping_rows(mapping_rows, logical, physical)
    _require(len(mapping_rows) == 3072, "layout map must have 3,072 rows")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-", dir=output_dir.parent
        )
    )
    try:
        payload_dir = staging / "payloads"
        payload_dir.mkdir()
        for role, values in physical.items():
            path = payload_dir / f"{role}.bin"
            path.write_bytes(values.tobytes(order="C"))
        with (staging / "layout-map.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=(
                    "role",
                    "logical_linear_index",
                    "physical_linear_index",
                    "value",
                ),
            )
            writer.writeheader()
            writer.writerows(mapping_rows)
        (staging / "payloads.h").write_text(_render_payloads(physical))
        (staging / "runner.c").write_text(
            _render_runner(projection, physical["output_oracle"].size)
        )
        emitted_calls = {
            "schema_version": 1,
            "semantic_source": {
                "path": "model.txt",
                "sha256": sha256_file(model_dir / "model.txt"),
            },
            "operation_id": OPERATION_ID,
            "emitted_operation_helper_count": 1,
            "emitted_backend_calls": [
                {
                    "ordinal": 0,
                    "helper": projection.helper,
                    "wrapper": "voyager_gemmini_operation",
                    "model_operation_name": facts.fused_name,
                    "fused_member_names": list(facts.member_names),
                }
            ],
            "lifecycle_calls": [
                {"ordinal": 0, "helper": "gemmini_flush", "argument": 0},
                {"ordinal": 1, "helper": "gemmini_fence"},
            ],
        }
        _write_json(staging / "emitted-calls.json", emitted_calls)
        generated = sorted(
            [
                staging / "layout-map.csv",
                staging / "payloads.h",
                staging / "runner.c",
                staging / "emitted-calls.json",
                *(payload_dir / f"{role}.bin" for role in physical),
            ],
            key=lambda path: path.relative_to(staging).as_posix(),
        )
        _validate_generated_inventory(
            staging, generated, manifest_written=False
        )
        manifest = {
            "schema_version": 1,
            "status": "converted",
            "performance_claimed": False,
            "contract_id": CONTRACT_ID,
            "contract": {
                "path": str(contract_path),
                "sha256": contract_sha256,
            },
            "native_dependencies": native_dependencies,
            "canonical_directory": str(model_dir),
            "canonical_artifacts": {
                "model.txt": {"sha256": sha256_file(model_dir / "model.txt")},
                **{
                    fact.source: {
                        "sha256": sha256_file(model_dir / fact.source),
                        "semantic_role": fact.role,
                    }
                    for fact in (
                        facts.model_input,
                        facts.weight,
                        facts.output,
                        facts.terminal_output,
                        facts.input_scale_tensor,
                        facts.accumulator_scale_tensor,
                        facts.output_scale_tensor,
                        facts.terminal_scale_tensor,
                    )
                },
                ORACLE_MANIFEST: {
                    "sha256": oracle_binding["manifest"]["sha256"],
                    "semantic_role": "independent-oracle-manifest",
                },
                ORACLE_ARTIFACT: {
                    "sha256": oracle_binding["code_reference"]["sha256"],
                    "semantic_role": "independent-code-oracle",
                },
                ORACLE_TERMINAL_ARTIFACT: {
                    "sha256": oracle_binding["terminal_reference"]["sha256"],
                    "semantic_role": "independent-terminal-oracle",
                },
            },
            "oracle_binding": oracle_binding,
            "operation_id": OPERATION_ID,
            "semantic_region": {
                "serialized_targets": list(FUSED_SERIALIZED_TARGETS),
                "semantic_members": list(FUSED_SEMANTIC_MEMBERS),
                "fused_name": facts.fused_name,
                "terminal_target": "quantized_ops::dequantize",
                "terminal_name": facts.terminal_name,
            },
            "operation_facts": asdict(facts),
            "input_quantize_projection": {
                "source_dtype": "float32",
                "destination_dtype": "int8",
                "formula": "clip(rint(float32(x / input_scale)))",
                "rounding": "nearest-even",
                "packing_rounding_domain": "float32",
                "shared_qmap_rounding_domain": "bfloat16",
                "bit_accuracy_limitation": (
                    "projection uses FP32 RNE while shared qmap normalizes "
                    "to BF16 before RNE"
                ),
                "serialized_rounding": facts.rounding,
                "saturation_bounds": [facts.quant_min, facts.quant_max],
                "input_scale": facts.input_scale,
                "input_zero_point": facts.input_zero_point,
                "runtime_float_input_supported": False,
                "projection_time_only": True,
                "verified_static_input_only": True,
            },
            "quantization_facts": {
                "qparams": _qparams(facts),
                "accumulator_scale": facts.accumulator_scale,
                "helper_acc_scale": float(
                    np.float32(
                        np.float32(facts.accumulator_scale)
                        / np.float32(facts.output_scale)
                    )
                ),
                "scale_representation": "float32",
                "zero_points_explicit_in_oracle": True,
                "zero_points_implicit_zero_in_shared_ops": True,
                "general_bit_exact_equivalence": False,
                "acceptance_scope": "fixed-static-vector-full-output-atol-1",
            },
            "native_projection": {
                "helper": projection.helper,
                "fields": {
                    name: asdict(value)
                    for name, value in projection.fields.items()
                },
                "schedule_authority": "endpoint",
                "schedule_fidelity": "helper_selected",
                "compiler_schedule_consumed": False,
                "helper_derived_fields": {
                    "in_stride": facts.input.shape[1],
                    "weight_stride": facts.weight.shape[0],
                    "out_stride": facts.output.shape[1],
                },
            },
            "emitted_call_artifact": "emitted-calls.json",
            "emitted_operation_helper_count": 1,
            "payloads": {
                role: {
                    "endpoint_consumed": role != "output_oracle",
                    "oracle_only": role == "output_oracle",
                    "logical_shape": list(logical[role].shape),
                    "physical_shape": list(physical[role].shape),
                    "logical_sha256": sha256_bytes(logical[role].tobytes()),
                    "physical_sha256": sha256_bytes(physical[role].tobytes()),
                    "abi_bytes": physical[role].nbytes,
                    "round_trip_equal": True,
                    "source_kind": (
                        "zero-initialized-native-destination"
                        if role == "output"
                        else (
                            "independent-code-oracle"
                            if role == "output_oracle"
                            else (
                                "serialized-input-quantize-projection"
                                if role == "input"
                                else "canonical-tensor-dump"
                            )
                        )
                    ),
                    "canonical_container_dtype": (
                        None if role == "output" else "float32"
                    ),
                    "native_dtype": "int8",
                }
                for role in physical
            },
            "layout_mapping_rows": len(mapping_rows),
            "layout_mapping_role_counts": {
                role: logical[role].size for role in logical
            },
            "generated_artifacts": [
                _artifact(path, staging) for path in generated
            ],
        }
        _write_json(staging / "conversion_manifest.json", manifest)
        _validate_generated_inventory(staging, generated, manifest_written=True)
        os.replace(staging, output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--contract-sha256", required=True)
    arguments = parser.parse_args(argv)
    try:
        manifest = convert(
            arguments.model_dir,
            arguments.output_dir,
            arguments.contract,
            arguments.contract_sha256,
        )
    except (ConversionError, OSError, ValueError) as error:
        print(f"conversion failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "operation_id": manifest["operation_id"],
                "emitted_operation_helper_count": 1,
                "layout_mapping_rows": manifest["layout_mapping_rows"],
                "output": str(arguments.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

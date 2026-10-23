#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Project shared workload regions to executable Gemmini collateral.

This target adapter deliberately leaves shared operator semantics and the R
registration converter unchanged. It validates one canonical workload region,
projects that region onto a public Gemmini helper, and can execute the emitted
collateral through the pinned native stack.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from google.protobuf import text_format

from conversion.gemmini import convert as anchor
from conversion.gemmini import run as native_run

WORKLOAD_CONTRACT_SHA256 = (
    "9256107f68a65fce92791f2736b660e70dc902496efeacecfc0a89ae2cce4a30"
)
OPERATION_ID = "aten::conv2d"
HELPER = "tiled_conv_auto"


@dataclass(frozen=True)
class RegionFacts:
    """Canonical shared facts consumed by one W-stage helper call."""

    fused_name: str
    member_names: tuple[str, ...]
    model_input: anchor.TensorFact
    input_is_quantized: bool
    input_codes: anchor.TensorFact
    weight: anchor.TensorFact
    bias: anchor.TensorFact
    accumulator_codes: anchor.TensorFact
    accumulator_float: anchor.TensorFact
    conv_output_codes: anchor.TensorFact
    output_codes: anchor.TensorFact
    input_scale_tensor: anchor.TensorFact | None
    accumulator_scale_tensor: anchor.TensorFact
    conv_output_scale_tensor: anchor.TensorFact
    output_scale_tensor: anchor.TensorFact
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int
    activation: str
    pool_size: int
    pool_stride: int
    pool_padding: int
    input_scale: float | None
    accumulator_scale: float
    conv_output_scale: float
    output_scale: float
    weight_scale: float | None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise anchor.ConversionError(message)


def _tensor_fact(
    tensor: Any,
    role: str,
    *,
    require_memory: bool = False,
    dtypes: frozenset[str] = frozenset({"float32", "int8", "int32"}),
) -> anchor.TensorFact:
    _require(bool(tensor.node), f"{role} tensor has no node identity")
    _require(
        anchor.TENSOR_NODE_RE.fullmatch(tensor.node) is not None,
        f"{role} tensor node is unsafe",
    )
    _require(bool(tensor.shape), f"{role} tensor has no shape")
    _require(tensor.dtype in dtypes, f"{role} has wrong dtype")
    has_memory = tensor.HasField("memory")
    _require(has_memory or not require_memory, f"{role} has no DRAM fact")
    if has_memory:
        memory = tensor.memory
        _require(
            memory.HasField("partition") and memory.partition == 0,
            f"{role} is not in DRAM partition zero",
        )
        _require(memory.HasField("address"), f"{role} DRAM address is missing")
        _require(
            memory.HasField("instance") and memory.instance == "DRAM",
            f"{role} is not in DRAM",
        )
    _require(not tensor.tiled_shape, f"{role} carries tiled shape metadata")
    _require(not tensor.tile_strides, f"{role} carries tile strides")
    _require(not tensor.HasField("scratchpad"), f"{role} carries scratchpad")
    _require(not tensor.HasField("dequant"), f"{role} carries dequant metadata")
    _require(not tensor.HasField("reshape"), f"{role} carries reshape metadata")
    _require(not tensor.is_none, f"{role} cannot be null")
    return anchor.TensorFact(
        role=role,
        node=tensor.node,
        shape=tuple(int(value) for value in tensor.shape),
        dtype=tensor.dtype,
        source=f"tensor_files/{tensor.node}.bin",
        memory_partition=int(tensor.memory.partition) if has_memory else None,
        memory_address=int(tensor.memory.address) if has_memory else None,
        memory_instance=tensor.memory.instance if has_memory else None,
    )


def _tensor_argument(
    argument: Any,
    name: str,
    *,
    require_memory: bool = False,
    dtypes: frozenset[str] = frozenset({"float32", "int8", "int32"}),
) -> anchor.TensorFact:
    kind, value = anchor._argument(argument, name)
    _require(kind == "tensor", f"{name} must be a tensor")
    return _tensor_fact(
        value,
        name,
        require_memory=require_memory,
        dtypes=dtypes,
    )


def _wire(fact: anchor.TensorFact) -> tuple[Any, ...]:
    return (
        fact.node,
        fact.shape,
        fact.dtype,
        fact.memory_partition,
        fact.memory_address,
        fact.memory_instance,
    )


def _operation(
    operation: Any,
    *,
    target: str,
    kwargs: frozenset[str],
    label: str,
) -> None:
    _require(bool(operation.name), f"{label} has no name")
    _require(operation.op == "call_function", f"{label} is not call_function")
    _require(operation.target == target, f"{label} has wrong target")
    _require(not operation.args, f"{label} has positional arguments")
    _require(set(operation.kwargs) == kwargs, f"{label} has wrong kwargs")


def _scale(model_dir: Path, fact: anchor.TensorFact, label: str) -> float:
    _require(fact.shape == (1,), f"{label} is not scalar")
    _require(fact.dtype == "float32", f"{label} is not FLOAT32")
    path = anchor._safe_artifact_path(model_dir, fact.source, f"{label} path")
    value = float(anchor._read_float_container(path, fact.shape).reshape(-1)[0])
    _require(math.isfinite(value) and value > 0, f"{label} is not positive")
    return float(np.float32(value))


def _load_workload_region_facts(
    model_dir: Path,
    fused_name: str,
) -> RegionFacts:
    """Read one helper-representable Conv region from a canonical workload."""
    proto = anchor._load_proto()
    model = proto.Model()
    text_format.Parse((model_dir / "model.txt").read_text(), model)
    candidates = [
        record
        for record in model.ops
        if record.WhichOneof("op_type") == "fused_op"
        and record.fused_op.name == fused_name
    ]
    _require(len(candidates) == 1, "requested fused Conv2d region is missing")
    fused_record = candidates[0]
    fused = fused_record.fused_op
    members = tuple(fused.op_list)
    _require(
        len(members) in {3, 4, 5, 6},
        "fused region has an unsupported member count",
    )
    has_input_quantize = members[0].target == "quantize"
    conv_index = 1 if has_input_quantize else 0
    _require(conv_index + 2 < len(members), "fused Conv core is incomplete")
    input_q = members[0] if has_input_quantize else None
    conv, acc_dq, conv_output_q = members[conv_index : conv_index + 3]
    epilogue = members[conv_index + 3 :]
    _require(len(epilogue) in {0, 1, 2}, "fused Conv epilogue is malformed")
    if input_q is not None:
        _operation(
            input_q,
            target="quantize",
            kwargs=anchor.QUANTIZE_KWARGS,
            label="input quantize",
        )
    _operation(
        conv,
        target="conv2d",
        kwargs=frozenset(
            {
                "input",
                "weight",
                "bias",
                "stride",
                "padding",
                "dilation",
                "groups",
            }
        ),
        label="shared conv2d",
    )
    _operation(
        acc_dq,
        target="dequantize",
        kwargs=anchor.DEQUANTIZE_KWARGS,
        label="accumulator dequantize",
    )
    _operation(
        conv_output_q,
        target="quantize",
        kwargs=anchor.QUANTIZE_KWARGS,
        label="Conv output quantize",
    )

    activation_op = None
    pool_op = None
    output_q = conv_output_q
    if epilogue:
        activation_op = epilogue[0]
        pool_op = epilogue[1] if len(epilogue) == 2 else None
        _operation(
            activation_op,
            target="relu",
            kwargs=frozenset({"input"}),
            label="quantized-value ReLU epilogue",
        )
        if pool_op is not None:
            _operation(
                pool_op,
                target="max_pool2d",
                kwargs=frozenset(
                    {
                        "input",
                        "kernel_size",
                        "stride",
                        "padding",
                        "dilation",
                        "ceil_mode",
                    }
                ),
                label="MaxPool epilogue",
            )

    def require_symmetric_quantize(operation: Any, label: str) -> None:
        _require(
            anchor._string_argument(operation.kwargs["rounding"], label)
            == "nearest",
            f"{label} is not nearest",
        )
        _require(
            (
                anchor._float_argument(operation.kwargs["quant_min"], label),
                anchor._float_argument(operation.kwargs["quant_max"], label),
            )
            == (-128.0, 127.0),
            f"{label} is not signed INT8",
        )

    if input_q is not None:
        require_symmetric_quantize(input_q, "input quantize")
    require_symmetric_quantize(conv_output_q, "Conv output quantize")
    input_codes = _tensor_argument(conv.kwargs["input"], "conv input codes")
    _require(input_codes.dtype == "int8", "Conv input codes are not INT8")
    if input_q is not None:
        model_input = _tensor_argument(
            input_q.kwargs["input"],
            "input quantize input",
            require_memory=True,
            dtypes=frozenset({"float32"}),
        )
        input_scale = _tensor_argument(
            input_q.kwargs["scale"],
            "input scale",
        )
        _require(
            input_codes.node == input_q.name,
            "Conv input does not consume input Q",
        )
    else:
        model_input = input_codes
        input_scale = None
        _require(
            model_input.memory_instance == "DRAM",
            "connected Conv input is not materialized in DRAM",
        )
    weight = _tensor_argument(
        conv.kwargs["weight"], "conv weight", require_memory=True
    )
    bias = _tensor_argument(
        conv.kwargs["bias"], "conv bias", require_memory=True
    )
    _require(weight.dtype == "int8", "Conv weight is not INT8")
    _require(bias.dtype == "int32", "folded Conv bias is not INT32")
    parameters = {
        tensor.node: _tensor_fact(
            tensor, f"parameter {tensor.node}", require_memory=True
        )
        for tensor in model.parameters
    }
    _require(weight.node in parameters, "Conv weight is not a parameter")
    _require(bias.node in parameters, "Conv bias is not a parameter")
    _require(
        _wire(parameters[weight.node]) == _wire(weight),
        "weight wire differs from parameter",
    )
    _require(
        _wire(parameters[bias.node]) == _wire(bias),
        "bias wire differs from parameter",
    )
    _require(
        bias.shape == (weight.shape[0],),
        "bias shape differs from output channels",
    )

    accumulator_codes = _tensor_argument(
        acc_dq.kwargs["input"], "accumulator codes"
    )
    _require(
        accumulator_codes.node == conv.name,
        "accumulator does not consume Conv",
    )
    accumulator_scale = _tensor_argument(
        acc_dq.kwargs["scale"], "accumulator scale"
    )
    accumulator_float = _tensor_argument(
        conv_output_q.kwargs["input"], "accumulator float"
    )
    _require(
        accumulator_float.node == acc_dq.name,
        "Conv output Q does not consume accumulator DQ",
    )
    conv_output_scale = _tensor_argument(
        conv_output_q.kwargs["scale"],
        "Conv output scale",
    )
    if activation_op is not None:
        conv_output_codes = _tensor_argument(
            activation_op.kwargs["input"],
            "Conv output codes",
        )
        _require(
            conv_output_codes.node == conv_output_q.name,
            "quantized-value ReLU does not consume Conv output Q",
        )
        if pool_op is not None:
            pool_input = _tensor_argument(pool_op.kwargs["input"], "pool input")
            _require(
                pool_input.node == activation_op.name,
                "MaxPool does not consume ReLU",
            )
    else:
        conv_output_codes = _tensor_fact(
            fused_record.output,
            "Conv output codes",
            require_memory=True,
        )
    output_scale = _tensor_argument(output_q.kwargs["scale"], "output scale")
    output_codes = _tensor_fact(
        fused_record.output, "fused output", require_memory=True
    )
    _require(output_codes.node == fused.name, "fused output wire differs")
    for fact, label, dtype in (
        # The shared Aten Conv produces integral accumulator codes in a
        # FLOAT32 semantic container.  The following DQ supplies their scale;
        # this is not the narrowed INT8 Conv-output Q tensor.
        (accumulator_codes, "accumulator codes", "float32"),
        (accumulator_float, "accumulator float", "float32"),
        (conv_output_codes, "Conv output codes", "int8"),
        (output_codes, "output codes", "int8"),
    ):
        _require(fact.dtype == dtype, f"{label} has wrong dtype")
    _require(
        accumulator_codes.shape == accumulator_float.shape,
        "Conv accumulator shapes differ",
    )
    _require(
        conv_output_codes.shape == accumulator_codes.shape,
        "Conv qcode shape differs from accumulator",
    )

    _require(len(model_input.shape) == 4, "input must be NCHW")
    _require(len(weight.shape) == 4, "weight must be OIHW")
    _require(len(output_codes.shape) == 4, "output must be NCHW")
    _require(
        model_input.shape[0] == output_codes.shape[0] == 1,
        "batch must be one",
    )
    _require(model_input.shape[1] == weight.shape[1], "input channels differ")
    _require(weight.shape[0] == output_codes.shape[1], "output channels differ")
    _require(weight.shape[2] == weight.shape[3], "kernel must be square")
    stride = anchor._pair_argument(conv.kwargs["stride"], "stride")
    padding = anchor._pair_argument(conv.kwargs["padding"], "padding")
    dilation = anchor._pair_argument(conv.kwargs["dilation"], "dilation")
    groups = anchor._integer_argument(conv.kwargs["groups"], "groups")
    _require(stride[0] == stride[1], "stride must be square")
    _require(padding[0] == padding[1], "padding must be square")
    _require(dilation == (1, 1), "W direct form requires unit dilation")
    _require(groups == 1, "W direct form requires groups one")

    input_scale_value = (
        None
        if input_scale is None
        else _scale(model_dir, input_scale, "input scale")
    )
    accumulator_scale_value = _scale(
        model_dir, accumulator_scale, "accumulator scale"
    )
    conv_output_scale_value = _scale(
        model_dir,
        conv_output_scale,
        "Conv output scale",
    )
    output_scale_value = _scale(model_dir, output_scale, "output scale")
    weight_scale = (
        None
        if input_scale_value is None
        else float(
            np.float32(
                np.float32(accumulator_scale_value)
                / np.float32(input_scale_value)
            )
        )
    )
    if weight_scale is not None:
        _require(
            math.isfinite(weight_scale) and weight_scale > 0,
            "weight scale is invalid",
        )

    activation = "RELU" if activation_op is not None else "NO_ACTIVATION"
    pool_size = 0
    pool_stride = 0
    pool_padding = 0
    if pool_op is not None:
        kernel = anchor._pair_argument(
            pool_op.kwargs["kernel_size"],
            "pool kernel",
        )
        pool_stride_pair = anchor._pair_argument(
            pool_op.kwargs["stride"],
            "pool stride",
        )
        pool_padding_pair = anchor._pair_argument(
            pool_op.kwargs["padding"],
            "pool padding",
        )
        pool_dilation = anchor._pair_argument(
            pool_op.kwargs["dilation"],
            "pool dilation",
        )
        _require(kernel[0] == kernel[1], "pool kernel must be square")
        _require(
            pool_stride_pair[0] == pool_stride_pair[1],
            "pool stride must be square",
        )
        _require(
            pool_padding_pair[0] == pool_padding_pair[1],
            "pool padding must be square",
        )
        _require(pool_dilation == (1, 1), "pool dilation must be one")
        ceil_kind, ceil_value = anchor._argument(
            pool_op.kwargs["ceil_mode"],
            "ceil mode",
        )
        _require(ceil_kind == "bool_value", "ceil mode must be a bool")
        _require(not bool(ceil_value), "ceil-mode MaxPool is unsupported")
        pool_size = kernel[0]
        pool_stride = pool_stride_pair[0]
        pool_padding = pool_padding_pair[0]
    return RegionFacts(
        fused_name=fused.name,
        member_names=tuple(operation.name for operation in members),
        model_input=model_input,
        input_is_quantized=input_q is None,
        input_codes=input_codes,
        weight=weight,
        bias=bias,
        accumulator_codes=accumulator_codes,
        accumulator_float=accumulator_float,
        conv_output_codes=conv_output_codes,
        output_codes=output_codes,
        input_scale_tensor=input_scale,
        accumulator_scale_tensor=accumulator_scale,
        conv_output_scale_tensor=conv_output_scale,
        output_scale_tensor=output_scale,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        activation=activation,
        pool_size=pool_size,
        pool_stride=pool_stride,
        pool_padding=pool_padding,
        input_scale=input_scale_value,
        accumulator_scale=accumulator_scale_value,
        conv_output_scale=conv_output_scale_value,
        output_scale=output_scale_value,
        weight_scale=weight_scale,
    )


def _read_int32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = anchor._read_float_container(path, shape)
    _require(np.all(np.isfinite(values)), "bias contains non-finite values")
    _require(np.all(values == np.rint(values)), "bias is not integral")
    _require(
        np.all(values >= np.iinfo(np.int32).min)
        and np.all(values <= np.iinfo(np.int32).max),
        "bias is outside INT32",
    )
    return values.astype(np.int32)


def _runtime_capture(
    capture_dir: Path,
    model_dir: Path,
    facts: RegionFacts,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load real transformed-region values captured before canonical dumping.

    The generic serializer uses shape propagation while it materializes
    per-operation files.  Those files are useful layout descriptors, but not
    execution oracles for a workload graph.  This capture is produced by one
    concrete execution of the same transformed graph and is bound to the
    canonical model text plus the exact fused-region name.
    """
    capture_dir = capture_dir.resolve()
    manifest = anchor._load_json(
        capture_dir / "runtime-capture.json",
        "runtime capture manifest",
    )
    _require(
        set(manifest)
        == {
            "schema_version",
            "producer",
            "canonical_model",
            "regions",
        },
        "runtime capture manifest fields differ",
    )
    _require(manifest["schema_version"] == 1, "runtime capture schema differs")
    producer = manifest["producer"]
    _require(
        isinstance(producer, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", producer) is not None,
        "runtime capture producer identity is invalid",
    )
    canonical_model = manifest["canonical_model"]
    _require(
        isinstance(canonical_model, dict),
        "runtime canonical model missing",
    )
    _require(
        canonical_model.get("path") == "model.txt",
        "runtime canonical model path differs",
    )
    _require(
        canonical_model.get("sha256")
        == anchor.sha256_file(model_dir / "model.txt"),
        "runtime capture is bound to another canonical model",
    )
    regions = manifest["regions"]
    _require(isinstance(regions, dict), "runtime capture regions missing")
    entry = regions.get(facts.fused_name)
    _require(isinstance(entry, dict), "runtime capture lacks requested region")
    _require(
        set(entry) == {"input", "output_codes"},
        "runtime capture region fields differ",
    )

    def load(
        entry_name: str,
        fact: anchor.TensorFact,
        logical_dtype: str,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        binding = entry[entry_name]
        _require(isinstance(binding, dict), f"runtime {entry_name} missing")
        _require(
            set(binding)
            == {
                "path",
                "sha256",
                "container_dtype",
                "logical_dtype",
                "shape",
            },
            f"runtime {entry_name} fields differ",
        )
        _require(
            binding["container_dtype"] == "float32",
            f"runtime {entry_name} container differs",
        )
        _require(
            binding["logical_dtype"] == logical_dtype,
            f"runtime {entry_name} dtype differs",
        )
        _require(
            binding["shape"] == list(fact.shape),
            f"runtime {entry_name} shape differs",
        )
        path = anchor._safe_artifact_path(
            capture_dir,
            binding["path"],
            f"runtime {entry_name} path",
        )
        _require(
            anchor.sha256_file(path) == binding["sha256"],
            f"runtime {entry_name} hash differs",
        )
        values = anchor._read_float_container(path, fact.shape)
        _require(
            np.all(np.isfinite(values)),
            f"runtime {entry_name} contains non-finite values",
        )
        return values, binding

    input_values, input_binding = load(
        "input",
        facts.model_input,
        "int8" if facts.input_is_quantized else "float32",
    )
    output_values, output_binding = load(
        "output_codes",
        facts.output_codes,
        "int8",
    )
    _require(
        np.all(output_values == np.rint(output_values)),
        "runtime output codes are not integral",
    )
    _require(
        np.all((-128 <= output_values) & (output_values <= 127)),
        "runtime output codes are outside INT8",
    )
    _require(
        int(np.count_nonzero(input_values)) > 0,
        "runtime region input is degenerate",
    )
    _require(
        int(np.count_nonzero(output_values)) > 0,
        "runtime region output is degenerate",
    )
    return (
        input_values.astype(np.float32),
        output_values.astype(np.int8),
        {
            "path": str(capture_dir),
            "producer": producer,
            "input": input_binding,
            "output_codes": output_binding,
        },
    )


def _quantize_input(values: np.ndarray, facts: RegionFacts) -> np.ndarray:
    if facts.input_is_quantized:
        _require(
            np.all(values == np.rint(values)),
            "connected input codes are not integral",
        )
        _require(
            np.all((-128 <= values) & (values <= 127)),
            "connected input codes are outside INT8",
        )
        return values.astype(np.int8)
    _require(facts.input_scale is not None, "FLOAT32 input has no scale")
    scaled = values.astype(np.float32) / np.float32(facts.input_scale)
    rounded = np.rint(scaled)
    return np.clip(rounded, -128, 127).astype(np.int8)


def _projection(facts: RegionFacts) -> dict[str, Any]:
    _, in_channels, input_rows, input_columns = facts.model_input.shape
    out_channels, weight_channels, kernel_rows, kernel_columns = (
        facts.weight.shape
    )
    _, result_channels, output_rows, output_columns = (
        facts.conv_output_codes.shape
    )
    _require(in_channels == weight_channels, "weight/input channels differ")
    _require(out_channels == result_channels, "weight/output channels differ")
    multiplier = np.float32(
        np.float32(facts.accumulator_scale) / np.float32(facts.output_scale)
    )
    _require(
        math.isfinite(float(multiplier)) and multiplier > 0, "bad helper scale"
    )
    return {
        "helper": HELPER,
        "fields": {
            "batch_size": 1,
            "in_row_dim": input_rows,
            "in_col_dim": input_columns,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "out_row_dim": output_rows,
            "out_col_dim": output_columns,
            "stride": facts.stride[0],
            "input_dilation": 1,
            "kernel_dilation": 1,
            "padding": facts.padding[0],
            "kernel_dim": kernel_rows,
            "wrot180": False,
            "trans_output_1203": False,
            "trans_input_3120": False,
            "trans_weight_1203": False,
            "trans_weight_0132": False,
            "input": "voyager_input",
            "weights": "voyager_weights",
            "bias": "voyager_bias",
            "output": "voyager_output",
            "activation": facts.activation,
            "acc_scale": float.hex(float(multiplier)) + "f",
            "pool_size": facts.pool_size,
            "pool_stride": facts.pool_stride,
            "pool_padding": facts.pool_padding,
            "dataflow": "WS",
        },
    }


def _initializer(name: str, values: np.ndarray, ctype: str) -> str:
    flattened = values.reshape(-1)
    body = ", ".join(str(int(value)) for value in flattened)
    declaration = f"static {ctype} {name}[{flattened.size}]"
    return f"{declaration} __attribute__((aligned(16))) = {{{body}}};"


def _render_payloads(payloads: dict[str, np.ndarray]) -> str:
    return "\n".join(
        (
            "#ifndef VOYAGER_GEMMINI_W_PAYLOADS_H",
            "#define VOYAGER_GEMMINI_W_PAYLOADS_H",
            '#include "include/gemmini.h"',
            _initializer("voyager_input", payloads["input"], "elem_t"),
            _initializer("voyager_weights", payloads["weight"], "elem_t"),
            _initializer("voyager_bias", payloads["bias"], "acc_t"),
            _initializer("voyager_expected", payloads["oracle"], "elem_t"),
            _initializer("voyager_output", payloads["output"], "elem_t"),
            "#endif",
            "",
        )
    )


def _render_runner(projection: dict[str, Any], element_count: int) -> str:
    fields = projection["fields"]
    arguments = ", ".join(
        (
            str(fields[name]).lower()
            if isinstance(fields[name], bool)
            else str(fields[name])
        )
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
    )
    return f"""#include <stdint.h>
#include <stdio.h>
#include "include/gemmini.h"
#include "payloads.h"

__attribute__((noinline, noclone, used))
void voyager_gemmini_operation(void) {{
  tiled_conv_auto(
      {arguments}, voyager_input, voyager_weights, voyager_bias,
      voyager_output, {fields["activation"]}, {fields["acc_scale"]},
      {fields["pool_size"]}, {fields["pool_stride"]},
      {fields["pool_padding"]}, {fields["dataflow"]});
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
  return tolerance == 0 ? 0 : 1;
}}
"""


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": anchor.sha256_file(path),
    }


def _convert_conv_region(
    model_dir: Path | str,
    output_dir: Path | str,
    contract: Path | str,
    contract_sha256: str,
    fused_name: str,
    runtime_capture_dir: Path | str,
) -> dict[str, Any]:
    """Project one named Conv2d region from a compiled workload model.

    The emitted helper receives real values captured from the transformed
    workload graph.  This is direct per-region coverage, not a connected
    connected workload execution.
    """
    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir).resolve()
    contract = Path(contract).resolve()
    runtime_capture_dir = Path(runtime_capture_dir).resolve()
    _require(model_dir.is_dir(), "canonical model directory is missing")
    _require(not output_dir.exists(), "conversion directory must be fresh")
    _require(contract.is_file(), "W injection contract is missing")
    _require(
        contract_sha256 == WORKLOAD_CONTRACT_SHA256,
        "wrong W injection contract hash",
    )
    _require(
        anchor.sha256_file(contract) == WORKLOAD_CONTRACT_SHA256,
        "W contract hash differs",
    )
    facts = _load_workload_region_facts(model_dir, fused_name)
    input_values, codes, runtime_capture = _runtime_capture(
        runtime_capture_dir,
        model_dir,
        facts,
    )
    input_codes = _quantize_input(input_values, facts)
    weight_values = anchor._read_integer_container(
        anchor._safe_artifact_path(
            model_dir, facts.weight.source, "weight path"
        ),
        facts.weight.shape,
        np.dtype(np.int8),
    )
    bias_values = _read_int32(
        anchor._safe_artifact_path(model_dir, facts.bias.source, "bias path"),
        facts.bias.shape,
    )
    payloads = {
        "input": anchor.pack_activation(input_codes),
        "weight": anchor.pack_weight(weight_values),
        "bias": bias_values,
        "oracle": anchor.pack_activation(codes),
        "output": np.zeros_like(anchor.pack_activation(codes)),
    }
    _require(
        np.array_equal(
            anchor.unpack_activation(payloads["input"]), input_codes
        ),
        "input packing round trip failed",
    )
    _require(
        np.array_equal(
            anchor.unpack_weight(payloads["weight"], facts.weight.shape),
            weight_values,
        ),
        "weight packing round trip failed",
    )
    projection = _projection(facts)
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
        (staging / "runner.c").write_text(
            _render_runner(projection, int(codes.size))
        )
        emitted = {
            "schema_version": 1,
            "semantic_source": {
                "path": "model.txt",
                "sha256": anchor.sha256_file(model_dir / "model.txt"),
            },
            "runtime_capture": runtime_capture,
            "operation_id": OPERATION_ID,
            "emitted_operation_helper_count": 1,
            "emitted_backend_calls": [
                {
                    "ordinal": 0,
                    "helper": HELPER,
                    "model_operation_name": facts.fused_name,
                    "fused_member_names": list(facts.member_names),
                }
            ],
            "lifecycle_calls": ["gemmini_flush", "gemmini_fence"],
        }
        anchor._write_json(staging / "emitted-calls.json", emitted)
        artifacts = [
            _artifact(path, staging)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        ]
        canonical_output = anchor._safe_artifact_path(
            model_dir, facts.output_codes.source, "canonical output path"
        )
        manifest = {
            "schema_version": 1,
            "status": "converted",
            "operation_id": OPERATION_ID,
            "contract": {"path": str(contract), "sha256": contract_sha256},
            "semantic_source": emitted["semantic_source"],
            "facts": asdict(facts),
            "projection": projection,
            "qparams": {
                "input_scale": facts.input_scale,
                "input_zero_point": 0,
                "weight_scale": facts.weight_scale,
                "weight_zero_point": 0,
                "output_scale": facts.output_scale,
                "output_zero_point": 0,
            },
            "compiler_layout_output": {
                "path": facts.output_codes.source,
                "sha256": anchor.sha256_file(canonical_output),
                "logical_dtype": "int8",
                "shape": list(facts.output_codes.shape),
            },
            "runtime_capture": runtime_capture,
            "accepted_scope": "captured-direct-conv-region-atol-1",
            "connected_workload_execution_claimed": False,
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


_OUTPUT_RE = re.compile(r"OUTPUT (?P<index>\d+) (?P<value>-?\d+)")
_SUMMARY_RE = re.compile(
    r"SUMMARY count=(?P<count>\d+) exact_mismatches=(?P<exact>\d+) "
    r"tolerance_mismatches=(?P<tolerance>\d+) max_abs=(?P<maximum>\d+)"
)
_GEMMINI_CONFIGURATION_BANNER = "Gemmini extension configured with:"
_GEMMINI_DIMENSION_RE = re.compile(r"\s+dim = \d+")


def _parse_output(text: str, count: int) -> tuple[np.ndarray, dict[str, int]]:
    values: list[int] = []
    summary: dict[str, int] | None = None
    configuration_banner = False
    configuration_dimension = False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("OUTPUT "):
            _require(summary is None, "output follows SUMMARY")
            match = _OUTPUT_RE.fullmatch(line)
            _require(match is not None, f"malformed OUTPUT line {number}")
            _require(
                int(match["index"]) == len(values), "output indices differ"
            )
            value = int(match["value"])
            _require(-128 <= value <= 127, "output code is outside INT8")
            values.append(value)
            continue
        if line.startswith("SUMMARY "):
            _require(summary is None, "duplicate SUMMARY")
            match = _SUMMARY_RE.fullmatch(line)
            _require(match is not None, "malformed SUMMARY")
            summary = {name: int(match[name]) for name in match.groupdict()}
            continue
        if line == _GEMMINI_CONFIGURATION_BANNER:
            _require(summary is not None, "Gemmini banner precedes SUMMARY")
            _require(
                not configuration_banner,
                "duplicate Gemmini configuration banner",
            )
            configuration_banner = True
            continue
        if _GEMMINI_DIMENSION_RE.fullmatch(line):
            _require(
                configuration_banner and not configuration_dimension,
                "unexpected Gemmini configuration dimension",
            )
            configuration_dimension = True
            continue
        if line:
            raise anchor.ConversionError(
                f"unexpected native output line {number}"
            )
    _require(len(values) == count, "native output count differs")
    _require(
        summary is not None and summary["count"] == count, "bad SUMMARY count"
    )
    return np.asarray(values, dtype=np.int8), summary


def validate_conv_conversion(converted: Path, contract: Path) -> dict[str, Any]:
    manifest = anchor._load_json(
        converted / "conversion_manifest.json", "manifest"
    )
    _require(manifest.get("status") == "converted", "conversion did not finish")
    _require(manifest.get("operation_id") == OPERATION_ID, "wrong operation")
    _require(
        manifest.get("contract")
        == {"path": str(contract), "sha256": WORKLOAD_CONTRACT_SHA256},
        "wrong conversion contract",
    )
    projection = manifest.get("projection")
    _require(isinstance(projection, dict), "projection is missing")
    _require(projection.get("helper") == HELPER, "wrong native helper")
    fields = projection.get("fields")
    _require(isinstance(fields, dict), "native fields are missing")
    _require(fields.get("bias") == "voyager_bias", "bias is not projected")
    _require(
        fields.get("activation") in {"NO_ACTIVATION", "RELU"},
        "wrong activation",
    )
    pool_size = fields.get("pool_size")
    pool_stride = fields.get("pool_stride")
    pool_padding = fields.get("pool_padding")
    _require(
        all(
            type(value) is int and value >= 0
            for value in (
                pool_size,
                pool_stride,
                pool_padding,
            )
        ),
        "invalid pooling projection",
    )
    _require(
        (pool_size == pool_stride == pool_padding == 0)
        or (pool_size > 0 and pool_stride > 0),
        "incomplete pooling projection",
    )
    calls = anchor._load_json(converted / "emitted-calls.json", "emitted calls")
    _require(
        calls.get("emitted_operation_helper_count") == 1, "wrong call count"
    )
    _require(calls.get("operation_id") == OPERATION_ID, "wrong call operation")
    for name in ("payloads.h", "runner.c"):
        _require((converted / name).is_file(), f"missing generated {name}")
    for name in ("input", "weight", "bias", "oracle", "output"):
        _require(
            (converted / "payloads" / f"{name}.bin").is_file(),
            "missing payload",
        )
    source = (converted / "runner.c").read_text()
    _require(source.count("tiled_conv_auto(") == 1, "wrong helper source count")
    _require("voyager_bias" in source, "generated source omits bias")
    return manifest


def _command(
    command: list[str],
    *,
    cwd: Path,
    stdout: Path,
    stderr: Path,
    records: list[dict[str, Any]],
    env: dict[str, str] | None = None,
    timeout: int = 180,
):
    return native_run._run_command(
        command,
        cwd=cwd,
        stdout_path=stdout,
        stderr_path=stderr,
        records=records,
        env=env,
        timeout=timeout,
    )


def _generic_elf_validation(
    output: Path,
    payload_sizes: dict[str, int],
) -> dict[str, Any]:
    readelf = (output / "elf.readelf.txt").read_text()
    nm = (output / "elf.nm.txt").read_text()
    unresolved = (output / "elf.unresolved.txt").read_text()
    objdump = (output / "elf.objdump.txt").read_text()
    _require(
        "Class:                             ELF64" in readelf,
        "ELF is not 64 bit",
    )
    _require(
        "Machine:                           RISC-V" in readelf,
        "ELF is not RISC-V",
    )
    _require(not unresolved.strip(), "ELF has unresolved symbols")
    symbols = native_run._parse_symbols(nm)
    required = {
        "voyager_input": payload_sizes["input"],
        "voyager_weights": payload_sizes["weight"],
        "voyager_bias": payload_sizes["bias"],
        "voyager_expected": payload_sizes["oracle"],
        "voyager_output": payload_sizes["output"],
    }
    for name, size in required.items():
        _require(name in symbols, f"ELF lacks {name}")
        _require(symbols[name]["size"] == size, f"ELF size differs: {name}")
    _require(native_run.WRAPPER_SYMBOL in symbols, "ELF lacks wrapper")
    wrapper = symbols[native_run.WRAPPER_SYMBOL]
    _require(
        isinstance(wrapper["size"], int) and wrapper["size"] > 0, "bad wrapper"
    )
    sites = native_run._static_custom_sites(objdump)
    start = int(wrapper["address"])
    end = start + int(wrapper["size"])
    inside = {pc: word for pc, word in sites.items() if start <= pc < end}
    _require(inside, "wrapper has no RoCC instructions")
    _require(
        any(not (start <= pc < end) for pc in sites), "missing lifecycle RoCC"
    )
    return {
        "wrapper": {
            "symbol": native_run.WRAPPER_SYMBOL,
            "start": start,
            "end": end,
        },
        "static_custom_sites": [
            {"pc": pc, "word": word} for pc, word in sites.items()
        ],
        "payload_sizes": required,
    }


def _validate_w_stack(
    manifest_path: Path,
    stack_root: Path,
) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    """Validate the shared portable runtime lock for a W execution."""
    return native_run.validate_stack_manifest(manifest_path, stack_root)


def _execute_converted_region(
    converted_dir: Path | str,
    output_dir: Path | str,
    stack_root: Path | str,
    stack_manifest: Path | str,
    contract: Path | str,
    contract_sha256: str,
    *,
    spike_timeout_seconds: int = 900,
    trace_instructions: bool = False,
    expected_contract_sha256: str | None = None,
    manifest_validator=None,
    operation_id: str = OPERATION_ID,
    output_fact_key: str = "output_codes",
    output_layout: str = "activation",
) -> dict[str, Any]:
    """Build and run one converted W direct region on the pinned Spike stack.

    Raw Spike instruction logging is deliberately opt-in.  It logs every host
    instruction, not just RoCC commands, and therefore turns a small native
    operation into a multi-gigabyte artifact.  Static ELF RoCC-site evidence
    plus the complete ordered native readback remain available by default.
    """
    converted = Path(converted_dir).resolve()
    output = Path(output_dir).resolve()
    stack_root = Path(stack_root).resolve()
    contract = Path(contract).resolve()
    stack_manifest = Path(stack_manifest).resolve()
    _require(not output.exists(), "execution output must be fresh")
    _require(
        isinstance(spike_timeout_seconds, int) and spike_timeout_seconds > 0,
        "Spike timeout must be a positive integer",
    )
    _require(
        output_layout in {"activation", "matrix"},
        "unknown native output layout",
    )
    expected_contract = expected_contract_sha256 or WORKLOAD_CONTRACT_SHA256
    _require(contract_sha256 == expected_contract, "wrong W contract hash")
    _require(
        anchor.sha256_file(contract) == expected_contract,
        "W contract changed",
    )
    validator = manifest_validator or validate_conv_conversion
    manifest = validator(converted, contract)
    stack, tools, frozen = _validate_w_stack(stack_manifest, stack_root)
    output.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": 1,
        "operation_id": operation_id,
        "status": "running",
        "converted_directory": str(converted),
        "stack_root": str(stack),
        "stack_manifest": str(stack_manifest),
        "contract": {"path": str(contract), "sha256": contract_sha256},
        "emitted_operation_helper_count": 1,
        "performance_claimed": False,
        "instruction_trace_enabled": trace_instructions,
        "spike_timeout_seconds": spike_timeout_seconds,
    }
    try:
        anchor._write_json(output / "frozen-stack.json", frozen)
        rocc = stack / native_run.ROCC_ROOT
        common = stack / native_run.COMMON_ROOT
        flags = [
            *native_run.COMPILE_FLAGS,
            f"-I{rocc / 'riscv-tests'}",
            f"-I{rocc / 'riscv-tests/env'}",
            f"-I{rocc}",
            f"-I{common}",
        ]
        gcc = tools["riscv64-unknown-elf-gcc"]
        readelf = tools["riscv64-unknown-elf-readelf"]
        nm = tools["riscv64-unknown-elf-nm"]
        objdump = tools["riscv64-unknown-elf-objdump"]
        runner_object = output / "runner.o"
        elf = output / "voyager-gemmini.elf"
        link_map = output / "voyager-gemmini.map"
        completed = _command(
            [
                str(gcc),
                *flags,
                "-c",
                str(converted / "runner.c"),
                "-o",
                str(runner_object),
            ],
            cwd=output,
            stdout=output / "compile.stdout",
            stderr=output / "compile.stderr",
            records=records,
        )
        _require(completed.returncode == 0, "native C compilation failed")
        completed = _command(
            [
                str(gcc),
                *flags,
                *native_run.LINK_FLAGS,
                "-T",
                str(common / "test.ld"),
                f"-Wl,-Map,{link_map}",
                str(runner_object),
                str(common / "syscalls.c"),
                str(common / "crt.S"),
                "-o",
                str(elf),
            ],
            cwd=output,
            stdout=output / "link.stdout",
            stderr=output / "link.stderr",
            records=records,
        )
        _require(completed.returncode == 0, "native ELF link failed")
        for stem, command in (
            ("elf.readelf", [str(readelf), "-hW", "-sW", str(elf)]),
            ("elf.nm", [str(nm), "-n", "-S", "--defined-only", str(elf)]),
            ("elf.unresolved", [str(nm), "-u", str(elf)]),
            ("elf.objdump", [str(objdump), "-dr", str(elf)]),
        ):
            completed = _command(
                command,
                cwd=output,
                stdout=output / f"{stem}.txt",
                stderr=output / f"{stem}.stderr",
                records=records,
            )
            _require(
                completed.returncode == 0, f"ELF inspection failed: {stem}"
            )
        payload_sizes = {
            name: (converted / "payloads" / f"{name}.bin").stat().st_size
            for name in ("input", "weight", "bias", "oracle", "output")
        }
        elf_evidence = _generic_elf_validation(output, payload_sizes)
        anchor._write_json(output / "elf-validation.json", elf_evidence)
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(
            (
                str(stack / "gemmini-stack/toolchain-env/bin"),
                str(stack / "gemmini-stack/toolchain-env/riscv-tools/bin"),
                "/usr/bin",
                "/bin",
            )
        )
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            (
                str(stack / "gemmini-stack/runtime-prefix/lib"),
                str(stack / "gemmini-stack/toolchain-env/lib"),
            )
        )
        spike_command = [
            "/usr/bin/timeout",
            str(spike_timeout_seconds),
            str(tools["spike"]),
        ]
        trace = output / "spike-execution.log"
        if trace_instructions:
            spike_command.extend(("-l", f"--log={trace}"))
        spike_command.extend(("--extension=gemmini", str(elf)))
        completed = _command(
            spike_command,
            cwd=output,
            stdout=output / "spike.stdout",
            stderr=output / "spike.stderr",
            records=records,
            env=env,
            timeout=spike_timeout_seconds + 15,
        )
        _require(completed.returncode == 0, "Spike/program failed")
        if trace_instructions:
            sites = {
                item["pc"]: item["word"]
                for item in elf_evidence["static_custom_sites"]
            }
            trace_evidence = native_run.parse_spike_trace(
                trace,
                elf_evidence["wrapper"]["start"],
                elf_evidence["wrapper"]["end"],
                sites,
            )
        else:
            trace_evidence = {
                "evidence_kind": "static-elf-rocc-sites-plus-native-readback",
                "instruction_trace_captured": False,
                "static_custom_site_count": len(
                    elf_evidence["static_custom_sites"]
                ),
                "actual_accelerator_call_count": None,
                "runtime_operation_rocc_instruction_record_count": None,
                "runtime_lifecycle_rocc_instruction_record_count": None,
            }
        anchor._write_json(output / "rocc-trace.json", trace_evidence)
        expected = np.frombuffer(
            (converted / "payloads" / "oracle.bin").read_bytes(),
            dtype=np.int8,
        ).copy()
        actual, summary = _parse_output(completed.stdout, int(expected.size))
        difference = actual.astype(np.int16) - expected.astype(np.int16)
        exact = np.flatnonzero(difference)
        tolerance = np.flatnonzero(np.abs(difference) > 1)
        _require(
            summary["exact"] == int(exact.size), "program exact count differs"
        )
        _require(
            summary["tolerance"] == int(tolerance.size),
            "program tolerance count differs",
        )
        _require(
            summary["maximum"] == int(np.max(np.abs(difference))),
            "program maximum differs",
        )
        _require(not tolerance.size, "native comparison exceeds atol 1")
        shape = manifest["facts"][output_fact_key]["shape"]
        _require(
            isinstance(shape, list)
            and all(isinstance(dimension, int) for dimension in shape),
            "logical output shape is invalid",
        )
        if output_layout == "activation":
            _require(len(shape) == 4, "activation output must be rank four")
            physical_shape = (shape[0], shape[2], shape[3], shape[1])
            logical_actual = anchor.unpack_activation(
                actual.reshape(physical_shape)
            )
            logical_expected = anchor.unpack_activation(
                expected.reshape(physical_shape)
            )
        else:
            _require(len(shape) == 2, "matrix output must be rank two")
            logical_actual = actual.reshape(shape)
            logical_expected = expected.reshape(shape)
        comparison = {
            "element_count": int(expected.size),
            "exact_match_count": int(expected.size - exact.size),
            "exact_mismatch_count": int(exact.size),
            "tolerance_mismatch_count": int(tolerance.size),
            "maximum_absolute_error": int(np.max(np.abs(difference))),
            "absolute_tolerance": 1,
            "relative_tolerance": 0,
            "exact_match_required": False,
            "logical_shape": shape,
            "logical_layout": output_layout,
            "logical_values_equal": bool(
                np.array_equal(logical_actual, logical_expected)
            ),
        }
        anchor._write_json(output / "comparison.json", comparison)
        anchor._write_json(
            output / "readback.json",
            {
                "record_count": int(actual.size),
                "values": [int(value) for value in actual],
                "stdout_sha256": anchor.sha256_file(output / "spike.stdout"),
            },
        )
        result.update(
            {
                "status": "passed",
                "conversion_manifest_sha256": anchor.sha256_file(
                    converted / "conversion_manifest.json"
                ),
                "elf_sha256": anchor.sha256_file(elf),
                "actual_accelerator_call_count": trace_evidence[
                    "actual_accelerator_call_count"
                ],
                "runtime_operation_rocc_instruction_count": trace_evidence[
                    "runtime_operation_rocc_instruction_record_count"
                ],
                "runtime_lifecycle_rocc_instruction_count": trace_evidence[
                    "runtime_lifecycle_rocc_instruction_record_count"
                ],
                "complete_output_comparison": comparison,
            }
        )
    except BaseException as error:
        result.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
    finally:
        result["overall_wall_seconds"] = time.monotonic() - started
        result["commands"] = records
        anchor._write_json(output / "result.json", result)
    return result


def _region_family(model_dir: Path, fused_name: str) -> str:
    """Classify one canonical fused region by its ordinary semantic member."""
    proto = anchor._load_proto()
    model = proto.Model()
    path = anchor._safe_artifact_path(model_dir, "model.txt", "model path")
    text_format.Parse(path.read_text(), model)
    matches = [
        record.fused_op
        for record in model.ops
        if record.WhichOneof("op_type") == "fused_op"
        and record.fused_op.name == fused_name
    ]
    _require(len(matches) == 1, "requested fused workload region is missing")
    targets = {member.target for member in matches[0].op_list}
    families = {
        "conv2d": {"conv2d"},
        "residual_add": {"add", "add_"},
        "linear": {"linear"},
        "global_average": {"adaptive_avg_pool2d"},
        "normalization": {"layer_norm", "softmax"},
    }
    matched = [
        family
        for family, operations in families.items()
        if targets & operations
    ]
    _require(bool(matched), "fused workload region has no Gemmini projection")
    _require(len(matched) == 1, "fused workload region is ambiguous")
    return matched[0]


def convert_region(
    model_dir: Path | str,
    output_dir: Path | str,
    contract: Path | str,
    contract_sha256: str,
    fused_name: str,
    runtime_capture_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Convert one named canonical region through the Gemmini W adapter."""
    model_dir = Path(model_dir).resolve()
    family = _region_family(model_dir, fused_name)
    if family == "conv2d":
        _require(
            runtime_capture_dir is not None,
            "Conv2d conversion requires a transformed runtime capture",
        )
        return _convert_conv_region(
            model_dir,
            output_dir,
            contract,
            contract_sha256,
            fused_name,
            runtime_capture_dir,
        )

    _require(
        runtime_capture_dir is None,
        "runtime capture is currently consumed only by Conv2d regions",
    )
    # Deferred import avoids a module cycle: helper projections reuse this
    # module's common native execution implementation.
    from conversion.gemmini import _workload_helpers

    return _workload_helpers.convert_helper_region(
        model_dir,
        Path(output_dir),
        Path(contract),
        contract_sha256,
        fused_name,
    )


def execute_region(
    converted_dir: Path | str,
    output_dir: Path | str,
    stack_root: Path | str,
    stack_manifest: Path | str,
    contract: Path | str,
    contract_sha256: str,
    *,
    spike_timeout_seconds: int = 900,
    trace_instructions: bool = False,
) -> dict[str, Any]:
    """Execute one converted region through the common native runner."""
    converted = Path(converted_dir).resolve()
    contract_path = Path(contract).resolve()
    manifest = anchor._load_json(
        converted / "conversion_manifest.json", "workload manifest"
    )
    operation_id = manifest.get("operation_id")
    if operation_id == OPERATION_ID:
        validator = validate_conv_conversion
        output_fact_key = "output_codes"
        output_layout = "activation"
    else:
        # See convert_region: helper projections depend on the shared runner.
        from conversion.gemmini import _workload_helpers

        _require(
            operation_id
            in {
                _workload_helpers.ADD_OPERATION_ID,
                _workload_helpers.AVERAGE_OPERATION_ID,
                _workload_helpers.LINEAR_OPERATION_ID,
            },
            "converted workload operation is unsupported",
        )
        validator = _workload_helpers.validate_helper_conversion
        output_fact_key = "output"
        output_layout = (
            "activation"
            if operation_id == _workload_helpers.ADD_OPERATION_ID
            else "matrix"
        )
    return _execute_converted_region(
        converted,
        output_dir,
        stack_root,
        stack_manifest,
        contract_path,
        contract_sha256,
        spike_timeout_seconds=spike_timeout_seconds,
        trace_instructions=trace_instructions,
        expected_contract_sha256=WORKLOAD_CONTRACT_SHA256,
        manifest_validator=validator,
        operation_id=operation_id,
        output_fact_key=output_fact_key,
        output_layout=output_layout,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    convert_parser = actions.add_parser("convert")
    convert_parser.add_argument("--model-dir", required=True, type=Path)
    convert_parser.add_argument("--output-dir", required=True, type=Path)
    convert_parser.add_argument("--contract", required=True, type=Path)
    convert_parser.add_argument("--contract-sha256", required=True)
    convert_parser.add_argument("--fused-region", required=True)
    convert_parser.add_argument("--runtime-capture", type=Path)
    run_parser = actions.add_parser("run")
    run_parser.add_argument("--converted-dir", required=True, type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser.add_argument("--stack-root", required=True, type=Path)
    run_parser.add_argument("--stack-manifest", required=True, type=Path)
    run_parser.add_argument("--contract", required=True, type=Path)
    run_parser.add_argument("--contract-sha256", required=True)
    run_parser.add_argument("--spike-timeout-seconds", type=int, default=900)
    run_parser.add_argument("--trace-instructions", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "convert":
            result = convert_region(
                args.model_dir,
                args.output_dir,
                args.contract,
                args.contract_sha256,
                args.fused_region,
                args.runtime_capture,
            )
        else:
            result = execute_region(
                args.converted_dir,
                args.output_dir,
                args.stack_root,
                args.stack_manifest,
                args.contract,
                args.contract_sha256,
                spike_timeout_seconds=args.spike_timeout_seconds,
                trace_instructions=args.trace_instructions,
            )
    except anchor.ConversionError as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") in {"converted", "passed"} else 1


__all__ = [
    "WORKLOAD_CONTRACT_SHA256",
    "convert_region",
    "execute_region",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())

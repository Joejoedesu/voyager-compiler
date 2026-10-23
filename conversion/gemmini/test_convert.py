# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Self-contained tests for the Gemmini converter and traced runner."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from google.protobuf import text_format


def _load(name: str):
    path = Path(__file__).with_name(f"{name}.py")
    specification = importlib.util.spec_from_file_location(
        f"gemmini_{name}_test", path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


converter = _load("convert")
runner = _load("run")
SHARED_FUSED_CONTRACT_SHA256 = (
    "a36b7a95d5d47bc8531abff4b72fa463731957f92ae5f34705c8cd7b5876d612"
)


def _tensor(
    message,
    node: str,
    shape: tuple[int, ...],
    dtype: str,
    address: int,
) -> None:
    message.node = node
    message.shape.extend(shape)
    message.dtype = dtype
    message.memory.partition = 0
    message.memory.address = address
    message.memory.instance = "DRAM"


def _tensor_argument(
    operation,
    name: str,
    node: str,
    shape: tuple[int, ...],
    dtype: str,
    address: int,
):
    _tensor(operation.kwargs[name].tensor, node, shape, dtype, address)


def _pair_argument(operation, name: str, values: tuple[int, int]) -> None:
    operation.kwargs[name].int_list.values.extend(values)


def _scale_argument(operation, name: str, node: str) -> None:
    _tensor(operation.kwargs[name].tensor, node, (1,), "float32", 0)
    operation.kwargs[name].tensor.ClearField("memory")


def _quantize_operation(
    operation,
    *,
    name: str,
    input_node: str,
    input_dtype: str,
    scale_node: str,
    address: int | None,
) -> None:
    operation.name = name
    operation.op = "call_function"
    operation.target = "quantize"
    _tensor_argument(
        operation,
        "input",
        input_node,
        (1, 16, 4, 4),
        input_dtype,
        address or 0,
    )
    if address is None:
        operation.kwargs["input"].tensor.ClearField("memory")
    _scale_argument(operation, "scale", scale_node)
    operation.kwargs["rounding"].str_value = "nearest"
    operation.kwargs["quant_min"].float_value = -128.0
    operation.kwargs["quant_max"].float_value = 127.0


def _canonical(tmp_path: Path) -> Path:
    """Create the exact shared fused serializer shape as a unit fixture."""
    model_dir = tmp_path / "canonical"
    tensor_dir = model_dir / "tensor_files"
    tensor_dir.mkdir(parents=True)
    model = converter._load_proto().Model()
    _tensor(model.inputs.add(), "input_1", (1, 16, 4, 4), "float32", 0x1000)
    _tensor(
        model.parameters.add(),
        "weight",
        (16, 16, 3, 3),
        "int8",
        0x2000,
    )
    fused_record = model.ops.add()
    fused = fused_record.fused_op
    fused.name = "fused_conv"
    input_quantize = fused.op_list.add()
    _quantize_operation(
        input_quantize,
        name="input_quantize",
        input_node="input_1",
        input_dtype="float32",
        scale_node="input_scale",
        address=0x1000,
    )
    convolution = fused.op_list.add()
    convolution.name = "conv"
    convolution.op = "call_function"
    convolution.target = "conv2d"
    _tensor_argument(
        convolution,
        "input",
        "input_quantize",
        (1, 16, 4, 4),
        "int8",
        0,
    )
    convolution.kwargs["input"].tensor.ClearField("memory")
    _tensor_argument(
        convolution,
        "weight",
        "weight",
        (16, 16, 3, 3),
        "int8",
        0x2000,
    )
    _pair_argument(convolution, "stride", (1, 1))
    _pair_argument(convolution, "padding", (1, 1))
    _pair_argument(convolution, "dilation", (1, 1))
    convolution.kwargs["groups"].int_value = 1
    accumulator_dq = fused.op_list.add()
    accumulator_dq.name = "accumulator_dq"
    accumulator_dq.op = "call_function"
    accumulator_dq.target = "dequantize"
    _tensor_argument(
        accumulator_dq,
        "input",
        "conv",
        (1, 16, 4, 4),
        "int8",
        0,
    )
    accumulator_dq.kwargs["input"].tensor.ClearField("memory")
    _scale_argument(accumulator_dq, "scale", "accumulator_scale")
    output_quantize = fused.op_list.add()
    _quantize_operation(
        output_quantize,
        name="output_quantize",
        input_node="accumulator_dq",
        input_dtype="float32",
        scale_node="output_scale",
        address=None,
    )
    _tensor(
        fused_record.output,
        "fused_conv",
        (1, 16, 4, 4),
        "int8",
        0x3000,
    )
    terminal_record = model.ops.add()
    terminal = terminal_record.op
    terminal.name = "terminal_dq"
    terminal.op = "call_function"
    terminal.target = "dequantize"
    _tensor_argument(
        terminal,
        "input",
        "fused_conv",
        (1, 16, 4, 4),
        "int8",
        0x3000,
    )
    _scale_argument(terminal, "scale", "terminal_scale")
    _tensor(
        terminal_record.output,
        "terminal_dq",
        (1, 16, 4, 4),
        "float32",
        0x4000,
    )
    (model_dir / "model.txt").write_text(text_format.MessageToString(model))

    input_values = (np.arange(256, dtype=np.int16) % 17 - 8) * 0.5
    weight_values = np.arange(2304, dtype=np.int16) % 7 - 3
    output_values = np.arange(256, dtype=np.int16) % 17 - 8
    input_scale = np.float32(0.5)
    weight_scale = np.float32(0.25)
    accumulator_scale = np.float32(input_scale * weight_scale)
    output_scale = np.float32(8.0 / 127.0)
    terminal_values = np.multiply(
        output_values.astype(np.float32), output_scale, dtype=np.float32
    )
    for name, values in (
        ("input_1", input_values),
        ("weight", weight_values),
        ("fused_conv", output_values),
        ("terminal_dq", terminal_values),
        ("input_scale", np.asarray([input_scale])),
        ("accumulator_scale", np.asarray([accumulator_scale])),
        ("output_scale", np.asarray([output_scale])),
        ("terminal_scale", np.asarray([output_scale])),
    ):
        values.astype("<f4").tofile(tensor_dir / f"{name}.bin")
    reference_path = model_dir / converter.ORACLE_ARTIFACT
    output_values.astype("<f4").tofile(reference_path)
    terminal_reference_path = model_dir / converter.ORACLE_TERMINAL_ARTIFACT
    terminal_values.astype("<f4").tofile(terminal_reference_path)
    canonical_output = tensor_dir / "fused_conv.bin"
    canonical_terminal = tensor_dir / "terminal_dq.bin"
    oracle_manifest = {
        "schema_version": 1,
        "producer": "natural-pytorch-pt2e-qdq",
        "operation_id": converter.OPERATION_ID,
        "scheme_id": converter.SCHEME_ID,
        "qdq_form": converter.QDQ_FORM,
        "code_reference": {
            "path": "pt2e_reference.bin",
            "sha256": converter.sha256_file(reference_path),
            "container_dtype": "float32",
            "logical_dtype": "int8",
            "shape": [1, 16, 4, 4],
        },
        "terminal_reference": {
            "path": converter.ORACLE_TERMINAL_ARTIFACT,
            "sha256": converter.sha256_file(terminal_reference_path),
            "dtype": "float32",
            "shape": [1, 16, 4, 4],
        },
        "canonical_code_output": {
            "path": "tensor_files/fused_conv.bin",
            "sha256": converter.sha256_file(canonical_output),
            "equal_to_reference": True,
        },
        "canonical_terminal_output": {
            "path": "tensor_files/terminal_dq.bin",
            "sha256": converter.sha256_file(canonical_terminal),
            "equal_to_reference": True,
        },
        "qparams": {
            "input_scale": float(input_scale),
            "input_zero_point": 0,
            "weight_scale": float(weight_scale),
            "weight_zero_point": 0,
            "output_scale": float(output_scale),
            "output_zero_point": 0,
        },
    }
    (model_dir / converter.ORACLE_MANIFEST).write_text(
        json.dumps(oracle_manifest, indent=2, sort_keys=True) + "\n"
    )
    return model_dir


def _contract(tmp_path: Path) -> tuple[Path, str]:
    del tmp_path
    path = converter._repo_root() / "docs/targets/gemmini/injection-contract.md"
    assert converter.INJECTION_CONTRACT_SHA256 == SHARED_FUSED_CONTRACT_SHA256
    assert runner.INJECTION_CONTRACT_SHA256 == SHARED_FUSED_CONTRACT_SHA256
    assert converter.sha256_file(path) == converter.INJECTION_CONTRACT_SHA256
    return path, converter.INJECTION_CONTRACT_SHA256


def _read_oracle_manifest(model_dir: Path) -> dict:
    return json.loads((model_dir / converter.ORACLE_MANIFEST).read_text())


def _write_oracle_manifest(model_dir: Path, manifest: dict) -> None:
    (model_dir / converter.ORACLE_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _read_model(model_dir: Path):
    model = converter._load_proto().Model()
    text_format.Parse((model_dir / "model.txt").read_text(), model)
    return model


def _write_model(model_dir: Path, model) -> None:
    (model_dir / "model.txt").write_text(text_format.MessageToString(model))


def test_conversion_binds_canonical_facts_and_independent_oracle(
    tmp_path: Path,
) -> None:
    model_dir = _canonical(tmp_path)
    contract, digest = _contract(tmp_path)
    output = tmp_path / "converted"
    manifest = converter.convert(model_dir, output, contract, digest)

    assert manifest["status"] == "converted"
    assert manifest["operation_id"] == converter.OPERATION_ID
    assert manifest["emitted_operation_helper_count"] == 1
    assert manifest["layout_mapping_rows"] == 3072
    assert set(manifest["payloads"]) == {
        "input",
        "weight",
        "output",
        "output_oracle",
    }
    assert manifest["payloads"]["output"]["oracle_only"] is False
    assert manifest["payloads"]["output"]["endpoint_consumed"] is True
    assert manifest["payloads"]["output_oracle"]["oracle_only"] is True
    assert manifest["payloads"]["output_oracle"]["endpoint_consumed"] is False
    assert (output / "payloads/output.bin").read_bytes() == bytes(256)
    assert (output / "payloads/output_oracle.bin").read_bytes() != bytes(256)
    assert manifest["performance_claimed"] is False
    assert manifest["operation_facts"]["input_scale"] == 0.5
    assert manifest["operation_facts"]["output_zero_point"] == 0
    assert manifest["operation_facts"]["model_input"]["memory_partition"] == 0
    assert (
        manifest["operation_facts"]["model_input"]["memory_address"] == 0x1000
    )
    assert (
        manifest["operation_facts"]["model_input"]["memory_instance"] == "DRAM"
    )
    assert manifest["input_quantize_projection"]["rounding"] == "nearest-even"
    assert manifest["input_quantize_projection"]["projection_time_only"] is True
    assert (
        manifest["input_quantize_projection"]["verified_static_input_only"]
        is True
    )
    assert (
        manifest["input_quantize_projection"]["runtime_float_input_supported"]
        is False
    )
    assert (
        manifest["quantization_facts"]["general_bit_exact_equivalence"] is False
    )
    sidecar = _read_oracle_manifest(model_dir)
    binding = manifest["oracle_binding"]
    assert set(binding) == {
        "manifest",
        "code_reference",
        "terminal_reference",
        "canonical_code_output",
        "canonical_terminal_output",
        "qparams",
    }
    assert binding["manifest"] == {
        "path": "oracle-manifest.json",
        "sha256": converter.sha256_file(model_dir / "oracle-manifest.json"),
    }
    assert binding["code_reference"] == {
        "path": "pt2e_reference.bin",
        "sha256": sidecar["code_reference"]["sha256"],
        "producer": "natural-pytorch-pt2e-qdq",
        "scheme_id": converter.SCHEME_ID,
        "qdq_form": converter.QDQ_FORM,
        "container_dtype": "float32",
        "logical_dtype": "int8",
        "shape": [1, 16, 4, 4],
    }
    assert binding["terminal_reference"] == {
        "path": "pt2e_terminal_reference.bin",
        "sha256": sidecar["terminal_reference"]["sha256"],
        "producer": "natural-pytorch-pt2e-qdq",
        "scheme_id": converter.SCHEME_ID,
        "qdq_form": converter.QDQ_FORM,
        "dtype": "float32",
        "shape": [1, 16, 4, 4],
    }
    assert binding["canonical_code_output"] == {
        "path": "tensor_files/fused_conv.bin",
        "sha256": sidecar["canonical_code_output"]["sha256"],
        "verified_equal": True,
    }
    assert binding["canonical_terminal_output"] == {
        "path": "tensor_files/terminal_dq.bin",
        "sha256": sidecar["canonical_terminal_output"]["sha256"],
        "verified_equal": True,
    }
    assert binding["qparams"] == sidecar["qparams"]
    projection = manifest["native_projection"]
    assert projection["helper"] == "tiled_conv_auto"
    assert projection["fields"]["input_dilation"] == {
        "value": 1,
        "provenance": "native:injection-contract:input_dilation",
    }
    assert projection["fields"]["kernel_dilation"]["provenance"] == (
        "canonical:model.txt:dilation"
    )
    assert projection["fields"]["acc_scale"]["value"] != "ACC_SCALE_IDENTITY"
    assert np.float32(
        float.fromhex(projection["fields"]["acc_scale"]["value"][:-1])
    ) == np.float32(0.125 / np.float32(8.0 / 127.0))
    assert "in_stride" not in projection["fields"]
    assert projection["helper_derived_fields"] == {
        "in_stride": 16,
        "weight_stride": 16,
        "out_stride": 16,
    }
    assert all(
        payload["round_trip_equal"] for payload in manifest["payloads"].values()
    )
    generated_paths = [
        artifact["path"] for artifact in manifest["generated_artifacts"]
    ]
    assert generated_paths == sorted(converter.GENERATED_ARTIFACT_PATHS)
    source = (output / "runner.c").read_text()
    assert source.count("tiled_conv_auto(") == 1
    assert "tiled_conv_stride_auto(" not in source
    assert "void voyager_gemmini_operation(void)" in source
    assert source.index("gemmini_flush(0)") < source.index(
        "voyager_gemmini_operation();"
    )
    assert source.index("voyager_gemmini_operation();") < source.index(
        "gemmini_fence()"
    )
    assert 'printf("OUTPUT %d %d\\n", index, actual);' in source
    assert "tolerance_mismatches=%d first=" in source
    assert 'squared_error=%llu\\n"' in source
    assert 'printf("OUTPUT %d %d\n"' not in source


def test_converter_rejects_unknown_model_kwargs(tmp_path: Path) -> None:
    model_dir = _canonical(tmp_path)
    model = _read_model(model_dir)
    model.ops[0].fused_op.op_list[1].kwargs[
        "gemmini_helper"
    ].str_value = "tiled_conv_auto"
    _write_model(model_dir, model)
    contract, digest = _contract(tmp_path)
    with pytest.raises(converter.ConversionError, match="extra"):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


@pytest.mark.parametrize(
    "member,name",
    [
        *((0, name) for name in sorted(converter.QUANTIZE_KWARGS)),
        *((1, name) for name in sorted(converter.CONV2D_KWARGS)),
        *((2, name) for name in sorted(converter.DEQUANTIZE_KWARGS)),
        *((3, name) for name in sorted(converter.QUANTIZE_KWARGS)),
    ],
)
def test_converter_rejects_every_missing_kwarg(
    tmp_path: Path, member: int, name: str
) -> None:
    model_dir = _canonical(tmp_path)
    model = _read_model(model_dir)
    del model.ops[0].fused_op.op_list[member].kwargs[name]
    _write_model(model_dir, model)
    contract, digest = _contract(tmp_path)
    with pytest.raises(converter.ConversionError, match="missing"):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


def test_converter_rejects_positional_arguments(tmp_path: Path) -> None:
    model_dir = _canonical(tmp_path)
    model = _read_model(model_dir)
    model.ops[0].fused_op.op_list[1].args.add().int_value = 1
    _write_model(model_dir, model)
    contract, digest = _contract(tmp_path)
    with pytest.raises(
        converter.ConversionError, match="must have no arguments"
    ):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


@pytest.mark.parametrize(
    "defect,message",
    [
        ("custom", "shared conv2d has wrong target"),
        ("mx", "shared conv2d has wrong target"),
        ("order", "input quantize has wrong target"),
        ("terminal", "model must have fused and terminal ops"),
    ],
)
def test_converter_rejects_noncanonical_fused_structure(
    tmp_path: Path, defect: str, message: str
) -> None:
    model_dir = _canonical(tmp_path)
    model = _read_model(model_dir)
    if defect == "custom":
        model.ops[0].fused_op.op_list[1].target = "conv2d_int8"
    elif defect == "mx":
        model.ops[0].fused_op.op_list[1].target = "conv2d_mx"
    elif defect == "order":
        first = type(model.ops[0].fused_op.op_list[0])()
        first.CopyFrom(model.ops[0].fused_op.op_list[0])
        model.ops[0].fused_op.op_list[0].CopyFrom(
            model.ops[0].fused_op.op_list[1]
        )
        model.ops[0].fused_op.op_list[1].CopyFrom(first)
    else:
        del model.ops[1]
    _write_model(model_dir, model)
    contract, digest = _contract(tmp_path)
    with pytest.raises(converter.ConversionError, match=message):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


def test_converter_rejects_self_attested_contract_hash(tmp_path: Path) -> None:
    model_dir = _canonical(tmp_path)
    contract = tmp_path / "substitute-contract.md"
    contract.write_text("not the pinned injection contract\n")
    digest = converter.sha256_file(contract)
    with pytest.raises(converter.ConversionError, match="contract"):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


def test_runner_rejects_pinned_native_hash_drift(tmp_path: Path) -> None:
    model_dir = _canonical(tmp_path)
    contract, digest = _contract(tmp_path)
    converted = tmp_path / "converted"
    converter.convert(model_dir, converted, contract, digest)
    manifest_path = converted / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["native_dependencies"]["gemmini.h"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(
        runner.RunError, match="native dependencies are not pinned"
    ):
        runner._validate_conversion(converted, contract, digest)


def test_converter_rejects_nonfinite_input_container(tmp_path: Path) -> None:
    model_dir = _canonical(tmp_path)
    path = model_dir / "tensor_files/input_1.bin"
    values = np.fromfile(path, dtype="<f4")
    values[0] = np.nan
    values.tofile(path)
    contract, digest = _contract(tmp_path)
    with pytest.raises(converter.ConversionError, match="non-finite"):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


@pytest.mark.parametrize(
    "control, message",
    [
        ("tiled_shape", "tiled shape"),
        ("tile_strides", "tile strides"),
        ("scratchpad", "scratchpad control"),
        ("dequant", "fused dequant"),
        ("reshape", "fused reshape"),
    ],
)
def test_converter_rejects_tensor_local_control(
    tmp_path: Path, control: str, message: str
) -> None:
    model_dir = _canonical(tmp_path)
    model = _read_model(model_dir)
    tensor = model.inputs[0]
    if control in {"tiled_shape", "tile_strides"}:
        getattr(tensor, control).append(1)
    elif control == "scratchpad":
        tensor.scratchpad.partition = 1
    else:
        getattr(tensor, control).name = "forbidden"
    _write_model(model_dir, model)
    contract, digest = _contract(tmp_path)
    with pytest.raises(converter.ConversionError, match=message):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


@pytest.mark.parametrize(
    "defect, message",
    [
        ("missing", "no DRAM memory fact"),
        ("partition", "DRAM partition zero"),
        ("address", "DRAM address is missing"),
        ("instance", "memory instance is not DRAM"),
    ],
)
def test_converter_requires_complete_dram_memory_facts(
    tmp_path: Path, defect: str, message: str
) -> None:
    model_dir = _canonical(tmp_path)
    model = _read_model(model_dir)
    tensor = model.inputs[0]
    if defect == "missing":
        tensor.ClearField("memory")
    elif defect == "partition":
        tensor.memory.partition = 1
    elif defect == "address":
        tensor.memory.ClearField("address")
    else:
        tensor.memory.instance = "SRAM"
    _write_model(model_dir, model)
    contract, digest = _contract(tmp_path)
    with pytest.raises(converter.ConversionError, match=message):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


def test_converter_rejects_tensor_path_escape(tmp_path: Path) -> None:
    model_dir = _canonical(tmp_path)
    model = _read_model(model_dir)
    model.inputs[0].node = "../input_1"
    model.ops[0].fused_op.op_list[0].kwargs["input"].tensor.node = "../input_1"
    _write_model(model_dir, model)
    contract, digest = _contract(tmp_path)
    with pytest.raises(converter.ConversionError, match="safe file identity"):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


def test_converter_rejects_aliased_tensor_identities(tmp_path: Path) -> None:
    model_dir = _canonical(tmp_path)
    model = _read_model(model_dir)
    model.ops[0].fused_op.name = "input_quantize"
    model.ops[0].output.node = "input_quantize"
    model.ops[1].op.kwargs["input"].tensor.node = "input_quantize"
    _write_model(model_dir, model)
    contract, digest = _contract(tmp_path)
    with pytest.raises(converter.ConversionError, match="identities.*distinct"):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


def test_converter_rejects_output_identity_mismatch(tmp_path: Path) -> None:
    model_dir = _canonical(tmp_path)
    model = _read_model(model_dir)
    model.ops[0].output.node = "different_output"
    _write_model(model_dir, model)
    contract, digest = _contract(tmp_path)
    with pytest.raises(
        converter.ConversionError, match="fused output identity"
    ):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


@pytest.mark.parametrize(
    "defect,message",
    [
        ("scale-kind", "must be a tensor"),
        ("negative-scale", "must be positive"),
        ("terminal-scale", "scales differ"),
        ("rounding", "nearest rounding"),
        ("bounds", "signed INT8 bounds"),
    ],
)
def test_converter_rejects_invalid_qparams(
    tmp_path: Path,
    defect: str,
    message: str,
) -> None:
    model_dir = _canonical(tmp_path)
    model = _read_model(model_dir)
    if defect == "scale-kind":
        model.ops[0].fused_op.op_list[0].kwargs["scale"].int_value = 1
        _write_model(model_dir, model)
    elif defect == "negative-scale":
        np.asarray([-0.5], dtype="<f4").tofile(
            model_dir / "tensor_files/input_scale.bin"
        )
    elif defect == "terminal-scale":
        np.asarray([0.25], dtype="<f4").tofile(
            model_dir / "tensor_files/terminal_scale.bin"
        )
    elif defect == "rounding":
        model.ops[0].fused_op.op_list[0].kwargs["rounding"].str_value = "floor"
        _write_model(model_dir, model)
    else:
        model.ops[0].fused_op.op_list[0].kwargs[
            "quant_min"
        ].float_value = -127.0
        _write_model(model_dir, model)
    contract, digest = _contract(tmp_path)
    with pytest.raises(converter.ConversionError, match=message):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


def test_float32_ingress_uses_ties_even_and_saturates(tmp_path: Path) -> None:
    values = np.asarray(
        [-200.0, -128.5, -127.5, -126.5, -0.5, 0.5, 1.5, 127.5, 200.0],
        dtype="<f4",
    )
    path = tmp_path / "input.bin"
    values.tofile(path)
    actual = converter._materialize_input(path, values.shape, 1.0, 0)
    expected = np.asarray(
        [-128, -128, -128, -126, 0, 0, 2, 127, 127], dtype=np.int8
    )
    assert np.array_equal(actual, expected)


def test_input_projection_records_known_bfloat16_bit_limit(
    tmp_path: Path,
) -> None:
    values = np.asarray([0.5001, 1.5001], dtype="<f4")
    path = tmp_path / "input.bin"
    values.tofile(path)
    actual = converter._materialize_input(path, values.shape, 1.0, 0)
    assert np.array_equal(actual, np.asarray([1, 2], dtype=np.int8))


@pytest.mark.parametrize("node", ["weight", "fused_conv"])
def test_weight_and_output_containers_remain_strict(
    tmp_path: Path, node: str
) -> None:
    model_dir = _canonical(tmp_path)
    path = model_dir / "tensor_files" / f"{node}.bin"
    values = np.fromfile(path, dtype="<f4")
    values[0] = 1.5
    values.tofile(path)
    if node == "fused_conv":
        manifest = _read_oracle_manifest(model_dir)
        manifest["canonical_code_output"]["sha256"] = converter.sha256_file(
            path
        )
        _write_oracle_manifest(model_dir, manifest)
    contract, digest = _contract(tmp_path)
    with pytest.raises(converter.ConversionError, match="fractional"):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


@pytest.mark.parametrize(
    "values, message",
    [
        (np.asarray([np.nan], dtype="<f4"), "non-finite"),
        (np.asarray([np.inf], dtype="<f4"), "non-finite"),
        (np.asarray([1.5], dtype="<f4"), "fractional"),
        (np.asarray([128.0], dtype="<f4"), "out-of-range"),
    ],
)
def test_integer_container_rejects_invalid_values(
    tmp_path: Path, values: np.ndarray, message: str
) -> None:
    path = tmp_path / "bad.bin"
    path.write_bytes(values.tobytes())
    with pytest.raises(converter.ConversionError, match=message):
        converter._read_integer_container(path, (1,), np.dtype(np.int8))


@pytest.mark.parametrize(
    "relative, message",
    [
        ("oracle-manifest.json", "oracle manifest is missing"),
        ("pt2e_reference.bin", "code reference is missing"),
        ("pt2e_terminal_reference.bin", "terminal reference is missing"),
    ],
)
def test_converter_rejects_missing_oracle_artifacts(
    tmp_path: Path, relative: str, message: str
) -> None:
    model_dir = _canonical(tmp_path)
    (model_dir / relative).unlink()
    contract, digest = _contract(tmp_path)
    with pytest.raises(converter.ConversionError, match=message):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


@pytest.mark.parametrize(
    "binding, message",
    [
        ("code_reference", "code reference hash mismatch"),
        ("terminal_reference", "terminal reference hash mismatch"),
        ("canonical_code_output", "canonical code output hash mismatch"),
        (
            "canonical_terminal_output",
            "canonical terminal output hash mismatch",
        ),
    ],
)
def test_converter_rejects_oracle_binding_hash_mismatch(
    tmp_path: Path, binding: str, message: str
) -> None:
    model_dir = _canonical(tmp_path)
    manifest = _read_oracle_manifest(model_dir)
    manifest[binding]["sha256"] = "0" * 64
    _write_oracle_manifest(model_dir, manifest)
    contract, digest = _contract(tmp_path)
    with pytest.raises(converter.ConversionError, match=message):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("producer", "self-generated", "wrong oracle producer"),
        ("scheme_id", "other-scheme", "wrong oracle scheme"),
        ("qdq_form", "other-form", "wrong oracle Q/DQ form"),
    ],
)
def test_converter_rejects_oracle_provenance_drift(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    model_dir = _canonical(tmp_path)
    manifest = _read_oracle_manifest(model_dir)
    manifest[field] = value
    _write_oracle_manifest(model_dir, manifest)
    contract, digest = _contract(tmp_path)
    with pytest.raises(converter.ConversionError, match=message):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


def test_converter_rejects_nonexact_oracle_manifest(tmp_path: Path) -> None:
    model_dir = _canonical(tmp_path)
    manifest = _read_oracle_manifest(model_dir)
    manifest["untrusted_note"] = "extra"
    _write_oracle_manifest(model_dir, manifest)
    contract, digest = _contract(tmp_path)
    with pytest.raises(converter.ConversionError, match="manifest fields"):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


def test_converter_rejects_oracle_qparam_drift(tmp_path: Path) -> None:
    model_dir = _canonical(tmp_path)
    manifest = _read_oracle_manifest(model_dir)
    manifest["qparams"]["output_zero_point"] = 1
    _write_oracle_manifest(model_dir, manifest)
    contract, digest = _contract(tmp_path)
    with pytest.raises(converter.ConversionError, match="qparam mismatch"):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


def test_converter_byte_compares_independent_oracle(tmp_path: Path) -> None:
    model_dir = _canonical(tmp_path)
    path = model_dir / converter.ORACLE_ARTIFACT
    values = np.fromfile(path, dtype="<f4")
    values[0] += 1.0
    values.tofile(path)
    manifest = _read_oracle_manifest(model_dir)
    manifest["code_reference"]["sha256"] = converter.sha256_file(path)
    _write_oracle_manifest(model_dir, manifest)
    contract, digest = _contract(tmp_path)
    with pytest.raises(
        converter.ConversionError,
        match="independent code reference differs from canonical fused output",
    ):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


def test_converter_validates_independent_terminal_relation(
    tmp_path: Path,
) -> None:
    model_dir = _canonical(tmp_path)
    path = model_dir / converter.ORACLE_TERMINAL_ARTIFACT
    values = np.fromfile(path, dtype="<f4")
    values[0] += np.float32(0.25)
    values.tofile(path)
    manifest = _read_oracle_manifest(model_dir)
    manifest["terminal_reference"]["sha256"] = converter.sha256_file(path)
    _write_oracle_manifest(model_dir, manifest)
    contract, digest = _contract(tmp_path)
    with pytest.raises(
        converter.ConversionError,
        match="independent terminal reference differs",
    ):
        converter.convert(model_dir, tmp_path / "converted", contract, digest)


def test_layout_round_trips_and_maps_every_element() -> None:
    activation = (
        np.arange(256, dtype=np.int16).astype(np.int8).reshape(1, 16, 4, 4)
    )
    weight = (
        np.arange(2304, dtype=np.int16).astype(np.int8).reshape(16, 16, 3, 3)
    )
    packed_activation = converter.pack_activation(activation)
    packed_weight = converter.pack_weight(weight)
    assert np.array_equal(
        converter.unpack_activation(packed_activation), activation
    )
    assert np.array_equal(
        converter.unpack_weight(packed_weight, weight.shape), weight
    )
    rows = converter._mapping_rows("input", activation)
    rows.extend(converter._mapping_rows("weight", weight))
    rows.extend(converter._mapping_rows("output", np.zeros_like(activation)))
    rows.extend(converter._mapping_rows("output_oracle", activation))
    assert len(rows) == 3072
    for role, size in (
        ("input", 256),
        ("weight", 2304),
        ("output", 256),
        ("output_oracle", 256),
    ):
        selected = [row for row in rows if row["role"] == role]
        assert {row["logical_linear_index"] for row in selected} == set(
            range(size)
        )
        assert {row["physical_linear_index"] for row in selected} == set(
            range(size)
        )


def _refresh_generated_artifact(converted: Path, relative: str) -> None:
    manifest_path = converted / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = next(
        item
        for item in manifest["generated_artifacts"]
        if item["path"] == relative
    )
    path = converted / relative
    record["bytes"] = path.stat().st_size
    record["sha256"] = runner.sha256_file(path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def test_runner_rejects_semantically_corrupt_layout_map(tmp_path: Path) -> None:
    model_dir = _canonical(tmp_path)
    contract, digest = _contract(tmp_path)
    converted = tmp_path / "converted"
    converter.convert(model_dir, converted, contract, digest)
    mapping_path = converted / "layout-map.csv"
    with mapping_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[1]["physical_linear_index"] = rows[0]["physical_linear_index"]
    with mapping_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=runner.LAYOUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    _refresh_generated_artifact(converted, "layout-map.csv")
    with pytest.raises(runner.RunError, match="layout"):
        runner._validate_conversion(converted, contract.resolve(), digest)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_runner_requires_exact_generated_artifact_inventory(
    tmp_path: Path, mutation: str
) -> None:
    model_dir = _canonical(tmp_path)
    contract, digest = _contract(tmp_path)
    converted = tmp_path / "converted"
    converter.convert(model_dir, converted, contract, digest)
    manifest_path = converted / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if mutation == "missing":
        manifest["generated_artifacts"].pop()
    else:
        path = converted / "extra.txt"
        path.write_text("undeclared\n")
        manifest["generated_artifacts"].append(
            {
                "path": "extra.txt",
                "bytes": path.stat().st_size,
                "sha256": runner.sha256_file(path),
            }
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(runner.RunError, match="artifact inventory"):
        runner._validate_conversion(converted, contract.resolve(), digest)


def _trace(tmp_path: Path, lines: list[bytes]) -> Path:
    path = tmp_path / "trace.log"
    path.write_bytes(b"\n".join(lines) + b"\n")
    return path


def test_trace_parser_separates_helper_call_and_rocc_instructions(
    tmp_path: Path,
) -> None:
    operation_word = 0x000000FB
    lifecycle_word = 0x0EF7B07B
    path = _trace(
        tmp_path,
        [
            b"core   0: 0x0000000000001000 (0x00000013) nop",
            b"core   0: 0x0000000000001004 (0x000000fb) unknown",
            b"core   0: Executed 3 times",
            b"core   0: 0x0000000000001008 (0x00000013) nop",
            b"core   0: Executed 5 times",
            b"core   0: 0x0000000000002000 (0x0ef7b07b) unknown",
        ],
    )
    result = runner.parse_spike_trace(
        path,
        0x1000,
        0x1100,
        {0x1004: operation_word, 0x2000: lifecycle_word},
    )
    assert result["actual_accelerator_call_count"] == 1
    assert result["runtime_rocc_instruction_record_count"] == 2
    assert result["runtime_rocc_dynamic_instruction_count"] == 4
    assert result["runtime_operation_rocc_dynamic_instruction_count"] == 3
    assert result["runtime_lifecycle_rocc_dynamic_instruction_count"] == 1


def test_trace_parser_rejects_missing_wrapper_entry(tmp_path: Path) -> None:
    path = _trace(
        tmp_path,
        [b"core 0: 0x0000000000002000 (0x0ef7b07b) unknown"],
    )
    with pytest.raises(runner.RunError, match="wrapper executed 0"):
        runner.parse_spike_trace(path, 0x1000, 0x1100, {0x2000: 0x0EF7B07B})


def test_trace_parser_rejects_static_runtime_site_mismatch(
    tmp_path: Path,
) -> None:
    path = _trace(
        tmp_path,
        [
            b"core 0: 0x0000000000001000 (0x00000013) nop",
            b"core 0: 0x0000000000001004 (0x000000fb) unknown",
            b"core 0: 0x0000000000002000 (0x0ef7b07b) unknown",
        ],
    )
    with pytest.raises(runner.RunError, match="differ"):
        runner.parse_spike_trace(
            path,
            0x1000,
            0x1100,
            {0x1004: 0x000000FB, 0x2004: 0x0EF7B07B},
        )


def test_objdump_decoder_uses_low_seven_opcode_bits() -> None:
    text = """
  1000: 000000fb .insn 4, 0x000000fb
  1004: 00000013 nop
  2000: 0ef7b07b .insn 4, 0x0ef7b07b
"""
    assert runner._static_custom_sites(text) == {
        0x1000: 0x000000FB,
        0x2000: 0x0EF7B07B,
    }


def test_nm_parser_accepts_sized_and_unsized_symbols() -> None:
    symbols = runner._parse_symbols(
        "0000000080000000 T _start\n"
        "0000000080001000 D tohost\n"
        "0000000080002090 0000000000000918 "
        "T voyager_gemmini_operation\n"
    )
    assert symbols["_start"] == {
        "address": 0x80000000,
        "size": None,
        "type": "T",
    }
    assert symbols["tohost"]["size"] is None
    assert symbols["voyager_gemmini_operation"]["size"] == 0x918


def _stack_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    stack = tmp_path / "stack"
    tools = {}
    tool_identities = {}
    for name, (relative, _) in runner.TOOL_IDENTITIES.items():
        path = stack / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"tool:{name}".encode())
        digest = runner.sha256_file(path)
        tools[name] = {"path": relative, "sha256": digest}
        tool_identities[name] = (relative, digest)
    monkeypatch.setattr(runner, "TOOL_IDENTITIES", tool_identities)
    sources = {}
    for name, (relative, revision) in runner.SOURCE_IDENTITIES.items():
        (stack / relative).mkdir(parents=True, exist_ok=True)
        sources[name] = {"path": relative, "revision": revision}
    for relative in runner.FROZEN_FILE_HASHES:
        path = stack / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"frozen:{relative}\n")
    files = {
        relative: runner.sha256_file(stack / relative)
        for relative in runner.FROZEN_FILE_HASHES
    }
    monkeypatch.setattr(runner, "FROZEN_FILE_HASHES", files)
    manifest = {
        "schema_version": 2,
        "tools": tools,
        "sources": sources,
        "files": files,
    }
    manifest_path = tmp_path / "stack-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, stack


def test_stack_manifest_validates_every_frozen_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, stack_root = _stack_manifest(tmp_path, monkeypatch)
    revisions = {
        (tmp_path / "stack" / relative).resolve(): revision
        for relative, revision in runner.SOURCE_IDENTITIES.values()
    }
    monkeypatch.setattr(runner, "_git_revision", revisions.__getitem__)
    stack, tools, evidence = runner.validate_stack_manifest(
        manifest_path, stack_root
    )
    assert stack == (tmp_path / "stack").resolve()
    assert set(tools) == set(runner.TOOL_IDENTITIES)
    assert evidence["manifest"]["sha256"] == runner.sha256_file(manifest_path)

    manifest = json.loads(manifest_path.read_text())
    tool_name = next(iter(runner.TOOL_IDENTITIES))
    manifest["tools"][tool_name]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(runner.RunError, match="pinned tool hash"):
        runner.validate_stack_manifest(manifest_path, stack_root)

    manifest["tools"][tool_name]["sha256"] = runner.TOOL_IDENTITIES[tool_name][
        1
    ]
    manifest["stack_root"] = str(stack_root)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(runner.RunError, match="wrong stack manifest fields"):
        runner.validate_stack_manifest(manifest_path, stack_root)


def test_stack_inventory_is_t5_local() -> None:
    flattened = "\n".join(
        [
            *(path for path, _ in runner.TOOL_IDENTITIES.values()),
            *(path for path, _ in runner.SOURCE_IDENTITIES.values()),
            *runner.FROZEN_FILE_HASHES,
        ]
    )
    assert "chipyard" not in flattened.lower()
    assert "CHIPYARD" not in flattened
    assert runner.ROCC_ROOT == "gemmini/software/gemmini-rocc-tests"
    assert runner.TOOL_IDENTITIES["spike"][0].startswith("gemmini-stack/")


def test_readback_parser_and_exact_comparison() -> None:
    lines = [f"OUTPUT {index} 0" for index in range(256)]
    lines.append(
        "SUMMARY count=256 exact_mismatches=0 tolerance_mismatches=0 "
        "first=none max_abs=0 squared_error=0"
    )
    actual, summary = runner.parse_output("\n".join(lines) + "\n")
    comparison = runner.compare_output(actual, np.zeros(256, dtype=np.int8))
    assert summary["exact_mismatch_count"] == 0
    assert summary["tolerance_mismatch_count"] == 0
    assert comparison["exact_mismatch_count"] == 0
    assert comparison["tolerance_mismatch_count"] == 0
    assert comparison["logical_equal"] is True


def test_readback_parser_accepts_only_pinned_extension_footer() -> None:
    outputs = [f"OUTPUT {index} 0" for index in range(256)]
    summary = (
        "SUMMARY count=256 exact_mismatches=0 tolerance_mismatches=0 "
        "first=none max_abs=0 squared_error=0"
    )
    footer = [
        "Gemmini extension configured with:",
        "    dim = 16",
    ]
    actual, parsed = runner.parse_output(
        "\n".join([*outputs, summary, *footer]) + "\n"
    )
    assert np.array_equal(actual, np.zeros(256, dtype=np.int8))
    assert parsed["element_count"] == 256

    invalid_streams = [
        [*outputs, summary, footer[0]],
        [*outputs, summary, footer[0], "    dim = 8"],
        [*outputs, summary, footer[0], "   dim = 16"],
        [*outputs, summary, *footer, "unexpected"],
        [*outputs, summary, *footer, *footer],
        [*outputs, footer[0], footer[1], summary],
        [*outputs[:-1], footer[0], footer[1], outputs[-1], summary],
        [*outputs, summary, "", *footer],
    ]
    for lines in invalid_streams:
        with pytest.raises(runner.RunError):
            runner.parse_output("\n".join(lines) + "\n")
    with pytest.raises(runner.RunError):
        runner.parse_output("\n".join([*outputs, summary, *footer]) + "\n\n")


def test_readback_parser_requires_terminal_summary_ordering() -> None:
    outputs = [f"OUTPUT {index} 0" for index in range(256)]
    summary = (
        "SUMMARY count=256 exact_mismatches=0 tolerance_mismatches=0 "
        "first=none max_abs=0 squared_error=0"
    )
    malformed = [
        "\n".join([summary, *outputs]) + "\n",
        "\n".join([*outputs[:-1], summary, outputs[-1]]) + "\n",
    ]
    for text in malformed:
        with pytest.raises(runner.RunError):
            runner.parse_output(text)


def test_comparison_accepts_one_code_but_records_nonexact_result() -> None:
    expected = np.zeros(256, dtype=np.int8)
    actual = expected.copy()
    actual[0] = 1
    comparison = runner.compare_output(actual, expected)
    assert comparison["exact_match_count"] == 255
    assert comparison["exact_mismatch_count"] == 1
    assert comparison["tolerance_mismatch_count"] == 0
    assert comparison["maximum_absolute_error"] == 1
    assert comparison["within_tolerance"] is True
    assert comparison["exact_match_required"] is False
    assert "exact_result_expected" not in comparison
    actual[1] = 2
    comparison = runner.compare_output(actual, expected)
    assert comparison["tolerance_mismatch_count"] == 1
    assert comparison["within_tolerance"] is False


def test_runner_records_exact_output_only_as_nonrequired_statistics() -> None:
    source = Path(runner.__file__).read_text()
    assert '"exact_output_match_count"' in source
    assert '"exact_output_mismatch_count"' in source
    assert '"exact_output_required": False' in source
    assert "exact_output_expected" not in source

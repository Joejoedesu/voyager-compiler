# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Focused tests for Gemmini workload runtime-capture contracts."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from google.protobuf import text_format

from conversion.gemmini import convert as anchor
from conversion.gemmini import workload

_FUSED_NAME = "downstream_conv_fused"
_SHAPE = (1, 2, 2, 2)
_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "docs/targets/gemmini/workload-injection-contract.md"
)


def _tensor(
    message,
    node: str,
    shape: tuple[int, ...],
    dtype: str,
    address: int | None = None,
) -> None:
    message.node = node
    message.shape.extend(shape)
    message.dtype = dtype
    if address is not None:
        message.memory.partition = 0
        message.memory.address = address
        message.memory.instance = "DRAM"


def _tensor_argument(
    operation,
    name: str,
    node: str,
    shape: tuple[int, ...],
    dtype: str,
    address: int | None = None,
) -> None:
    _tensor(
        operation.kwargs[name].tensor,
        node,
        shape,
        dtype,
        address,
    )


def _pair_argument(operation, name: str, values: tuple[int, int]) -> None:
    operation.kwargs[name].int_list.values.extend(values)


def _write_model(model_dir: Path) -> None:
    tensor_dir = model_dir / "tensor_files"
    tensor_dir.mkdir(parents=True)
    model = anchor._load_proto().Model()
    _tensor(model.inputs.add(), "prior_codes", _SHAPE, "int8", 0x1000)
    _tensor(
        model.parameters.add(),
        "weight",
        (2, 2, 1, 1),
        "int8",
        0x2000,
    )
    _tensor(model.parameters.add(), "bias", (2,), "int32", 0x2100)

    record = model.ops.add()
    fused = record.fused_op
    fused.name = _FUSED_NAME

    conv = fused.op_list.add()
    conv.name = "conv"
    conv.op = "call_function"
    conv.target = "conv2d"
    _tensor_argument(
        conv,
        "input",
        "prior_codes",
        _SHAPE,
        "int8",
        0x1000,
    )
    _tensor_argument(
        conv,
        "weight",
        "weight",
        (2, 2, 1, 1),
        "int8",
        0x2000,
    )
    _tensor_argument(conv, "bias", "bias", (2,), "int32", 0x2100)
    for name, values in (
        ("stride", (1, 1)),
        ("padding", (0, 0)),
        ("dilation", (1, 1)),
    ):
        _pair_argument(conv, name, values)
    conv.kwargs["groups"].int_value = 1

    dequantize = fused.op_list.add()
    dequantize.name = "accumulator_dq"
    dequantize.op = "call_function"
    dequantize.target = "dequantize"
    _tensor_argument(
        dequantize,
        "input",
        "conv",
        _SHAPE,
        "float32",
    )
    _tensor_argument(
        dequantize,
        "scale",
        "accumulator_scale",
        (1,),
        "float32",
    )

    quantize = fused.op_list.add()
    quantize.name = "output_q"
    quantize.op = "call_function"
    quantize.target = "quantize"
    _tensor_argument(
        quantize,
        "input",
        "accumulator_dq",
        _SHAPE,
        "float32",
    )
    _tensor_argument(
        quantize,
        "scale",
        "output_scale",
        (1,),
        "float32",
    )
    quantize.kwargs["rounding"].str_value = "nearest"
    quantize.kwargs["quant_min"].float_value = -128.0
    quantize.kwargs["quant_max"].float_value = 127.0
    _tensor(record.output, _FUSED_NAME, _SHAPE, "int8", 0x3000)

    (model_dir / "model.txt").write_text(
        text_format.MessageToString(model),
        encoding="utf-8",
    )
    tensor_values = {
        "weight": np.asarray([1, -1, 2, -2], dtype=np.float32),
        "bias": np.asarray([0, 1], dtype=np.float32),
        "accumulator_scale": np.asarray([0.25], dtype=np.float32),
        "output_scale": np.asarray([0.5], dtype=np.float32),
        _FUSED_NAME: np.arange(-3, 5, dtype=np.float32).reshape(_SHAPE),
    }
    for name, values in tensor_values.items():
        values.astype("<f4").tofile(tensor_dir / f"{name}.bin")


def _binding(path: Path, logical_dtype: str) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "container_dtype": "float32",
        "logical_dtype": logical_dtype,
        "shape": list(_SHAPE),
    }


def _write_capture(
    capture_dir: Path,
    model_dir: Path,
    input_values: np.ndarray,
    *,
    input_logical_dtype: str = "int8",
) -> Path:
    capture_dir.mkdir()
    input_path = capture_dir / "input.bin"
    output_path = capture_dir / "output-codes.bin"
    input_values.astype("<f4").reshape(_SHAPE).tofile(input_path)
    np.arange(-3, 5, dtype="<f4").reshape(_SHAPE).tofile(output_path)
    manifest = {
        "schema_version": 1,
        "producer": "gemmini-workload-test",
        "canonical_model": {
            "path": "model.txt",
            "sha256": anchor.sha256_file(model_dir / "model.txt"),
        },
        "regions": {
            _FUSED_NAME: {
                "input": _binding(input_path, input_logical_dtype),
                "output_codes": _binding(output_path, "int8"),
            }
        },
    }
    manifest_path = capture_dir / "runtime-capture.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class GemminiRuntimeCaptureContractTest(unittest.TestCase):
    """Exercise the downstream Conv logical-dtype boundary."""

    def _fixture(
        self,
        root: Path,
        input_values: np.ndarray,
        *,
        input_logical_dtype: str = "int8",
    ) -> tuple[Path, Path, Path]:
        model_dir = root / "model"
        capture_dir = root / "capture"
        _write_model(model_dir)
        manifest_path = _write_capture(
            capture_dir,
            model_dir,
            input_values,
            input_logical_dtype=input_logical_dtype,
        )
        return model_dir, capture_dir, manifest_path

    def _convert(
        self,
        root: Path,
        model_dir: Path,
        capture_dir: Path,
        name: str = "converted",
    ) -> dict[str, object]:
        return workload.convert_region(
            model_dir,
            root / name,
            _CONTRACT,
            workload.WORKLOAD_CONTRACT_SHA256,
            _FUSED_NAME,
            capture_dir,
        )

    def test_integral_float32_container_is_accepted_as_logical_int8(self):
        values = np.asarray(
            [-128, -8, -1, 0, 1, 7, 42, 127],
            dtype=np.float32,
        ).reshape(_SHAPE)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            model_dir, capture_dir, _ = self._fixture(root, values)

            manifest = self._convert(root, model_dir, capture_dir)

            self.assertEqual(manifest["status"], "converted")
            packed = np.fromfile(
                root / "converted/payloads/input.bin",
                dtype=np.int8,
            )
            expected = anchor.pack_activation(values.astype(np.int8))
            np.testing.assert_array_equal(packed, expected.reshape(-1))

    def test_downstream_float32_label_fails_without_changing_bytes(self):
        values = np.asarray(
            [-128, -8, -1, 0, 1, 7, 42, 127],
            dtype=np.float32,
        ).reshape(_SHAPE)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            model_dir, capture_dir, manifest_path = self._fixture(
                root,
                values,
            )
            input_path = capture_dir / "input.bin"
            before = input_path.read_bytes()
            manifest = _load_manifest(manifest_path)
            region = manifest["regions"][_FUSED_NAME]
            region["input"]["logical_dtype"] = "float32"
            _write_manifest(manifest_path, manifest)

            with self.assertRaisesRegex(
                anchor.ConversionError,
                "runtime input dtype differs",
            ):
                self._convert(root, model_dir, capture_dir)
            self.assertEqual(input_path.read_bytes(), before)

    def test_fractional_logical_int8_input_fails_closed(self):
        values = np.asarray(
            [-8, -1, 0, 0.5, 1, 2, 3, 7],
            dtype=np.float32,
        ).reshape(_SHAPE)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            model_dir, capture_dir, _ = self._fixture(root, values)

            with self.assertRaisesRegex(
                anchor.ConversionError,
                "connected input codes are not integral",
            ):
                self._convert(root, model_dir, capture_dir)

    def test_out_of_range_logical_int8_input_fails_closed(self):
        values = np.asarray(
            [-129, -8, -1, 0, 1, 2, 3, 128],
            dtype=np.float32,
        ).reshape(_SHAPE)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            model_dir, capture_dir, _ = self._fixture(root, values)

            with self.assertRaisesRegex(
                anchor.ConversionError,
                "connected input codes are outside INT8",
            ):
                self._convert(root, model_dir, capture_dir)

    def test_capture_rejects_canonical_model_hash_drift(self):
        values = np.arange(-4, 4, dtype=np.float32).reshape(_SHAPE)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            model_dir, capture_dir, _ = self._fixture(root, values)
            model_path = model_dir / "model.txt"
            model_path.write_text(
                model_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                anchor.ConversionError,
                "runtime capture is bound to another canonical model",
            ):
                self._convert(root, model_dir, capture_dir)


if __name__ == "__main__":
    unittest.main()

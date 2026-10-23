#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Build, inspect, trace, and compare converted Gemmini collateral."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

CONTRACT_ID = "gemmini-R-default-s8-operations-20260816"
INJECTION_CONTRACT_SHA256 = (
    "a36b7a95d5d47bc8531abff4b72fa463731957f92ae5f34705c8cd7b5876d612"
)
OPERATION_ID = "aten::conv2d"
WRAPPER_SYMBOL = "voyager_gemmini_operation"
OUTPUT_COUNT = 256
CUSTOM_3_OPCODE = 0x7B
REPO_ROOT = Path(__file__).resolve().parents[2]
INJECTION_CONTRACT_PATH = (
    REPO_ROOT / "docs/targets/gemmini/injection-contract.md"
).resolve()

COMPILE_FLAGS = (
    "-DPREALLOCATE=1",
    "-DMULTITHREAD=1",
    "-mcmodel=medany",
    "-std=gnu99",
    "-O2",
    "-ffast-math",
    "-fno-common",
    "-fno-builtin-printf",
    "-fno-tree-loop-distribute-patterns",
    "-march=rv64gc",
    "-Wa,-march=rv64gc",
    "-DID_STRING=",
    "-DPRINT_TILE=0",
    "-DBAREMETAL=1",
)
LINK_FLAGS = ("-nostdlib", "-nostartfiles", "-static", "-lm", "-lgcc")

TOOL_IDENTITIES = {
    "riscv64-unknown-elf-gcc": (
        "gemmini-stack/toolchain-env/riscv-tools/bin/"
        "riscv64-unknown-elf-gcc",
        "4edd51fc7e8aecc1967d95cff423685dfe6565739fa208f876413fa8277ab8a4",
    ),
    "riscv64-unknown-elf-readelf": (
        "gemmini-stack/toolchain-env/riscv-tools/bin/"
        "riscv64-unknown-elf-readelf",
        "4289d8c8d012dd7ffd8ea53f43d5bbcbb1d2b4ae8ed0b831e1bb0b748f8b0a5a",
    ),
    "riscv64-unknown-elf-nm": (
        "gemmini-stack/toolchain-env/riscv-tools/bin/riscv64-unknown-elf-nm",
        "1637fa2367d7e758962025edc1163e6395f300591eb2c2cdb483cdaf10ad7215",
    ),
    "riscv64-unknown-elf-objdump": (
        "gemmini-stack/toolchain-env/riscv-tools/bin/"
        "riscv64-unknown-elf-objdump",
        "1c0a65ea4d86a0fdc57e22a49b777f80881b0a3df9b2ac0634dc651b295c0410",
    ),
    "spike": (
        "gemmini-stack/runtime-prefix/bin/spike",
        "d74f929c75d07dfaf6da2dd85f881f2743cba4aa23590c4cfcb1f1d3dc8cbb09",
    ),
    "libgemmini.so": (
        "gemmini-stack/runtime-prefix/lib/libgemmini.so",
        "9f7f95a0f131bec9c39810491f8accc42badce4a47da6709676c4c96ce8e68cc",
    ),
}
SOURCE_IDENTITIES = {
    "gemmini": (
        "gemmini",
        "8c3f9923a44a2fe2c7930587be297d6d4f8c09ca",
    ),
    "gemmini-rocc-tests": (
        "gemmini/software/gemmini-rocc-tests",
        "7c540b3adf1b86ad93d07f893abe3a73489b568e",
    ),
    "riscv-tests": (
        "gemmini/software/gemmini-rocc-tests/riscv-tests",
        "c84daca8824635b7d896003c78f9c6245997cf7a",
    ),
    "rocc-software": (
        "gemmini/software/gemmini-rocc-tests/rocc-software",
        "fddb795a0b52e82f8f4ce9ead9b1428440a62ab0",
    ),
    "libgemmini": (
        "gemmini-stack/libgemmini",
        "ea8f7ed7afd68e001fb06ddccad9a023c990961d",
    ),
    "spike": (
        "gemmini-stack/spike-src",
        "d182829e68f1393f223c1233cf22d7adcc89042d",
    ),
}
ROCC_ROOT = "gemmini/software/gemmini-rocc-tests"
COMMON_ROOT = f"{ROCC_ROOT}/riscv-tests/benchmarks/common"
FROZEN_FILE_HASHES = {
    "gemmini/src/main/scala/gemmini/Configs.scala": (
        "178c21ac741c89c08158efa71fb34da35d27b75bd50bde5116a5e19178f8391b"
    ),
    "gemmini/src/main/scala/gemmini/GemminiConfigs.scala": (
        "7294a082a1df4ae7aa7d81adc5bde75b7d5118683c083f6a4b998c9ce64d102e"
    ),
    f"{ROCC_ROOT}/include/gemmini.h": (
        "18801f4eab0cd2e81e25a6c2aa97c7ed644b693e8d2c24cdeb41ab3a7208336d"
    ),
    f"{ROCC_ROOT}/include/gemmini_params.h": (
        "3758ae967af3a179497660970201093a7fb624be00173990ce33d3f5c38da924"
    ),
    f"{ROCC_ROOT}/include/gemmini_counter.h": (
        "5826242287784b9a383d45fdcb6791f8a1651d9d320a1a5f011585a9885b5669"
    ),
    f"{ROCC_ROOT}/rocc-software/src/xcustom.h": (
        "9582eb00cee774e4680fde6408a0d607bba85196a0a4fd803f7b7ddd801aad69"
    ),
    f"{COMMON_ROOT}/test.ld": (
        "77ff20ffb473836e779809ff47b34fbcc391152025d19de4d6fd7380948658f4"
    ),
    f"{COMMON_ROOT}/crt.S": (
        "2794cf1a7ae21537060453864c94a811869e72335e98b9eb1e8aeee4de7f0508"
    ),
    f"{COMMON_ROOT}/syscalls.c": (
        "71b02fbad1f1d29c7f13fb6ab96a18f90cd7c257398b4a52cd8c9f2b459dfcab"
    ),
    "gemmini-stack/libgemmini/gemmini.cc": (
        "c807b51c7b84b86a2fb270e75af54aae9ef82312b228ce666f716f7f653be4f4"
    ),
    "gemmini-stack/libgemmini/gemmini.h": (
        "264f8858449123efd554642058d69966cdc35d194a3878d550142b3e9594284b"
    ),
    "gemmini-stack/toolchain-env/bin/dtc": (
        "cf4a6fcf83e0f8c2d01bb62be1a93c332bf55819e93bf266d80da0705cabd9b7"
    ),
}
PINNED_NATIVE_DEPENDENCIES = {
    "gemmini.h": (
        f"{ROCC_ROOT}/include/gemmini.h",
        FROZEN_FILE_HASHES[f"{ROCC_ROOT}/include/gemmini.h"],
    ),
    "gemmini_params.h": (
        f"{ROCC_ROOT}/include/gemmini_params.h",
        FROZEN_FILE_HASHES[f"{ROCC_ROOT}/include/gemmini_params.h"],
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
LAYOUT_COLUMNS = (
    "role",
    "logical_linear_index",
    "physical_linear_index",
    "value",
)
LAYOUT_ROLE_COUNTS = {
    "input": 256,
    "weight": 2304,
    "output": 256,
    "output_oracle": 256,
}
LAYOUT_ROW_COUNT = sum(LAYOUT_ROLE_COUNTS.values())
ORACLE_MANIFEST = "oracle-manifest.json"
ORACLE_ARTIFACT = "pt2e_reference.bin"
ORACLE_TERMINAL_ARTIFACT = "pt2e_terminal_reference.bin"
ORACLE_PRODUCER = "natural-pytorch-pt2e-qdq"
ORACLE_SCHEME_ID = "gemmini.s8-symmetric-per-tensor"
ORACLE_QDQ_FORM = "gemmini.s8s8-s32-to-s8-f32-rne-sat"
ORACLE_QPARAM_NAMES = frozenset(
    {
        "input_scale",
        "input_zero_point",
        "weight_scale",
        "weight_zero_point",
        "output_scale",
        "output_zero_point",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CSV_UNSIGNED_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
CSV_SIGNED_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)$")

OUTPUT_RE = re.compile(r"^OUTPUT ([0-9]+) (-?[0-9]+)$")
SUMMARY_RE = re.compile(
    r"^SUMMARY count=([0-9]+) exact_mismatches=([0-9]+) "
    r"tolerance_mismatches=([0-9]+) first=(none|[0-9]+) "
    r"max_abs=([0-9]+) squared_error=([0-9]+)$"
)
EXTENSION_FOOTER_HEADER = "Gemmini extension configured with:"
PINNED_GEMMINI_DIM = 16
EXTENSION_FOOTER_DIM = f"    dim = {PINNED_GEMMINI_DIM}"
TRACE_INSTRUCTION_RE = re.compile(
    rb"^core\s+(?P<core>[0-9]+):\s+"
    rb"0x(?P<pc>[0-9a-fA-F]+)\s+"
    rb"\(0x(?P<word>[0-9a-fA-F]+)\)\s*(?P<text>.*)$"
)
TRACE_RLE_RE = re.compile(
    rb"^core\s+(?P<core>[0-9]+):\s+Executed\s+" rb"(?P<count>[0-9]+)\s+times$"
)
NM_RE = re.compile(
    r"^([0-9a-fA-F]+)\s+(?:([0-9a-fA-F]+)\s+)?" r"([A-Za-z])\s+(\S+)$"
)
OBJDUMP_WORD_RE = re.compile(
    r"^\s*([0-9a-fA-F]+):\s+" r"([0-9a-fA-F]{4}|[0-9a-fA-F]{8})\s+"
)


class RunError(RuntimeError):
    """Frozen build, trace, or comparison validation failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RunError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RunError(f"cannot load {label} {path}: {error}") from error
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def _normalized_relative(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be a path")
    candidate = value.replace("\\", "/")
    path = PurePosixPath(candidate)
    _require(not path.is_absolute(), f"{label} must be stack-relative")
    _require(
        candidate == path.as_posix()
        and all(part not in {"", ".", ".."} for part in path.parts),
        f"{label} is not normalized: {value}",
    )
    return candidate


def _stack_path(stack: Path, relative: Any, label: str) -> Path:
    normalized = _normalized_relative(relative, label)
    path = (stack / normalized).resolve()
    try:
        path.relative_to(stack)
    except ValueError as error:
        raise RunError(f"{label} escapes the frozen stack") from error
    return path


def _git_revision(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    _require(completed.returncode == 0, f"cannot identify revision: {path}")
    return completed.stdout.strip()


def validate_stack_manifest(
    manifest_path: Path,
    stack_root: Path,
) -> tuple[Path, dict[str, Path], dict[str, Any]]:
    """Validate the portable runtime lock against one explicit stack root."""
    _require(
        manifest_path.is_file(), f"stack manifest is missing: {manifest_path}"
    )
    manifest = _load_json(manifest_path, "stack manifest")
    expected_keys = {
        "schema_version",
        "tools",
        "sources",
        "files",
    }
    _require(set(manifest) == expected_keys, "wrong stack manifest fields")
    _require(manifest["schema_version"] == 2, "wrong stack manifest schema")
    stack = Path(stack_root).resolve()
    _require(stack.is_dir(), f"frozen stack root is missing: {stack}")

    tools = manifest["tools"]
    _require(isinstance(tools, dict), "stack tools must be an object")
    _require(set(tools) == set(TOOL_IDENTITIES), "wrong frozen tool inventory")
    resolved_tools = {}
    tool_evidence = {}
    for name, (expected_relative, expected_sha256) in TOOL_IDENTITIES.items():
        record = tools[name]
        _require(isinstance(record, dict), f"invalid tool record: {name}")
        _require(
            set(record) == {"path", "sha256"}, f"wrong tool fields: {name}"
        )
        _require(
            record["path"] == expected_relative, f"wrong tool path: {name}"
        )
        _require(
            record["sha256"] == expected_sha256,
            f"wrong pinned tool hash: {name}",
        )
        path = _stack_path(stack, record["path"], f"tool {name}")
        _require(path.is_file(), f"frozen tool is missing: {path}")
        actual = sha256_file(path)
        _require(
            actual == expected_sha256, f"frozen tool hash mismatch: {name}"
        )
        resolved_tools[name] = path
        tool_evidence[name] = {
            "path": str(path),
            "sha256": actual,
        }

    sources = manifest["sources"]
    _require(isinstance(sources, dict), "stack sources must be an object")
    _require(set(sources) == set(SOURCE_IDENTITIES), "wrong source inventory")
    source_evidence = {}
    for name, (
        expected_relative,
        expected_revision,
    ) in SOURCE_IDENTITIES.items():
        record = sources[name]
        _require(isinstance(record, dict), f"invalid source record: {name}")
        _require(
            set(record) == {"path", "revision"},
            f"wrong source fields: {name}",
        )
        _require(
            record["path"] == expected_relative, f"wrong source path: {name}"
        )
        _require(
            record["revision"] == expected_revision,
            f"wrong frozen source revision: {name}",
        )
        path = _stack_path(stack, record["path"], f"source {name}")
        actual = _git_revision(path)
        _require(
            actual == expected_revision, f"source revision mismatch: {name}"
        )
        source_evidence[name] = {
            "path": str(path),
            "revision": actual,
        }

    files = manifest["files"]
    _require(isinstance(files, dict), "stack files must be an object")
    _require(
        files == FROZEN_FILE_HASHES,
        "frozen file inventory or pinned hashes differ",
    )
    file_evidence = {}
    for relative, expected_hash in FROZEN_FILE_HASHES.items():
        path = _stack_path(stack, relative, f"frozen file {relative}")
        _require(path.is_file(), f"frozen file is missing: {path}")
        actual = sha256_file(path)
        _require(
            actual == expected_hash,
            f"frozen file hash mismatch: {relative}",
        )
        file_evidence[relative] = actual
    evidence = {
        "schema_version": 2,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "stack_root": str(stack),
        "tools": tool_evidence,
        "sources": source_evidence,
        "files": file_evidence,
        "compile_flags": list(COMPILE_FLAGS),
        "link_flags": list(LINK_FLAGS),
    }
    return stack, resolved_tools, evidence


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    records: list[dict[str, Any]],
    env: dict[str, str] | None = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout_path.write_text(error.stdout or "")
        stderr_path.write_text(error.stderr or "")
        records.append(
            {
                "argv": command,
                "shell": shlex.join(command),
                "cwd": str(cwd),
                "returncode": "python-timeout",
                "wall_seconds": time.monotonic() - started,
                "stdout": stdout_path.name,
                "stderr": stderr_path.name,
            }
        )
        raise RunError(f"command timed out: {shlex.join(command)}") from error
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    records.append(
        {
            "argv": command,
            "shell": shlex.join(command),
            "cwd": str(cwd),
            "returncode": completed.returncode,
            "wall_seconds": time.monotonic() - started,
            "stdout": stdout_path.name,
            "stderr": stderr_path.name,
            "environment": (
                {key: env[key] for key in ("PATH", "LD_LIBRARY_PATH")}
                if env is not None
                else None
            ),
        }
    )
    return completed


def _validate_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _expected_native_dependencies() -> dict[str, dict[str, str]]:
    return {
        name: {
            "path": relative,
            "sha256": expected_sha256,
        }
        for name, (
            relative,
            expected_sha256,
        ) in PINNED_NATIVE_DEPENDENCIES.items()
    }


def _validate_native_dependencies(
    manifest: dict[str, Any], stack: Path | None
) -> None:
    expected = _expected_native_dependencies()
    _require(
        manifest.get("native_dependencies") == expected,
        "conversion native dependencies are not pinned",
    )
    if stack is not None:
        for name, record in expected.items():
            path = _stack_path(
                stack, record["path"], f"native dependency {name}"
            )
            _require(
                path.is_file(), f"pinned native dependency is missing: {name}"
            )
            _require(
                sha256_file(path) == record["sha256"],
                f"pinned native dependency hash mismatch: {name}",
            )


def _validate_qparams(
    value: Any,
    label: str,
    *,
    expected: dict[str, float | int] | None = None,
) -> dict[str, float | int]:
    _require(isinstance(value, dict), f"{label} must be an object")
    _require(set(value) == ORACLE_QPARAM_NAMES, f"wrong {label} fields")
    for name in sorted(ORACLE_QPARAM_NAMES):
        expected_type = float if name.endswith("_scale") else int
        _require(
            type(value[name]) is expected_type,
            f"wrong {label} type: {name}",
        )
        if expected_type is float:
            _require(
                bool(np.isfinite(value[name])) and value[name] > 0,
                f"invalid {label} value: {name}",
            )
            _require(
                float(np.float32(value[name])) == value[name],
                f"{label} is not exactly FLOAT32: {name}",
            )
        else:
            _require(value[name] == 0, f"wrong {label} value: {name}")
        if expected is not None:
            if expected_type is float:
                equal = (
                    np.float32(value[name]).tobytes()
                    == np.float32(expected[name]).tobytes()
                )
            else:
                equal = value[name] == expected[name]
            _require(equal, f"wrong {label} value: {name}")
    return value


def _read_oracle_codes(path: Path, label: str) -> np.ndarray:
    _require(path.is_file(), f"{label} is missing: {path}")
    payload = path.read_bytes()
    _require(
        len(payload) == OUTPUT_COUNT * 4,
        f"{label} has the wrong FLOAT32 extent",
    )
    values = np.frombuffer(payload, dtype="<f4")
    _require(np.isfinite(values).all(), f"{label} contains a non-finite value")
    _require(
        np.equal(values, np.rint(values)).all(),
        f"{label} contains a fractional INT8 code",
    )
    _require(
        ((values >= -128) & (values <= 127)).all(),
        f"{label} contains an out-of-range INT8 code",
    )
    return np.ascontiguousarray(values.astype(np.int8).reshape(1, 16, 4, 4))


def _validate_oracle_binding(converted: Path, manifest: dict[str, Any]) -> None:
    canonical_value = manifest.get("canonical_directory")
    _require(
        isinstance(canonical_value, str) and bool(canonical_value),
        "canonical directory is missing",
    )
    canonical = Path(canonical_value).resolve()
    _require(
        str(canonical) == canonical_value,
        "canonical directory must be an absolute normalized path",
    )
    _require(canonical.is_dir(), f"canonical directory is missing: {canonical}")

    facts = manifest.get("operation_facts")
    _require(isinstance(facts, dict), "operation facts are missing")
    output_fact = facts.get("output")
    terminal_fact = facts.get("terminal_output")
    _require(isinstance(output_fact, dict), "output operation fact is missing")
    _require(
        isinstance(terminal_fact, dict), "terminal operation fact is missing"
    )
    output_source = _normalized_relative(
        output_fact.get("source"), "canonical code output source"
    )
    terminal_source = _normalized_relative(
        terminal_fact.get("source"), "canonical terminal output source"
    )
    quantization = manifest.get("quantization_facts")
    _require(isinstance(quantization, dict), "quantization facts are missing")
    expected_qparams = _validate_qparams(
        quantization.get("qparams"), "conversion qparams"
    )

    binding = manifest.get("oracle_binding")
    _require(isinstance(binding, dict), "oracle binding is missing")
    _require(
        set(binding)
        == {
            "manifest",
            "code_reference",
            "terminal_reference",
            "canonical_code_output",
            "canonical_terminal_output",
            "qparams",
        },
        "wrong oracle binding fields",
    )
    _validate_qparams(
        binding["qparams"], "oracle qparams", expected=expected_qparams
    )

    manifest_record = binding["manifest"]
    _require(
        isinstance(manifest_record, dict), "oracle manifest binding is invalid"
    )
    _require(
        set(manifest_record) == {"path", "sha256"},
        "wrong oracle manifest binding fields",
    )
    _require(
        manifest_record["path"] == ORACLE_MANIFEST,
        "wrong oracle manifest path",
    )
    manifest_sha256 = _validate_sha256(
        manifest_record["sha256"], "oracle manifest hash"
    )

    reference = binding["code_reference"]
    _require(isinstance(reference, dict), "code reference binding is invalid")
    expected_reference_fields = {
        "path",
        "sha256",
        "producer",
        "scheme_id",
        "qdq_form",
        "container_dtype",
        "logical_dtype",
        "shape",
    }
    _require(
        set(reference) == expected_reference_fields,
        "wrong code reference binding fields",
    )
    _require(reference["path"] == ORACLE_ARTIFACT, "wrong code reference path")
    _require(reference["producer"] == ORACLE_PRODUCER, "wrong oracle producer")
    _require(
        reference["scheme_id"] == ORACLE_SCHEME_ID,
        "wrong oracle quantization scheme",
    )
    _require(reference["qdq_form"] == ORACLE_QDQ_FORM, "wrong oracle Q/DQ form")
    _require(
        reference["container_dtype"] == "float32",
        "oracle container must be FLOAT32",
    )
    _require(
        reference["logical_dtype"] == "int8", "oracle must be logical INT8"
    )
    _require(reference["shape"] == [1, 16, 4, 4], "wrong oracle shape")
    reference_sha256 = _validate_sha256(
        reference["sha256"], "oracle reference hash"
    )

    terminal_reference = binding["terminal_reference"]
    _require(
        isinstance(terminal_reference, dict),
        "terminal reference binding is invalid",
    )
    _require(
        set(terminal_reference)
        == {
            "path",
            "sha256",
            "producer",
            "scheme_id",
            "qdq_form",
            "dtype",
            "shape",
        },
        "wrong terminal reference binding fields",
    )
    _require(
        terminal_reference["path"] == ORACLE_TERMINAL_ARTIFACT,
        "wrong terminal reference path",
    )
    _require(
        terminal_reference["producer"] == ORACLE_PRODUCER,
        "wrong terminal reference producer",
    )
    _require(
        terminal_reference["scheme_id"] == ORACLE_SCHEME_ID,
        "wrong terminal reference scheme",
    )
    _require(
        terminal_reference["qdq_form"] == ORACLE_QDQ_FORM,
        "wrong terminal reference Q/DQ form",
    )
    _require(
        terminal_reference["dtype"] == "float32",
        "terminal reference must be FLOAT32",
    )
    _require(
        terminal_reference["shape"] == [1, 16, 4, 4],
        "wrong terminal reference shape",
    )
    terminal_reference_sha256 = _validate_sha256(
        terminal_reference["sha256"], "terminal reference hash"
    )

    canonical_output = binding["canonical_code_output"]
    _require(
        isinstance(canonical_output, dict),
        "canonical code output binding is invalid",
    )
    _require(
        set(canonical_output) == {"path", "sha256", "verified_equal"},
        "wrong canonical code output fields",
    )
    _require(
        canonical_output["path"] == output_source,
        "oracle binds the wrong canonical code output",
    )
    _require(
        canonical_output["verified_equal"] is True,
        "oracle binding does not assert equality",
    )
    canonical_sha256 = _validate_sha256(
        canonical_output["sha256"], "canonical code output hash"
    )

    canonical_terminal = binding["canonical_terminal_output"]
    _require(
        isinstance(canonical_terminal, dict),
        "canonical terminal output binding is invalid",
    )
    _require(
        set(canonical_terminal) == {"path", "sha256", "verified_equal"},
        "wrong canonical terminal output fields",
    )
    _require(
        canonical_terminal["path"] == terminal_source,
        "oracle binds the wrong canonical terminal output",
    )
    _require(
        canonical_terminal["verified_equal"] is True,
        "terminal oracle binding does not assert equality",
    )
    canonical_terminal_sha256 = _validate_sha256(
        canonical_terminal["sha256"], "canonical terminal output hash"
    )

    oracle_manifest_path = _stack_path(
        canonical, ORACLE_MANIFEST, "oracle manifest"
    )
    reference_path = _stack_path(canonical, ORACLE_ARTIFACT, "oracle reference")
    terminal_reference_path = _stack_path(
        canonical, ORACLE_TERMINAL_ARTIFACT, "terminal reference"
    )
    canonical_path = _stack_path(
        canonical, output_source, "canonical code output"
    )
    canonical_terminal_path = _stack_path(
        canonical, terminal_source, "canonical terminal output"
    )
    _require(
        sha256_file(oracle_manifest_path) == manifest_sha256,
        "oracle manifest hash mismatch",
    )
    _require(
        sha256_file(reference_path) == reference_sha256,
        "oracle reference hash mismatch",
    )
    _require(
        sha256_file(terminal_reference_path) == terminal_reference_sha256,
        "terminal reference hash mismatch",
    )
    _require(
        sha256_file(canonical_path) == canonical_sha256,
        "canonical code output hash mismatch",
    )
    _require(
        sha256_file(canonical_terminal_path) == canonical_terminal_sha256,
        "canonical terminal output hash mismatch",
    )

    sidecar = _load_json(oracle_manifest_path, "independent oracle manifest")
    expected_sidecar = {
        "schema_version": 1,
        "producer": ORACLE_PRODUCER,
        "operation_id": OPERATION_ID,
        "scheme_id": ORACLE_SCHEME_ID,
        "qdq_form": ORACLE_QDQ_FORM,
        "code_reference": {
            "path": ORACLE_ARTIFACT,
            "sha256": reference_sha256,
            "container_dtype": "float32",
            "logical_dtype": "int8",
            "shape": [1, 16, 4, 4],
        },
        "terminal_reference": {
            "path": ORACLE_TERMINAL_ARTIFACT,
            "sha256": terminal_reference_sha256,
            "dtype": "float32",
            "shape": [1, 16, 4, 4],
        },
        "canonical_code_output": {
            "path": output_source,
            "sha256": canonical_sha256,
            "equal_to_reference": True,
        },
        "canonical_terminal_output": {
            "path": terminal_source,
            "sha256": canonical_terminal_sha256,
            "equal_to_reference": True,
        },
        "qparams": expected_qparams,
    }
    _require(sidecar == expected_sidecar, "independent oracle sidecar differs")
    _validate_qparams(
        sidecar["qparams"],
        "oracle sidecar qparams",
        expected=expected_qparams,
    )

    reference_codes = _read_oracle_codes(reference_path, "oracle reference")
    canonical_codes = _read_oracle_codes(canonical_path, "canonical output")
    _require(
        np.array_equal(reference_codes, canonical_codes),
        "independent oracle differs from canonical code output",
    )
    terminal_values = np.frombuffer(
        terminal_reference_path.read_bytes(), dtype="<f4"
    )
    canonical_terminal_values = np.frombuffer(
        canonical_terminal_path.read_bytes(), dtype="<f4"
    )
    _require(
        terminal_values.size == OUTPUT_COUNT
        and canonical_terminal_values.size == OUTPUT_COUNT,
        "terminal reference has wrong extent",
    )
    _require(
        np.isfinite(terminal_values).all()
        and np.isfinite(canonical_terminal_values).all(),
        "terminal reference contains non-finite values",
    )
    _require(
        np.array_equal(terminal_values, canonical_terminal_values),
        "independent terminal differs from canonical terminal output",
    )
    expected_terminal = np.multiply(
        reference_codes.astype(np.float32),
        np.float32(expected_qparams["output_scale"]),
        dtype=np.float32,
    ).reshape(-1)
    _require(
        np.array_equal(terminal_values, expected_terminal),
        "terminal reference is not dequantized code output",
    )
    expected_physical = np.ascontiguousarray(
        reference_codes.transpose(0, 2, 3, 1)
    ).tobytes()
    _require(
        (converted / "payloads/output_oracle.bin").read_bytes()
        == expected_physical,
        "packed output oracle is not bound to the independent reference",
    )

    artifacts = manifest.get("canonical_artifacts")
    _require(
        isinstance(artifacts, dict), "canonical artifact evidence is missing"
    )
    _require(
        artifacts.get(ORACLE_MANIFEST)
        == {
            "sha256": manifest_sha256,
            "semantic_role": "independent-oracle-manifest",
        },
        "canonical oracle manifest evidence differs",
    )
    _require(
        artifacts.get(ORACLE_ARTIFACT)
        == {
            "sha256": reference_sha256,
            "semantic_role": "independent-code-oracle",
        },
        "canonical oracle reference evidence differs",
    )
    _require(
        artifacts.get(output_source)
        == {"sha256": canonical_sha256, "semantic_role": "fused code output"},
        "canonical output evidence differs from oracle binding",
    )
    _require(
        artifacts.get(ORACLE_TERMINAL_ARTIFACT)
        == {
            "sha256": terminal_reference_sha256,
            "semantic_role": "independent-terminal-oracle",
        },
        "terminal oracle reference evidence differs",
    )
    _require(
        artifacts.get(terminal_source)
        == {
            "sha256": canonical_terminal_sha256,
            "semantic_role": "terminal output",
        },
        "canonical terminal evidence differs from oracle binding",
    )


def _parse_csv_integer(value: Any, label: str, *, signed: bool) -> int:
    pattern = CSV_SIGNED_RE if signed else CSV_UNSIGNED_RE
    _require(
        isinstance(value, str) and pattern.fullmatch(value) is not None,
        f"{label} is not a canonical integer",
    )
    return int(value)


def _expected_physical_index(role: str, logical_index: int) -> int:
    if role == "weight":
        output, channel, kernel_row, kernel_column = np.unravel_index(
            logical_index, (16, 16, 3, 3)
        )
        return int(
            ((kernel_row * 3 + kernel_column) * 16 + channel) * 16 + output
        )
    batch, channel, row, column = np.unravel_index(logical_index, (1, 16, 4, 4))
    return int(((batch * 4 + row) * 4 + column) * 16 + channel)


def _validate_layout_map(converted: Path) -> None:
    payloads: dict[str, np.ndarray] = {}
    for role, count in LAYOUT_ROLE_COUNTS.items():
        path = converted / f"payloads/{role}.bin"
        _require(path.is_file(), f"layout payload is missing: {role}")
        payload = path.read_bytes()
        _require(
            len(payload) == count, f"layout payload has wrong size: {role}"
        )
        payloads[role] = np.frombuffer(payload, dtype=np.int8)
    _require(
        not np.any(payloads["output"]),
        "native output destination is not zero-initialized",
    )

    rows_by_role: dict[str, list[tuple[int, int, int]]] = {
        role: [] for role in LAYOUT_ROLE_COUNTS
    }
    path = converted / "layout-map.csv"
    _require(path.is_file(), f"layout map is missing: {path}")
    try:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            _require(
                tuple(reader.fieldnames or ()) == LAYOUT_COLUMNS,
                "layout map has wrong columns",
            )
            for row_number, row in enumerate(reader, start=2):
                _require(
                    set(row) == set(LAYOUT_COLUMNS),
                    f"layout row {row_number} has wrong fields",
                )
                role = row["role"]
                _require(
                    role in LAYOUT_ROLE_COUNTS,
                    f"layout row {row_number} has unknown role",
                )
                logical = _parse_csv_integer(
                    row["logical_linear_index"],
                    f"layout row {row_number} logical index",
                    signed=False,
                )
                physical = _parse_csv_integer(
                    row["physical_linear_index"],
                    f"layout row {row_number} physical index",
                    signed=False,
                )
                value = _parse_csv_integer(
                    row["value"],
                    f"layout row {row_number} value",
                    signed=True,
                )
                count = LAYOUT_ROLE_COUNTS[role]
                _require(
                    logical < count,
                    f"layout logical index out of range: {role}",
                )
                _require(
                    physical < count,
                    f"layout physical index out of range: {role}",
                )
                _require(
                    -128 <= value <= 127, f"layout value out of range: {role}"
                )
                _require(
                    physical == _expected_physical_index(role, logical),
                    f"layout permutation differs: {role} logical {logical}",
                )
                _require(
                    value == int(payloads[role][physical]),
                    "layout value differs from payload: "
                    f"{role} physical {physical}",
                )
                rows_by_role[role].append((logical, physical, value))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise RunError(f"cannot parse layout map {path}: {error}") from error

    _require(
        sum(len(rows) for rows in rows_by_role.values()) == LAYOUT_ROW_COUNT,
        "layout map has wrong total row count",
    )
    for role, count in LAYOUT_ROLE_COUNTS.items():
        rows = rows_by_role[role]
        _require(len(rows) == count, f"layout role has wrong count: {role}")
        _require(
            {logical for logical, _, _ in rows} == set(range(count)),
            f"layout logical indices are not a bijection: {role}",
        )
        _require(
            {physical for _, physical, _ in rows} == set(range(count)),
            f"layout physical indices are not a bijection: {role}",
        )


def _validate_generated_artifacts(
    converted: Path, manifest: dict[str, Any]
) -> None:
    records = manifest.get("generated_artifacts")
    _require(
        isinstance(records, list), "generated artifact inventory is missing"
    )
    _require(
        len(records) == len(GENERATED_ARTIFACT_PATHS),
        "generated artifact inventory has wrong size",
    )
    seen = set()
    for record in records:
        _require(isinstance(record, dict), "malformed generated artifact")
        _require(
            set(record) == {"path", "bytes", "sha256"},
            "wrong generated artifact fields",
        )
        relative = _normalized_relative(record["path"], "generated artifact")
        _require(
            relative not in seen, f"duplicate generated artifact: {relative}"
        )
        _require(
            relative in GENERATED_ARTIFACT_PATHS,
            f"unexpected generated artifact: {relative}",
        )
        seen.add(relative)
        path = _stack_path(converted, relative, "generated artifact")
        _require(path.is_file(), f"generated artifact is missing: {path}")
        _require(
            not path.is_symlink(),
            f"generated artifact is a symlink: {relative}",
        )
        _require(
            type(record["bytes"]) is int and record["bytes"] >= 0,
            f"invalid generated artifact size: {relative}",
        )
        _require(
            path.stat().st_size == record["bytes"],
            f"generated artifact size changed: {relative}",
        )
        expected_sha256 = _validate_sha256(
            record["sha256"], f"generated artifact hash: {relative}"
        )
        _require(
            sha256_file(path) == expected_sha256,
            f"generated artifact hash changed: {relative}",
        )
    _require(
        seen == GENERATED_ARTIFACT_PATHS, "generated artifacts are incomplete"
    )

    actual = set()
    for path in converted.rglob("*"):
        _require(
            not path.is_symlink(), f"converted collateral has symlink: {path}"
        )
        if path.is_file():
            actual.add(path.relative_to(converted).as_posix())
    _require(
        actual == GENERATED_ARTIFACT_PATHS | {"conversion_manifest.json"},
        "converted directory contains omitted or extra files",
    )


def _validate_conversion(
    converted: Path,
    contract_path: Path,
    contract_sha256: str,
    stack: Path | None = None,
) -> dict[str, Any]:
    manifest_path = converted / "conversion_manifest.json"
    manifest = _load_json(manifest_path, "conversion manifest")
    expected_fields = {
        "schema_version",
        "status",
        "performance_claimed",
        "contract_id",
        "contract",
        "native_dependencies",
        "canonical_directory",
        "canonical_artifacts",
        "oracle_binding",
        "operation_id",
        "semantic_region",
        "operation_facts",
        "input_quantize_projection",
        "quantization_facts",
        "native_projection",
        "emitted_call_artifact",
        "emitted_operation_helper_count",
        "payloads",
        "layout_mapping_rows",
        "layout_mapping_role_counts",
        "generated_artifacts",
    }
    _require(
        set(manifest) == expected_fields, "wrong conversion manifest fields"
    )
    _require(manifest.get("schema_version") == 1, "wrong conversion schema")
    _require(manifest.get("status") == "converted", "conversion did not pass")
    _require(
        manifest.get("performance_claimed") is False,
        "conversion makes a performance claim",
    )
    _require(manifest.get("contract_id") == CONTRACT_ID, "wrong contract ID")
    _require(manifest.get("operation_id") == OPERATION_ID, "wrong operation ID")
    _require(
        contract_path == INJECTION_CONTRACT_PATH,
        "runner received the wrong injection contract path",
    )
    _require(
        contract_sha256 == INJECTION_CONTRACT_SHA256,
        "runner received the wrong injection contract hash",
    )
    _require(
        manifest.get("contract")
        == {
            "path": str(INJECTION_CONTRACT_PATH),
            "sha256": INJECTION_CONTRACT_SHA256,
        },
        "conversion contract identity mismatch",
    )
    _validate_native_dependencies(manifest, stack)
    _validate_oracle_binding(converted, manifest)
    _require(
        manifest.get("semantic_region")
        == {
            "serialized_targets": [
                "quantize",
                "conv2d",
                "dequantize",
                "quantize",
            ],
            "semantic_members": [
                "quantized_ops::quantize",
                "aten::conv2d",
                "quantized_ops::dequantize",
                "quantized_ops::quantize",
            ],
            "fused_name": manifest["operation_facts"].get("fused_name"),
            "terminal_target": "quantized_ops::dequantize",
            "terminal_name": manifest["operation_facts"].get("terminal_name"),
        },
        "conversion semantic region differs",
    )
    input_projection = manifest.get("input_quantize_projection")
    _require(
        isinstance(input_projection, dict)
        and input_projection.get("runtime_float_input_supported") is False
        and input_projection.get("projection_time_only") is True
        and input_projection.get("verified_static_input_only") is True
        and input_projection.get("packing_rounding_domain") == "float32"
        and input_projection.get("shared_qmap_rounding_domain") == "bfloat16"
        and input_projection.get("serialized_rounding") == "nearest"
        and input_projection.get("saturation_bounds") == [-128.0, 127.0]
        and input_projection.get("input_zero_point") == 0,
        "input quantize projection is not fail closed",
    )
    _require(
        manifest.get("emitted_operation_helper_count") == 1,
        "conversion did not emit exactly one operation helper",
    )
    _require(
        manifest.get("layout_mapping_rows") == LAYOUT_ROW_COUNT,
        "conversion layout map is incomplete",
    )
    _require(
        manifest.get("layout_mapping_role_counts") == LAYOUT_ROLE_COUNTS,
        "conversion layout role counts differ",
    )
    projection = manifest.get("native_projection")
    _require(isinstance(projection, dict), "native projection is missing")
    _require(
        projection.get("helper") == "tiled_conv_auto",
        "conversion selected the wrong native helper",
    )
    _require(
        projection.get("schedule_authority") == "endpoint"
        and projection.get("schedule_fidelity") == "helper_selected",
        "conversion claims compiler-owned native scheduling",
    )
    _require(
        projection.get("compiler_schedule_consumed") is False,
        "converter consumed a compiler schedule",
    )
    _require(
        projection.get("helper_derived_fields")
        == {"in_stride": 16, "weight_stride": 16, "out_stride": 16},
        "helper-derived stride evidence differs",
    )
    quantization = manifest["quantization_facts"]
    _require(
        set(quantization)
        == {
            "qparams",
            "accumulator_scale",
            "helper_acc_scale",
            "scale_representation",
            "zero_points_explicit_in_oracle",
            "zero_points_implicit_zero_in_shared_ops",
            "general_bit_exact_equivalence",
            "acceptance_scope",
        },
        "wrong quantization fact fields",
    )
    _require(
        quantization["general_bit_exact_equivalence"] is False,
        "conversion makes a general bit-exact-equivalence claim",
    )
    _require(
        quantization["acceptance_scope"]
        == "fixed-static-vector-full-output-atol-1",
        "conversion claims the wrong arithmetic acceptance scope",
    )
    qparams = _validate_qparams(quantization["qparams"], "conversion qparams")
    expected_accumulator = np.float32(
        np.float32(qparams["input_scale"]) * np.float32(qparams["weight_scale"])
    )
    _require(
        np.float32(quantization["accumulator_scale"]).tobytes()
        == expected_accumulator.tobytes(),
        "conversion accumulator scale differs",
    )
    expected_helper_scale = np.float32(
        expected_accumulator / np.float32(qparams["output_scale"])
    )
    _require(
        np.float32(quantization["helper_acc_scale"]).tobytes()
        == expected_helper_scale.tobytes(),
        "helper accumulator scale differs",
    )
    fields = projection.get("fields")
    _require(isinstance(fields, dict), "native projection fields are missing")
    acc_scale = fields.get("acc_scale")
    _require(
        isinstance(acc_scale, dict)
        and set(acc_scale) == {"value", "provenance"}
        and isinstance(acc_scale["value"], str)
        and acc_scale["value"].endswith("f"),
        "native accumulator scale literal is invalid",
    )
    try:
        literal_value = float.fromhex(acc_scale["value"][:-1])
    except ValueError as error:
        raise RunError("native accumulator scale literal is invalid") from error
    _require(
        np.float32(literal_value).tobytes() == expected_helper_scale.tobytes(),
        "native accumulator scale literal differs",
    )
    _require(
        manifest.get("emitted_call_artifact") == "emitted-calls.json",
        "wrong emitted-call artifact path",
    )
    calls = _load_json(converted / "emitted-calls.json", "emitted calls")
    _require(
        set(calls)
        == {
            "schema_version",
            "semantic_source",
            "operation_id",
            "emitted_operation_helper_count",
            "emitted_backend_calls",
            "lifecycle_calls",
        },
        "wrong emitted-call evidence fields",
    )
    _require(calls["schema_version"] == 1, "wrong emitted-call schema")
    _require(calls["operation_id"] == OPERATION_ID, "wrong emitted operation")
    _require(
        calls.get("emitted_operation_helper_count") == 1,
        "emitted call count is not one",
    )
    canonical_artifacts = manifest["canonical_artifacts"]
    _require(
        isinstance(canonical_artifacts, dict)
        and isinstance(canonical_artifacts.get("model.txt"), dict),
        "canonical model evidence is missing",
    )
    model_sha256 = _validate_sha256(
        canonical_artifacts["model.txt"].get("sha256"),
        "canonical model hash",
    )
    _require(
        calls["semantic_source"]
        == {"path": "model.txt", "sha256": model_sha256},
        "emitted calls bind the wrong semantic source",
    )
    operation_name = manifest["operation_facts"].get("fused_name")
    _require(
        isinstance(operation_name, str) and bool(operation_name),
        "canonical operation name is missing",
    )
    backend_calls = calls.get("emitted_backend_calls")
    _require(
        backend_calls
        == [
            {
                "ordinal": 0,
                "helper": "tiled_conv_auto",
                "wrapper": WRAPPER_SYMBOL,
                "model_operation_name": operation_name,
                "fused_member_names": manifest["operation_facts"].get(
                    "member_names"
                ),
            }
        ],
        "wrong emitted operation helper",
    )
    _require(
        calls.get("lifecycle_calls")
        == [
            {"ordinal": 0, "helper": "gemmini_flush", "argument": 0},
            {"ordinal": 1, "helper": "gemmini_fence"},
        ],
        "wrong emitted lifecycle calls",
    )
    source = (converted / "runner.c").read_text()
    _require(
        source.count("tiled_conv_auto(") == 1,
        "runner source must contain one helper invocation",
    )
    _require(
        "tiled_conv_stride_auto(" not in source,
        "runner calls the internal stride helper",
    )
    _require("voyager_gemmini_operation" in source, "runner wrapper is missing")
    _require(
        "gemmini::conv2d_int8" not in source and "conv2d_int8" not in source,
        "runner contains obsolete target-private operation text",
    )
    _validate_generated_artifacts(converted, manifest)
    _validate_layout_map(converted)
    return manifest


def _parse_symbols(text: str) -> dict[str, dict[str, int | str | None]]:
    symbols = {}
    for line in text.splitlines():
        match = NM_RE.fullmatch(line.strip())
        if match:
            size = match.group(2)
            symbols[match.group(4)] = {
                "address": int(match.group(1), 16),
                "size": None if size is None else int(size, 16),
                "type": match.group(3),
            }
    return symbols


def _static_custom_sites(objdump_text: str) -> dict[int, int]:
    sites = {}
    for line in objdump_text.splitlines():
        match = OBJDUMP_WORD_RE.match(line)
        if not match:
            continue
        pc = int(match.group(1), 16)
        word = int(match.group(2), 16)
        if word & 0x7F == CUSTOM_3_OPCODE:
            sites[pc] = word
    return sites


def validate_elf(
    readelf_text: str,
    nm_text: str,
    unresolved_text: str,
    objdump_text: str,
    map_text: str,
) -> dict[str, Any]:
    _require(
        "Class:                             ELF64" in readelf_text, "not ELF64"
    )
    _require(
        "Data:                              2's complement, little endian"
        in readelf_text,
        "ELF is not little-endian",
    )
    _require(
        "Machine:                           RISC-V" in readelf_text,
        "not RISC-V",
    )
    _require(
        "Type:                              EXEC" in readelf_text,
        "not executable",
    )
    _require(" INTERP " not in readelf_text, "ELF has an interpreter")
    _require(not unresolved_text.strip(), "ELF contains unresolved symbols")
    symbols = _parse_symbols(nm_text)
    required_sizes = {
        "voyager_input": 256,
        "voyager_weights": 2304,
        "voyager_expected": 256,
        "voyager_output": 256,
    }
    lifecycle_types = {
        "_start": "T",
        "main": "T",
        "tohost": "D",
        "fromhost": "D",
        WRAPPER_SYMBOL: "T",
    }
    for name, symbol_type in lifecycle_types.items():
        _require(name in symbols, f"ELF lacks required symbol {name}")
        _require(
            str(symbols[name]["type"]).upper() == symbol_type,
            f"ELF symbol has wrong type: {name}",
        )
    intervals = []
    for name, size in required_sizes.items():
        _require(name in symbols, f"ELF lacks payload symbol {name}")
        symbol = symbols[name]
        _require(symbol["size"] == size, f"wrong symbol size: {name}")
        address = int(symbol["address"])
        _require(address % 16 == 0, f"symbol is not row-aligned: {name}")
        _require(name in map_text, f"link map lacks symbol {name}")
        intervals.append((address, address + size, name))
    intervals.sort()
    _require(
        all(
            left[1] <= right[0] for left, right in zip(intervals, intervals[1:])
        ),
        "ELF payload symbols overlap",
    )
    wrapper = symbols[WRAPPER_SYMBOL]
    wrapper_start = int(wrapper["address"])
    _require(
        isinstance(wrapper["size"], int) and wrapper["size"] > 0,
        "operation wrapper lacks a nonzero size",
    )
    wrapper_size = int(wrapper["size"])
    wrapper_end = wrapper_start + wrapper_size
    sites = _static_custom_sites(objdump_text)
    inside = {
        pc: word
        for pc, word in sites.items()
        if wrapper_start <= pc < wrapper_end
    }
    outside = {
        pc: word
        for pc, word in sites.items()
        if not wrapper_start <= pc < wrapper_end
    }
    _require(inside, "operation wrapper contains no CUSTOM_3 instructions")
    _require(outside, "ELF lacks lifecycle CUSTOM_3 instruction")
    _require(
        re.search(r"\bfence\b", objdump_text) is not None, "ELF lacks fence"
    )
    return {
        "format": "ELF64 little-endian static RISC-V executable",
        "symbols": {
            name: symbols[name] for name in (*required_sizes, *lifecycle_types)
        },
        "payload_intervals": [
            {"start": start, "end_exclusive": end, "symbol": name}
            for start, end, name in intervals
        ],
        "wrapper": {
            "symbol": WRAPPER_SYMBOL,
            "start": wrapper_start,
            "end_exclusive": wrapper_end,
            "size": wrapper_size,
        },
        "static_custom_3_sites": [
            {
                "pc": pc,
                "word": word,
                "inside_operation_wrapper": pc in inside,
            }
            for pc, word in sorted(sites.items())
        ],
        "static_custom_3_site_count": len(sites),
        "static_operation_custom_3_site_count": len(inside),
        "static_lifecycle_custom_3_site_count": len(outside),
        "fence_present": True,
    }


def parse_spike_trace(
    path: Path,
    wrapper_start: int,
    wrapper_end: int,
    static_sites: dict[int, int],
) -> dict[str, Any]:
    """Stream a Spike ``-l`` trace and correlate the emitted ELF sites."""
    _require(path.is_file(), f"Spike execution trace is missing: {path}")
    digest = hashlib.sha256()
    line_count = 0
    byte_count = 0
    instruction_record_count = 0
    wrapper_entries = []
    rocc_records = []
    last_relevant: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            digest.update(raw_line)
            line_count += 1
            byte_count += len(raw_line)
            line = raw_line.rstrip(b"\r\n")
            instruction = TRACE_INSTRUCTION_RE.fullmatch(line)
            if instruction:
                instruction_record_count += 1
                core = int(instruction.group("core"))
                pc = int(instruction.group("pc"), 16)
                word = int(instruction.group("word"), 16)
                record = {
                    "trace_line": line_number,
                    "core": core,
                    "pc": pc,
                    "word": word,
                    "dynamic_count": 1,
                }
                last_relevant = []
                if pc == wrapper_start:
                    wrapper_entries.append(record.copy())
                    last_relevant.append(wrapper_entries[-1])
                if word & 0x7F == CUSTOM_3_OPCODE:
                    rocc_record = {
                        **record,
                        "opcode": CUSTOM_3_OPCODE,
                        "funct": (word >> 25) & 0x7F,
                        "inside_operation_wrapper": wrapper_start
                        <= pc
                        < wrapper_end,
                    }
                    rocc_records.append(rocc_record)
                    last_relevant.append(rocc_records[-1])
                continue
            repeated = TRACE_RLE_RE.fullmatch(line)
            if repeated:
                for record in last_relevant:
                    _require(
                        int(repeated.group("core")) == record["core"],
                        "Spike RLE core differs from preceding instruction",
                    )
                    record["dynamic_count"] = int(repeated.group("count"))

    _require(instruction_record_count > 0, "Spike trace has no instructions")
    wrapper_count = sum(record["dynamic_count"] for record in wrapper_entries)
    _require(
        wrapper_count == 1, f"operation wrapper executed {wrapper_count} times"
    )
    runtime_sites = {record["pc"]: record["word"] for record in rocc_records}
    _require(
        runtime_sites == static_sites,
        "runtime CUSTOM_3 sites differ from emitted ELF sites",
    )
    operation_records = [
        record for record in rocc_records if record["inside_operation_wrapper"]
    ]
    lifecycle_records = [
        record
        for record in rocc_records
        if not record["inside_operation_wrapper"]
    ]
    _require(
        operation_records, "runtime trace lacks operation RoCC instructions"
    )
    _require(
        lifecycle_records, "runtime trace lacks lifecycle RoCC instruction"
    )
    return {
        "schema_version": 1,
        "trace": {
            "path": path.name,
            "bytes": byte_count,
            "lines": line_count,
            "sha256": digest.hexdigest(),
            "instruction_record_count": instruction_record_count,
        },
        "helper_call_boundary": WRAPPER_SYMBOL,
        "actual_accelerator_call_count": wrapper_count,
        "wrapper_entries": wrapper_entries,
        "runtime_rocc_instruction_record_count": len(rocc_records),
        "runtime_rocc_dynamic_instruction_count": sum(
            record["dynamic_count"] for record in rocc_records
        ),
        "runtime_operation_rocc_instruction_record_count": len(
            operation_records
        ),
        "runtime_operation_rocc_dynamic_instruction_count": sum(
            record["dynamic_count"] for record in operation_records
        ),
        "runtime_lifecycle_rocc_instruction_record_count": len(
            lifecycle_records
        ),
        "runtime_lifecycle_rocc_dynamic_instruction_count": sum(
            record["dynamic_count"] for record in lifecycle_records
        ),
        "rocc_records": rocc_records,
        "static_runtime_site_match": True,
    }


def parse_output(
    text: str, expected_count: int = OUTPUT_COUNT
) -> tuple[np.ndarray, dict]:
    values = []
    summary = None
    records_started = False
    footer_pending = False
    footer_complete = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        if footer_complete:
            raise RunError(
                "unexpected program text after extension footer at line "
                f"{line_number}: {line!r}"
            )
        if footer_pending:
            _require(
                line == EXTENSION_FOOTER_DIM,
                "Gemmini extension footer has the wrong pinned DIM line: "
                f"{line!r}",
            )
            footer_pending = False
            footer_complete = True
            continue
        if line.startswith("OUTPUT"):
            _require(summary is None, "output record appears after summary")
            records_started = True
            match = OUTPUT_RE.fullmatch(line)
            _require(match is not None, f"malformed output record: {line!r}")
            index = int(match.group(1))
            value = int(match.group(2))
            _require(
                len(values) < expected_count,
                "program emitted extra output records",
            )
            _require(index == len(values), f"out-of-order output index {index}")
            _require(-128 <= value <= 127, f"output {index} is outside INT8")
            values.append(value)
        elif line.startswith("SUMMARY"):
            _require(summary is None, "duplicate summary")
            _require(
                len(values) == expected_count,
                "program summary precedes complete output",
            )
            match = SUMMARY_RE.fullmatch(line)
            _require(match is not None, f"malformed summary: {line!r}")
            summary = {
                "element_count": int(match.group(1)),
                "exact_mismatch_count": int(match.group(2)),
                "tolerance_mismatch_count": int(match.group(3)),
                "first_mismatch": (
                    None if match.group(4) == "none" else int(match.group(4))
                ),
                "maximum_absolute_error": int(match.group(5)),
                "squared_error": int(match.group(6)),
            }
        elif line == EXTENSION_FOOTER_HEADER:
            _require(
                summary is not None,
                "Gemmini extension footer precedes complete summary",
            )
            footer_pending = True
        elif records_started:
            position = (
                "after summary" if summary is not None else "among records"
            )
            raise RunError(
                f"unexpected program text {position} at line {line_number}: "
                f"{line!r}"
            )
    _require(not footer_pending, "Gemmini extension footer is incomplete")
    _require(len(values) == expected_count, "output record count is incomplete")
    _require(summary is not None, "program summary is missing")
    _require(summary["element_count"] == expected_count, "wrong summary count")
    return np.asarray(values, dtype=np.int8), summary


def compare_output(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    _require(
        actual.shape == expected.shape == (OUTPUT_COUNT,), "wrong output extent"
    )
    difference = actual.astype(np.int16) - expected.astype(np.int16)
    exact_mismatches = np.flatnonzero(difference)
    tolerance_mismatches = np.flatnonzero(np.abs(difference) > 1)
    physical_actual = actual.reshape(1, 4, 4, 16)
    physical_expected = expected.reshape(1, 4, 4, 16)
    logical_actual = physical_actual.transpose(0, 3, 1, 2)
    logical_expected = physical_expected.transpose(0, 3, 1, 2)
    return {
        "element_count": OUTPUT_COUNT,
        "exact_match_count": int(OUTPUT_COUNT - exact_mismatches.size),
        "exact_mismatch_count": int(exact_mismatches.size),
        "tolerance_mismatch_count": int(tolerance_mismatches.size),
        "first_mismatch": (
            int(exact_mismatches[0]) if exact_mismatches.size else None
        ),
        "maximum_absolute_error": int(np.max(np.abs(difference))),
        "squared_error": int(np.sum(difference.astype(np.int64) ** 2)),
        "physical_equal": bool(
            np.array_equal(physical_actual, physical_expected)
        ),
        "logical_equal": bool(np.array_equal(logical_actual, logical_expected)),
        "within_tolerance": bool(tolerance_mismatches.size == 0),
        "absolute_tolerance": 1,
        "relative_tolerance": 0,
        "exact_match_required": False,
        "logical_shape": [1, 16, 4, 4],
        "physical_shape": [1, 4, 4, 16],
    }


def execute(
    converted_dir: Path | str,
    output_dir: Path | str,
    stack_root: Path | str,
    stack_manifest: Path | str,
    contract_path: Path | str,
    contract_sha256: str,
) -> dict[str, Any]:
    """Run the frozen build, trace, and complete tolerance comparison."""
    converted = Path(converted_dir).resolve()
    output = Path(output_dir).resolve()
    stack = Path(stack_root).resolve()
    manifest_path = Path(stack_manifest).resolve()
    contract = Path(contract_path).resolve()
    _require(converted.is_dir(), f"converted directory is missing: {converted}")
    _require(not output.exists(), f"execution output must be fresh: {output}")
    _require(
        contract == INJECTION_CONTRACT_PATH,
        "injection contract path is not the pinned T5 contract",
    )
    _require(
        contract_sha256 == INJECTION_CONTRACT_SHA256,
        "injection contract hash is not pinned",
    )
    _require(contract.is_file(), f"injection contract is missing: {contract}")
    _require(
        sha256_file(contract) == INJECTION_CONTRACT_SHA256,
        "contract hash mismatch",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    records: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "operation_id": OPERATION_ID,
        "converted_directory": str(converted),
        "stack_root": str(stack),
        "stack_manifest": str(manifest_path),
        "injection_contract": {
            "path": str(contract),
            "sha256": contract_sha256,
        },
        "emitted_backend_calls": [{"ordinal": 0, "helper": "tiled_conv_auto"}],
        "emitted_operation_helper_count": 1,
        "performance_claimed": False,
        "stage_runtimes_seconds": {},
    }
    started_overall = time.monotonic()
    try:
        stack, tools, frozen = validate_stack_manifest(manifest_path, stack)
        conversion = _validate_conversion(
            converted, contract, contract_sha256, stack
        )
        _write_json(output / "frozen-stack.json", frozen)
        rocc = stack / ROCC_ROOT
        common = stack / COMMON_ROOT
        compile_flags = [
            *COMPILE_FLAGS,
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

        started = time.monotonic()
        completed = _run_command(
            [
                str(gcc),
                *compile_flags,
                "-c",
                str(converted / "runner.c"),
                "-o",
                str(runner_object),
            ],
            cwd=output,
            stdout_path=output / "compile.stdout",
            stderr_path=output / "compile.stderr",
            records=records,
        )
        result["stage_runtimes_seconds"]["compile"] = time.monotonic() - started
        _require(completed.returncode == 0, "runner compilation failed")
        _require(runner_object.is_file(), "runner object is missing")

        for stem, command in (
            (
                "runner.readelf",
                [str(readelf), "-hW", "-SW", "-sW", "-rW", str(runner_object)],
            ),
            ("runner.objdump", [str(objdump), "-dr", str(runner_object)]),
        ):
            completed = _run_command(
                command,
                cwd=output,
                stdout_path=output / f"{stem}.txt",
                stderr_path=output / f"{stem}.stderr",
                records=records,
            )
            _require(
                completed.returncode == 0, f"object inspection failed: {stem}"
            )

        started = time.monotonic()
        completed = _run_command(
            [
                str(gcc),
                *compile_flags,
                "-nostdlib",
                "-nostartfiles",
                "-static",
                "-T",
                str(common / "test.ld"),
                f"-Wl,-Map,{link_map}",
                str(runner_object),
                str(common / "syscalls.c"),
                str(common / "crt.S"),
                "-lm",
                "-lgcc",
                "-o",
                str(elf),
            ],
            cwd=output,
            stdout_path=output / "link.stdout",
            stderr_path=output / "link.stderr",
            records=records,
        )
        result["stage_runtimes_seconds"]["link"] = time.monotonic() - started
        _require(completed.returncode == 0, "ELF link failed")
        _require(
            elf.is_file() and link_map.is_file(), "link outputs are missing"
        )

        inspections = (
            (
                "elf.readelf",
                [str(readelf), "-hW", "-lW", "-SW", "-sW", "-rW", str(elf)],
            ),
            ("elf.nm", [str(nm), "-n", "-S", "--defined-only", str(elf)]),
            ("elf.unresolved", [str(nm), "-u", str(elf)]),
            ("elf.objdump", [str(objdump), "-dr", str(elf)]),
        )
        for stem, command in inspections:
            completed = _run_command(
                command,
                cwd=output,
                stdout_path=output / f"{stem}.txt",
                stderr_path=output / f"{stem}.stderr",
                records=records,
            )
            _require(
                completed.returncode == 0, f"ELF inspection failed: {stem}"
            )
        elf_evidence = validate_elf(
            (output / "elf.readelf.txt").read_text(),
            (output / "elf.nm.txt").read_text(),
            (output / "elf.unresolved.txt").read_text(),
            (output / "elf.objdump.txt").read_text(),
            link_map.read_text(),
        )
        _write_json(output / "elf-validation.json", elf_evidence)

        trace_path = output / "spike-execution.log"
        run_env = os.environ.copy()
        run_env["PATH"] = os.pathsep.join(
            [
                str(stack / "gemmini-stack/toolchain-env/bin"),
                str(stack / "gemmini-stack/toolchain-env/riscv-tools/bin"),
                "/usr/bin",
                "/bin",
            ]
        )
        run_env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [
                str(stack / "gemmini-stack/runtime-prefix/lib"),
                str(stack / "gemmini-stack/toolchain-env/lib"),
            ]
        )
        started = time.monotonic()
        completed = _run_command(
            [
                "/usr/bin/timeout",
                "120",
                str(tools["spike"]),
                "-l",
                f"--log={trace_path}",
                "--extension=gemmini",
                str(elf),
            ],
            cwd=output,
            stdout_path=output / "spike.stdout",
            stderr_path=output / "spike.stderr",
            records=records,
            env=run_env,
            timeout=135,
        )
        result["stage_runtimes_seconds"]["spike"] = time.monotonic() - started
        _require(
            completed.returncode == 0,
            f"Spike/program returned {completed.returncode}",
        )
        static_sites = {
            item["pc"]: item["word"]
            for item in elf_evidence["static_custom_3_sites"]
        }
        wrapper = elf_evidence["wrapper"]
        trace_evidence = parse_spike_trace(
            trace_path,
            wrapper["start"],
            wrapper["end_exclusive"],
            static_sites,
        )
        _write_json(output / "rocc-trace.json", trace_evidence)
        actual, in_program = parse_output(completed.stdout)
        expected_path = converted / "payloads/output_oracle.bin"
        expected = np.frombuffer(
            expected_path.read_bytes(), dtype=np.int8
        ).copy()
        comparison = compare_output(actual, expected)
        _require(
            in_program["exact_mismatch_count"]
            == comparison["exact_mismatch_count"],
            "program and host exact mismatch counts differ",
        )
        _require(
            in_program["tolerance_mismatch_count"]
            == comparison["tolerance_mismatch_count"],
            "program and host tolerance mismatch counts differ",
        )
        _require(
            in_program["first_mismatch"] == comparison["first_mismatch"],
            "program and host first mismatch differ",
        )
        _require(
            in_program["maximum_absolute_error"]
            == comparison["maximum_absolute_error"],
            "program and host maximum error differ",
        )
        _require(
            in_program["squared_error"] == comparison["squared_error"],
            "program and host squared error differ",
        )
        _require(
            comparison["tolerance_mismatch_count"] == 0,
            "host comparison exceeded tolerance",
        )
        _write_json(
            output / "readback.json",
            {
                "record_count": OUTPUT_COUNT,
                "ordered_indices": list(range(OUTPUT_COUNT)),
                "values": [int(value) for value in actual],
                "in_program_summary": in_program,
                "stdout_sha256": sha256_file(output / "spike.stdout"),
            },
        )
        _write_json(output / "comparison.json", comparison)
        result.update(
            {
                "status": "passed",
                "conversion_manifest_sha256": sha256_file(
                    converted / "conversion_manifest.json"
                ),
                "model_artifact_sha256": conversion["canonical_artifacts"][
                    "model.txt"
                ]["sha256"],
                "elf_sha256": sha256_file(elf),
                "emitted_call_artifact": "emitted-calls.json",
                "actual_execution_trace": "rocc-trace.json",
                "actual_accelerator_call_count": 1,
                "runtime_rocc_instruction_record_count": trace_evidence[
                    "runtime_rocc_instruction_record_count"
                ],
                "runtime_rocc_dynamic_instruction_count": trace_evidence[
                    "runtime_rocc_dynamic_instruction_count"
                ],
                "semantic_accounting_complete": True,
                "all_emitted_calls_executed": True,
                "implicit_cpu_fallback": False,
                "no_fallback_proof": [
                    "generated runner has one operation wrapper/helper call",
                    "wrapper entry was observed exactly once in Spike trace",
                    "all emitted CUSTOM_3 sites were observed at runtime",
                    "CPU code only initializes, launches, reads, and compares",
                    "all 256 target-produced elements passed atol=1, rtol=0",
                ],
                "exact_output_observed": comparison["logical_equal"],
                "exact_output_match_count": comparison["exact_match_count"],
                "exact_output_mismatch_count": comparison[
                    "exact_mismatch_count"
                ],
                "exact_output_required": False,
                "complete_output_comparison": comparison,
            }
        )
    except BaseException as error:
        result["status"] = "failed"
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
    finally:
        result["overall_wall_seconds"] = time.monotonic() - started_overall
        result["command_count"] = len(records)
        _write_json(output / "commands.json", records)
        artifacts = []
        for path in sorted(output.iterdir()):
            if path.is_file() and path.name not in {
                "result.json",
                "artifacts.json",
            }:
                artifacts.append(
                    {
                        "path": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        _write_json(output / "artifacts.json", artifacts)
        _write_json(output / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stack-root", required=True, type=Path)
    parser.add_argument("--stack-manifest", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--contract-sha256", required=True)
    arguments = parser.parse_args(argv)
    try:
        result = execute(
            arguments.converted_dir,
            arguments.output_dir,
            arguments.stack_root,
            arguments.stack_manifest,
            arguments.contract,
            arguments.contract_sha256,
        )
    except RunError as error:
        print(
            f"execution failed before evidence initialization: {error}",
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

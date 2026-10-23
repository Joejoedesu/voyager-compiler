# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Serialization of auxiliary compiler output artifacts."""

import json
from collections.abc import Mapping
from pathlib import Path

from voyager_compiler.codegen.hardware import HardwareConfig

HARDWARE_CONFIG_SCHEMA_VERSION = 1
SUPPLEMENTAL_ARTIFACTS_SCHEMA_VERSION = 1
_SUPPLEMENTAL_ARTIFACTS_META = "supplemental_compile_artifacts"


def set_supplemental_artifact(
    model,
    name: str,
    payload: Mapping[str, object],
) -> None:
    """Set one target-owned payload in the supplemental manifest."""
    if not isinstance(name, str) or not name:
        raise ValueError("Supplemental artifact name must be nonempty")
    schema_version = payload.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version <= 0
    ):
        raise ValueError(
            "Supplemental artifact requires a positive schema_version"
        )
    artifacts = model.meta.setdefault(_SUPPLEMENTAL_ARTIFACTS_META, {})
    artifacts[name] = dict(payload)


def discard_supplemental_artifact(model, name: str) -> None:
    """Discard one target-owned supplemental payload if present."""
    artifacts = model.meta.get(_SUPPLEMENTAL_ARTIFACTS_META)
    if artifacts is None:
        return
    artifacts.pop(name, None)
    if not artifacts:
        model.meta.pop(_SUPPLEMENTAL_ARTIFACTS_META, None)


def prepare_output_directory(output_dir) -> Path:
    """Create a fresh output directory or reject stale/user-owned contents."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"Compiler output directory is not empty: {path}")
    return path


def _compute_unit_to_dict(unit) -> dict:
    return {
        "name": unit.name,
        "unroll_dimensions": [
            {
                "name": dimension.name,
                "extent": dimension.extent,
                "dtype_extents": [
                    {
                        "input_bits": bits,
                        "extent": extent,
                    }
                    for bits, extent in dimension.dtype_extents
                ],
            }
            for dimension in unit.unroll_dimensions
        ],
        "accepted_input_bits": sorted(unit.accepted_input_bits),
        "accepted_operations": sorted(unit.accepted_operations),
        "input_word_bits": unit.input_word_bits,
        "scalar_bits": unit.scalar_bits,
    }


def _memory_instance_to_dict(memory) -> dict:
    return {
        "name": memory.name,
        "bank_count": memory.bank_count,
        "depth_per_bank": memory.depth_per_bank,
        "bits_per_bank_entry": memory.bits_per_bank_entry,
        "accepted_accesses": [
            {
                "role": role.value,
                "compute_unit": compute_unit,
            }
            for role, compute_unit in sorted(
                memory.accepted_accesses,
                key=lambda access: (
                    access[0].value,
                    access[1] or "",
                ),
            )
        ],
        "access_cost": memory.access_cost,
        "logical_capacity": memory.logical_capacity,
        "alignment_bytes": memory.alignment_bytes,
        "access_granularity_bits": memory.access_granularity_bits,
        "read_bandwidth": memory.read_bandwidth,
        "write_bandwidth": memory.write_bandwidth,
        "double_buffered": memory.double_buffered,
    }


def _memory_level_to_dict(level) -> dict:
    return {
        "name": level.name,
        "instances": [
            _memory_instance_to_dict(memory) for memory in level.instances
        ],
        "input_memory": level.input_memory,
        "output_memory": level.output_memory,
        "weight_memory": level.weight_memory,
    }


def serialize_hardware_config(hardware: HardwareConfig) -> dict:
    """Serialize the exact typed compiler hardware configuration."""
    payload = {
        "schema_version": HARDWARE_CONFIG_SCHEMA_VERSION,
        "config_type": type(hardware).__name__,
        "target": hardware.target,
        "compute_unit_bindings": dict(
            sorted(hardware.compute_unit_bindings().items())
        ),
        "compute_units": [
            _compute_unit_to_dict(unit) for unit in hardware.compute_units
        ],
        "dtype_modes": [
            {
                "operand_bits": mode.operand_bits,
                "banks_per_value": mode.banks_per_value,
                "output_channel_alignment": (mode.output_channel_alignment),
            }
            for mode in hardware.dtype_modes
        ],
        "local_memory": _memory_level_to_dict(hardware.local_memory),
        "global_memory": _memory_level_to_dict(hardware.global_memory),
        "external_memory": (
            _memory_level_to_dict(hardware.external_memory)
            if hardware.external_memory is not None
            else None
        ),
        "output_collector": {
            "name": hardware.output_collector.name,
            "depth_words": hardware.output_collector.depth_words,
            "lanes_per_word": (hardware.output_collector.lanes_per_word),
            "accumulator_bits": (hardware.output_collector.accumulator_bits),
        },
        "hierarchy_bandwidths": [
            {
                "name": route.name,
                "source": route.source,
                "destination": route.destination,
                "bandwidth_bits_per_cycle": (route.bandwidth_bits_per_cycle),
                "roles": sorted(role.value for role in route.roles),
                "shared_group": route.shared_group,
            }
            for route in hardware.hierarchy_bandwidths
        ],
        "mapping_num_banks": hardware.mapping_num_banks,
        "mapping_bank_width": hardware.mapping_bank_width,
        "technology_nm": hardware.technology_nm,
        "pad_depthwise_channels": hardware.pad_depthwise_channels,
        "partial_matrix_input_channels": sorted(
            hardware.partial_matrix_input_channels
        ),
    }
    return payload


def dump_hardware_config(hardware, output_dir) -> None:
    """Write hardware_config.json and remove superseded plan files."""
    output_dir = Path(output_dir)
    for stale_name in ("target_plan.json", "target_schedules.json"):
        (output_dir / stale_name).unlink(missing_ok=True)

    path = output_dir / "hardware_config.json"
    if hardware is None:
        path.unlink(missing_ok=True)
        return
    payload = serialize_hardware_config(hardware)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def dump_performance_estimates(model, output_dir) -> None:
    """Write scheduler-reported cycles and scalar memory accesses."""
    path = Path(output_dir) / "performance_estimates.json"
    layers = {}
    for node in model.graph.nodes:
        result = node.meta.get("local_schedule")
        if result is None:
            continue
        if result.runtime is None and not result.access_counts:
            continue

        record = {"backend": result.backend}
        if result.runtime is not None:
            record["execution_cycles"] = float(result.runtime)
        if result.access_counts:
            record["level_access_counts"] = [
                {
                    "input": int(level.input),
                    "output": int(level.output),
                    "weight": int(level.weight),
                }
                for level in result.access_counts
            ]
        layers[node.name] = record

    if not layers:
        path.unlink(missing_ok=True)
        return
    payload = {
        "schema_version": 1,
        "count_unit": "scalar_accesses",
        "level_order": "inner_to_outer",
        "runtime_unit": "cycles",
        "layers": dict(sorted(layers.items())),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def dump_storage_plan(model, output_dir) -> None:
    """Write exact logical and physical tensor identities for debugging."""
    path = Path(output_dir) / "storage_plan.json"
    plan = model.meta.get("storage_plan")
    if plan is None:
        path.unlink(missing_ok=True)
        return
    payload = {
        "schema_version": 1,
        **plan.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def dump_supplemental_artifacts(model, output_dir) -> None:
    """Write target-owned payloads under one versioned shared manifest."""
    path = Path(output_dir) / "supplemental_artifacts.json"
    artifacts = model.meta.get(_SUPPLEMENTAL_ARTIFACTS_META)
    if not artifacts:
        path.unlink(missing_ok=True)
        return
    payload = {
        "schema_version": SUPPLEMENTAL_ARTIFACTS_SCHEMA_VERSION,
        "artifacts": dict(sorted(artifacts.items())),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dump_compile_artifacts(
    model,
    hardware,
    output_dir,
) -> None:
    """Finalize auxiliary artifacts for one terminal compile path."""
    output_dir = Path(output_dir)
    dump_hardware_config(hardware, output_dir)
    dump_performance_estimates(model, output_dir)
    dump_storage_plan(model, output_dir)
    dump_supplemental_artifacts(model, output_dir)


__all__ = [
    "HARDWARE_CONFIG_SCHEMA_VERSION",
    "SUPPLEMENTAL_ARTIFACTS_SCHEMA_VERSION",
    "discard_supplemental_artifact",
    "dump_compile_artifacts",
    "dump_hardware_config",
    "dump_performance_estimates",
    "dump_storage_plan",
    "dump_supplemental_artifacts",
    "prepare_output_directory",
    "set_supplemental_artifact",
    "serialize_hardware_config",
]

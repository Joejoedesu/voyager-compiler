"""Canonical bundle projections with stable identities and payload checksums."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path


def to_data(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: to_data(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): to_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_data(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (to_data(v) for v in value),
            key=lambda v: json.dumps(v, sort_keys=True),
        )
    if hasattr(value, "loop_blockings"):
        return dict(
            provider="interstellar",
            loops=["FX", "FY", "OX", "OY", "OC", "IC", "ON"],
            orders=to_data(value.loop_orders),
            blockings=to_data(value.loop_blockings),
            partitionings=to_data(value.loop_partitionings),
            parallel_dimensions=to_data(value.para_loop_dim),
        )
    # Dtypes are schema data. Arbitrary objects must never silently become reprs
    # containing process-specific addresses or lose their semantic fields.
    if type(value).__module__ == "torch" and type(value).__name__ == "dtype":
        return str(value)
    raise TypeError(f"no artifact encoding for {type(value).__name__}")


def hardware_digest(config):
    """Physical identity excludes software choices and fitted model parameters."""
    data = to_data(config)
    data.pop("execution", None)
    data.pop("profile", None)
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def content_digest(value):
    return hashlib.sha256(
        json.dumps(to_data(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def save_bundle(program, config, output_dir, provenance=None):
    program.assert_frozen()
    output = Path(output_dir)
    plan = to_data(program.plan)
    hardware = to_data(config)
    (output / "compilation_plan.json").write_text(
        json.dumps(plan, indent=2, allow_nan=False) + "\n"
    )
    (output / "hardware_config.json").write_text(
        json.dumps(hardware, indent=2, allow_nan=False) + "\n"
    )
    (output / "buffer_bindings.json").write_text(
        json.dumps(to_data(program.bindings), indent=2) + "\n"
    )
    invocations = {}
    for scope, module in program.graph.named_modules():
        if not hasattr(module, "graph"):
            continue
        for node in module.graph.nodes:
            fields = {
                key: to_data(node.meta[key])
                for key in (
                    "target_invoke",
                    "target_schedule",
                    "target_layout",
                    "physical_component",
                )
                if key in node.meta
            }
            if fields:
                invocations[f"{scope}/{node.name}"] = fields
    (output / "target_invocations.json").write_text(
        json.dumps(invocations, indent=2, allow_nan=False) + "\n"
    )
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "bundle.json":
            with path.open("rb") as stream:
                files[str(path.relative_to(output))] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    bundle = dict(
        schema_version=1,
        target=program.plan.target,
        graph_digest=program.frozen_digest,
        plan_digest=program.plan_digest,
        validation=to_data(program.validation),
        files=files,
        provenance=provenance or {},
    )
    (output / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    return bundle

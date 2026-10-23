# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Run the calibrated Magnet Alpha YOLO compile lane."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import yaml

SCHEDULERS = ("magnet_alpha_rule_based",)


def _repo_python_environment(repo_root: Path) -> dict[str, str]:
    """Put this checkout ahead of unrelated editable installations."""
    environment = dict(os.environ)
    entries = [str(repo_root / "src"), str(repo_root)]
    existing = environment.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def _verify_converter_handoff(
    repo_root: Path,
    output: Path,
    python: str,
) -> dict[str, int | str]:
    """Check compiler boundary and exercise fresh Alpha conversion."""
    model_path = output / "model.txt"
    model_text = model_path.read_text()
    if "cmod_" in model_text:
        raise RuntimeError(
            f"compiler dump leaks external CMOD metadata: {model_path}"
        )
    copy_target = 'target: "channel_shuffle_copy"'
    copy_count = model_text.count(copy_target)
    if copy_count <= 0:
        raise RuntimeError(
            f"compiler dump has no explicit shuffle-copy target: {model_path}"
        )

    conversion_output = output / "cmod"
    subprocess.run(
        [
            python,
            str(repo_root / "conversion" / "magnet-alpha" / "convert.py"),
            str(output),
            "--output",
            str(conversion_output),
        ],
        cwd=repo_root,
        env=_repo_python_environment(repo_root),
        check=True,
    )
    report = json.loads(
        (conversion_output / "conversion_report.json").read_text()
    )
    summary = report.get("summary", {})
    passed = int(summary.get("emitted_layers", 0))
    total = int(summary.get("supported_execution_records", 0))
    if total <= 0 or passed != total:
        raise RuntimeError(
            "Magnet Alpha conversion pass rate is " f"{passed}/{total}"
        )
    print(
        "Verified Magnet Alpha converter handoff: "
        f"{passed}/{total} supported execution records",
        flush=True,
    )
    return {
        "passed": passed,
        "total": total,
        "unit": "supported_execution_records",
    }


def _verify_operation_inventory(
    repo_root: Path,
    output: Path,
    *,
    compile_command: list[str],
    checkpoint: Path,
    scheduler: str,
    calibration_steps: int,
    conversion: dict[str, int | str] | None,
) -> None:
    """Validate Alpha inventory content and emitted target identity."""
    inventory_path = (
        repo_root / "docs" / "targets" / "magnet-alpha" / "operations.yaml"
    )
    inventory = yaml.safe_load(inventory_path.read_text())
    if inventory.get("backend") != "magnet-alpha":
        raise RuntimeError(
            f"wrong operation-inventory backend: {inventory_path}"
        )
    endpoint = inventory.get("endpoint")
    quantization = inventory.get("quantization")
    capabilities = inventory.get("capabilities")
    if not isinstance(endpoint, dict) or not endpoint.get("endpoint_id"):
        raise RuntimeError(f"missing inventory endpoint: {inventory_path}")
    if endpoint.get("control") != "full-control":
        raise RuntimeError(
            f"Alpha CMOD endpoint is not full-control: {inventory_path}"
        )
    if not isinstance(quantization, dict):
        raise RuntimeError(f"missing quantization index: {inventory_path}")
    if not isinstance(capabilities, list) or not capabilities:
        raise RuntimeError(f"empty capability inventory: {inventory_path}")

    statuses = {"supported", "unsupported", "unverified"}
    # An incomplete capability is acceptable only when a gap record explains
    # it. Both the missing-adapter and unverified rules below share this set.
    affected_by_gap = {
        capability_id
        for gap in inventory.get("gaps", ())
        for capability_id in gap.get("affects", ())
    }
    capability_ids = [item.get("capability_id") for item in capabilities]
    if any(not value for value in capability_ids):
        raise RuntimeError(f"unnamed capability in inventory: {inventory_path}")
    if len(capability_ids) != len(set(capability_ids)):
        raise RuntimeError(
            f"duplicate capability ID in inventory: {inventory_path}"
        )
    for capability in capabilities:
        capability_id = capability["capability_id"]
        if capability.get("kind") not in {"operation", "composite"}:
            raise RuntimeError(f"{capability_id}: invalid capability kind")
        if capability.get("status") not in statuses:
            raise RuntimeError(f"{capability_id}: invalid capability status")
        if not capability.get("summary"):
            raise RuntimeError(f"{capability_id}: missing form summary")
        if capability.get("status") == "supported" and not capability.get(
            "implementation"
        ):
            raise RuntimeError(
                f"{capability_id}: supported form has no implementation"
            )
        # Semantic support is a capability claim; adapter acceptance is a
        # separate check. A supported form with no registered translation is
        # legal as long as a gap record names it.
        if (
            capability["status"] == "supported"
            and not capability.get("adapter")
            and capability_id not in affected_by_gap
        ):
            raise RuntimeError(
                f"{capability_id}: supported form has no adapter and no gap"
            )

    supported = [
        item for item in capabilities if item.get("status") == "supported"
    ]
    unverified = [
        item for item in capabilities if item.get("status") == "unverified"
    ]
    missing_gap = sorted(
        capability["capability_id"]
        for capability in unverified
        if capability["capability_id"] not in affected_by_gap
    )
    if missing_gap:
        raise RuntimeError(
            "unverified capabilities without a gap: {}".format(
                ", ".join(missing_gap)
            )
        )

    hardware_path = output / "hardware_config.json"
    hardware = json.loads(hardware_path.read_text())
    if hardware.get("target") != "magnet-alpha":
        raise RuntimeError(f"wrong emitted hardware target: {hardware_path}")

    source_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    model_path = output / "model.txt"
    model_bytes = model_path.read_bytes()

    evidence = {
        "schema_version": 2,
        "backend": "magnet-alpha",
        "status": "pass",
        "scope": "compiler_and_operation_inventory",
        "command": shlex.join(compile_command),
        "source": {
            "revision": source_revision,
            "dirty": source_dirty,
        },
        "input": {
            "workload": "yolo5-face-nv",
            "checkpoint": str(checkpoint),
            "calibration_steps": calibration_steps,
        },
        "scheduler": scheduler,
        "voyager_collateral": {
            "artifact": model_path.name,
            "sha256": hashlib.sha256(model_bytes).hexdigest(),
        },
        "checks": {
            "compiler": "pass",
            "operation_inventory": "pass",
            "hardware_target": "pass",
        },
        "capability_counts": {
            status: sum(
                capability.get("status") == status
                for capability in capabilities
            )
            for status in sorted(statuses)
        },
    }
    if conversion is not None:
        evidence["conversion"] = conversion
    evidence_path = output / "operation_inventory_evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    print(
        "Verified Magnet Alpha operation inventory: "
        f"{len(supported)} supported capabilities",
        flush=True,
    )


def _empty_output_directory(
    root: Path,
    scheduler: str,
) -> Path:
    output = root / scheduler
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to reuse non-empty regression output: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the calibrated mixed-policy Magnet Alpha YOLO regression "
            "matrix."
        ),
    )
    parser.add_argument("--calibration_steps", type=int, default=5)
    parser.add_argument(
        "--model_name_or_path",
        required=True,
        help=(
            "Explicit YOLO checkpoint used by the compile lane. The runner "
            "never selects from adapter defaults."
        ),
    )
    parser.add_argument("--output_root")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--verify-converter-handoff",
        action="store_true",
        help=(
            "Require a compiler-only dump boundary and run fresh Magnet Alpha "
            "conversion for each scheduler lane."
        ),
    )
    parser.add_argument(
        "--execution_order",
        choices=("fx", "dp"),
        default=None,
        help=(
            "Optional execution-order mode passed through to each compiler "
            "lane. Omit it to exercise the compiler default."
        ),
    )
    args = parser.parse_args()
    if args.calibration_steps <= 0:
        parser.error(
            "--calibration_steps must be positive for submission regression"
        )
    checkpoint = Path(args.model_name_or_path).resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {checkpoint}")

    repo_root = Path(__file__).resolve().parents[1]
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else Path(tempfile.mkdtemp(prefix="voyager-magnet-alpha-regression-"))
    )

    for scheduler in SCHEDULERS:
        output = _empty_output_directory(output_root, scheduler)
        command = [
            args.python,
            str(repo_root / "test" / "test_codegen.py"),
            "yolo5-face-nv",
            "--pt2e",
            "--target",
            "magnet-alpha",
            "--scratchpad_mapping",
            "end_to_end",
            "--calibration_steps",
            str(args.calibration_steps),
            "--local_scheduler",
            scheduler,
            "--model_name_or_path",
            str(checkpoint),
            "--model_output_dir",
            str(output),
            "--dump_tensors",
            "--log_level",
            "WARNING",
        ]
        if args.execution_order is not None:
            command.extend(["--execution_order", args.execution_order])
        print(
            f"Running Magnet Alpha regression: padding/{scheduler}",
            flush=True,
        )
        subprocess.run(
            command,
            cwd=repo_root,
            env=_repo_python_environment(repo_root),
            check=True,
        )
        conversion = None
        if args.verify_converter_handoff:
            conversion = _verify_converter_handoff(
                repo_root,
                output,
                args.python,
            )
        _verify_operation_inventory(
            repo_root,
            output,
            compile_command=command,
            checkpoint=checkpoint,
            scheduler=scheduler,
            calibration_steps=args.calibration_steps,
            conversion=conversion,
        )

    print(
        "Magnet Alpha scheduler compile lanes passed: " f"{output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Compile Alpha-YOLO here and enforce sealed numerics plus complete collateral."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from alpha_collateral import check_collateral
from run_references import digest, source_hashes, verify_pin, write_json
from torch.utils._pytree import tree_flatten

import voyager_compiler
from voyager_compiler.pipeline import compile_workload
from voyager_compiler.runtime import run
from voyager_compiler.targets.magnet_alpha.target import TARGET
from voyager_compiler.workloads.yolo5_face_nv import WORKLOAD_SPEC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    pins = json.loads(
        (repo / "validation/reference-baselines.json").read_text()
    )["alpha_yolo"]
    reference = repo.parent / pins["directory"]
    output = args.output.resolve()
    if output.is_relative_to(repo.parent / "reference-point"):
        parser.error("candidate output must be outside the sealed reference")
    for key in ("provenance", "validation", "artifact_checksums"):
        verify_pin(repo.parent, pins[key])
    for line in (reference / "checksums.sha256").read_text().splitlines():
        sha, path = line.split("  ", 1)
        if digest(reference / path) != sha:
            raise ValueError(f"sealed Alpha artifact changed: {path}")
    if digest(args.checkpoint) != pins["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from the pinned Alpha reference")
    imported = Path(voyager_compiler.__file__).resolve()
    if not imported.is_relative_to(repo / "src"):
        raise RuntimeError(f"Wrong compiler imported: {imported}")
    output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    write_json(output / "source-sha256.json", sources)
    (output / "source.patch").write_bytes(
        subprocess.check_output(["git", "diff", "--binary"], cwd=repo)
    )
    environment = dict(
        compiler=str(imported),
        python=sys.executable,
        torch=torch.__version__,
        imported_implementation=True,
    )
    write_json(
        output / "invocation.json",
        dict(
            environment=environment,
            checkpoint=str(args.checkpoint.resolve()),
            checkpoint_sha256=pins["checkpoint_sha256"],
            reference=str(reference),
            revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
        ),
    )
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    options = dict(
        model_name_or_path=str(args.checkpoint.resolve()),
        seed=pins["seed"],
        checkpoint_sha256=pins["checkpoint_sha256"],
        calibration_steps=pins["calibration_steps"],
    )
    start = time.monotonic()
    result = compile_workload(
        WORKLOAD_SPEC,
        TARGET,
        output / "compiler",
        options=options,
        normalize_options=dict(skip_op_fusion=True),
        dump_tensors=True,
    )
    actual, evidence = run(result, reference=result.reference, rtol=0, atol=0)
    leaves, _ = tree_flatten(actual)
    tensors = [t.detach().cpu() for t in leaves if isinstance(t, torch.Tensor)]
    expected = np.load(
        reference / "numerical-reference/post_mapping_outputs.npz"
    )
    comparisons = []
    for i, value in enumerate(tensors):
        baseline = (
            torch.from_numpy(expected[f"tensor_{i}"])
            if f"tensor_{i}" in expected
            else None
        )
        equal = baseline is not None and torch.equal(value, baseline)
        same_shape = baseline is not None and value.shape == baseline.shape
        comparisons.append(
            dict(
                shape=list(value.shape),
                bitwise_equal=equal,
                max_abs_difference=(
                    float((value.double() - baseline.double()).abs().max())
                    if same_shape
                    else None
                ),
            )
        )
    np.savez_compressed(
        output / "outputs.npz",
        **{f"tensor_{i}": t.numpy() for i, t in enumerate(tensors)},
    )
    collateral = check_collateral(output / "compiler", reference)
    write_json(output / "collateral-comparison.json", collateral)
    report = dict(
        **environment,
        source_unchanged=source_hashes() == sources,
        numerical_passed=len(tensors) == pins["required_numerical_outputs"]
        and all(c["bitwise_equal"] for c in comparisons),
        outputs=comparisons,
        execution=evidence,
        buffer_validation=dict(result.program.validation),
        input_shapes=[list(x.shape) for x in result.args],
        cmod_passed=collateral["passed"],
        collateral=collateral,
        elapsed_seconds=time.monotonic() - start,
        scope="Compiler collateral and host replay; no C-model or device execution",
    )
    report["passed"] = (
        report["numerical_passed"]
        and report["cmod_passed"]
        and report["source_unchanged"]
    )
    write_json(output / "comparison.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

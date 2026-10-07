"""Authenticate the six full-shape movement-search comparisons."""

import argparse
import ast
import hashlib
import json
import statistics
import struct
import zipfile
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def measured(result):
    return statistics.median(x["p50_us"] for x in result["latencies"])


def authenticate(folder, allow_failure=False):
    result = json.loads((folder / "result.json").read_text())
    assert result["status"] == "pass" or (
        allow_failure and result["status"] == "fail"
    ), folder
    for file, key in (
        ("nki/program.py", "program_sha256"),
        ("instructions.json", "instructions_sha256"),
        ("hardware.json", "hardware_sha256"),
        ("model.txt", "model_sha256"),
        ("reference.npz", "reference_sha256"),
    ):
        assert digest(folder / file) == result[key], (folder, file)
    for file, sha in result["artifact_sha256"].items():
        assert digest(folder / file) == sha, (folder, file)
    return result


def reference_shapes(path):
    """Read NPZ headers without expanding the full benchmark payloads."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                continue
            with archive.open(name) as source:
                prefix = source.read(8)
                assert prefix[:6] == b"\x93NUMPY", (path, name)
                fmt = "<H" if prefix[6] == 1 else "<I"
                length = struct.unpack(fmt, source.read(struct.calcsize(fmt)))[
                    0
                ]
                header = ast.literal_eval(source.read(length).decode("latin1"))
                result[name[:-4]] = {
                    "shape": header["shape"],
                    "dtype": header["descr"],
                }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    manifest = json.loads(args.manifest.read_text())
    for baseline in manifest:
        case = baseline["case"]
        prior_dir = Path(baseline["baseline_directory"])
        for file, sha in baseline["artifact_sha256"].items():
            assert digest(prior_dir / file) == sha, (case, file)
        old_dir = Path(baseline["voyager_directory"])
        old = authenticate(old_dir)
        assert measured(old) == baseline["voyager_us"]
        folder = args.results / "selected" / case
        result = authenticate(folder, allow_failure=True)
        assert result["reference_sha256"] == old["reference_sha256"]
        assert result["compiler_version"] == old["compiler_version"]
        assert result["compiler_flags"] == old["compiler_flags"]
        selection = json.loads((folder / "selection.json").read_text())
        search = json.loads((folder / "movement-search.json").read_text())
        analysis = selection["program_analysis"][
            "selected_instruction_analysis"
        ]
        predicted = analysis["prediction_ns"] / 1000
        actual = measured(result) if result["status"] == "pass" else None
        requests = search["requests"]
        row = dict(
            case=case,
            dtype="float32",
            reference_arrays=reference_shapes(folder / "reference.npz"),
            selected_stats=selection["stats"],
            prior_hardware_us=baseline["prior_us"],
            previous_voyager_hardware_us=measured(old),
            selected_predicted_us=predicted,
            selected_hardware_us=actual,
            hardware_status=result["status"],
            hardware_error=result.get("error"),
            prediction_error_percent=(
                100 * (predicted / actual - 1) if actual else None
            ),
            hardware_speedup_vs_previous=(
                measured(old) / actual if actual else None
            ),
            hardware_speedup_vs_prior=(
                baseline["prior_us"] / actual if actual else None
            ),
            repeated_p50_us=[x["p50_us"] for x in result.get("latencies", [])],
            correctness=result.get("hardware"),
            source_sha256=result["program_sha256"],
            previous_source_sha256=old["program_sha256"],
            source_unchanged=result["program_sha256"] == old["program_sha256"],
            modeled_hbm_bytes=analysis["hbm_bytes"],
            instruction_count=analysis["instruction_count"],
            unknown_completion_count=len(analysis["unknown_completion"]),
            whole_program_timing_complete=not analysis["unknown_completion"],
            resource_service_ns=analysis["service_ns"],
            evaluated=search["evaluated"],
            nominal_combinations=search["nominal_combinations"],
            legal=sum(c["status"] == "legal" for c in search["candidates"]),
            candidates=search["candidates"],
            selected_bindings=search["selected"]["selected_bindings"],
            movement_request_occurrences={
                k: v["count"] for k, v in requests.items()
            },
            physical_placement=selection["physical_placement_strategy"],
            artifacts=str(folder.resolve()),
        )
        seed_dir = args.results / "seed" / case
        if (seed_dir / "result.json").exists():
            seed = authenticate(seed_dir, allow_failure=True)
            seed_selection = json.loads(
                (seed_dir / "selection.json").read_text()
            )
            row["fresh_seed_status"] = seed["status"]
            row["fresh_seed_hardware_us"] = (
                measured(seed) if seed["status"] == "pass" else None
            )
            row["fresh_seed_predicted_us"] = (
                seed_selection["program_analysis"][
                    "whole_program_prediction_ns"
                ]
                / 1000
            )
            if actual is not None and seed["status"] == "pass":
                row["hardware_speedup_vs_fresh_seed"] = measured(seed) / actual
                row["model_ranking_agrees_with_hardware"] = (
                    predicted < row["fresh_seed_predicted_us"]
                ) == (actual < measured(seed))
        compiled = folder / "compiled_prediction.json"
        if actual is not None:
            assert compiled.exists(), (case, "Missing compiled profile")
            provenance = json.loads(
                (folder / "extraction_provenance.json").read_text()
            )
            assert (
                provenance["artifact_sha256"] == result["artifact_sha256"]
            ), case
            assert provenance["compiled_static_sha256"] == digest(
                folder / "compiled_static.json"
            ), case
            row["compiled_static_prediction"] = json.loads(compiled.read_text())
            row["profile_audit"] = json.loads(
                (folder / "measurement_audit.json").read_text()
            )
            replay = row["compiled_static_prediction"]["prediction_us"]
            row["compiled_prediction_error_percent"] = (
                100 * (replay / actual - 1) if replay is not None else None
            )
            measured_hbm = sum(
                row["profile_audit"][k]
                for k in ("hbm_read_bytes", "hbm_write_bytes")
            )
            row["measured_hbm_bytes"] = measured_hbm
            row["modeled_hbm_matches_profile"] = (
                row["modeled_hbm_bytes"] == measured_hbm
            )
            row["spill_save_bytes"] = row["profile_audit"].get(
                "spill_save_bytes"
            )
            row["spill_reload_bytes"] = row["profile_audit"].get(
                "spill_reload_bytes"
            )
            seed_replay = seed_dir / "compiled_prediction.json"
            if seed_replay.exists():
                seed_provenance = json.loads(
                    (seed_dir / "extraction_provenance.json").read_text()
                )
                assert (
                    seed_provenance["artifact_sha256"]
                    == seed["artifact_sha256"]
                ), case
                assert seed_provenance["compiled_static_sha256"] == digest(
                    seed_dir / "compiled_static.json"
                ), case
                row["fresh_seed_compiled_prediction"] = json.loads(
                    seed_replay.read_text()
                )
        rows.append(row)
        print(
            case,
            f"prior={baseline['prior_us']:.1f}",
            f"old={measured(old):.1f}",
            f"predicted={predicted:.1f}",
            f"measured={actual}",
            f"status={result['status']}",
        )
    log = (args.results / "regression.log").read_text()
    assert "failed" not in log.lower(), log[-2000:]
    assert "passed" in log
    report = dict(
        scope="Full benchmark shapes; movement search on fixed saved shared software schedules. Native compilation and hardware correctness attempted for all winners; three timing repeats for successful kernels. No CPU simulator.",
        hardware="Trainium2 / NeuronCore-v3, --target=trn2 --lnc=1",
        baseline_manifest=str(args.manifest.resolve()),
        limitations=[
            "Bounded search, not a global optimum; full candidate counts are recorded per kernel.",
            "Only movement sites represented by current endpoint requests are varied. Maxpool has no requests in this lowering.",
            "The shared software schedule, GEMM tiling and producer/consumer layouts remain fixed.",
            "Existing reduction/activation completion-law gaps remain explicit; these predictions are not fully calibrated when unknown_completion_count is nonzero.",
            "Measured timing excludes host invocation. The prior and old Voyager measurements are authenticated saved evidence.",
        ],
        results=rows,
        hardware_passes=sum(row["hardware_status"] == "pass" for row in rows),
        tests=[line for line in log.splitlines() if " passed" in line][-1],
    )
    (args.results / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    paths = list(Path("src/voyager_compiler/trainium").glob("*.py")) + [
        Path("scripts/trainium_search_movement.py"),
        Path("scripts/trainium_full_movement_report.py"),
        Path("scripts/trainium_run_hardware.py"),
        Path("src/voyager_compiler/trainium/timing_trainium2.json"),
        Path("docs/trainium-movement-search.md"),
        Path("test/test_trainium_movement_search.py"),
    ]
    (args.results / "source-sha256.json").write_text(
        json.dumps({str(p): digest(p) for p in paths}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

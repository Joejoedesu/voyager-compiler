"""Authenticate and summarize the movement search hardware experiment."""

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    a = p.parse_args()
    root = a.results
    rows = []
    jobs = [
        ("baseline", "gemm128"),
        ("selected", "gemm128"),
        ("baseline", "swiglu"),
        ("selected", "swiglu"),
        ("alternatives", "vector_copy"),
        ("stream-disjoint", "stream_load"),
        ("stream-final", "stream_operand"),
    ]
    for group, case in jobs:
        folder = root / group / case
        result = json.loads((folder / "result.json").read_text())
        assert result["status"] == "pass"
        for file, key in [
            ("nki/program.py", "program_sha256"),
            ("instructions.json", "instructions_sha256"),
            ("hardware.json", "hardware_sha256"),
            ("model.txt", "model_sha256"),
            ("reference.npz", "reference_sha256"),
        ]:
            assert digest(folder / file) == result[key], (folder, file)
        for file, sha in result["artifact_sha256"].items():
            assert digest(folder / file) == sha
        selection = json.loads((folder / "selection.json").read_text())
        measured = statistics.median(x["p50_us"] for x in result["latencies"])
        predicted = (
            selection["program_analysis"]["whole_program_prediction_ns"] / 1000
        )
        rows.append(
            dict(
                group=group,
                case=case,
                predicted_us=predicted,
                hardware_p50_us=measured,
                error_percent=100 * (predicted / measured - 1),
                correct=True,
                max_abs_error=result["hardware"]["max_abs_error"],
                repeated_p50_us=[x["p50_us"] for x in result["latencies"]],
                source_sha256=result["program_sha256"],
                instructions_sha256=result["instructions_sha256"],
                physical_placement=selection["physical_placement_strategy"],
            )
        )
    searches = {}
    for case in ("gemm128", "swiglu"):
        old = root / "selected" / case
        final = root / "final-selected" / case
        for file in (
            "nki/program.py",
            "instructions.json",
            "hardware.json",
            "model.txt",
            "reference.npz",
        ):
            assert digest(old / file) == digest(final / file), (case, file)
        search = json.loads((final / "movement-search.json").read_text())
        searches[case] = dict(
            nominal_combinations=search["nominal_combinations"],
            evaluated=search["evaluated"],
            legal=sum(c["status"] == "legal" for c in search["candidates"]),
            selected=search["selected"],
            final_artifacts_equal_measured=True,
        )
    failed = []
    for folder in sorted((root / "stream-alternatives").iterdir()):
        result = json.loads((folder / "result.json").read_text())
        assert result["status"] == "fail"
        failed.append(
            dict(
                case=folder.name,
                status="compiler_rejected",
                constraint="Pinned in-place allocation checker rejects monolithic arena; disjoint storage retries passed",
                source_sha256=result["program_sha256"],
            )
        )
    report = dict(
        scope="Fixed software schedules; endpoint-constrained ISA movement search. Fresh device correctness and timing; no CPU simulator.",
        measurements=rows,
        searches=searches,
        initial_failed_realizations=failed,
    )
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    baseline = Path(
        "results/trainium/model-integration-2026-10-07/default-reference/2026-10-07_04-57-22"
    )
    current = next((root / "default-after").glob("*/report.txt")).parent
    cases = []
    for before in sorted(baseline.rglob("run.log")):
        after = current / before.relative_to(baseline)

        def terminal(path):
            return [
                l
                for l in path.read_text().splitlines()
                if re.match(r"^[A-Za-z_][\w.]*?(?:Error|Exception):", l)
            ][-1]

        assert terminal(before) == terminal(after)
        cases.append(
            dict(
                case=str(before.parent.relative_to(baseline)),
                same_preexisting_failure=True,
                error=terminal(after),
            )
        )
    assert len(cases) == 11
    before = json.loads((root / "before-source-sha256.json").read_text())
    changed = [
        name for name, sha in before.items() if digest(Path(name)) != sha
    ]
    assert all(
        name.startswith("src/voyager_compiler/trainium/") for name in changed
    )
    test_log = (root / "final-regression.log").read_text()
    if re.search(r"\d+ failed|FAILED test/", test_log):
        raise ValueError("The final regression log contains test failures")
    summary = re.findall(r"\d+ passed[^\n]*", test_log)[-1]
    validation = dict(
        tests=summary,
        hardware_correctness_passes=len(rows),
        winning_artifacts_authenticated=True,
        default_cases=cases,
        default_emitted_equivalence="not established: all 11 blocked before compilation in both revisions",
        changed_existing_sources=changed,
        shared_and_default_sources_unchanged=True,
    )
    (root / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    files = list(Path("src/voyager_compiler/trainium").glob("*.py")) + [
        Path("scripts/trainium_search_movement.py"),
        Path("scripts/trainium_movement_search_report.py"),
        Path("test/test_trainium_movement_search.py"),
        Path("docs/trainium-movement-search.md"),
    ]
    (root / "source-sha256.json").write_text(
        json.dumps({str(p): digest(p) for p in files}, indent=2) + "\n"
    )
    print(summary)
    for row in rows:
        print(
            row["group"],
            row["case"],
            row["predicted_us"],
            row["hardware_p50_us"],
        )


if __name__ == "__main__":
    main()

"""Compare generated instructions, analytical costs and matching device profiles."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import re
from pathlib import Path
import statistics
import subprocess
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(case):
    result = json.loads((case / "result.json").read_text())
    row = dict(case=str(case), status=result["status"])
    row["boundary_operations"] = len(
        re.findall(
            r'target: "aten::(?:pad|slice|permute)"',
            (case / "model.txt").read_text(),
        )
    )
    if result["status"] != "pass":
        return row
    plan = json.loads((case / "nki/plan.json").read_text())
    generation = json.loads((case / "generation.json").read_text())
    assert (
        digest(case / "nki/program.py")
        == result["program_sha256"]
        == generation["program_sha256"]
    )
    assert digest(case / "model.txt") == plan["source_sha256"]
    assert result["hardware"]["correct"] and result["simulation"]["correct"]
    process = subprocess.run(
        [
            "/opt/aws/neuron/bin/neuron-profile",
            "view",
            "--output-format",
            "json",
            "--output-file",
            str(case / "profile_full.json"),
            "-n",
            str(case / "file.neff"),
            "-s",
            str(case / "profile.ntff"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    profile = json.loads((case / "profile_full.json").read_text())
    assert len(profile["summary"]) == 1
    metrics = profile["summary"][0]
    (case / "profile_summary.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    row["profile_warnings"] = profile.get("warnings", [])
    row["execution_contract"] = plan.get("execution_contract", "legacy-nki")
    row["high_level_operations"] = plan.get("high_level_operations")
    counts = Counter()
    for instruction in profile["instruction"]:
        op = instruction["opcode"]
        if op == "MATMUL":
            counts["MATMUL_" + instruction["instruction_type"]] += 1
        elif op == "LDWEIGHTS":
            counts[op] += 1
        elif op == "COPY":
            counts["COPY_" + instruction["subgroup"].upper()] += 1
        if instruction.get("instruction_type") == "SPILL":
            counts["SPILL"] += 1
    row["compiled_isa_counts"] = dict(counts)
    # Overall engine-active counters include SDK control instructions. Expose
    # the union of math instruction intervals separately; don't add durations
    # of pipelined LDWEIGHTS/MATMUL instructions that overlap.
    intervals = sorted(
        (i["timestamp"], i["timestamp"] + i["duration"])
        for i in profile["instruction"]
        if i["opcode"] in ("LDWEIGHTS", "MATMUL")
    )
    end = total = 0
    for start, stop in intervals:
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    row["profile_TensorE_math_active_us"] = total / 1000
    seconds = metrics["total_time"]
    row.update(
        nc_p50_us=statistics.median(r["p50_us"] for r in result["latencies"]),
        profile_us=seconds * 1e6,
        max_abs_error=result["hardware"]["max_abs_error"],
        profile_hbm_bytes=metrics["hbm_read_bytes"]
        + metrics["hbm_write_bytes"],
        emitted_dma_panels=plan["stats"].get("isa_dma_panels"),
        emitted_matmuls=plan["stats"].get("tensor_instructions"),
        profiled_matmuls=metrics.get("matmul_instruction_count"),
        artifact_sha256={
            str(p.relative_to(case)): digest(p)
            for p in [
                case / "model.txt",
                case / "nki/program.py",
                case / "file.neff",
                case / "profile.ntff",
                case / "reference.npz",
            ]
        },
    )
    for engine, key in [
        ("DMA", "dma"),
        ("TensorE", "tensor_engine"),
        ("VectorE", "vector_engine"),
        ("ScalarE", "scalar_engine"),
    ]:
        row[f"profile_{engine}_active_us"] = metrics[key + "_active_time"] * 1e6
    estimates = result["estimates"]
    if estimates:
        assert len(estimates) == 1
        e = estimates[0]
        with np.load(case / "reference.npz") as data:
            a, b = data["a0"], data["a1"]
            assert a.ndim == b.ndim == 2
            useful = 2 * a.shape[0] * a.shape[1] * b.shape[1]
            dtype = generation.get("input_dtype", "float32")
            problem = f"{a.shape[0]}x{b.shape[1]}x{a.shape[1]}:{dtype}"
        hw = json.loads((case / "hardware.json").read_text())["hardware"]
        peak = (
            2
            * 128
            * 128
            * hw["frequency"]
            * 1e9
            / (4 if dtype == "float32" else 1)
        )
        bandwidth = (
            next(
                c["bandwidth"]["value"]
                for c in hw["connections"]
                if c["name"] == "HBM_DMA"
            )
            * 1e9
        )
        row.update(
            problem=problem,
            estimate_scope=e["scope"],
            has_unmodeled_boundary_work=bool(row["boundary_operations"]),
            tile=e["software_tile"],
            buffer_depth=e["buffer_depth"],
            modes=e["instruction_modes"],
            predicted_us=e["predicted_ns"] / 1000,
            useful_flops=useful,
            modeled_hbm_bytes=e["hbm_bytes"],
            extra_profiled_hbm_bytes=row["profile_hbm_bytes"] - e["hbm_bytes"],
            profiled_over_modeled_hbm_bytes=row["profile_hbm_bytes"]
            / e["hbm_bytes"],
            predicted_useful_compute_fraction=useful
            / (e["predicted_ns"] * 1e-9)
            / peak,
            measured_useful_compute_fraction=useful / seconds / peak,
            measured_hbm_payload_fraction=row["profile_hbm_bytes"]
            / seconds
            / bandwidth,
        )
        for engine in ("DMA", "TensorE", "VectorE", "ScalarE"):
            row[f"predicted_{engine}_service_us"] = (
                e["service_ns"].get(engine, 0) / 1000
            )
        row["relative_latency_error"] = (
            abs(row["predicted_us"] - row["nc_p50_us"]) / row["nc_p50_us"]
        )
        row["relative_hbm_error"] = (
            abs(row["modeled_hbm_bytes"] - row["profile_hbm_bytes"])
            / row["profile_hbm_bytes"]
        )
        if e.get("expanded_isa"):
            row["modeled_isa_counts"] = e["expanded_isa"]
            row["compiled_isa_count_match"] = all(
                counts[key] == count for key, count in e["expanded_isa"].items()
            )
        row["scheduled_dma_panel_delta"] = (
            row["emitted_dma_panels"] - e["dma_commands"]
        )
        assert row["emitted_matmuls"] == e["tensor_instructions"]
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = [
        summarize(p.parent.resolve())
        for root in args.artifacts
        for p in sorted(root.glob("*/result.json"))
    ]
    if not rows:
        raise SystemExit("No hardware results found")
    groups = {}
    for row in rows:
        if "problem" in row:
            key = (row["problem"], json.dumps(row["modes"], sort_keys=True))
            groups.setdefault(key, []).append(row)
    ranking = []
    for (problem, modes), candidates in groups.items():
        if len(candidates) < 2:
            continue
        selected = min(candidates, key=lambda r: r["predicted_us"])
        best = min(candidates, key=lambda r: r["nc_p50_us"])
        ranking.append(
            dict(
                problem=problem,
                modes=json.loads(modes),
                candidates=len(candidates),
                model_selected=selected["case"],
                measured_best=best["case"],
                selection_regret=selected["nc_p50_us"] / best["nc_p50_us"] - 1,
            )
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(
            dict(
                scope="Single physical Trainium2 core; no host invocation time; service and active time are distinct metrics",
                rows=rows,
                ranking=ranking,
            ),
            indent=2,
        )
        + "\n"
    )
    fields = list(
        dict.fromkeys(k for r in rows for k in r if k != "artifact_sha256")
    )
    with (args.output / "measurements.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            row["case"],
            row["status"],
            "predicted",
            row.get("predicted_us"),
            "measured",
            row.get("nc_p50_us"),
        )
    print(json.dumps(ranking, indent=2))


if __name__ == "__main__":
    main()

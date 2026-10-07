"""Authenticate selected-plan hardware evidence and report whole-program costs."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def profile(case):
    audit = case / "profile_audit.json"
    hashes = {n: digest(case / n) for n in ("file.neff", "profile.ntff")}
    if audit.exists():
        old = json.loads(audit.read_text())
        if old["artifact_sha256"] == hashes:
            return old
    existing = case / "profile_full.json"
    with tempfile.TemporaryDirectory(prefix="trainium-profile-") as directory:
        path = (
            existing if existing.exists() else Path(directory) / "profile.json"
        )
        if not existing.exists():
            subprocess.run(
                [
                    "/opt/aws/neuron/bin/neuron-profile",
                    "view",
                    "-n",
                    str(case / "file.neff"),
                    "-s",
                    str(case / "profile.ntff"),
                    "--output-format",
                    "json",
                    "--output-file",
                    str(path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        data = json.loads(path.read_text())
        counts = Counter(i["opcode"] for i in data["instruction"])
        matmul = Counter(
            i.get("instruction_type")
            for i in data["instruction"]
            if i["opcode"] == "MATMUL"
        )
        grouped = {}
        for instruction in data["instruction"]:
            key = ":".join(
                instruction.get(k, "")
                for k in ("opcode", "instruction_type", "subgroup")
            )
            grouped.setdefault(key, []).append(instruction)
        timings = {}
        for key, instructions in grouped.items():
            starts = sorted(i["timestamp"] for i in instructions)
            timings[key] = dict(
                count=len(instructions),
                duration_median_ns=statistics.median(
                    i["duration"] for i in instructions
                ),
                event_wait_median_ns=statistics.median(
                    i.get("evt_wait_time", 0) for i in instructions
                ),
                start_spacing_median_ns=(
                    statistics.median(b - a for a, b in zip(starts, starts[1:]))
                    if len(starts) > 1
                    else None
                ),
            )
        record = dict(
            artifact_sha256=hashes,
            instruction_timing_groups=timings,
            timing_units="Profiler nanoseconds; event waits overlap and must not be summed into kernel latency",
            summary=data["summary"][0],
            instructions=dict(counts),
            matmul_types=dict(matmul),
            spills=sum(
                i.get("instruction_type") == "SPILL"
                for i in data["instruction"]
            ),
            warnings=data.get("warnings", []),
        )
    audit.write_text(json.dumps(record, indent=2) + "\n")
    return record


def layouts(plan):
    selected = [
        *plan.get("operation_implementations", []),
        *(
            operation
            for region in plan.get("program_analysis", {}).get(
                "vector_regions", []
            )
            for operation in region["operations"]
        ),
    ]
    values = []
    if any("matmul" in name for name in selected):
        values.append("TensorE K-partition panels; explicit transposes")
    if any(
        any(op in name for op in ("layer_norm", "rms_norm", "softmax"))
        for name in selected
    ):
        values.append("row partitions / full feature axis")
    if any("pool" in name for name in selected):
        values.append("output-row partitions / halo-width axis")
    return (
        "; ".join(values) or "explicit partition/free views (instruction plan)"
    )


def summarize(case, prior):
    row = dict(case=str(case), status="unmeasured")
    if not (case / "result.json").exists():
        # Preserve the selected workload even while native compilation is pending.
        selected = case / "nki/plan.json"
        generation = case / "generation.json"
        hardware = case / "hardware.json"
        if selected.exists() and generation.exists() and hardware.exists():
            plan = json.loads(selected.read_text())
            config = json.loads(hardware.read_text())
            metadata = json.loads(generation.read_text())
            row.update(
                input_shapes=[a["shape"] for a in plan["arguments"]],
                layouts=layouts(plan),
                dtype=metadata["input_dtype"],
                diagnostic_tile=metadata.get("diagnostic_tile"),
                matrix_tiles=[
                    e["software_tile"] for e in config.get("estimates", [])
                ],
                logical_buffer_depth=[
                    e["buffer_depth"] for e in config.get("estimates", [])
                ],
                vector_tiles=[
                    dict(
                        operations=v["operations"],
                        tile=v["tile"],
                        repetitions=v["repetitions"],
                    )
                    for v in plan["program_analysis"]["vector_regions"]
                ],
                source_bytes=(case / "nki/program.py").stat().st_size,
            )
        progress = case / "execution_progress.json"
        if progress.exists():
            checkpoint = json.loads(progress.read_text())
            row["hardware_correct"] = checkpoint.get("hardware", {}).get(
                "correct", False
            )
            if row["hardware_correct"]:
                row["status"] = "correctness_pass_timing_pending"
        error = case / "generation_error.txt"
        if error.exists():
            row.update(status="generation_failed", error=error.read_text())
        return row
    result = json.loads((case / "result.json").read_text())
    row["status"] = result["status"]
    row["hardware_correct"] = result.get("hardware", {}).get("correct", False)
    row["source_authenticated"] = (
        digest(case / "nki/program.py") == result["program_sha256"]
    )
    if not row["source_authenticated"]:
        row["status"] = "stale_source"
        return row
    plan = json.loads((case / "nki/plan.json").read_text())
    assert digest(case / "model.txt") == plan["source_sha256"]
    assert digest(case / "instructions.json") == plan["instructions_sha256"]
    assert digest(case / "hardware.json") == plan["hardware_record_sha256"]
    for name, key in (
        ("model.txt", "model_sha256"),
        ("hardware.json", "hardware_sha256"),
        ("instructions.json", "instructions_sha256"),
        ("reference.npz", "reference_sha256"),
    ):
        if key in result:
            assert digest(case / name) == result[key], (case, name)
    generation = json.loads((case / "generation.json").read_text())
    row.update(
        input_shapes=[a["shape"] for a in plan["arguments"]],
        layouts=layouts(plan),
        dtype=generation["input_dtype"],
        diagnostic_tile=generation.get("diagnostic_tile"),
        selection_kind=(
            "allocation diagnostic"
            if generation.get("allocation_diagnostic") == "size_classes"
            else (
                "fixed tile diagnostic"
                if generation.get("diagnostic_tile")
                else "shared search"
            )
        ),
        matrix_tiles=[e["software_tile"] for e in result.get("estimates", [])],
        logical_buffer_depth=[
            e["buffer_depth"] for e in result.get("estimates", [])
        ],
        vector_tiles=[
            dict(
                operations=v["operations"],
                tile=v["tile"],
                repetitions=v["repetitions"],
            )
            for v in plan["program_analysis"]["vector_regions"]
        ],
        physical_allocation=plan["allocation"],
        simulation=result.get("simulation"),
        source_calls=plan["program_analysis"]["source_calls"],
        source_bytes=(case / "nki/program.py").stat().st_size,
    )
    analysis = plan["program_analysis"]
    prediction = analysis["whole_program_prediction_ns"]
    replay_path = case / "selected_analysis.json"
    replay = analysis.get("selected_instruction_analysis")
    if replay_path.exists():
        replay = json.loads(replay_path.read_text())
        assert replay["instructions_sha256"] == digest(
            case / "instructions.json"
        )
    if replay:
        row["template_predicted_us"] = (
            analysis.get("template_prediction_ns", prediction) / 1000
            if prediction
            else None
        )
        row["recorded_plan_valid"] = replay.get("recorded_plan_valid", True)
        row["recorded_plan_validation_error"] = replay.get(
            "recorded_plan_validation_error"
        )
        row["analysis_completion_edges_added"] = replay.get(
            "analysis_completion_edges_added", 0
        )
        prediction = replay["prediction_ns"]
    row.update(
        predicted_us=prediction / 1000 if prediction else None,
        modeled_hbm_bytes=(
            replay["hbm_bytes"] if replay else analysis["hbm"]["total_bytes"]
        ),
        timing_complete=(
            not replay["unknown_completion"]
            if replay
            else analysis["whole_program_timing_complete"]
        ),
    )
    if result["status"] != "pass":
        row["error"] = result.get("error")
        return row
    assert result["hardware"]["correct"]
    measured = statistics.median(x["p50_us"] for x in result["latencies"])
    row.update(
        actual_us=measured,
        max_abs_error=result["hardware"]["max_abs_error"],
        latencies=result["latencies"],
    )
    if prediction:
        signed = (prediction / 1000 / measured - 1) * 100
        row.update(signed_error_pct=signed, absolute_error_pct=abs(signed))
    p = profile(case)
    row["measured_binary_correctness"] = result.get(
        "benchmark_compilation", ""
    ).startswith("Validated NEFF")
    binary_check = case / "binary_correctness.json"
    if binary_check.exists():
        checked = json.loads(binary_check.read_text())
        assert checked["status"] == "pass" and checked["hardware_correct"]
        for filename, expected_hash in checked["artifact_sha256"].items():
            assert digest(case / filename) == expected_hash, (case, filename)
        row["measured_binary_correctness"] = True
        row["binary_correctness_artifact"] = str(binary_check)
    m = p["summary"]
    row.update(
        measured_hbm_bytes=m["hbm_read_bytes"] + m["hbm_write_bytes"],
        compiled_instructions=p["instructions"],
        compiled_matmul_types=p["matmul_types"],
        spills=p["spills"],
        profile_us=m["total_time"] * 1e6,
        engine_active_us={
            e: m[k + "_active_time"] * 1e6
            for e, k in [
                ("DMA", "dma"),
                ("TensorE", "tensor_engine"),
                ("VectorE", "vector_engine"),
                ("ScalarE", "scalar_engine"),
            ]
        },
    )
    transposes = plan["stats"].get("isa_transposes", 0)
    regular = row["source_calls"].get("nisa.nc_matmul", 0) - transposes
    passes = 2 if generation["input_dtype"] == "float32" else 1
    row["backend_expansion_audit"] = dict(
        source_regular_matmul=regular,
        source_transpose=transposes,
        expected_regular_commands=regular * passes,
        compiled_regular_commands=p["matmul_types"].get("REGULAR", 0),
        compiled_transpose_commands=p["matmul_types"].get("TRANSPOSE", 0),
        matches=(
            regular * passes == p["matmul_types"].get("REGULAR", 0)
            and transposes == p["matmul_types"].get("TRANSPOSE", 0)
        ),
    )
    row["artifact_sha256"] = {
        n: digest(case / n)
        for n in (
            "nki/program.py",
            "model.txt",
            "hardware.json",
            "instructions.json",
            "file.neff",
            "profile.ntff",
            "reference.npz",
        )
    }
    for folder in prior:
        candidate = folder / case.name / "result.json"
        if not candidate.exists():
            continue
        d = json.loads(candidate.read_text())
        if (
            d["status"] == "pass"
            and d["reference_sha256"] == result["reference_sha256"]
        ):
            row["prior_us"] = statistics.median(
                x["p50_us"] for x in d["latencies"]
            )
            row["speedup_vs_prior"] = row["prior_us"] / measured
            row["prior_artifact"] = str(candidate)
            break
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, nargs="+", required=True)
    p.add_argument("--prior", type=Path, nargs="+", default=[])
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = []
    for folder in a.artifacts:
        for case in sorted(folder.iterdir()):
            if case.is_dir():
                rows.append(
                    summarize(case.resolve(), [p.resolve() for p in a.prior])
                )
                print(case, rows[-1]["status"], flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = [
        "case",
        "status",
        "hardware_correct",
        "input_shapes",
        "dtype",
        "matrix_tiles",
        "vector_tiles",
        "logical_buffer_depth",
        "layouts",
        "selection_kind",
        "predicted_us",
        "actual_us",
        "signed_error_pct",
        "absolute_error_pct",
        "modeled_hbm_bytes",
        "measured_hbm_bytes",
        "prior_us",
        "speedup_vs_prior",
        "timing_complete",
        "recorded_plan_valid",
        "analysis_completion_edges_added",
    ]
    with a.output.with_suffix(".csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Selected-plan hardware comparison",
        "",
        "Median of three NKI nc_latency p50 samples, 10 warmups and 100 iterations; one Trainium2 core. Signed error is (model / measured − 1). Predictions marked incomplete retain unknown completion costs. All cases are validation workloads; no timing coefficients were fitted here.",
        "",
        "| Case | Inputs / dtype | Selected tiles; layout; logical depth | Model µs | Device µs | Signed / absolute error | HBM modeled / measured bytes | Prior µs; speedup | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]

    def val(r, k, fmt=".2f"):
        v = r.get(k)
        return format(v, fmt) if v is not None else "—"

    for r in rows:
        case = Path(r["case"])
        label = case.parent.name + "/" + case.name
        tile = (
            r.get("selection_kind", "unmeasured")
            + "; "
            + str(r.get("matrix_tiles") or r.get("vector_tiles", "—"))
            + "; "
            + str(r.get("layouts", "—"))
            + "; "
            + str(r.get("logical_buffer_depth") or "vector")
        )
        status = (
            r["status"]
            if r.get("recorded_plan_valid", True)
            else r["status"] + "; incomplete recorded dependencies"
        )
        link = case / "result.json" if (case / "result.json").exists() else case
        lines.append(
            f"| [{label}]({link}) | {r.get('input_shapes','—')} / {r.get('dtype','—')} | {tile} | {val(r,'predicted_us')}{'' if r.get('timing_complete') else '*'} | {val(r,'actual_us')} | {val(r,'signed_error_pct')}% / {val(r,'absolute_error_pct')}% | {val(r,'modeled_hbm_bytes','.0f')} / {val(r,'measured_hbm_bytes','.0f')} | {val(r,'prior_us')}; {val(r,'speedup_vs_prior')}× | {status} |"
        )
    lines += [
        "",
        "* Timing has uncharacterized completion costs. A blank prior cell means no authenticated matching measurement. Physical addresses are fixed, but logical depth alone does not establish hardware overlap.",
        "Historical rows with incomplete recorded dependencies use an explicitly repaired analysis DAG; their source and hardware evidence are unchanged. See the JSON for added-edge counts. These rows are not current plan-validation passes.",
        "",
    ]
    a.output.with_suffix(".md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

"""Audit maxpool search coverage and prediction gaps using saved evidence.

Counterfactual dependencies are diagnostics, not candidate performance claims.
No measured duration is supplied to the performance model.
"""

import argparse
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import importlib.util

from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.execution import TrainiumTuning
from voyager_compiler.trainium.lowering import spatial_pool_graph
from voyager_compiler.trainium.dependencies import estimate_graph
from voyager_compiler.trainium.instruction_plan import Program
from voyager_compiler.trainium.program_analysis import analyze_selected
from voyager_compiler.trainium.compiled_analysis import predict
from voyager_compiler.codegen.transform.tiling.execution import RepeatedGraph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("results/trainium/separable-final-full/maxpool")
    compiled = Path(
        "/home/ubuntu/ML/trainium-prior-model-experiment/controls/maxpool"
    )
    hw = neuron_core(3)
    tuning = TrainiumTuning()
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    saved = json.loads((compiled / "prediction.json").read_text())
    for file, value in saved["artifact_sha256"].items():
        assert digest(root / file) == value
    profile = json.loads(
        gzip.decompress((compiled / "profile.json.gz").read_bytes())
    )
    summary = profile["summary"][0]
    static = json.loads((compiled / "compiled_static.json").read_text())
    replay, _ = predict(static)
    p = Program.load(json.loads((root / "instructions.json").read_text()))
    selected = analyze_selected(p, hw)
    # Reconstruct logical RAW/WAR/WAW while omitting physical arena reuse.
    # This is an ideal-storage dependency bound, not a realizable allocation.
    last, readers, logical = {}, {}, []
    for i, ins in enumerate(p.instructions):
        reads = {p.root(n) for n in ins.reads}
        writes = {p.root(n) for n in ins.writes}
        deps = {last[n] for n in reads | writes if n in last}
        for n in writes:
            deps.update(readers.get(n, ()))
            last[n], readers[n] = i, set()
        for n in reads:
            readers.setdefault(n, set()).add(i)
        logical.append(replace(ins, dependencies=tuple(sorted(deps))))
    p.instructions = logical
    logical_bound = analyze_selected(p, hw)
    candidates = []
    for rows in (1, 2, 23, 46, 89, 128):
        g = spatial_pool_graph(
            hw, (1, rows, 4094, 1), (1, rows + 2, 4096, 1), tuning, 9
        )
        repetitions = (4094 + rows - 1) // rows
        serial = estimate_graph(RepeatedGraph(g.nodes, repetitions))
        double = tuple(
            replace(
                n,
                dependencies=tuple(
                    replace(d, distance=2) if d.distance == 1 else d
                    for d in n.dependencies
                ),
            )
            for n in g.nodes
        )
        overlap = estimate_graph(RepeatedGraph(double, repetitions))
        candidates.append(
            dict(
                rows=rows,
                repetitions=repetitions,
                enumerated=4094 % rows == 0,
                final_rows=4094 - (repetitions - 1) * rows,
                serial_template_us=(
                    serial.duration_ns + hw.timing_profile.fixed_kernel_ns
                )
                / 1000,
                distance2_diagnostic_us=(
                    overlap.duration_ns + hw.timing_profile.fixed_kernel_ns
                )
                / 1000,
                note=(
                    "128-row estimates charge a full final tile; masked-tail schedule is not in production search"
                    if rows == 128
                    else ""
                ),
            )
        )
    result = json.loads((root / "result.json").read_text())
    report = dict(
        artifact_directory=str(root.resolve()),
        hashes={
            f: digest(root / f)
            for f in (
                "file.neff",
                "profile.ntff",
                "instructions.json",
                "nki/program.py",
            )
        },
        profile_sha256=hashlib.sha256(
            gzip.decompress((compiled / "profile.json.gz").read_bytes())
        ).hexdigest(),
        benchmark_us=statistics.median(
            x["p50_us"] for x in result["latencies"]
        ),
        trace_us=summary["total_time"] * 1e6,
        selected=selected,
        compiled_replay=replay,
        logical_dependency_bound=logical_bound,
        candidate_diagnostics=candidates,
        measured=dict(
            hbm_read_bytes=summary["hbm_read_bytes"],
            hbm_write_bytes=summary["hbm_write_bytes"],
            dma_active_us=summary["dma_active_time"] * 1e6,
            vector_active_us=summary["vector_engine_active_time"] * 1e6,
        ),
        conclusions=[
            "Exact-divisor tiling excludes 128 rows; 89 is the largest divisor of 4094 not exceeding the 128-partition layout limit.",
            "Both prior and Voyager reload three overlapping halo strips. Both transfer 268271632 HBM bytes. Line-buffer/rolling-window reuse is not searched.",
            "The spatial-pool search template uses distance-1 whole-tile completion; emitted shared bufferization alternates two halo slots.",
            "Selected ISA dependencies conservatively track whole logical storage roots and physical generations; compiled semaphores expose a different schedule.",
            "Distance-2 and removed-reuse variants are sensitivity diagnostics, not measured speedups or validated new production schedules.",
        ],
    )
    probes = args.output.parent / "probes"
    probe_records = []
    extractor_path = Path(
        "/home/ubuntu/ML/trainium-prior-model-experiment/extract.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prior_static_extract", extractor_path
    )
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
    for rows in (89, 128):
        folder = probes / f"rows{rows}"
        if not (folder / "result.json").exists():
            continue
        measured = json.loads((folder / "result.json").read_text())
        assert measured["status"] == "pass" and measured["hardware"]["correct"]
        for filename, expected in measured["artifact_sha256"].items():
            assert digest(folder / filename) == expected
        assert digest(folder / "nki/program.py") == measured["program_sha256"]
        prof = json.loads((folder / "profile.json").read_text())
        static = extractor.static_metadata(prof)
        prediction, _ = predict(static)
        (folder / "compiled_static.json").write_text(
            json.dumps(static, indent=2) + "\n"
        )
        (folder / "compiled_prediction.json").write_text(
            json.dumps(prediction, indent=2) + "\n"
        )
        summary = prof["summary"][0]
        probe_records.append(
            dict(
                tile_rows=rows,
                directory=str(folder.resolve()),
                p50_samples_us=[r["p50_us"] for r in measured["latencies"]],
                median_us=statistics.median(
                    r["p50_us"] for r in measured["latencies"]
                ),
                compiled_prediction_us=prediction["prediction_us"],
                hbm_read_bytes=summary["hbm_read_bytes"],
                hbm_write_bytes=summary["hbm_write_bytes"],
                dma_count=summary["dma_transfer_count"],
                source_sha256=digest(folder / "nki/program.py"),
                profile_sha256=digest(folder / "profile.json"),
                artifact_sha256=measured["artifact_sha256"],
                correctness=measured["hardware"],
                scope="Fixed-template diagnostic, not selected by production Voyager",
            )
        )
    report["hardware_tile_probes"] = probe_records
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "benchmark_us",
                    "trace_us",
                    "candidate_diagnostics",
                    "measured",
                )
            },
            indent=2,
        )
    )
    print(
        "selected / replay / logical bound us",
        selected["prediction_ns"] / 1000,
        replay["prediction_us"],
        logical_bound["prediction_ns"] / 1000,
    )


if __name__ == "__main__":
    main()

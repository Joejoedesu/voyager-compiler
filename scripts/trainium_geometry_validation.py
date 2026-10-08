"""Re-score authenticated fixed programs, without changing their measured artifacts."""

import argparse, hashlib, json, statistics
from dataclasses import replace
from pathlib import Path
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.instruction_plan import Program
from voyager_compiler.trainium.program_analysis import analyze_selected
from voyager_compiler.trainium.orientation import orientation_graph
from voyager_compiler.trainium.execution import TrainiumTuning
from voyager_compiler.trainium.dependencies import evaluate_graph


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    hw = neuron_core(3)
    root = Path("results/trainium")
    paths = []
    for folder in [
        "weight-layout-2026-10-08/full-auto",
        "weight-layout-2026-10-08/full-k-activations",
        "weight-layout-2026-10-08/small-auto",
        "bounded-buffering-2026-10-08/depth2",
        "bounded-buffering-2026-10-08/depth4",
        "bounded-buffering-2026-10-08/depth8",
        "final-matrix-fp32",
        "final-matrix-bf16",
        "operand-reuse-2026-10-07/heldout-fp32-v2",
        "operand-reuse-2026-10-07/heldout-bf16-v2",
        "orientation-2026-10-08/gemm-activations",
    ]:
        paths.extend(sorted((root / folder).glob("*/result.json")))
    rows = []
    for rp in paths:
        folder = rp.parent
        r = json.loads(rp.read_text())
        ip = folder / "instructions.json"
        sp = folder / "selection.json"
        if not ip.exists():
            continue
        assert r["status"] == "pass"
        assert r["instructions_sha256"] == digest(ip)
        assert r["hardware_sha256"] == digest(folder / "hardware.json")
        assert r["model_sha256"] == digest(folder / "model.txt")
        assert r["reference_sha256"] == digest(
            Path(r.get("reference_path", str(folder / "reference.npz")))
        )
        assert r["program_sha256"] == digest(folder / "nki/program.py")
        s = json.loads(sp.read_text())
        assert s["instructions_sha256"] == digest(ip)
        for f, value in r.get("artifact_sha256", {}).items():
            p = folder / f
            assert digest(p) == value, (p, "hash mismatch")
        prog = Program.load(json.loads(ip.read_text()))
        added = 0
        for previous, current in prog.reuse_edges():
            ins = prog.instructions[current]
            if previous not in ins.dependencies:
                prog.instructions[current] = replace(
                    ins,
                    dependencies=tuple(sorted((*ins.dependencies, previous))),
                )
                added += 1
        prog.validate()
        analysis = analyze_selected(prog, hw)
        times = (
            [x["p50_us"] for x in r["latencies"]]
            if "latencies" in r
            else r.get("p50_us", r.get("latency_us", None))
        )
        if times is None:
            times = r.get("latencies_us")
        if times is None:
            raise ValueError((str(rp), list(r)))
        median = statistics.median(times) if isinstance(times, list) else times
        row = dict(
            case=str(folder),
            instructions_sha256=digest(ip),
            measured_median_us=median,
            selected_prediction_us=analysis["prediction_ns"] / 1000,
            service_ns=analysis["service_ns"],
            matmul_geometry_events=analysis["matmul_geometry_events"],
            historical_missing_reuse_edges_added=added,
        )
        if "weight-layout" in str(folder):
            h = json.loads((folder / "hardware.json").read_text())
            selected = h["row_regions"][0]["selected"]
            choice = selected["matrix_choices"][0]
            tuning = TrainiumTuning(
                matmul_operands="reuse",
                matmul_orientation=choice["orientation"],
            )
            graph, _ = orientation_graph(
                hw,
                choice["m"],
                choice["n"],
                choice["k"],
                32,
                choice["transposed"],
                tuning,
                input_row=choice["input_row"],
                output_row=choice["output_row"],
                weight_layout=choice["weight_layout"],
            )
            row["matrix_tile_us"] = evaluate_graph(graph).duration_ns / 1000
            row["saved_matrix_tile_us"] = (
                next(
                    c["prediction_ns"]
                    for c in choice["candidates"]
                    if c["orientation"] == choice["orientation"]
                )
                / 1000
            )
            row["matrix_choice"] = choice
        rows.append(row)
        print(str(folder), row["selected_prediction_us"], median, flush=True)
    source = Path("src/voyager_compiler/trainium")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(
            dict(
                scope="Fixed measured programs, new model evaluation. Hardware is authenticated reuse, not new execution.",
                sources={
                    str(p): digest(p)
                    for p in [
                        *source.glob("*.py"),
                        source / "timing_trainium2.json",
                    ]
                },
                results=rows,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

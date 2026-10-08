"""Report geometry calibration, fixed-plan validation, and fresh search results."""

import hashlib, json, re, statistics
from pathlib import Path
import numpy as np
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.trainium.instruction_plan import Program
from voyager_compiler.trainium.program_analysis import analyze_selected


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def measured(p):
    r = json.loads((p / "result.json").read_text())
    assert r["status"] == "pass"
    assert r["program_sha256"] == digest(p / "nki/program.py")
    assert r["instructions_sha256"] == digest(p / "instructions.json")
    for f, h in r["artifact_sha256"].items():
        assert digest(p / f) == h
    return statistics.median(x["p50_us"] for x in r["latencies"])


def main():
    root = Path("results/trainium/matmul-geometry-2026-10-08")
    hw = neuron_core(3)
    before = json.loads((root / "before.json").read_text())
    after = json.loads((root / "after.json").read_text())
    cal = json.loads((root / "calibration.json").read_text())
    old = {r["case"]: r for r in before["results"]}
    rows = []
    for r in after["results"]:
        b = old[r["case"]]
        assert b["instructions_sha256"] == r["instructions_sha256"]
        rows.append(
            dict(
                case=r["case"].removeprefix("results/trainium/"),
                group=(
                    "Fused" if "weight-layout" in r["case"] else "Previous GEMM"
                ),
                before_us=b["selected_prediction_us"],
                after_us=r["selected_prediction_us"],
                hardware_us=r["measured_median_us"],
                before_error_pct=100
                * (b["selected_prediction_us"] / r["measured_median_us"] - 1),
                after_error_pct=100
                * (r["selected_prediction_us"] / r["measured_median_us"] - 1),
            )
        )
    selected = []
    for name, mode in [
        ("matmul_add_rmsnorm", "full-k-activations"),
        ("add_rmsnorm_matmul", "full-auto"),
    ]:
        p = root / "full-auto-final" / name
        prior = root.parent / "weight-layout-2026-10-08" / mode / name
        assert digest(p / "nki/program.py") == digest(prior / "nki/program.py")
        with (
            np.load(p / "reference.npz") as a,
            np.load(prior / "reference.npz") as b,
        ):
            assert a.files == b.files and all(
                np.array_equal(a[k], b[k]) for k in a.files
            )
        program = Program.load(
            json.loads((p / "instructions.json").read_text())
        )
        program.validate()
        analysis = analyze_selected(program, hw)
        h = json.loads((p / "hardware.json").read_text())
        s = h["row_regions"][0]["selected"]
        selected.append(
            dict(
                case=name,
                orientation=s["matrix_choices"][0]["orientation"],
                search_us=s["prediction_ns"] / 1000,
                selected_us=analysis["prediction_ns"] / 1000,
                hardware_us=measured(prior),
                hardware_evidence=str(prior),
                measurement="Authenticated unchanged NKI and input arrays; hardware measurement reused",
                source_sha256=digest(p / "nki/program.py"),
            )
        )
    fresh = []
    for n in ["gemm128", "gemm256", "gemm512"]:
        pair = []
        for folder in ["matrix-auto-before", "matrix-auto"]:
            p = root / folder / n
            program = Program.load(
                json.loads((p / "instructions.json").read_text())
            )
            program.validate()
            a = analyze_selected(program, hw)
            h = json.loads((p / "hardware.json").read_text())
            pair.append(
                dict(
                    mode=folder,
                    hardware_us=measured(p),
                    search_us=sum(e["predicted_ns"] for e in h["estimates"])
                    / 1000,
                    new_model_selected_us=a["prediction_ns"] / 1000,
                    matrix_choices=json.loads(
                        (p / "selection.json").read_text()
                    )["matrix_choices"],
                    source_sha256=digest(p / "nki/program.py"),
                )
            )
        fresh.append(
            dict(
                case=n,
                before=pair[0],
                after=pair[1],
                runtime_change_pct=100
                * (pair[1]["hardware_us"] / pair[0]["hardware_us"] - 1),
            )
        )
    oldtile = root / "matrix-oldtile-newmodel/gemm512"
    assert digest(oldtile / "nki/program.py") == digest(
        root / "matrix-auto-before/gemm512/nki/program.py"
    )
    oldtile_search = (
        sum(
            e["predicted_ns"]
            for e in json.loads((oldtile / "hardware.json").read_text())[
                "estimates"
            ]
        )
        / 1000
    )
    gemm512 = next(r for r in fresh if r["case"] == "gemm512")
    ranking_audit = dict(
        old_tile=[128, 256, 512],
        new_tile=[256, 256, 512],
        old_tile_new_model_search_us=oldtile_search,
        new_tile_new_model_search_us=gemm512["after"]["search_us"],
        old_tile_new_model_selected_us=gemm512["before"][
            "new_model_selected_us"
        ],
        new_tile_new_model_selected_us=gemm512["after"][
            "new_model_selected_us"
        ],
        old_tile_hardware_us=gemm512["before"]["hardware_us"],
        new_tile_hardware_us=gemm512["after"]["hardware_us"],
        conclusion="Compact search reverses the ranking that the same model's selected physical ISA audit gets right. Remaining mismatch is candidate realization/overlap accounting, not evidence for fitting another matmul rate.",
    )
    report = dict(
        scope="Device nc_latency on one Trainium2 core, FP32 unless specified. 24 unchanged historical programs re-scored; 17 fresh independent geometry probes; old/new standalone search programs freshly measured. No application latency used for calibration.",
        calibration=cal,
        gemm512_ranking_audit=ranking_audit,
        validation=rows,
        fused_selected=selected,
        fresh_search=fresh,
        limits=[
            "Geometry law characterized for FP32 K=N=128, moving width128..512 and regular strides1/2/4/8/16; other cases retain earlier laws.",
            "Primitive feed regimes are empirical, not a claimed physical SBUF bank topology.",
            "Stream restart uses producer/source-order context, not a finite-queue cycle model; readiness and backend issue ordering remain approximate.",
            "Geometry improves the fused orientation ranking but does not fix all absolute predictions or standalone search rankings.",
            "Selected-ISA prediction and compact search prediction remain separate. BF16 and short-K laws unchanged.",
        ],
    )
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "Trainium matmul geometry validation (2026-10-08)",
        report["scope"],
        "",
        "Fused selected schedules (search / selected ISA / measured, us):",
    ]
    lines += [
        f"{r['case']}: {r['orientation']}, {r['search_us']:.3f} / {r['selected_us']:.3f} / {r['hardware_us']:.3f}"
        for r in selected
    ]
    lines += [
        "",
        "Fresh standalone search (old measured -> new measured, us):",
    ] + [
        f"{r['case']}: {r['before']['hardware_us']} -> {r['after']['hardware_us']} ({r['runtime_change_pct']:+.2f}%)"
        for r in fresh
    ]
    lines += [
        "",
        "Fixed programs (before prediction / after prediction / hardware, us):",
    ] + [
        f"{r['case']}: {r['before_us']:.3f} / {r['after_us']:.3f} / {r['hardware_us']:.3f}"
        for r in rows
    ]
    lines += ["", *report["limits"]]
    (root / "report.txt").write_text("\n".join(lines) + "\n")
    section = """<!-- MATMUL_GEOMETRY_START -->
<section id="matmul-geometry-update" class="panel"><h2>Geometry-aware matmul model · October 8</h2>
<p>Matched hardware probes isolate the cost of operand strides. The model now prices moving and stationary feeds separately, allows their overlap, and distinguishes pipeline startup, steady completion, and accumulator forwarding. Parameters come from isolated primitives; application runtimes were held out.</p>
<div id="geometry-summary"></div>
<details><summary>Model and limits</summary><p>At the characterized FP32 geometry, moving feed = 4M × access factor, stationary feed = 4N × access factor (cycles). The measured access factor is 1 for strides 1–2 and 2 for strides 4–16. Effective issue = max(feeds) + 24 cycles. Steady completion = 432 + max(feeds) + 0.5 × moving feed. Divide by the 2.4 GHz Tensor Engine clock for ns. Startup retains the existing isolated completion bound. These are empirical feed regimes, not a claim about undocumented physical banks.</p><p>Producer writes after the previous matmul and intervening Tensor Engine operations break the modeled stream. Operand retirement and physical memory-reuse edges still wait for completion; only accumulator dependencies use forwarding. Short-K, BF16, unsupported strides, and irregular views retain their previous laws.</p><p>The compact search still differs from the selected physical instruction graph. Backend queues, readiness-dependent pipeline restarts, and non-matmul costs remain approximate. The tables retain prediction errors and regressions.</p></details>
<label for="geometry-filter">Validation group </label><select id="geometry-filter"><option>All</option><option>Fused</option><option>Previous GEMM</option></select>
<div id="geometry-table" style="overflow-x:auto"></div>
<details><summary>Fresh standalone searches and independent probes</summary><div id="geometry-fresh" style="overflow-x:auto"></div><div id="geometry-probes" style="overflow-x:auto"></div></details>
<p>Evidence: <code>voyager-trainium-10-06-isa/results/trainium/matmul-geometry-2026-10-08/report.json</code>. Historical sections below preserve earlier model results.</p></section>
<script id="matmul-geometry-evidence" type="application/json">__DATA__</script>
<script>(()=>{const d=JSON.parse(document.getElementById('matmul-geometry-evidence').textContent);const f=x=>x.toFixed(3);const table=(heads,rows)=>'<table><thead><tr>'+heads.map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody></table>';
document.getElementById('geometry-summary').innerHTML='<div style="overflow-x:auto">'+table(['Selected fused kernel','Orientation','Search ms','Selected ISA ms','Measured ms'],d.fused_selected.map(r=>[r.case,r.orientation,f(r.search_us/1000),f(r.selected_us/1000),f(r.hardware_us/1000)]))+'</div><p>Correct orientation selected for both fused kernels. Whole-kernel accuracy remains uneven; this does not establish reliable ranking across all schedules.</p>';
const select=document.getElementById('geometry-filter');function render(){document.getElementById('geometry-table').innerHTML=table(['Fixed program','Old prediction µs','New prediction µs','Hardware µs','New error %'],d.validation.filter(r=>select.value==='All'||r.group===select.value).map(r=>[r.case,f(r.before_us),f(r.after_us),f(r.hardware_us),r.after_error_pct.toFixed(1)]));}select.addEventListener('change',render);render();
const g=d.gemm512_ranking_audit;document.getElementById('geometry-fresh').innerHTML='<p>Remaining GEMM512 ranking gap: the new compact model scores the old/new tiles at '+f(g.old_tile_new_model_search_us)+' / '+f(g.new_tile_new_model_search_us)+' µs. The same model scores their selected physical ISA at '+f(g.old_tile_new_model_selected_us)+' / '+f(g.new_tile_new_model_selected_us)+' µs, correctly preferring the old tile. Hardware is '+f(g.old_tile_hardware_us)+' / '+f(g.new_tile_hardware_us)+' µs. The remaining search error lies in candidate realization and overlap accounting.</p>'+table(['Fresh search','Old selection hardware µs','New selection hardware µs','Runtime change %'],d.fresh_search.map(r=>[r.case,f(r.before.hardware_us),f(r.after.hardware_us),r.runtime_change_pct.toFixed(1)]));
document.getElementById('geometry-probes').innerHTML=table(['Probe','Split','Issue prediction ns','Measured ns','Error %'],d.calibration.results.map(r=>[r.case,r.split,f(r.predicted_issue_ns),f(r.measured_issue_ns),r.issue_error_pct.toFixed(1)]));})();</script>
<!-- MATMUL_GEOMETRY_END -->
""".replace("__DATA__", json.dumps(report).replace("</", "<\\/"))
    html = Path(
        "/home/ubuntu/ML/AGEN-voyager/trainium-kernel-analysis-2026-10-07.html"
    )
    page = html.read_text()
    old_data = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>',
        page,
        re.S,
    ).group(1)
    page = re.sub(
        r"<!-- MATMUL_GEOMETRY_START -->.*?<!-- MATMUL_GEOMETRY_END -->\n?",
        "",
        page,
        flags=re.S,
    )
    page = page.replace("<main>", "<main>\n" + section, 1)
    assert (
        re.search(
            r'<script id="report-data" type="application/json">(.*?)</script>',
            page,
            re.S,
        ).group(1)
        == old_data
    )
    html.write_text(page)
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()

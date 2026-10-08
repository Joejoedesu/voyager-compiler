"""Authenticate row-region measurements and append the kernel-analysis update."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measured(path):
    result = json.loads((path / "result.json").read_text())
    assert result["status"] == "pass" and result["hardware"]["correct"]
    for filename, expected in result["artifact_sha256"].items():
        assert digest(path / filename) == expected
    assert digest(path / "nki/program.py") == result["program_sha256"]
    assert digest(path / "reference.npz") == result["reference_sha256"]
    selection = json.loads((path / "selection.json").read_text())
    analysis = selection["program_analysis"]["selected_instruction_analysis"]
    program = json.loads((path / "instructions.json").read_text())
    assert len(program["outputs"]) == 1
    record = dict(
        directory=str(path.resolve()),
        median_us=statistics.median(r["p50_us"] for r in result["latencies"]),
        p50_samples_us=[r["p50_us"] for r in result["latencies"]],
        correctness=result["hardware"],
        selected_prediction_us=analysis["prediction_ns"] / 1000,
        incomplete_completion_events=len(analysis["unknown_completion"]),
        hbm_bytes=analysis["hbm_bytes"],
        stats=selection["stats"],
        artifact_sha256=result["artifact_sha256"],
        source_sha256=result["program_sha256"],
        compiler_version=result["compiler_version"],
        compiler_flags=result["compiler_flags"],
        semantic_outputs=1,
        hbm_workspace_outputs=0,
    )
    profile = path / "profile.json"
    if profile.exists():
        data = json.loads(profile.read_text())
        summary = data["summary"][0]
        assert (
            summary["hbm_read_bytes"] + summary["hbm_write_bytes"]
            == analysis["hbm_bytes"]
        )
        record["profile"] = dict(
            sha256=digest(profile),
            trace_us=summary["total_time"] * 1e6,
            hbm_read_bytes=summary["hbm_read_bytes"],
            hbm_write_bytes=summary["hbm_write_bytes"],
            spill_save_bytes=summary["spill_save_bytes"],
            spill_reload_bytes=summary["spill_reload_bytes"],
        )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    args = parser.parse_args()
    root = Path("results/trainium/row-regions-2026-10-08")
    rows = []
    for case, folder, old_us, prior_us in [
        ("add_rmsnorm_matmul", "full-reuse", 8318, 4114),
        ("matmul_add_rmsnorm", "full-fixed", 17647, 8466),
    ]:
        record = measured(root / folder / case)
        final = root / "final" / case
        record["final_equivalence"] = {}
        for filename in (
            "model.txt",
            "instructions.json",
            "nki/program.py",
            "reference.npz",
        ):
            assert (final / filename).read_bytes() == (
                root / folder / case / filename
            ).read_bytes()
            record["final_equivalence"][filename] = digest(final / filename)
        hardware = json.loads((final / "hardware.json").read_text())
        record["region_selection"] = hardware["row_regions"][0]
        old = (
            Path("results/trainium/operand-reuse-2026-10-07/shared-full-v2")
            / case
        )
        before = json.loads((old / "selection.json").read_text())
        rows.append(
            dict(
                case=case,
                previous_voyager_us=old_us,
                prior_work_us=prior_us,
                previous_hbm_bytes=before["program_analysis"][
                    "selected_instruction_analysis"
                ]["hbm_bytes"],
                **record,
            )
        )
    orders = []
    for path in sorted((root / "heldout-fixed").iterdir()):
        if not path.is_dir():
            continue
        source = (
            path
            if (path / "result.json").exists()
            else root / "heldout" / path.name
        )
        assert (path / "nki/program.py").read_bytes() == (
            source / "nki/program.py"
        ).read_bytes()
        assert (path / "instructions.json").read_bytes() == (
            source / "instructions.json"
        ).read_bytes()
        orders.append(
            dict(
                order=path.name.split("_"),
                shape=dict(M=192, N=256, K=128),
                **measured(source),
            )
        )
    assert len(orders) == 6
    pool = json.loads(
        Path(
            "results/trainium/maxpool-gap-2026-10-08/analysis.json"
        ).read_text()
    )
    regression = (root / "regression-final.log").read_text()
    assert "162 passed" in regression and "23 subtests passed" in regression
    report = dict(
        date="2026-10-08",
        full=rows,
        heldout_orders=orders,
        maxpool=pool,
        validation="162 tests and 23 subtests passed; two small and two full application kernels plus all six rectangular orderings pass hardware",
        limits=[
            "Opt-in row_regions; default per-kernel flow preserved",
            "One 2-D GEMM per discovered region; FP32; invariant matrix must fit SBUF",
            "Whole feature/reduction axes; exact-divisor row tiles <=128; no ragged tail search",
            "Same-shape pointwise operands; broadcast tensor edges, padding/view boundaries and pre-fused submodules may break regions",
            "No streaming of an oversized invariant weight, multi-GEMM region, softmax/attention, alternate matrix orientation, or joint region/unfused comparison",
            "Search template omits final physical reuse; selected model retains missing reduce/activation completion laws. Timing accuracy is not established by correct HBM accounting.",
        ],
    )
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    body = []
    labels = ["Residual → RMSNorm → GEMM", "GEMM → residual → RMSNorm"]
    for label, row in zip(labels, rows):
        body.append(
            f'<tr><td>{label}</td><td>{row["previous_voyager_us"]/1000:.3f}</td><td><b>{row["median_us"]/1000:.3f}</b></td><td>{row["prior_work_us"]/1000:.3f}</td><td>{row["previous_hbm_bytes"]/1e6:.1f} → {row["hbm_bytes"]/1e6:.1f}</td></tr>'
        )
    timing_rows = []
    for label, row in zip(labels, rows):
        timing_rows.append(
            f'<tr><td>{label}</td><td>{row["region_selection"]["selected"]["prediction_ns"]/1e6:.3f}</td><td>{row["selected_prediction_us"]/1000:.3f}</td><td>{row["median_us"]/1000:.3f}</td><td>{row["incomplete_completion_events"]}</td></tr>'
        )
    options = "".join(
        f'<option value="{i}">{" → ".join(r["order"])}</option>'
        for i, r in enumerate(orders)
    )
    probes = "".join(
        f'<tr><td>{p["tile_rows"]}</td><td>{p["median_us"]/1000:.3f}</td><td>{p["compiled_prediction_us"]/1000:.3f}</td><td>{p["dma_count"]}</td></tr>'
        for p in pool["hardware_tile_probes"]
    )
    encoded = json.dumps(report).replace("</", "<\\/")
    section = r"""<!-- ROW_REGION_UPDATE_START -->
<section id="row-region-update">
<h2>8 October update: maxpool coverage and general row bufferization</h2>
<p><span class="badge measured">Hardware validated</span> <span class="badge policy">Opt-in shared pass</span> Latest results below supersede the older per-kernel timings for these two fused workloads. FP32, Trainium2, one core; median of three p50 measurements, 10 warmups and 100 iterations.</p>
<div class="card"><h3>Maxpool misses a faster legal tile</h3>
<p>The production pool tiler only enumerates exact divisors. For 4,094 output rows, 89 is its largest legal row tile below the 128-partition limit. A controlled 128-row version with a 126-row final tile runs in <b>0.729 ms</b> versus <b>1.016 ms</b> for the 89-row control. Both are exactly correct. Prior work takes 1.358 ms.</p>
<table><thead><tr><th>Diagnostic row tile</th><th>Hardware ms</th><th>Compiled replay ms</th><th>DMA commands</th></tr></thead><tbody>POOL_ROWS</tbody></table>
<p>Both move exactly <b>268,271,632 bytes</b> from/to HBM. More partitions participate in each command; the 128-row form needs fewer DMA and vector commands. It uses the same four-max expansion, two halo buffers, and 16 MiB explicit SBUF arena. These are fixed-template diagnostics, not newly selected production kernels.</p>
<p><b>Classification:</b> schedule representation/enumeration restriction, not the model rejecting the winning tile. Even the current serial template ranks 128 rows ahead of 89. Masked/ragged pool tiling is still missing. Rolling halo reuse is another absent alternative: both tested forms reload three vertically overlapping strips.</p>
<details><summary>Why the maxpool predictions differ</summary><p>The selected instruction model predicts 1.321 ms; compiled replay predicts 1.074 ms; hardware measures 1.016 ms. Removing physical-reuse edges diagnostically gives 1.077 ms. This localizes the plan/replay discrepancy to conservative whole-buffer reuse ordering, but is not permission to remove those edges without proving byte-range lifetimes. The tile-search template also serializes successive tiles: 2.045 ms plus 9.403 µs fixed overhead = 2.054 ms. A distance-two recurrence diagnostic gives 1.175 ms. That diagnostic is not a complete corrected two-slot model.</p><p>Compiled replay overpredicts hardware by 5.7%. Modeled DMA service is 1.048 ms versus approximately 0.986 ms DMA-active time in the trace; these are related accounting views, not identical measurement boundaries. No timing coefficients were fitted.</p></details></div>
<div class="card"><h3>One bufferizer, six operation orders</h3>
<p>The new shared pass proves a common independent row axis from operand roles. Pointwise operations preserve that axis; RMSNorm keeps its entire feature axis inside the tile; GEMM maps <code>(rows,K) × (K,N)</code> to <code>(rows,N)</code>. External invariant weights load once, tile intermediates stay in SBUF, and only the region output is stored to HBM. The pass does not match application names or maintain six kernel recipes.</p>
<p>Use <code>trainium_advanced.py --row-regions --matmul-operands reuse</code>, or <code>BufferizationOptions(row_regions=True)</code>. Shared exact-divisor enumeration calls the target resource/cost hook, then the existing pipeline scheduler constructs buffers, copies and waits. Layout conversion and bounded activation-panel reuse are selected before formal NKI emission. Strict mode still assigns addresses in Voyager.</p>
<label for="row-order-select">Inspect a hardware-tested ordering: </label><select id="row-order-select">ORDER_OPTIONS</select><pre id="row-order-code"></pre><p id="row-order-result"></p>
<table><thead><tr><th>Full workload</th><th>Previous Voyager ms</th><th>Row-region ms</th><th>Prior work ms</th><th>HBM MB</th></tr></thead><tbody>FULL_ROWS</tbody></table>
<p>Both full kernels use 128-row tiles and pass the same tolerance checks. Profiles agree with the selected HBM counts and show no spills. Keeping converted activation panels across output-column blocks was essential: without that reuse, residual→RMSNorm→GEMM took 13.861 ms despite eliminating HBM intermediates.</p>
<details><summary>Prediction accuracy and the remaining GEMM-first gap</summary>
<table><thead><tr><th>Workload</th><th>Search ms</th><th>Selected ISA ms</th><th>Hardware ms</th><th>Unknown completion events</th></tr></thead><tbody>TIMING_ROWS</tbody></table>
<p>The search sums existing boundary and stage templates; final physical reuse is only exposed in selected-ISA analysis. Reduce/activation completion laws remain incomplete. These estimates are not claimed as an accurate joint search model.</p>
<p>The prior GEMM-first kernel uses a 128-row × 512-column result tile. It puts activation columns on the stationary operand and weight columns on the 512-wide moving operand, producing the row layout needed by RMSNorm. Voyager currently fixes the opposite operand orientation and uses 128×128 result panels here: 8,192 logical GEMMs versus 2,048 in the prior algorithm, plus local layout conversions. The invariant weight payload is retained in SBUF, but its panel transposes are repeated per row tile. These are confirmed structural differences; a controlled orientation experiment is still needed to apportion their timing effects.</p></details>
<details><summary>Scope, validation and reproducible evidence</summary><ul>LIMITS</ul><p>VALIDATION. Default GEMM128/512 bufferized collateral and emitted NKI remain byte-identical to the preserved comparator. ImageNet/Llama regressions were waived. CPU simulation was not used.</p><p>Code: <code>codegen/transform/bufferize/row_regions.py</code>, <code>trainium/row_regions.py</code>, and early layout edges in <code>trainium/planning.py</code>. Reports: <code>results/trainium/row-regions-2026-10-08/report.json</code> and <code>results/trainium/maxpool-gap-2026-10-08/analysis.json</code>.</p><button id="row-evidence-download">Download update evidence</button><pre id="row-evidence-json"></pre></details></div>
</section>
<script id="row-region-evidence" type="application/json">EVIDENCE_JSON</script>
<script>(()=>{const d=JSON.parse(document.getElementById('row-region-evidence').textContent),s=document.getElementById('row-order-select');function draw(){const r=d.heldout_orders[+s.value];let width=128;const lines=['retain W[128,256] and gamma in SBUF','for rows in [0:96, 96:192]:','  x = load input[rows,:]'];for(const op of r.order){if(op==='gemm'){lines.push('  x = GEMM_local(x, W)    # keep K whole');width=256}else if(op==='norm')lines.push('  x = RMSNorm_local(x, gamma)    # reduce all '+width+' features');else lines.push('  x = x + load residual[rows,:]    # '+width+' features')}lines.push('  store output[rows,:] = x','  # all intermediate x values stay in SBUF');document.getElementById('row-order-code').textContent=lines.join('\n');document.getElementById('row-order-result').textContent='Hardware correctness passed · '+r.median_us+' µs · 192×128 input, 128×256 weight · two 96-row tiles · no HBM workspace output.'}s.addEventListener('change',draw);draw();document.getElementById('row-evidence-json').textContent=JSON.stringify(d,null,2);document.getElementById('row-evidence-download').onclick=()=>{const a=document.createElement('a'),u=URL.createObjectURL(new Blob([JSON.stringify(d,null,2)],{type:'application/json'}));a.href=u;a.download='trainium-row-regions-maxpool-2026-10-08.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}})();</script>
<!-- ROW_REGION_UPDATE_END -->
"""
    for key, value in dict(
        POOL_ROWS=probes,
        ORDER_OPTIONS=options,
        FULL_ROWS="".join(body),
        TIMING_ROWS="".join(timing_rows),
        LIMITS="".join(
            "<li>" + html.escape(x) + "</li>" for x in report["limits"]
        ),
        VALIDATION=html.escape(report["validation"]),
        EVIDENCE_JSON=encoded,
    ).items():
        section = section.replace(key, value)
    section = section.replace('<table>', '<div style="overflow-x:auto"><table>').replace('</table>', '</table></div>')
    page = args.html.read_text()
    old_data = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>',
        page,
        re.S,
    ).group(1)
    page = re.sub(
        r"<!-- ROW_REGION_UPDATE_START -->.*?<!-- ROW_REGION_UPDATE_END -->\n?",
        "",
        page,
        flags=re.S,
    )
    page = page.replace("<main>", "<main>\n" + section, 1)
    nav = '<a href="#row-region-update">8 Oct: pool &amp; row regions</a>'
    if nav not in page:
        page = page.replace("<nav>", "<nav>" + nav, 1)
    assert (
        re.search(
            r'<script id="report-data" type="application/json">(.*?)</script>',
            page,
            re.S,
        ).group(1)
        == old_data
    )
    assert page.count('id="row-region-update"') == 1
    args.html.write_text(page)
    print(args.html)
    for row in rows:
        print(row["case"], row["median_us"], row["hbm_bytes"])


if __name__ == "__main__":
    main()

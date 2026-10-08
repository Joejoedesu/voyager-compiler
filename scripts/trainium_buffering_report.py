"""Report a measured bounded-buffer sweep without fitting hardware parameters."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
from trainium_allocation_report import record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--html", type=Path, required=True)
    a = p.parse_args()
    rows = []
    for case in ["gemm128", "gemm256", "gemm512"]:
        baseline = a.baseline / ("strict-verified" if case == "gemm256" else "strict") / case
        old = record(baseline)
        for f in ["model.txt", "nki/program.py"]:
            assert (a.root / "depth1" / case / f).read_bytes() == (baseline / f).read_bytes()
        old.update(case=case, depth=1, measurement="Authenticated previous hardware run; default source/model unchanged")
        rows.append(old)
        original = json.loads((baseline / "instructions.json").read_text())
        for depth in [2, 4, 8, 16]:
            root = a.root / f"depth{depth}" / case
            if not root.exists():
                continue
            r = record(root)
            current = json.loads((root / "instructions.json").read_text())
            assert r["hashes"]["model.txt"] == old["hashes"]["model.txt"]
            assert len(original["instructions"]) == len(current["instructions"])
            for x, y in zip(original["instructions"], current["instructions"]):
                assert {k:v for k,v in x.items() if k != "dependencies"} == {k:v for k,v in y.items() if k != "dependencies"}
            assert {k:v for k,v in original["placements"].items() if v["memory"] == "PSUM"} == {k:v for k,v in current["placements"].items() if v["memory"] == "PSUM"}
            with np.load(root / "reference.npz") as x, np.load(baseline / "reference.npz") as y:
                assert all(np.array_equal(x[k], y[k]) for k in x.files)
            r.update(case=case, depth=depth, measurement="Fresh hardware run")
            rows.append(r)
    ranking = []
    for case in ["gemm128", "gemm256", "gemm512"]:
        candidates = [r for r in rows if r["case"] == case]
        predicted = min(candidates, key=lambda r:(r["selected_prediction_us"], r["sbuf_reserved_bytes"]))
        measured = min(candidates, key=lambda r:(r["median_us"], r["sbuf_reserved_bytes"]))
        ranking.append(dict(case=case, model_ranked_depth=predicted["depth"], best_measured_depth=measured["depth"], measured_regret=predicted["median_us"] / measured["median_us"] - 1))
    report = dict(
        scope="One-core FP32 Trainium2; staged operands; identical software tiles, instruction operands/order and PSUM placement. Three p50 repeats, 100 timed iterations after 10 warmups. CPU simulator not run.",
        knob="--temporary-buffer-depth N (positive integer; default 1)",
        semantics="1 preserves existing best-fit allocation. N>1 rotates oldest-free whole slots per byte-size class, with at most N times the minimum source-live slot count, capped by the number of values. Live values cannot overlap; reuse adds completion edges. Physical SBUF capacity remains 28 MiB. This is a pool multiplier, not N total buffers or N guaranteed simultaneous panels.",
        rows=rows, ranking=ranking,
        limitations=["Depth is applied to the already selected logical program; joint software-tile/buffer-depth search is not integrated.", "The model underestimates several bounded configurations and cannot reliably distinguish near-tied candidates.", "Size-class pools are a heuristic and can reserve more storage than an operation-aware pool.", "The six full application kernels were not rerun.", "Strict default depth 1 remains unchanged; native allocation rejects an explicit Voyager pool depth above 1."],
        validation="118 focused tests; 33 shared tests and 23 subtests. Default model.txt and NKI source are byte-identical for all three GEMMs. The bounded-pool unit test verifies fixed storage across 16 versus 160 iterations and all physical reuse edges remain explicit. Full default model suite not rerun; no shared production source changed.",
    )
    a.root.mkdir(parents=True, exist_ok=True)
    (a.root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [report["scope"], "", report["knob"], report["semantics"], "", "case       depth   SBUF_MiB   model_us   hardware_us"]
    for r in rows:
        lines.append(f'{r["case"]:10} {r["depth"]:5} {r["sbuf_reserved_bytes"]/1048576:10.4f} {r["selected_prediction_us"]:10.3f} {r["median_us"]:12.1f}')
    lines += ["", "Retrospective model ranking among measured candidates (not a claim of joint mapping search):", json.dumps(ranking, indent=2), "", *report["limitations"], "", report["validation"]]
    (a.root / "analysis.txt").write_text("\n".join(lines) + "\n")
    data = json.dumps(report).replace("</", "<\\/")
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bounded temporary buffering</title><style>body{font:16px/1.6 system-ui;background:#f3f6fa;color:#17283b;margin:0}main{max-width:1000px;margin:auto;padding:28px 18px}section{background:white;border:1px solid #d6e0ed;border-radius:12px;margin:20px 0;padding:22px}select,button{font:inherit;padding:7px;background:white;border:1px solid #97aac0;border-radius:5px}table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}.scroll{overflow-x:auto}svg{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}small{color:#4b5f77}code{background:#edf2f8;padding:2px 4px}a{color:#185caa}</style><main><p>Trainium2 · 8 October 2026 · <a href="trainium-gemm-allocation-2026-10-08.html">Original allocation investigation</a></p><h1>Tuning bounded temporary buffering</h1><p>The compiler can trade SBUF footprint for overlap by changing <code>--temporary-buffer-depth</code>. All settings below retain the same selected operations and PSUM banks.</p><section><h2>Measured tradeoff</h2><label>Kernel <select id="case"><option>gemm128</option><option>gemm256</option><option selected>gemm512</option></select></label><svg id="chart" viewBox="0 0 900 360" role="img" aria-label="Latency by buffer depth"></svg><div class="scroll"><table><thead><tr><th>Depth</th><th>SBUF MiB</th><th>Plan estimate µs</th><th>Hardware µs</th></tr></thead><tbody id="rows"></tbody></table></div><p id="selection"></p><small>Blue bars: measured latency. Black markers: selected-plan estimates. Each fresh setting passes hardware correctness. Three p50 repeats; 100 iterations after 10 warmups. Depth 1 reuses authenticated earlier results after checking that the regenerated source is byte-identical.</small></section><section><h2>What the depth means</h2><p>Depth 1 preserves existing placement. For depth N above 1, each byte-size class gets a reusable slot pool capped at N times its minimum source-live slot count. The compiler selects the oldest free slot and preserves completion dependencies before its next overwrite.</p><p>This is a <b>pool multiplier</b>, not N buffers total. Pool capacity is bounded independently of loop iterations for a fixed set of tile shapes and live-value requirements. Larger pools must still fit the physical 28 MiB SBUF. PSUM placement is unchanged.</p><pre>PYTHONPATH=src python scripts/trainium_generate.py \
  --output OUT --cases gemm128 gemm256 gemm512 \
  --temporary-buffer-depth 8</pre><p>Also available in <code>trainium_advanced.py</code>. The knob is recorded in compiler policy and restored during conversion. It requires strict ISA realization; the native-allocation path owns its own local storage.</p></section><section><h2>Search and model limits</h2><p>The knob currently controls physical allocation for the selected software schedule. It does not yet participate jointly in Interstellar's software-tile search. The model underestimates several bounded settings, and tiny predicted differences should not be treated as reliable hardware rankings.</p><p>The size-class policy is a starting heuristic. Different data-movement roles can eventually use different pool depths. Payload retention is independent: these experiments all use staged operands.</p><p>Validation: 118 focused tests, 33 shared tests and 23 subtests. Default source remains byte-identical. The six full application kernels were not rerun and depth 1 remains the default.</p><button id="download">Download evidence JSON</button><details><summary>Detailed evidence and retrospective ranking</summary><pre id="evidence"></pre></details></section></main><script id="data" type="application/json">DATA</script><script>const data=JSON.parse(document.querySelector('#data').textContent);function draw(){const k=document.querySelector('#case').value,rs=data.rows.filter(r=>r.case===k),max=Math.max(...rs.map(r=>Math.max(r.median_us,r.selected_prediction_us)))*1.14;document.querySelector('#rows').replaceChildren();let svg='';rs.forEach((r,i)=>{const tr=document.createElement('tr');for(const t of [r.depth,(r.sbuf_reserved_bytes/1048576).toFixed(4),r.selected_prediction_us.toFixed(2),r.median_us]){const td=document.createElement('td');td.textContent=t;tr.append(td)}document.querySelector('#rows').append(tr);const y=18+i*67,w=580*r.median_us/max,x=180+580*r.selected_prediction_us/max;svg+=`<text x="0" y="${y+21}">Depth ${r.depth}</text><rect x="180" y="${y}" width="${w}" height="30" fill="#287da6" rx="3"/><line x1="${x}" x2="${x}" y1="${y-4}" y2="${y+34}" stroke="#17283b" stroke-width="3"/><text x="${190+w}" y="${y+21}">${r.median_us} µs</text><text x="180" y="${y+48}" font-size="12">${(r.sbuf_reserved_bytes/1048576).toFixed(3)} MiB SBUF</text>`});document.querySelector('#chart').innerHTML=svg;const rank=data.ranking.find(r=>r.case===k);document.querySelector('#selection').textContent=`Best measured depth: ${rank.best_measured_depth}. Model-ranked depth: ${rank.model_ranked_depth}. Measured regret within this candidate set: ${(100*rank.measured_regret).toFixed(1)}%.`};document.querySelector('#case').onchange=draw;draw();document.querySelector('#evidence').textContent=JSON.stringify(data,null,2);document.querySelector('#download').onclick=()=>{const a=document.createElement('a'),u=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.href=u;a.download='bounded-buffering-evidence.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};</script></html>'''.replace('DATA</script>', data + '</script>')
    a.html.write_text(html)
    print("\n".join(lines))


if __name__ == "__main__":
    main()

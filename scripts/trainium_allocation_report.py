"""Authenticate allocation probes and produce an offline interactive report."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics
import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    result = json.loads((path / "result.json").read_text())
    assert result["status"] == "pass" and result["hardware"]["correct"]
    assert digest(path / "nki/program.py") == result["program_sha256"]
    for filename, expected in result["artifact_sha256"].items():
        assert digest(path / filename) == expected, (path, filename)
    assert digest(path / "instructions.json") == result["instructions_sha256"]
    assert digest(path / "reference.npz") == result["reference_sha256"]
    selection = json.loads((path / "selection.json").read_text())
    program = json.loads((path / "instructions.json").read_text())
    placements = [p for p in program["placements"].values() if p["memory"] == "SBUF"]
    samples = [r["p50_us"] for r in result["latencies"]]
    return dict(
        directory=str(path.resolve()), p50_samples_us=samples,
        median_us=statistics.median(samples), correctness=result["hardware"],
        candidate_prediction_us=result["estimates"][0]["predicted_ns"] / 1000,
        selected_prediction_us=selection["program_analysis"]["selected_instruction_analysis"]["prediction_ns"] / 1000,
        selected_scope=selection["program_analysis"]["selected_instruction_analysis"]["scope"],
        sbuf_reserved_bytes=max((p["byte_address"] + p["bytes_per_partition"] for p in placements), default=0) * 128 if placements else None,
        compiler_version=result["compiler_version"], compiler_flags=result["compiler_flags"],
        software_tile=result["estimates"][0]["software_tile"],
        hashes={f: digest(path / f) for f in ["model.txt", "instructions.json", "nki/program.py", "reference.npz", "file.neff", "profile.ntff"]},
    )


def profile(path):
    p = json.loads((path / "profile.json").read_text())
    ins, summary = p["instruction"], p["summary"][0]
    loads = sorted((i for i in ins if i["opcode"] == "DMA_DIRECT2D" and "src_table_index=0 " in i["operands"]), key=lambda i: i["timestamp"])[:8]
    gaps = [b["timestamp"] - a["timestamp"] for a, b in zip(loads, loads[1:])]
    return dict(
        profile_sha256=digest(path / "profile.json"),
        trace_us=summary["total_time"] * 1e6,
        opcode_counts=dict(Counter(i["opcode"] for i in ins)),
        hbm_read_bytes=summary["hbm_read_bytes"], hbm_write_bytes=summary["hbm_write_bytes"],
        dram_spill_bytes=sum(x["amount_bytes"] for x in p["nc_mem_usage"] if x["usage_type"] == "DRAM Spill"),
        first_eight_weight_dma=[{k: i[k] for k in ["timestamp", "duration", "operands"]} for i in loads],
        weight_issue_gaps_ns=gaps, median_weight_issue_gap_ns=statistics.median(gaps),
        regular_matmul_strides=dict(Counter(re.search(r'src=\w+@0x[0-9a-f]+\[([^]]+)\]', i["operands"]).group(1) for i in ins if i["opcode"] == "MATMUL" and i["instruction_type"] == "REGULAR")),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    cases = []
    for case in ["gemm128", "gemm256", "gemm512"]:
        strict = root / ("strict-verified" if case == "gemm256" else "strict") / case
        paths = dict(strict=strict, distinct=root / "distinct" / case, native=root / "native" / case)
        records = {name: record(path) for name, path in paths.items()}
        assert len({r["hashes"]["model.txt"] for r in records.values()}) == 1
        with np.load(strict / "reference.npz") as reference:
            for path in paths.values():
                with np.load(path / "reference.npz") as other:
                    assert set(reference.files) == set(other.files)
                    assert all(np.array_equal(reference[k], other[k]) for k in reference.files)
        a, b = [json.loads((paths[k] / "instructions.json").read_text()) for k in ["strict", "distinct"]]
        assert len(a["instructions"]) == len(b["instructions"])
        for x, y in zip(a["instructions"], b["instructions"]):
            assert {k: v for k, v in x.items() if k != "dependencies"} == {k: v for k, v in y.items() if k != "dependencies"}
        assert {k:v for k,v in a["placements"].items() if v["memory"] == "PSUM"} == {k:v for k,v in b["placements"].items() if v["memory"] == "PSUM"}
        cases.append(dict(case=case, identical_bufferized_program=True, identical_inputs=True, strict_distinct_instruction_semantics_and_psum_placements_equal=True, variants=records))
    profiles = {name: profile(root / name / "gemm512") for name in ["strict", "distinct", "native"]}
    for opcode in ["MATMUL", "LDWEIGHTS", "COPY", "MEMSET", "DMA_DIRECT2D"]:
        assert profiles["strict"]["opcode_counts"][opcode] == profiles["distinct"]["opcode_counts"][opcode]
    for field in ["hbm_read_bytes", "hbm_write_bytes", "regular_matmul_strides", "dram_spill_bytes"]:
        assert profiles["strict"][field] == profiles["distinct"][field]
    report = dict(
        date="2026-10-08", scope="Fresh Trainium2 one-core FP32 device benchmarks, same software tiles and inputs. Three p50 repeats, 100 iterations and 10 warmups each. CPU simulator not run.",
        cases=cases, gemm512_profiles=profiles,
        conclusion="SBUF address reuse introduces write-after-read waits and is the dominant regression. Distinct SBUF preserves operations and PSUM placement and recovers most of the performance; native allocation also removes clears and folds copies.",
        limitations=["Trace duration differs from benchmark nc_latency; do not equate their boundaries.", "Distinct storage is a diagnostic, not a bounded scalable allocation policy.", "Native allocation predictions do not certify physical placement, folding, reuse or spills.", "Six full application kernels were not rerun with the new flag.", "No performance timing coefficients were fitted or changed in this experiment."],
        next_step="Search bounded physical temporary generations and include their reuse edges before ranking mappings. Keep payload retention independent of address recycling.",
        reproduction=["PYTHONPATH=src python scripts/trainium_generate.py --output OUT/native --cases gemm128 gemm256 gemm512 --no-strict-realization", "PYTHONPATH=src python scripts/trainium_allocation_probe.py --source STRICT --output OUT/distinct --cases gemm128 gemm256 gemm512", "NEURON_RT_VISIBLE_CORES=0 NKI_PYTHON scripts/trainium_run_hardware.py --artifacts OUT/native --repeats 3"],
    )
    (root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [report["scope"], "", "Case         strict_us  distinct_SBUF_us  native_us"]
    for case in cases:
        v = case["variants"]
        lines.append(f'{case["case"]:12} {v["strict"]["median_us"]:9.1f} {v["distinct"]["median_us"]:17.1f} {v["native"]["median_us"]:10.1f}')
    lines += ["", report["conclusion"], "", "GEMM512 strict versus distinct: 1 MiB versus 11.5625 MiB SBUF; 137.325 versus 44.654 us selected-DAG prediction; identical 128 MATMUL, 160 COPY, 72 MEMSET, 33 DMA; identical HBM traffic; no spills.", "Weight DMA median spacing: 3695 ns versus 599 ns; distinct addresses remove ScalarE reader waits.", "Native GEMM512: 136 COPY, no MEMSET; 128 MATMUL and 33 DMA. HBM identity is 16 KiB instead of 64 KiB.", "", "Remaining: " + report["next_step"], "", *report["limitations"], "", "Validation: 110 focused tests; 33 shared tests plus 23 subtests. Strict GEMM128/512 emitted NKI and model.txt byte-identical to pre-flag baseline. BF16 GEMM128 hardware correctness passed (15 us, one repeat). Full default model regression suite not rerun; no shared production files changed."]
    (root / "analysis.txt").write_text("\n".join(lines) + "\n")
    data = json.dumps(report).replace("</", "<\\/")
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GEMM allocation investigation</title>
<style>body{font:16px/1.6 system-ui;margin:0;background:#f3f6fa;color:#16253a}main{max-width:1000px;margin:auto;padding:32px 20px}h1{line-height:1.2}section{background:white;border:1px solid #d8e1ed;border-radius:12px;padding:22px;margin:20px 0}table{border-collapse:collapse;width:100%}th,td{padding:10px;text-align:left;border-bottom:1px solid #ddd}select,button{font:inherit;padding:7px;border:1px solid #9aabc0;border-radius:6px;background:white}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}svg{width:100%;height:auto}small{color:#50617a}.table{overflow-x:auto}code{background:#edf2f8;padding:2px 4px}a{color:#185caa}</style>
<main><p>Trainium2 · 8 October 2026 · measured on hardware</p><h1>Address reuse caused the GEMM regression</h1><p>GEMM512 falls from <b>169 to 44 µs</b> by assigning distinct SBUF storage, with the same selected operations and PSUM placement. Native allocation reaches <b>42 µs</b>.</p>
<section><h2>Compare the three kernels</h2><label>Kernel <select id="kernel"><option>gemm128</option><option>gemm256</option><option selected>gemm512</option></select></label> <label><input type="checkbox" id="prediction"> Show selected-plan predictions</label><svg id="chart" viewBox="0 0 900 230" role="img" aria-label="Kernel execution latency comparison"></svg><div class="table"><table><thead><tr><th>Kernel</th><th>Strict</th><th>Distinct SBUF</th><th>Native allocation</th></tr></thead><tbody id="rows"></tbody></table></div><small>Device nc_latency, median of three p50 repeats, 100 iterations after 10 warmups. Same FP32 inputs and bufferized software schedule in each row. All pass correctness. Native plan predictions do not include the compiler's final allocation and folding.</small></section>
<section><h2>Controlled GEMM512 experiment</h2><p>Only SBUF placements and the resulting physical reuse dependencies change. Reserved SBUF grows from <b>1 MiB to 11.5625 MiB</b>, within the 28 MiB capacity. Instruction order, operands and PSUM banks stay fixed.</p><div class="table"><table><thead><tr><th>Native command</th><th>Strict</th><th>Distinct SBUF</th><th>Native allocation</th></tr></thead><tbody id="counts"></tbody></table></div><p>The weight DMA issue spacing changes from <b>3.695 µs to 0.599 µs</b>. Independent loads use distinct addresses, so they no longer wait for ScalarE to finish reading a reused buffer. Strict and distinct variants transfer the same HBM bytes and neither spills.</p><p>The selected-DAG model changes from 137.325 to 44.654 µs. The compact mapping-search prediction does not include the eventual placement-induced serialization; this becomes visible too late, in the selected-plan audit.</p><small>Trace durations (169.868, 53.419 and 50.428 µs) have different boundaries/conditions from benchmark nc_latency. The timing comparison above uses only benchmark measurements.</small></section>
<section><h2>Using relaxed realization</h2><p><code>--no-strict-realization</code> is available in <code>trainium_generate.py</code> and <code>trainium_advanced.py</code>. Shared compilation still chooses the software tiles, layouts and operand policy. Logical SBUF/PSUM buffers are allocated by the native compiler. Strict mode remains the default.</p><p>The pinned SDK rejects our explicit identity-matmul transpose under native allocation. Relaxed mode selects native TensorE <code>nc_transpose</code>, whose implicit identity DMA is recorded separately. This comparison includes native instruction folding as well as allocation: it removes clears and folds some copies.</p><p>Hardware coverage: three FP32 GEMMs and BF16 GEMM128. The six full application kernels have not been rerun. Movement-chain search and <code>--no-isa</code> are rejected with this flag.</p></section>
<section><h2>What remains</h2><p>Search a bounded number of temporary buffer generations and score the resulting reuse dependencies before selecting a mapping. Fully distinct storage is useful evidence, but its footprint grows with the kernel. Payload reuse is a separate choice; every comparison here uses the staged operand policy.</p><p>No timing coefficients were changed. Validation: 110 focused tests, 33 shared tests and 23 subtests. Default strict GEMM128/512 source stays byte-identical.</p><button id="download">Download evidence JSON</button><details><summary>Commands and detailed evidence</summary><pre id="evidence"></pre></details></section></main>
<script id="data" type="application/json">DATA</script><script>
const data=JSON.parse(document.querySelector('#data').textContent),names=['strict','distinct','native'],labels=['Strict','Distinct SBUF','Native allocation'],colors=['#ab4057','#307aa4','#25816b'];
for(const row of data.cases){const tr=document.createElement('tr');for(const text of [row.case,...names.map(n=>row.variants[n].median_us+' µs')]){const td=document.createElement('td');td.textContent=text;tr.append(td)}document.querySelector('#rows').append(tr)}
for(const op of ['MATMUL','COPY','MEMSET','DMA_DIRECT2D']){const tr=document.createElement('tr');for(const text of [op,...names.map(n=>data.gemm512_profiles[n].opcode_counts[op]||0)]){const td=document.createElement('td');td.textContent=text;tr.append(td)}document.querySelector('#counts').append(tr)}
function draw(){const c=data.cases.find(c=>c.case===document.querySelector('#kernel').value),show=document.querySelector('#prediction').checked,max=Math.max(...names.map(n=>c.variants[n].median_us),...(show?names.map(n=>c.variants[n].selected_prediction_us):[]))*1.15;let svg='';names.forEach((n,i)=>{const v=c.variants[n],y=25+i*70,w=620*v.median_us/max;svg+=`<text x="0" y="${y+20}" font-size="16">${labels[i]}</text><rect x="170" y="${y}" height="30" width="${w}" fill="${colors[i]}" rx="3"/><text x="${180+w}" y="${y+20}">${v.median_us} µs</text>`;if(show){const x=170+620*v.selected_prediction_us/max;svg+=`<line x1="${x}" x2="${x}" y1="${y-5}" y2="${y+35}" stroke="#17283b" stroke-width="3"/><text x="170" y="${y+52}" font-size="12">Plan estimate: ${v.selected_prediction_us.toFixed(2)} µs${n==='native'?' (logical only)':''}</text>`}});document.querySelector('#chart').innerHTML=svg}document.querySelector('#kernel').onchange=draw;document.querySelector('#prediction').onchange=draw;draw();document.querySelector('#evidence').textContent=JSON.stringify(data,null,2);document.querySelector('#download').onclick=()=>{const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.href=url;a.download='gemm-allocation-evidence.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script></html>'''.replace('DATA</script>', data + '</script>')
    args.html.write_text(html)
    print("\n".join(lines))
    print(args.html)


if __name__ == "__main__":
    main()

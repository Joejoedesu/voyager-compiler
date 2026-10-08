"""Authenticate joint weight-layout/orientation experiments and extend the HTML."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from trainium_orientation_report import collect


def detail(path):
    data = collect(path)
    selection = json.loads((path / "selection.json").read_text())
    hardware = json.loads((path / "hardware.json").read_text())
    chosen = hardware["row_regions"][0]["selected"]
    data["weight_layout"] = chosen.get("weight_layout", "generic")
    data["weight_contract"] = chosen.get("weight_contract")
    data["layout_candidates"] = [
        dict(
            layout=x["weight_layout"],
            legal=x["legal"],
            reason=x.get("reason"),
            sbuf_bytes=x.get("sbuf_bytes"),
            search_prediction_us=x.get("prediction_ns", 0) / 1000,
            matrix_choices=x.get("matrix_choices", []),
        )
        for x in chosen.get("layout_candidates", [])
    ]
    names = set(selection.get("k_partitioned_weight_buffers", []))
    data["weight_load_events"] = [
        x
        for x in selection["events"]
        if x["kind"] == "copy" and x["dst"] in names
    ]
    if names:
        assert len(names) == len(data["weight_load_events"]) == 1
        assert (
            data["weight_load_events"][0]["sizes"]
            == data["weight_contract"]["logical_shape"]
        )
        assert all(
            x["weight_layout"] == "k_partitioned"
            for x in selection["matrix_choices"]
        )
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/trainium/weight-layout-2026-10-08"),
    )
    parser.add_argument("--html", type=Path, required=True)
    args = parser.parse_args()
    previous = args.root.parent / "orientation-2026-10-08/full-auto"
    cases = []
    for name, label, prior in (
        ("matmul_add_rmsnorm", "GEMM → residual → RMSNorm", 8466),
        ("add_rmsnorm_matmul", "Residual → RMSNorm → GEMM", 4114),
    ):
        old, auto, alternate = (
            detail(p / name)
            for p in (
                previous,
                args.root / "full-auto",
                args.root / "full-k-activations",
            )
        )
        refs = [
            np.load(p / name / "reference.npz")
            for p in (
                previous,
                args.root / "full-auto",
                args.root / "full-k-activations",
            )
        ]
        try:
            assert all(x.files == refs[0].files for x in refs)
            assert all(
                np.array_equal(refs[0][key], r[key])
                for key in refs[0].files
                for r in refs[1:]
            )
        finally:
            for ref in refs:
                ref.close()
        assert old["hbm_bytes"] == auto["hbm_bytes"] == alternate["hbm_bytes"]
        cases.append(
            dict(
                name=name,
                label=label,
                prior_work_us=prior,
                before=old,
                auto=auto,
                alternate=alternate,
                speedup=old["median_us"] / auto["median_us"],
                selection_regret=auto["median_us"]
                / min(auto["median_us"], alternate["median_us"])
                - 1,
            )
        )
    heldout = {
        d.name: detail(d)
        for d in sorted((args.root / "heldout-blocks").iterdir())
        if d.is_dir()
    }
    assert len(heldout) == 6
    regression = (args.root / "regression-final.log").read_text()
    assert "171 passed" in regression and "23 subtests passed" in regression
    defaults = {}
    for name in ("gemm128", "gemm512"):
        a = args.root / "default-check" / name
        b = args.root.parent / "row-regions-2026-10-08/default-check" / name
        defaults[name] = {
            f: hashlib.sha256((a / f).read_bytes()).hexdigest()
            for f in ("model.txt", "instructions.json", "nki/program.py")
        }
        assert all(
            (a / f).read_bytes() == (b / f).read_bytes() for f in defaults[name]
        )
    report = dict(
        cases=cases,
        heldout=heldout,
        default_equivalence=defaults,
        validation="171 tests + 23 subtests passed; full candidates and six multi-block operation orders pass real Trainium2 correctness checks",
        limits=[
            "New layout eligibility: whole invariant private 2-D FP32 row-region weight, one matrix consumer; whole buffer must fit SBUF.",
            "Rounded K-partition storage is charged independently; a tested capacity case rejects it and retains generic layout.",
            "Saved bindings, contiguous full-load geometry, compatible buffer uses, and final physical placement are checked before emission.",
            "Multiple GEMMs, oversized streamed weights, arbitrary alias/view layouts, and padding-crossing region fusion remain outside this extension.",
            "M=192/N=320/K=192 probes encounter existing padding boundaries and fall back; not claimed as K-layout hardware coverage.",
            "The performance model still misranks the two K-layout orientations for GEMM-first. No application-fitted timing coefficient was added.",
        ],
    )
    (args.root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    section = r"""<!-- WEIGHT_LAYOUT_UPDATE_START -->
<section id="weight-layout-update" class="card" style="margin-bottom:24px">
<h2>Joint weight layout and operand orientation · 8 October</h2>
<p>The fixed N-partitioned weight restriction is removed for eligible invariant row-region weights. Search can load the existing representation or DMA directly into K-partitioned SBUF, then retain that payload across all row tiles. Orientation remains a separate choice. No new timing rate was fitted.</p>
<label for="weight-layout-case">Kernel </label><select id="weight-layout-case"></select>
<div id="weight-layout-results"></div>
<details><summary>Data movement, reuse, and conservative checks</summary>
<pre style="overflow-x:auto">Existing: HBM W → N-partitioned SBUF → repeated local transpose/assembly → TensorE
New:      HBM W → K-partitioned SBUF → direct panel views across row tiles → TensorE</pre>
<p>The new physical layout has at most 128 K partitions and packs K blocks along the free dimension: W[k,n] maps to [k mod P, floor(k/P)·N+n], P=min(128,K). The DMA writes this representation directly. It replaces the old weight buffer rather than retaining a second converted copy. Buffer generations and shared load/wait lifetimes remain explicit.</p>
<p>Search prices each layout's DMA rectangles, matrix preparation, output conversion, and partition-rounded storage. K padding can increase storage; an automated capacity test verifies fallback to the old layout when only it fits. The ISA planner checks saved bindings and full contiguous loads, and physical SBUF/PSUM allocation remains checked. Aliased or conflicting consumers are not silently reinterpreted.</p>
<p>CLI: <code>--row-regions --matmul-operands reuse --matmul-weight-layout auto|generic|k_partitioned --matmul-orientation auto|weights|activations</code>. Both searches default to auto. Ordinary GEMM128/GEMM512 model, ISA, and NKI files remain byte-identical to the preceding implementation.</p>
</details>
<details><summary>Validation and remaining limitations</summary><p id="weight-layout-validation"></p></details>
</section>
<script id="weight-layout-evidence" type="application/json">__DATA__</script>
<script>
(()=>{const data=JSON.parse(document.getElementById('weight-layout-evidence').textContent);
const select=document.getElementById('weight-layout-case');
for(const c of data.cases){const o=document.createElement('option');o.value=c.name;o.textContent=c.label;select.appendChild(o);}
const ms=x=>(x/1000).toFixed(3);
function render(){const c=data.cases.find(x=>x.name===select.value);
const rows=[['Before: fixed weight layout',c.before],['Auto: K layout / '+c.auto.orientation,c.auto],['K layout / forced activations',c.alternate]];
document.getElementById('weight-layout-results').innerHTML='<div style="overflow-x:auto"><table><thead><tr><th>Candidate</th><th>Search (ms)</th><th>Selected ISA (ms)</th><th>Hardware (ms)</th><th>GEMM calls</th><th>Transposes</th></tr></thead><tbody>'+rows.map(([label,r])=>'<tr><td>'+label+'</td><td>'+ms(r.search_prediction_us)+'</td><td>'+ms(r.selected_prediction_us)+'</td><td>'+ms(r.median_us)+'</td><td>'+r.matrix_calls+'</td><td>'+r.stats.isa_transposes+'</td></tr>').join('')+'</tbody></table></div><p>Automatic-mode speedup: '+c.speedup.toFixed(2)+'×. Prior-work reference: '+ms(c.prior_work_us)+' ms. Selection regret among the two measured K-layout candidates: '+(100*c.selection_regret).toFixed(1)+'%. All candidates are numerically correct; hardware values are median p50 over three runs, excluding host invocation. The older search prediction is the preserved historical model.</p><p>HBM payload remains '+c.auto.hbm_bytes.toLocaleString()+' bytes. One invariant weight load feeds all row tiles; the model has not claimed fewer bytes by omitting an intermediate.</p>';
}
select.addEventListener('change',render);render();
document.getElementById('weight-layout-validation').textContent=data.validation+'. Held-out shape: M=192, N=512, K=256, with two 96-row tiles. '+data.limits.join(' ')+' Evidence: voyager-trainium-10-06-isa/results/trainium/weight-layout-2026-10-08/report.json.';
})();
</script>
<!-- WEIGHT_LAYOUT_UPDATE_END -->
""".replace("__DATA__", json.dumps(report).replace("</", "<\\/"))
    page = args.html.read_text()
    old = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>',
        page,
        re.S,
    ).group(1)
    page = re.sub(
        r"<!-- WEIGHT_LAYOUT_UPDATE_START -->.*?<!-- WEIGHT_LAYOUT_UPDATE_END -->\n?",
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
        == old
    )
    args.html.write_text(page)
    for c in cases:
        print(
            c["name"],
            c["auto"]["median_us"],
            c["alternate"]["median_us"],
            c["selection_regret"],
        )


if __name__ == "__main__":
    main()

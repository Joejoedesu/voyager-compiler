"""Authenticate the controlled orientation experiment and update the HTML report."""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from trainium_row_region_report import measured


def collect(path):
    result = measured(path)
    evidence = json.loads((path / "result.json").read_text())
    for filename, field in (
        ("model.txt", "model_sha256"),
        ("hardware.json", "hardware_sha256"),
        ("instructions.json", "instructions_sha256"),
    ):
        assert (
            hashlib.sha256((path / filename).read_bytes()).hexdigest()
            == evidence[field]
        )
    selection = json.loads((path / "selection.json").read_text())
    choices = selection["matrix_choices"]
    result["orientation"] = choices[0]["orientation"]
    result["matrix_calls"] = sum(
        next(
            x["matmul_calls"]
            for x in c["candidates"]
            if x["orientation"] == c["orientation"]
        )
        for c in choices
    )
    result["matrix_tile"] = {
        k: choices[0][k] for k in ("m", "n", "k", "input_row", "output_row")
    }
    result["local_candidates"] = [
        {k: v for k, v in c.items() if k != "unknown_completion"}
        | {"unknown_completion_events": len(c["unknown_completion"])}
        for c in choices[0]["candidates"]
    ]
    hardware = json.loads((path / "hardware.json").read_text())
    regions = hardware["row_regions"]
    if regions:
        selected = regions[0]["selected"]
        result["search_prediction_us"] = selected["prediction_ns"] / 1000
        result["search_sbuf_bytes"] = selected["sbuf_bytes"]
    result["sbuf_reserved_bytes"] = selection["temporary_buffering"][
        "sbuf_reserved_bytes"
    ]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("results/trainium/orientation-2026-10-08"),
    )
    p.add_argument("--html", type=Path, required=True)
    args = p.parse_args()
    cases = []
    for name, label, prior in (
        ("matmul_add_rmsnorm", "GEMM → residual → RMSNorm", 8466),
        ("add_rmsnorm_matmul", "Residual → RMSNorm → GEMM", 4114),
    ):
        auto = collect(args.root / "full-auto" / name)
        alternate = collect(args.root / "full-activations" / name)
        with np.load(args.root / "full-auto" / name / "reference.npz") as a, np.load(
            args.root / "full-activations" / name / "reference.npz"
        ) as b:
            assert a.files == b.files
            assert all(np.array_equal(a[key], b[key]) for key in a.files)
        cases.append(
            dict(
                name=name,
                label=label,
                prior_work_us=prior,
                auto=auto,
                alternate=alternate,
                selection_regret=auto["median_us"]
                / min(auto["median_us"], alternate["median_us"])
                - 1,
            )
        )
    heldout = {
        d.name: collect(d)
        for d in sorted((args.root / "heldout-activations").iterdir())
        if d.is_dir()
    }
    assert len(heldout) == 6
    regression = (args.root / "regression-final.log").read_text()
    assert "167 passed" in regression and "23 subtests passed" in regression
    report = dict(
        date="2026-10-08",
        cases=cases,
        heldout=heldout,
        rectangular=collect(args.root / "gemm-activations/gemm_custom"),
        bf16=collect(args.root / "bf16-whole-k/gemm_custom"),
        validation="167 tests + 23 subtests; real Trainium2 outputs; no CPU simulator",
        limitations=[
            "The two orientations retain the existing input/weight SBUF layouts. Weight-layout and invariant-conversion hoisting are not yet search choices.",
            "Local orientation ranking uses dependency templates before final physical allocation; the shared mapping candidate retains conservative storage allowances and final ISA capacity checks.",
            "Row-region estimates serialize boundaries/stages and retain conservative norm layout charges; they are not calibrated full-kernel latency predictions.",
            "Convolution keeps its existing lowering. Row regions remain FP32 with one 2-D GEMM and whole feature axes.",
            "BF16 external K reductions failed the same CPU reference check for both orientations (32/32768 elements); whole-K BF16 is validated separately.",
        ],
    )
    (args.root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    section = r"""<!-- ORIENTATION_UPDATE_START -->
<section id="orientation-update" class="card" style="margin-bottom:24px">
<h2>TensorE operand orientation search · 8 October</h2>
<p>Trainium now searches weight-stationary (M≤512, N≤128) and activation-stationary (M≤128, N≤512) panels, both with K≤128. The selected orientation is recorded before instruction emission. HBM software tiles and shared buffer lifetimes still come from Voyager.</p>
<label for="orientation-case">Kernel comparison </label><select id="orientation-case"></select>
<div id="orientation-results"></div>
<details><summary>How the choice is costed and constrained</summary>
<p>Existing primitive timing laws price operand preparation, matmul, PSUM eviction, and output layout conversion. Reuse caches are local to a software GEMM invocation. Matrix candidates retain the shared SBUF certificate (including converted weights for reuse and a 4 MiB default temporary reserve), three reserved PSUM banks, and final physical placement checks. No new fitted rate was added.</p>
<p>Activation-stationary produces M×N directly for a row reduction. With the current N-partitioned row-region weight buffer, it must first transpose N≤128 fragments and copy them into a K×N≤512 moving panel. A generic N-partitioned result requires an explicit conversion back. Native compilation retains engine scheduling; strict mode still uses Voyager-assigned addresses.</p>
<p>The two orientations share the existing weight layout. The prior kernel’s direct K-partitioned weight loads and cross-row conversion reuse are additional choices still missing from this search. Search-template timing and selected-ISA timing remain distinct, and neither is a substitute for device measurement.</p>
<p>CLI: <code>--matmul-orientation auto|weights|activations</code>. Auto is the Trainium default; forcing one mode gives a controlled comparison. Convolution retains its current lowering.</p>
</details>
<details><summary>Validation and evidence</summary><div id="orientation-validation"></div></details>
</section>
<script id="orientation-evidence" type="application/json">__DATA__</script>
<script>
(()=>{
const data=JSON.parse(document.getElementById('orientation-evidence').textContent);
const select=document.getElementById('orientation-case');
for(const c of data.cases){const o=document.createElement('option');o.value=c.name;o.textContent=c.label;select.appendChild(o);}
const ms=x=>(x/1000).toFixed(3);
function render(){const c=data.cases.find(x=>x.name===select.value);
const rows=[['Auto ('+c.auto.orientation+')',c.auto],['Forced activation-stationary',c.alternate]];
document.getElementById('orientation-results').innerHTML='<div style="overflow-x:auto"><table><thead><tr><th>Candidate</th><th>Search (ms)</th><th>Selected ISA (ms)</th><th>Hardware (ms)</th><th>GEMM calls</th><th>Transposes</th></tr></thead><tbody>'+rows.map(([label,r])=>'<tr><td>'+label+'</td><td>'+ms(r.search_prediction_us)+'</td><td>'+ms(r.selected_prediction_us)+'</td><td>'+ms(r.median_us)+'</td><td>'+r.matrix_calls+'</td><td>'+r.stats.isa_transposes+'</td></tr>').join('')+'</tbody></table></div><p>Prior-work reference: '+ms(c.prior_work_us)+' ms. Measured selection regret across these two candidates: '+(100*c.selection_regret).toFixed(1)+'%. Hardware values are median p50 over three runs, excluding host invocation. Both candidates pass numerical checks.</p>';
}
select.addEventListener('change',render);render();
document.getElementById('orientation-validation').textContent=data.validation+'. Six operation orderings at M=192, N=256, K=128 pass with activation-stationary panels, along with a rectangular FP32 GEMM and a BF16 whole-K GEMM. Evidence: voyager-trainium-10-06-isa/results/trainium/orientation-2026-10-08/report.json. '+data.limitations.join(' ');
})();
</script>
<!-- ORIENTATION_UPDATE_END -->
""".replace("__DATA__", json.dumps(report).replace("</", "<\\/"))
    page = args.html.read_text()
    original = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>',
        page,
        re.S,
    ).group(1)
    page = re.sub(
        r"<!-- ORIENTATION_UPDATE_START -->.*?<!-- ORIENTATION_UPDATE_END -->\n?",
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
        == original
    )
    args.html.write_text(page)
    for c in cases:
        print(
            c["name"],
            "auto",
            c["auto"]["median_us"],
            "alternate",
            c["alternate"]["median_us"],
            "regret",
            c["selection_regret"],
        )


if __name__ == "__main__":
    main()

"""Summarize authenticated Trainium ISA fusion measurements."""
import argparse
import hashlib
import html
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('--html',type=Path)
    a=ap.parse_args(); root=a.root.resolve()
    rows=[]
    for p in sorted(root.glob('*/result.json')):
        d=json.loads(p.read_text())
        if d['status']!='pass': raise ValueError(f'Failed probe: {p}')
        for name,digest in d['artifact_sha256'].items():
            assert hashlib.sha256((p.parent/name).read_bytes()).hexdigest()==digest,(p,name)
        profile=json.loads((p.parent/'profile.json').read_text())
        ops={'COPY','TENSOR_TENSOR','TENSOR_SCALAR','ACTIVATE','SCALAR_TENSOR_TENSOR'}
        ins=[i for i in profile['instruction'] if i['opcode'] in ops]
        d['compute_span_us']=(max(i['timestamp']+i['duration'] for i in ins)-min(i['timestamp'] for i in ins))/1000
        d['compute_opcodes']={k:v for k,v in d['opcode_counts'].items() if k in ops}
        d['directory']=p.parent.name
        rows.append(d)
    assert len(rows)==12,len(rows)
    pairs=[]
    for kind,width in sorted({(d['kind'],d['width']) for d in rows}):
        plain=next(d for d in rows if d['kind']==kind and d['width']==width and not d['fused'])
        fused=next(d for d in rows if d['kind']==kind and d['width']==width and d['fused'])
        import numpy as np
        with np.load(root/plain['directory']/'reference.npz') as x, np.load(root/fused['directory']/'reference.npz') as y:
            assert set(x.files)==set(y.files)
            for k in x.files: assert np.array_equal(x[k],y[k]),(kind,width,k)
        actual0=np.load(root/plain['directory']/'actual.npy');actual1=np.load(root/fused['directory']/'actual.npy')
        pairs.append(dict(kind=kind,width=width,separate=plain,fused=fused,speedup=plain['median_us']/fused['median_us'],latency_reduction_pct=100*(1-fused['median_us']/plain['median_us']),compute_span_reduction_pct=100*(1-fused['compute_span_us']/plain['compute_span_us']),pair_max_abs_error=float(np.max(np.abs(actual0-actual1)))))
    notes=[
        'Fresh real Trainium2 measurements, one visible NeuronCore (0), FP32, 128 partitions, contiguous free axis. Pinned compiler 2.22.12471.0+b4a00d10; --target=trn2 --lnc=1.',
        'Each kernel contains 32 dependent steps. Reported device latency is the median of three p50 measurements, each with 10 warmup and 100 timed iterations. Host invocation is excluded. Correctness and all timing repeats reuse the same native binary.',
        'Compute span is first selected arithmetic/copy start to last completion in the final hardware trace. It includes instruction dependencies and, for PSUM/add, a common SBUF-to-PSUM refill per step. It is not a sum of instruction durations or a standalone throughput measurement.',
        'PSUM/add models the epilogue, with a common copy into PSUM to generate each dependent input. It contains no GEMM; benefit in a complete GEMM pipeline remains to be measured.',
        'RMS final scaling starts with an already expanded full-shape gamma tile. Its broadcast/expansion cost is excluded equally from both variants. Width 1024 matches the current RMS feature width.',
        'Scale/rsqrt width 1 matches per-row RMS statistics; width 128 tests a larger activation tile. Epsilon is a preloaded [128,1] vector in both variants. Their gamma/unused input DMA is optimized away equally.',
        'Repeated arithmetic amplifies local latency differences; it is not an actual repeated-normalization application. Accuracy uses atol=rtol=5e-5; per-pair outputs are also compared.',
        'These observations are candidates for search and characterization, not production timing constants. No compiler model or default fusion policy was changed.',
        'Initial root-level and validated/ attempts contain harness artifact-retention errors and are excluded. Only measured/ results are used here.'
    ]
    report=dict(pairs=pairs,notes=notes)
    (root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['Trainium ISA fusion experiment — 2026-10-08','']+notes+['','kind | shape | separate us | fused us | latency reduction | compute span separate/fused us | pair max abs error']
    for d in pairs:
        s,f=d['separate'],d['fused']
        lines.append(f"{d['kind']} | 128x{d['width']} | {s['median_us']:.3f} | {f['median_us']:.3f} | {d['latency_reduction_pct']:.1f}% | {s['compute_span_us']:.3f}/{f['compute_span_us']:.3f} | {d['pair_max_abs_error']:.3g}")
        lines.append(f"  native arithmetic/copy counts: {s['compute_opcodes']} -> {f['compute_opcodes']}")
    lines+=['','Reproduce:','  /home/ubuntu/ML/.venv-nki/bin/python scripts/trainium_fusion_probe.py --output results/trainium/isa-fusion-2026-10-08/new-run','  /home/ubuntu/ML/AGEN-voyager/.venv-compiler/bin/python scripts/trainium_fusion_report.py results/trainium/isa-fusion-2026-10-08/new-run --html /tmp/trainium-isa-fusion.html']
    (root/'analysis.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    if a.html:
        body=[]
        for d in pairs:
            s,f=d['separate'],d['fused']
            body.append(f"<tr><td>{d['kind']}</td><td>128 × {d['width']}</td><td>{s['median_us']:.3f}</td><td>{f['median_us']:.3f}</td><td>{d['latency_reduction_pct']:.1f}%</td><td>{s['compute_span_us']:.3f} → {f['compute_span_us']:.3f}</td><td>{d['pair_max_abs_error']:.3g}</td></tr>")
        page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Trainium ISA fusion measurements</title><style>body{font:16px/1.55 system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#182c42;background:#f6f8fc}h1{line-height:1.15}table{border-collapse:collapse;width:100%;background:white}th,td{padding:12px;text-align:left;border-bottom:1px solid #dae1ec}th{background:#eaf0f8}.scroll{overflow:auto}summary,button{cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}button{padding:10px;margin:16px 0}</style><h1>Trainium ISA fusion: real hardware results</h1><p>2026-10-08 · FP32 · one Trainium2 NeuronCore · 32 dependent steps per kernel</p><p>All 12 variants passed hardware correctness. These are local fusion microbenchmarks, not full application speedups.</p><div class="scroll"><table><thead><tr><th>Sequence</th><th>Tile</th><th>Separate µs</th><th>Fused µs</th><th>Latency reduction</th><th>Compute span µs</th><th>Pair max error</th></tr></thead><tbody>'''+''.join(body)+'''</tbody></table></div><h2>Interpretation and scope</h2><ul>'''+''.join('<li>'+html.escape(n)+'</li>' for n in notes)+'''</ul><details><summary>Native instructions, timing repeats, and authenticated artifact hashes</summary><pre id="details"></pre></details><button id="download">Download evidence JSON</button><script id="data" type="application/json">'''+json.dumps(report).replace('<','\\u003c')+'''</script><script>const raw=document.getElementById('data').textContent;document.getElementById('details').textContent=JSON.stringify(JSON.parse(raw),null,2);document.getElementById('download').onclick=()=>{const u=URL.createObjectURL(new Blob([raw],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download='trainium-isa-fusion-evidence.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};</script></html>'''
        a.html.write_text(page)

if __name__=='__main__':main()

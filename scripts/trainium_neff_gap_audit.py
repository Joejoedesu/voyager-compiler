"""Diagnostic NEFF/profile attribution; never feeds measured timing to search.

Replays a saved compiled-static graph, then performs explicitly labeled oracle
latency substitutions to locate missing costs. All original evidence is read-only.
"""
import argparse
import collections
import csv
import gzip
from dataclasses import replace
import hashlib
import heapq
import json
from pathlib import Path
import re
import statistics

from voyager_compiler.trainium import compiled_analysis as ca
from voyager_compiler.trainium.hardware import neuron_core
from voyager_compiler.codegen.transform.tiling.execution import evaluate_graph, Dependency, OperationEvent


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def timeline(graph):
    """Same ready-resource policy as shared evaluate_graph; expose event starts."""
    nodes=graph.nodes
    assert graph.repetitions==1
    successors=[[] for _ in nodes];pending=[];ready_at=[0.]*len(nodes)
    why=[None]*len(nodes)
    for dest,n in enumerate(nodes):
        pending.append(len(n.dependencies))
        for d in n.dependencies:successors[d.source].append((dest,d.milestone))
    future={};eligible={}
    for i,n in enumerate(nodes):
        future.setdefault(n.resource,[]);eligible.setdefault(n.resource,[])
        if not pending[i]:heapq.heappush(future[n.resource],(0.,i))
    available={};last={};starts=[0.]*len(nodes);ends=[0.]*len(nodes);critical=[None]*len(nodes)
    for _ in nodes:
        choices=[]
        for resource,waiting in future.items():
            now=available.get(resource,0.)
            while waiting and waiting[0][0]<=now:
                _,i=heapq.heappop(waiting);heapq.heappush(eligible[resource],i)
            if eligible[resource]:choices.append((now,eligible[resource][0],resource,True))
            elif waiting:choices.append((waiting[0][0],waiting[0][1],resource,False))
        start,i,res,is_ready=min(choices)
        heapq.heappop(eligible[res] if is_ready else future[res])
        n=nodes[i];starts[i]=start;ends[i]=start+n.offset('result')
        critical[i]=((last[res],'resource',max(nodes[last[res]].issue_ns,nodes[last[res]].occupancy_ns)) if res in last and available[res]>ready_at[i] else why[i])
        available[res]=start+max(n.issue_ns,n.occupancy_ns);last[res]=i
        for dest,milestone in successors[i]:
            when=start+n.offset(milestone)
            if when>ready_at[dest]:ready_at[dest]=when;why[dest]=(i,milestone,n.offset(milestone))
            pending[dest]-=1
            if not pending[dest]:heapq.heappush(future[nodes[dest].resource],(ready_at[dest],dest))
    assert abs(max(ends)-evaluate_graph(graph).duration_ns)<1e-5
    return starts,ends,critical


def stat(xs):
    xs=sorted(xs)
    if not xs:return None
    return dict(count=len(xs),min=min(xs),median=statistics.median(xs),mean=statistics.mean(xs),p90=xs[int((len(xs)-1)*.9)],max=max(xs),sum=sum(xs))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('case',type=Path);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--compare',type=Path,nargs='*',default=[]);a=ap.parse_args()
    root=a.case;out=a.output;out.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((root/'extraction_provenance.json').read_text())
    for name,value in provenance['artifact_sha256'].items():assert digest(root/name)==value,(name,'artifact hash mismatch')
    assert digest(root/'profile.json')==provenance['profile_uncompressed_sha256']
    assert digest(root/'compiled_static.json')==provenance['compiled_static_sha256']
    profile=json.loads((root/'profile.json').read_text());data=json.loads((root/'compiled_static.json').read_text())
    prediction,graph=ca.predict(data);saved=json.loads((root/'compiled_prediction.json').read_text())
    assert abs(prediction['prediction_us']-saved['prediction_us'])<1e-6
    hw=neuron_core(3);fixed=hw.timing_profile.fixed_kernel_ns
    starts,ends,crit=timeline(graph)
    # Reconstruct the predictor's deterministic groups, never dynamic order.
    seen={}
    for ins in data['instructions']:
        key=ins['subgroup'],ins['compiler_pc']
        if key not in seen or 'hbm_read_bytes' in ins or 'hbm_write_bytes' in ins:seen[key]=ins
    groups=collections.defaultdict(list)
    for i in sorted(seen.values(),key=lambda x:(x['subgroup'],x['compiler_pc'])):
        key=('tensor',i['raw_bir_id']) if i['opcode'] in ('MATMUL','LDWEIGHTS') else (i['subgroup'],i['compiler_pc'])
        groups[key].append(i)
    units=[sorted(g,key=lambda x:x['compiler_pc']) for g in groups.values()]
    units.sort(key=lambda g:(g[0]['subgroup'],g[0]['compiler_pc']))
    measured=collections.defaultdict(list)
    for i in profile['instruction']:
        if i.get('raw_bir_id') and 'compiler_pc' in i:measured[i['subgroup'],i['compiler_pc']].append(i)
    aggs=collections.defaultdict(list)
    for d in profile['dma']:
        if d.get('aggregated')=='yes':aggs[int(re.search(r'S\[(\d+)\]',d['semaphore_id'])[1])].append(d)
    for ds in aggs.values():ds.sort(key=lambda d:d['timestamp'])
    updates=collections.defaultdict(dict)
    for e in profile['semaphore_update']:
        m=re.match(r'S\[(\d+)\]',e['id'])
        if m:updates[int(m[1])][e['value']]=e['timestamp']
    dma_index=collections.Counter();actual={};unit_info=[]
    byuid=collections.defaultdict(list)
    for idx,n in enumerate(graph.nodes):
        m=re.match(r'u(\d+)_',n.name)
        if m:byuid[int(m[1])].append(idx)
    for uid,g in enumerate(units):
        ms=[]
        for ins in g:
            opts=measured[ins['subgroup'],ins['compiler_pc']]
            opts=[x for x in opts if x['opcode']==ins['opcode'] and x.get('raw_bir_id')==ins['raw_bir_id']]
            if len(opts)>1:opts=[x for x in opts if 'hbm_read_bytes' in x or 'hbm_write_bytes' in x]
            assert len(opts)==1,(uid,ins,opts)
            ms.extend(opts)
        begin=min(x['timestamp'] for x in ms);end=max(x['timestamp']+x['duration'] for x in ms)
        op=g[-1]['opcode'];ids=byuid[uid];category=graph.nodes[ids[-1]].implementation or 'control'
        dma=None
        if op=='DMA_DIRECT2D' or g[-1].get('compiler_opcode')=='PSEUDO_DMA_TRIGGER':
            ins=g[-1]
            sem=ca.number(ins.get('compiler_operands',ins['operands']),'semaphore') if op=='DMA_DIRECT2D' else int(re.search(r'S\[(\d+)\]',data['static_dma'][0]['semaphore_id'])[1])
            rank=dma_index[sem];dma_index[sem]+=1;dma=aggs[sem][rank]
            assert dma['timestamp']>=begin,(uid,'DMA precedes trigger')
            notify=updates[sem][16*(rank+1)]
            assert notify>=dma['timestamp']+dma['duration']-5,(uid,'bad notification')
            category='DMA.store' if ins.get('hbm_write_bytes') else 'DMA.load'
            actual[ids[0]]=(begin,dma['timestamp'])
            actual[ids[1]]=(dma['timestamp'],notify)
            end=notify
        else:
            assert len(ids)==1
            actual[ids[0]]=(begin,end)
        unit_info.append(dict(uid=uid,category=category,engine=g[0]['subgroup'],pc=g[0]['compiler_pc'],last_pc=g[-1]['compiler_pc'],bir=g[-1]['raw_bir_id'],opcode=op,measured_start_ns=begin,measured_end_ns=end,measured_span_ns=end-begin,model_start_ns=starts[ids[0]],model_end_ns=ends[ids[-1]],model_latency_ns=sum(graph.nodes[i].offset('result') for i in ids),model_occupancy_ns=sum(graph.nodes[i].occupancy_ns for i in ids),node_ids=ids,operands=' '.join(x['operands'] for x in g)))
    assert sum(dma_index.values())==data['dma_audit']['aggregate_count']
    # Profile-conditioned replay is a diagnostic oracle, NOT a new prediction.
    kinds=['DMA','matmul','transpose','copy','other_compute','control']
    def kind(n):
        if n.resource in ('DMAIssue','DMA'):return 'DMA'
        if n.implementation.startswith('nki.matmul.'):return 'matmul'
        if 'transpose' in n.implementation:return 'transpose'
        if 'copy.' in n.implementation:return 'copy'
        if n.resource.startswith(('Control.','Semaphore.')):return 'control'
        return 'other_compute'
    def oracle(selected):
        return replace(graph,nodes=tuple(replace(n,latency_ns=actual[i][1]-actual[i][0]) if i in actual and kind(n) in selected else n for i,n in enumerate(graph.nodes)))
    variants={}
    for k in kinds+['all']:
        gg=oracle(set(kinds) if k=='all' else {k});variants[k]=(evaluate_graph(gg).duration_ns+fixed)/1000
    # Measured-time readiness audit: no inferred queue/port penalties.
    observed={};residuals=collections.defaultdict(list)
    for i,n in enumerate(graph.nodes):
        if i in actual:observed[i]=actual[i]
        else:
            assert n.resource.startswith('Semaphore.')
            t=max(observed[d.source][1] for d in n.dependencies);observed[i]=(t,t)
    for i,n in enumerate(graph.nodes):
        if i not in actual:continue
        available=[]
        for d in n.dependencies:
            s,e=observed[d.source]
            available.append(s+graph.nodes[d.source].issue_ns if d.milestone=='issue' else e)
        t=max(available,default=0)
        residuals[kind(n)].append(actual[i][0]-t)
    groupsum=collections.defaultdict(list)
    for row in unit_info:groupsum[row['category']].append(row)
    stats={k:dict(count=len(v),observed_span_ns=stat([x['measured_span_ns'] for x in v]),modeled_latency_ns=stat([x['model_latency_ns'] for x in v]),span_difference_ns=stat([x['measured_span_ns']-x['model_latency_ns'] for x in v])) for k,v in groupsum.items()}
    # Timed phase markers are corresponding instruction IDs, not guessed windows.
    markers=[]
    for label,pred in [('first_add',lambda r:r['opcode']=='TENSOR_TENSOR'),('first_reduce',lambda r:r['opcode']=='TENSOR_REDUCE'),('last_rms_activation',lambda r:r['opcode']=='ACTIVATE'),('first_main_gemm',lambda r:r['category']=='nki.matmul.float32' and r['opcode']=='MATMUL' and '128*128' in r['operands'])]:
        rs=[r for r in unit_info if pred(r)];r=(max if label.startswith('last') else min)(rs,key=lambda r:r['measured_start_ns'])
        markers.append(dict(label=label,uid=r['uid'],actual_us=r['measured_start_ns']/1000,model_us=(r['model_start_ns']+fixed)/1000))
    main_gemm=[r for r in unit_info if r['category']=='nki.matmul.float32' and r['operands'].rstrip().endswith('128*128')]
    spacing=[]
    for left,right in zip(main_gemm,main_gemm[1:]):
        if right['pc']==left['last_pc']+1:
            spacing.append(dict(left_uid=left['uid'],right_uid=right['uid'],observed_ns=right['measured_start_ns']-left['measured_start_ns'],modeled_ns=right['model_start_ns']-left['model_start_ns']))
    issue_probe=statistics.median(r['observed_ns'] for r in spacing)
    mat_ids={i for r in main_gemm for i in r['node_ids']}
    # Sensitivity probe: observed launch spacing is NOT independently isolated
    # engine throughput. Keep this entirely outside the hardware timing profile.
    for measured_latency in (False,True):
        gg=oracle(set(kinds)) if measured_latency else graph
        gg=replace(gg,nodes=tuple(replace(n,issue_ns=issue_probe,occupancy_ns=issue_probe) if i in mat_ids else n for i,n in enumerate(gg.nodes)))
        variants['all_spans_plus_observed_matmul_spacing' if measured_latency else 'observed_matmul_spacing_only']=(evaluate_graph(gg).duration_ns+fixed)/1000
    dma_wait_gaps=[]
    for r in unit_info:
        if not r['category'].startswith('DMA.'):continue
        waits=[(int(s),int(v)) for s,cmp,v in ca.WAIT.findall(r['operands']) if int(v)>0]
        if waits:
            ready=max(updates[s][v] for s,v in waits)
            dma_wait_gaps.append(dict(uid=r['uid'],pc=r['pc'],category=r['category'],ready_ns=ready,trigger_ns=r['measured_start_ns'],gap_ns=r['measured_start_ns']-ready))
    admission=statistics.median(r['gap_ns'] for r in dma_wait_gaps)
    sensitivity_nodes=[];remap={}
    source_graph=oracle(set(kinds))
    for i,n in enumerate(source_graph.nodes):
        ds=tuple(replace(d,source=remap[d.source]) for d in n.dependencies)
        if n.resource=='DMAIssue':
            gate=len(sensitivity_nodes)
            sensitivity_nodes.append(OperationEvent(n.name+'_diagnostic_admission','DiagnosticAdmission',0,0,admission,dependencies=ds))
            ds=(Dependency(gate),)
        nn=replace(n,dependencies=ds)
        if i in mat_ids:nn=replace(nn,issue_ns=issue_probe,occupancy_ns=issue_probe)
        remap[i]=len(sensitivity_nodes);sensitivity_nodes.append(nn)
    variants['all_spans_observed_matmul_spacing_and_dma_admission']=(evaluate_graph(replace(graph,nodes=tuple(sensitivity_nodes))).duration_ns+fixed)/1000
    stride_copies=collections.defaultdict(list)
    for r in unit_info:
        if r['category']=='nki.copy.SBUF.float32.ScalarE':
            fs=ca.fields(r['operands']);key=str((fs['src'][2],fs['dst'][2],fs['dst'][3]))
            stride_copies[key].append(r)
    extra=dict(consecutive_matmul_groups=dict(count=len(spacing),observed_spacing_ns=stat([r['observed_ns'] for r in spacing]),modeled_spacing_ns=stat([r['modeled_ns'] for r in spacing])),dma_explicit_wait_to_trigger_ns=stat([r['gap_ns'] for r in dma_wait_gaps]),copy_stride_groups={k:dict(count=len(v),observed_ns=stat([r['measured_span_ns'] for r in v]),modeled_ns=stat([r['model_latency_ns'] for r in v])) for k,v in stride_copies.items()})
    (out/'dma-wait-to-trigger.json').write_text(json.dumps(dma_wait_gaps,indent=2)+'\n')
    (out/'matmul-spacing.json').write_text(json.dumps(spacing,indent=2)+'\n')
    summary=profile['summary'][0]
    comparators=[]
    for directory in a.compare:
        prov=json.loads((directory/'extraction_provenance.json').read_text())
        for name,value in prov['artifact_sha256'].items():assert digest(directory/name)==value,(directory,name)
        path=directory/'profile.json'
        raw=path.read_bytes() if path.exists() else gzip.decompress(path.with_suffix('.json.gz').read_bytes())
        expected_profile_hash=prov.get('profile_uncompressed_sha256',prov.get('profile_sha256'))
        assert hashlib.sha256(raw).hexdigest()==expected_profile_hash
        prof=json.loads(raw);gg=collections.defaultdict(list)
        for ins in prof['instruction']:
            if ins.get('raw_bir_id') and ins['opcode'] in ('LDWEIGHTS','MATMUL'):gg[ins['raw_bir_id']].append(ins)
        forms=collections.defaultdict(list)
        for g in gg.values():
            mm=[x for x in g if x['opcode']=='MATMUL'];ins=mm[0]
            if ins['instruction_type']!='REGULAR' or not ins['operands'].rstrip().endswith('128*128'):continue
            fs=ca.fields(ins['operands']);src=fs['src']
            forms[str((src[2],src[3]))].append(max(i['timestamp']+i['duration'] for i in g)-min(i['timestamp'] for i in g))
        comparators.append(dict(directory=str(directory),artifact_sha256=prov['artifact_sha256'],profile_sha256=expected_profile_hash,trace_us=prof['summary'][0]['total_time']*1e6,matmul_forms={k:stat(v) for k,v in forms.items()},throttle={k:v for k,v in prof['summary'][0].items() if 'throttle' in k}))
    report=dict(scope='Read-only authenticated NEFF + matching NTFF/profile analysis. Oracle replays use measured event spans only as diagnostic substitutions, never production inputs or calibration.',prediction=prediction,benchmark_us=json.loads((root/'result.json').read_text())['latencies'],trace_us=summary['total_time']*1e6,artifact_sha256=provenance['artifact_sha256'],profile_sha256=provenance['profile_uncompressed_sha256'],static_sha256=provenance['compiled_static_sha256'],source_sha256={__file__:digest(Path(__file__)),ca.__file__:digest(Path(ca.__file__))},units=len(units),comparators=comparators,additional_diagnostics=extra,profile_conditioned_latency_replays_us=variants,primitive_stats=stats,phase_markers=markers,readiness_residual_ns={k:stat(v) for k,v in residuals.items()},throttle={k:v for k,v in summary.items() if 'throttle' in k},ham_intervals=profile['ham'],spill_bytes={k:summary.get(k) for k in ['spill_save_bytes','spill_reload_bytes']})
    (out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'units.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(unit_info[0]));w.writeheader();w.writerows(unit_info)
    end=max(range(len(ends)),key=ends.__getitem__);chain=[];visited=set()
    while end is not None:
        assert end not in visited;visited.add(end);n=graph.nodes[end]
        edge=crit[end];chain.append(dict(index=end,name=n.name,resource=n.resource,implementation=n.implementation,start_ns=starts[end],end_ns=ends[end],critical_predecessor=edge))
        end=edge[0] if edge else None
    (out/'model-critical-path.json').write_text(json.dumps(list(reversed(chain)),indent=2)+'\n')
    (out/'unit-mapping.json').write_text(json.dumps(unit_info,separators=(',',':'))+'\n')
    print(json.dumps({k:report[k] for k in ['trace_us','profile_conditioned_latency_replays_us','primitive_stats','phase_markers','readiness_residual_ns']},indent=2))


if __name__=='__main__':main()

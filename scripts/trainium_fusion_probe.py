"""Matched native Trainium2 ISA fusion probes; no production model changes."""
import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import traceback
import numpy as np
import neuronxcc.nki as nki


def source(kind, width, fused, steps):
    lines = ['import neuronxcc.nki as nki', 'import neuronxcc.nki.language as nl',
             'import neuronxcc.nki.isa as nisa', 'import neuronxcc.nki.compiler as ncc',
             '@nki.compiler.skip_middle_end_transformations', '@nki.jit', 'def kernel(a,b,c):']
    def put(s): lines.append('    ' + s)
    offset = 0
    for name, w in [('x',width),('z',width),('tmp',width),('y',width),('v',1)]:
        put(f'{name} = nl.ndarray((128,{w}), dtype=nl.float32, buffer=ncc.sbuf.alloc(lambda idx,pdim_size,fdim_size: (0,{offset})))')
        offset += max(128,w*4)
    put('nisa.dma_copy(dst=x,src=a)')
    put('nisa.dma_copy(dst=y,src=b)')
    put('nisa.dma_copy(dst=v,src=c)')
    if kind == 'psum_add':
        put(f'p = nl.ndarray((128,{width}), dtype=nl.float32, buffer=ncc.psum.alloc(lambda idx,pdim_size,fdim_size: (0,0,0)))')
    for i in range(steps):
        src,dst = ('x','z') if i%2==0 else ('z','x')
        if kind == 'psum_add':
            put(f'p[...] = nisa.tensor_copy({src},engine=nisa.scalar_engine)')
            if fused:
                put(f'{dst}[...] = nisa.tensor_tensor(p,y,op=nl.add,engine=nisa.vector_engine)')
            else:
                put('tmp[...] = nisa.tensor_copy(p,engine=nisa.scalar_engine)')
                put(f'{dst}[...] = nisa.tensor_tensor(tmp,y,op=nl.add,engine=nisa.vector_engine)')
        elif kind == 'rms_scale':
            if fused:
                put(f'{dst}[...] = nisa.scalar_tensor_tensor(data={src},op0=nl.multiply,operand0=v,op1=nl.multiply,operand1=y)')
            else:
                put(f'tmp[...] = nisa.tensor_scalar({src},op0=nl.multiply,operand0=v,engine=nisa.vector_engine)')
                put(f'{dst}[...] = nisa.tensor_tensor(tmp,y,op=nl.multiply,engine=nisa.vector_engine)')
        else:
            if fused:
                put(f'{dst}[...] = nisa.activation(op=nl.rsqrt,data={src},scale={1/1024},bias=v)')
            else:
                put(f'tmp[...] = nisa.tensor_scalar({src},op0=nl.multiply,operand0={1/1024},op1=nl.add,operand1=v,engine=nisa.vector_engine)')
                put(f'{dst}[...] = nisa.activation(op=nl.rsqrt,data=tmp)')
    put(f'out = nl.ndarray((128,{width}),dtype=nl.float32,buffer=nl.shared_hbm)')
    put(f'nisa.dma_copy(dst=out,src={dst})')
    put('return out')
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=['psum_add','rms_scale','scale_rsqrt'])
    p.add_argument('--width',type=int)
    p.add_argument('--fused',type=int,choices=[0,1])
    p.add_argument('--steps',type=int,default=32)
    args=p.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.kind is None:
        for kind,widths in [('psum_add',[128,512]),('rms_scale',[128,1024]),('scale_rsqrt',[1,128])]:
            for width in widths:
                for fused in [0,1]:
                    name=f'{kind}-{width}-'+('fused' if fused else 'separate')
                    case=root/name;case.mkdir(exist_ok=True)
                    with (case/'run.log').open('w') as log:
                        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(root),'--kind',kind,'--width',str(width),'--fused',str(fused),'--steps',str(args.steps)],stdout=log,stderr=subprocess.STDOUT)
                    print(name, 'pass' if result.returncode==0 else 'FAIL',flush=True)
        rows=[json.loads(f.read_text()) for f in sorted(root.glob('*/result.json'))]
        (root/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        if len(rows)!=12 or any(r['status']!='pass' for r in rows):
            raise SystemExit('Some probes failed; inspect per-case logs.')
        return
    case=root/(f'{args.kind}-{args.width}-'+('fused' if args.fused else 'separate'))
    case.mkdir(exist_ok=True);os.chdir(case)
    os.environ['PATH']=str(Path(sys.executable).parent)+':/opt/aws_neuronx_venv_pytorch_2_9/bin:/opt/aws/neuron/bin:'+os.environ['PATH']
    os.environ['NEURON_RT_VISIBLE_CORES']='0'
    os.environ['NEURON_RT_ENABLE_DGE_NOTIFICATIONS']='1'
    record=dict(kind=args.kind,width=args.width,fused=bool(args.fused),steps=args.steps,partitions=128,dtype='float32',compiler=importlib.metadata.version('neuronx-cc'),flags='--target=trn2 --lnc=1',warmup=10,iterations=100,repeats=3,visible_core=0)
    try:
        path=case/'program.py';path.write_text(source(args.kind,args.width,args.fused,args.steps))
        spec=importlib.util.spec_from_file_location('probe',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        rng=np.random.default_rng(873)
        a=rng.uniform(.5,1.5,(128,args.width)).astype(np.float32)
        b=rng.uniform(.001,.01,a.shape).astype(np.float32) if args.kind=='psum_add' else rng.uniform(.99,1.01,a.shape).astype(np.float32)
        c=np.full((128,1),1e-5 if args.kind=='scale_rsqrt' else .99,dtype=np.float32)
        expected=a.copy()
        for _ in range(args.steps):
            if args.kind=='psum_add':expected=expected+b
            elif args.kind=='rms_scale':expected=(expected*c)*b
            else:expected=1/np.sqrt(expected*np.float32(1/1024)+c)
        np.savez(case/'reference.npz',a=a,b=b,c=c,expected=expected)
        def lock(kernel,save=False):
            execute=kernel.execute_neff
            def wrapped(*values,**kwargs):
                if save:
                    neff=Path(values[0] if values else kwargs['neff'])
                    if neff.resolve()!=Path('file.neff').resolve(): shutil.copy2(neff,'file.neff')
                    kernel.replay_values=('file.neff',*values[1:]) if values else ()
                    kernel.replay_kwargs=dict(kwargs)
                    if not values:kernel.replay_kwargs['neff']='file.neff'
                with open('/tmp/voyager-trainium-device.lock','w') as f:
                    fcntl.flock(f,fcntl.LOCK_EX)
                    return execute(*values,**kwargs)
            kernel.execute_neff=wrapped
            return kernel
        run=lock(nki.baremetal(mod.kernel,additional_compile_opt=record['flags'],save_neff_name='file.neff'),True)
        actual=np.asarray(run(a,b,c))
        np.testing.assert_allclose(actual,expected,atol=5e-5,rtol=5e-5)
        np.save(case/'actual.npy',actual)
        record['max_abs_error']=float(np.max(np.abs(actual-expected)))
        benchmark=lock(nki.benchmark(mod.kernel,warmup=10,iters=100,additional_compile_opt=record['flags'],save_neff_name='file.neff',save_trace_name='profile.ntff'))
        times=[]
        for _ in range(3):
            benchmark.execute_neff(*run.replay_values,**run.replay_kwargs)
            times.append(float(benchmark.benchmark_result.nc_latency.get_latency_percentile(50)))
        record.update(p50_us=times,median_us=statistics.median(times),timing_scope=f'Device nc_latency for {args.steps} dependent steps plus identical DMA and setup; excludes host invocation. Same correctness-validated NEFF reused.')
        subprocess.run(['/opt/aws/neuron/bin/neuron-profile','view','-n','file.neff','-s','profile.ntff','--output-format','json','--output-file','profile.json'],check=True,stdout=subprocess.DEVNULL)
        data=json.loads(Path('profile.json').read_text())
        ins=data['instruction'];record['opcode_counts']=dict(Counter(x['opcode'] for x in ins))
        groups=defaultdict(list)
        for x in ins:groups[x['opcode']].append(x['duration'])
        record['opcode_duration_ns']={k:dict(count=len(v),median=statistics.median(v),sum=sum(v)) for k,v in groups.items()}
        record['profile_keys']=list(data)
        record['status']='pass'
        record['artifact_sha256']={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ['program.py','reference.npz','actual.npy','file.neff','profile.ntff','profile.json']}
    except Exception:
        record.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        (case/'result.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)

if __name__=='__main__':main()

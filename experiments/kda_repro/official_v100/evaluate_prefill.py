"""Full official GDN prefill inputs versus the unmodified reference."""
import argparse
import functools
import hashlib
import json
import statistics
from pathlib import Path
import torch
from safetensors.torch import load_file
from gdn_prefill import candidate
from evaluate_decode import graph_for, measure, load_candidate

NAME='gdn_prefill_qk4_v8_d128_k_last'
ORDER=('q','k','v','state','A_log','a','dt_bias','b','cu_seqlens','scale')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--indices',default='',help='comma-separated official workload indices')
    p.add_argument('--baseline-source')
    p.add_argument('--candidate-source')
    p.add_argument('--timing-repeats',type=int,default=7)
    args=p.parse_args()
    data,out=Path(args.data),Path(args.output)
    if out.exists():
        raise SystemExit('Evidence exists')
    out.parent.mkdir(parents=True,exist_ok=True)
    ns={}
    exec(compile(json.loads((data/f'definitions/gdn/{NAME}.json').read_text())['reference'],'official_reference','exec'),ns)
    ref=ns['run']
    workloads=[json.loads(x)['workload'] for x in (data/f'workloads/gdn/{NAME}.jsonl').read_text().splitlines()]
    if args.indices:
        indices=[int(value) for value in args.indices.split(',')]
        if any(index<0 or index>=len(workloads) for index in indices):
            raise SystemExit('workload index out of range')
        workloads=[workloads[index] for index in indices]
    elif args.limit:
        workloads=workloads[:args.limit]
    if args.timing_repeats<3:
        raise SystemExit('timing repeats must be at least 3')
    functions={}
    baseline_path=Path(args.baseline_source).resolve() if args.baseline_source else None
    source_path=Path(args.candidate_source).resolve() if args.candidate_source else Path(__file__).with_name('gdn_prefill.py')
    if baseline_path:
        functions['baseline']=load_candidate(baseline_path,'kda_prefill_baseline')
    if args.candidate_source:
        functions['candidate']=load_candidate(source_path,'kda_prefill_candidate')
    elif not baseline_path:
        for rows in (4,8,16):
            functions[str(rows)]=functools.partial(candidate,rows=rows)
    report=dict(status='running',scope='official data and reference; adapted V100 runner, not official evaluator',
                dataset=json.loads((data/'dataset-lock.json').read_text()),device=torch.cuda.get_device_name(0),
                torch=torch.__version__,atol=.01,rtol=.01,workloads=[],branches=[],
                source_path=str(source_path),source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                baseline_source_path=str(baseline_path) if baseline_path else None,
                baseline_source_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest() if baseline_path else None,
                timing_repeats=args.timing_repeats)
    try:
        for w in workloads:
            specs=w['inputs']
            blob=data/next(x['path'] for x in specs.values() if x['type']=='safetensors')
            tensors=load_file(str(blob),device='cuda')
            inputs=[tensors[specs[n]['tensor_key']] if specs[n]['type']=='safetensors' else specs[n]['value'] for n in ORDER]
            snapshots=[x.clone() if isinstance(x,torch.Tensor) else x for x in inputs]
            expected=ref(*inputs)
            item=dict(uuid=w['uuid'],axes=w['axes'],candidates={})
            valid={}
            for name,fn in functions.items():
                try:
                    torch.cuda.reset_peak_memory_stats()
                    before=torch.cuda.memory_allocated()
                    actual=fn(*inputs)
                    torch.cuda.synchronize()
                    peak=torch.cuda.max_memory_allocated()-before
                    for a,e in zip(actual,expected):
                        torch.testing.assert_close(a,e,atol=.01,rtol=.01)
                    for a,e in zip(inputs,snapshots):
                        if isinstance(a,torch.Tensor):
                            torch.testing.assert_close(a,e,atol=0,rtol=0)
                    errors=[float((a.float()-e.float()).abs().max()) for a,e in zip(actual,expected)]
                    valid[name]=fn
                    item['candidates'][name]=dict(status='correct',max_abs=errors,peak_extra_bytes=peak)
                except Exception as error:
                    item['candidates'][name]=dict(status='rejected',reason=str(error))
            for required in ('baseline','candidate'):
                if required in functions and item['candidates'][required]['status']!='correct':
                    report['workloads'].append(item)
                    out.write_text(json.dumps(report,indent=2))
                    raise RuntimeError(f"{required} rejected on {w['uuid']}: {item['candidates'][required]['reason']}")
            if not valid:
                raise RuntimeError(f"no candidate passed on {w['uuid']}")
            graphs={name:graph_for(fn,inputs,repetitions=3) for name,fn in valid.items()}
            samples={name:[] for name in graphs}
            names=list(graphs)
            for repeat in range(args.timing_repeats):
                for name in (names if repeat%2==0 else names[::-1]):
                    samples[name].append(measure(graphs[name][0],repetitions=3))
            for name in graphs:
                item['candidates'][name].update(raw_ms=samples[name],median_ms=statistics.median(samples[name]))
            item['winner']=min(graphs,key=lambda name:item['candidates'][name]['median_ms'])
            report['workloads'].append(item)
            out.write_text(json.dumps(report,indent=2))
            print(json.dumps(dict(uuid=w['uuid'],axes=w['axes'],statuses={k:v['status'] for k,v in item['candidates'].items()})),flush=True)
            del graphs
        # Original reference defines empty sequences' state as zero, even with supplied state.
        small=list(inputs)
        small[0],small[1],small[2]=[x[:6].contiguous() for x in inputs[:3]]
        small[5],small[7]=inputs[5][:6].contiguous(),inputs[7][:6].contiguous()
        small[8]=torch.tensor([0,0,2,6],device='cuda',dtype=inputs[8].dtype)
        for has_state in (False,True):
            small[3]=torch.randn(3,8,128,128,device='cuda')*.01 if has_state else None
            small[9]=0.0
            expected=ref(*small)
            for name,fn in functions.items():
                actual=fn(*small)
                for a,e in zip(actual,expected):
                    torch.testing.assert_close(a,e,atol=.01,rtol=.01)
            report['branches'].append(dict(state_provided=has_state,empty_sequence=True,default_scale=True,status='passed'))
        report['status']='complete' if all(v['status']=='correct' for w in report['workloads'] for v in w['candidates'].values()) else 'complete_with_rejections'
    except Exception as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        out.write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    main()

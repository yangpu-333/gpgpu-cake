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
from evaluate_decode import graph_for, measure

NAME='gdn_prefill_qk4_v8_d128_k_last'
ORDER=('q','k','v','state','A_log','a','dt_bias','b','cu_seqlens','scale')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    data,out=Path(args.data),Path(args.output)
    if out.exists():
        raise SystemExit('Evidence exists')
    out.parent.mkdir(parents=True,exist_ok=True)
    ns={}
    exec(compile(json.loads((data/f'definitions/gdn/{NAME}.json').read_text())['reference'],'official_reference','exec'),ns)
    ref=ns['run']
    workloads=[json.loads(x)['workload'] for x in (data/f'workloads/gdn/{NAME}.jsonl').read_text().splitlines()]
    if args.limit:
        workloads=workloads[:args.limit]
    report=dict(status='running',scope='official data and reference; adapted V100 runner, not official evaluator',
                dataset=json.loads((data/'dataset-lock.json').read_text()),device=torch.cuda.get_device_name(0),
                torch=torch.__version__,atol=.01,rtol=.01,workloads=[],branches=[],
                source_sha256=hashlib.sha256(Path(__file__).with_name('gdn_prefill.py').read_bytes()).hexdigest())
    try:
        for w in workloads:
            specs=w['inputs']
            blob=data/next(x['path'] for x in specs.values() if x['type']=='safetensors')
            tensors=load_file(str(blob),device='cuda')
            inputs=[tensors[specs[n]['tensor_key']] if specs[n]['type']=='safetensors' else specs[n]['value'] for n in ORDER]
            snapshots=[x.clone() if isinstance(x,torch.Tensor) else x for x in inputs]
            expected=ref(*inputs)
            item=dict(uuid=w['uuid'],axes=w['axes'],candidates={})
            for rows in (4,8,16):
                fn=functools.partial(candidate,rows=rows)
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
                    graph,retained=graph_for(fn,inputs,repetitions=3)
                    times=[measure(graph,repetitions=3) for _ in range(7)]
                    item['candidates'][str(rows)]=dict(status='correct',max_abs=errors,peak_extra_bytes=peak,raw_ms=times,median_ms=statistics.median(times))
                    del graph,retained
                except Exception as error:
                    item['candidates'][str(rows)]=dict(status='rejected',reason=str(error))
            report['workloads'].append(item)
            out.write_text(json.dumps(report,indent=2))
            print(json.dumps(dict(uuid=w['uuid'],axes=w['axes'],statuses={k:v['status'] for k,v in item['candidates'].items()})),flush=True)
        # Original reference defines empty sequences' state as zero, even with supplied state.
        small=list(inputs)
        small[0],small[1],small[2]=[x[:6].contiguous() for x in inputs[:3]]
        small[5],small[7]=inputs[5][:6].contiguous(),inputs[7][:6].contiguous()
        small[8]=torch.tensor([0,0,2,6],device='cuda',dtype=inputs[8].dtype)
        for has_state in (False,True):
            small[3]=torch.randn(3,8,128,128,device='cuda')*.01 if has_state else None
            small[9]=0.0
            expected=ref(*small)
            for rows in (4,8,16):
                actual=candidate(*small,rows=rows)
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

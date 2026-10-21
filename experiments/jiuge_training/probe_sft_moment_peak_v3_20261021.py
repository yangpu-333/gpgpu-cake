"""Isolate moment initialization RSS in fresh bounded CPU-only subprocesses."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT = PREP/'sft-moment-direct-zero-preparation-20261021-01/memory-probe'
PARAMETER_BYTES = 128*1024**2


def rss():
    values = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:', 'VmHWM:', 'RssAnon:', 'RssShmem:')):
            name, value = line.split(':',1)
            values[name] = int(value.split()[0])*1024
    return values


def child(args):
    os.sched_setaffinity(0,{args.cpu})
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    parameter = torch.nn.Parameter(torch.full((PARAMETER_BYTES//4,),.25,dtype=torch.float32))
    parameter.grad = torch.full_like(parameter,.01)
    optimizer = torch.optim.AdamW([parameter],foreach=False,fused=False)
    events = []
    undo = None
    if args.arm == 'copy-v1':
        from sft_local_moments_20261021 import install
        undo = install(torch,events)
    elif args.arm == 'direct-zero-v3':
        from sft_local_moments_v3_20261021 import install
        undo = install(torch,events)
    before = rss()
    params, grads, avgs, squares, maxima, steps = [], [], [], [], [], []
    try:
        optimizer._init_group(optimizer.param_groups[0],params,grads,False,avgs,squares,maxima,steps)
    finally:
        if undo is not None:undo()
    after = rss()
    assert params == [parameter] and len(avgs) == len(squares) == len(steps) == 1
    assert avgs[0] is optimizer.state[parameter]['exp_avg']
    assert squares[0] is optimizer.state[parameter]['exp_avg_sq']
    assert avgs[0].count_nonzero().item() == squares[0].count_nonzero().item() == 0
    assert steps[0].item() == 0 and not torch.cuda.is_initialized()
    if args.arm == 'direct-zero-v3':
        assert len(events)==2 and all(e['initialization_path']=='direct_local_zero' for e in events)
        assert all(not e['native_source_moment_created'] and e['copied_source_moment_bytes']==0 for e in events)
    result = {'arm':args.arm,'node':args.node,'cpu':args.cpu,'parameter_bytes':PARAMETER_BYTES,
              'final_moment_bytes':2*PARAMETER_BYTES,'rss_before_initialization':before,
              'rss_after_initialization':after,'initialization_peak_rss_bytes':after['VmHWM'],
              'initialization_peak_increase_bytes':after['VmHWM']-before['VmRSS'],
              'events':events,'GPU_used':False,'CUDA_initialized':False,
              'functional_state_zero_and_native_list_identity_verified':True,
              'this_is_not_7B_memory_acceptance':True,'formal_performance_evidence':False}
    with (ROOT/f'node-{args.node}-{args.arm}.json').open('x') as stream:json.dump(result,stream,indent=2)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=('native','copy-v1','direct-zero-v3'))
    parser.add_argument('--node',type=int)
    parser.add_argument('--cpu',type=int)
    args=parser.parse_args()
    if args.arm:
        child(args);return
    ROOT.mkdir(exist_ok=False)
    probe=json.loads((ROOT.parent/'page-policy-probe.json').read_text())
    if not probe['mbind_all_nodes_passed'] or not probe['first_touch_all_nodes_passed']:
        raise ValueError('Actual NUMA capability checks required')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory',
                                  '--format=csv,noheader'],text=True,timeout=15)
    if apps.strip():raise ValueError('Other GPU workload appeared; defer')
    rows=[]
    for node,cpu in sorted(probe['nodes_and_probe_CPUs'].items()):
        for arm in ('native','copy-v1','direct-zero-v3'):
            with (ROOT/f'node-{node}-{arm}.log').open('x') as stream:
                subprocess.run([sys.executable,__file__,'--arm',arm,'--node',str(node),'--cpu',str(cpu)],
                               stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=150)
            rows.append(json.loads((ROOT/f'node-{node}-{arm}.json').read_text()))
    comparisons=[]
    for node in sorted({r['node'] for r in rows}):
        by={r['arm']:r for r in rows if r['node']==node}
        reduction=by['copy-v1']['initialization_peak_increase_bytes']-by['direct-zero-v3']['initialization_peak_increase_bytes']
        comparisons.append({'node':node,'old_copy_peak_increase_bytes':by['copy-v1']['initialization_peak_increase_bytes'],
                            'direct_zero_peak_increase_bytes':by['direct-zero-v3']['initialization_peak_increase_bytes'],
                            'observed_peak_reduction_bytes':reduction,
                            'at_least_half_one_parameter_buffer_reduction':reduction>=PARAMETER_BYTES//2})
    result={'schema':'jiuge7b-sft-moment-initialization-CPU-memory-v3',
            'utc':datetime.now(timezone.utc).isoformat(),'rows':rows,'comparisons':comparisons,
            'all_direct_fresh_source_copy_bytes_zero':True,
            'measured_peak_reduction_passed':all(x['at_least_half_one_parameter_buffer_reduction'] for x in comparisons),
            'GPU_used':False,'new_7B_OOM_fixed':False,'formal_performance_evidence':False,
            'scope':'fresh subprocesses;one128MiBCPUparameter;native initialization only;no training timing'}
    with (ROOT/'summary.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()

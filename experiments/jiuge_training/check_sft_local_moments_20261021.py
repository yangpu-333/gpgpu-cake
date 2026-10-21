"""CPU fixtures for storage, full AdamW state, native resume and lifetime only.

These small CPU fixtures do not substitute for7B SFT acceptance or GPU timing.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import weakref

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT = PREP/'sft-moment-placement-preparation-20261021-01/cpu-regression'


def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2)


def child(node,cpu):
    os.sched_setaffinity(0,{cpu})
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    import torch
    import copy
    from sft_local_moments_20261021 import install,node_for_current_cpus,relocate_moment
    assert node_for_current_cpus()==node and not torch.cuda.is_initialized()
    folder = ROOT/f'node-{node}'
    folder.mkdir(exist_ok=False)
    shapes = ((65536,),(512,16),(),(27,))

    def model():
        return [torch.nn.Parameter(torch.linspace(-.2,.2,steps=max(1,__import__('math').prod(s)),
                                                 dtype=torch.float32).reshape(s)) for s in shapes]

    def optimizer(params):
        return torch.optim.AdamW([{'params':params[:2]},{'params':params[2:]}],
            lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=.1)

    def update(params,opt,step):
        for i,p in enumerate(params):
            p.grad = None if i==3 and step==1 else torch.full_like(p,.01*(i+1)+.001*step)
        torch.nn.utils.clip_grad_norm_(params,1.)
        opt.step();opt.zero_grad(set_to_none=True)

    def snapshot(params,opt):
        return {'params':[p.detach().clone() for p in params], 'optimizer':copy.deepcopy(opt.state_dict())}

    compared = 0
    def equal(left,right):
        nonlocal compared
        if torch.is_tensor(left):
            compared += 1
            assert torch.is_tensor(right) and left.dtype==right.dtype and left.shape==right.shape
            assert torch.equal(left,right), 'CPU fixture full-state values differ'
        elif isinstance(left,dict):
            assert left.keys()==right.keys()
            for key in left:equal(left[key],right[key])
        elif isinstance(left,(list,tuple)):
            assert type(left)==type(right) and len(left)==len(right)
            for a,b in zip(left,right):equal(a,b)
        else:assert left==right

    native_params = model();native = optimizer(native_params)
    references = []
    for step in range(1,6):
        update(native_params,native,step)
        reference = snapshot(native_params,native)
        references.append(reference)
        with (folder/f'native-state-step-{step}.pt').open('xb') as stream:torch.save(reference,stream)
    candidate_params = model();candidate = optimizer(candidate_params)
    events = []
    uninstall = install(torch,events)
    try:
        for step in range(1,6):
            update(candidate_params,candidate,step)
            equal(references[step-1],snapshot(candidate_params,candidate))
            expected_events = 6 if step==1 else 8
            assert len(events)==expected_events, 'Moment relocation repeated after initialization'
        # Native checkpoint is the only resume source. The candidate must handle
        # ordinary loaded storages, not a self-produced preferred checkpoint.
        loaded = torch.load(folder/'native-state-step-2.pt',weights_only=False,map_location='cpu')
        restored_params = model();restored = optimizer(restored_params)
        with torch.no_grad():
            for p,value in zip(restored_params,loaded['params']):p.copy_(value)
        restored.load_state_dict(loaded['optimizer'])
        equal(references[1],snapshot(restored_params,restored))
        before = len(events)
        update(restored_params,restored,3)
        equal(references[2],snapshot(restored_params,restored))
        assert len(events)==before+8
        weak = weakref.ref(restored)
        del restored
        gc.collect();assert weak() is None, 'Patch retained an optimizer'
    finally:
        uninstall()
    # A tensor view must keep its mmap storage alive after the returned owner
    # goes out of scope; then both tensor and view must be collectable.
    source = torch.arange(1024,dtype=torch.float32)
    moved,receipt = relocate_moment(source,node,torch)
    weak = weakref.ref(moved)
    view = moved.view(-1)
    del moved;gc.collect();assert torch.equal(view,source)
    del view;gc.collect();assert weak() is None
    assert not torch.cuda.is_initialized()
    value = {'passed':True,'node':node,'bound_CPU':cpu,'torch_version':torch.__version__,
             'GPU_used':False,'CUDA_initialized':False,'five_updates_full_state_exact':True,
             'native_step2_resume_and_step3_exact':True,'compared_tensor_fields':compared,
             'max_absolute_error':0.,'lazy_none_gradient_and_multiple_groups_checked':True,
             'placement_events':events,'placement_not_repeated_on_later_steps':True,
             'observer_retains_optimizer':False,'mmap_view_lifetime_checked':True,
             'new_SFT_full_state_acceptance':False,'formal_performance_evidence':False}
    save(folder/'result.json',value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--node',type=int)
    parser.add_argument('--cpu',type=int)
    args = parser.parse_args()
    if args.node is not None:
        child(args.node,args.cpu);return
    assert socket.gethostname()=='sg-huids25-260917-94135-default0-0'
    probe = json.loads((ROOT.parent/'page-policy-probe.json').read_text())
    if not probe['mbind_all_nodes_passed'] or not probe['first_touch_all_nodes_passed']:
        raise ValueError('Live NUMA capability gate required before CPU fixture')
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory',
                                   '--format=csv,noheader'],text=True,timeout=15)
    if apps.strip():raise ValueError('Other GPU workload appeared; defer')
    ROOT.mkdir(exist_ok=False)
    sources = (Path(__file__),PREP/'sft_local_moments_20261021.py')
    for path in sources:(ROOT/path.name).write_bytes(path.read_bytes())
    rows = []
    for node,cpu in sorted(probe['nodes_and_probe_CPUs'].items()):
        with (ROOT/f'node-{node}.log').open('x') as stream:
            subprocess.run([sys.executable,str(Path(__file__)),'--node',node,'--cpu',str(cpu)],
                           stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=120)
        rows.append(json.loads((ROOT/f'node-{node}/result.json').read_text()))
    files = {str(f.relative_to(ROOT)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
             for f in ROOT.rglob('*') if f.is_file()}
    save(ROOT/'manifest.json',{'files':files,'ACT_only_tensor_fixtures':True})
    for name,expected in files.items():
        path = ROOT/name
        assert path.stat().st_size==expected['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==expected['sha256']
    result = {'schema':'jiuge7b-sft-local-moments-CPU-regression-v1','passed':all(x['passed'] for x in rows),
              'ACT_root':str(ROOT),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'manifest_sha256':hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest(),
              'ACT_verified_files':len(files),'ACT_bytes':sum(f['bytes'] for f in files.values()),
              'nodes':[{'node':x['node'],'five_updates_full_state_exact':x['five_updates_full_state_exact'],
                        'native_resume_next_step_exact':x['native_step2_resume_and_step3_exact'],
                        'compared_tensor_fields':x['compared_tensor_fields'],'max_absolute_error':x['max_absolute_error'],
                        'placement_events':len(x['placement_events']),'mmap_view_lifetime_checked':x['mmap_view_lifetime_checked']}
                       for x in rows],
              'GPU_used':False,'new_SFT_acceptance':False,'formal_performance_evidence':False}
    save(ROOT/'summary.json',result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()

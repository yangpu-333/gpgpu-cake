"""Observe actual compiled CE launches outside all promotion timed regions."""
import argparse
import json
from pathlib import Path
import sys
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parent))
import driver
import torch
from triton.compiler.compiler import CompiledKernel

parser=argparse.ArgumentParser()
parser.add_argument('candidate',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args()
if args.output.exists():
    parser.error('immutable audit already exists')
candidate=driver.base.inspect_candidate(args.candidate)
launches=[]
original=CompiledKernel.__getitem__
phase=['forward']
def observe(kernel,grid):
    runner=original(kernel,grid)
    def launch(*a,**kw):
        launches.append({'phase':phase[0],'grid':list(grid),
            'name':str(getattr(kernel,'name',getattr(kernel,'metadata',{}))),
            'asm_keys':sorted(getattr(kernel,'asm',{})),
            'scope':'observed CompiledKernel runner invocation; no instruction equivalence claim'})
        return runner(*a,**kw)
    return launch
CompiledKernel.__getitem__=observe
try:
    torch.manual_seed(24007)
    for _ in range(3):
        logits=torch.randn(128,2,1024,device='cuda',requires_grad=True)
        target=torch.randint(1024,(128,2),device='cuda')
        phase[0]='forward'
        loss=candidate.cross_entropy(logits,target)
        phase[0]='backward'
        torch.autograd.grad(loss,logits,torch.randn_like(loss))
    torch.cuda.synchronize()
finally:
    CompiledKernel.__getitem__=original
forward=[r for r in launches if r['phase']=='forward']
backward=[r for r in launches if r['phase']=='backward']
from stage19_megatron_route_step import load_megatron
load_megatron(Path('/private/atrex-megatron/src/megatron-lm'))
import os
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
os.environ['MASTER_ADDR']='127.0.0.1'
os.environ['MASTER_PORT']='29833'
dist.init_process_group('nccl',rank=0,world_size=1)
parallel_state.initialize_model_parallel()
group=parallel_state.get_tensor_model_parallel_group()
xr=torch.randn(2,2,17,device='cuda',requires_grad=True)
xc=xr.detach().clone().requires_grad_()
target=torch.tensor([-(1<<63),1<<32,(1<<32)+16,(1<<63)-1],dtype=torch.int64,device='cuda').reshape(2,2)
native=vocab_parallel_cross_entropy(xr,target,tp_group=group)
actual=candidate.cross_entropy(xc,target)
upstream=torch.tensor([1.0,-1.0,0.5,-0.5],device='cuda').reshape(2,2)
ng=torch.autograd.grad(native,xr,upstream)[0]
cg=torch.autograd.grad(actual,xc,upstream)[0]
wide_checks={'loss':driver.base.check(actual,native,3e-4,1e-3),
             'dlogits':driver.base.check(cg,ng,3e-4,1e-3)}
parallel_state.destroy_model_parallel()
dist.destroy_process_group()
report={'candidate_sha256':driver.sha256(args.candidate),
    'base_harness_sha256':driver.sha256(driver.base.__file__),
    'base_contract_sha256':driver.sha256(driver.BASE_ROOT/'contract.json'),
    'extension_harness_sha256':driver.sha256(driver.__file__),
    'contract_sha256':driver.sha256(driver.ROOT/'contract.json'),
    'operation':'cross_entropy','shape':[128,2,1024],
    'logits_dtype':'torch.float32','loss_dtype':'torch.float32','launches':launches,
    'forward':{'compiled_kernel_names':sorted(set(r['name'] for r in forward)),
        'launch_counts':dict(Counter(r['name'] for r in forward))},
    'backward':{'compiled_kernel_names':sorted(set(r['name'] for r in backward)),
        'launch_counts':dict(Counter(r['name'] for r in backward))},
    'forward_compiled_launches':len(forward),'backward_compiled_launches':len(backward),
    'wide_int64_out_of_range_probe':wide_checks,
    'passed':bool(forward and backward) and all(v['passed'] for v in wide_checks.values()),
    'scope':'Independent CE-only observed compiled-runner audit; not timed throughput'}
args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(report['passed'],flush=True)

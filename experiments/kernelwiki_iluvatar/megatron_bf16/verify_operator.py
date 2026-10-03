"""Fixed BF16 vocabulary CE checks against untouched pinned native Megatron.

No timing, production-speedup claim, or tolerance adjustment. Every numerical
comparison is performed on CPU float64 copies, including BF16 dlogits.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
FORMAL = ROOT.parent / 'megatron_training_entry'
SHAPES = ((128,2,1024), (17,2,512), (7,3,769), (8,1,7), (8,1,1),
          (128,2,4096), (256,2,4096), (512,2,8192))
DISTRIBUTIONS = ('random-0.001','random-1','random-100','zero-ties',
                 'offset-10000','dominant-positive-1000','dominant-negative-1000')
LABELS = ('valid','outside','wide')
UPSTREAM = ('ones','signed-zero','strided-signed')
ATOL, RTOL = 3e-4, 1e-3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(torch, actual, expected):
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    error = (a-b).abs()
    limit = ATOL + RTOL*b.abs()
    return {'passed':bool(torch.isfinite(a).all() and torch.isfinite(b).all()
                           and torch.all(error <= limit)),
            'max_abs_error':error.max().item(),
            'failed_elements':int((error > limit).sum()), 'elements':a.numel(),
            'actual_dtype':str(actual.dtype),'native_dtype':str(expected.dtype)}


def values(torch, shape, kind, generator):
    result = torch.randn(shape,generator=generator)
    if kind.startswith('random-'):
        return result * float(kind.split('-',1)[1])
    if kind=='offset-10000':
        return result + 10000
    result.zero_()
    if kind!='zero-ties':
        result[...,0] = 1000 if kind=='dominant-positive-1000' else -1000
    return result


def targets(torch, shape, kind):
    v=shape[-1]
    choices={'valid':[0,v-1], 'outside':[-100,-1,v,v+1,0,v-1],
             'wide':[2**32,2**32+v-1,2**63-1,-2**63,-2**32,0,v-1]}[kind]
    n=shape[0]*shape[1]
    return torch.tensor((choices*((n+len(choices)-1)//len(choices)))[:n],
                        dtype=torch.int64,device='cuda').reshape(shape[:2])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--megatron-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=26011)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('immutable output already exists')
    report={'status':'initializing','started_utc':datetime.now(timezone.utc).isoformat(),
            'harness_sha256':sha(__file__),'candidate_sha256':sha(args.candidate),
            'contract_sha256':sha(args.contract),'seed':args.seed,
            'comparison_device':'CPU float64 copies','atol':ATOL,'rtol':RTOL,
            'scope':'Contiguous BF16 logits, TP1 zero smoothing, no autocast; native loss/gradient/input semantics, no timer.',
            'cases':[]}
    dist=None
    parallel_state=None
    try:
        contract=json.loads(args.contract.read_text())
        assert contract['candidate_sha256']==report['candidate_sha256']
        assert contract['correctness']['atol']==ATOL and contract['correctness']['rtol']==RTOL
        manifest=json.loads((FORMAL/'source_manifest.json').read_text())
        report['native_sources_sha256']={name:sha(args.megatron_root/name) for name in manifest['files']}
        assert report['native_sources_sha256']==manifest['files']
        assert (args.megatron_root/'.git/HEAD').read_text().strip()==contract['megatron_commit']
        sys.path.insert(0,str(FORMAL))
        from runtime_compat import prepare_local_runtime
        report['runtime_compatibility']=prepare_local_runtime(args.megatron_root.resolve())
        import torch
        import torch.distributed as dist
        import triton
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
        report['environment']={'torch':torch.__version__,'triton':triton.__version__,
                               'gpu':torch.cuda.get_device_name(0),'autocast':torch.is_autocast_enabled()}
        assert report['environment']['gpu'] in ('BI-V150','Iluvatar BI-V150')
        assert not torch.is_autocast_enabled()
        spec=importlib.util.spec_from_file_location('verified_ce_candidate',args.candidate)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT='29932',RANK='0',WORLD_SIZE='1',LOCAL_RANK='0')
        torch.cuda.set_device(0)
        dist.init_process_group('nccl',rank=0,world_size=1)
        parallel_state.initialize_model_parallel()
        group=parallel_state.get_tensor_model_parallel_group()
        generator=torch.Generator().manual_seed(args.seed)
        for shape,distribution,label_kind,upstream_kind in itertools.product(SHAPES,DISTRIBUTIONS,LABELS,UPSTREAM):
            source=values(torch,shape,distribution,generator).to(device='cuda',dtype=torch.bfloat16)
            native=source.clone().requires_grad_()
            candidate=source.clone().requires_grad_()
            target=targets(torch,shape,label_kind)
            target_before=target.clone()
            gradient=torch.ones(shape[:2],device='cuda')
            if upstream_kind!='ones':
                gradient=torch.randn(shape[:2],generator=generator).to('cuda')
                gradient.reshape(-1)[::3]=0
            if upstream_kind=='strided-signed':
                storage=torch.empty((*shape[:1],shape[1]*2),device='cuda')
                storage[:,::2].copy_(gradient)
                gradient=storage[:,::2]
            native_loss=vocab_parallel_cross_entropy(native,target,tp_group=group)
            candidate_loss=module.cross_entropy(candidate,target)
            forward_unchanged=bool(torch.equal(native,source) and torch.equal(candidate,source))
            native_loss.backward(gradient)
            candidate_loss.backward(gradient)
            record={'shape':list(shape),'distribution':distribution,'labels':label_kind,
                    'upstream':upstream_kind,'loss':compare(torch,candidate_loss,native_loss),
                    'dlogits':compare(torch,candidate.grad,native.grad),
                    'forward_input_unchanged':forward_unchanged,
                    'backward_input_unchanged':bool(torch.equal(candidate,source) and torch.equal(native,source)),
                    'target_unchanged':bool(torch.equal(target,target_before)),
                    'output_fp32':candidate_loss.dtype==torch.float32,
                    'gradient_bf16':candidate.grad.dtype==torch.bfloat16}
            record['passed']=all((record['loss']['passed'],record['dlogits']['passed'],
                record['forward_input_unchanged'],record['backward_input_unchanged'],
                record['target_unchanged'],record['output_fp32'],record['gradient_bf16']))
            report['cases'].append(record)
        expected=len(SHAPES)*len(DISTRIBUTIONS)*len(LABELS)*len(UPSTREAM)
        report['expected_cases']=expected
        report['passed_cases']=sum(row['passed'] for row in report['cases'])
        report['status']='passed' if len(report['cases'])==expected and report['passed_cases']==expected else 'failed'
    except BaseException as error:
        report.update(status='error',error=str(error),traceback=traceback.format_exc())
    finally:
        if parallel_state is not None and dist is not None and dist.is_initialized():
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report.get(key) for key in ('status','expected_cases','passed_cases','error')}))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        raise SystemExit(main())

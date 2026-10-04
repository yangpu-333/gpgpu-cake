"""Explore installed vendor SDPA backends versus captured native fused attention.

This is a backend feasibility/numerics probe, not an accepted candidate or
training performance measurement. Internal SDPA rounding can differ from the
native BF16 score/probability boundaries and is reported rather than assumed.
"""
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import torch

BASE = Path('/tmp/kda-attention-stage31')
output = BASE / 'sdpa-backends-01.json'
assert not output.exists()
sys.path.insert(0, '/tmp/kda-bf16-continuation/megatron_training_entry')
from runtime_compat import prepare_local_runtime
runtime = prepare_local_runtime(Path('/tmp/kda-bf16-continuation/megatron-native-5be9626'))
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import attention_mask_func
native_softmax = FusedScaleMaskSoftmax(False,True,AttnMaskType.causal,True,
                                     attention_mask_func,False,None)
capture = BASE / 'native-default-capture-02/attention'

def compare(a,b):
    a,b=a.detach().cpu().double(),b.detach().cpu().double()
    difference=(a-b).abs();tolerance=.0003+.001*b.abs()
    return {'passed':bool(torch.isfinite(a).all() and torch.isfinite(b).all()
                          and (difference<=tolerance).all()),
            'max_abs':difference.max().item(),'bad_elements':int((difference>tolerance).sum()),
            'elements':a.numel(),'exact_equal':bool(torch.equal(a,b))}

def native(q,k,v):
    s,b,h,d=q.shape
    scores=torch.baddbmm(torch.empty((b*h,s,s),device='cuda',dtype=q.dtype),
                       q.reshape(s,b*h,d).transpose(0,1),
                       k.view(s,b*h,d).permute(1,2,0),beta=0.,alpha=.125)
    probs=native_softmax(scores.view(b,h,s,s),None).view(b*h,s,s)
    context=torch.bmm(probs,v.view(s,b*h,d).transpose(0,1))
    return context.view(b,h,s,d).permute(2,0,1,3).contiguous().view(s,b,h*d)

with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    payload=torch.load(capture/'inputs.pt',map_location='cuda',weights_only=False)
    s,b,h,d=payload['query'].shape
    backing=torch.empty((s,b,h,3*d),device='cuda',dtype=torch.bfloat16)
    for index,name in enumerate(('query','key','value')):
        backing[...,index*d:(index+1)*d].copy_(payload[name])
    q,k,v=[backing[...,index*d:(index+1)*d].detach().requires_grad_(True) for index in range(3)]
    dy=torch.load(capture/'upstream-grad.pt',map_location='cuda',weights_only=False)
    expected=native(q,k,v);gradients=torch.autograd.grad(expected,(q,k,v),dy)
    binding={'output':compare(expected,torch.load(capture/'output.pt',map_location='cuda',weights_only=False))}
    for name,gradient in zip(('query','key','value'),gradients):
        binding[name+'_grad']=compare(gradient,torch.load(capture/(name+'-grad.pt'),map_location='cuda',weights_only=False))
    assert all(row['passed'] and row['exact_equal'] for row in binding.values()),'native functional replay differs from actual formal capture'
    from torch.nn.attention import SDPBackend,sdpa_kernel
    rows=[]
    for name in ('FLASH_ATTENTION','EFFICIENT_ATTENTION','MATH','CUDNN_ATTENTION'):
        row={'backend':name,'internal_BF16_stages_assumed_equal':False}
        started=time.perf_counter()
        try:
            backend=getattr(SDPBackend,name)
            with sdpa_kernel(backend):
                result=torch.nn.functional.scaled_dot_product_attention(
                    q.permute(1,2,0,3),k.permute(1,2,0,3),v.permute(1,2,0,3),
                    dropout_p=0.,is_causal=True,scale=.125)
                result=result.permute(2,0,1,3).contiguous().view(s,b,h*d)
                actual_grads=torch.autograd.grad(result,(q,k,v),dy)
                torch.cuda.synchronize()
            row['checks']={'output':compare(result,expected)}
            for label,actual,reference in zip(('query_grad','key_grad','value_grad'),actual_grads,gradients):
                row['checks'][label]=compare(actual,reference)
            row['status']='passed_capture' if all(r['passed'] for r in row['checks'].values()) else 'numerically_rejected'
            row['grad_fn']=str(result.grad_fn)
        except Exception as error:
            row.update(status='unsupported',error_type=type(error).__name__,error=str(error))
        row['diagnostic_elapsed_seconds']=time.perf_counter()-started
        rows.append(row);print(json.dumps(row),flush=True)
    report={'schema':'vendor-sdpa-backend-exploration-v1','runtime':runtime,
            'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'shape':[s,b,h,d],'strides':list(q.stride()),'atol':.0003,'rtol':.001,
            'native_capture_replay':binding,'backends':rows,
            'scope':'Single captured main-shape diagnostic; no training or throughput acceptance; native default CUDA fused softmax reference'}
    output.write_text(json.dumps(report,indent=2)+'\n')

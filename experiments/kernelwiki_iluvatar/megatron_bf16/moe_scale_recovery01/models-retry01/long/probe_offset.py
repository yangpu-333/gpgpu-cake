"""Supplemental exact storage-offset oracle: main probe clones aligned tensors.

The frozen main probe remains unchanged. This adds two genuinely unaligned
contiguous views, with exact entry addresses recorded, using the same tolerance.
"""
import argparse,fcntl,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--version',required=True);v=a.parse_args().version
out=ROOT/('offset-'+v+'.json');assert not out.exists()
sys.path.insert(0,'/tmp/kda-bf16-continuation/megatron_training_entry')
from runtime_compat import prepare_local_runtime
prepare_local_runtime(Path('/tmp/kda-bf16-continuation/megatron-native-5be9626'))
import torch
torch.set_num_threads(4)
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
path=ROOT/'candidates'/v/'candidate.py'
spec=importlib.util.spec_from_file_location('weighted_candidate_offset',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r={'schema':'weighted-stage34-real-offset-probe-v1','candidate_sha256':sha(path),'verifier_sha256':sha(__file__),
 'main_probe_sha256':sha(ROOT/'probe.py'),'reason':'Main probe run() clones inputs, losing original storage offset; this additional gate preserves views. No original tolerance changed.',
 'atol':3e-4,'rtol':1e-3,'checks':[]}
with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  torch.manual_seed(34120)
  for n in (17,165):
   x=torch.randn(n*4096+1,device='cuda',dtype=torch.bfloat16)[1:].view(n,4096).requires_grad_()
   w=torch.rand(n+1,device='cuda',dtype=torch.bfloat16)[1:].view(n,1).requires_grad_()
   g=torch.randn(n,2048,device='cuda',dtype=torch.bfloat16)
   assert x.storage_offset()==w.storage_offset()==1 and x.data_ptr()%16==w.data_ptr()%16==2
   assert x.is_contiguous() and w.is_contiguous()
   before=(x.detach().clone(),w.detach().clone());seen=[]
   native=weighted_bias_swiglu_impl(x,None,w,False)
   native_grads=torch.autograd.grad(native,(x,w),g)
   cand=m.weighted_swiglu(x,w,native=lambda xx,ww:(seen.append(True) or weighted_bias_swiglu_impl(xx,None,ww,False)))
   cand_grads=torch.autograd.grad(cand,(x,w),g)
   checks={}
   for name,l,z in zip(('output','input_grad','weights_grad'),(native,)+native_grads,(cand,)+cand_grads):
    ll,zz=l.detach().cpu().to(torch.float64),z.detach().cpu().to(torch.float64)
    bad=int(torch.count_nonzero((ll-zz).abs()>3e-4+1e-3*ll.abs()))
    checks[name]={'passed':bool(torch.isfinite(ll).all() and torch.isfinite(zz).all()) and bad==0,
      'bad_elements':bad,'exact_equal':bool(torch.equal(l,z)),'max_abs':float((ll-zz).abs().max())}
   row={'n':n,'input_storage_offset':x.storage_offset(),'weights_storage_offset':w.storage_offset(),
      'pointer_alignment_mod16':[x.data_ptr()%16,w.data_ptr()%16],'checks':checks,
      'no_fallback':not seen,'inputs_unchanged':torch.equal(x,before[0]) and torch.equal(w,before[1])}
   row['passed']=row['no_fallback'] and row['inputs_unchanged'] and all(z['passed'] for z in checks.values())
   r['checks'].append(row)
  r['passed']=all(z['passed'] for z in r['checks']) and len(r['checks'])==2
 except Exception as e:r.update(passed=False,error_type=type(e).__name__,error=str(e))
 out.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r));raise SystemExit(0 if r['passed'] else 1)

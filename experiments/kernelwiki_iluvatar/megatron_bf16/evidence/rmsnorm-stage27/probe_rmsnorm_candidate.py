"""Independent native Torch/CPU verification and synchronized wall benchmark."""
import argparse,hashlib,importlib.util,json,time,statistics
from pathlib import Path
import torch

p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
assert not a.output.exists()
s=importlib.util.spec_from_file_location('rms_candidate',a.candidate);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
torch.manual_seed(27102);records=[]
for rows in (17,256,513):
 for scale in (0.0,0.01,1.0,10.0):
  x=(torch.randn(rows,512,device='cuda',dtype=torch.bfloat16)*scale).requires_grad_()
  w=(torch.randn(512,device='cuda',dtype=torch.bfloat16)*.1+1).requires_grad_()
  g=torch.randn_like(x)
  y=torch.nn.functional.rms_norm(x,(512,),w,1e-5);ng=torch.autograd.grad(y,(x,w),g)
  z=m.rms_norm(x,w,1e-5);cg=torch.autograd.grad(z,(x,w),g)
  rec={'rows':rows,'scale':scale,'checks':[]}
  for name,left,right in zip(('output','dx','dw'),(z,)+cg,(y,)+ng):
   l=left.detach().cpu().double();r=right.detach().cpu().double()
   rec['checks'].append({'name':name,'passed':bool(torch.isfinite(l).all() and torch.isfinite(r).all() and torch.allclose(l,r,atol=3e-4,rtol=1e-3)),
                         'bitwise_equal':torch.equal(l,r),'max_abs':float((l-r).abs().max())})
  records.append(rec)
passed=all(c['passed'] for r in records for c in r['checks'])
perf=[]
if passed:
 x=torch.randn(128,2,512,device='cuda',dtype=torch.bfloat16,requires_grad=True);w=torch.ones(512,device='cuda',dtype=torch.bfloat16,requires_grad=True);g=torch.randn_like(x)
 for backward in (False,True):
  samples={'native':[],'candidate':[]}
  for i in range(13):
   for arm in (('native','candidate') if i%2==0 else ('candidate','native')):
    def call():
     y=torch.nn.functional.rms_norm(x,(512,),w,1e-5) if arm=='native' else m.rms_norm(x,w,1e-5)
     if backward:torch.autograd.grad(y,(x,w),g)
    call();torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(20):call()
    torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000/20
    if i>=4:samples[arm].append(ms)
  perf.append({'backward':backward,'samples_ms':samples,'median_ms':{k:statistics.median(v) for k,v in samples.items()}})
out={'status':'passed' if passed else 'rejected','candidate_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),'checks':records,'performance':perf,
     'timer':'GPU synchronized time.perf_counter wall; standalone forward or forward+backward, not formal training'}
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'performance':perf,'failed':[r for r in records if not all(c['passed'] for c in r['checks'])]}))

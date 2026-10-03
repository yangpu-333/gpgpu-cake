"""Independent actual three-dimensional layout and fresh-input launch-cache gates."""
import argparse,hashlib,importlib.util,json,torch
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('output',type=Path);a=p.parse_args();assert not a.output.exists()
s=importlib.util.spec_from_file_location('c',a.candidate);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
torch.manual_seed(27103);checks=[]
for rows in (128,63):
 for strided in (False,True):
  for i in range(3):
   x=torch.randn((2,rows,512) if strided else (rows,2,512),device='cuda',dtype=torch.bfloat16)
   if strided:x=x.transpose(0,1)
   x.requires_grad_();w=torch.randn(512,device='cuda',dtype=torch.bfloat16,requires_grad=True);g=torch.randn_like(x)
   n=torch.nn.functional.rms_norm(x,(512,),w,1e-5);ng=torch.autograd.grad(n,(x,w),g)
   y=m.rms_norm(x,w,1e-5);cg=torch.autograd.grad(y,(x,w),g)
   for name,l,r in zip(('output','dx','dw'),(y,)+cg,(n,)+ng):
    lc,rc=l.detach().cpu().double(),r.detach().cpu().double()
    checks.append({'rows':rows,'strided':strided,'repeat':i,'name':name,'passed':bool(torch.isfinite(lc).all() and torch.isfinite(rc).all() and torch.allclose(lc,rc,atol=3e-4,rtol=1e-3)),
                   'bitwise_equal':torch.equal(lc,rc),'max_abs':float((lc-rc).abs().max())})
report={'candidate_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),'status':'passed' if all(c['passed'] for c in checks) else 'rejected','checks':checks}
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'checks':len(checks),'all_bitwise':all(c['bitwise_equal'] for c in checks)}))

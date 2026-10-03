"""Explore documented installed CompiledKernel call arity, CPU check output."""
import importlib.util,json,statistics,time,torch
from pathlib import Path
s=importlib.util.spec_from_file_location('c',Path('/tmp/kda-rmsnorm-stage27/candidate-002-corex.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
x=torch.randn(256,512,device='cuda',dtype=torch.bfloat16);r=torch.ones(256,1,device='cuda',dtype=torch.bfloat16);w=torch.ones(512,device='cuda',dtype=torch.bfloat16);o=torch.empty_like(x);t=torch.empty_like(x)
args=(x,r,w,o,t,x.numel(),512)
kernel=m._fused_affine_fwd_kernel[(128,)](*args,BLOCK=1024,num_warps=4)
results=[]
for n in (0,1):
 try:
  values=args if n==0 else args+(1024,)
  kernel[(128,1,1)](*values);torch.cuda.synchronize()
  exact=torch.equal(o.cpu(),x.cpu())
  samples=[]
  for i in range(9):
   torch.cuda.synchronize();start=time.perf_counter()
   for j in range(100):kernel[(128,1,1)](*values)
   torch.cuda.synchronize();samples.append((time.perf_counter()-start)*10)
  results.append({'extra_constexpr':n,'output_exact':exact,'median_ms':statistics.median(samples),'samples_ms':samples})
 except Exception as e:results.append({'extra_constexpr':n,'error':repr(e)})
print(json.dumps(results,indent=2))

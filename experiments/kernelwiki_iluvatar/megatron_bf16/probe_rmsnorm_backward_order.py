"""Pin native BF16 analytic backward operation order before fusion."""
import json,torch
torch.manual_seed(27104);records=[]
for scale in (0.0,.01,1.,10.):
 x=(torch.randn(256,512,device='cuda',dtype=torch.bfloat16)*scale).requires_grad_();w=torch.randn(512,device='cuda',dtype=torch.bfloat16,requires_grad=True);g=torch.randn_like(x)
 y=torch.nn.functional.rms_norm(x,(512,),w,1e-5);dx,dw=torch.autograd.grad(y,(x,w),g)
 with torch.no_grad():
  v=x.pow(2).mean(-1,keepdim=True);v.add_(1.0013580322265625e-05);r=torch.rsqrt(v);t=x*r;gw=g*w
  cube=r.pow(3).cpu();records.append({'scale':scale,'name':'pow3_order','staged_bf16_equal':torch.equal(cube,((r*r)*r).cpu()),'float_cube_equal':torch.equal(cube,((r.float()*r.float())*r.float()).bfloat16().cpu())})
  dr=(gw*x).sum_to_size(r.shape)
  dv=dr*(-.5*r.pow(3));dextra=(dv/512)*(2*x);manual=gw*r+dextra;mw=(g*t).sum_to_size(w.shape)
  for name,a,b in (('dx',manual,dx),('dw',mw,dw)):
   ac,bc=a.cpu().double(),b.cpu().double();records.append({'scale':scale,'name':name,'bitwise_equal':torch.equal(ac,bc),'max_abs':float((ac-bc).abs().max())})
print(json.dumps(records,indent=2))

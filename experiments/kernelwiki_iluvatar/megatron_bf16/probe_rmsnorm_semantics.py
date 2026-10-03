"""Independent CPU comparison of vendor RMSNorm BF16 rounding formulas."""
import json
import torch

torch.manual_seed(27101)
records=[]
for shape in ((17,512),(256,512),(513,512)):
    x=torch.randn(shape,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    w=torch.randn(512,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    y=torch.nn.functional.rms_norm(x,(512,),w,1e-5)
    xf=x.float(); r=torch.rsqrt(xf.square().mean(-1,keepdim=True)+1e-5)
    formulas={'fp32_all_then_cast':(xf*r*w.float()).to(x.dtype),
              'cast_normalized_then_weight':(xf*r).to(x.dtype)*w,
              'cast_inv_then_bf16_products':(x*r.to(x.dtype))*w,
              'native_bf16_stages':(x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+torch.tensor(1e-5,dtype=x.dtype,device=x.device)))*w}
    entry={'shape':shape,'native_dtype':str(y.dtype),'formulas':{}}
    for name,v in formulas.items():
        yc,vc=y.detach().cpu().float(),v.detach().cpu().float()
        entry['formulas'][name]={'exact':torch.equal(yc,vc),'max_abs':float((yc-vc).abs().max())}
    records.append(entry)
print(json.dumps({'torch':torch.__version__,'records':records},indent=2))

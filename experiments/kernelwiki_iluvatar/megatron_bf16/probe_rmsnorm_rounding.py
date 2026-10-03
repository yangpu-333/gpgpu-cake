"""Observe vendor BF16 RMSNorm rounding; diagnostic only, no fused candidate."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path


def metrics(torch,a,b):
    a,b=a.detach().cpu().double(),b.detach().cpu().double()
    return {'exact':bool(torch.equal(a,b)),'max_abs_error':float((a-b).abs().max()),
            'failed_elements':int(((a-b).abs()>3e-4+1e-3*b.abs()).sum()),'elements':a.numel()}


def formula(torch,x,w,kind):
    dims=(-1,)
    if kind=='bf16_every_stage':
        return x*torch.rsqrt((x*x).mean(dim=dims,keepdim=True)+1e-5)*w
    z=x.float()
    inverse=torch.rsqrt((z*z).mean(dim=dims,keepdim=True)+1e-5)
    if kind=='fp32_stat_bf16_inverse':
        return (x*inverse.to(x.dtype))*w
    if kind=='fp32_normalize_bf16_weight_multiply':
        return (z*inverse).to(x.dtype)*w
    return (z*inverse*w.float()).to(x.dtype)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('immutable evidence exists')
    import torch
    torch.cuda.set_device(0)
    g=torch.Generator().manual_seed(26021)
    report={'scope':'Vendor Torch BF16 RMSNorm mathematical decomposition probe, no fusion or timing evidence.',
            'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'rows':[]}
    for shape in ((128,2,512),(256,2,512),(512,2,512)):
        for scale in (0.001,1,100):
            source=(torch.randn(shape,generator=g)*scale).to(device='cuda',dtype=torch.bfloat16)
            weight=(torch.randn(shape[-1],generator=g)*0.1+1).to(device='cuda',dtype=torch.bfloat16)
            upstream=torch.randn(shape,generator=g).to(device='cuda',dtype=torch.bfloat16)
            native_x=source.clone().requires_grad_();native_w=weight.clone().requires_grad_()
            native_y=torch.nn.functional.rms_norm(native_x,(shape[-1],),native_w,1e-5)
            native_y.backward(upstream)
            for kind in ('fp32_every_stage','bf16_every_stage','fp32_stat_bf16_inverse','fp32_normalize_bf16_weight_multiply'):
                x=source.clone().requires_grad_();w=weight.clone().requires_grad_()
                y=formula(torch,x,w,kind);y.backward(upstream)
                report['rows'].append({'shape':list(shape),'scale':scale,'formula':kind,
                    'output':metrics(torch,y,native_y),'dx':metrics(torch,x.grad,native_x.grad),
                    'dweight':metrics(torch,w.grad,native_w.grad)})
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ('fp32_every_stage','bf16_every_stage','fp32_stat_bf16_inverse','fp32_normalize_bf16_weight_multiply'):
        rows=[r for r in report['rows'] if r['formula']==kind]
        print(kind,{key:sum(r[key]['exact'] for r in rows) for key in ('output','dx','dweight')})


if __name__=='__main__':
    with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()

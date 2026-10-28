"""Stream full tensors through original CPU-float64 tolerances; no giant pack."""
import hashlib,math
import numpy as np
import torch
ATOL=3e-4;RTOL=1e-3

class Metrics:
    def __init__(self):self.count=0;self.bad=0;self.max_abs=0.;self.diff2=0.;self.ref2=0.;self.finite=True
    def add(self,a,b):
        a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
        self.finite &= bool(np.isfinite(a).all()and np.isfinite(b).all())
        diff=np.abs(a-b);self.count+=a.size;self.bad+=int(np.count_nonzero(diff>ATOL+RTOL*np.abs(b)))
        self.max_abs=max(self.max_abs,float(diff.max(initial=0)));self.diff2+=float(np.sum(diff*diff,dtype=np.float64));self.ref2+=float(np.sum(b*b,dtype=np.float64))
    def result(self,relative_limit):
        rel=math.sqrt(self.diff2/self.ref2)if self.ref2 else (0. if not self.diff2 else math.inf)
        return {'elements':self.count,'nonfinite':not self.finite,'bad_original_tolerance':self.bad,'max_abs':self.max_abs,'relative_L2':rel,'additional_relative_L2_limit':relative_limit,'passed':self.finite and self.bad==0 and rel<=relative_limit}

def full_check(x,g,mask,ny,nd,cy,cd,scale,guard):
    """Native vs candidate AND independent CPU-double softmax/derivative, all rows."""
    N=x.shape[-1];rows=x.numel()//N;chunk_rows=128
    tensors=[x,g,ny,nd,cy,cd];flat=[a.detach().reshape(rows,N)for a in tensors]
    metrics={k:Metrics()for k in ('forward_vs_native','backward_vs_native','native_forward_vs_CPU64','candidate_forward_vs_CPU64','native_backward_vs_CPU64','candidate_backward_vs_CPU64')}
    hashes={k:hashlib.sha256()for k in ('input','upstream','native_forward','native_backward','candidate_forward','candidate_backward')}
    max_row_sum=0.;masked_gradient_bad=0
    # Broadcast masks are copied in row chunks, never materialized as a full CPU tensor.
    expanded=mask.expand(x.shape)
    B,H=x.shape[:2]
    for start in range(0,rows,chunk_rows):
        end=min(rows,start+chunk_rows);guard.check_live(128*1024**2)
        host=[a[start:end].cpu()for a in flat]
        for key,t in zip(hashes,host):hashes[key].update(memoryview(t.contiguous().view(torch.uint8).numpy()).cast('B'))
        xx,gg,nn,ng,cc,cg=[t.float().numpy().astype(np.float64)for t in host]
        # Mask indexing honors batch/head broadcasting and strided mask views.
        rid=torch.arange(start,end,device=x.device);bb=rid//(H*N);hh=(rid//N)%H;qq=rid%N
        mm=expanded[bb,hh,qq,:].cpu().numpy()
        scaled=(xx.astype(np.float32)*np.float32(scale)).astype(np.float64)
        scaled[mm]=-10000.;scaled-=np.max(scaled,axis=-1,keepdims=True)
        ex=np.exp(scaled);ref=ex/np.sum(ex,axis=-1,keepdims=True)
        dx=ref*(gg-np.sum(gg*ref,axis=-1,keepdims=True));dx[mm]=0.;dx=dx*float(np.float32(scale))
        if x.dtype==torch.bfloat16:dx=torch.from_numpy(dx.astype(np.float32)).to(torch.bfloat16).float().numpy().astype(np.float64)
        metrics['forward_vs_native'].add(cc,nn);metrics['backward_vs_native'].add(cg,ng)
        metrics['native_forward_vs_CPU64'].add(nn,ref);metrics['candidate_forward_vs_CPU64'].add(cc,ref)
        metrics['native_backward_vs_CPU64'].add(ng,dx);metrics['candidate_backward_vs_CPU64'].add(cg,dx)
        max_row_sum=max(max_row_sum,float(np.max(np.abs(cc.sum(-1)-1))))
        masked_gradient_bad+=int(np.count_nonzero(cg[mm]))
        del host,xx,gg,nn,ng,cc,cg,mm,scaled,ex,ref,dx,rid,bb,hh,qq
    out={k:m.result(1e-3 if 'backward'in k and x.dtype==torch.bfloat16 else 1e-5)for k,m in metrics.items()}
    return {'CPU_float64_all_elements':True,'atol':ATOL,'rtol':RTOL,'metrics':out,'raw_checked_tensor_SHA256':{k:h.hexdigest()for k,h in hashes.items()},'max_probability_row_sum_error':max_row_sum,'masked_input_gradient_nonzero':masked_gradient_bad,'passed':all(v['passed']for v in out.values())and max_row_sum<1e-5 and masked_gradient_bad==0}

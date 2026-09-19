"""Sequential-in-time fused kernel; exact external official GDN prefill contract."""
import math
import torch
import triton
import triton.language as tl
from gdn_decode import bf16_load


@triton.jit
def prefill(Q,K,V,S,AL,A,DB,B,CU,O,NS,SCALE:tl.constexpr,HAS_STATE:tl.constexpr,ROWS:tl.constexpr):
    seq_head = tl.program_id(0)
    seq, head = seq_head//8, seq_head%8
    rows = tl.program_id(1)*ROWS+tl.arange(0,ROWS)
    cols = tl.arange(0,128)
    offsets = seq_head*128*128+rows[:,None]*128+cols[None,:]
    start, end = tl.load(CU+seq), tl.load(CU+seq+1)
    state = tl.full((ROWS,128),0.,tl.float32)
    if HAS_STATE:
        if end > start:
            state = tl.load(S+offsets)
    al, db = tl.load(AL+head), tl.load(DB+head)
    for t in range(start,end):
        q = bf16_load(Q,(t*4+head//2)*128+cols)
        k = bf16_load(K,(t*4+head//2)*128+cols)
        v = bf16_load(V,(t*8+head)*128+rows)
        av, bv = bf16_load(A,t*8+head), bf16_load(B,t*8+head)
        x = av+db
        softplus = tl.where(x>20.,x,tl.log(1.+tl.exp(tl.minimum(x,20.))))
        g = tl.exp(-tl.exp(al)*softplus)
        beta = 1./(1.+tl.exp(-bv))
        old = state*g
        old_v = tl.sum(old*k[None,:],1)
        new_v = beta*v+(1.-beta)*old_v
        state = old-k[None,:]*old_v[:,None]+k[None,:]*new_v[:,None]
        out = tl.sum(state*q[None,:],1)*SCALE
        bits = out.to(tl.uint32,bitcast=True)
        rounded = ((bits+0x7fff+((bits>>16)&1))>>16).to(tl.uint16)
        tl.store(O+(t*8+head)*128+rows,rounded)
    tl.store(NS+offsets,state)


def candidate(q,k,v,state,A_log,a,dt_bias,b,cu_seqlens,scale,rows=8):
    if rows not in (4,8,16):
        raise ValueError('unsupported schedule')
    ns = torch.empty((cu_seqlens.numel()-1,8,128,128),device=q.device,dtype=torch.float32)
    out = torch.empty((q.shape[0],8,128),device=q.device,dtype=torch.bfloat16)
    prefill[((cu_seqlens.numel()-1)*8,128//rows)](
        q.view(torch.uint16),k.view(torch.uint16),v.view(torch.uint16),state if state is not None else ns,
        A_log,a.view(torch.uint16),dt_bias,b.view(torch.uint16),cu_seqlens,out.view(torch.uint16),ns,
        SCALE=float(scale or 1/math.sqrt(128)),HAS_STATE=state is not None,ROWS=rows,
        num_warps=4,enable_fp_fusion=False)
    return out,ns


def triton_candidate(q,k,v,state,A_log,a,dt_bias,b,cu_seqlens,scale,rows=8):
    """Stable export used by the isolated KDA candidate evaluator."""
    return candidate(q,k,v,state,A_log,a,dt_bias,b,cu_seqlens,scale,rows=rows)

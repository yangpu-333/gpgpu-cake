"""Fuse cast/scale/mask around the exact installed native softmax algorithm."""
import torch,triton
import triton.language as tl
from fused_scale_mask_softmax import validate

@triton.jit
def _prepare(X,Y,M,SCALE,N:tl.constexpr,H:tl.constexpr,TOTAL:tl.constexpr,M0:tl.constexpr,M1:tl.constexpr,M2:tl.constexpr,M3:tl.constexpr,CAUSAL:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<TOTAL;col=i%N;row=i//N;q=row%N;head=(row//N)%H;batch=row//(N*H)
 x=tl.load(X+i,mask=valid,other=0).to(tl.float32);value=x*SCALE
 if CAUSAL:excluded=col>q
 else:excluded=tl.load(M+batch*M0+head*M1+q*M2+col*M3,mask=valid,other=1).to(tl.int1)
 tl.store(Y+i,tl.where(excluded,-10000.0,value),mask=valid)

@triton.jit
def _prepare_backward(G,DX,M,SCALE,N:tl.constexpr,H:tl.constexpr,TOTAL:tl.constexpr,M0:tl.constexpr,M1:tl.constexpr,M2:tl.constexpr,M3:tl.constexpr,CAUSAL:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);valid=i<TOTAL;col=i%N;row=i//N;q=row%N;head=(row//N)%H;batch=row//(N*H)
 g=tl.load(G+i,mask=valid,other=0).to(tl.float32)
 if CAUSAL:excluded=col>q
 else:excluded=tl.load(M+batch*M0+head*M1+q*M2+col*M3,mask=valid,other=1).to(tl.int1)
 tl.store(DX+i,tl.where(excluded,0.0,g)*SCALE,mask=valid)

class _Prepare(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,scale,mask):
  mask=validate(x,scale,mask);strides=(0,0,0,0)if mask is None else mask.stride();y=torch.empty(x.shape,device=x.device,dtype=torch.float32)
  _prepare[(triton.cdiv(x.numel(),1024),)](x,y,x if mask is None else mask,float(scale),x.shape[-1],x.shape[1],x.numel(),*strides,mask is None,1024,num_warps=4,enable_fp_fusion=False)
  ctx.scale=float(scale);ctx.dtype=x.dtype;ctx.shape=x.shape;ctx.strides=strides;ctx.causal=mask is None
  if mask is not None:ctx.save_for_backward(mask)
  return y
 @staticmethod
 def backward(ctx,g):
  g=g.contiguous();mask=None if ctx.causal else ctx.saved_tensors[0];dx=torch.empty(ctx.shape,device=g.device,dtype=ctx.dtype)
  _prepare_backward[(triton.cdiv(g.numel(),1024),)](g,dx,g if mask is None else mask,ctx.scale,g.shape[-1],g.shape[1],g.numel(),*ctx.strides,ctx.causal,1024,num_warps=4,enable_fp_fusion=False)
  return dx,None,None

def native_softmax_fused_boundaries(x,scale,mask=None):
 return torch.softmax(_Prepare.apply(x,float(scale),mask),dim=-1)

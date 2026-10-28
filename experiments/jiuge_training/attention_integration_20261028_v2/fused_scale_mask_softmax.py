"""Two-launch scale/mask/softmax forward + backward for CoreX Triton.

Contract: contiguous CUDA BF16/FP32 [B,H,S,S], FP32 output/intermediates,
finite mask fill -10000, finite scale. mask=None means causal; explicit bool
mask broadcasts to X. No dropout, QK/PV, optimizer or model integration.
Row reduction follows the general Triton fused-softmax algorithm:
https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
Target compatibility and timings must be established on the actual BI-V150.
"""
import math
import torch
import triton
import triton.language as tl

@triton.jit
def _forward(X,Y,M,SCALE,N:tl.constexpr,H:tl.constexpr,
             M0:tl.constexpr,M1:tl.constexpr,M2:tl.constexpr,M3:tl.constexpr,
             CAUSAL:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BLOCK);valid=col<N
    q=row%N;head=(row//N)%H;batch=row//(N*H)
    x=tl.load(X+row*N+col,mask=valid,other=0).to(tl.float32)
    scaled=x*SCALE
    if CAUSAL:
        excluded=col>q
    else:
        excluded=tl.load(M+batch*M0+head*M1+q*M2+col*M3,mask=valid,other=1).to(tl.int1)
    logits=tl.where(excluded,-10000.0,scaled)
    logits=tl.where(valid,logits,-float('inf'))
    shifted=logits-tl.max(logits,axis=0)
    numerator=tl.exp(shifted)
    output=numerator/tl.sum(numerator,axis=0)
    tl.store(Y+row*N+col,output,mask=valid)

@triton.jit
def _backward(Y,G,DX,M,SCALE,N:tl.constexpr,H:tl.constexpr,
              M0:tl.constexpr,M1:tl.constexpr,M2:tl.constexpr,M3:tl.constexpr,
              CAUSAL:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);col=tl.arange(0,BLOCK);valid=col<N
    q=row%N;head=(row//N)%H;batch=row//(N*H)
    y=tl.load(Y+row*N+col,mask=valid,other=0).to(tl.float32)
    g=tl.load(G+row*N+col,mask=valid,other=0).to(tl.float32)
    weighted=g*y
    # Keep the FP32 product-before-reduction and epilogue ordering.
    grad=weighted-y*tl.sum(weighted,axis=0)
    if CAUSAL:
        excluded=col>q
    else:
        excluded=tl.load(M+batch*M0+head*M1+q*M2+col*M3,mask=valid,other=1).to(tl.int1)
    grad=tl.where(excluded,0.0,grad)
    grad=grad*SCALE
    tl.store(DX+row*N+col,grad,mask=valid)

COMPILED={}

def validate(x,scale,mask):
    if not x.is_cuda or x.dtype not in (torch.bfloat16,torch.float32):
        raise ValueError('CUDA BF16/FP32 required')
    if x.ndim!=4 or x.shape[-2]!=x.shape[-1] or not all(x.shape) or x.shape[-1]>4096:
        raise ValueError('Nonempty [B,H,S,S], S<=4096 required')
    if not x.is_contiguous():raise ValueError('Contiguous input required')
    if not math.isfinite(float(scale)):raise ValueError('Finite scale required')
    if mask is not None:
        if mask.device!=x.device or mask.dtype!=torch.bool or mask.ndim>4:
            raise ValueError('Boolean mask on same CUDA device required')
        try:mask=mask.expand(x.shape)
        except RuntimeError as e:raise ValueError('Mask must broadcast to input')from e
    return mask

class _ScaleMaskSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,scale,mask):
        mask=validate(x,scale,mask);N=x.shape[-1];H=x.shape[1]
        strides=(0,0,0,0)if mask is None else mask.stride()
        y=torch.empty(x.shape,device=x.device,dtype=torch.float32)
        block=triton.next_power_of_2(N);warps=4 if N<=1024 else 8
        k=_forward[(x.numel()//N,)](x,y,x if mask is None else mask,float(scale),N,H,*strides,mask is None,block,num_warps=warps,num_stages=1,enable_fp_fusion=False)
        COMPILED['forward']=k
        ctx.save_for_backward(y)if mask is None else ctx.save_for_backward(y,mask)
        ctx.causal=mask is None;ctx.dtype=x.dtype;ctx.scale=float(scale);ctx.strides=strides;ctx.warps=warps
        return y
    @staticmethod
    def backward(ctx,dy):
        y=ctx.saved_tensors[0];mask=None if ctx.causal else ctx.saved_tensors[1]
        dy=dy.contiguous();dx=torch.empty(y.shape,device=y.device,dtype=ctx.dtype)
        N=y.shape[-1];H=y.shape[1];block=triton.next_power_of_2(N)
        k=_backward[(y.numel()//N,)](y,dy,dx,y if mask is None else mask,ctx.scale,N,H,*ctx.strides,ctx.causal,block,num_warps=ctx.warps,num_stages=1,enable_fp_fusion=False)
        COMPILED['backward']=k
        return dx,None,None

def fused_scale_mask_softmax(x,scale,mask=None):
    return _ScaleMaskSoftmax.apply(x,float(scale),mask)

def native_scale_mask_softmax(x,scale,mask):
    """The precomputed native boolean mask is outside the timed call."""
    return torch.softmax((x.float()*float(scale)).masked_fill(mask,-10000.0),dim=-1)

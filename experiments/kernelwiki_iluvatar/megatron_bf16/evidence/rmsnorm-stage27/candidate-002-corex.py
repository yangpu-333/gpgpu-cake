import torch
import triton
import triton.language as tl

# BF16(1e-5) materialized as an exact host float; avoids a per-call CUDA alloc.
_EPS_BF16 = 1.0013580322265625e-05
_BLOCK = 1024


@triton.jit
def _fused_affine_fwd_kernel(
    x_ptr, rstd_ptr, w_ptr, out_ptr, t_ptr,
    numel, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    row = offs // N
    col = offs % N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)
    # First BF16 product; round to BF16 before the second product (native parity).
    t = x * rstd
    t = t.to(tl.bfloat16).to(tl.float32)
    out = t * w
    tl.store(t_ptr + offs, t.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + offs, out.to(tl.bfloat16), mask=mask)


@triton.jit
def _fused_affine_bwd_kernel(
    g_ptr, w_ptr, rstd_ptr, x_ptr, t_ptr,
    dx_ptr, rstd_term_ptr, wt_term_ptr,
    numel, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    row = offs // N
    col = offs % N
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(t_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # Native BF16 order; round to BF16 after each elementwise product.
    gw = g * w
    gw = gw.to(tl.bfloat16).to(tl.float32)
    dx = gw * rstd
    rstd_term = gw * x
    wt_term = g * t
    tl.store(dx_ptr + offs, dx.to(tl.bfloat16), mask=mask)
    tl.store(rstd_term_ptr + offs, rstd_term.to(tl.bfloat16), mask=mask)
    tl.store(wt_term_ptr + offs, wt_term.to(tl.bfloat16), mask=mask)


class _FusedAffine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rstd, weight):
        x_c = x.contiguous()
        rstd_c = rstd.contiguous()
        w_c = weight.contiguous()
        out = torch.empty_like(x_c)
        t = torch.empty_like(x_c)
        N = x_c.shape[-1]
        numel = x_c.numel()
        grid = (triton.cdiv(numel, _BLOCK),)
        _fused_affine_fwd_kernel[grid](
            x_c, rstd_c, w_c, out, t, numel, N,
            BLOCK=_BLOCK, num_warps=4,
        )
        ctx.save_for_backward(x_c, rstd_c, w_c, t)
        return out

    @staticmethod
    def backward(ctx, g):
        x, rstd, weight, t = ctx.saved_tensors
        g = g.contiguous()
        dx = torch.empty_like(x)
        rstd_term = torch.empty_like(x)
        wt_term = torch.empty_like(x)
        N = x.shape[-1]
        numel = x.numel()
        grid = (triton.cdiv(numel, _BLOCK),)
        _fused_affine_bwd_kernel[grid](
            g, weight, rstd, x, t, dx, rstd_term, wt_term, numel, N,
            BLOCK=_BLOCK, num_warps=4,
        )
        # Native reductions kept in Torch, order unchanged.
        g_rstd = rstd_term.sum_to_size(rstd.shape)
        g_weight = wt_term.sum_to_size(weight.shape)
        return dx, g_rstd, g_weight


def _supported(x, weight, eps):
    if not (x.is_cuda and weight.is_cuda):
        return False
    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        return False
    if x.dim() < 1 or x.shape[-1] != 512:
        return False
    if weight.dim() != 1 or weight.numel() != 512:
        return False
    if eps != 1e-5:
        return False
    return True


def rms_norm(x, weight, eps):
    if not _supported(x, weight, eps):
        raise NotImplementedError(
            "fused rms_norm supports only BF16 CUDA tensors with hidden=512, "
            "1D weight=512, and eps==1e-5"
        )
    # Native normalization branch kept in Torch for exact autograd parity.
    var = x.pow(2).mean(-1, keepdim=True)
    var.add_(_EPS_BF16)
    rstd = torch.rsqrt(var)
    # rstd stays outside the Function so its grad flows through pow/mean/rsqrt.
    return _FusedAffine.apply(x, rstd, weight)

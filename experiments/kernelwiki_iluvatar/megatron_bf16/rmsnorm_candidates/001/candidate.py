import torch
import triton
import triton.language as tl


@triton.jit
def _fused_affine_fwd_kernel(
    x_ptr, rstd_ptr, w_ptr, out_ptr,
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
    # First BF16 elementwise product; materialize BF16 rounding before the
    # second product so no single multiply spans both stages (native parity).
    t = x * rstd
    t = t.to(tl.bfloat16).to(tl.float32)
    out = t * w
    tl.store(out_ptr + offs, out.to(tl.bfloat16), mask=mask)


def _launch_fwd(x, rstd, w, out, numel, N):
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    try:
        _fused_affine_fwd_kernel[grid](
            x, rstd, w, out, numel, N,
            BLOCK=BLOCK, num_warps=4, enable_fp_fusion=False,
        )
    except TypeError:
        _fused_affine_fwd_kernel[grid](
            x, rstd, w, out, numel, N,
            BLOCK=BLOCK, num_warps=4,
        )


class _FusedAffine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rstd, weight):
        x_c = x.contiguous()
        rstd_c = rstd.contiguous()
        w_c = weight.contiguous()
        out = torch.empty_like(x_c)
        N = x_c.shape[-1]
        numel = x_c.numel()
        _launch_fwd(x_c, rstd_c, w_c, out, numel, N)
        ctx.save_for_backward(x_c, rstd_c, w_c)
        return out

    @staticmethod
    def backward(ctx, g):
        x, rstd, weight = ctx.saved_tensors
        g = g.contiguous()
        # Native BF16 order, unfused: out = (bf16(x*rstd)) * weight.
        gw = g * weight
        g_input = gw * rstd
        g_rstd = (gw * x).sum_to_size(rstd.shape)
        g_weight = (g * (x * rstd)).sum_to_size(weight.shape)
        return g_input, g_rstd, g_weight


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
    eps_bf16 = torch.tensor(eps, dtype=torch.bfloat16, device=x.device)
    var = x.pow(2).mean(-1, keepdim=True)
    var = var + eps_bf16
    rstd = torch.rsqrt(var)
    # rstd stays outside the Function so its grad flows through pow/mean/rsqrt.
    return _FusedAffine.apply(x, rstd, weight)

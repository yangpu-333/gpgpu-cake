import torch
import triton
import triton.language as tl

# BF16(1e-5) materialized as an exact host float; avoids a per-call CUDA alloc.
_EPS_BF16 = 1.0013580322265625e-05
_BLOCK = 1024

# Process-local caches of compiled launch closures ONLY (never data/results).
_FWD_RUNNERS = {}
_BWD1_RUNNERS = {}
_BWD2_RUNNERS = {}


@triton.jit
def _fused_affine_fwd_kernel(
    x_ptr, r_ptr, w_ptr, out_ptr, t_ptr,
    numel, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    row = offs // N
    col = offs % N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)
    # First BF16 product; round to BF16 before the second product (native parity).
    t = x * r
    t = t.to(tl.bfloat16).to(tl.float32)
    out = t * w
    tl.store(t_ptr + offs, t.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + offs, out.to(tl.bfloat16), mask=mask)


@triton.jit
def _bwd1_kernel(
    g_ptr, w_ptr, x_ptr, t_ptr,
    gw_ptr, dr_term_ptr, dw_term_ptr,
    numel, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    col = offs % N
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + col, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    t = tl.load(t_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # Native BF16 order; gw rounded once, then reused.
    gw = g * w
    gw = gw.to(tl.bfloat16).to(tl.float32)
    dr_term = gw * x
    dw_term = g * t
    tl.store(gw_ptr + offs, gw.to(tl.bfloat16), mask=mask)
    tl.store(dr_term_ptr + offs, dr_term.to(tl.bfloat16), mask=mask)
    tl.store(dw_term_ptr + offs, dw_term.to(tl.bfloat16), mask=mask)


@triton.jit
def _bwd2_kernel(
    gw_ptr, r_ptr, x_ptr, dr_ptr, dx_ptr,
    numel, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    row = offs // N
    gw = tl.load(gw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    dr = tl.load(dr_ptr + row, mask=mask, other=0.0).to(tl.float32)
    # BF16 barrier after each native stage, native order preserved.
    cube = r * r
    cube = cube.to(tl.bfloat16).to(tl.float32)
    cube = cube * r
    cube = cube.to(tl.bfloat16).to(tl.float32)
    scale = -0.5 * cube
    scale = scale.to(tl.bfloat16).to(tl.float32)
    dv = dr * scale
    dv = dv.to(tl.bfloat16).to(tl.float32)
    dsq = dv / N
    dsq = dsq.to(tl.bfloat16).to(tl.float32)
    two_x = 2.0 * x
    two_x = two_x.to(tl.bfloat16).to(tl.float32)
    extra = dsq * two_x
    extra = extra.to(tl.bfloat16).to(tl.float32)
    direct = gw * r
    direct = direct.to(tl.bfloat16).to(tl.float32)
    dx = direct + extra
    tl.store(dx_ptr + offs, dx.to(tl.bfloat16), mask=mask)


def _fwd_launch(x, r, w, out, t, numel, N):
    grid0 = triton.cdiv(numel, _BLOCK)
    key = (x.device.index, numel, N)
    runner = _FWD_RUNNERS.get(key)
    if runner is None:
        compiled = _fused_affine_fwd_kernel[(grid0,)](
            x, r, w, out, t, numel, N, BLOCK=_BLOCK, num_warps=4,
        )
        _FWD_RUNNERS[key] = compiled[(grid0, 1, 1)]
    else:
        runner(x, r, w, out, t, numel, N)


def _bwd1_launch(g, w, x, t, gw, dr_term, dw_term, numel, N):
    grid0 = triton.cdiv(numel, _BLOCK)
    key = (x.device.index, numel, N)
    runner = _BWD1_RUNNERS.get(key)
    if runner is None:
        compiled = _bwd1_kernel[(grid0,)](
            g, w, x, t, gw, dr_term, dw_term, numel, N, BLOCK=_BLOCK, num_warps=4,
        )
        _BWD1_RUNNERS[key] = compiled[(grid0, 1, 1)]
    else:
        runner(g, w, x, t, gw, dr_term, dw_term, numel, N)


def _bwd2_launch(gw, r, x, dr, dx, numel, N):
    grid0 = triton.cdiv(numel, _BLOCK)
    key = (x.device.index, numel, N)
    runner = _BWD2_RUNNERS.get(key)
    if runner is None:
        compiled = _bwd2_kernel[(grid0,)](
            gw, r, x, dr, dx, numel, N, BLOCK=_BLOCK, num_warps=4,
        )
        _BWD2_RUNNERS[key] = compiled[(grid0, 1, 1)]
    else:
        runner(gw, r, x, dr, dx, numel, N)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        x_c = x.contiguous()
        w_c = weight.contiguous()
        # Native BF16 normalization, now inside the Function (no tracked branch).
        v = x_c.pow(2).mean(-1, keepdim=True)
        v.add_(_EPS_BF16)
        r = torch.rsqrt(v)
        y = torch.empty_like(x_c)
        t = torch.empty_like(x_c)
        N = x_c.shape[-1]
        numel = x_c.numel()
        _fwd_launch(x_c, r, w_c, y, t, numel, N)
        ctx.save_for_backward(x_c, r, w_c, t)
        return y

    @staticmethod
    def backward(ctx, g):
        x, r, weight, t = ctx.saved_tensors
        g = g.contiguous()
        N = x.shape[-1]
        numel = x.numel()
        gw = torch.empty_like(x)
        dr_terms = torch.empty_like(x)
        dw_terms = torch.empty_like(x)
        _bwd1_launch(g, weight, x, t, gw, dr_terms, dw_terms, numel, N)
        # Native reductions in Torch, order unchanged.
        dr = dr_terms.sum_to_size(r.shape).contiguous()
        dw = dw_terms.sum_to_size(weight.shape)
        dx = torch.empty_like(x)
        _bwd2_launch(gw, r, x, dr, dx, numel, N)
        return dx, dw


def _supported(x, weight, eps):
    if not (x.is_cuda and weight.is_cuda):
        return False
    if x.device != weight.device:
        return False
    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        return False
    if x.numel() == 0 or weight.numel() == 0:
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
            "fused rms_norm supports only nonempty BF16 CUDA tensors on the same "
            "device with hidden=512, 1D weight=512, and eps==1e-5"
        )
    return _RMSNorm.apply(x, weight)

import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_fwd_kernel(x_ptr, y_ptr, H, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    row = offs // H
    col = offs % H
    gate_off = row * (2 * H) + col
    up_off = gate_off + H
    a = tl.load(x_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + up_off, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-a))
    y = a * sig * b
    tl.store(y_ptr + offs, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _swiglu_bwd_kernel(g_ptr, x_ptr, dx_ptr, H, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    row = offs // H
    col = offs % H
    gate_off = row * (2 * H) + col
    up_off = gate_off + H
    a = tl.load(x_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + up_off, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-a))
    dxgate = g * sig * (1.0 + a * (1.0 - sig)) * b
    dxup = g * (a * sig)
    tl.store(dx_ptr + gate_off, dxgate.to(tl.bfloat16), mask=mask)
    tl.store(dx_ptr + up_off, dxup.to(tl.bfloat16), mask=mask)


def _guard(x):
    if not x.is_cuda:
        raise NotImplementedError("swiglu: CUDA tensor required")
    if x.dtype != torch.bfloat16:
        raise NotImplementedError("swiglu: BF16 dtype required")
    if x.dim() not in (2, 3):
        raise NotImplementedError("swiglu: only 2D/3D inputs supported")
    if not x.is_contiguous():
        raise NotImplementedError("swiglu: contiguous input required")
    if x.shape[-1] % 2 != 0:
        raise NotImplementedError("swiglu: even last dim required")


_BLOCK = 1024


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        _guard(x)
        H = x.shape[-1] // 2
        y = torch.empty(x.shape[:-1] + (H,), dtype=x.dtype, device=x.device)
        n_elements = y.numel()
        if n_elements > 0:
            grid = ((n_elements + _BLOCK - 1) // _BLOCK,)
            _swiglu_fwd_kernel[grid](x, y, H, n_elements, BLOCK=_BLOCK)
        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        g = grad_out.contiguous()
        H = x.shape[-1] // 2
        dx = torch.empty_like(x)
        n_elements = g.numel()
        if n_elements > 0:
            grid = ((n_elements + _BLOCK - 1) // _BLOCK,)
            _swiglu_bwd_kernel[grid](g, x, dx, H, n_elements, BLOCK=_BLOCK)
        return dx


def swiglu(x):
    return SwiGLUFunction.apply(x)


def bias_swiglu_impl(x, bias=None):
    if bias is not None:
        raise NotImplementedError("swiglu: bias path not supported in this candidate")
    return swiglu(x)

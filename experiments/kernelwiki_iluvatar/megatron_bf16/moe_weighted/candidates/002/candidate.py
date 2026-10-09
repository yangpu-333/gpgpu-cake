import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4


@triton.jit
def _wswiglu_fwd_kernel(x_ptr, w_ptr, y_ptr, numel, H: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    row = offs // H
    col = offs % H
    gate = row * (2 * H) + col
    up = gate + H
    a = tl.load(x_ptr + gate, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + up, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + row, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(a)
    y = (a * sig) * b
    y = y * w
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _wswiglu_bwd_input_kernel(x_ptr, g_ptr, w_ptr, dx_ptr, numel, H: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    row = offs // H
    col = offs % H
    gate = row * (2 * H) + col
    up = gate + H
    a = tl.load(x_ptr + gate, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + up, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + row, mask=mask, other=0.0).to(tl.float32)
    gg = g * w
    sig = tl.sigmoid(a)
    tmp8 = gg * sig
    tmp10 = 1 - sig
    tmp11 = a * tmp10
    tmp12 = tmp11 + 1
    tmp13 = tmp8 * tmp12
    dx_gate = tmp13 * b
    dx_up = gg * (a * tl.sigmoid(a))
    tl.store(dx_ptr + gate, dx_gate, mask=mask)
    tl.store(dx_ptr + up, dx_up, mask=mask)


@torch.compile
def _weighted_swiglu_wgrad(y, g):
    y_1, y_2 = torch.chunk(y, 2, -1)
    swig = (y_1 * torch.sigmoid(y_1)) * y_2
    weights_grad = swig * g
    return torch.sum(weights_grad, dim=-1, keepdim=True)


def _grid(numel):
    return ((numel + _BLOCK - 1) // _BLOCK, 1, 1)


class _WeightedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights):
        H = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (H,), dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x, weights)
        ctx.H = H
        numel = out.numel()
        if numel != 0:
            _wswiglu_fwd_kernel[_grid(numel)](
                x, weights, out, numel, H=H, BLOCK=_BLOCK, num_warps=_NUM_WARPS)
        return out

    @staticmethod
    def backward(ctx, g):
        x, weights = ctx.saved_tensors
        H = ctx.H
        g = g.contiguous()
        dx = torch.empty_like(x)
        numel = g.numel()
        if numel == 0:
            return torch.empty_like(x), torch.empty_like(weights)
        _wswiglu_bwd_input_kernel[_grid(numel)](
            x, g, weights, dx, numel, H=H, BLOCK=_BLOCK, num_warps=_NUM_WARPS)
        dw = _weighted_swiglu_wgrad(x.view(-1, 2 * H), g.view(-1, H)).to(weights.dtype)
        return dx, dw


def weighted_swiglu(x, weights, *, native):
    if (isinstance(x, torch.Tensor) and isinstance(weights, torch.Tensor)
            and x.is_cuda and weights.is_cuda
            and x.dtype == torch.bfloat16 and weights.dtype == torch.bfloat16
            and x.dim() in (2, 3) and weights.dim() == 2 and weights.shape[-1] == 1
            and x.is_contiguous() and weights.is_contiguous()
            and x.shape[-1] % 2 == 0
            and weights.shape[0] == x.numel() // x.shape[-1]):
        return _WeightedSwiGLU.apply(x, weights)
    return native(x, weights)

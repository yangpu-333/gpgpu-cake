import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_launch_cache = {}
_state = {"closure_ok": True}


@triton.jit
def _swiglu_fwd_kernel(x_ptr, y_ptr, numel, H: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    row = offs // H
    col = offs % H
    gate = row * (2 * H) + col
    up = gate + H
    a = tl.load(x_ptr + gate, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + up, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(a)
    y = (a * sig) * b
    tl.store(y_ptr + offs, y, mask=mask)


@triton.jit
def _swiglu_bwd_kernel(x_ptr, g_ptr, dx_ptr, numel, H: tl.constexpr, BLOCK: tl.constexpr):
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
    sig = tl.sigmoid(a)
    tmp8 = g * sig
    tmp10 = 1 - sig
    tmp11 = a * tmp10
    tmp12 = tmp11 + 1
    tmp13 = tmp8 * tmp12
    dx_gate = tmp13 * b
    dx_up = g * (a * tl.sigmoid(a))
    tl.store(dx_ptr + gate, dx_gate, mask=mask)
    tl.store(dx_ptr + up, dx_up, mask=mask)


def _grid_x(numel):
    return (numel + _BLOCK - 1) // _BLOCK


def _launch_fwd(x, out, numel, H):
    gx = _grid_x(numel)
    key = ("fwd", x.get_device(), numel, H, gx,
           x.data_ptr() % 16, out.data_ptr() % 16)
    if _state["closure_ok"]:
        closure = _launch_cache.get(key)
        if closure is not None:
            try:
                closure(x, out, numel)
                return
            except Exception:
                _state["closure_ok"] = False
                _launch_cache.clear()
    compiled = _swiglu_fwd_kernel[(gx, 1, 1)](
        x, out, numel, H=H, BLOCK=_BLOCK, num_warps=_NUM_WARPS)
    if _state["closure_ok"]:
        try:
            _launch_cache[key] = compiled[(gx, 1, 1)]
        except Exception:
            _state["closure_ok"] = False
            _launch_cache.clear()


def _launch_bwd(x, g, dx, numel, H):
    gx = _grid_x(numel)
    key = ("bwd", x.get_device(), numel, H, gx,
           x.data_ptr() % 16, g.data_ptr() % 16, dx.data_ptr() % 16)
    if _state["closure_ok"]:
        closure = _launch_cache.get(key)
        if closure is not None:
            try:
                closure(x, g, dx, numel)
                return
            except Exception:
                _state["closure_ok"] = False
                _launch_cache.clear()
    compiled = _swiglu_bwd_kernel[(gx, 1, 1)](
        x, g, dx, numel, H=H, BLOCK=_BLOCK, num_warps=_NUM_WARPS)
    if _state["closure_ok"]:
        try:
            _launch_cache[key] = compiled[(gx, 1, 1)]
        except Exception:
            _state["closure_ok"] = False
            _launch_cache.clear()


class _SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        H = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (H,), dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x)
        ctx.H = H
        numel = out.numel()
        if numel != 0:
            _launch_fwd(x, out, numel, H)
        return out

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        H = ctx.H
        g = g.contiguous()
        dx = torch.empty_like(x)
        numel = g.numel()
        if numel != 0:
            _launch_bwd(x, g, dx, numel, H)
        return dx


def swiglu(x):
    if (not isinstance(x, torch.Tensor) or not x.is_cuda
            or x.dtype != torch.bfloat16 or x.dim() not in (2, 3)
            or not x.is_contiguous() or x.shape[-1] % 2 != 0):
        raise NotImplementedError(
            "swiglu expects a contiguous CUDA bfloat16 2D/3D tensor with an "
            "even last dimension")
    return _SwiGLU.apply(x)

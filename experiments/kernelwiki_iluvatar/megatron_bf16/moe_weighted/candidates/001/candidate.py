import torch
import triton
import triton.language as tl

_BLOCK = 1024
_NUM_WARPS = 4
_launch_cache = {}
_state = {"closure_ok": True}


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


@triton.jit
def _wswiglu_wgrad_kernel(x_ptr, g_ptr, dw_ptr, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    gate = row * (2 * H) + offs
    up = gate + H
    a = tl.load(x_ptr + gate, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + up, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(a)
    swig = (a * sig) * b
    summand = swig * g
    s = tl.sum(summand, axis=0)
    tl.store(dw_ptr + row, s)


def _grid_x(numel):
    return (numel + _BLOCK - 1) // _BLOCK


def _launch_fwd(x, w, out, numel, H):
    gx = _grid_x(numel)
    key = ("fwd", x.get_device(), numel, H, gx,
           x.data_ptr() % 16, w.data_ptr() % 16, out.data_ptr() % 16)
    if _state["closure_ok"]:
        closure = _launch_cache.get(key)
        if closure is not None:
            try:
                closure(x, w, out, numel)
                return
            except Exception:
                _state["closure_ok"] = False
                _launch_cache.clear()
    compiled = _wswiglu_fwd_kernel[(gx, 1, 1)](
        x, w, out, numel, H=H, BLOCK=_BLOCK, num_warps=_NUM_WARPS)
    if _state["closure_ok"]:
        try:
            _launch_cache[key] = compiled[(gx, 1, 1)]
        except Exception:
            _state["closure_ok"] = False
            _launch_cache.clear()


def _launch_bwd_input(x, g, w, dx, numel, H):
    gx = _grid_x(numel)
    key = ("bwd_input", x.get_device(), numel, H, gx,
           x.data_ptr() % 16, g.data_ptr() % 16, w.data_ptr() % 16, dx.data_ptr() % 16)
    if _state["closure_ok"]:
        closure = _launch_cache.get(key)
        if closure is not None:
            try:
                closure(x, g, w, dx, numel)
                return
            except Exception:
                _state["closure_ok"] = False
                _launch_cache.clear()
    compiled = _wswiglu_bwd_input_kernel[(gx, 1, 1)](
        x, g, w, dx, numel, H=H, BLOCK=_BLOCK, num_warps=_NUM_WARPS)
    if _state["closure_ok"]:
        try:
            _launch_cache[key] = compiled[(gx, 1, 1)]
        except Exception:
            _state["closure_ok"] = False
            _launch_cache.clear()


def _launch_wgrad(x, g, dw, N, H):
    block = triton.next_power_of_2(H)
    nw = 8 if block >= 2048 else 4
    key = ("wgrad", x.get_device(), N, H, block, nw,
           x.data_ptr() % 16, g.data_ptr() % 16, dw.data_ptr() % 16)
    if _state["closure_ok"]:
        closure = _launch_cache.get(key)
        if closure is not None:
            try:
                closure(x, g, dw)
                return
            except Exception:
                _state["closure_ok"] = False
                _launch_cache.clear()
    compiled = _wswiglu_wgrad_kernel[(N, 1, 1)](
        x, g, dw, H=H, BLOCK=block, num_warps=nw)
    if _state["closure_ok"]:
        try:
            _launch_cache[key] = compiled[(N, 1, 1)]
        except Exception:
            _state["closure_ok"] = False
            _launch_cache.clear()


class _WeightedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights):
        H = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (H,), dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x, weights)
        ctx.H = H
        numel = out.numel()
        if numel != 0:
            _launch_fwd(x, weights, out, numel, H)
        return out

    @staticmethod
    def backward(ctx, g):
        x, weights = ctx.saved_tensors
        H = ctx.H
        g = g.contiguous()
        dx = torch.empty_like(x)
        dw = torch.empty_like(weights)
        numel = g.numel()
        if numel != 0:
            _launch_bwd_input(x, g, weights, dx, numel, H)
            _launch_wgrad(x, g, dw, numel // H, H)
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

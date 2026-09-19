"""Independent V100 candidates for the official BF16 GDN decode contract."""
import math
import torch
import triton
import triton.language as tl


@torch.no_grad()
def vectorized(q, k, v, state, A_log, a, dt_bias, b, scale):
    scale = scale or 1 / math.sqrt(128)
    qf = q[:, 0].float().repeat_interleave(2, 1)
    kf = k[:, 0].float().repeat_interleave(2, 1)
    vf = v[:, 0].float()
    g = torch.exp(-torch.exp(A_log.float()) * torch.nn.functional.softplus(a[:, 0].float() + dt_bias.float()))
    beta = torch.sigmoid(b[:, 0].float())
    s = state if state is not None else torch.zeros(q.shape[0], 8, 128, 128, device=q.device)
    old = s * g[..., None, None]
    old_v = (old * kf[..., None, :]).sum(-1)
    new_v = beta[..., None] * vf + (1-beta[..., None]) * old_v
    updated = old - kf[..., None, :] * old_v[..., None] + kf[..., None, :] * new_v[..., None]
    output = (updated * qf[..., None, :]).sum(-1) * scale
    return output[:, None].to(torch.bfloat16), updated


@triton.jit
def bf16_load(ptr, offset):
    bits = tl.load(ptr + offset).to(tl.uint32) << 16
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def fused(Q, K, V, S, AL, A, DB, B, O, NS, SCALE: tl.constexpr,
          HAS_STATE: tl.constexpr, ROWS: tl.constexpr):
    batch_head = tl.program_id(0)
    batch = batch_head // 8
    head = batch_head % 8
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, 128)
    q = bf16_load(Q, (batch * 4 + head // 2) * 128 + cols)
    k = bf16_load(K, (batch * 4 + head // 2) * 128 + cols)
    v = bf16_load(V, batch_head * 128 + rows)
    av = bf16_load(A, batch_head)
    bv = bf16_load(B, batch_head)
    x = av + tl.load(DB + head)
    # Stable softplus, equivalent to PyTorch's thresholded expression.
    softplus = tl.where(x > 20, x, tl.log(1. + tl.exp(tl.minimum(x, 20.))))
    g = tl.exp(-tl.exp(tl.load(AL + head)) * softplus)
    beta = 1. / (1. + tl.exp(-bv))
    offsets = batch_head * 128 * 128 + rows[:, None] * 128 + cols[None, :]
    old = tl.full((ROWS, 128), 0., tl.float32)
    if HAS_STATE:
        old = tl.load(S + offsets) * g
    old_v = tl.sum(old * k[None, :], 1)
    new_v = beta * v + (1. - beta) * old_v
    updated = old - k[None, :] * old_v[:, None] + k[None, :] * new_v[:, None]
    output = tl.sum(updated * q[None, :], 1) * SCALE
    # BF16 round-to-nearest-even via integer bits works on sm70.
    bits = output.to(tl.uint32, bitcast=True)
    rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).to(tl.uint16)
    tl.store(O + batch_head * 128 + rows, rounded)
    tl.store(NS + offsets, updated)


def triton_candidate(q, k, v, state, A_log, a, dt_bias, b, scale, rows=4, warps=4):
    if rows not in (1, 2, 4, 8, 16) or warps not in (4, 8):
        raise ValueError('unsupported schedule')
    output = torch.empty((q.shape[0], 1, 8, 128), dtype=torch.bfloat16, device=q.device)
    ns = torch.empty((q.shape[0], 8, 128, 128), dtype=torch.float32, device=q.device)
    fused[(q.shape[0]*8, 128//rows)](
        q.view(torch.uint16), k.view(torch.uint16), v.view(torch.uint16), state if state is not None else ns,
        A_log, a.view(torch.uint16), dt_bias, b.view(torch.uint16), output.view(torch.uint16), ns,
        SCALE=float(scale or 1/math.sqrt(128)), HAS_STATE=state is not None, ROWS=rows,
        num_warps=warps, enable_fp_fusion=False)
    return output, ns

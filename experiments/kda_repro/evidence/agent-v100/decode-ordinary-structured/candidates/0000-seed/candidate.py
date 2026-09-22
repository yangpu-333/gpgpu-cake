"""Correctness-first PyTorch seed for the official GDN decode contract."""
import math
import torch


@torch.no_grad()
def triton_candidate(q, k, v, state, A_log, a, dt_bias, b, scale):
    """Ordinary eager PyTorch baseline supplied to the optimization controller."""
    scale = scale or 1 / math.sqrt(128)
    qf = q[:, 0].float().repeat_interleave(2, 1)
    kf = k[:, 0].float().repeat_interleave(2, 1)
    vf = v[:, 0].float()
    g = torch.exp(
        -torch.exp(A_log.float())
        * torch.nn.functional.softplus(a[:, 0].float() + dt_bias.float())
    )
    beta = torch.sigmoid(b[:, 0].float())
    current = (
        state
        if state is not None
        else torch.zeros(q.shape[0], 8, 128, 128, device=q.device)
    )
    old = current * g[..., None, None]
    old_v = (old * kf[..., None, :]).sum(-1)
    new_v = beta[..., None] * vf + (1 - beta[..., None]) * old_v
    updated = (
        old
        - kf[..., None, :] * old_v[..., None]
        + kf[..., None, :] * new_v[..., None]
    )
    output = (updated * qf[..., None, :]).sum(-1) * scale
    return output[:, None].to(torch.bfloat16), updated

"""Independent FP32-reduction reference for the task contract."""

from __future__ import annotations

import torch


def forward(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized output and the observable residual sum."""
    if hidden.ndim != 2 or residual.shape != hidden.shape:
        raise ValueError("hidden and residual must be rank-2 tensors of the same shape")
    if weight.ndim != 1 or weight.numel() != hidden.shape[-1]:
        raise ValueError("weight must match hidden_size")

    residual_out = hidden + residual
    inv_rms = torch.rsqrt(residual_out.float().square().mean(dim=-1, keepdim=True) + eps)
    output = residual_out * inv_rms.to(dtype=residual_out.dtype) * weight
    return output, residual_out

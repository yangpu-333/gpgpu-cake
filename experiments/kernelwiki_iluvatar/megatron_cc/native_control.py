"""Verifier control: native operations with no claimed kernel optimization."""
import torch


def fused(x, residual, weight, epsilon):
    if x.dtype != residual.dtype:
        x = x.to(residual.dtype)
    summed = residual + torch.nn.functional.dropout(x, p=0.0, training=True)
    return torch.nn.functional.rms_norm(summed, (summed.shape[-1],), weight, epsilon), summed

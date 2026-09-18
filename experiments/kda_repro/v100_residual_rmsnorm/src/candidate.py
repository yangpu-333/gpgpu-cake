"""First KDA candidate: a one-row Triton fused residual RMSNorm forward path."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _residual_rmsnorm_forward_kernel(
    hidden_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    residual_out_ptr,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(axis=0)
    columns = tl.arange(0, block)
    mask = columns < hidden_size
    offsets = row * hidden_size + columns

    # Match the FP16 residual observable used by the task contract before
    # starting the FP32 reduction.
    merged = (tl.load(hidden_ptr + offsets, mask=mask, other=0.0) + tl.load(
        residual_ptr + offsets, mask=mask, other=0.0
    )).to(tl.float16)
    squared = merged.to(tl.float32) * merged.to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(squared, axis=0) / hidden_size + eps)

    normalized = (merged * inv_rms.to(tl.float16)).to(tl.float16)
    output = (normalized * tl.load(weight_ptr + columns, mask=mask, other=0.0)).to(tl.float16)
    tl.store(output_ptr + offsets, output, mask=mask)
    tl.store(residual_out_ptr + offsets, merged, mask=mask)


class _FusedResidualRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (hidden.is_cuda and residual.is_cuda and weight.is_cuda):
            raise ValueError("candidate requires CUDA tensors")
        if not (hidden.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()):
            raise ValueError("candidate requires contiguous tensors")
        if hidden.ndim != 2 or residual.shape != hidden.shape or weight.shape != (hidden.shape[1],):
            raise ValueError("invalid residual RMSNorm tensor shapes")
        if hidden.dtype is not torch.float16 or residual.dtype is not torch.float16 or weight.dtype is not torch.float16:
            raise ValueError("candidate currently supports FP16 inputs only")

        rows, hidden_size = hidden.shape
        block = triton.next_power_of_2(hidden_size)
        output = torch.empty_like(hidden)
        residual_out = torch.empty_like(hidden)
        _residual_rmsnorm_forward_kernel[(rows,)](
            hidden,
            residual,
            weight,
            output,
            residual_out,
            hidden_size=hidden_size,
            eps=eps,
            block=block,
            num_warps=4,
        )
        ctx.save_for_backward(residual_out, weight)
        ctx.eps = eps
        return output, residual_out

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor | None,
        grad_residual_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        residual_out, weight = ctx.saved_tensors
        if grad_output is None:
            grad_output = torch.zeros_like(residual_out)
        if grad_residual_out is None:
            grad_residual_out = torch.zeros_like(residual_out)

        x = residual_out.float()
        grad_y = grad_output.float()
        scale = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + ctx.eps)
        weighted_grad = grad_y * weight.float()
        projection = (weighted_grad * x).sum(dim=-1, keepdim=True)
        grad_x = weighted_grad * scale - x * projection * scale.pow(3) / x.shape[-1]
        grad_x = grad_x + grad_residual_out.float()
        grad_weight = (grad_y * x * scale).sum(dim=0)
        return (
            grad_x.to(dtype=residual_out.dtype),
            grad_x.to(dtype=residual_out.dtype),
            grad_weight.to(dtype=weight.dtype),
            None,
        )


def forward(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the candidate under the task template's standard interface."""
    return _FusedResidualRMSNorm.apply(hidden, residual, weight, eps)

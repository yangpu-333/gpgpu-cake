'''Candidate 0001: fused residual-add + RMSNorm for BI-V150.

Optimized path covers FP32 residual and FP32 weight, contiguous 2D/3D tensors
with hidden <= 8192, and x in {float32, float16, bfloat16}. Everything outside
that scope uses an exact native fallback that preserves Megatron native
semantics, including native low-precision nonfinite behavior.
'''

import torch
import triton
import triton.language as tl


@triton.jit
def _forward_kernel(X, RESIDUAL, WEIGHT, OUTPUT, RESIDUAL_OUT,
                    hidden, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < hidden
    offset = row * hidden + col
    x = tl.load(X + offset, mask=valid, other=0.0).to(tl.float32)
    residual = tl.load(RESIDUAL + offset, mask=valid, other=0.0).to(tl.float32)
    summed = residual + x
    sum_squares = tl.sum(summed * summed, axis=0)
    inv_rms = 1.0 / tl.sqrt(sum_squares / hidden + eps)
    weight = tl.load(WEIGHT + col, mask=valid, other=0.0).to(tl.float32)
    tl.store(RESIDUAL_OUT + offset, summed, mask=valid)
    tl.store(OUTPUT + offset, summed * inv_rms * weight, mask=valid)


@triton.jit
def _backward_rows_kernel(R, W, DY, H, DR, PARTIAL, hidden, eps,
                          BLOCK: tl.constexpr, HAS_DY: tl.constexpr,
                          HAS_H: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < hidden
    offset = row * hidden + col
    r = tl.load(R + offset, mask=valid, other=0.0).to(tl.float32)
    inv_rms = 1.0 / tl.sqrt(tl.sum(r * r, axis=0) / hidden + eps)
    dr = tl.zeros([BLOCK], dtype=tl.float32)
    partial = tl.zeros([BLOCK], dtype=tl.float32)
    if HAS_DY:
        w = tl.load(W + col, mask=valid, other=0.0).to(tl.float32)
        dy = tl.load(DY + offset, mask=valid, other=0.0).to(tl.float32)
        weighted = dy * w
        dot = tl.sum(weighted * r, axis=0)
        dr = weighted * inv_rms - r * ((inv_rms * inv_rms * inv_rms) / hidden) * dot
        partial = dy * r * inv_rms
    if HAS_H:
        h = tl.load(H + offset, mask=valid, other=0.0).to(tl.float32)
        dr = dr + h
    tl.store(DR + offset, dr, mask=valid)
    tl.store(PARTIAL + offset, partial, mask=valid)


@triton.jit
def _backward_weight_kernel(PARTIAL, DW, rows, hidden,
                            ROW_BLOCK: tl.constexpr, COL_BLOCK: tl.constexpr):
    row = tl.arange(0, ROW_BLOCK)
    col = tl.program_id(0) * COL_BLOCK + tl.arange(0, COL_BLOCK)
    mask = (row[:, None] < rows) & (col[None, :] < hidden)
    values = tl.load(PARTIAL + row[:, None] * hidden + col[None, :], mask=mask, other=0.0)
    summed = tl.sum(values, axis=0)
    tl.store(DW + col, summed, mask=col < hidden)


class _FusedResidualRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, epsilon):
        shape = tuple(x.shape)
        hidden = shape[-1]
        rows = x.numel() // hidden
        x2 = x.reshape(rows, hidden)
        residual2 = residual.reshape(rows, hidden)
        output = torch.empty((rows, hidden), dtype=torch.float32, device=x.device)
        residual_out = torch.empty((rows, hidden), dtype=torch.float32, device=x.device)
        block = triton.next_power_of_2(hidden)
        _forward_kernel[(rows,)](
            x2, residual2, weight, output, residual_out,
            hidden, epsilon, BLOCK=block, num_warps=4,
        )
        ctx.save_for_backward(residual_out, weight)
        ctx.epsilon = epsilon
        ctx.x_dtype = x.dtype
        ctx.residual_dtype = residual.dtype
        ctx.weight_dtype = weight.dtype
        ctx.shape = shape
        return output.reshape(shape), residual_out.reshape(shape)

    @staticmethod
    def backward(ctx, grad_output, grad_residual_out):
        residual_out, weight = ctx.saved_tensors
        rows, hidden = residual_out.shape
        device = residual_out.device
        has_dy = grad_output is not None
        has_h = grad_residual_out is not None
        dy = grad_output.reshape(rows, hidden).contiguous() if has_dy else residual_out
        h = grad_residual_out.reshape(rows, hidden).contiguous() if has_h else residual_out
        dr = torch.empty((rows, hidden), dtype=torch.float32, device=device)
        partial = torch.empty((rows, hidden), dtype=torch.float32, device=device)
        block = triton.next_power_of_2(hidden)
        _backward_rows_kernel[(rows,)](
            residual_out, weight, dy, h, dr, partial,
            hidden, ctx.epsilon, BLOCK=block,
            HAS_DY=has_dy, HAS_H=has_h, num_warps=4,
        )
        dw = torch.empty((hidden,), dtype=torch.float32, device=device)
        _backward_weight_kernel[(triton.cdiv(hidden, 32),)](
            partial, dw, rows, hidden,
            ROW_BLOCK=triton.next_power_of_2(rows), COL_BLOCK=32,
            num_warps=4,
        )
        grad_x = dr.to(ctx.x_dtype).reshape(ctx.shape)
        grad_residual = dr.to(ctx.residual_dtype).reshape(ctx.shape)
        grad_weight = dw.to(ctx.weight_dtype)
        return grad_x, grad_residual, grad_weight, None


def _supported(x, residual, weight):
    if not (x.is_cuda and residual.is_cuda and weight.is_cuda):
        return False
    if residual.dtype != torch.float32 or weight.dtype != torch.float32:
        return False
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if x.dim() not in (2, 3) or tuple(residual.shape) != tuple(x.shape):
        return False
    if weight.dim() != 1 or weight.shape[0] != x.shape[-1]:
        return False
    if x.shape[-1] > 8192:
        return False
    if not (x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()):
        return False
    return True


def _native_fallback(x, residual, weight, epsilon):
    if x.dtype != residual.dtype:
        x = x.to(residual.dtype)
    summed = residual + torch.nn.functional.dropout(x, p=0.0, training=True)
    normalized = torch.nn.functional.rms_norm(summed, (summed.shape[-1],), weight, epsilon)
    return normalized, summed


def fused(x, residual, weight, epsilon):
    if _supported(x, residual, weight):
        return _FusedResidualRMSNorm.apply(x, residual, weight, epsilon)
    return _native_fallback(x, residual, weight, epsilon)

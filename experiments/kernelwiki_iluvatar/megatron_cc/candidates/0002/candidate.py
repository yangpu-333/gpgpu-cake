'''Candidate 0002: fused residual-add + RMSNorm for BI-V150 with a
process-local compiled-kernel cache.

Child of candidate 0001. Identical mathematics, epsilon semantics, forward
outputs, two-kernel backward, native low-precision-residual fallback and
num_warps=4. The only behavioral change is a safe process-local cache of
compiled Triton KERNEL CODE plus launch metadata: the first call for a given
specialization goes through normal CoreX JIT (and launches exactly once);
later calls launch the already-compiled kernel directly on the CURRENT stream
via CompiledKernel[(gx,gy,gz)], skipping repeated argument/device/SME/shape
dispatch. No tensor data, results, gradient values or device buffers are ever
cached. Any first-JIT failure stays visible.

hidden and epsilon are promoted to constexpr (useful: removes runtime scalar
dispatch and bakes loop/division bounds) and are part of every cache key. The
cache key also includes device, dtype, block, constexpr configuration and
per-pointer 16-byte alignment, so a differing alignment simply misses the
cache and falls back to a fresh, correctly specialized JIT compile rather than
unsafe reuse.

Optimized path covers FP32 residual and FP32 weight, contiguous 2D/3D tensors
with rows <= 2048 and hidden <= 8192, and x in {float32, float16, bfloat16}.
Everything outside that scope uses the exact native fallback that preserves
Megatron native semantics, including native low-precision nonfinite behavior.
'''

import torch
import triton
import triton.language as tl


@triton.jit
def _forward_kernel(X, RESIDUAL, WEIGHT, OUTPUT, RESIDUAL_OUT,
                    hidden: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
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
def _backward_rows_kernel(R, W, DY, H, DR, PARTIAL,
                          hidden: tl.constexpr, eps: tl.constexpr,
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
def _backward_weight_kernel(PARTIAL, DW,
                            rows: tl.constexpr, hidden: tl.constexpr,
                            ROW_BLOCK: tl.constexpr, COL_BLOCK: tl.constexpr):
    row = tl.arange(0, ROW_BLOCK)
    col = tl.program_id(0) * COL_BLOCK + tl.arange(0, COL_BLOCK)
    mask = (row[:, None] < rows) & (col[None, :] < hidden)
    values = tl.load(PARTIAL + row[:, None] * hidden + col[None, :], mask=mask, other=0.0)
    summed = tl.sum(values, axis=0)
    tl.store(DW + col, summed, mask=col < hidden)


# Process-local cache of compiled kernel code + launch metadata only.
# Never holds tensors, results, gradients or device buffers.
_COMPILED_KERNELS = {}


def _alignment(*tensors):
    return tuple((t.data_ptr() % 16 == 0) for t in tensors)


def _launch_forward(x, residual, weight, output, residual_out, hidden, epsilon, rows):
    block = triton.next_power_of_2(hidden)
    key = ("forward", x.device.index, x.dtype, hidden, float(epsilon), block,
           _alignment(x, residual, weight, output, residual_out))
    compiled = _COMPILED_KERNELS.get(key)
    if compiled is None:
        compiled = _forward_kernel[(rows,)](
            x, residual, weight, output, residual_out,
            hidden=hidden, eps=epsilon, BLOCK=block, num_warps=4,
        )
        _COMPILED_KERNELS[key] = compiled
    else:
        compiled[(rows, 1, 1)](x, residual, weight, output, residual_out)


def _launch_backward_rows(residual_out, weight, dy, h, dr, partial,
                          hidden, epsilon, rows, has_dy, has_h):
    block = triton.next_power_of_2(hidden)
    key = ("backward_rows", residual_out.device.index, hidden, float(epsilon),
           block, has_dy, has_h,
           _alignment(residual_out, weight, dy, h, dr, partial))
    compiled = _COMPILED_KERNELS.get(key)
    if compiled is None:
        compiled = _backward_rows_kernel[(rows,)](
            residual_out, weight, dy, h, dr, partial,
            hidden=hidden, eps=epsilon, BLOCK=block,
            HAS_DY=has_dy, HAS_H=has_h, num_warps=4,
        )
        _COMPILED_KERNELS[key] = compiled
    else:
        compiled[(rows, 1, 1)](residual_out, weight, dy, h, dr, partial)


def _launch_backward_weight(partial, dw, rows, hidden):
    grid = triton.cdiv(hidden, 32)
    row_block = triton.next_power_of_2(rows)
    key = ("backward_weight", partial.device.index, rows, hidden, row_block, 32,
           _alignment(partial, dw))
    compiled = _COMPILED_KERNELS.get(key)
    if compiled is None:
        compiled = _backward_weight_kernel[(grid,)](
            partial, dw, rows=rows, hidden=hidden,
            ROW_BLOCK=row_block, COL_BLOCK=32, num_warps=4,
        )
        _COMPILED_KERNELS[key] = compiled
    else:
        compiled[(grid, 1, 1)](partial, dw)


class _FusedResidualRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight, epsilon):
        hidden = x.shape[-1]
        rows = x.numel() // hidden
        output = torch.empty_like(residual)
        residual_out = torch.empty_like(residual)
        _launch_forward(x, residual, weight, output, residual_out,
                        hidden, epsilon, rows)
        ctx.save_for_backward(residual_out, weight)
        ctx.epsilon = epsilon
        ctx.x_dtype = x.dtype
        ctx.residual_dtype = residual.dtype
        ctx.weight_dtype = weight.dtype
        return output, residual_out

    @staticmethod
    def backward(ctx, grad_output, grad_residual_out):
        residual_out, weight = ctx.saved_tensors
        hidden = residual_out.shape[-1]
        rows = residual_out.numel() // hidden
        has_dy = grad_output is not None
        has_h = grad_residual_out is not None
        dy = grad_output.contiguous() if has_dy else residual_out
        h = grad_residual_out.contiguous() if has_h else residual_out
        dr = torch.empty_like(residual_out)
        partial = torch.empty_like(residual_out)
        _launch_backward_rows(residual_out, weight, dy, h, dr, partial,
                              hidden, ctx.epsilon, rows, has_dy, has_h)
        dw = torch.empty((hidden,), dtype=torch.float32, device=residual_out.device)
        _launch_backward_weight(partial, dw, rows, hidden)
        grad_x = dr.to(ctx.x_dtype)
        grad_residual = dr.to(ctx.residual_dtype)
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
    if x.numel() // x.shape[-1] > 2048:
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

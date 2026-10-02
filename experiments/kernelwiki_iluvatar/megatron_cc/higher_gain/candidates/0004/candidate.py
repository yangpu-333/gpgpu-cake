'''Candidate 0003: fused residual-add + RMSNorm for BI-V150, a minimal child
of candidate 0002 that adds a single autocast guard.

Identical to 0002 in every respect -- same mathematics, epsilon semantics,
forward outputs, two-kernel backward, process-local compiled-kernel cache,
native low-precision-residual fallback, num_warps=4 and supported row/hidden
range -- EXCEPT for one behavioral change at the public entry point.

Why the change: under the contract's bfloat16 autocast context the real fused
input x is BF16 while residual/weight/y are FP32. The optimized FP32-residual
kernels do their normalized math in FP32 and the tiny perturbations, once the
results propagate back through BF16 rounding inside the surrounding autocast
region, exceeded the UNCHANGED model correctness tolerance on 0002 (a few
token-loss elements near max~0.00789 and one parameter-gradient element near
max~0.0003133). That is a correctness/compatibility failure, not something to
be fixed by relaxing tolerances or touching the baseline/harness.

Minimal repair: whenever a CUDA autocast context is active, the public fused
entry dispatches to _native_fallback, which executes exactly the native BDA
dtype cast / dropout(p=0) / residual add and native torch.nn.functional.
rms_norm. It is invoked directly inside the caller's still-active autocast
context (no autocast disable/enable wrapping), so both forward and backward
rounding match the native Megatron path bit-for-bit under autocast. This is a
correctness/compatibility guarantee via NATIVE FALLBACK -- it is NOT a claim of
fused BF16 training performance. Note also that the harness's
candidate_calls.fused counter only counts adapter API calls and does not by
itself prove Triton kernels executed.

Outside autocast, behavior is byte-for-byte the 0002 contract: the optimized
path still covers FP32 residual and FP32 weight, contiguous 2D/3D tensors with
rows <= 2048 and hidden <= 8192, and x in {float32, float16, bfloat16}.
Everything outside that scope continues to use the exact native fallback that
preserves Megatron native semantics, including native low-precision nonfinite
behavior.
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


def _cuda_autocast_enabled():
    # Detect an active CUDA autocast context using whichever is_autocast_enabled
    # signature this torch build exposes (2.4.x accepts an optional device_type
    # positional; older signatures take none and report the CUDA state).
    try:
        return bool(torch.is_autocast_enabled("cuda"))
    except TypeError:
        return bool(torch.is_autocast_enabled())


def _native_fallback(x, residual, weight, epsilon):
    if x.dtype != residual.dtype:
        x = x.to(residual.dtype)
    summed = residual + torch.nn.functional.dropout(x, p=0.0, training=True)
    normalized = torch.nn.functional.rms_norm(summed, (summed.shape[-1],), weight, epsilon)
    return normalized, summed


def fused(x, residual, weight, epsilon):
    # Under any active CUDA autocast context, defer to the exact native path so
    # forward/backward rounding matches Megatron native under autocast. The
    # fallback runs inside the caller's still-active autocast region (no
    # disable/enable wrapping). This is correctness/compatibility via native
    # fallback, not a fused BF16 performance claim.
    if _cuda_autocast_enabled():
        return _native_fallback(x, residual, weight, epsilon)
    if _supported(x, residual, weight):
        return _FusedResidualRMSNorm.apply(x, residual, weight, epsilon)
    return _native_fallback(x, residual, weight, epsilon)


# ---------------------------------------------------------------------------
# Candidate 0004 extension (child of 0003): TP=1, zero-label-smoothing fused
# vocabulary cross entropy. The residual/RMSNorm source above is byte-identical
# to parent 0003; all new work is appended so the incremental effect of the CE
# fusion is attributable. Reuses the same process-local compiled-code cache and
# the vendor CompiledKernel[(gx,gy,gz)] repeated-launch API on the current
# stream. No mark_dirty: native mutates the FP32 logits buffer in place without
# it, and the narrow final-logits consumption scope is identical here.
# ---------------------------------------------------------------------------


@triton.jit
def _ce_forward_kernel(LOGITS, TARGET, LOSS,
                       V: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < V
    offset = row * V + col
    x = tl.load(LOGITS + offset, mask=mask, other=-float('inf'))
    row_max = tl.max(x, axis=0)
    z = x - row_max
    exp_z = tl.exp(z)
    sum_exp = tl.sum(exp_z, axis=0)
    target = tl.load(TARGET + row).to(tl.int32)
    # (col == target) & mask excludes out-of-range targets that happen to land
    # on a padding lane in [V, BLOCK); those rows keep the native log(sumexp)
    # loss with a zero predicted term.
    selected = (col == target) & mask
    predicted = tl.sum(tl.where(selected, z, 0.0), axis=0)
    tl.store(LOSS + row, tl.log(sum_exp) - predicted)
    # Overwrite the FP32 buffer with softmax probabilities, mirroring the native
    # observable forward mutation.
    tl.store(LOGITS + offset, exp_z / sum_exp, mask=mask)


@triton.jit
def _ce_backward_kernel(PROB, TARGET, GRAD_OUTPUT, DLOGITS,
                        V: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < V
    offset = row * V + col
    prob = tl.load(PROB + offset, mask=mask, other=0.0)
    target = tl.load(TARGET + row).to(tl.int32)
    grad_output = tl.load(GRAD_OUTPUT + row)
    onehot = tl.where((col == target) & mask, 1.0, 0.0)
    tl.store(DLOGITS + offset, (prob - onehot) * grad_output, mask=mask)


def _launch_ce_forward(logits, target, loss, vocab, rows):
    block = triton.next_power_of_2(vocab)
    key = ("ce_forward", logits.device.index, vocab, block,
           _alignment(logits, target, loss))
    compiled = _COMPILED_KERNELS.get(key)
    if compiled is None:
        compiled = _ce_forward_kernel[(rows,)](
            logits, target, loss, V=vocab, BLOCK=block, num_warps=4,
        )
        _COMPILED_KERNELS[key] = compiled
    else:
        compiled[(rows, 1, 1)](logits, target, loss)


def _launch_ce_backward(prob, target, grad_output, dlogits, vocab, rows):
    block = triton.next_power_of_2(vocab)
    key = ("ce_backward", prob.device.index, vocab, block,
           _alignment(prob, target, grad_output, dlogits))
    compiled = _COMPILED_KERNELS.get(key)
    if compiled is None:
        compiled = _ce_backward_kernel[(rows,)](
            prob, target, grad_output, dlogits, V=vocab, BLOCK=block, num_warps=4,
        )
        _COMPILED_KERNELS[key] = compiled
    else:
        compiled[(rows, 1, 1)](prob, target, grad_output, dlogits)


class _FusedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):
        sequence, batch, vocab = logits.shape
        rows = sequence * batch
        # Mirror native vocab_parallel_logits.float(): for FP32 input this is the
        # same tensor, so the softmax overwrite is observable on the input; for
        # low precision it is a separate FP32 buffer and the original low
        # precision tensor stays exactly unchanged.
        probabilities = logits.float()
        loss = torch.empty((sequence, batch), dtype=torch.float32, device=logits.device)
        _launch_ce_forward(probabilities, target, loss, vocab, rows)
        ctx.save_for_backward(probabilities, target)
        ctx.vocab = vocab
        ctx.rows = rows
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        probabilities, target = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        # Reuse the saved probability storage as the gradient buffer exactly as
        # native overwrites its softmax buffer, avoiding an extra [S,B,V]
        # temporary. (P - onehot) * grad_output, with no subtraction for
        # out-of-range targets. Autograd casts this FP32 gradient back to the
        # logits input dtype, matching native.
        _launch_ce_backward(probabilities, target, grad_output, probabilities,
                            ctx.vocab, ctx.rows)
        return probabilities, None


def cross_entropy(logits, target):
    # TP=1, zero-label-smoothing fused vocabulary cross entropy returning
    # float32 loss[S,B]. The protected adapter owns the singleton-TP, zero
    # smoothing, contiguity, dtype and native-fallback guards, so this entry
    # assumes supported contiguous CUDA logits[S,B,V] and contiguous int64
    # target[S,B]. Out-of-range labels (including -100) are NOT ignore_index:
    # their loss is log(sumexp) and their gradient is plain softmax*grad_output.
    return _FusedCrossEntropy.apply(logits, target)

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
# Candidate 0007 (minimal child of 0006, non-autocast Triton CE branch only):
# remove the separate logits.float() conversion launch from the fused CE
# forward. 0005/0006 built probabilities = logits.float() -- a full extra device
# launch producing an FP32 copy -- and then had the single forward kernel read
# AND write that FP32 buffer in place. 0007 removes that launch: it loads the
# native-dtype logits directly inside the fused forward and widens them to FP32
# in-kernel, writing softmax into a SEPARATE FP32 probability buffer. For
# low-precision (BF16) input the buffer is a fresh torch.empty_like(float32), so
# the BF16 input is only read and stays bitwise unchanged (matching native
# .float() which leaves a low-precision input untouched). For FP32 input the
# probability buffer aliases the input itself, reproducing the native observable
# in-place softmax overwrite and the native gradient-alias behavior.
#
# Numerics are unchanged from the parent: BF16 -> FP32 widening is exact, and
# both torch.float() and tl.to(tl.float32) perform the same lossless widening,
# so the FP32 value entering the max/exp/sum/log math is bitwise identical to
# 0005/0006. Loss and softmax outputs are therefore identical; only the extra
# .float() launch is eliminated. No new approximate math is introduced.
#
# The forward kernel now takes a distinct read pointer (LOGITS, native dtype)
# and write pointer (PROB, FP32). The process-local compiled-code cache key for
# the forward additionally encodes logits.dtype and prob.dtype (alongside the
# existing device index, vocab, block and 16-byte alignment flags) so a
# BF16-read compilation and an FP32-read compilation never collide. The backward
# kernel, its reuse of the probability buffer as the gradient buffer, the int64
# wide/out-of-range label handling (-100 is NOT ignore-index), the autocast
# native-equivalent Torch CE branch and every public entry point are
# byte-identical to 0006. The residual/RMSNorm prefix above is byte-identical to
# parent 0003. Scope stays TP=1, zero smoothing, tested seq128 / vocab1024;
# seq256/512 and vocab4096/8192 are not claimed. Screen-only; not a promotion
# claim, and all formal processes must be rerun before any promotion.
# ---------------------------------------------------------------------------


@triton.jit
def _ce_forward_kernel(LOGITS, PROB, TARGET, LOSS,
                       V: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < V
    offset = row * V + col
    # Load the native-dtype logits directly and widen to FP32 inside the kernel,
    # removing the parent's separate logits.float() launch. BF16 -> FP32 is an
    # exact widening, so x is bitwise identical to loading the FP32 result of a
    # prior .float() copy; the CE math below is unchanged from 0005/0006. LOGITS
    # is only read here, so a BF16 input is never mutated.
    x = tl.load(LOGITS + offset, mask=mask, other=-float('inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    z = x - row_max
    exp_z = tl.exp(z)
    sum_exp = tl.sum(exp_z, axis=0)
    # Keep the native int64 target. Loading as int64 promotes the int32 lane
    # index col to int64 in the comparison below, so the range check is exact for
    # every signed 64-bit label and no narrowing/overflow can happen.
    target = tl.load(TARGET + row).to(tl.int64)
    # (col == target) & mask selects the ground-truth lane only when target lies
    # in [0, V). Any out-of-range target (target < 0 or target >= V, matching
    # native vocab_start/vocab_end masking at world_size=1) matches no lane, so
    # the row keeps the native log(sumexp) loss with a zero predicted term.
    selected = (col == target) & mask
    predicted = tl.sum(tl.where(selected, z, 0.0), axis=0)
    tl.store(LOSS + row, tl.log(sum_exp) - predicted)
    # Write FP32 softmax into the separate probability buffer. When PROB aliases
    # an FP32 input (FP32 case) this reproduces the native observable in-place
    # overwrite; for BF16 input PROB is a fresh FP32 buffer and the BF16 input
    # stays bitwise unchanged.
    tl.store(PROB + offset, exp_z / sum_exp, mask=mask)


@triton.jit
def _ce_backward_kernel(PROB, TARGET, GRAD_OUTPUT, DLOGITS,
                        V: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < V
    offset = row * V + col
    prob = tl.load(PROB + offset, mask=mask, other=0.0)
    # Same int64 preservation as the forward kernel: a wide/negative int64 label
    # matches no lane, so no onehot subtraction is applied, exactly as native
    # skips softmax_update on masked targets.
    target = tl.load(TARGET + row).to(tl.int64)
    grad_output = tl.load(GRAD_OUTPUT + row)
    onehot = tl.where((col == target) & mask, 1.0, 0.0)
    tl.store(DLOGITS + offset, (prob - onehot) * grad_output, mask=mask)


def _launch_ce_forward(logits, prob, target, loss, vocab, rows):
    block = triton.next_power_of_2(vocab)
    # Key encodes device, native-logits dtype, probability-buffer dtype, vocab
    # (shape), block (layout) and 16-byte alignment so the BF16-read and
    # FP32-read compilations never collide. No tensors/results/gradients stored.
    key = ("ce_forward", logits.device.index, logits.dtype, prob.dtype, vocab,
           block, _alignment(logits, prob, target, loss))
    compiled = _COMPILED_KERNELS.get(key)
    if compiled is None:
        compiled = _ce_forward_kernel[(rows,)](
            logits, prob, target, loss, V=vocab, BLOCK=block, num_warps=4,
        )
        _COMPILED_KERNELS[key] = compiled
    else:
        compiled[(rows, 1, 1)](logits, prob, target, loss)


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
        # Remove the parent's separate logits.float() conversion launch. For
        # low-precision (BF16) input allocate a fresh FP32 probability buffer and
        # let the fused forward widen the BF16 logits in-kernel; the BF16 input
        # is only read and stays bitwise unchanged, matching native .float()
        # which leaves a low-precision input untouched. For FP32 input alias the
        # input itself so the native observable in-place softmax overwrite (and
        # native gradient alias) are preserved.
        if logits.dtype == torch.float32:
            probabilities = logits
        else:
            probabilities = torch.empty_like(logits, dtype=torch.float32)
        loss = torch.empty((sequence, batch), dtype=torch.float32, device=logits.device)
        _launch_ce_forward(logits, probabilities, target, loss, vocab, rows)
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


class _NativeAutocastCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):
        # calculate_logits_max: native vocab_parallel_logits.float() then max.
        # Under BF16 autocast .float() allocates a separate FP32 buffer, so the
        # original low-precision input stays unchanged; for FP32 input it is the
        # same tensor and the native in-place mutation is observable.
        vocab_parallel_logits = logits.float()
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        # world_size == 1: all_reduce(logits_max, MAX) is the identity, omitted.

        # vocab range for rank 0 at world_size 1.
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        vocab_start_index = 0
        vocab_end_index = partition_vocab_size

        # calculate_predicted_logits (native ordering, in-place subtract_).
        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # world_size == 1: all_reduce(predicted_logits, SUM) is the identity.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        # world_size == 1: all_reduce(sum_exp_logits, SUM) is the identity.

        # calculate_cross_entropy_loss: loss then normalize softmax in place.
        loss = torch.log(sum_exp_logits) - predicted_logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        # Store softmax, target-mask and masked-target for backward, as native.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        # prepare_gradient_calculation_operands: softmax is the gradient.
        grad_input = softmax
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        softmax_update = 1.0 - target_mask.view(-1).float()
        # calculate_gradients (zero smoothing): onehot subtraction then scale.
        # Out-of-range labels have softmax_update 0, so no subtraction applies.
        grad_2d[arange_1d, masked_target_1d] -= softmax_update
        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        # grad_input is FP32; autograd casts it to the original logits dtype,
        # exactly as native.
        return grad_input, None


def cross_entropy(logits, target):
    # TP=1, zero-label-smoothing vocabulary cross entropy returning float32
    # loss[S,B]. Out-of-range labels (including -100) are NOT ignore_index: their
    # loss is log(sumexp) and their gradient is plain softmax*grad_output. The
    # protected adapter owns the singleton-TP, zero-smoothing, contiguity, dtype
    # and native-fallback guards, so this entry assumes supported contiguous CUDA
    # logits[S,B,V] and contiguous int64 target[S,B].
    #
    # Under an active CUDA autocast context dispatch to the Torch autograd CE that
    # reproduces the pinned native operations and ordering exactly (0006 repair of
    # 0005's autocast re-rounding output failures). Outside autocast the FP32
    # Triton CE runs; 0007 removes its separate logits.float() launch by widening
    # the native-dtype logits in-kernel into a separate FP32 probability buffer.
    if _cuda_autocast_enabled():
        return _NativeAutocastCrossEntropy.apply(logits, target)
    return _FusedCrossEntropy.apply(logits, target)

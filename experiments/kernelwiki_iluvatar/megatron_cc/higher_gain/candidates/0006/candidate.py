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
# Candidate 0005 (minimal child of 0004): repair wide int64 cross-entropy target
# semantics. Candidate 0004 loaded target as int64 but immediately narrowed it
# with .to(tl.int32) in both CE kernels; any label outside the signed 32-bit
# range (the audit probed 2**32, 2**32+16, 2**63-1 and -2**63 at V=17) wraps
# modulo 2**32 into a valid [0, V) class, so those rows wrongly received the
# predicted-logit subtraction and a onehot gradient -- the independent compiled
# runner saw loss max error 2.7277 and dlogits max error 1.0000 on 3/4 rows.
# The ONLY change from 0004 is preserving the loaded target as int64 in both CE
# kernels: the int32 lane index col is promoted to int64 for the (col == target)
# & mask comparison, so the in-range test is exact across the full signed 64-bit
# label space and no narrowing/overflow can happen. Every out-of-range label
# (target < 0 or target >= V, matching native (target < vocab_start) |
# (target >= vocab_end) masking at world_size=1) now matches no lane, so it keeps
# the native log(sumexp) loss with a zero predicted term in the forward and takes
# no onehot subtraction in the backward. The residual/RMSNorm prefix above is
# byte-identical to parent 0003, and the CE kernels, process-local compiled-code
# cache, autograd Functions and entry points are otherwise unchanged from 0004
# (same max-shift CE math and buffers, observable FP32 softmax overwrite with the
# original low-precision input left unchanged, reused probability gradient buffer,
# num_warps=4, V>=1 / padded-lane support, arbitrary signed/zero upstream dloss,
# and native fallback). Screen-only evidence, not a promotion claim.
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
    # Keep the native int64 target. Narrowing to int32 (as 0004 did) let wide
    # labels wrap modulo 2**32 into a valid [0, V) class and wrongly collect the
    # predicted term; loading as int64 promotes the int32 lane index col to
    # int64 in the comparison below, so the range check is exact for every signed
    # 64-bit label.
    target = tl.load(TARGET + row).to(tl.int64)
    # (col == target) & mask selects the ground-truth lane only when target lies
    # in [0, V). Any out-of-range target (target < 0 or target >= V, matching
    # native vocab_start/vocab_end masking at world_size=1, including labels that
    # land on a padding lane in [V, BLOCK)) matches no lane, so the row keeps the
    # native log(sumexp) loss with a zero predicted term.
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
    # Same int64 preservation as the forward kernel: a wide/negative int64 label
    # matches no lane, so no onehot subtraction is applied, exactly as native
    # skips softmax_update on masked targets.
    target = tl.load(TARGET + row).to(tl.int64)
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


# ---------------------------------------------------------------------------
# Candidate 0006 (minimal child of 0005): repair the BF16-autocast cross-entropy
# output failures. 0005 passed all residual/RMSNorm, all full CE matrices, the
# wide-int64 semantic + actual-GPU-runner audit, every FP32 model output / loss /
# gradient / SGD check and the 2-layer holdout (FP32 complete-step ratios
# 1.173235 / 1.178758 / 1.149903, geomean ~1.167). But under BF16 autocast the
# FP32 Triton CE math, once its FP32 result propagated back through BF16 rounding
# in the surrounding autocast region, perturbed five token-loss elements per step
# on steps 2/3 by max~0.00799, exceeding the UNCHANGED atol 3e-4 / rtol 1e-3. The
# model gradient checks still passed, but that does NOT excuse the output
# failures, so 0005 is rejected; tolerances / verifier / config / timer stay
# untouched.
#
# The ONLY change from 0005 is at the public cross_entropy entry: it now adds an
# active-CUDA-autocast guard (parent's _cuda_autocast_enabled()) that dispatches
# to a separate TP=1 native-equivalent Torch autograd CE below. That branch
# reproduces the PINNED native VocabParallelCrossEntropy operations and their
# exact ordering -- logits.float(), torch.max, in-place subtract_ of the
# unsqueezed max, target_mask = (target < 0) | (target >= V), masked_target =
# target.clone() - 0 with masked_target[target_mask] = 0, the 2D view + arange +
# advanced index + clone().contiguous() + view_as(target), predicted[target_mask]
# = 0.0, torch.exp(out=same buffer), sum(dim=-1), log(sum_exp) - predicted, then
# div_ to normalize -- and saves exp_logits (probabilities), target_mask and
# masked_target_1d. Backward reproduces the native view / arange / softmax_update
# = 1.0 - target_mask.float(), the in-place grad_2d[arange, masked_target] -=
# softmax_update onehot subtraction, then mul_ by grad_output.unsqueeze(-1). It
# does NOT use F.cross_entropy / log_softmax / simplified autograd. Loss and
# grad_input stay FP32 internally; autograd casts the gradient to the original
# logits dtype exactly as native does. World_size is 1, so the native
# all_reduce(MAX) on logits_max and the two all_reduce(SUM) on predicted_logits /
# sum_exp_logits are the identity and are omitted; vocab_start=0, vocab_end=V.
# The default world group is never touched and Megatron is never imported -- the
# external adapter owns the TP group, the TP=1 / zero-smoothing / contiguity /
# dtype guards and the native fallback. Wide/out-of-range int64 labels are caught
# by target_mask (no onehot term, native log(sumexp) loss), there is no
# ignore_index, and zero smoothing only. Under autocast with BF16 input,
# logits.float() allocates a separate FP32 buffer so the original low-precision
# input is unchanged; native FP32-input overwrite is preserved when the input is
# already FP32. This Torch branch is a compatibility path, NOT Triton CE fusion
# under autocast: the adapter's optimized call counter only proves the candidate
# API was selected, not which internal branch ran. The 0003 residual/RMSNorm
# prefix and the 0005 FP32 Triton CE kernels, math, launch cache and int64
# comparison are byte-identical above. Screen-only; not a promotion claim, and
# all 14 formal processes must be rerun before any promotion.
# ---------------------------------------------------------------------------


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
    # Under an active CUDA autocast context the verified FP32 Triton CE path
    # perturbed a few token-loss elements once its FP32 result re-rounded through
    # BF16 in the surrounding autocast region (0005 autocast output failures,
    # max~0.00799 > atol 3e-4 / rtol 1e-3), so dispatch to the Torch autograd CE
    # that reproduces the pinned native operations and ordering exactly, matching
    # Megatron native rounding under autocast. This is a compatibility path, not
    # Triton CE fusion under autocast. Outside autocast the verified FP32 Triton
    # CE runs unchanged.
    if _cuda_autocast_enabled():
        return _NativeAutocastCrossEntropy.apply(logits, target)
    return _FusedCrossEntropy.apply(logits, target)

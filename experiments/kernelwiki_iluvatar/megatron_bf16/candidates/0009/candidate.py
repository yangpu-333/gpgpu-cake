import math

import torch
import triton
import triton.language as tl


_COMPILED_KERNELS = {}


def _alignment(*tensors):
    return tuple((t.data_ptr() % 16) == 0 for t in tensors)


def _cuda_autocast_enabled():
    return torch.is_autocast_enabled()


@triton.jit
def _ce_backward_kernel(PROB, TARGET, GRAD_OUTPUT, DLOGITS, DLOGITS_BF16,
                        V: tl.constexpr, BLOCK: tl.constexpr,
                        CAST: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < V
    offset = row * V + col
    prob = tl.load(PROB + offset, mask=mask, other=0.0)
    # A negative (e.g. -100) or >= V int64 label matches no lane, so no onehot
    # subtraction applies, mirroring native skipping softmax_update on masked
    # targets. -100 is NOT an ignore index here.
    target = tl.load(TARGET + row).to(tl.int64)
    grad_output = tl.load(GRAD_OUTPUT + row)
    onehot = tl.where((col == target) & mask, 1.0, 0.0)
    # Subtract the onehot in FP32, then multiply the FP32 upstream, matching
    # native grad -= softmax_update; grad.mul_(grad_output). mul(sub(.), .) has
    # no add-of-product so no FMA contraction applies; enable_fp_fusion=False is
    # belt-and-suspenders.
    dgrad = (prob - onehot) * grad_output
    # Overwrite the saved FP32 probability buffer in place, exactly as native
    # mutates its softmax storage, for every dtype.
    tl.store(DLOGITS + offset, dgrad, mask=mask)
    if CAST:
        # BF16-only: round the SAME FP32 dgrad to BF16 with the default RNE,
        # matching the CoreX native cast autograd would otherwise launch, and
        # write it to the separate BF16 output so autograd does not recast.
        tl.store(DLOGITS_BF16 + offset, dgrad.to(tl.bfloat16), mask=mask)


def _launch_ce_backward(prob, target, grad_output, dlogits, dlogits_bf16,
                        vocab, rows, cast):
    block = triton.next_power_of_2(vocab)
    key = ('ce_backward', prob.device.index, vocab, block, bool(cast),
           dlogits_bf16.dtype,
           _alignment(prob, target, grad_output, dlogits, dlogits_bf16))
    compiled = _COMPILED_KERNELS.get(key)
    if compiled is None:
        try:
            compiled = _ce_backward_kernel[(rows,)](
                prob, target, grad_output, dlogits, dlogits_bf16,
                V=vocab, BLOCK=block, CAST=cast,
                num_warps=4, enable_fp_fusion=False,
            )
        except TypeError:
            compiled = _ce_backward_kernel[(rows,)](
                prob, target, grad_output, dlogits, dlogits_bf16,
                V=vocab, BLOCK=block, CAST=cast,
                num_warps=4,
            )
        _COMPILED_KERNELS[key] = compiled
    else:
        compiled[(rows, 1, 1)](prob, target, grad_output, dlogits,
                               dlogits_bf16)


class _NativeForwardFusedBackwardCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):
        # EXACT native _NativeAutocastCrossEntropy.forward math and ordering; the
        # three world_size==1 all-reduces (MAX/SUM/SUM) are identities, omitted.
        vocab_parallel_logits = logits.float()
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        vocab_start_index = 0
        vocab_end_index = partition_vocab_size
        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                                 device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        loss = torch.log(sum_exp_logits) - predicted_logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        # Save the FP32 softmax buffer and the ORIGINAL int64 target; the fused
        # backward rebuilds the onehot from the raw label so out-of-range labels
        # reproduce native's skipped subtraction without a mask tensor.
        ctx.save_for_backward(exp_logits, target)
        ctx.vocab = partition_vocab_size
        ctx.rows = logits_2d.size()[0]
        # Record the original input dtype so backward restricts the fused
        # FP32->BF16 cast to BF16 inputs and leaves all other dtypes on the
        # parent 0008 path.
        ctx.input_dtype = logits.dtype
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        probabilities, target = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if ctx.input_dtype == torch.bfloat16:
            # BF16 training: fuse the FP32->BF16 dlogits cast into the backward
            # kernel. Still overwrite the FP32 probability buffer in place
            # (native mutation) and additionally emit a rounded BF16 gradient,
            # returned directly so autograd performs no extra cast launch.
            dlogits_bf16 = torch.empty_like(probabilities,
                                            dtype=torch.bfloat16)
            _launch_ce_backward(probabilities, target, grad_output,
                                probabilities, dlogits_bf16, ctx.vocab,
                                ctx.rows, cast=True)
            return dlogits_bf16, None
        # Parent 0008 path for FP32 (and any non-BF16) input: reuse the saved
        # FP32 softmax storage as the gradient buffer and let autograd cast the
        # returned FP32 dlogits to the logits dtype. The BF16 output pointer is
        # a dummy alias that CAST=False never stores to.
        _launch_ce_backward(probabilities, target, grad_output, probabilities,
                            probabilities, ctx.vocab, ctx.rows, cast=False)
        return probabilities, None


class _NativeAutocastCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target):
        vocab_parallel_logits = logits.float()
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        vocab_start_index = 0
        vocab_end_index = partition_vocab_size
        vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                                 device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        loss = torch.log(sum_exp_logits) - predicted_logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        grad_input = softmax
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0],
                                 device=grad_2d.device)
        softmax_update = 1.0 - target_mask.view(-1).float()
        grad_2d[arange_1d, masked_target_1d] -= softmax_update
        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        return grad_input, None


def cross_entropy(logits, target):
    # TP=1, zero-label-smoothing vocabulary cross entropy -> FP32 loss[S,B].
    # Out-of-range labels (including -100) are NOT ignore_index.
    #
    # Under active CUDA autocast dispatch to the fully native Torch CE (exact
    # loss/dlogits, zero Triton launches), preserving the 0006 autocast repair.
    # Outside autocast (true BF16 training) run the native-equivalent FP32
    # forward and fuse only the backward in Triton.
    if _cuda_autocast_enabled():
        return _NativeAutocastCrossEntropy.apply(logits, target)
    return _NativeForwardFusedBackwardCrossEntropy.apply(logits, target)

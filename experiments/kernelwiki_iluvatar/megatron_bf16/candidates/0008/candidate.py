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
def _ce_backward_kernel(PROB, TARGET, GRAD_OUTPUT, DLOGITS,
                        V: tl.constexpr, BLOCK: tl.constexpr):
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
    tl.store(DLOGITS + offset, (prob - onehot) * grad_output, mask=mask)


def _launch_ce_backward(prob, target, grad_output, dlogits, vocab, rows):
    block = triton.next_power_of_2(vocab)
    key = ('ce_backward', prob.device.index, vocab, block,
           _alignment(prob, target, grad_output, dlogits))
    compiled = _COMPILED_KERNELS.get(key)
    if compiled is None:
        try:
            compiled = _ce_backward_kernel[(rows,)](
                prob, target, grad_output, dlogits, V=vocab, BLOCK=block,
                num_warps=4, enable_fp_fusion=False,
            )
        except TypeError:
            compiled = _ce_backward_kernel[(rows,)](
                prob, target, grad_output, dlogits, V=vocab, BLOCK=block,
                num_warps=4,
            )
        _COMPILED_KERNELS[key] = compiled
    else:
        compiled[(rows, 1, 1)](prob, target, grad_output, dlogits)


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
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        probabilities, target = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        # Reuse the saved FP32 softmax storage as the gradient buffer, exactly as
        # native overwrites its softmax in place; autograd casts the FP32 dlogits
        # to the BF16 logits dtype.
        _launch_ce_backward(probabilities, target, grad_output, probabilities,
                            ctx.vocab, ctx.rows)
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

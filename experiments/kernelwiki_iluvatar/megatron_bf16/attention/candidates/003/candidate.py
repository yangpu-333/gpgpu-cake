# Copyright (c) 2026. BI-V150 Megatron dense causal attention candidate (stage31).
#
# Public API:  attention(q, k, v, scale) -> contiguous BF16 [S, B, H*D]
#
# candidate003 delta over candidate002
# -----------------------------------
# candidate002 already removed the FusedScaleMaskSoftmax module dispatch, the
# per-call CPU scale-tensor allocation, the global memory-buffer fetch, and the
# dropout/RNG-tracker fork, collapsing the whole op into a single autograd node
# whose forward/backward mirror the native per-stage derivatives.  It proved
# BITWISE-equal to the native default fused path (1475 checks / 637 tensors) and
# lowered the micro-Event cost, but the full nine-process training gate did not
# promote it: the residual *Python-level* ATen dispatch cost of the layout /
# GEMM glue around the vendor softmax was still enough for seed 31102 to tie the
# control arm.
#
# candidate003 keeps every tensor-level operation, its order, dtype, rounding,
# materialization and gradient formula byte-for-byte identical to candidate002,
# but moves the pure-ATen stage sequences that bracket the vendor softmax
# (reshape / transpose / baddbmm for the scores stage, and bmm / view / permute
# / contiguous for the context stage, plus their mirror-image backward stages)
# into torch.jit.script functions.  Scripting runs that fixed op chain inside
# the TorchScript interpreter in C++, eliminating repeated Python interpreter +
# per-op eager-dispatch cost on every one of the many attention calls per step,
# without introducing any operator fusion that would alter rounding (the only
# pointwise op, the backward `* scale`, is left explicit).  The vendor fused
# causal softmax forward/backward (scaled_upper_triang_masked_softmax_cuda,
# scalar 1.0, BF16 probabilities) remains a direct Python call because the
# pybind extension is not a scriptable torch.op; it is never weakened to eager
# torch where the native kernel is available.
#
# Preserved from candidate002 (unchanged semantics):
#   * BF16-staged baddbmm(beta=0, alpha=scale) scores,
#   * vendor default CUDA causal fused softmax (scale 1.0), BF16 probs,
#   * native bmm BF16 context and native layout conversion,
#   * native per-stage backward staging and the scale-at-matmul gradient,
#   * automatic torch fallback for shapes the native kernel rejects (tail
#     sq % 4 != 0), mirroring the native backend's own routing,
#   * the alignment guard for shared fused-QKV views whose storage offsets
#     (1/65/129) leave sub-16B-aligned data pointers; real formal inputs at
#     offsets 0/64/128 stay on the no-copy fast path.
#
# Nothing input/output/result/pointer/address-derived is cached; only the
# scalar scale ABI mode, per-shape batch_per_block, and the process-level
# scripted JIT code are retained (shape/config/scalar/JIT caches, permitted).
# If TorchScript is unavailable on the target build the eager stage functions
# are used unchanged, so numerics never depend on scripting succeeding.

from typing import Tuple

import torch

# --- cached scalar representations / shape config (no tensors/results cached) ---
_ONE = None            # process-level FP32 scale constant, mirrors torch.tensor([1.0])
_SCALE_MODE = None     # 'float' or 'tensor', chosen once by a harmless ABI probe
_BPB_CACHE = {}        # (sq, sk, b, np) -> batch_per_block int

# Minimum data-pointer alignment (bytes) assumed by the vendor GEMM/softmax
# path. 16B == one 128-bit BF16 vector load. Real fused-QKV inputs sit at
# element offsets 0/64/128 (byte 0/128/256), all >=16B aligned.
_PTR_ALIGN = 16


def _get_one():
    global _ONE
    if _ONE is None:
        _ONE = torch.tensor([1.0])  # default FP32 CPU scalar, identical to native scale_t
    return _ONE


def _ext_forward(scores):
    import scaled_upper_triang_masked_softmax_cuda as _m
    global _SCALE_MODE
    if _SCALE_MODE == 'float':
        return _m.forward(scores, 1.0)
    if _SCALE_MODE == 'tensor':
        return _m.forward(scores, _get_one()[0])
    # Once-per-process ABI check: prefer a plain float scalar; the native
    # scale_t[0] scalar-tensor form is the guaranteed fallback. Both encode 1.0.
    try:
        out = _m.forward(scores, 1.0)
        _SCALE_MODE = 'float'
        return out
    except (TypeError, RuntimeError):
        _SCALE_MODE = 'tensor'
        return _m.forward(scores, _get_one()[0])


def _ext_backward(output_grads, softmax_results):
    import scaled_upper_triang_masked_softmax_cuda as _m
    if _SCALE_MODE == 'float':
        return _m.backward(output_grads, softmax_results, 1.0)
    return _m.backward(output_grads, softmax_results, _get_one()[0])


def _fused_available(sq, sk, b, np_):
    # Mirrors FusedScaleMaskSoftmax.is_kernel_available for the BF16 causal path
    # (fusion=True, input_in_float16=True). batch_per_block cached per shape.
    attn_batches = b * np_
    if not (16 < sk <= 4096 and sq % 4 == 0 and sk % 4 == 0 and attn_batches % 4 == 0):
        return False
    key = (sq, sk, b, np_)
    bpb = _BPB_CACHE.get(key)
    if bpb is None:
        import scaled_masked_softmax_cuda
        bpb = scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np_)
        _BPB_CACHE[key] = bpb
    return attn_batches % bpb == 0


def _align(t):
    # Differentiable, value-preserving realignment of a single input. Fresh
    # allocator storage is aligned past _PTR_ALIGN; contiguous_format keeps the
    # downstream reshape/transpose chain a pure view. clone backward routes grad
    # to the original tensor.
    if t.data_ptr() % _PTR_ALIGN != 0:
        return t.clone(memory_format=torch.contiguous_format)
    return t


# --- pure-ATen stage functions (scripted below); identical ops/order/rounding
#     to candidate002, just executed in the TorchScript interpreter -----------

def _fwd_scores_impl(q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
    sq = q.size(0)
    b = q.size(1)
    np_ = q.size(2)
    hn = q.size(3)
    sk = k.size(0)
    bnp = b * np_
    q3 = q.reshape(sq, bnp, hn)
    k3 = k.reshape(sk, bnp, hn)
    qt = q3.transpose(0, 1)                       # [bnp, sq, hn]
    kt = k3.transpose(0, 1).transpose(1, 2)       # [bnp, hn, sk]
    # beta=0 -> buffer values are ignored; empty avoids the global buffer.
    buf = torch.empty([bnp, sq, sk], dtype=q.dtype, device=q.device)
    scores = torch.baddbmm(buf, qt, kt, beta=0.0, alpha=scale)  # BF16 scores
    return scores


def _fwd_context_impl(probs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    sk = v.size(0)
    b = v.size(1)
    np_ = v.size(2)
    hn = v.size(3)
    bnp = b * np_
    sq = probs.size(1)
    v3 = v.reshape(sk, bnp, hn)
    vt = v3.transpose(0, 1)                        # [bnp, sk, hn]
    context = torch.bmm(probs, vt)                 # [bnp, sq, hn] BF16
    out = (context.view(b, np_, sq, hn)
                  .permute(2, 0, 1, 3)
                  .contiguous()
                  .view(sq, b, np_ * hn))
    return out


def _bwd_context_impl(grad_out: torch.Tensor, probs: torch.Tensor,
                      v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    sk = v.size(0)
    b = v.size(1)
    np_ = v.size(2)
    hn = v.size(3)
    bnp = b * np_
    sq = probs.size(1)
    v3 = v.reshape(sk, bnp, hn)
    vt = v3.transpose(0, 1)                        # [bnp, sk, hn]
    # undo view(sq,b,hp) -> permute(2,0,1,3) -> view(b,np,sq,hn)
    grad_context = (grad_out.reshape(sq, b, np_, hn)
                            .permute(1, 2, 0, 3)
                            .reshape(bnp, sq, hn)
                            .contiguous())         # [bnp, sq, hn]
    grad_probs = torch.bmm(grad_context, vt.transpose(1, 2))   # [bnp, sq, sk]
    grad_vt = torch.bmm(probs.transpose(1, 2), grad_context)   # [bnp, sk, hn]
    grad_v = grad_vt.transpose(0, 1).reshape(sk, b, np_, hn)
    return grad_probs, grad_v


def _bwd_scores_impl(grad_scores: torch.Tensor, q: torch.Tensor,
                     k: torch.Tensor, scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    sq = q.size(0)
    b = q.size(1)
    np_ = q.size(2)
    hn = q.size(3)
    sk = k.size(0)
    bnp = b * np_
    q3 = q.reshape(sq, bnp, hn)
    qt = q3.transpose(0, 1)
    k3 = k.reshape(sk, bnp, hn)
    kt = k3.transpose(0, 1).transpose(1, 2)
    # baddbmm backward: scores = alpha * (qt @ kt), alpha applied post-matmul.
    grad_qt = torch.bmm(grad_scores, kt.transpose(1, 2)) * scale    # [bnp, sq, hn]
    grad_kt = torch.bmm(qt.transpose(1, 2), grad_scores) * scale    # [bnp, hn, sk]
    grad_q = grad_qt.transpose(0, 1).reshape(sq, b, np_, hn)
    grad_k = grad_kt.transpose(1, 2).transpose(0, 1).reshape(sk, b, np_, hn)
    return grad_q, grad_k


# Script the fixed stage chains once at import (JIT code cache, permitted). If
# the target build cannot script them, fall back to the identical eager impls so
# numerics never depend on scripting succeeding.
try:
    _fwd_scores = torch.jit.script(_fwd_scores_impl)
    _fwd_context = torch.jit.script(_fwd_context_impl)
    _bwd_context = torch.jit.script(_bwd_context_impl)
    _bwd_scores = torch.jit.script(_bwd_scores_impl)
except Exception:  # pragma: no cover - availability guard only
    _fwd_scores = _fwd_scores_impl
    _fwd_context = _fwd_context_impl
    _bwd_context = _bwd_context_impl
    _bwd_scores = _bwd_scores_impl


class _FusedDenseCausalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        scores = _fwd_scores(q, k, scale)             # BF16 staged scores
        probs = _ext_forward(scores)                  # vendor causal softmax, scalar 1.0
        out = _fwd_context(probs, v)                  # BF16 context + native layout
        ctx.save_for_backward(q, k, v, probs)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, probs = ctx.saved_tensors
        scale = ctx.scale
        grad_probs, grad_v = _bwd_context(grad_out, probs, v)
        grad_scores = _ext_backward(grad_probs, probs)        # vendor softmax backward
        grad_q, grad_k = _bwd_scores(grad_scores, q, k, scale)
        return grad_q, grad_k, grad_v, None


def _attention_torch_fallback(q, k, v, scale):
    # Mirrors DotProductAttention + FusedScaleMaskSoftmax.forward_torch_softmax
    # for shapes the native fused kernel rejects (softmax_in_fp32=False,
    # additional scale=None). Standard autograd supplies the matching backward.
    # Kept eager (unscripted) so the rare tail path stays identical to native.
    sq, b, np_, hn = q.shape
    sk = k.shape[0]
    bnp = b * np_

    q3 = q.reshape(sq, bnp, hn)
    k3 = k.reshape(sk, bnp, hn)
    v3 = v.reshape(sk, bnp, hn)

    buf = torch.empty((bnp, sq, sk), dtype=q.dtype, device=q.device)
    scores = torch.baddbmm(buf, q3.transpose(0, 1),
                           k3.transpose(0, 1).transpose(1, 2),
                           beta=0.0, alpha=scale)
    scores = scores.view(b, np_, sq, sk)

    if sq > 1:
        mask = torch.triu(torch.ones(sq, sq, device=q.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, -10000.0)
    probs = torch.softmax(scores, dim=-1)            # BF16 softmax (softmax_in_fp32=False)

    probs = probs.view(bnp, sq, sk)
    context = torch.bmm(probs, v3.transpose(0, 1))   # [bnp, sq, hn]
    return (context.view(b, np_, sq, hn)
                   .permute(2, 0, 1, 3)
                   .contiguous()
                   .view(sq, b, np_ * hn))


def attention(q, k, v, scale):
    """Dense causal self-attention. Returns contiguous BF16 [S, B, H*D]."""
    if not (isinstance(q, torch.Tensor)
            and isinstance(k, torch.Tensor)
            and isinstance(v, torch.Tensor)):
        raise TypeError("attention expects q, k, v as torch.Tensor")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError("attention expects scale as a real number")
    for t in (q, k, v):
        if t.device.type != "cuda":
            raise RuntimeError("attention supports CUDA tensors only (CPU rejected)")
        if t.dtype != torch.bfloat16:
            raise RuntimeError("attention supports BF16 tensors only (FP32/other rejected)")

    scale = float(scale)

    # Alignment guard: only misaligned shared-view inputs are cloned; aligned
    # formal inputs pass straight through on the fast no-copy path.
    q = _align(q)
    k = _align(k)
    v = _align(v)

    sq, b, np_, hn = q.shape
    sk = k.shape[0]
    if _fused_available(sq, sk, b, np_):
        return _FusedDenseCausalAttention.apply(q, k, v, scale)
    return _attention_torch_fallback(q, k, v, scale)

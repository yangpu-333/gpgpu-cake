# Copyright (c) 2026. BI-V150 Megatron dense causal attention candidate (stage31).
#
# Public API:  attention(q, k, v, scale) -> contiguous BF16 [S, B, H*D]
#
# Reproduces the original pinned DotProductAttention staged math
# (baddbmm(beta=0, alpha=scale) BF16 scores  ->  native
#  scaled_upper_triang_masked_softmax_cuda forward/backward with scalar 1.0 and
#  BF16 probabilities  ->  bmm BF16 context  ->  native layout conversion)
# while removing host/launch/wrapper overhead:
#   * no per-call CPU scale-tensor allocation (scale constant cached once),
#   * no FusedScaleMaskSoftmax module dispatch / per-call is_kernel_available
#     extension query (availability + batch_per_block cached per shape),
#   * no global memory buffer fetch, dropout module, or RNG-tracker fork
#     (dropout == 0 draws no randomness and must not touch global RNG),
#   * a single autograd.Function node whose backward mirrors the native
#     per-stage derivatives instead of the Python-level module graph.
#
# The native default fused masked softmax stays the baseline and is invoked
# directly; it is never weakened to torch eager where the native kernel is
# actually available.  For sequence lengths the native kernel rejects (e.g. the
# 17-tail: sq % 4 != 0), the native backend itself routes to its torch softmax
# path, so this candidate mirrors that exact fallback to match the reference.
#
# candidate002 delta over 001: a single alignment guard.  When q/k/v are views
# into a shared fused-QKV buffer whose storage offset leaves the data pointer
# sub-16B-aligned (observed: offsets 1/65/129 -> 2-byte-misaligned pointers),
# the vendor baddbmm/bmm/softmax path produced catastrophically wrong results
# (output max_abs ~0.88-0.96, not BF16 ULP).  Such a tensor is replaced by a
# fresh, contiguous, allocator-aligned clone carrying identical values; the
# clone is differentiable, so gradients still flow to the original q/k/v.  The
# real formal inputs at offsets 0/64/128 are >=16B aligned and are left exactly
# on the fast native staged view path (no copy), so their proven exact match is
# untouched.  The clone is per call and transient: no input/output/pointer/
# result is cached.

import torch

# --- cached scalar representations / shape config (no tensors/results cached) ---
_ONE = None            # process-level FP32 scale constant, mirrors torch.tensor([1.0])
_SCALE_MODE = None     # 'float' or 'tensor', chosen once by a harmless ABI probe
_BPB_CACHE = {}        # (sq, sk, b, np) -> batch_per_block int

# Minimum data-pointer alignment (bytes) assumed by the vendor GEMM/softmax
# path.  16B == one 128-bit BF16 vector load.  Real fused-QKV inputs sit at
# element offsets 0/64/128 (byte 0/128/256), all >=16B aligned, so they never
# trip this guard; only the misaligned 2-byte-offset shared views do.
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
    # Differentiable, value-preserving realignment of a single input.  Fresh
    # allocator storage is aligned well past _PTR_ALIGN; contiguous_format keeps
    # the downstream reshape/transpose chain a pure view, matching the aligned
    # fast path. clone's backward routes grad to the original tensor.
    if t.data_ptr() % _PTR_ALIGN != 0:
        return t.clone(memory_format=torch.contiguous_format)
    return t


class _FusedDenseCausalAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        sq, b, np_, hn = q.shape
        sk = k.shape[0]
        bnp = b * np_

        q3 = q.reshape(sq, bnp, hn)
        k3 = k.reshape(sk, bnp, hn)
        v3 = v.reshape(sk, bnp, hn)
        qt = q3.transpose(0, 1)                      # [bnp, sq, hn]
        kt = k3.transpose(0, 1).transpose(1, 2)      # [bnp, hn, sk]

        # beta=0 -> input buffer values are ignored; empty avoids the global buffer.
        buf = torch.empty((bnp, sq, sk), dtype=q.dtype, device=q.device)
        scores = torch.baddbmm(buf, qt, kt, beta=0.0, alpha=scale)   # BF16 scores

        probs = _ext_forward(scores)                 # BF16 causal softmax, scalar 1.0
        vt = v3.transpose(0, 1)                      # [bnp, sk, hn]
        context = torch.bmm(probs, vt)               # [bnp, sq, hn] BF16
        out = (context.view(b, np_, sq, hn)
                      .permute(2, 0, 1, 3)
                      .contiguous()
                      .view(sq, b, np_ * hn))

        ctx.save_for_backward(q, k, v, probs)
        ctx.scale = scale
        ctx.dims = (sq, b, np_, hn, sk, bnp)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, probs = ctx.saved_tensors
        scale = ctx.scale
        sq, b, np_, hn, sk, bnp = ctx.dims

        q3 = q.reshape(sq, bnp, hn); qt = q3.transpose(0, 1)
        k3 = k.reshape(sk, bnp, hn); kt = k3.transpose(0, 1).transpose(1, 2)
        v3 = v.reshape(sk, bnp, hn); vt = v3.transpose(0, 1)

        # undo view(sq,b,hp) -> permute(2,0,1,3) -> view(b,np,sq,hn)
        grad_context = (grad_out.reshape(sq, b, np_, hn)
                                .permute(1, 2, 0, 3)
                                .reshape(bnp, sq, hn)
                                .contiguous())                       # [bnp, sq, hn]

        # bmm backward: context = probs @ vt
        grad_probs = torch.bmm(grad_context, vt.transpose(1, 2))     # [bnp, sq, sk]
        grad_vt = torch.bmm(probs.transpose(1, 2), grad_context)     # [bnp, sk, hn]

        # native extension softmax backward (scalar 1.0, BF16)
        grad_scores = _ext_backward(grad_probs, probs)               # [bnp, sq, sk]

        # baddbmm backward: scores = alpha * (qt @ kt), alpha applied post-matmul
        grad_qt = torch.bmm(grad_scores, kt.transpose(1, 2)) * scale  # [bnp, sq, hn]
        grad_kt = torch.bmm(qt.transpose(1, 2), grad_scores) * scale  # [bnp, hn, sk]

        grad_q3 = grad_qt.transpose(0, 1)                             # [sq, bnp, hn]
        grad_k3 = grad_kt.transpose(1, 2).transpose(0, 1)             # [sk, bnp, hn]
        grad_v3 = grad_vt.transpose(0, 1)                             # [sk, bnp, hn]

        grad_q = grad_q3.reshape(sq, b, np_, hn)
        grad_k = grad_k3.reshape(sk, b, np_, hn)
        grad_v = grad_v3.reshape(sk, b, np_, hn)
        return grad_q, grad_k, grad_v, None


def _attention_torch_fallback(q, k, v, scale):
    # Mirrors DotProductAttention + FusedScaleMaskSoftmax.forward_torch_softmax
    # for shapes the native fused kernel rejects (softmax_in_fp32=False,
    # additional scale=None). Standard autograd supplies the matching backward.
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

    # Alignment guard (candidate002): only misaligned shared-view inputs are
    # cloned; aligned formal inputs pass straight through on the fast path.
    q = _align(q)
    k = _align(k)
    v = _align(v)

    sq, b, np_, hn = q.shape
    sk = k.shape[0]
    if _fused_available(sq, sk, b, np_):
        return _FusedDenseCausalAttention.apply(q, k, v, scale)
    return _attention_torch_fallback(q, k, v, scale)

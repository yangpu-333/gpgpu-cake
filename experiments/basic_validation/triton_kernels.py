"""Small, masked Triton candidates; import only on a configured GPU host.

These handwritten kernels exercise the runner, not Halide or Poly integration.
"""

import triton
import triton.language as tl


@triton.jit
def vector_add(X, Y, Z, LENGTH: tl.constexpr, BLOCK: tl.constexpr):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = indices < LENGTH
    left = tl.load(X + indices, mask=valid, other=0.0)
    right = tl.load(Y + indices, mask=valid, other=0.0)
    tl.store(Z + indices, left + right, mask=valid)


@triton.jit
def matrix_multiply(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # The runner supplies contiguous FP16 inputs and a preallocated FP16 output.
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for chunk in range(triton.cdiv(K, BK)):
        ks = chunk * BK + inner
        left = tl.load(A + rows[:, None] * K + ks[None, :],
                       mask=(rows[:, None] < M) & (ks[None, :] < K), other=0.0)
        right = tl.load(B + ks[:, None] * N + cols[None, :],
                        mask=(ks[:, None] < K) & (cols[None, :] < N), other=0.0)
        accumulator += tl.dot(left, right)
    tl.store(C + rows[:, None] * N + cols[None, :], accumulator.to(tl.float16),
             mask=(rows[:, None] < M) & (cols[None, :] < N))

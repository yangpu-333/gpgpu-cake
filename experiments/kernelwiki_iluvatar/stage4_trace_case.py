"""Short, focused GEMM bias pair for ixSYS CUDA/NVTX capture."""

import argparse

import torch
import triton

from stage4_gemm_epilogue import gemm_bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    torch.manual_seed(20260928)
    m = n = k = 256
    bm = bn = bk = 32
    a = torch.randn((m, k), device="cuda:0", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda:0", dtype=torch.float16)
    bias = torch.randn((n,), device="cuda:0", dtype=torch.float16)
    intermediate = torch.empty((m, n), device="cuda:0", dtype=torch.float16)
    baseline = torch.empty_like(intermediate)
    fused = torch.empty_like(intermediate)
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))

    def run_unfused():
        gemm_bias[grid](a, b, bias, intermediate, m, n, k,
                        bm, bn, bk, False, num_warps=4)
        torch.add(intermediate, bias, out=baseline)

    def run_fused():
        gemm_bias[grid](a, b, bias, fused, m, n, k,
                        bm, bn, bk, True, num_warps=4)

    for _ in range(3):
        run_unfused()
        run_fused()
    torch.cuda.synchronize()
    for name, fn in (("unfused_gemm_plus_bias", run_unfused),
                     ("fused_gemm_bias_epilogue", run_fused)):
        torch.cuda.nvtx.range_push(name)
        for _ in range(args.iterations):
            fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    print("TRACE_CASE_PASS", args.iterations, flush=True)


if __name__ == "__main__":
    main()

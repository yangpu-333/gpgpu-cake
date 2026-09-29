"""Narrow FP8 GEMM execution probe; does not imply NVFP4/block scaling."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def fp8_dot(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    cols = tl.program_id(1) * 32 + tl.arange(0, 32)
    inner = tl.arange(0, 32)
    acc = tl.zeros((32, 32), tl.float32)
    for chunk in range((K + 31) // 32):
        ks = chunk * 32 + inner
        a = tl.load(A + rows[:, None] * K + ks[None, :],
                    mask=(rows[:, None] < M) & (ks[None, :] < K), other=0)
        b = tl.load(B + ks[:, None] * N + cols[None, :],
                    mask=(ks[:, None] < K) & (cols[None, :] < N), other=0)
        acc += tl.dot(a, b)
    tl.store(C + rows[:, None] * N + cols[None, :], acc.to(tl.float16),
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(20260928)
    torch.cuda.set_device(0)
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "triton": triton.__version__,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cases": []}
    try:
        for name in ("float8_e4m3fn", "float8_e5m2"):
            for m, n, k in ((32, 32, 32), (64, 64, 64)):
                item = {"dtype": name, "mnk": [m, n, k]}
                try:
                    dtype = getattr(torch, name)
                    a = torch.randn((m, k), dtype=torch.float32).to(dtype).to("cuda")
                    b = torch.randn((k, n), dtype=torch.float32).to(dtype).to("cuda")
                    out = torch.empty((m, n), dtype=torch.float16, device="cuda")
                    compiled = fp8_dot[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
                        a, b, out, m, n, k, num_warps=4)
                    reference = (a.float().cpu().double() @ b.float().cpu().double()).half()
                    error = (out.cpu().float() - reference.float()).abs()
                    item["max_abs_error"] = float(error.max().item())
                    item["check_passed"] = bool(torch.allclose(out.cpu(), reference,
                                                                atol=0.05, rtol=0.05))
                    item["n_regs"] = getattr(compiled, "n_regs", None)
                    item["status"] = "passed" if item["check_passed"] else "numerical_failure"
                except Exception as exc:
                    item["status"] = "error"
                    item["error"] = repr(exc)
                report["cases"].append(item)
                print(name, (m, n, k), item["status"], flush=True)
    finally:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

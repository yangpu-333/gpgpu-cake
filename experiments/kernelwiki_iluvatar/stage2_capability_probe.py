"""BI-V150 CoreX 4.2 capability probe: BF16 GEMM and masked RMSNorm.

Run on the target Pod's existing CoreX environment. This checks correctness,
compilation, and a short event timing sample; it is not a tuning benchmark.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def bf16_gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for chunk in range((K + BK - 1) // BK):
        ks = chunk * BK + inner
        left = tl.load(A + rows[:, None] * K + ks[None, :],
                       mask=(rows[:, None] < M) & (ks[None, :] < K), other=0)
        right = tl.load(B + ks[:, None] * N + cols[None, :],
                        mask=(ks[:, None] < K) & (cols[None, :] < N), other=0)
        acc += tl.dot(left, right)
    tl.store(C + rows[:, None] * N + cols[None, :], acc.to(tl.bfloat16),
             mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def rmsnorm(X, W, Y, ROWS: tl.constexpr, COLS: tl.constexpr,
            EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * COLS + col, mask=col < COLS, other=0).to(tl.float32)
    w = tl.load(W + col, mask=col < COLS, other=0).to(tl.float32)
    variance = tl.sum(x * x, 0) / COLS
    result = x * (1.0 / tl.sqrt(variance + EPS)) * w
    tl.store(Y + row * COLS + col, result, mask=col < COLS)


def check(actual, reference, atol, rtol):
    diff = (actual.float() - reference.float()).abs()
    allowed = atol + rtol * reference.float().abs()
    return {
        "passed": bool(torch.isfinite(actual).all().item() and torch.all(diff <= allowed).item()),
        "max_abs_error": float(diff.max().item()),
        "max_allowed": float(allowed.max().item()),
        "fail_count": int((diff > allowed).sum().item()),
        "atol": atol, "rtol": rtol,
    }


def timed(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) * 1000 / 20)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20260926)
    torch.cuda.set_device(0)
    report = {
        "environment": {
            "device": torch.cuda.get_device_name(0), "device_index": 0,
            "torch": torch.__version__, "triton": triton.__version__,
            "python": platform.python_version(),
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [],
    }
    try:
        for m, n, k in ((64, 64, 64), (127, 65, 33), (256, 256, 256)):
            a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            fn = lambda: bf16_gemm[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
                a, b, out, m, n, k, 32, 32, 32, num_warps=4)
            item = {"operator": "bf16_gemm", "shape": [m, n, k], "num_warps": 4}
            try:
                fn()
                reference = (a.double().cpu() @ b.double().cpu()).to(torch.bfloat16)
                item["check"] = check(out.cpu(), reference, 0.04, 0.04)
                if item["check"]["passed"]:
                    item["event_us_per_launch"] = timed(fn)
            except Exception as exc:
                item["error"] = repr(exc)
            report["cases"].append(item)
        for dtype in (torch.float16, torch.bfloat16):
            for rows, cols in ((3, 768), (7, 1024), (16, 4099)):
                x = torch.randn((rows, cols), device="cuda", dtype=dtype)
                w = torch.randn((cols,), device="cuda", dtype=dtype)
                out = torch.empty_like(x)
                block = triton.next_power_of_2(cols)
                fn = lambda: rmsnorm[(rows,)](x, w, out, rows, cols, 1e-5,
                                               block, num_warps=4)
                item = {"operator": "rmsnorm", "shape": [rows, cols],
                        "dtype": str(dtype), "block": block, "num_warps": 4}
                try:
                    fn()
                    xc, wc = x.double().cpu(), w.double().cpu()
                    reference = ((xc * torch.rsqrt(xc.square().mean(-1, keepdim=True)
                                             + 1e-5)) * wc).to(dtype)
                    atol = 0.02 if dtype == torch.float16 else 0.04
                    item["check"] = check(out.cpu(), reference, atol, atol)
                    if item["check"]["passed"]:
                        item["event_us_per_launch"] = timed(fn)
                except Exception as exc:
                    item["error"] = repr(exc)
                report["cases"].append(item)
    finally:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    failed = [x for x in report["cases"] if "error" in x or not x["check"]["passed"]]
    print(f"{len(report['cases'])} cases, {len(failed)} failed, report={args.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""BI-V150 portable tile swizzle and dual-accumulator GEMM probes.

These are algorithm-level variations, not NVIDIA shared-memory swizzle,
TMEM ping-pong, warp specialization, or CLC equivalents.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def gemm_variant(A, B, BIAS, C, M: tl.constexpr, N: tl.constexpr,
                 K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                 BK: tl.constexpr, GM: tl.constexpr, GN: tl.constexpr,
                 SWIZZLE: tl.constexpr, DUAL_ACC: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    if SWIZZLE:
        pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, GM, GN, 2)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    if DUAL_ACC:
        acc0 = tl.zeros((BM, BN), tl.float32)
        acc1 = tl.zeros((BM, BN), tl.float32)
        for pair in range((K + 2 * BK - 1) // (2 * BK)):
            ks0 = pair * 2 * BK + inner
            ks1 = ks0 + BK
            a0 = tl.load(A + rows[:, None] * K + ks0[None, :],
                         mask=(rows[:, None] < M) & (ks0[None, :] < K), other=0)
            b0 = tl.load(B + ks0[:, None] * N + cols[None, :],
                         mask=(ks0[:, None] < K) & (cols[None, :] < N), other=0)
            a1 = tl.load(A + rows[:, None] * K + ks1[None, :],
                         mask=(rows[:, None] < M) & (ks1[None, :] < K), other=0)
            b1 = tl.load(B + ks1[:, None] * N + cols[None, :],
                         mask=(ks1[:, None] < K) & (cols[None, :] < N), other=0)
            acc0 += tl.dot(a0, b0)
            acc1 += tl.dot(a1, b1)
        acc = acc0 + acc1
    else:
        acc = tl.zeros((BM, BN), tl.float32)
        for chunk in range((K + BK - 1) // BK):
            ks = chunk * BK + inner
            a = tl.load(A + rows[:, None] * K + ks[None, :],
                        mask=(rows[:, None] < M) & (ks[None, :] < K), other=0)
            b = tl.load(B + ks[:, None] * N + cols[None, :],
                        mask=(ks[:, None] < K) & (cols[None, :] < N), other=0)
            acc += tl.dot(a, b)
    bias = tl.load(BIAS + cols, mask=cols < N, other=0)
    result = (acc.to(tl.float16).to(tl.float32) + bias.to(tl.float32)).to(tl.float16)
    tl.store(C + rows[:, None] * N + cols[None, :], result,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


def check(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    allowed = 0.03 + 0.03 * expected.float().abs()
    return {"passed": bool(torch.isfinite(actual).all().item() and
                           torch.all(diff <= allowed).item()),
            "max_abs_error": float(diff.max().item()),
            "failed_elements": int((diff > allowed).sum().item())}


def paired_samples(base, candidate):
    for _ in range(10):
        base()
        candidate()
    torch.cuda.synchronize()
    samples = {"base_us": [], "candidate_us": []}
    for sample in range(9):
        order = (("base", base), ("candidate", candidate)) if sample % 2 == 0 else (
            ("candidate", candidate), ("base", base))
        for name, fn in order:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(30):
                fn()
            stop.record()
            stop.synchronize()
            samples[f"{name}_us"].append(float(start.elapsed_time(stop)) * 1000 / 30)
    samples["base_median_us"] = statistics.median(samples["base_us"])
    samples["candidate_median_us"] = statistics.median(samples["candidate_us"])
    samples["speedup"] = samples["base_median_us"] / samples["candidate_median_us"]
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260928)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "triton": triton.__version__, "seed": args.seed,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "timing": {"warmup": 10, "samples": 9, "launches_per_sample": 30,
                         "baseline": "same Triton GEMM+bias tile, sequential accumulator"},
              "cases": []}
    try:
        for m, n, k in ((127, 65, 33), (256, 256, 256), (512, 512, 512)):
            a = torch.randn((m, k), dtype=torch.float16, device="cuda")
            b = torch.randn((k, n), dtype=torch.float16, device="cuda")
            bias = torch.randn((n,), dtype=torch.float16, device="cuda")
            reference = ((a.double().cpu() @ b.double().cpu()).half().float()
                         + bias.float().cpu()).half()
            gm, gn = triton.cdiv(m, 32), triton.cdiv(n, 32)
            grid = (gm, gn)
            baseline = torch.empty((m, n), dtype=torch.float16, device="cuda")

            def launch(out, swizzle, dual):
                return gemm_variant[grid](a, b, bias, out, m, n, k,
                                          32, 32, 32, gm, gn, swizzle, dual,
                                          num_warps=4)

            def base():
                return launch(baseline, False, False)

            base_kernel = base()
            item = {"mnk": [m, n, k], "tile": [32, 32, 32], "warps": 4,
                    "baseline_check": check(baseline.cpu(), reference),
                    "baseline_n_regs": getattr(base_kernel, "n_regs", None),
                    "variants": []}
            for name, swizzle, dual in (("swizzle2d_group2", True, False),
                                        ("dual_accumulator", False, True)):
                out = torch.empty_like(baseline)
                variant = {"name": name}

                def candidate():
                    return launch(out, swizzle, dual)

                try:
                    compiled = candidate()
                    variant["check"] = check(out.cpu(), reference)
                    variant["pair_check"] = check(out.cpu(), baseline.cpu())
                    variant["n_regs"] = getattr(compiled, "n_regs", None)
                    variant["shared"] = getattr(compiled, "shared", None)
                    if item["baseline_check"]["passed"] and variant["check"]["passed"] and variant["pair_check"]["passed"]:
                        variant["timing"] = paired_samples(base, candidate)
                        variant["status"] = "measured"
                    else:
                        variant["status"] = "numerical_failure"
                except Exception as exc:
                    variant["status"] = "error"
                    variant["error"] = repr(exc)
                item["variants"].append(variant)
            report["cases"].append(item)
            print((m, n, k), [(v["name"], v["status"]) for v in item["variants"]], flush=True)
    finally:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

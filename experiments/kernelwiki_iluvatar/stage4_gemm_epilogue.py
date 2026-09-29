"""BI-V150 GEMM bias epilogue and tile-search experiment.

Compare identical Triton GEMM tiles with bias applied by a separate PyTorch
launch or inside the GEMM store path. Keep all failed configurations and raw
GPU-event samples; this is a forward-only microbenchmark.
"""

import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def gemm_bias(A, B, BIAS, C, M: tl.constexpr, N: tl.constexpr,
              K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
              BK: tl.constexpr, FUSE: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for chunk in range((K + BK - 1) // BK):
        ks = chunk * BK + inner
        left = tl.load(A + rows[:, None] * K + ks[None, :],
                       mask=(rows[:, None] < M) & (ks[None, :] < K), other=0)
        right = tl.load(B + ks[:, None] * N + cols[None, :],
                        mask=(ks[:, None] < K) & (cols[None, :] < N), other=0)
        accumulator += tl.dot(left, right)
    result = accumulator.to(C.dtype.element_ty)
    if FUSE:
        bias = tl.load(BIAS + cols, mask=cols < N, other=0)
        result = (result.to(tl.float32) + bias.to(tl.float32)).to(C.dtype.element_ty)
    tl.store(C + rows[:, None] * N + cols[None, :], result,
             mask=(rows[:, None] < M) & (cols[None, :] < N))


CONFIGS = [
    {"bm": 32, "bn": 32, "bk": 32, "warps": 4},
    {"bm": 32, "bn": 64, "bk": 32, "warps": 4},
    {"bm": 64, "bn": 64, "bk": 32, "warps": 4},
    {"bm": 32, "bn": 64, "bk": 32, "warps": 8},
]


def correctness(output, reference, atol, rtol):
    actual = output.float()
    expected = reference.float()
    difference = (actual - expected).abs()
    allowed = atol + rtol * expected.abs()
    return {
        "passed": bool(torch.isfinite(actual).all().item() and
                       torch.all(difference <= allowed).item()),
        "max_abs_error": float(difference.max().item()),
        "failed_elements": int((difference > allowed).sum().item()),
        "atol": atol, "rtol": rtol,
    }


def timing_pair(unfused, fused, warmup=10, samples=9, launches=30):
    for _ in range(warmup):
        unfused()
        fused()
    torch.cuda.synchronize()
    out = {"unfused_us": [], "fused_us": []}
    for sample in range(samples):
        names = ("unfused", "fused") if sample % 2 == 0 else ("fused", "unfused")
        for name in names:
            fn = unfused if name == "unfused" else fused
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(launches):
                fn()
            stop.record()
            stop.synchronize()
            out[f"{name}_us"].append(float(start.elapsed_time(stop)) * 1000 / launches)
    out["unfused_median_us"] = statistics.median(out["unfused_us"])
    out["fused_median_us"] = statistics.median(out["fused_us"])
    out["speedup"] = out["unfused_median_us"] / out["fused_median_us"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), required=True)
    parser.add_argument("--shapes", default="127x65x33,256x256x256,512x512x512")
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    atol = 0.03 if dtype == torch.float16 else 0.05
    report = {
        "environment": {"gpu": torch.cuda.get_device_name(args.device),
                        "device_index": args.device, "torch": torch.__version__,
                        "triton": triton.__version__, "python": platform.python_version()},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dtype": args.dtype, "seed": args.seed,
        "timing": {"scope": "forward, warm-buffer GPU events",
                   "warmup": 10, "samples": 9, "launches_per_sample": 30,
                   "baseline": "same Triton GEMM tile plus separate torch.add(out=)"},
        "shapes": [],
    }
    try:
        for spec in args.shapes.split(","):
            m, n, k = map(int, spec.split("x"))
            a = torch.randn((m, k), device="cuda", dtype=dtype)
            b = torch.randn((k, n), device="cuda", dtype=dtype)
            bias = torch.randn((n,), device="cuda", dtype=dtype)
            cpu_dot = a.double().cpu() @ b.double().cpu()
            reference = (cpu_dot.to(dtype).float() + bias.float().cpu()).to(dtype)
            shape = {"mnk": [m, n, k], "configs": []}
            for config in CONFIGS:
                item = {"config": config.copy()}
                unfused_dot = torch.empty((m, n), device="cuda", dtype=dtype)
                unfused_output = torch.empty_like(unfused_dot)
                fused_output = torch.empty_like(unfused_dot)
                grid = (triton.cdiv(m, config["bm"]), triton.cdiv(n, config["bn"]))

                def launch(out, fuse):
                    gemm_bias[grid](a, b, bias, out, m, n, k,
                                    config["bm"], config["bn"], config["bk"],
                                    fuse, num_warps=config["warps"])

                def unfused():
                    launch(unfused_dot, False)
                    torch.add(unfused_dot, bias, out=unfused_output)

                def fused():
                    launch(fused_output, True)

                try:
                    unfused()
                    fused()
                    item["unfused_check"] = correctness(unfused_output.cpu(), reference, atol, atol)
                    item["fused_check"] = correctness(fused_output.cpu(), reference, atol, atol)
                    item["pair_check"] = correctness(fused_output.cpu(), unfused_output.cpu(), atol, atol)
                    if all(item[key]["passed"] for key in
                           ("unfused_check", "fused_check", "pair_check")):
                        item["timing"] = timing_pair(unfused, fused)
                        item["status"] = "measured"
                    else:
                        item["status"] = "numerical_failure"
                except Exception as exc:
                    item["status"] = "error"
                    item["error"] = repr(exc)
                shape["configs"].append(item)
            report["shapes"].append(shape)
            print(args.dtype, spec, [(x["config"], x["status"]) for x in shape["configs"]], flush=True)
    finally:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if all(x["status"] == "measured" for s in report["shapes"]
                    for x in s["configs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

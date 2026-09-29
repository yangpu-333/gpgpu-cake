"""BI-V150 software exp variant and chunk-parallel scan probes."""

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
def exp_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr,
               USE_POLY: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + idx, mask=idx < N, other=0)
    if USE_POLY:
        y = 1.0 + x * (1.0 + x * (0.5 + x * (1.0 / 6.0 +
            x * (1.0 / 24.0 + x * (1.0 / 120.0 + x / 720.0)))))
    else:
        y = tl.exp(x)
    tl.store(Y + idx, y, mask=idx < N)


@triton.jit
def scan_chunk_sums(X, SUMS, N: tl.constexpr, BLOCK: tl.constexpr):
    chunk = tl.program_id(0)
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, mask=offsets < N, other=0)
    tl.store(SUMS + chunk, tl.sum(values, 0))


@triton.jit
def scan_chunks(X, SUMS, Y, N: tl.constexpr, BLOCK: tl.constexpr,
                SUM_BLOCK: tl.constexpr):
    chunk = tl.program_id(0)
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, mask=offsets < N, other=0)
    chunk_ids = tl.arange(0, SUM_BLOCK)
    previous = tl.load(SUMS + chunk_ids, mask=chunk_ids < chunk, other=0)
    carry = tl.sum(previous, 0)
    result = tl.cumsum(values, 0) + carry
    tl.store(Y + offsets, result, mask=offsets < N)


def samples(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(9):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(30):
            fn()
        b.record()
        b.synchronize()
        vals.append(float(a.elapsed_time(b)) * 1000 / 30)
    return vals


def compare(actual, expected, atol, rtol):
    diff = (actual - expected).abs()
    allowed = atol + rtol * expected.abs()
    return {"passed": bool(torch.isfinite(actual).all().item() and
                           torch.all(diff <= allowed).item()),
            "max_abs_error": float(diff.max().item()),
            "failed_elements": int((diff > allowed).sum().item()),
            "atol": atol, "rtol": rtol}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260928)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    report = {"environment": {"gpu": torch.cuda.get_device_name(0),
                              "torch": torch.__version__, "triton": triton.__version__,
                              "python": platform.python_version()},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "seed": args.seed, "exp": [], "scan": [],
              "timing": {"warmup": 10, "samples": 9, "launches_per_sample": 30,
                         "scope": "forward warm-buffer GPU event"}}
    try:
        for n in (1 << 20, 1 << 22):
            x = torch.empty((n,), device="cuda", dtype=torch.float32).uniform_(-1, 0)
            expected = torch.exp(x)
            item = {"n": n, "domain": "uniform[-1,0]", "variants": []}
            for use_poly in (False, True):
                output = torch.empty_like(x)

                def launch():
                    return exp_kernel[(triton.cdiv(n, 1024),)](
                        x, output, n, 1024, use_poly, num_warps=4)

                variant = {"implementation": "degree6_taylor" if use_poly else "tl.exp"}
                try:
                    launch()
                    variant["check"] = compare(output, expected, 0.0003, 0.001)
                    if variant["check"]["passed"]:
                        variant["samples_us"] = samples(launch)
                        variant["median_us"] = statistics.median(variant["samples_us"])
                        variant["status"] = "measured"
                    else:
                        variant["status"] = "numerical_failure"
                except Exception as exc:
                    variant["status"] = "error"
                    variant["error"] = repr(exc)
                item["variants"].append(variant)
            report["exp"].append(item)
            print("exp", n, [v["status"] for v in item["variants"]], flush=True)
        for n in (4099, 65537):
            block = 1024
            chunks = triton.cdiv(n, block)
            x = torch.randn((n,), device="cuda", dtype=torch.float32)
            sums = torch.empty((chunks,), device="cuda", dtype=torch.float32)
            output = torch.empty_like(x)
            baseline = torch.empty_like(x)
            item = {"n": n, "block": block, "chunks": chunks}

            def run_candidate():
                scan_chunk_sums[(chunks,)](x, sums, n, block, num_warps=4)
                scan_chunks[(chunks,)](x, sums, output, n, block,
                                        triton.next_power_of_2(chunks), num_warps=4)

            def run_baseline():
                torch.cumsum(x, 0, out=baseline)

            try:
                run_baseline()
                run_candidate()
                item["check"] = compare(output, baseline, 0.01, 0.001)
                if item["check"]["passed"]:
                    item["candidate_samples_us"] = samples(run_candidate)
                    item["baseline_samples_us"] = samples(run_baseline)
                    item["candidate_median_us"] = statistics.median(item["candidate_samples_us"])
                    item["baseline_median_us"] = statistics.median(item["baseline_samples_us"])
                    item["speedup"] = item["baseline_median_us"] / item["candidate_median_us"]
                    item["status"] = "measured"
                else:
                    item["status"] = "numerical_failure"
            except Exception as exc:
                item["status"] = "error"
                item["error"] = repr(exc)
            report["scan"].append(item)
            print("scan", n, item["status"], flush=True)
    finally:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if all(v["status"] == "measured" for x in report["exp"] for v in x["variants"]) \
        and all(x["status"] == "measured" for x in report["scan"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

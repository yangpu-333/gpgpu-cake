"""Portable persistent-grid idea on BI-V150; no CLC or NVIDIA sync used."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def add_regular(A, B, C, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + idx, mask=idx < N, other=0)
    b = tl.load(B + idx, mask=idx < N, other=0)
    tl.store(C + idx, a + b, mask=idx < N)


@triton.jit
def add_persistent(A, B, C, N: tl.constexpr, BLOCK: tl.constexpr,
                   GRID: tl.constexpr):
    for tile in range(tl.program_id(0), (N + BLOCK - 1) // BLOCK, GRID):
        idx = tile * BLOCK + tl.arange(0, BLOCK)
        a = tl.load(A + idx, mask=idx < N, other=0)
        b = tl.load(B + idx, mask=idx < N, other=0)
        tl.store(C + idx, a + b, mask=idx < N)


def time_fn(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(9):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(30):
            fn()
        b.record()
        b.synchronize()
        values.append(float(a.elapsed_time(b)) * 1000 / 30)
    return values


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
              "timing": {"warmup": 10, "samples": 9, "launches_per_sample": 30},
              "cases": []}
    try:
        for n in ((1 << 20) + 17, (1 << 22) + 17):
            a = torch.randn((n,), device="cuda", dtype=torch.float16)
            b = torch.randn((n,), device="cuda", dtype=torch.float16)
            expected = (a + b).cpu()
            output = torch.empty_like(a)
            tiles = triton.cdiv(n, 1024)
            for grid in (tiles, 1024, 256):
                item = {"n": n, "block": 1024, "grid": grid,
                        "kind": "regular" if grid == tiles else "persistent"}

                def launch():
                    if grid == tiles:
                        return add_regular[(tiles,)](a, b, output, n, 1024,
                                                     num_warps=4)
                    return add_persistent[(grid,)](a, b, output, n, 1024, grid,
                                                   num_warps=4)

                try:
                    compiled = launch()
                    item["check_passed"] = bool(torch.equal(output.cpu(), expected))
                    item["n_regs"] = getattr(compiled, "n_regs", None)
                    if item["check_passed"]:
                        item["samples_us"] = time_fn(launch)
                        item["median_us"] = statistics.median(item["samples_us"])
                        item["status"] = "measured"
                    else:
                        item["status"] = "numerical_failure"
                except Exception as exc:
                    item["status"] = "error"
                    item["error"] = repr(exc)
                report["cases"].append(item)
            print(n, [(x["grid"], x["status"]) for x in report["cases"][-3:]], flush=True)
    finally:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

"""BI-V150 vector access layout and compiler resource observation."""

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
def affine_copy(X, Y, N: tl.constexpr, STRIDE: tl.constexpr,
                BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + offset * STRIDE, mask=offset < N, other=0)
    tl.store(Y + offset, value * 1.25 + 0.5, mask=offset < N)


def event_samples(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    result = []
    for _ in range(9):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(30):
            fn()
        end.record()
        end.synchronize()
        result.append(float(start.elapsed_time(end)) * 1000 / 30)
    return result


def metadata(kernel):
    info = {}
    for key in ("n_regs", "n_spills", "shared", "num_warps"):
        value = getattr(kernel, key, None)
        if isinstance(value, (int, float, str)):
            info[key] = value
    assembly = getattr(kernel, "asm", None)
    if isinstance(assembly, dict):
        info["assembly_kinds"] = sorted(assembly.keys())
        info["assembly_sha256"] = {
            key: hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()
            for key, value in assembly.items() if isinstance(value, (str, bytes))
        }
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    report = {
        "environment": {"gpu": torch.cuda.get_device_name(args.device),
                        "torch": torch.__version__, "triton": triton.__version__,
                        "python": platform.python_version()},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": args.seed,
        "timing": {"warmup": 10, "samples": 9, "launches_per_sample": 30,
                   "scope": "warm-buffer GPU event, one affine elementwise launch"},
        "cases": [],
    }
    try:
        for n in (1 << 20, 1 << 22):
            for stride in (1, 2):
                backing = torch.randn((n * stride,), device="cuda", dtype=torch.float16)
                output = torch.empty((n,), device="cuda", dtype=torch.float16)
                expected = backing[::stride].float() * 1.25 + 0.5
                for block, warps in ((256, 4), (1024, 4), (4096, 4), (1024, 8)):
                    item = {"n": n, "stride": stride, "block": block, "warps": warps}

                    def launch():
                        return affine_copy[(triton.cdiv(n, block),)](
                            backing, output, n, stride, block, num_warps=warps)

                    try:
                        compiled = launch()
                        actual = output.float()
                        difference = (actual - expected).abs()
                        item["check"] = {"passed": bool(torch.allclose(actual, expected, atol=0.002, rtol=0.002)),
                                         "max_abs_error": float(difference.max().item())}
                        item["compiler"] = metadata(compiled)
                        if item["check"]["passed"]:
                            item["samples_us"] = event_samples(launch)
                            item["median_us"] = statistics.median(item["samples_us"])
                            item["status"] = "measured"
                        else:
                            item["status"] = "numerical_failure"
                    except Exception as exc:
                        item["status"] = "error"
                        item["error"] = repr(exc)
                    report["cases"].append(item)
                print("access", n, stride, [x["status"] for x in report["cases"][-4:]], flush=True)
    finally:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0 if all(c["status"] == "measured" for c in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

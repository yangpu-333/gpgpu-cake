"""CoreX 4.2 API audit and minimal stages/cache compilation probes."""

import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl

from stage4_gemm_epilogue import gemm_bias


@triton.jit
def cache_copy(X, Y, N: tl.constexpr, BLOCK: tl.constexpr,
               CACHE: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + idx, mask=idx < N, other=0, cache_modifier=CACHE)
    tl.store(Y + idx, x, mask=idx < N)


def time_kernel(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    result = []
    for _ in range(9):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(30):
            fn()
        b.record()
        b.synchronize()
        result.append(float(a.elapsed_time(b)) * 1000 / 30)
    return result


def metadata(compiled):
    result = {}
    for key in ("n_regs", "n_spills", "shared", "num_warps"):
        value = getattr(compiled, key, None)
        if isinstance(value, (int, float, str)):
            result[key] = value
    asm = getattr(compiled, "asm", None)
    if isinstance(asm, dict):
        result["assembly_kinds"] = sorted(asm)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260928)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    names = ("exp2", "cumsum", "swizzle2d", "make_block_ptr", "debug_barrier",
             "async_copy", "atomic_add")
    report = {
        "environment": {"gpu": torch.cuda.get_device_name(0),
                        "torch": torch.__version__, "triton": triton.__version__,
                        "python": platform.python_version()},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": args.seed,
        "language_symbol_present": {name: hasattr(tl, name) for name in names},
        "float8_types": {}, "stages": [], "cache": [],
        "timing": {"warmup": 10, "samples": 9, "launches_per_sample": 30,
                   "scope": "warm-buffer GPU event"},
    }
    try:
        for name in ("float8_e4m3fn", "float8_e5m2"):
            dtype = getattr(torch, name, None)
            entry = {"attribute_present": dtype is not None}
            if dtype is not None:
                try:
                    tensor = torch.empty((16,), device="cuda", dtype=dtype)
                    entry["allocation_passed"] = tensor.numel() == 16
                except Exception as exc:
                    entry["allocation_error"] = repr(exc)
            report["float8_types"][name] = entry
        m = n = k = 256
        a = torch.randn((m, k), device="cuda", dtype=torch.float16)
        b = torch.randn((k, n), device="cuda", dtype=torch.float16)
        bias = torch.randn((n,), device="cuda", dtype=torch.float16)
        output = torch.empty((m, n), device="cuda", dtype=torch.float16)
        expected = ((a.double().cpu() @ b.double().cpu()).half().float()
                    + bias.float().cpu()).half()
        for stages in (1, 2, 3):
            item = {"num_stages": stages, "tile": [32, 32, 32], "warps": 4}

            def launch():
                return gemm_bias[(8, 8)](a, b, bias, output, m, n, k,
                                         32, 32, 32, True,
                                         num_warps=4, num_stages=stages)

            try:
                compiled = launch()
                item["check_passed"] = bool(torch.allclose(output.cpu(), expected,
                                                           atol=0.03, rtol=0.03))
                item["compiler"] = metadata(compiled)
                if item["check_passed"]:
                    item["samples_us"] = time_kernel(launch)
                    item["median_us"] = statistics.median(item["samples_us"])
                    item["status"] = "measured"
                else:
                    item["status"] = "numerical_failure"
            except Exception as exc:
                item["status"] = "error"
                item["error"] = repr(exc)
            report["stages"].append(item)
        x = torch.randn((1 << 20,), device="cuda", dtype=torch.float16)
        y = torch.empty_like(x)
        for cache in ("", ".ca", ".cg"):
            item = {"cache_modifier": cache}

            def launch():
                return cache_copy[(1024,)](x, y, 1 << 20, 1024, cache,
                                           num_warps=4)

            try:
                compiled = launch()
                item["check_passed"] = bool(torch.equal(x, y))
                item["compiler"] = metadata(compiled)
                if item["check_passed"]:
                    item["samples_us"] = time_kernel(launch)
                    item["median_us"] = statistics.median(item["samples_us"])
                    item["status"] = "measured"
                else:
                    item["status"] = "numerical_failure"
            except Exception as exc:
                item["status"] = "error"
                item["error"] = repr(exc)
            report["cache"].append(item)
        print("stages", [x["status"] for x in report["stages"]], flush=True)
        print("cache", [x["status"] for x in report["cache"]], flush=True)
    finally:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

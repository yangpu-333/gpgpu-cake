"""BI-V150 Residual Add RMSNorm backward: two Triton kernels vs analytical PyTorch.

This is a downstream KDA task experiment, not a claim of NVIDIA instruction
equivalence or full-training throughput. All timing is backward-only GPU events.
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.language as tl


EPSILON = 1e-6


@triton.jit
def backward_rows(R, W, DY, H, DX, DRES, PARTIAL,
                  HIDDEN: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
                  HAS_H: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < HIDDEN
    offset = row * HIDDEN + col
    r = tl.load(R + offset, mask=valid, other=0).to(tl.float32)
    w = tl.load(W + col, mask=valid, other=0).to(tl.float32)
    dy = tl.load(DY + offset, mask=valid, other=0).to(tl.float32)
    inv = 1.0 / tl.sqrt(tl.sum(r * r, axis=0) / HIDDEN + EPS)
    weighted = dy * w
    dot = tl.sum(weighted * r, axis=0)
    dr = weighted * inv - r * ((inv * inv * inv) / HIDDEN) * dot
    if HAS_H:
        h = tl.load(H + offset, mask=valid, other=0).to(tl.float32)
        dr = dr + h
    tl.store(DX + offset, dr, mask=valid)
    tl.store(DRES + offset, dr, mask=valid)
    tl.store(PARTIAL + offset, dy * r * inv, mask=valid)


@triton.jit
def backward_weight(PARTIAL, DW, ROWS: tl.constexpr, HIDDEN: tl.constexpr,
                    ROW_BLOCK: tl.constexpr, COL_BLOCK: tl.constexpr):
    rows = tl.arange(0, ROW_BLOCK)
    cols = tl.program_id(0) * COL_BLOCK + tl.arange(0, COL_BLOCK)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < HIDDEN)
    values = tl.load(PARTIAL + rows[:, None] * HIDDEN + cols[None, :],
                     mask=mask, other=0)
    summed = tl.sum(values, axis=0)
    tl.store(DW + cols, summed, mask=cols < HIDDEN)


def candidate(r, w, dy, h, dx, dres, dw, partial, num_warps):
    rows, hidden = r.shape
    backward_rows[(rows,)](
        r, w, dy, h if h is not None else r, dx, dres, partial,
        HIDDEN=hidden, EPS=EPSILON, BLOCK=triton.next_power_of_2(hidden),
        HAS_H=h is not None, num_warps=num_warps,
    )
    backward_weight[(triton.cdiv(hidden, 32),)](
        partial, dw, ROWS=rows, HIDDEN=hidden,
        ROW_BLOCK=triton.next_power_of_2(rows), COL_BLOCK=32,
        num_warps=4,
    )


def analytical_reference(r, w, dy, h):
    values = r.to(torch.float32)
    gradient = dy.to(torch.float32) * w.to(torch.float32)
    inv = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + EPSILON)
    dot = (gradient * values).sum(dim=-1, keepdim=True)
    dr = gradient * inv - values * (inv.pow(3) / r.shape[1]) * dot
    if h is not None:
        dr = dr + h.to(torch.float32)
    dw = (dy.to(torch.float32) * values * inv).sum(dim=0)
    return dr.to(r.dtype), dr.to(r.dtype), dw.to(w.dtype)


def autograd_reference(r, w, dy, h):
    x = r.detach().clone().requires_grad_(True)
    residual = torch.zeros_like(x).requires_grad_(True)
    weight = w.detach().clone().requires_grad_(True)
    residual_out = x + residual
    values = residual_out.to(torch.float32)
    inv = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + EPSILON)
    output = (values * inv * weight.to(torch.float32)).to(r.dtype)
    if h is None:
        return torch.autograd.grad(output, (x, residual, weight), grad_outputs=dy)
    return torch.autograd.grad((output, residual_out), (x, residual, weight),
                               grad_outputs=(dy, h))


def compare(actual, expected, atol=0.03, rtol=0.03):
    delta = (actual.to(torch.float32) - expected.to(torch.float32)).abs()
    tolerance = atol + rtol * expected.to(torch.float32).abs()
    return {
        "passed": bool(torch.all(delta <= tolerance).item()),
        "max_abs_error": float(delta.max().item()),
        "bad_elements": int((delta > tolerance).sum().item()),
    }


def timed_ms(fn, launches):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(launches):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / launches


def run_case(rows, hidden, dtype_name, mode, seed, num_warps, warmup, samples, launches):
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    def random(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype=dtype, device="cuda")
    r, w, dy = random((rows, hidden)), random((hidden,)), random((rows, hidden))
    h = random((rows, hidden)) if mode == "present" else None
    initial = [tensor.clone() for tensor in (r, w, dy)]
    if h is not None:
        initial.append(h.clone())
    dx, dres = torch.empty_like(r), torch.empty_like(r)
    dw = torch.empty_like(w)
    partial = torch.empty((rows, hidden), dtype=torch.float32, device="cuda")
    output = {"shape": [rows, hidden], "dtype": dtype_name, "grad_residual_out": mode,
              "seed": seed, "num_warps": num_warps}
    try:
        expected = analytical_reference(r, w, dy, h)
        autograd_expected = autograd_reference(r, w, dy, h)
        candidate_fn = lambda: candidate(r, w, dy, h, dx, dres, dw, partial, num_warps)
        baseline_fn = lambda: analytical_reference(r, w, dy, h)
        candidate_fn()
        torch.cuda.synchronize()
        checks = {
            name: compare(actual, reference)
            for name, actual, reference in zip(
                ("grad_x", "grad_residual", "grad_weight"),
                (dx, dres, dw), expected)
        }
        checks["autograd"] = {
            name: compare(actual, reference)
            for name, actual, reference in zip(
                ("grad_x", "grad_residual", "grad_weight"),
                (dx, dres, dw), autograd_expected)
        }
        checks["grad_x_equals_grad_residual"] = bool(torch.equal(dx, dres))
        tensors = [r, w, dy] + ([h] if h is not None else [])
        checks["inputs_unchanged"] = all(torch.equal(a, b) for a, b in zip(tensors, initial))
        output["checks"] = checks
        output["passed"] = (all(checks[name]["passed"] for name in
                                ("grad_x", "grad_residual", "grad_weight"))
                            and all(item["passed"] for item in checks["autograd"].values())
                            and checks["grad_x_equals_grad_residual"]
                            and checks["inputs_unchanged"])
        if not output["passed"]:
            return output
        for _ in range(warmup):
            candidate_fn()
            baseline_fn()
        torch.cuda.synchronize()
        timings = {"candidate": [], "analytical_pytorch": []}
        order = ["candidate", "analytical_pytorch"]
        funcs = {"candidate": candidate_fn, "analytical_pytorch": baseline_fn}
        for sample in range(samples):
            for name in (order if sample % 2 == 0 else list(reversed(order))):
                timings[name].append(timed_ms(funcs[name], launches))
        candidate_median = statistics.median(timings["candidate"])
        baseline_median = statistics.median(timings["analytical_pytorch"])
        output["timings_ms_per_call"] = timings
        output["median_ms"] = {"candidate": candidate_median,
                               "analytical_pytorch": baseline_median}
        output["speedup_vs_analytical_pytorch"] = baseline_median / candidate_median
    except Exception as exc:
        output["passed"] = False
        output["error"] = f"{type(exc).__name__}: {exc}"
    return output


def parse_shapes(value):
    return [tuple(map(int, item.lower().split("x"))) for item in value.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default="64x768,64x1024,64x4096,7x769,16x4099,17x1537")
    parser.add_argument("--dtypes", default="float16,bfloat16")
    parser.add_argument("--modes", default="present,absent")
    parser.add_argument("--seed", type=int, default=13001)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--launches", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shapes = parse_shapes(args.shapes)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "triton": triton.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "epsilon": EPSILON, "tolerance": {"atol": 0.03, "rtol": 0.03},
        "baseline": "stage3 analytical PyTorch backward; backward-only warm GPU event",
        "arguments": vars(args).copy(), "cases": [],
    }
    report["arguments"]["output"] = str(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for dtype_name in args.dtypes.split(","):
            for mode in args.modes.split(","):
                for rows, hidden in shapes:
                    case = run_case(rows, hidden, dtype_name, mode, args.seed, args.num_warps,
                                    args.warmup, args.samples, args.launches)
                    report["cases"].append(case)
                    print(f"{dtype_name} {mode} {rows}x{hidden}: passed={case['passed']} "
                          f"speedup={case.get('speedup_vs_analytical_pytorch')}", flush=True)
        report["status"] = "passed" if all(case["passed"] for case in report["cases"]) else "failed"
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"status={report['status']} cases={len(report['cases'])} output={args.output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

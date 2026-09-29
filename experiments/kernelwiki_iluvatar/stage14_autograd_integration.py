"""Synthetic forward/backward integration check for the BI-V150 RMSNorm kernels.

The shapes mimic the BDA-to-RMSNorm tensor rank but this is not a Megatron
training step. It checks a custom autograd wrapper and one weight update.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton

from stage3_residual_rmsnorm import residual_add_rmsnorm_kernel
from stage13_backward import EPSILON, candidate, compare


class FusedResidualRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight):
        if not (x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()):
            raise ValueError("stage14 wrapper requires contiguous tensors")
        if x.shape != residual.shape or x.shape[-1] != weight.numel():
            raise ValueError("shape mismatch")
        rows, hidden = x.numel() // x.shape[-1], x.shape[-1]
        y, residual_out = torch.empty_like(x), torch.empty_like(x)
        residual_add_rmsnorm_kernel[(rows,)](
            x, residual, weight, y, residual_out,
            ROWS=rows, HIDDEN=hidden, EPSILON=EPSILON,
            BLOCK=triton.next_power_of_2(hidden), num_warps=4,
        )
        ctx.save_for_backward(residual_out, weight)
        return y, residual_out

    @staticmethod
    def backward(ctx, dy, dres_observed):
        residual_out, weight = ctx.saved_tensors
        rows, hidden = residual_out.numel() // residual_out.shape[-1], residual_out.shape[-1]
        dy = torch.zeros_like(residual_out) if dy is None else dy.contiguous()
        dres_observed = None if dres_observed is None else dres_observed.contiguous()
        dx, dres = torch.empty_like(residual_out), torch.empty_like(residual_out)
        dw = torch.empty_like(weight)
        partial = torch.empty((rows, hidden), dtype=torch.float32, device=residual_out.device)
        candidate(residual_out.view(rows, hidden), weight, dy.view(rows, hidden),
                  None if dres_observed is None else dres_observed.view(rows, hidden),
                  dx.view(rows, hidden), dres.view(rows, hidden), dw, partial, num_warps=4)
        return dx, dres, dw


def reference(x, residual, weight):
    summed = x + residual
    values = summed.to(torch.float32)
    inv = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + EPSILON)
    output = (values * inv * weight.to(torch.float32)).to(x.dtype)
    return output, summed


def loss_for(y, summed):
    return y.to(torch.float32).square().mean() + 0.01 * summed.to(torch.float32).square().mean()


def run_case(shape, dtype_name, seed):
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    def random(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype=dtype, device="cuda")
    x0, residual0, w0 = random(shape), random(shape), random((shape[-1],))
    copies = []
    for _ in range(2):
        copies.append([v.detach().clone().requires_grad_(True) for v in (x0, residual0, w0)])
    (xc, rc, wc), (xr, rr, wr) = copies
    try:
        yc, summed_c = FusedResidualRMSNorm.apply(xc, rc, wc)
        yr, summed_r = reference(xr, rr, wr)
        lc, lr = loss_for(yc, summed_c), loss_for(yr, summed_r)
        lc.backward()
        lr.backward()
        torch.cuda.synchronize()
        checks = {
            "output": compare(yc, yr),
            "residual_out": compare(summed_c, summed_r),
            "loss": compare(lc, lr, atol=0.001, rtol=0.001),
            "grad_x": compare(xc.grad, xr.grad),
            "grad_residual": compare(rc.grad, rr.grad),
            "grad_weight": compare(wc.grad, wr.grad),
            "weight_after_sgd": compare(wc.detach() - 0.01 * wc.grad,
                                        wr.detach() - 0.01 * wr.grad),
        }
        checks["grad_x_equals_grad_residual"] = bool(torch.equal(xc.grad, rc.grad))
        checks["inputs_unchanged"] = all(
            torch.equal(actual.detach(), initial)
            for actual, initial in zip((xc, rc, wc), (x0, residual0, w0))
        )
        passed = all(v["passed"] for v in checks.values() if isinstance(v, dict)) and checks["grad_x_equals_grad_residual"] and checks["inputs_unchanged"]
        return {"shape": list(shape), "dtype": dtype_name, "seed": seed,
                "passed": passed, "checks": checks}
    except Exception as exc:
        return {"shape": list(shape), "dtype": dtype_name, "seed": seed,
                "passed": False, "error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=14001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shapes = ((8, 2, 1024), (7, 1, 769), (4, 4, 4099))
    cases = [run_case(shape, dtype, args.seed) for dtype in ("float16", "bfloat16") for shape in shapes]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(case["passed"] for case in cases) else "failed",
        "scope": "synthetic custom-autograd forward/backward and one SGD weight update; not Megatron training",
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "triton": triton.__version__,
        "seed": args.seed,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for case in cases:
        print(f"{case['dtype']} {case['shape']}: passed={case['passed']} error={case.get('error')}")
    print(f"status={report['status']} output={args.output}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

"""Verify autocast compatibility uses native fallback, with no compiled GPU kernels."""
import argparse
import json
from pathlib import Path
import torch
import benchmark

parser = argparse.ArgumentParser()
parser.add_argument("candidate", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
if args.output.exists():
    parser.error("output exists")
candidate = benchmark.inspect_candidate(args.candidate)
torch.manual_seed(23103)
x = torch.randn(128, 2, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
r = torch.randn_like(x, dtype=torch.float32, requires_grad=True)
w = torch.randn(512, device="cuda", requires_grad=True)
xr, rr, wr = [v.detach().clone().requires_grad_() for v in (x, r, w)]
with torch.autocast("cuda", dtype=torch.bfloat16):
    enabled = bool(torch.is_autocast_enabled())
    actual = candidate.fused(x, r, w, 1e-5)
    summed = rr + torch.nn.functional.dropout(xr.to(rr.dtype), p=0.0, training=True)
    expected = torch.nn.functional.rms_norm(summed, (512,), wr, 1e-5), summed
dy, dr = [torch.randn_like(v) for v in expected]
ag = torch.autograd.grad(actual, (x, r, w), (dy, dr))
eg = torch.autograd.grad(expected, (xr, rr, wr), (dy, dr))
checks = [benchmark.check(a, b, 0, 0) for a, b in zip((*actual, *ag), (*expected, *eg))]
cache_entries = len(candidate._COMPILED_KERNELS)
report = {"candidate_sha256": benchmark.digest(args.candidate), "cuda_autocast_enabled": enabled,
          "compiled_kernel_cache_entries_after_forward_backward": cache_entries, "checks": checks,
          "passed": enabled and cache_entries == 0 and all(x["passed"] for x in checks),
          "scope": "Exact native fallback compatibility; not optimized BF16 training"}
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(report["passed"], flush=True)

"""Independent CPU numerical audit, including evidence that training is nontrivial."""
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


def cpu_check(actual, expected, atol, rtol):
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    delta = (a - b).abs()
    finite = bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item())
    return {"passed": actual.dtype == expected.dtype and a.shape == b.shape and finite
            and bool((delta <= atol + rtol * b.abs()).all().item()),
            "max_abs": float(delta.max().item()), "reference_abs_max": float(b.abs().max().item()),
            "reference_nonzero": int((b != 0).sum().item()), "elements": b.numel(),
            "atol": atol, "rtol": rtol, "finite": finite}


benchmark.check = cpu_check
from stage19_megatron_route_step import load_megatron
parts = load_megatron(Path("/private/atrex-megatron/src/megatron-lm"))
candidate = benchmark.inspect_candidate(args.candidate)
settings = argparse.Namespace(seed=23102, layers=4, hidden=512, heads=8,
    sequence=128, micro_batch=2, vocab=1024, autocast=False, warmup=1, samples=2, steps=2)
result = benchmark.model_test(settings, candidate, parts)
args.output.write_text(json.dumps({"scope": "independent CPU correctness audit; throughput diagnostic only",
    "candidate_sha256": benchmark.digest(args.candidate), "model": result}, indent=2))
print(args.output, flush=True)

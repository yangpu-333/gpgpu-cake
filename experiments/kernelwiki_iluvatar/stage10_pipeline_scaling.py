"""Paired num_stages sweep for larger BI-V150 GEMM+bias workloads."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
import triton

from stage4_gemm_epilogue import correctness, gemm_bias


def measure_pair(base, candidate, samples=9, launches=20):
    for _ in range(10):
        base()
        candidate()
    torch.cuda.synchronize()
    out = {"base_us": [], "candidate_us": []}
    for i in range(samples):
        order = (("base", base), ("candidate", candidate)) if i % 2 == 0 else (
            ("candidate", candidate), ("base", base))
        for name, fn in order:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(launches):
                fn()
            stop.record()
            stop.synchronize()
            out[name + "_us"].append(float(start.elapsed_time(stop)) * 1000 / launches)
    out["base_median_us"] = statistics.median(out["base_us"])
    out["candidate_median_us"] = statistics.median(out["candidate_us"])
    out["speedup"] = out["base_median_us"] / out["candidate_median_us"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260928)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    report = {
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "triton": triton.__version__, "seed": args.seed,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing": {"scope": "forward, warm-buffer GPU events", "warmup": 10,
                   "samples": 9, "launches_per_sample": 20,
                   "pair_order": "alternates; same Triton tile, only num_stages differs"},
        "cases": [],
    }
    try:
        for m, n, k in ((512, 512, 2048), (1024, 1024, 1024)):
            a = torch.randn((m, k), device="cuda", dtype=torch.float16)
            b = torch.randn((k, n), device="cuda", dtype=torch.float16)
            bias = torch.randn((n,), device="cuda", dtype=torch.float16)
            reference = ((a.double().cpu() @ b.double().cpu()).half().float()
                         + bias.float().cpu()).half()
            outputs = {stage: torch.empty((m, n), device="cuda", dtype=torch.float16)
                       for stage in (1, 2, 3)}
            grid = (triton.cdiv(m, 32), triton.cdiv(n, 32))

            def launch(stage):
                return gemm_bias[grid](a, b, bias, outputs[stage], m, n, k,
                                       32, 32, 32, True,
                                       num_warps=4, num_stages=stage)

            case = {"mnk": [m, n, k], "tile": [32, 32, 32], "warps": 4,
                    "stages": [], "comparisons": []}
            for stage in (1, 2, 3):
                row = {"num_stages": stage}
                try:
                    compiled = launch(stage)
                    row["check"] = correctness(outputs[stage].cpu(), reference, 0.03, 0.03)
                    row["n_regs"] = getattr(compiled, "n_regs", None)
                    row["n_spills"] = getattr(compiled, "n_spills", None)
                    row["shared"] = getattr(compiled, "shared", None)
                    row["status"] = "passed" if row["check"]["passed"] else "numerical_failure"
                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = repr(exc)
                case["stages"].append(row)
            if all(row["status"] == "passed" for row in case["stages"]):
                for candidate in (2, 3):
                    base = lambda: launch(1)
                    variant = lambda stage=candidate: launch(stage)
                    timing = measure_pair(base, variant)
                    case["comparisons"].append({"base_stage": 1,
                                                "candidate_stage": candidate,
                                                "timing": timing})
            report["cases"].append(case)
            print((m, n, k), [(x["num_stages"], x["status"]) for x in case["stages"]],
                  flush=True)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

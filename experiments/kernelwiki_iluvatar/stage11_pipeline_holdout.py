"""BI-V150 GEMM+bias stage-count holdout across dtype and tile shape."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import triton

from stage4_gemm_epilogue import correctness, gemm_bias
from stage10_pipeline_scaling import measure_pair


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
                   "pair_order": "alternating stage 1 and candidate"},
        "cases": [],
    }
    try:
        for dtype in (torch.float16, torch.bfloat16):
            atol = rtol = 0.03 if dtype == torch.float16 else 0.05
            for m, n, k in ((512, 512, 2048), (1024, 1024, 1024)):
                a = torch.randn((m, k), device="cuda", dtype=dtype)
                b = torch.randn((k, n), device="cuda", dtype=dtype)
                bias = torch.randn((n,), device="cuda", dtype=dtype)
                reference = ((a.double().cpu() @ b.double().cpu()).to(dtype).float()
                             + bias.float().cpu()).to(dtype)
                for bm, bn, bk in ((32, 32, 32), (64, 64, 32)):
                    outputs = {stage: torch.empty((m, n), device="cuda", dtype=dtype)
                               for stage in (1, 2, 3)}
                    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))

                    def launch(stage):
                        return gemm_bias[grid](a, b, bias, outputs[stage], m, n, k,
                                               bm, bn, bk, True,
                                               num_warps=4, num_stages=stage)

                    case = {"dtype": str(dtype).split(".")[-1], "mnk": [m, n, k],
                            "tile": [bm, bn, bk], "warps": 4,
                            "atol": atol, "rtol": rtol,
                            "stages": [], "comparisons": []}
                    for stage in (1, 2, 3):
                        row = {"num_stages": stage}
                        try:
                            compiled = launch(stage)
                            row["check"] = correctness(outputs[stage].cpu(), reference,
                                                       atol, rtol)
                            row["n_regs"] = getattr(compiled, "n_regs", None)
                            row["n_spills"] = getattr(compiled, "n_spills", None)
                            row["shared"] = getattr(compiled, "shared", None)
                            row["status"] = ("passed" if row["check"]["passed"]
                                             else "numerical_failure")
                        except Exception as exc:
                            row["status"] = "error"
                            row["error"] = repr(exc)
                        case["stages"].append(row)
                    if all(row["status"] == "passed" for row in case["stages"]):
                        for candidate in (2, 3):
                            base = lambda: launch(1)
                            variant = lambda stage=candidate: launch(stage)
                            try:
                                timing = measure_pair(base, variant)
                                case["comparisons"].append({"base_stage": 1,
                                                            "candidate_stage": candidate,
                                                            "timing": timing,
                                                            "status": "measured"})
                            except Exception as exc:
                                case["comparisons"].append({"base_stage": 1,
                                                            "candidate_stage": candidate,
                                                            "status": "error",
                                                            "error": repr(exc)})
                    report["cases"].append(case)
                    print(case["dtype"], case["mnk"], case["tile"],
                          [row["status"] for row in case["stages"]], flush=True)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()

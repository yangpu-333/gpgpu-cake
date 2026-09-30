"""Run the stage21 full GPT comparison under BI-V150 BF16 autocast.

This exploratory wrapper changes only the process-local forward precision.
The stage21 model and candidate code remain the benchmark implementation.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

import stage21_megatron_gpt_fusion as stage21


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()
    original_run_step = stage21.run_step
    original_forward = stage21.FusedResidualRMSNorm.forward
    observed = []

    def observed_forward(ctx, x, residual, weight):
        result = original_forward(ctx, x, residual, weight)
        if not observed:
            observed.append({
                "shape": list(x.shape),
                "input_dtype": str(x.dtype),
                "residual_dtype": str(residual.dtype),
                "weight_dtype": str(weight.dtype),
                "normalized_output_dtype": str(result[0].dtype),
            })
        return result

    def bf16_run_step(model, inputs, optimizer, *, timed=False):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return original_run_step(model, inputs, optimizer, timed=timed)

    stage21.run_step = bf16_run_step
    stage21.FusedResidualRMSNorm.forward = staticmethod(observed_forward)
    try:
        code = stage21.main()
    finally:
        stage21.run_step = original_run_step
        stage21.FusedResidualRMSNorm.forward = original_forward
    result = json.loads(args.output.read_text(encoding="utf-8"))
    result["autocast_dtype"] = "torch.bfloat16"
    result["wrapper_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["candidate_observed"] = observed[0] if observed else None
    result["scope"] += "; BF16 autocast around each training step"
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())

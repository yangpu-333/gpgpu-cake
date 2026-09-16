"""Check whether the pinned Megatron checkout can construct fused Residual RMSNorm.

This is a small module-level smoke test for a host with no known training launch
command.  It uses the exact Megatron checkout provided at runtime, does not edit
that checkout, requires no dataset or checkpoint, and performs one forward and
backward pass only when the selected module is available.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_COMMIT = "5be9626709af2722333bf54797c954c09edeada3"
EXPECTED_CLASS = "TEFusedResidualRMSNorm"


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def validate_dimensions(sequence_length, micro_batch_size, hidden_size, num_attention_heads):
    if hidden_size % num_attention_heads:
        raise ValueError("hidden size must be divisible by number of attention heads")
    return {
        "shape": [sequence_length, micro_batch_size, hidden_size],
        "flattened_rows": sequence_length * micro_batch_size,
        "hidden_size": hidden_size,
    }


def git_head(repository):
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False, capture_output=True, text=True, timeout=20,
    )
    return {"returncode": completed.returncode, "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}


def smoke_status(module_class_name, forward_passed, backward_passed):
    if module_class_name != EXPECTED_CLASS:
        return "fused_module_not_selected"
    return "module_smoke_passed" if forward_passed and backward_passed else "module_smoke_failed"


def tensor_info(tensor):
    return {
        "shape": [int(value) for value in tensor.shape],
        "stride": [int(value) for value in tensor.stride()],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
    }


def finite(torch, tensor):
    return bool(torch.isfinite(tensor.detach()).all())


def run_smoke(args, report):
    root = args.megatron_root.expanduser().resolve()
    report["megatron"] = {"root": str(root), "expected_commit": EXPECTED_COMMIT, "head": git_head(root)}
    if report["megatron"]["head"]["stdout"] != EXPECTED_COMMIT:
        report["status"] = "megatron_commit_mismatch"
        return 2
    if not (root / "megatron").is_dir():
        report["status"] = "megatron_package_missing"
        return 2
    dimensions = validate_dimensions(
        args.sequence_length, args.micro_batch_size, args.hidden_size, args.num_attention_heads
    )
    report["requested_workload"] = dimensions
    sys.path.insert(0, str(root))
    try:
        import torch
        from megatron.core.extensions import transformer_engine as megatron_te
        from megatron.core.transformer.transformer_config import TransformerConfig
    except Exception as exc:
        report.update(status="megatron_import_failed", error_type=type(exc).__name__, error=str(exc))
        return 3
    finally:
        if sys.path and sys.path[0] == str(root):
            sys.path.pop(0)
    report["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count(),
        "transformer_engine_available": bool(getattr(megatron_te, "HAVE_TE", False)),
        "fused_class_available": getattr(megatron_te, EXPECTED_CLASS, None) is not None,
    }
    if not report["environment"]["cuda_available"]:
        report["status"] = "cuda_unavailable"
        return 3
    if not report["environment"]["transformer_engine_available"]:
        report["status"] = "transformer_engine_unavailable"
        return 3
    if not report["environment"]["fused_class_available"]:
        report["status"] = "fused_module_unavailable"
        return 3
    if args.device >= torch.cuda.device_count():
        report["status"] = "invalid_device"
        return 3

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        normalization="RMSNorm",
        fused_residual_rmsnorm=True,
        sequence_parallel=False,
    )
    report["active_stage"] = "construct_same_factory_path"
    try:
        # TENorm is the exact factory Megatron's transformer-engine spec uses.
        module = megatron_te.TENorm(
            config, args.hidden_size, args.epsilon, has_residual=True
        ).to(device)
        report["selected_module"] = {"class_name": type(module).__name__, "module_path": type(module).__module__}
        if type(module).__name__ != EXPECTED_CLASS:
            report["status"] = "fused_module_not_selected"
            return 3
        dtype = getattr(torch, args.dtype)
        input_tensor = torch.randn(
            dimensions["shape"], device=device, dtype=dtype, requires_grad=True
        )
        report["active_stage"] = "forward"
        normalized, residual = module(input_tensor)
        forward_passed = (
            tuple(normalized.shape) == tuple(input_tensor.shape)
            and tuple(residual.shape) == tuple(input_tensor.shape)
            and finite(torch, normalized)
            and finite(torch, residual)
        )
        report["forward"] = {
            "passed": forward_passed,
            "input": tensor_info(input_tensor),
            "normalized_output": tensor_info(normalized),
            "residual_output": tensor_info(residual),
            "residual_aliases_input": bool(residual.data_ptr() == input_tensor.data_ptr()),
        }
        report["active_stage"] = "backward"
        loss = normalized.float().square().mean() + residual.float().square().mean()
        loss.backward()
        weight_grad = getattr(module, "weight").grad
        backward_passed = input_tensor.grad is not None and weight_grad is not None and finite(torch, input_tensor.grad) and finite(torch, weight_grad)
        report["backward"] = {
            "passed": backward_passed,
            "input_grad_finite": finite(torch, input_tensor.grad) if input_tensor.grad is not None else False,
            "weight_grad_finite": finite(torch, weight_grad) if weight_grad is not None else False,
        }
        torch.cuda.synchronize()
        report["status"] = smoke_status(type(module).__name__, forward_passed, backward_passed)
        return 0 if report["status"] == "module_smoke_passed" else 1
    except Exception as exc:
        report.update(status="module_smoke_failed", error_type=type(exc).__name__, error=str(exc))
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sequence-length", type=positive, default=8)
    parser.add_argument("--micro-batch-size", type=positive, default=2)
    parser.add_argument("--hidden-size", type=positive, default=1024)
    parser.add_argument("--num-attention-heads", type=positive, default=8)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.device < 0 or args.epsilon <= 0:
        parser.error("device must be non-negative and epsilon must be positive")
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite evidence")
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operator": "megatron_fused_residual_rmsnorm_preflight",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        code = run_smoke(args, report)
    except Exception as exc:
        report.update(status="run_failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        code = 1
    report.pop("active_stage", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print("status:", report["status"])
    print("report:", args.output.resolve())
    return code


if __name__ == "__main__":
    raise SystemExit(main())

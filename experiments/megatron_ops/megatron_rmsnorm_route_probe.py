"""Identify and smoke-test the RMSNorm route selected by Megatron's local GPT spec.

``TransformerBlock`` has a process-wide default normalization factory, but it
does not decide a local GPT RMSNorm experiment. In this pinned version,
``get_gpt_layer_local_spec(normalization="RMSNorm")`` explicitly requests
``WrappedTorchNorm`` even if Apex is importable. This probe builds that actual
spec, then executes Megatron's bias/dropout/add helper followed by the selected
RMSNorm once, with dropout and bias disabled. It is a path and correctness
check, never a performance benchmark.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_COMMIT = "5be9626709af2722333bf54797c954c09edeada3"


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def git_head(repository):
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def select_route(factory_name):
    """Describe the normalization factory selected by the local GPT spec."""
    if factory_name == "WrappedTorchNorm":
        return {
            "kind": "gpt_local_wrapped_torch_rmsnorm",
            "rmsnorm_supported_by_selected_factory": True,
            "next_action": "run the local GPT BDA-to-RMSNorm smoke test",
        }
    return {
        "kind": "unexpected_gpt_local_rmsnorm_factory",
        "rmsnorm_supported_by_selected_factory": False,
        "next_action": "inspect the locally modified GPT layer specification",
    }


def tensor_info(tensor):
    return {
        "shape": [int(value) for value in tensor.shape],
        "stride": [int(value) for value in tensor.stride()],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
    }


def finite(torch, tensor):
    return bool(tensor is not None and torch.isfinite(tensor.detach()).all())


def run_probe(args, report):
    root = args.megatron_root.expanduser().resolve()
    report["megatron"] = {"root": str(root), "expected_commit": EXPECTED_COMMIT, "head": git_head(root)}
    if report["megatron"]["head"]["stdout"] != EXPECTED_COMMIT:
        report["status"] = "megatron_commit_mismatch"
        return 2
    if not (root / "megatron").is_dir():
        report["status"] = "megatron_package_missing"
        return 2

    sys.path.insert(0, str(root))
    try:
        import torch
        from megatron.core.extensions.transformer_engine import HAVE_TE
        from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
        from megatron.core.transformer.transformer_block import HAVE_APEX, LayerNormImpl
        from megatron.core.transformer.transformer_config import TransformerConfig
    except Exception as exc:
        report.update(status="megatron_import_failed", error_type=type(exc).__name__, error=str(exc))
        return 3
    finally:
        if sys.path and sys.path[0] == str(root):
            sys.path.pop(0)

    default_factory_name = getattr(LayerNormImpl, "__name__", type(LayerNormImpl).__name__)
    report["active_stage"] = "build_gpt_local_rmsnorm_spec"
    local_spec = get_gpt_layer_local_spec(normalization="RMSNorm")
    input_norm_builder = local_spec.submodules.input_layernorm
    pre_mlp_norm_builder = local_spec.submodules.pre_mlp_layernorm
    factory_name = getattr(input_norm_builder, "__name__", type(input_norm_builder).__name__)
    route = select_route(factory_name)
    report["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count(),
        "transformer_engine_available": bool(HAVE_TE),
        "apex_available": bool(HAVE_APEX),
        "transformer_block_default_factory": default_factory_name,
        "gpt_local_input_layernorm_factory": factory_name,
        "gpt_local_pre_mlp_layernorm_factory": getattr(
            pre_mlp_norm_builder, "__name__", type(pre_mlp_norm_builder).__name__
        ),
    }
    report["selected_route"] = route
    if not report["environment"]["cuda_available"]:
        report["status"] = "cuda_unavailable"
        return 3
    if args.device >= torch.cuda.device_count():
        report["status"] = "invalid_device"
        return 3
    if route["kind"] != "gpt_local_wrapped_torch_rmsnorm":
        report["status"] = "rmsnorm_not_available_in_gpt_local_spec"
        return 4

    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    shape = [args.sequence_length, args.micro_batch_size, args.hidden_size]
    config = TransformerConfig(
        num_layers=1,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        normalization="RMSNorm",
        sequence_parallel=False,
    )
    report["active_stage"] = "construct_wrapped_torch_rmsnorm"
    try:
        norm = input_norm_builder(config, args.hidden_size, args.epsilon).to(device)
        if type(norm).__name__ != "RMSNorm":
            report["status"] = "gpt_local_factory_did_not_construct_rmsnorm"
            report["constructed_norm_class"] = type(norm).__name__
            return 4
        dtype = getattr(torch, args.dtype)
        branch_output = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        residual = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        report["active_stage"] = "megatron_bias_dropout_add"
        # The probe fixes bias=None and dropout=0.0.  Those are the safe, explicit
        # semantics for a first add-plus-RMSNorm candidate; real training capture
        # decides whether a broader dropout/bias-capable candidate is needed.
        added = get_bias_dropout_add(training=True, fused=args.fused_bda)(
            (branch_output, None), residual, 0.0
        )
        report["active_stage"] = "rmsnorm"
        normalized = norm(added)
        forward_passed = (
            tuple(added.shape) == tuple(branch_output.shape)
            and tuple(normalized.shape) == tuple(branch_output.shape)
            and finite(torch, added)
            and finite(torch, normalized)
        )
        report["forward"] = {
            "passed": forward_passed,
            "branch_output": tensor_info(branch_output),
            "residual": tensor_info(residual),
            "residual_added_output": tensor_info(added),
            "normalized_output": tensor_info(normalized),
            "bias": None,
            "dropout_probability": 0.0,
            "bias_dropout_fusion_requested": bool(args.fused_bda),
        }
        report["active_stage"] = "backward"
        normalized.float().square().mean().backward()
        weight_grad = norm.weight.grad
        backward_passed = all(
            (
                finite(torch, branch_output.grad),
                finite(torch, residual.grad),
                finite(torch, weight_grad),
            )
        )
        report["backward"] = {
            "passed": backward_passed,
            "branch_output_grad_finite": finite(torch, branch_output.grad),
            "residual_grad_finite": finite(torch, residual.grad),
            "weight_grad_finite": finite(torch, weight_grad),
        }
        torch.cuda.synchronize()
        report["status"] = "fallback_path_smoke_passed" if forward_passed and backward_passed else "fallback_path_smoke_failed"
        return 0 if report["status"] == "fallback_path_smoke_passed" else 1
    except Exception as exc:
        report.update(status="fallback_path_smoke_failed", error_type=type(exc).__name__, error=str(exc))
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
    parser.add_argument("--fused-bda", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.device < 0 or args.epsilon <= 0:
        parser.error("device must be non-negative and epsilon must be positive")
    if args.hidden_size % args.num_attention_heads:
        parser.error("hidden size must be divisible by number of attention heads")
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite evidence")
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operator": "megatron_residual_add_rmsnorm_route_probe",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        code = run_probe(args, report)
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

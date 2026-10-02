"""Compare a BI-V150 fused step with pinned Megatron's local BDA/RMSNorm route.

This is a real Megatron-selected operator path with synthetic tensors, not a
complete GPT model step. Transformer Engine is masked only in this process;
CoreX DTensor classes are exposed under the import path this Megatron expects.
"""

import argparse
import hashlib
import json
import sys
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton

from stage3_residual_rmsnorm import residual_add_rmsnorm_kernel
from stage13_backward import EPSILON, candidate


PINNED_MEGATRON = "5be9626709af2722333bf54797c954c09edeada3"


def load_megatron(root):
    root = root.resolve()
    head = root / ".git" / "HEAD"
    if not head.is_file() or head.read_text().strip() != PINNED_MEGATRON:
        raise RuntimeError("pinned Megatron detached HEAD does not match")
    import torch.distributed._tensor as legacy_tensor_namespace
    import torch.distributed.tensor as tensor_namespace

    for name in ("DTensor", "DeviceMesh", "Partial", "Placement", "Replicate",
                 "Shard", "distribute_module", "distribute_tensor"):
        if not hasattr(tensor_namespace, name):
            setattr(tensor_namespace, name, getattr(legacy_tensor_namespace, name))
    sys.modules.setdefault(
        "torch.distributed.tensor.placement_types",
        legacy_tensor_namespace.placement_types,
    )
    sys.modules["transformer_engine"] = None
    namespace = types.ModuleType("megatron")
    namespace.__path__ = [str(root / "megatron")]
    namespace.__package__ = "megatron"
    sys.modules["megatron"] = namespace

    from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.transformer.transformer_config import TransformerConfig
    return get_bias_dropout_add, get_gpt_layer_local_spec, TransformerConfig


class FusedResidualRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, residual, weight):
        if not (x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()):
            raise ValueError("contiguous tensors required")
        if x.shape != residual.shape or x.shape[-1] != weight.numel():
            raise ValueError("shape mismatch")
        rows, hidden = x.numel() // x.shape[-1], x.shape[-1]
        y = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        residual_out = torch.empty_like(x)
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
        dy = torch.zeros_like(residual_out, dtype=torch.float32) if dy is None else dy.contiguous()
        dres_observed = None if dres_observed is None else dres_observed.contiguous()
        dx = torch.empty_like(residual_out)
        dres = torch.empty_like(residual_out)
        dw = torch.empty_like(weight)
        partial = torch.empty((rows, hidden), dtype=torch.float32, device=residual_out.device)
        candidate(
            residual_out.view(rows, hidden), weight, dy.view(rows, hidden),
            None if dres_observed is None else dres_observed.view(rows, hidden),
            dx.view(rows, hidden), dres.view(rows, hidden), dw, partial, num_warps=4,
        )
        return dx, dres, dw


def compare(actual, expected, atol=0.03, rtol=0.03):
    a, b = actual.detach().float(), expected.detach().float()
    return {
        "passed": bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
        "max_abs_error": float((a - b).abs().max().item()),
    }


def run_case(args, report):
    bda_factory, spec_factory, config_type = load_megatron(args.megatron_root)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    shape = (args.sequence_length, args.micro_batch_size, args.hidden_size)
    spec = spec_factory(normalization="RMSNorm")
    norm_builder = spec.submodules.input_layernorm
    report["route"] = {
        "factory": getattr(norm_builder, "__name__", str(norm_builder)),
        "shape": list(shape),
        "dtype": args.dtype,
        "bias": None,
        "dropout": 0.0,
        "fused_bda": False,
    }
    config = config_type(
        num_layers=1, hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        normalization="RMSNorm", sequence_parallel=False,
    )
    norm = norm_builder(config, args.hidden_size, EPSILON).cuda()
    if type(norm).__name__ != "RMSNorm":
        raise RuntimeError(f"unexpected norm: {type(norm).__name__}")
    bda = bda_factory(training=True, fused=False)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    def random():
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            dtype=dtype, device="cuda"
        )

    x0, residual0 = random(), random()
    xr, rr = (v.detach().clone().requires_grad_(True) for v in (x0, residual0))
    xc, rc = (v.detach().clone().requires_grad_(True) for v in (x0, residual0))
    wc = norm.weight.detach().clone().requires_grad_(True)
    yr_input = bda((xr, None), rr, 0.0)
    yr = norm(yr_input)
    yc, yc_residual = FusedResidualRMSNorm.apply(xc, rc, wc)
    report["route"]["reference_output_dtype"] = str(yr.dtype)
    report["route"]["candidate_output_dtype"] = str(yc.dtype)
    loss_ref = yr.float().square().mean() + 0.01 * yr_input.float().square().mean()
    loss_candidate = yc.float().square().mean() + 0.01 * yc_residual.float().square().mean()
    loss_ref.backward()
    loss_candidate.backward()
    with torch.no_grad():
        ref_updated = norm.weight - args.learning_rate * norm.weight.grad
        candidate_updated = wc - args.learning_rate * wc.grad
    checks = {
        "output": compare(yc, yr),
        "residual_out": compare(yc_residual, yr_input),
        "loss": compare(loss_candidate.reshape(1), loss_ref.reshape(1)),
        "grad_x": compare(xc.grad, xr.grad),
        "grad_residual": compare(rc.grad, rr.grad),
        "grad_weight": compare(wc.grad, norm.weight.grad),
        "weight_after_sgd": compare(candidate_updated, ref_updated),
    }
    report["checks"] = checks
    report["status"] = "passed" if all(v["passed"] for v in checks.values()) else "failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=19001)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    args = parser.parse_args()
    if args.hidden_size % args.num_attention_heads:
        parser.error("hidden size must be divisible by num attention heads")
    if args.output.exists():
        parser.error("output exists")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "megatron_commit": PINNED_MEGATRON,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "seed": args.seed,
        "scope": "pinned Megatron local BDA/RMSNorm operator path with synthetic inputs and one SGD weight update; not a full GPT training step",
    }
    try:
        run_case(args, report)
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(report["status"], args.output)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

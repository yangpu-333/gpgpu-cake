"""Supplemental CE verifier and candidate-only Megatron adapter.

The parent benchmark owns the model, batches, correctness checks and full-step
timer. This module extends only its optimized-model installation in the
``extended`` variant. Standalone CE checks never produce timing evidence.
"""

import argparse
import contextlib
import hashlib
import json
import os
import sys
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE_ROOT = ROOT.parent
sys.path.insert(0, str(BASE_ROOT))
import benchmark as base


SHAPES = ((128, 2, 1024), (17, 2, 512), (7, 3, 769), (1, 1, 7), (2, 1, 1))
DISTRIBUTIONS = ("random-0.001", "random-1", "random-100", "zero-ties",
                 "offset-10000", "dominant-positive-1000", "dominant-negative-1000")
LABEL_PATTERNS = ("valid-boundaries", "outside-and-boundaries")
LAYOUTS = ("contiguous", "last-stride-2", "transpose-first-two")
ATOL, RTOL = 3e-4, 1e-3


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fusion_reason(logits, target, tp_group, loss_fusion=False):
    """Return an unsupported reason; value-dependent checks do not synchronize."""
    import torch
    if loss_fusion:
        return "configured native loss fusion"
    if tp_group is None or tp_group.size() != 1:
        return "tensor parallel group is not singleton"
    if not (logits.is_cuda and target.is_cuda and logits.device == target.device):
        return "device"
    if logits.ndim != 3 or tuple(target.shape) != tuple(logits.shape[:2]):
        return "rank or shape"
    if logits.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return "logits dtype"
    if target.dtype != torch.int64:
        return "target dtype"
    if logits.shape[-1] < 1 or logits.numel() == 0:
        return "empty logits"
    if not (logits.is_contiguous() and target.is_contiguous()):
        return "layout"
    return None


def tensor_metadata(value):
    return {"shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "device": str(value.device)}


def safe_cache_key(value):
    """Describe launch metadata without rendering tensors or result contents."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return [safe_cache_key(item) for item in value]
    value_type = type(value)
    if value_type.__module__ == "torch" and value_type.__name__ in ("dtype", "device"):
        return str(value)
    return {"opaque_type": value_type.__module__ + "." + value_type.__name__}


def compiled_kernel_inventory(module):
    """Inventory compiled code metadata; assembly keys do not prove instructions."""
    caches = []
    for name, cache in vars(module).items():
        if name.startswith("__") or not isinstance(cache, dict):
            continue
        entries = []
        for key, value in cache.items():
            if not (hasattr(value, "asm") or hasattr(value, "metadata")):
                continue
            asm = getattr(value, "asm", None)
            metadata = getattr(value, "metadata", None)
            entries.append({"cache_key": safe_cache_key(key),
                            "compiled_type": type(value).__module__ + "." + type(value).__name__,
                            "available_asm_keys": sorted(str(item) for item in asm) if isinstance(asm, dict) else [],
                            "metadata_type": (type(metadata).__module__ + "." + type(metadata).__name__)
                            if metadata is not None else None})
        if entries:
            caches.append({"cache_name": name, "compiled_entries": entries})
    return {"scope": "process-local compiled-kernel cache and code-format metadata only; "
                     "no tensor/result values or assembly contents; presence does not establish "
                     "particular hardware instructions or per-call kernel execution",
            "compiled_objects": sum(len(cache["compiled_entries"]) for cache in caches),
            "caches": caches}


def extended_install(model, candidate, original_install):
    counts = original_install(model, candidate)
    native = model.compute_language_model_loss
    ce_counts = {"optimized": 0, "fallback": 0, "fallback_reasons": {},
                 "observations": [],
                 "scope": "adapter dispatch counts; kernel execution is verified separately"}
    counts["cross_entropy"] = ce_counts

    def compute_loss(self, labels, logits):
        # Preserve the native [b,s] -> contiguous [s,b] conversion. Do not change
        # model configuration, labels, logits, optimizer or parent step timing.
        target = labels.transpose(0, 1).contiguous() if labels.ndim == 2 else labels
        reason = fusion_reason(logits, target, self.tp_group,
                               self.config.cross_entropy_loss_fusion)
        if reason is not None:
            ce_counts["fallback"] += 1
            ce_counts["fallback_reasons"][reason] = ce_counts["fallback_reasons"].get(reason, 0) + 1
            return native(labels, logits)
        loss = candidate.cross_entropy(logits, target)
        ce_counts["optimized"] += 1
        if len(ce_counts["observations"]) < 4:
            ce_counts["observations"].append({"logits": tensor_metadata(logits),
                                               "target": tensor_metadata(target),
                                               "loss": tensor_metadata(loss)})
        return loss.transpose(0, 1).contiguous()

    model.compute_language_model_loss = types.MethodType(compute_loss, model)
    return counts


@contextlib.contextmanager
def candidate_installation(variant):
    original = base.install
    if variant == "extended":
        base.install = lambda model, candidate: extended_install(model, candidate, original)
    try:
        yield
    finally:
        base.install = original


def make_values(torch, shape, distribution, generator):
    if distribution == "zero-ties":
        return torch.zeros(shape, dtype=torch.float32)
    values = torch.randn(shape, generator=generator, dtype=torch.float32)
    if distribution.startswith("random-"):
        return values * float(distribution.split("-", 1)[1])
    if distribution == "offset-10000":
        return values + 10000
    values.zero_()
    values[..., 0] = 1000 if distribution == "dominant-positive-1000" else -1000
    return values


def make_logits(torch, values, dtype, layout):
    values = values.to(device="cuda", dtype=dtype)
    if layout == "contiguous":
        result = values.clone()
    elif layout == "last-stride-2":
        storage = torch.empty((*values.shape[:-1], values.shape[-1] * 2),
                              device="cuda", dtype=dtype)
        result = storage[..., ::2]
        result.copy_(values)
    else:
        # Materialize reversed leading dimensions before transposition. This
        # preserves the requested logical shape and creates the native view()
        # incompatibility when both leading dimensions exceed one.
        result = values.transpose(0, 1).contiguous().transpose(0, 1)
    return result.detach().requires_grad_()


def make_target(torch, shape, pattern):
    vocab = shape[-1]
    labels = [0, vocab - 1] if pattern == "valid-boundaries" else [-100, -1, vocab, vocab + 1, 0, vocab - 1]
    rows = shape[0] * shape[1]
    repeated = (labels * ((rows + len(labels) - 1) // len(labels)))[:rows]
    return torch.tensor(repeated, dtype=torch.int64, device="cuda").reshape(shape[:2])


def exception_record(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


def cross_entropy_cases(args, candidate):
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29700 + args.seed % 100))
    dist.init_process_group("nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    group = parallel_state.get_tensor_model_parallel_group()
    generator = torch.Generator().manual_seed(args.seed + 300)
    specifications = [(shape, dtype, distribution, labels, layout)
                      for shape in SHAPES
                      for dtype in (torch.float32, torch.float16, torch.bfloat16)
                      for distribution in DISTRIBUTIONS
                      for labels in LABEL_PATTERNS
                      for layout in LAYOUTS]
    if args.smoke:
        specifications = specifications[:6]
    records = []
    dispatch = {"optimized": 0, "fallback": 0, "fallback_reasons": {}}
    try:
        for shape, dtype, distribution, labels, layout in specifications:
            values = make_values(torch, shape, distribution, generator)
            xr, xc = (make_logits(torch, values, dtype, layout) for _ in range(2))
            native_before, candidate_before = xr.detach().cpu().clone(), xc.detach().cpu().clone()
            tr = make_target(torch, shape, labels)
            tc = tr.clone()
            target_before = tr.detach().cpu().clone()
            rows = shape[0] * shape[1]
            upstream = torch.linspace(-1.25, 1.75, rows, device="cuda").reshape(shape[:2])
            if rows > 1:
                upstream.reshape(-1)[::3] = 0
            reason = fusion_reason(xc, tc, group)
            route = "optimized" if reason is None else "native-fallback"
            key = "optimized" if reason is None else "fallback"
            dispatch[key] += 1
            if reason is not None:
                dispatch["fallback_reasons"][reason] = dispatch["fallback_reasons"].get(reason, 0) + 1
            record = {"shape": list(shape), "dtype": str(dtype), "distribution": distribution,
                      "labels": labels, "layout": layout, "route": route,
                      "fallback_reason": reason, "logits": tensor_metadata(xc),
                      "label_values": sorted(set(tc.detach().cpu().reshape(-1).tolist())),
                      "timing": "none; correctness only", "checks": {}}
            native_error = candidate_error = None
            native_loss = candidate_loss = None
            try:
                native_loss = vocab_parallel_cross_entropy(xr, tr, tp_group=group)
            except Exception as exc:
                native_error = exception_record(exc)
            try:
                candidate_loss = (candidate.cross_entropy(xc, tc) if reason is None else
                                  vocab_parallel_cross_entropy(xc, tc, tp_group=group))
            except Exception as exc:
                candidate_error = exception_record(exc)
            record["native_error"], record["candidate_error"] = native_error, candidate_error
            if native_error is not None:
                record.update(baseline_valid=False, passed=False,
                              compatibility_passed=route == "native-fallback" and native_error == candidate_error,
                              compatibility_scope="native layout error; not finite correctness")
                records.append(record)
                continue
            if candidate_error is not None:
                record.update(baseline_valid=bool(torch.isfinite(native_loss).all().item()),
                              passed=False, compatibility_passed=False)
                records.append(record)
                continue

            # Capture forward mutation before either backward overwrites the
            # saved probabilities. FP32 must mirror native; low precision must
            # remain exact because native creates a separate .float() tensor.
            native_after, candidate_after = xr.detach().cpu().clone(), xc.detach().cpu().clone()
            if dtype == torch.float32:
                record["checks"]["forward_input_matches_native_probabilities"] = base.check(
                    candidate_after, native_after, ATOL, RTOL)
            else:
                record["checks"]["native_input_exactly_unchanged"] = base.check(native_after, native_before, 0, 0)
                record["checks"]["candidate_input_exactly_unchanged"] = base.check(candidate_after, candidate_before, 0, 0)
            record["checks"]["native_targets_exactly_unchanged"] = base.check(tr.detach().cpu(), target_before, 0, 0)
            record["checks"]["candidate_targets_exactly_unchanged"] = base.check(tc.detach().cpu(), target_before, 0, 0)
            record["checks"]["loss"] = base.check(candidate_loss, native_loss, ATOL, RTOL)
            native_grad = torch.autograd.grad(native_loss, xr, upstream)[0]
            candidate_grad = torch.autograd.grad(candidate_loss, xc, upstream)[0]
            record["checks"]["dlogits"] = base.check(candidate_grad, native_grad, ATOL, RTOL)
            record["loss"] = tensor_metadata(candidate_loss)
            record["baseline_valid"] = bool(torch.isfinite(native_loss).all().item()
                                               and torch.isfinite(native_grad).all().item())
            record["passed"] = record["baseline_valid"] and all(x["passed"] for x in record["checks"].values())
            if not record["baseline_valid"]:
                record["native_invalid_compatibility"] = {
                    "loss": base.native_invalid_compatibility(candidate_loss, native_loss, ATOL, RTOL),
                    "dlogits": base.native_invalid_compatibility(candidate_grad, native_grad, ATOL, RTOL)}
                record["compatibility_passed"] = all(x["compatible"] for x in record["native_invalid_compatibility"].values())
            records.append(record)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()
    valid = [record for record in records if record["baseline_valid"]]
    invalid = [record for record in records if not record["baseline_valid"]]
    summary = {"expected_cases": len(specifications), "observed_cases": len(records),
               "finite_native_cases": len(valid), "finite_correctness_passed": sum(r["passed"] for r in valid),
               "native_invalid_cases": len(invalid),
               "native_invalid_compatibility_passed": sum(r.get("compatibility_passed", False) for r in invalid)}
    passed = (bool(valid) and len(records) == len(specifications)
              and all(record["passed"] for record in valid)
              and all(record.get("compatibility_passed", False) for record in invalid))
    return records, summary, dispatch, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("model", "cross-entropy"), required=True)
    parser.add_argument("--variant", choices=("extended", "current-best"), default="extended")
    parser.add_argument("--seed", type=int, default=24001)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; evidence is immutable")
    if args.mode == "cross-entropy" and args.variant != "extended":
        parser.error("cross-entropy requires --variant extended")
    report = {"utc": datetime.now(timezone.utc).isoformat(),
              "mode": "ce" if args.mode == "cross-entropy" else "model",
              "requested_mode": args.mode, "variant": args.variant, "seed": args.seed,
              "base_harness_sha256": sha256(base.__file__),
              "base_contract_sha256": sha256(BASE_ROOT / "contract.json"),
              "extension_harness_sha256": sha256(__file__),
              "contract_sha256": sha256(ROOT / "contract.json"),
              "candidate_sha256": sha256(args.candidate),
              "baseline": "pinned Megatron native local PyTorch backend", "status": "running"}
    candidate = None
    try:
        import torch
        import triton
        from stage19_megatron_route_step import load_megatron, PINNED_MEGATRON
        report["environment"] = {"torch": torch.__version__, "triton": triton.__version__,
                                 "gpu": torch.cuda.get_device_name(0),
                                 "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                                 "megatron_commit": PINNED_MEGATRON}
        report["native_sources_sha256"] = {path: sha256(args.megatron_root / path) for path in (
            "megatron/core/transformer/torch_norm.py",
            "megatron/core/fusions/fused_bias_dropout.py",
            "megatron/core/tensor_parallel/cross_entropy.py",
            "megatron/core/models/common/language_module/language_module.py",
            "megatron/core/models/gpt/gpt_model.py")}
        parts = load_megatron(args.megatron_root)
        candidate = base.inspect_candidate(args.candidate)
        if args.variant == "extended" and not callable(getattr(candidate, "cross_entropy", None)):
            raise ValueError("extended candidate must expose cross_entropy(logits, target)")
        if args.mode == "model":
            with candidate_installation(args.variant):
                report["model"] = base.model_test(args, candidate, parts)
            passed = report["model"]["passed"]
            if args.variant == "extended":
                ce = report["model"]["candidate_calls"]["cross_entropy"]
                passed = passed and ce["optimized"] > 0
                report["model"]["passed"] = bool(passed)
        else:
            records, summary, dispatch, passed = cross_entropy_cases(args, candidate)
            report.update(cases=records, case_summary=summary, cross_entropy_dispatch=dispatch)
        report["status"] = "passed" if passed else "failed"
    except Exception as exc:
        report.update(status="error", error=str(exc), traceback=traceback.format_exc())
    if candidate is not None:
        report["compiled_kernel_inventory"] = compiled_kernel_inventory(candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["status"], args.output, flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

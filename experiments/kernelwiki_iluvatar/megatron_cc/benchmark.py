"""Fixed verifier for Claude Code candidates against pinned native Megatron.

Only the candidate module is replaceable. GPU events measure operator calls;
synchronized wall-clock blocks measure complete in-memory training steps.
"""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_candidate(path):
    source = Path(path).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    forbidden = {"open", "eval", "exec", "compile", "__import__", "input", "breakpoint"}
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in forbidden:
                raise ValueError("forbidden candidate call: " + node.func.id)
    if any(name.split('.')[0] not in {"torch", "triton", "math"} for name in imports):
        raise ValueError("candidate imports outside torch/triton/math")
    spec = importlib.util.spec_from_file_location("cc_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "fused", None)):
        raise ValueError("candidate must expose fused(x, residual, weight, epsilon)")
    return module


def check(a, b, atol, rtol):
    import torch
    # CoreX GPU float64 subtraction returned zeros for deliberately unequal
    # inputs in the archived comparator probe. Compare CPU copies instead.
    ac, bc = a.detach().cpu().double(), b.detach().cpu().double()
    if a.shape != b.shape:
        return {"passed": False, "error": "shape mismatch"}
    delta = (ac - bc).abs()
    allowed = atol + rtol * bc.abs()
    finite = bool(torch.isfinite(ac).all().item() and torch.isfinite(bc).all().item())
    return {"passed": finite and a.dtype == b.dtype and a.shape == b.shape
            and bool((delta <= allowed).all().item()),
            "max_abs": float(delta.max().item()), "bad_elements": int((delta > allowed).sum().item()),
            "actual_dtype": str(a.dtype), "reference_dtype": str(b.dtype),
            "atol": atol, "rtol": rtol, "finite": finite,
            "comparison_device": "CPU float64", "reference_abs_max": float(bc.abs().max().item())}


def native_invalid_compatibility(a, b, atol, rtol):
    """A nonfinite native case establishes compatibility, not finite correctness."""
    import torch
    ac, bc = a.detach().cpu().double(), b.detach().cpu().double()
    finite = torch.isfinite(bc)
    patterns = (torch.equal(torch.isnan(ac), torch.isnan(bc))
                and torch.equal(torch.isposinf(ac), torch.isposinf(bc))
                and torch.equal(torch.isneginf(ac), torch.isneginf(bc)))
    close = bool(((ac[finite] - bc[finite]).abs() <= atol + rtol * bc[finite].abs()).all().item())
    return {"compatible": a.dtype == b.dtype and a.shape == b.shape and patterns and close,
            "reference_nonfinite_elements": int((~finite).sum().item()),
            "matching_nonfinite_pattern": patterns, "finite_elements_close": close,
            "scope": "native behavior compatibility only; not finite correctness"}


def event_pair(funcs, warmup=10, samples=15, launches=30):
    import torch
    for _ in range(warmup):
        for fn in funcs.values():
            fn()
    torch.cuda.synchronize()
    values = {name: [] for name in funcs}
    names = list(funcs)
    for i in range(samples):
        for name in names if i % 2 == 0 else names[::-1]:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(launches):
                funcs[name]()
            end.record()
            end.synchronize()
            value = float(start.elapsed_time(end)) * 1000 / launches
            if not math.isfinite(value) or value <= 0:
                raise ValueError("invalid event time")
            values[name].append(value)
    medians = {name: statistics.median(samples) for name, samples in values.items()}
    return {"metric": "GPU event us per call", "samples": values, "medians": medians,
            "speedup": medians["native"] / medians["candidate"],
            "warmup": warmup, "launches_per_sample": launches}


def operators(args, candidate, native_parts):
    import torch
    bda_factory, spec_factory, config_type = native_parts
    result = []
    shape_list = [(16, 256)] if args.smoke else [(16, 256), (256, 512), (7, 769), (17, 1537)]
    dtype_list = [(torch.float32, torch.float32)] if args.smoke else [
        (torch.float32, torch.float32), (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16), (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32)]
    generator = torch.Generator().manual_seed(args.seed + 100)
    bda = bda_factory(training=True, fused=False)
    for rows, hidden in shape_list:
        for dtype, residual_dtype in dtype_list:
            for epsilon in ([1e-5] if args.smoke else [1e-5, 1e-6]):
                config = config_type(num_layers=1, hidden_size=hidden, num_attention_heads=1,
                                     normalization="RMSNorm", layernorm_epsilon=epsilon)
                norm = spec_factory(normalization="RMSNorm").submodules.input_layernorm(
                    config, hidden, epsilon).cuda()
                with torch.no_grad():
                    norm.weight.copy_(torch.randn((hidden,), generator=generator).cuda())
                for scale in ([1] if args.smoke else [1, 0.001, 100]):
                    for has_residual_grad in ([True] if args.smoke else [True, False]):
                        x0 = (torch.randn((rows, hidden), generator=generator) * scale).to("cuda", dtype)
                        r0 = (torch.randn((rows, hidden), generator=generator) * scale).to("cuda", residual_dtype)
                        xr, rr, xc, rc = [v.clone().requires_grad_() for v in (x0, r0, x0, r0)]
                        wc = norm.weight.detach().clone().requires_grad_()
                        native_r = bda((xr, None), rr, 0.0)
                        native_y = norm(native_r)
                        cy, cr = candidate.fused(xc, rc, wc, epsilon)
                        dy = torch.randn(native_y.shape, generator=generator).to("cuda", native_y.dtype)
                        dr = torch.randn(native_r.shape, generator=generator).to("cuda", native_r.dtype)
                        if has_residual_grad:
                            nr_grads = torch.autograd.grad((native_y, native_r), (xr, rr, norm.weight), (dy, dr))
                            cc_grads = torch.autograd.grad((cy, cr), (xc, rc, wc), (dy, dr))
                        else:
                            nr_grads = torch.autograd.grad(native_y, (xr, rr, norm.weight), dy)
                            cc_grads = torch.autograd.grad(cy, (xc, rc, wc), dy)
                        tolerances = {torch.float32: (3e-4, 1e-3), torch.float16: (0.003, 0.003),
                                      torch.bfloat16: (0.02, 0.02)}
                        tol = tolerances[residual_dtype]
                        checks = {"normalized": check(cy, native_y, *tol),
                                  "residual": check(cr, native_r, 0, 0)}
                        checks.update({name: check(a, b, *tolerances[b.dtype]) for name, a, b in
                                       zip(("dx", "dresidual", "dweight"), cc_grads, nr_grads)})
                        unchanged = all(torch.equal(a.detach(), b) for a, b in
                                        ((xr, x0), (rr, r0), (xc, x0), (rc, r0), (wc, norm.weight)))
                        record = {"shape": [rows, hidden], "dtype": str(dtype),
                                  "residual_dtype": str(residual_dtype), "epsilon": epsilon,
                                  "scale": scale, "residual_grad_present": has_residual_grad,
                                  "checks": checks, "inputs_unchanged": unchanged,
                                  "passed": unchanged and all(x["passed"] for x in checks.values())}
                        references = (native_y, native_r, *nr_grads)
                        record["baseline_valid"] = all(bool(torch.isfinite(v).all().item()) for v in references)
                        if not record["baseline_valid"]:
                            record["native_invalid_compatibility"] = {
                                name: native_invalid_compatibility(a, b, *tolerances[b.dtype])
                                for name, a, b in zip(("normalized", "residual", "dx", "dresidual", "dweight"),
                                                      (cy, cr, *cc_grads), references)}
                            record["compatibility_passed"] = unchanged and all(
                                x["compatible"] for x in record["native_invalid_compatibility"].values())
                        result.append(record)
                        if not record["passed"]:
                            continue
                        if scale == 1 and has_residual_grad and epsilon == 1e-5:
                            def native_fn():
                                return norm(bda((xr, None), rr, 0.0))
                            def candidate_fn():
                                return candidate.fused(xc, rc, wc, epsilon)
                            record["forward"] = event_pair({"native": native_fn, "candidate": candidate_fn},
                                                           samples=3 if args.smoke else 15,
                                                           launches=5 if args.smoke else 30)
                            def native_train():
                                nr = bda((xr, None), rr, 0.0)
                                ny = norm(nr)
                                return torch.autograd.grad((ny, nr), (xr, rr, norm.weight), (dy, dr))
                            def candidate_train():
                                yy, rs = candidate.fused(xc, rc, wc, epsilon)
                                return torch.autograd.grad((yy, rs), (xc, rc, wc), (dy, dr))
                            record["forward_backward"] = event_pair(
                                {"native": native_train, "candidate": candidate_train},
                                samples=3 if args.smoke else 15, launches=5 if args.smoke else 30)
    return result


def install(model, candidate):
    counts = {"fused": 0, "fallback": 0, "observations": []}
    for layer in model.decoder.layers:
        def setup(layer):
            norm = layer.pre_mlp_layernorm
            native_norm, native_factory = norm.forward, layer.self_attn_bda
            cache = {}
            def factory(training, fused):
                original = native_factory(training, fused)
                def bda(values, residual, dropout):
                    x, bias = values
                    if (not training or fused or dropout != 0 or bias is not None
                            or not x.is_contiguous() or not residual.is_contiguous()):
                        counts["fallback"] += 1
                        return original(values, residual, dropout)
                    y, summed = candidate.fused(x, residual, norm.weight, norm.eps)
                    cache["y"] = y
                    counts["fused"] += 1
                    if len(counts["observations"]) < len(model.decoder.layers):
                        counts["observations"].append({"shape": list(x.shape), "stride": list(x.stride()),
                            "x_dtype": str(x.dtype), "residual_dtype": str(residual.dtype),
                            "weight_dtype": str(norm.weight.dtype), "y_dtype": str(y.dtype), "epsilon": norm.eps})
                    return summed
                return bda
            def norm_forward(self, hidden):
                return cache.pop("y") if "y" in cache else native_norm(hidden)
            layer.self_attn_bda = factory
            norm.forward = types.MethodType(norm_forward, norm)
        setup(layer)
    return counts


def model_test(args, candidate, parts):
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    import os
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29700 + args.seed % 100))
    dist.init_process_group("nccl", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(args.seed)
    torch.manual_seed(args.seed)
    _, spec_factory, config_type = parts
    cfg = config_type(num_layers=args.layers, hidden_size=args.hidden,
                      num_attention_heads=args.heads, normalization="RMSNorm",
                      layernorm_epsilon=1e-5, attention_dropout=0.0, hidden_dropout=0.0,
                      add_bias_linear=False, sequence_parallel=False)
    def make():
        return GPTModel(config=cfg, transformer_layer_spec=spec_factory(normalization="RMSNorm"),
                        vocab_size=args.vocab, max_sequence_length=args.sequence,
                        position_embedding_type="learned_absolute").cuda()
    native, optimized = make(), make()
    optimized.load_state_dict(native.state_dict())
    counts = install(optimized, candidate)
    opts = {"native": torch.optim.SGD(native.parameters(), lr=0.001),
            "candidate": torch.optim.SGD(optimized.parameters(), lr=0.001)}
    models = {"native": native, "candidate": optimized}
    generator = torch.Generator().manual_seed(args.seed + 101)
    batches = []
    for _ in range(4):
        shape = (args.micro_batch, args.sequence)
        tokens = torch.randint(args.vocab, shape, generator=generator).cuda()
        positions = torch.arange(args.sequence, device="cuda").unsqueeze(0).expand_as(tokens)
        mask = torch.triu(torch.ones((1, 1, args.sequence, args.sequence), dtype=torch.bool, device="cuda"), 1)
        labels = torch.randint(args.vocab, shape, generator=generator).cuda()
        batches.append((tokens, positions, mask, labels))
    def precision():
        return torch.autocast("cuda", dtype=torch.bfloat16) if args.autocast else contextlib.nullcontext()
    def step(name, batch, inspect=False):
        opts[name].zero_grad(set_to_none=True)
        with precision():
            output = models[name](*batch[:3], labels=batch[3])
            loss = output.float().mean()
        loss.backward()
        grads = [p.grad.detach().clone() if p.grad is not None else None
                 for p in models[name].parameters()] if inspect else None
        opts[name].step()
        return output.detach(), loss.detach(), grads
    checks = []
    for batch in batches[:3]:
        a, la, ga = step("native", batch, True)
        b, lb, gb = step("candidate", batch, True)
        record = {"output": check(b, a, 3e-4, 1e-3), "loss": check(lb, la, 3e-4, 1e-3)}
        record["gradients"] = [check(cb, na, 3e-4, 1e-3) if na is not None and cb is not None
                               else {"passed": na is None and cb is None} for na, cb in zip(ga, gb)]
        record["parameters"] = [check(b, a, 3e-4, 1e-3) for a, b in zip(native.parameters(), optimized.parameters())]
        record["passed"] = all(record[x]["passed"] for x in ("output", "loss")) and all(
            x["passed"] for key in ("gradients", "parameters") for x in record[key])
        checks.append(record)
    answer = {"correctness": checks, "candidate_calls": counts,
              "configuration": {x: getattr(args, x) for x in ("layers", "hidden", "heads", "sequence", "micro_batch", "vocab", "autocast")},
              "passed": all(x["passed"] for x in checks) and counts["fused"] == args.layers * 3}
    if answer["passed"]:
        for i in range(args.warmup):
            for name in models:
                step(name, batches[i % 4])
        torch.cuda.synchronize()
        raw = {name: [] for name in models}
        peaks = {name: [] for name in models}
        for sample in range(args.samples):
            for name in (list(models) if sample % 2 == 0 else list(models)[::-1]):
                for opt in opts.values():
                    opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                for j in range(args.steps):
                    step(name, batches[(sample * args.steps + j) % 4])
                torch.cuda.synchronize()
                raw[name].append((time.perf_counter() - started) * 1000 / args.steps)
                peaks[name].append(torch.cuda.max_memory_allocated() - before)
        medians = {name: statistics.median(times) for name, times in raw.items()}
        answer["training"] = {"metric": "synchronized wall ms per complete in-memory step",
            "includes": ["zero_grad", "forward", "loss", "backward", "SGD"],
            "excludes": ["data loading", "checkpoint I/O", "initial compilation"],
            "raw_ms": raw, "median_ms": medians,
            "tokens_per_second": {name: args.sequence * args.micro_batch * 1000 / ms for name, ms in medians.items()},
            "speedup": medians["native"] / medians["candidate"],
            "incremental_peak_bytes": peaks, "memory_scope": "temporary peak over two resident models; not single-model total memory",
            "warmup": args.warmup, "samples": args.samples, "steps_per_sample": args.steps}
        answer["after_timing_parameters"] = [check(b, a, 0.003, 0.003) for a, b in zip(native.parameters(), optimized.parameters())]
        answer["passed"] &= all(x["passed"] for x in answer["after_timing_parameters"])
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("operator", "model"), required=True)
    parser.add_argument("--seed", type=int, default=23001)
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
    report = {"utc": datetime.now(timezone.utc).isoformat(), "mode": args.mode,
              "candidate_sha256": digest(args.candidate), "harness_sha256": digest(__file__),
              "contract_sha256": digest(ROOT / "contract.json"), "seed": args.seed,
              "baseline": "pinned Megatron native local PyTorch backend", "status": "running"}
    try:
        import torch
        import triton
        from stage19_megatron_route_step import load_megatron, PINNED_MEGATRON
        report["environment"] = {"torch": torch.__version__, "triton": triton.__version__,
            "gpu": torch.cuda.get_device_name(0), "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "megatron_commit": PINNED_MEGATRON}
        report["native_sources_sha256"] = {path: digest(args.megatron_root / path) for path in
            ("megatron/core/transformer/torch_norm.py", "megatron/core/fusions/fused_bias_dropout.py")}
        parts = load_megatron(args.megatron_root)
        candidate = inspect_candidate(args.candidate)
        if args.mode == "operator":
            report["cases"] = operators(args, candidate, parts)
            valid = [x for x in report["cases"] if x["baseline_valid"]]
            invalid = [x for x in report["cases"] if not x["baseline_valid"]]
            report["case_summary"] = {"finite_native_cases": len(valid),
                "finite_correctness_passed": sum(x["passed"] for x in valid),
                "native_invalid_cases": len(invalid),
                "native_invalid_compatibility_passed": sum(x["compatibility_passed"] for x in invalid)}
            passed = bool(valid) and all(x["passed"] for x in valid) and all(x["compatibility_passed"] for x in invalid)
        else:
            report["model"] = model_test(args, candidate, parts)
            passed = report["model"]["passed"]
        report["status"] = "passed" if passed else "failed"
    except Exception as exc:
        report.update(status="error", error=str(exc), traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["status"], args.output, flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

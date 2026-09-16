"""Automated Residual Add RMSNorm schedule search for BI-V150.

The tool deliberately separates three small, inspectable stages:

1. normalize the operator contract into a stable Halide-style tensor IR record;
2. generate and statically filter Poly-style row/reduction schedules;
3. on a configured GPU, validate forward and backward semantics before timing
   only the forward implementations with GPU events.

It is an initial, standalone compiler-workflow prototype.  The Triton kernel is
handwritten and the Python schedule model is not a replacement for the project's
future Halide or Poly implementation.  The result JSON records that boundary.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

try:
    import triton
    import triton.language as tl
except Exception:  # CPU-plan mode must remain usable without the GPU runtime.
    triton = None
    tl = None


ROOT = Path(__file__).resolve().parent
IR_NORMALIZATION_VERSION = "residual-rmsnorm-v1"
MAX_REDUCTION_BLOCK = 8192


if triton is not None:
    @triton.jit
    def residual_add_rmsnorm_kernel(X, RESIDUAL, WEIGHT, OUTPUT, RESIDUAL_OUT,
                                    ROWS: tl.constexpr, HIDDEN: tl.constexpr,
                                    EPSILON: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        valid = columns < HIDDEN
        offset = row * HIDDEN + columns
        values = tl.load(X + offset, mask=valid, other=0.0) + tl.load(RESIDUAL + offset, mask=valid, other=0.0)
        values_fp32 = values.to(tl.float32)
        sum_squares = tl.sum(values_fp32 * values_fp32, axis=0)
        inverse_rms = tl.rsqrt(sum_squares / HIDDEN + EPSILON)
        scale = tl.load(WEIGHT + columns, mask=valid, other=0.0).to(tl.float32)
        tl.store(RESIDUAL_OUT + offset, values, mask=valid)
        tl.store(OUTPUT + offset, values_fp32 * inverse_rms * scale, mask=valid)
else:
    residual_add_rmsnorm_kernel = None


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def parse_shapes(value):
    """Parse comma-separated ROWSxHIDDEN shapes without accepting ambiguity."""
    shapes = []
    for item in value.split(","):
        pieces = item.strip().lower().split("x")
        if len(pieces) != 2:
            raise argparse.ArgumentTypeError("shapes must use ROWSxHIDDEN, separated by commas")
        try:
            rows, hidden = (int(part) for part in pieces)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("shape dimensions must be integers") from exc
        if rows <= 0 or hidden <= 0:
            raise argparse.ArgumentTypeError("shape dimensions must be positive")
        shapes.append((rows, hidden))
    if not shapes:
        raise argparse.ArgumentTypeError("at least one shape is required")
    return shapes


def next_power_of_two(value):
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def stable_json_hash(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalized_ir(rows, hidden, dtype, epsilon):
    """Return a canonical tensor-IR description and workload identity.

    The expression order is explicit because RMSNorm's reduction and the exposed
    residual output are part of the training operator contract.  It avoids
    claiming that ordinary floating-point reassociation is semantics preserving.
    """
    if rows <= 0 or hidden <= 0:
        raise ValueError("rows and hidden must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    ir = {
        "ir_kind": "halide_style_tensor_ir",
        "normalization_version": IR_NORMALIZATION_VERSION,
        "operator": "residual_add_rmsnorm",
        "domain": {"rows": rows, "hidden": hidden},
        "dtype": dtype,
        "constants": {"epsilon": epsilon, "reduction_extent": hidden},
        "buffers": [
            {"name": "x", "shape": [rows, hidden], "access": "read", "layout": "row_major_contiguous"},
            {"name": "residual", "shape": [rows, hidden], "access": "read", "layout": "row_major_contiguous"},
            {"name": "weight", "shape": [hidden], "access": "read", "layout": "contiguous"},
            {"name": "output", "shape": [rows, hidden], "access": "write", "layout": "row_major_contiguous"},
            {"name": "residual_out", "shape": [rows, hidden], "access": "write", "layout": "row_major_contiguous"},
        ],
        "statements": [
            {"id": "residual_add", "domain": ["row", "col"], "expr": "residual_out[row,col] = x[row,col] + residual[row,col]"},
            {"id": "sum_squares", "domain": ["row"], "reduction": "col", "expr": "sum(residual_out[row,col]^2)"},
            {"id": "inverse_rms", "domain": ["row"], "expr": "rsqrt(sum_squares[row] / hidden + epsilon)"},
            {"id": "scale", "domain": ["row", "col"], "expr": "output[row,col] = residual_out[row,col] * inverse_rms[row] * weight[col]"},
        ],
        "semantics": {
            "residual_out_is_observable": True,
            "reduction_precision": "fp32_accumulate",
            "reduction_reassociation": "not_assumed_legal",
            "input_mutation": "forbidden",
        },
    }
    return {"canonical_ir": ir, "workload_hash": stable_json_hash(ir)}


def candidate_is_legal(candidate, rows, hidden):
    """Check the limited affine schedule model before attempting compilation."""
    if rows <= 0 or hidden <= 0:
        return False, "non_positive_domain"
    if hidden > MAX_REDUCTION_BLOCK:
        return False, f"hidden_exceeds_supported_single_program_reduction:{MAX_REDUCTION_BLOCK}"
    if candidate.get("row_tile") != 1:
        return False, "only_one_row_per_program_is_implemented"
    block = candidate.get("reduction_tile")
    if not isinstance(block, int) or block < hidden or block & (block - 1):
        return False, "reduction_tile_must_be_power_of_two_and_cover_hidden"
    if block > MAX_REDUCTION_BLOCK:
        return False, "reduction_tile_exceeds_implementation_limit"
    if candidate.get("num_warps") not in (1, 2, 4, 8):
        return False, "unsupported_num_warps"
    if candidate.get("vector_width") != 1:
        return False, "only_contiguous_scalar_lane_mapping_is_implemented"
    return True, None


def generate_poly_candidates(rows, hidden):
    """Generate candidate schedules from the normalized affine row/reduction loops."""
    block = next_power_of_two(hidden)
    # num_warps changes the reduction tree and resource allocation selected by
    # Triton.  Each candidate keeps the required row->reduction->scale order.
    warp_options = (1, 2, 4) if block <= 1024 else (2, 4, 8)
    candidates = []
    for warps in warp_options:
        candidate = {
            "schedule_kind": "poly_row_reduction_schedule",
            "row_tile": 1,
            "reduction_tile": block,
            "reduction_order": "row_then_full_hidden_reduction_then_scale",
            "vector_width": 1,
            "num_warps": warps,
            "num_stages": 1,
        }
        legal, reason = candidate_is_legal(candidate, rows, hidden)
        candidates.append({"config": candidate, "static_legality": {"passed": legal, "reason": reason}})
    return candidates


def choose_fastest(records):
    valid = [record for record in records
             if record.get("kind") == "candidate"
             and record.get("forward_correctness", {}).get("passed") is True
             and record.get("backward_correctness", {}).get("passed") is True
             and math.isfinite(record.get("median_ms", math.inf))
             and record.get("median_ms", 0) > 0]
    return min(valid, key=lambda record: record["median_ms"]) if valid else None


def cpu_plan(args, report):
    report["scope"] = "Static IR normalization and Poly-style schedule validation only; no GPU execution."
    report["shapes"] = []
    for rows, hidden in args.shapes:
        ir = normalized_ir(rows, hidden, args.dtype, args.epsilon)
        candidates = generate_poly_candidates(rows, hidden)
        report["shapes"].append({"shape": [rows, hidden], **ir, "candidates": candidates})
    report["status"] = "cpu_plan_complete"
    return 0


def compare_tensor(torch, actual, expected, atol, rtol):
    actual64 = actual.detach().to("cpu", dtype=torch.float64)
    expected64 = expected.detach().to("cpu", dtype=torch.float64)
    finite = bool(torch.isfinite(actual64).all() and torch.isfinite(expected64).all())
    if not finite:
        return {"passed": False, "reason": "non_finite_value", "atol": atol, "rtol": rtol}
    error = (actual64 - expected64).abs()
    passed = bool((error <= atol + rtol * expected64.abs()).all())
    return {
        "passed": passed,
        "max_abs_error": float(error.max()) if error.numel() else 0.0,
        "atol": atol,
        "rtol": rtol,
    }


def event_sample(torch, launch, launches):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(launches):
        launch()
    end.record()
    end.synchronize()
    elapsed = float(start.elapsed_time(end)) / launches
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("invalid GPU event timing result")
    return elapsed


def load_gpu_dependencies():
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("GPU mode requires the preinstalled PyTorch runtime") from exc
    if triton is None or tl is None or residual_add_rmsnorm_kernel is None:
        raise RuntimeError("GPU mode requires the preinstalled Triton runtime")
    return torch, triton, tl


def make_autograd_function(torch, triton, kernel):
    class TritonResidualAddRMSNorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, residual, weight, epsilon, config):
            rows, hidden = x.shape
            output = torch.empty_like(x)
            residual_out = torch.empty_like(x)
            kernel[(rows,)](
                x, residual, weight, output, residual_out, ROWS=rows, HIDDEN=hidden,
                EPSILON=epsilon, BLOCK=config["reduction_tile"],
                num_warps=config["num_warps"], num_stages=config["num_stages"],
            )
            ctx.save_for_backward(residual_out, weight)
            ctx.epsilon = epsilon
            ctx.input_dtype = x.dtype
            ctx.weight_dtype = weight.dtype
            return output, residual_out

        @staticmethod
        def backward(ctx, grad_output, grad_residual_out):
            residual_out, weight = ctx.saved_tensors
            hidden = residual_out.shape[-1]
            values = residual_out.to(torch.float32)
            gradient = grad_output.to(torch.float32) * weight.to(torch.float32)
            inverse_rms = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + ctx.epsilon)
            dot = (gradient * values).sum(dim=-1, keepdim=True)
            grad_values = gradient * inverse_rms - values * (inverse_rms.pow(3) / hidden) * dot
            if grad_residual_out is not None:
                grad_values = grad_values + grad_residual_out.to(torch.float32)
            grad_weight = (grad_output.to(torch.float32) * values * inverse_rms).sum(dim=0)
            return (grad_values.to(ctx.input_dtype), grad_values.to(ctx.input_dtype),
                    grad_weight.to(ctx.weight_dtype), None, None)

    return TritonResidualAddRMSNorm


def reference_forward(torch, x, residual, weight, epsilon):
    residual_out = x + residual
    inverse_rms = torch.rsqrt(residual_out.to(torch.float32).square().mean(dim=-1, keepdim=True) + epsilon)
    output = (residual_out.to(torch.float32) * inverse_rms * weight.to(torch.float32)).to(x.dtype)
    return output, residual_out


def baseline_launch(torch, x, residual, weight, output, residual_out, epsilon):
    """Use preallocated observable outputs while retaining PyTorch's unfused operations."""
    torch.add(x, residual, out=residual_out)
    inverse_rms = torch.rsqrt(residual_out.to(torch.float32).square().mean(dim=-1, keepdim=True) + epsilon)
    output.copy_((residual_out.to(torch.float32) * inverse_rms * weight.to(torch.float32)).to(output.dtype))


def candidate_launch(kernel, triton, x, residual, weight, output, residual_out, config, epsilon):
    rows, hidden = x.shape
    kernel[(rows,)](
        x, residual, weight, output, residual_out, ROWS=rows, HIDDEN=hidden,
        EPSILON=epsilon, BLOCK=config["reduction_tile"],
        num_warps=config["num_warps"], num_stages=config["num_stages"],
    )


def validate_candidate(torch, triton, kernel, autograd_function, x_cpu, residual_cpu, weight_cpu,
                       device, config, epsilon, atol, rtol):
    """Validate both returned tensors and all three first-order gradients."""
    x_ref = x_cpu.to(device).detach().requires_grad_(True)
    residual_ref = residual_cpu.to(device).detach().requires_grad_(True)
    weight_ref = weight_cpu.to(device).detach().requires_grad_(True)
    x_candidate = x_cpu.to(device).detach().requires_grad_(True)
    residual_candidate = residual_cpu.to(device).detach().requires_grad_(True)
    weight_candidate = weight_cpu.to(device).detach().requires_grad_(True)
    seed_generator = torch.Generator(device="cpu").manual_seed(9173)
    grad_output = torch.randn(x_cpu.shape, generator=seed_generator, dtype=x_cpu.dtype).to(device)
    grad_residual_out = torch.randn(x_cpu.shape, generator=seed_generator, dtype=x_cpu.dtype).to(device)

    reference_output, reference_residual = reference_forward(torch, x_ref, residual_ref, weight_ref, epsilon)
    reference_grads = torch.autograd.grad(
        (reference_output, reference_residual), (x_ref, residual_ref, weight_ref),
        grad_outputs=(grad_output, grad_residual_out),
    )
    candidate_output, candidate_residual = autograd_function.apply(
        x_candidate, residual_candidate, weight_candidate, epsilon, config)
    candidate_grads = torch.autograd.grad(
        (candidate_output, candidate_residual), (x_candidate, residual_candidate, weight_candidate),
        grad_outputs=(grad_output, grad_residual_out),
    )
    torch.cuda.synchronize()
    output_check = compare_tensor(torch, candidate_output, reference_output, atol, rtol)
    residual_check = compare_tensor(torch, candidate_residual, reference_residual, atol, rtol)
    grad_checks = {
        name: compare_tensor(torch, actual, expected, atol, rtol)
        for name, actual, expected in zip(
            ("x", "residual", "weight"), candidate_grads, reference_grads)
    }
    forward_passed = output_check["passed"] and residual_check["passed"]
    backward_passed = all(check["passed"] for check in grad_checks.values())
    input_unchanged = bool(torch.equal(x_candidate.detach().cpu(), x_cpu)
                           and torch.equal(residual_candidate.detach().cpu(), residual_cpu)
                           and torch.equal(weight_candidate.detach().cpu(), weight_cpu))
    return {
        "forward_correctness": {
            "passed": forward_passed and input_unchanged,
            "output": output_check,
            "residual_out": residual_check,
            "inputs_unchanged": input_unchanged,
        },
        "backward_correctness": {
            "passed": backward_passed,
            "gradients": grad_checks,
            "implementation": "analytical PyTorch backward for validation; not timed or selected as an optimized backward kernel",
        },
    }


def gpu_autotune(args, report):
    torch, triton, tl = load_gpu_dependencies()
    report["active_stage"] = "probe_gpu"
    if not torch.cuda.is_available():
        report["status"] = "gpu_unavailable"
        return 2
    if args.device < 0 or args.device >= torch.cuda.device_count():
        report["status"] = "invalid_device"
        report["available_device_count"] = torch.cuda.device_count()
        return 2
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    properties = torch.cuda.get_device_properties(args.device)
    try:
        triton_target = str(triton.runtime.driver.active.get_current_target())
    except Exception as exc:
        triton_target = None
        report["triton_target_probe_error"] = str(exc)
    report["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "device_index": args.device,
        "device_name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "triton_target": triton_target,
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
    }
    report["timing_protocol"] = {
        "scope": "forward only, warm-buffer GPU-event intervals",
        "warmup_launches": args.warmup,
        "samples": args.samples,
        "launches_per_sample": args.launches,
        "compilation_and_correctness_timed": False,
        "candidate_order": "rotated each sample",
        "backward_timing": "not included; this prototype validates gradients with an analytical PyTorch fallback",
    }
    kernel = residual_add_rmsnorm_kernel
    autograd_function = make_autograd_function(torch, triton, kernel)
    candidate_rejections = 0
    incomplete_workloads = 0
    report["workloads"] = []
    for rows, hidden in args.shapes:
        report["active_stage"] = f"prepare:{rows}x{hidden}"
        ir = normalized_ir(rows, hidden, args.dtype, args.epsilon)
        planned = generate_poly_candidates(rows, hidden)
        workload = {"shape": [rows, hidden], **ir, "candidate_records": []}
        report["workloads"].append(workload)
        dtype = getattr(torch, args.dtype)
        generator = torch.Generator(device="cpu").manual_seed(args.seed + rows * 100003 + hidden)
        x_cpu = torch.randn((rows, hidden), generator=generator, dtype=dtype)
        residual_cpu = torch.randn((rows, hidden), generator=generator, dtype=dtype)
        weight_cpu = torch.randn((hidden,), generator=generator, dtype=dtype)
        x = x_cpu.to(device)
        residual = residual_cpu.to(device)
        weight = weight_cpu.to(device)

        baseline_output = torch.empty_like(x)
        baseline_residual_out = torch.empty_like(x)
        report["active_stage"] = f"validate_baseline:{rows}x{hidden}"
        baseline_launch(torch, x, residual, weight, baseline_output, baseline_residual_out, args.epsilon)
        expected_output, expected_residual = reference_forward(torch, x, residual, weight, args.epsilon)
        torch.cuda.synchronize()
        baseline_forward = {
            "passed": compare_tensor(torch, baseline_output, expected_output, args.atol, args.rtol)["passed"]
            and compare_tensor(torch, baseline_residual_out, expected_residual, args.atol, args.rtol)["passed"],
        }
        baseline_record = {
            "kind": "baseline",
            "implementation": "unfused PyTorch residual add, fp32 RMS reduction, scale",
            "forward_correctness": baseline_forward,
            "backward_correctness": {"passed": True, "implementation": "PyTorch autograd reference"},
            "status": "correct" if baseline_forward["passed"] else "numerical_failure",
        }
        workload["candidate_records"].append(baseline_record)
        if not baseline_forward["passed"]:
            raise RuntimeError("PyTorch baseline failed its own reference contract")

        runnable = [(baseline_record, lambda: baseline_launch(
            torch, x, residual, weight, baseline_output, baseline_residual_out, args.epsilon))]
        for planned_candidate in planned:
            config = planned_candidate["config"]
            record = {
                "kind": "candidate",
                "implementation": "Triton fused forward kernel generated from the row/reduction schedule",
                "config": config,
                "static_legality": planned_candidate["static_legality"],
                "status": "static_rejected" if not planned_candidate["static_legality"]["passed"] else "validating",
            }
            workload["candidate_records"].append(record)
            if record["status"] == "static_rejected":
                candidate_rejections += 1
                continue
            report["active_stage"] = f"validate_candidate:{rows}x{hidden}:warps{config['num_warps']}"
            try:
                record.update(validate_candidate(
                    torch, triton, kernel, autograd_function, x_cpu, residual_cpu, weight_cpu,
                    device, config, args.epsilon, args.atol, args.rtol))
            except Exception as exc:
                record.update(status="execution_error", error_type=type(exc).__name__, error=str(exc))
                candidate_rejections += 1
                continue
            if not (record["forward_correctness"]["passed"] and record["backward_correctness"]["passed"]):
                record["status"] = "numerical_failure"
                candidate_rejections += 1
                continue
            record["status"] = "correct"
            candidate_output = torch.empty_like(x)
            candidate_residual_out = torch.empty_like(x)
            runnable.append((record, lambda output=candidate_output, residual_out=candidate_residual_out,
                             config=config: candidate_launch(
                kernel, triton, x, residual, weight, output, residual_out, config, args.epsilon)))

        for _, launch in runnable:
            for _ in range(args.warmup):
                launch()
        torch.cuda.synchronize()
        for record, _ in runnable:
            record["samples_ms"] = []
        for sample in range(args.samples):
            offset = sample % len(runnable)
            for record, launch in runnable[offset:] + runnable[:offset]:
                record["samples_ms"].append(event_sample(torch, launch, args.launches))
        for record, _ in runnable:
            samples = record["samples_ms"]
            record.update(status="measured", median_ms=statistics.median(samples),
                          min_ms=min(samples), max_ms=max(samples))
        baseline_ms = baseline_record["median_ms"]
        for record, _ in runnable:
            record["speedup_vs_unfused_pytorch_forward"] = baseline_ms / record["median_ms"]
        winner = choose_fastest(workload["candidate_records"])
        workload["selection"] = {
            "best_observed_config": winner["config"] if winner else None,
            "speedup_vs_unfused_pytorch_forward": winner["speedup_vs_unfused_pytorch_forward"] if winner else None,
            "dispatch_scope": "this exact normalized workload hash only",
            "fallback": "unfused PyTorch reference when no candidate is both correct and timed",
        }
        if winner is None:
            incomplete_workloads += 1
    report["active_stage"] = "completed"
    report["candidate_rejections"] = candidate_rejections
    report["incomplete_workloads"] = incomplete_workloads
    report["status"] = "gpu_autotune_incomplete" if incomplete_workloads else "gpu_autotune_passed"
    return 1 if incomplete_workloads else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cpu-plan", "gpu"), default="gpu")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shapes", type=parse_shapes, default=parse_shapes("64x768,64x1024,64x4096"))
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--samples", type=positive_int, default=9)
    parser.add_argument("--launches", type=positive_int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("--device must be non-negative")
    if args.epsilon <= 0 or args.atol < 0 or args.rtol < 0:
        parser.error("epsilon must be positive; tolerances must be non-negative")
    stamp = datetime.now(timezone.utc)
    output = args.output or ROOT / "results" / ("residual-rmsnorm-" + stamp.strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    if output.exists():
        parser.error("output already exists; choose a new path to preserve prior evidence")
    report = {
        "schema_version": 1,
        "created_at_utc": stamp.isoformat(),
        "operator": "residual_add_rmsnorm",
        "mode": args.mode,
        "implementation_boundary": {
            "halide_ir": "stable Halide-style normalized contract record; not integrated with the project's Halide source tree yet",
            "poly_module": "explicit affine row/reduction schedule generator and legality filter; not a full external Poly runtime",
            "automated_flow": "generate -> filter -> compile -> forward/backward validate -> event-time -> select/fallback",
            "backend": "handwritten Triton forward kernel for the preinstalled BI-V150-compatible runtime",
        },
        "source_sha256": {"residual_rmsnorm_autotune.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
    }
    started = time.perf_counter()
    try:
        code = cpu_plan(args, report) if args.mode == "cpu-plan" else gpu_autotune(args, report)
    except Exception as exc:
        report.update(status="run_failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        code = 1
    report["wall_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print("status:", report["status"])
    print("report:", output.resolve())
    for workload in report.get("workloads", []):
        selection = workload.get("selection", {})
        print("shape:", workload["shape"], "best:", selection.get("best_observed_config"),
              "speedup:", selection.get("speedup_vs_unfused_pytorch_forward"))
    return code


if __name__ == "__main__":
    raise SystemExit(main())

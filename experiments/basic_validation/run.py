"""CPU workflow check and optional CUDA-compatible GPU/Triton experiments.

Usage: python experiments/basic_validation/run.py --mode cpu
No framework is imported by the CPU mode. GPU failures never fall back to CPU.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ADD_CONFIGS = [{"block": 128, "warps": 4}, {"block": 256, "warps": 4},
               {"block": 512, "warps": 8}]
MATMUL_CONFIGS = [{"bm": 16, "bn": 16, "bk": 16, "warps": 4},
                  {"bm": 32, "bn": 32, "bk": 32, "warps": 4},
                  {"bm": 64, "bn": 32, "bk": 32, "warps": 4}]


def compare_values(actual, expected, atol=1e-12, rtol=1e-12):
    """Every element must be finite and within the fixed, per-element tolerance."""
    actual, expected = list(actual), list(expected)
    if len(actual) != len(expected):
        return {"passed": False, "reason": "length_mismatch"}
    max_error = 0.0
    passed = True
    for value, reference in zip(actual, expected):
        if not math.isfinite(value) or not math.isfinite(reference):
            return {"passed": False, "reason": "non_finite_value"}
        error = abs(value - reference)
        max_error = max(max_error, error)
        passed = passed and error <= atol + rtol * abs(reference)
    return {"passed": passed, "max_abs_error": max_error, "atol": atol, "rtol": rtol}


def choose_fastest(records):
    valid = [r for r in records if r.get("kind") == "candidate"
             and r.get("correctness", {}).get("passed") is True
             and math.isfinite(r.get("median_ms", math.inf))
             and r.get("median_ms", 0) > 0]
    return min(valid, key=lambda r: r["median_ms"]) if valid else None


def reference_matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def chunked_add(a, b, block):
    if block <= 0:
        raise ValueError("block must be positive")
    if len(a) != len(b):
        raise ValueError("input lengths must match")
    output = [math.nan] * len(a)
    for begin in range(0, len(a), block):
        for i in range(begin, min(begin + block, len(a))):
            output[i] = a[i] + b[i]
    return output


def tiled_matmul(a, b, tile):
    if tile <= 0:
        raise ValueError("tile must be positive")
    m, k, n = len(a), len(b), len(b[0])
    output = [[0.0] * n for _ in range(m)]
    for ii in range(0, m, tile):
        for jj in range(0, n, tile):
            for kk in range(0, k, tile):
                for i in range(ii, min(ii + tile, m)):
                    for j in range(jj, min(jj + tile, n)):
                        for p in range(kk, min(kk + tile, k)):
                            output[i][j] += a[i][p] * b[p][j]
    return output


def flatten(matrix):
    return [value for row in matrix for value in row]


def cpu_check(args, report):
    rng = random.Random(args.seed)
    records = report["records"]
    if args.operator in ("add", "all"):
        for length in (1, 17, 129):
            a = [rng.uniform(-1, 1) for _ in range(length)]
            b = [rng.uniform(-1, 1) for _ in range(length)]
            expected = [x + y for x, y in zip(a, b)]
            for block in (4, 16, 32):
                actual = chunked_add(a, b, block)
                records.append({"operator": "add", "shape": [length], "block": block,
                                "correctness": compare_values(actual, expected)})
    if args.operator in ("matmul", "all"):
        for m, n, k in ((1, 1, 1), (5, 7, 3), (16, 12, 8)):
            a = [[rng.uniform(-1, 1) for _ in range(k)] for _ in range(m)]
            b = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(k)]
            expected = flatten(reference_matmul(a, b))
            for tile in (2, 4, 8):
                records.append({"operator": "matmul", "shape_mnk": [m, n, k], "tile": tile,
                                "correctness": compare_values(
                                    flatten(tiled_matmul(a, b, tile)), expected)})
    report["scope"] = "Python CPU functional self-check; no GPU timing, Halide, or Poly"
    report["status"] = ("cpu_check_passed" if all(r["correctness"]["passed"] for r in records)
                        else "cpu_check_failed")
    return 0 if report["status"] == "cpu_check_passed" else 1


def probe_environment(device):
    result = {"python": sys.version, "platform": platform.platform(),
              "corex_root": (os.environ.get("COREX_ROOT")
                             or os.environ.get("ILUVATAR_SOFTWARE_ROOT")),
              "requested_device": device}
    for package in ("torch", "triton"):
        try:
            module = importlib.import_module(package)
            result[package] = {"imported": True, "version": getattr(module, "__version__", None)}
        except Exception as exc:
            result[package] = {"imported": False, "error": str(exc)}
    result["gpu_available"] = False
    if result["torch"]["imported"]:
        import torch
        try:
            result["framework_cuda_version"] = getattr(torch.version, "cuda", None)
            result["device_count"] = torch.cuda.device_count()
            if torch.cuda.is_available() and 0 <= device < result["device_count"]:
                props = torch.cuda.get_device_properties(device)
                result.update(gpu_available=True, device_name=props.name,
                              total_memory_bytes=props.total_memory)
        except Exception as exc:
            result["gpu_probe_error"] = str(exc)
    result["ready_for_gpu_attempt"] = bool(result["gpu_available"] and result["triton"]["imported"])
    return result


def tensor_check(torch, output, expected_cpu, atol, rtol):
    actual = output.detach().cpu().to(torch.float64)
    expected = expected_cpu.to(torch.float64)
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    error = (actual - expected).abs()
    passed = finite and bool((error <= atol + rtol * expected.abs()).all())
    return {"passed": passed, "atol": atol, "rtol": rtol,
            "max_abs_error": float(error.max()) if finite else None}


def event_sample(torch, launch, launches):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(launches):
        launch()
    end.record()
    end.synchronize()
    milliseconds = float(start.elapsed_time(end)) / launches
    if not math.isfinite(milliseconds) or milliseconds <= 0:
        raise RuntimeError("Invalid GPU event timing; no valid performance measurement")
    return milliseconds


def gpu_experiments(args, report):
    report["active_stage"] = "probe_environment"
    environment = report["environment"] = probe_environment(args.device)
    if not environment["ready_for_gpu_attempt"]:
        report["status"] = "gpu_unavailable"
        return 2
    import torch
    import triton
    import triton_kernels as kernels

    report["active_stage"] = "set_device"
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    # CUDA-compatible API is also used by the vendor's adapted PyTorch.
    # This does NOT assert the device is NVIDIA or that a vendor backend is installed.
    report["active_stage"] = "create_cpu_generator"
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    try:
        report["triton_target"] = str(triton.runtime.driver.active.get_current_target())
    except Exception as exc:
        report["triton_target_probe_error"] = str(exc)
    report["timing_protocol"] = {
        "method": "GPU stream events, repeated launches, median of batch means",
        "warmup_launches": args.warmup, "samples": args.samples,
        "launches_per_sample": args.launches, "preallocated_output": True,
        "compilation_and_validation_timed": False, "l2_flushed": False,
        "cuda_graph": False, "order": "rotated per round",
        "scope": "warm-buffer event interval; may include launch gaps; not isolated kernel profiling",
    }
    report["scope"] = "Fixed handwritten Triton candidate sweep; no Halide/Poly or training integration"
    jobs = []
    if args.operator in ("add", "all"):
        jobs += [("add", (size,)) for size in (1, 4099, 1048576)]
    if args.operator in ("matmul", "all"):
        jobs += [("matmul", shape) for shape in ((64, 64, 64), (127, 65, 33), (256, 256, 256))]
    report["selections"] = []
    any_failure = False
    for operator, shape in jobs:
        report["active_stage"] = f"prepare_cpu_inputs:{operator}:{shape}"
        dtype = torch.float32 if operator == "add" else torch.float16
        if operator == "add":
            left_shape, right_shape, out_shape = shape, shape, shape
            configs, atol, rtol = ADD_CONFIGS, 1e-6, 1e-6
        else:
            m, n, k = shape
            left_shape, right_shape, out_shape = (m, k), (k, n), (m, n)
            configs, atol, rtol = MATMUL_CONFIGS, 1e-2, 1e-2
        left_cpu = torch.randn(left_shape, generator=generator, dtype=torch.float32).to(dtype)
        right_cpu = torch.randn(right_shape, generator=generator, dtype=torch.float32).to(dtype)
        report["active_stage"] = f"copy_inputs_to_device:{operator}:{shape}"
        left, right = left_cpu.to(device), right_cpu.to(device)

        def oracle(a_cpu, b_cpu):
            a64, b64 = a_cpu.to(torch.float64), b_cpu.to(torch.float64)
            return (a64 + b64 if operator == "add" else a64 @ b64).to(dtype)

        report["active_stage"] = f"compute_cpu_reference:{operator}:{shape}"
        normal_expected = oracle(left_cpu, right_cpu)
        zero_expected = torch.zeros(out_shape, dtype=dtype)
        case_records, runnable = [], []
        routes = [("baseline", {})] + [("candidate", cfg) for cfg in configs]
        for kind, config in routes:
            report["active_stage"] = f"validate_route:{operator}:{shape}:{kind}:{config}"
            record = {"operator": operator, "shape": list(shape), "kind": kind,
                      "shape_axes": "N" if operator == "add" else "M,N,K",
                      "dtype": str(dtype), "config": config, "status": "validating"}
            report["records"].append(record)
            case_records.append(record)
            out = torch.empty(out_shape, dtype=dtype, device=device)
            if kind == "baseline":
                def launch(out=out):
                    if operator == "add":
                        torch.add(left, right, out=out)
                    else:
                        torch.mm(left, right, out=out)
            elif operator == "add":
                def launch(out=out, config=config):
                    kernels.vector_add[(triton.cdiv(shape[0], config["block"]),)](
                        left, right, out, shape[0], BLOCK=config["block"], num_warps=config["warps"])
            else:
                def launch(out=out, config=config):
                    kernels.matrix_multiply[(triton.cdiv(m, config["bm"]), triton.cdiv(n, config["bn"]))](
                        left, right, out, m, n, k, BM=config["bm"], BN=config["bn"],
                        BK=config["bk"], num_warps=config["warps"], num_stages=1)
            # Reset inputs/output for each route and distribution; expose unwritten tails.
            try:
                checks = []
                for distribution in ("normal", "zeros"):
                    if distribution == "normal":
                        left.copy_(left_cpu)
                        right.copy_(right_cpu)
                        expected = normal_expected
                    else:
                        left.zero_()
                        right.zero_()
                        expected = zero_expected
                    out.fill_(float("nan"))
                    launch()  # JIT compilation, allocation and copies are outside timing.
                    torch.cuda.synchronize()
                    check = tensor_check(torch, out, expected, atol, rtol)
                    check["distribution"] = distribution
                    # Inputs are read-only according to the workload contract.
                    check["inputs_unchanged"] = bool(
                        torch.equal(left.cpu(), left_cpu if distribution == "normal" else torch.zeros_like(left_cpu))
                        and torch.equal(right.cpu(), right_cpu if distribution == "normal" else torch.zeros_like(right_cpu)))
                    check["passed"] = check["passed"] and check["inputs_unchanged"]
                    checks.append(check)
                record["correctness"] = {"passed": all(c["passed"] for c in checks), "checks": checks}
                if not record["correctness"]["passed"]:
                    record["status"] = "numerical_failure"
                    any_failure = True
                    continue
                record["status"] = "correct"
                runnable.append((record, launch))
            except Exception as exc:
                record.update(status="execution_error", error_type=type(exc).__name__, error=str(exc))
                raise  # A device error may poison the context. Do not continue or silently fall back.

        if not case_records[0].get("correctness", {}).get("passed"):
            raise RuntimeError("PyTorch baseline failed the independent CPU reference")
        left.copy_(left_cpu)
        right.copy_(right_cpu)
        for record, launch in runnable:
            for _ in range(args.warmup):
                launch()
            record["samples_ms"] = []
        torch.cuda.synchronize()
        for sample in range(args.samples):
            offset = sample % len(runnable)
            for record, launch in runnable[offset:] + runnable[:offset]:
                record["samples_ms"].append(event_sample(torch, launch, args.launches))
        for record, _ in runnable:
            samples = record["samples_ms"]
            record.update(status="measured", median_ms=statistics.median(samples),
                          min_ms=min(samples), max_ms=max(samples))
        baseline_ms = case_records[0]["median_ms"]
        for record, _ in runnable:
            record["speedup_vs_pytorch"] = baseline_ms / record["median_ms"]
        winner = choose_fastest(case_records)
        report["selections"].append({
            "operator": operator, "shape": list(shape),
            "best_observed_config": winner["config"] if winner else None,
            "speedup_vs_pytorch": winner["speedup_vs_pytorch"] if winner else None,
            "note": "Selection on the measured inputs only; not a validated general-purpose dispatcher",
        })
    report["active_stage"] = "completed"
    report["status"] = "gpu_completed_with_failures" if any_failure else "gpu_check_passed"
    return 1 if any_failure else 0


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "cpu", "gpu"), default="probe")
    parser.add_argument("--operator", choices=("add", "matmul", "all"), default="all")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=positive, default=5)
    parser.add_argument("--samples", type=positive, default=7)
    parser.add_argument("--launches", type=positive, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("--device must be non-negative")
    stamp = datetime.now(timezone.utc)
    output = args.output or ROOT / "results" / (stamp.strftime("%Y%m%dT%H%M%S%fZ") + "-" + args.mode + ".json")
    if output.exists():
        parser.error("output already exists; choose a new filename to preserve the previous run")
    report = {"schema_version": 1, "created_at_utc": stamp.isoformat(), "mode": args.mode,
              "seed": args.seed, "operator": args.operator, "records": [],
              "implementation": "fixed candidates; no agent calls, Halide passes, or Poly integration",
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ("run.py", "triton_kernels.py")}}
    started = time.perf_counter()
    try:
        if args.mode == "cpu":
            report["environment"] = {"python": sys.version, "platform": platform.platform()}
            code = cpu_check(args, report)
        elif args.mode == "probe":
            report["environment"] = probe_environment(args.device)
            report["status"] = "probe_complete"
            code = 0
        else:
            code = gpu_experiments(args, report)
    except Exception as exc:
        report.update(
            status="run_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        code = 1
    report["wall_seconds"] = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print("status:", report["status"])
    print("report:", output.resolve())
    for selection in report.get("selections", []):
        print(selection["operator"], selection["shape"], "best observed:",
              selection["best_observed_config"], "speedup:", selection["speedup_vs_pytorch"])
    return code


if __name__ == "__main__":
    raise SystemExit(main())

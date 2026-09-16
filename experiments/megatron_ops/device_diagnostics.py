"""Isolate PyTorch and Triton device operations in fresh subprocesses.

Every case runs with CUDA_LAUNCH_BLOCKING=1 in a separate process. One failed
operation therefore cannot hide the phase that failed or poison later cases.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


CASES = (
    "metadata",
    "allocate",
    "cpu_to_device",
    "device_to_cpu",
    "torch_add",
    "event_timing",
    "triton_target",
    "triton_add_warp1",
    "triton_add_warp4",
)
MARKER = "GPGPU_DIAGNOSTIC_JSON="


def execute_case(name, device_index):
    context = {"case": name, "active_stage": "import_torch"}
    try:
        import torch

        context.update(
            torch_version=getattr(torch, "__version__", None),
            torch_file=getattr(torch, "__file__", None),
        )
        context["active_stage"] = "set_device"
        torch.cuda.set_device(device_index)
        device = torch.device("cuda", device_index)

        if name == "metadata":
            context["active_stage"] = "read_metadata"
            properties = torch.cuda.get_device_properties(device_index)
            details = {
                "device_count": int(torch.cuda.device_count()),
                "device_name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "capability": list(torch.cuda.get_device_capability(device_index)),
            }
        elif name == "allocate":
            context["active_stage"] = "allocate_device_tensor"
            tensor = torch.empty(257, dtype=torch.float32, device=device)
            context["active_stage"] = "fill_device_tensor"
            tensor.fill_(1.0)
            context["active_stage"] = "synchronize_after_fill"
            torch.cuda.synchronize()
            details = {"numel": tensor.numel(), "dtype": str(tensor.dtype)}
        elif name == "cpu_to_device":
            generator = torch.Generator(device="cpu").manual_seed(0)
            copied_sizes = []
            for size in (1, 4099):
                context["active_stage"] = f"create_cpu_tensor:{size}"
                cpu_tensor = torch.randn(size, generator=generator, dtype=torch.float32)
                context["active_stage"] = f"copy_cpu_tensor_to_device:{size}"
                device_tensor = cpu_tensor.to(device)
                context["active_stage"] = f"synchronize_after_h2d:{size}"
                torch.cuda.synchronize()
                copied_sizes.append(device_tensor.numel())
            details = {"copied_sizes": copied_sizes, "dtype": str(device_tensor.dtype)}
        elif name == "device_to_cpu":
            context["active_stage"] = "create_device_tensor"
            device_tensor = torch.arange(17, dtype=torch.float32, device=device)
            context["active_stage"] = "copy_device_tensor_to_cpu"
            cpu_tensor = device_tensor.cpu()
            details = {"values": cpu_tensor.tolist()}
        elif name == "torch_add":
            context["active_stage"] = "create_add_inputs"
            left = torch.ones(257, dtype=torch.float32, device=device)
            right = torch.full((257,), 2.0, dtype=torch.float32, device=device)
            context["active_stage"] = "launch_torch_add"
            output = torch.add(left, right)
            context["active_stage"] = "synchronize_after_torch_add"
            torch.cuda.synchronize()
            context["active_stage"] = "check_torch_add"
            actual = output.cpu()
            details = {"correct": bool(torch.equal(actual, torch.full_like(actual, 3.0)))}
            if not details["correct"]:
                raise RuntimeError("torch.add returned an incorrect value")
        elif name == "event_timing":
            context["active_stage"] = "create_event_inputs"
            left = torch.ones(4096, dtype=torch.float32, device=device)
            right = torch.ones(4096, dtype=torch.float32, device=device)
            context["active_stage"] = "create_events"
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            context["active_stage"] = "record_events"
            start.record()
            output = torch.add(left, right)
            end.record()
            context["active_stage"] = "synchronize_event"
            end.synchronize()
            context["active_stage"] = "read_elapsed_time"
            elapsed_ms = float(start.elapsed_time(end))
            if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
                raise RuntimeError(f"invalid event timing result: {elapsed_ms}")
            details = {"elapsed_ms": elapsed_ms, "output_numel": output.numel()}
        elif name == "triton_target":
            context["active_stage"] = "import_triton"
            import triton

            context["active_stage"] = "read_triton_target"
            details = {
                "triton_version": getattr(triton, "__version__", None),
                "triton_file": getattr(triton, "__file__", None),
                "target": str(triton.runtime.driver.active.get_current_target()),
            }
        elif name in ("triton_add_warp1", "triton_add_warp4"):
            context["active_stage"] = "import_triton"
            import triton

            basic_dir = Path(__file__).resolve().parents[1] / "basic_validation"
            sys.path.insert(0, str(basic_dir))
            context["active_stage"] = "import_project_triton_kernel"
            import triton_kernels

            warps = 1 if name.endswith("warp1") else 4
            length = 257
            block = 256
            context["active_stage"] = "create_triton_inputs_on_device"
            left = torch.ones(length, dtype=torch.float32, device=device)
            right = torch.full((length,), 2.0, dtype=torch.float32, device=device)
            output = torch.empty_like(left)
            context["active_stage"] = "compile_and_launch_triton_add"
            triton_kernels.vector_add[(triton.cdiv(length, block),)](
                left,
                right,
                output,
                length,
                BLOCK=block,
                num_warps=warps,
            )
            context["active_stage"] = "synchronize_after_triton_add"
            torch.cuda.synchronize()
            context["active_stage"] = "check_triton_add"
            actual = output.cpu()
            details = {
                "warps": warps,
                "block": block,
                "correct": bool(torch.equal(actual, torch.full_like(actual, 3.0))),
            }
            if not details["correct"]:
                raise RuntimeError("Triton vector add returned an incorrect value")
        else:
            raise ValueError(f"unknown diagnostic case: {name}")

        return {**context, "passed": True, "completed_stage": context["active_stage"], "details": details}
    except Exception as exc:
        return {
            **context,
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def run_child(case, device):
    command = [sys.executable, str(Path(__file__).resolve()), "--case", case, "--device", str(device)]
    environment = os.environ.copy()
    environment["CUDA_LAUNCH_BLOCKING"] = "1"
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "case": case,
            "passed": False,
            "active_stage": "child_process_timeout",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "child_stdout": exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            "child_stderr": exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
        }
    except OSError as exc:
        return {
            "case": case,
            "passed": False,
            "active_stage": "start_child_process",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "child_stdout": "",
            "child_stderr": "",
        }
    parsed = None
    for line in completed.stdout.splitlines():
        if line.startswith(MARKER):
            try:
                candidate = json.loads(line[len(MARKER) :])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                parsed = candidate
    if parsed is None:
        parsed = {
            "case": case,
            "passed": False,
            "active_stage": "child_process",
            "error_type": "MissingDiagnosticResult",
            "error": "child process did not emit a diagnostic JSON record",
        }
    if completed.returncode != 0:
        parsed["passed"] = False
        parsed.setdefault("error_type", "ChildProcessError")
        parsed.setdefault("error", f"child process exited with code {completed.returncode}")
    parsed["child_returncode"] = completed.returncode
    parsed["child_stdout"] = completed.stdout
    parsed["child_stderr"] = completed.stderr
    return parsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", choices=CASES)
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("--device must be non-negative")

    if args.case:
        print(MARKER + json.dumps(execute_case(args.case, args.device), ensure_ascii=False))
        return 0
    if args.output is None:
        parser.error("--output is required unless --case is used")
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite evidence")

    records = [run_child(case, args.device) for case in CASES]
    failed = [record["case"] for record in records if not record.get("passed")]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "method": "one fresh subprocess per case with CUDA_LAUNCH_BLOCKING=1",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": records,
        "failed_cases": failed,
        "status": "diagnostics_passed" if not failed else "diagnostics_complete_with_failures",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print("status:", report["status"])
    print("failed cases:", ", ".join(failed) if failed else "none")
    print("report:", args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

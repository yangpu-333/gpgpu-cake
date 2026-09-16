"""Isolate PyTorch stream initialization from direct CoreX runtime allocation.

Only child-process library search paths change. No packages or devices are modified.
"""

import argparse
import ctypes
import json
import os
import stat
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from runtime_diagnostics import (HERE, MARKER, STAGE_MARKER, corex_roots, inventory,
                                 loaded_libraries, parse_smoke, run_command,
                                 selected_environment)

DEFAULT_LIBRARY = "/usr/local/corex-4.4.0/lib64/libcudart.so.10.2.89"
CASES = ("torch_current_stream", "raw_priority_range", "raw_malloc_copy")


def device_info(path):
    path = Path(path)
    info = {"path": str(path)}
    try:
        details = path.stat()
        info.update(exists=True, character_device=stat.S_ISCHR(details.st_mode),
                    mode=oct(details.st_mode), uid=details.st_uid, gid=details.st_gid,
                    major=os.major(details.st_rdev), minor=os.minor(details.st_rdev))
    except OSError as exc:
        info.update(exists=False, error=str(exc))
    return info


def build_variants(environment, directory=Path("/usr/local/iluvatar/lib64")):
    baseline = environment.copy()
    baseline.update(CUDA_LAUNCH_BLOCKING="1", TORCH_SHOW_CPP_STACKTRACES="1")
    variants = [("baseline", baseline)]
    candidates, seen = [], set()
    directory = Path(directory)
    for pattern in ("libcuda.so*", "libixthunk.so*", "libixml.so*"):
        for path in sorted(directory.glob(pattern)):
            try:
                resolved = path.resolve(strict=True)
                if str(resolved) in seen or not resolved.is_file():
                    continue
                seen.add(str(resolved))
                details = resolved.stat()
                candidates.append({"path": str(path), "resolved": str(resolved),
                                   "size": details.st_size, "mode": oct(details.st_mode),
                                   "mtime_ns": details.st_mtime_ns})
            except OSError as exc:
                candidates.append({"path": str(path), "error": str(exc)})
    if any("error" not in item and Path(item["path"]).name.startswith(
            ("libcuda.so", "libixthunk.so")) for item in candidates):
        adjusted = baseline.copy()
        paths = [str(directory)] + baseline.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        unique, seen_paths = [], set()
        for path in filter(None, paths):
            key = str(Path(path).resolve())
            if key not in seen_paths:
                unique.append(path)
                seen_paths.add(key)
        adjusted["LD_LIBRARY_PATH"] = os.pathsep.join(unique)
        variants.append(("iluvatar_path_first", adjusted))
    return variants, {"directory": str(directory), "libraries": candidates}


class RuntimeAPI:
    """CUDA 10.2 C signatures; raw cases deliberately do not import torch."""

    def __init__(self, library, result, stage):
        self.result, self.stage = result, stage
        self.library = ctypes.CDLL(library)
        pointer = ctypes.POINTER
        signatures = {
            "cudaSetDevice": [ctypes.c_int],
            "cudaDeviceGetStreamPriorityRange": [pointer(ctypes.c_int), pointer(ctypes.c_int)],
            "cudaMalloc": [pointer(ctypes.c_void_p), ctypes.c_size_t],
            "cudaMemcpy": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
            "cudaDeviceSynchronize": [],
            "cudaFree": [ctypes.c_void_p],
        }
        # Resolve functions on demand: a missing optional API must not suppress
        # the separate allocation experiment.
        self.signatures = signatures
        for name in ("cudaGetErrorName", "cudaGetErrorString"):
            if hasattr(self.library, name):
                function = getattr(self.library, name)
                function.argtypes, function.restype = [ctypes.c_int], ctypes.c_char_p

    def call(self, name, *arguments, cleanup=False, label=None):
        previous = self.result.get("active_stage")
        self.stage(label or name)
        try:
            function = getattr(self.library, name)
            function.argtypes, function.restype = self.signatures[name], ctypes.c_int
            code = function(*arguments)
            record = {"name": name, "returncode": code, "cleanup": cleanup,
                      "stage": label or name, "error": None}
            for key, symbol in (("error_name", "cudaGetErrorName"),
                                ("error", "cudaGetErrorString")):
                if hasattr(self.library, symbol):
                    value = getattr(self.library, symbol)(code)
                    record[key] = value.decode("utf-8", errors="replace") if value else None
            self.result["api_calls"].append(record)
            if code:
                raise RuntimeError(f"{name}: code={code}, {record.get('error_name')}: "
                                   f"{record.get('error')}")
            return code
        finally:
            if cleanup:
                self.result["active_stage"] = previous


def execute_case(case, device, library):
    result = {"case": case, "device": device, "passed": False, "api_calls": [],
              "environment": selected_environment(os.environ), "library": library}

    def stage(name):
        result.update(active_stage=name, loaded_libraries=loaded_libraries())
        print(STAGE_MARKER + json.dumps(result), flush=True)

    allocation, api = ctypes.c_void_p(), None
    try:
        if case == "torch_current_stream":
            stage("import_torch")
            import torch
            result.update(torch_version=str(torch.__version__), torch_file=torch.__file__)
            stage("set_device")
            torch.cuda.set_device(device)
            stage("torch_current_stream")
            result["stream"] = str(torch.cuda.current_stream(device))
        elif case in CASES:
            stage("load_runtime_library")
            api = RuntimeAPI(library, result, stage)
            api.call("cudaSetDevice", device)
            if case == "raw_priority_range":
                least, greatest = ctypes.c_int(-999), ctypes.c_int(-999)
                api.call("cudaDeviceGetStreamPriorityRange", ctypes.byref(least), ctypes.byref(greatest))
                result.update(least_priority=least.value, greatest_priority=greatest.value)
            else:
                api.call("cudaMalloc", ctypes.byref(allocation), 4)
                original, copied = ctypes.c_uint32(0x12345678), ctypes.c_uint32(0)
                api.call("cudaMemcpy", allocation, ctypes.byref(original), 4, 1, label="cudaMemcpy_H2D")
                api.call("cudaMemcpy", ctypes.byref(copied), allocation, 4, 2, label="cudaMemcpy_D2H")
                api.call("cudaDeviceSynchronize")
                stage("verify_roundtrip")
                result.update(expected=original.value, actual=copied.value)
                if copied.value != original.value:
                    raise RuntimeError("raw runtime copy roundtrip mismatch")
                pointer_to_free, allocation = allocation, ctypes.c_void_p()
                api.call("cudaFree", pointer_to_free)
        else:
            raise ValueError(f"unknown case: {case}")
        result.update(passed=True, active_stage="completed")
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        if api is not None and allocation.value:
            try:
                api.call("cudaFree", allocation, cleanup=True)
            except Exception as exc:
                result["cleanup_error"] = str(exc)
        result["loaded_libraries"] = loaded_libraries()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", default=DEFAULT_LIBRARY)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case-child", choices=CASES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.device < 0 or args.timeout <= 0:
        parser.error("device must be non-negative and timeout must be positive")
    if args.case_child:
        result = execute_case(args.case_child, args.device, args.library)
        print(MARKER + json.dumps(result), flush=True)
        return 0 if result["passed"] else 1
    if args.output is None:
        parser.error("--output is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = args.output.open("x", encoding="utf-8")
    except FileExistsError:
        parser.error("output already exists; refusing to overwrite evidence")
    with stream:
        report = {"schema_version": 1, "status": "running", "device": args.device,
                  "library": args.library, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "records": [], "confirmed_variants": []}

        def save():
            stream.seek(0)
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.truncate()
            stream.flush()

        save()
        environment = os.environ.copy()
        print("Collecting stream/runtime evidence ...", flush=True)
        report["inventory"] = inventory(environment, corex_roots(environment))
        report["itrctl"] = device_info("/dev/itrctl")
        try:
            report["relevant_mounts"] = [line for line in Path("/proc/self/mountinfo").read_text().splitlines()
                                         if any(word in line.lower() for word in ("corex", "iluvatar", "itrctl"))]
        except OSError as exc:
            report["mountinfo_error"] = str(exc)
        variants, report["iluvatar_candidates"] = build_variants(environment)
        save()

        def attempt(name, case, settings, command):
            print(f"Trying {name}/{case} ...", flush=True)
            result = {"name": name, "case": case, "environment": selected_environment(settings),
                      **parse_smoke(run_command(command, settings, args.timeout))}
            report["records"].append(result)
            save()
            print(f"  {'PASS' if result['passed'] else 'FAIL'} at {result['active_stage']}", flush=True)
            for call in result.get("api_calls", []):
                print(f"  {call['name']}: code={call['returncode']} {call.get('error', '')}", flush=True)
            if not result["passed"]:
                print("  " + result["error"].splitlines()[0], flush=True)
            return result["passed"]

        for name, settings in variants:
            passed = []
            for case in CASES:
                passed.append(attempt(name, case, settings, [sys.executable, str(Path(__file__).resolve()),
                    "--case-child", case, "--device", str(args.device), "--library", args.library]))
            if all(passed) and attempt(name, "torch_add_confirmation", settings,
                    [sys.executable, str(HERE / "runtime_diagnostics.py"), "--smoke-child", "--device", str(args.device)]):
                report["confirmed_variants"].append(name)
        report["status"] = "smoke_confirmed" if report["confirmed_variants"] else "diagnostics_complete_with_failures"
        save()
        print("status: " + report["status"], flush=True)
        print("confirmed variants: " + (", ".join(report["confirmed_variants"]) or "none"), flush=True)
        print("report: " + str(args.output.resolve()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

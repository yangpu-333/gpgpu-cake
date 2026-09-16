"""Compare allocator and existing CoreX paths in isolated GPU smoke processes.

No packages, driver settings, visibility mappings or shell profiles are changed.
The parent uses only the standard library and saves partial evidence after each run.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
MARKER = "GPGPU_RUNTIME_JSON="
STAGE_MARKER = "GPGPU_RUNTIME_STAGE="
ENV_KEYS = (
    "COREX_ROOT", "COREX_HOME", "ILUVATAR_SOFTWARE_ROOT", "LD_LIBRARY_PATH",
    "LD_PRELOAD", "CUDA_VISIBLE_DEVICES", "IX_VISIBLE_DEVICES",
    "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF",
    "PYTORCH_NO_CUDA_MEMORY_CACHING", "CUDA_LAUNCH_BLOCKING",
    "TORCH_SHOW_CPP_STACKTRACES",
)
NATIVE_CONFIG = "backend:native,expandable_segments:False"


def selected_environment(environment):
    return {key: environment.get(key) for key in ENV_KEYS}


def loaded_libraries():
    """Actual Linux process mappings, including libraries loaded with dlopen."""
    try:
        lines = Path("/proc/self/maps").read_text().splitlines()
    except OSError:
        return []
    paths = set()
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) == 6:
            path = fields[5]
            if ".so" in path and any(word in path.lower() for word in (
                "corex", "iluvatar", "libcuda", "libix", "libtorch", "libc10",
            )):
                paths.add(path)
    return sorted(paths)


def smoke(device):
    result = {"passed": False, "device": device}

    def stage(name):
        result["active_stage"] = name
        result["loaded_libraries"] = loaded_libraries()
        print(STAGE_MARKER + json.dumps(result), flush=True)

    try:
        stage("import_torch")
        import torch

        result.update(torch_version=str(torch.__version__), torch_file=torch.__file__,
                      torch_cuda_version=torch.version.cuda)
        stage("set_device")
        torch.cuda.set_device(device)
        stage("read_metadata")
        result.update(device_name=torch.cuda.get_device_name(device),
                      device_count=torch.cuda.device_count())
        try:
            result["allocator_backend"] = torch.cuda.memory.get_allocator_backend()
        except Exception as exc:
            result["allocator_backend_query_error"] = str(exc)
        stage("allocate_one_float")
        tiny = torch.empty(1, dtype=torch.float32, device=f"cuda:{device}")
        stage("synchronize_allocation")
        torch.cuda.synchronize(device)
        stage("allocate_257_floats")
        tensor = torch.empty(257, dtype=torch.float32, device=f"cuda:{device}")
        stage("fill_tensor")
        tensor.fill_(1.0)
        stage("synchronize_fill")
        torch.cuda.synchronize(device)
        stage("torch_add")
        output = tensor + tensor
        stage("synchronize_add")
        torch.cuda.synchronize(device)
        stage("copy_and_check")
        actual = output.cpu()
        if not torch.equal(actual, torch.full((257,), 2.0)):
            raise RuntimeError("incorrect GPU addition result")
        result.update(passed=True, active_stage="completed", numel=actual.numel())
        # Keep allocations alive until after synchronization and the CPU check.
        del output, tensor, tiny
    except Exception as exc:
        result.update(error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    result["loaded_libraries"] = loaded_libraries()
    print(MARKER + json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


def as_text(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")


def run_command(command, environment, timeout):
    try:
        with subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8", errors="replace",
                              start_new_session=(os.name == "posix")) as process:
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return {"returncode": process.returncode, "stdout": as_text(stdout),
                        "stderr": as_text(stderr), "timed_out": False}
            except subprocess.TimeoutExpired as exc:
                if os.name == "posix":
                    # Follow-up diagnostics spawn GPU children of their own.
                    # Terminate only the new session/group created above.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
                stdout, stderr = process.communicate()
                return {"returncode": process.returncode, "stdout": as_text(stdout),
                        "stderr": as_text(stderr), "timed_out": True, "error": str(exc)}
    except OSError as exc:
        return {"returncode": None, "stdout": "", "stderr": "",
                "timed_out": False, "error": str(exc)}


def parse_smoke(process):
    last_stage = {}
    final = None
    for line in process["stdout"].splitlines():
        prefix = next((p for p in (MARKER, STAGE_MARKER) if line.startswith(p)), None)
        if prefix:
            try:
                parsed = json.loads(line[len(prefix):])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                if prefix == MARKER:
                    final = parsed
                else:
                    last_stage = parsed
    result = dict(final or last_stage)
    result["passed"] = bool(final and final.get("passed") and process["returncode"] == 0
                            and not process["timed_out"])
    if not result["passed"]:
        result.setdefault("error", process.get("error") or "child exited without a successful result")
    result.setdefault("active_stage", "start_child")
    return {**result, "process": process}


def corex_roots(environment):
    # Only explicitly configured roots and the documented default. No scanning
    # unrelated SDK versions and no selection of linker stub directories.
    candidates = [environment.get(k) for k in ("COREX_ROOT", "COREX_HOME", "ILUVATAR_SOFTWARE_ROOT")]
    candidates.append("/usr/local/corex")
    roots = []
    for candidate in candidates:
        if candidate:
            root = Path(candidate).resolve()
            if root.is_dir() and "stubs" not in root.parts and root not in roots:
                roots.append(root)
    return roots[:2]


def build_variants(environment, roots):
    common = environment.copy()
    common.update(CUDA_LAUNCH_BLOCKING="1", TORCH_SHOW_CPP_STACKTRACES="1")

    def native(source, no_cache=False):
        target = source.copy()
        # 2.7.1 uses CUDA_ALLOC_CONF; set the newer alias consistently in case
        # the vendor build backports it. A rejected option is diagnostic evidence.
        target.update(PYTORCH_CUDA_ALLOC_CONF=NATIVE_CONFIG, PYTORCH_ALLOC_CONF=NATIVE_CONFIG)
        target.pop("PYTORCH_NO_CUDA_MEMORY_CACHING", None)
        if no_cache:
            target["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
        return target

    variants = [("baseline", common), ("native", native(common)),
                ("native_no_cache", native(common, no_cache=True))]
    for index, root in enumerate(roots):
        directories = []
        for relative in ("lib", "lib64", "targets/x86_64-linux/lib"):
            directory = (root / relative).resolve()
            if directory.is_dir() and "stubs" not in directory.parts and str(directory) not in directories:
                directories.append(str(directory))
        if not directories:
            continue
        adjusted = common.copy()
        adjusted.update(COREX_ROOT=str(root), COREX_HOME=str(root))
        old = adjusted.get("LD_LIBRARY_PATH")
        adjusted["LD_LIBRARY_PATH"] = os.pathsep.join(directories + ([old] if old else []))
        variants.extend([(f"corex{index}_path", adjusted),
                         (f"corex{index}_native", native(adjusted)),
                         (f"corex{index}_native_no_cache", native(adjusted, no_cache=True))])
    return variants


def inventory(environment, roots):
    packages = {}
    for name in ("torch", "triton"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    info = {"python": sys.executable, "python_version": sys.version,
            "platform": platform.platform(), "packages": packages,
            "environment": selected_environment(environment),
            "corex_roots": [str(root) for root in roots], "device_nodes": []}
    for pattern in ("iluvatar*", "ix*", "nvidia*"):
        for path in sorted(Path("/dev").glob(pattern)):
            try:
                stat = path.stat()
                info["device_nodes"].append({"path": str(path), "mode": oct(stat.st_mode),
                                             "uid": stat.st_uid, "gid": stat.st_gid})
            except OSError as exc:
                info["device_nodes"].append({"path": str(path), "error": str(exc)})
    ixsmi = shutil.which("ixsmi", path=environment.get("PATH"))
    info["ixsmi_path"] = ixsmi
    if ixsmi:
        info["ixsmi"] = run_command([ixsmi], environment, 15)
        ldd = shutil.which("ldd")
        if ldd:
            info["ixsmi_ldd"] = run_command([ldd, ixsmi], environment, 15)
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=90, help="seconds per smoke process")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--followup", action="store_true", help="run full device diagnostics after a repeated pass")
    parser.add_argument("--smoke-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.device < 0 or args.timeout <= 0:
        parser.error("device must be non-negative and timeout must be positive")
    if args.smoke_child:
        return smoke(args.device)
    if args.output is None:
        parser.error("--output is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = args.output.open("x", encoding="utf-8")
    except FileExistsError:
        parser.error("output already exists; refusing to overwrite evidence")
    with stream:
        report = {"schema_version": 1, "status": "running", "device": args.device,
                  "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "records": [], "confirmed_variants": []}

        def save():
            stream.seek(0)
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.truncate()
            stream.flush()

        save()
        environment = os.environ.copy()
        roots = corex_roots(environment)
        print("Collecting runtime metadata (no package installation)...", flush=True)
        report["inventory"] = inventory(environment, roots)
        save()
        variants = build_variants(environment, roots)

        def attempt(name, settings):
            print(f"Trying {name} ...", flush=True)
            process = run_command([sys.executable, str(Path(__file__).resolve()), "--smoke-child",
                                   "--device", str(args.device)], settings, args.timeout)
            result = {"name": name, "environment": selected_environment(settings), **parse_smoke(process)}
            report["records"].append(result)
            save()
            print(f"  {'PASS' if result['passed'] else 'FAIL'} at {result['active_stage']}", flush=True)
            if not result["passed"]:
                print("  " + result["error"].splitlines()[0], flush=True)
            return result["passed"]

        selected = None
        for name, settings in variants:
            if attempt(name, settings) and attempt(name + "_confirm", settings):
                report["confirmed_variants"].append(name)
                if selected is None:
                    selected = (name, settings)
        # Detect transient recovery before treating a changed setting as causal.
        if selected and selected[0] != "baseline":
            if attempt("baseline_recheck", variants[0][1]):
                report["baseline_recovered"] = True
        report["status"] = "smoke_confirmed" if selected else "no_working_variant"
        save()

        if selected and args.followup:
            name, settings = selected
            followup_path = args.output.with_name(args.output.stem + "-device.json")
            report["followup"] = {"variant": name, "report": str(followup_path.resolve()), "status": "running"}
            save()
            print(f"Running PyTorch/Triton device diagnostics with {name} ...", flush=True)
            process = run_command([sys.executable, str(HERE / "device_diagnostics.py"),
                                   "--device", str(args.device), "--output", str(followup_path)],
                                  settings, 1800)
            followup = report["followup"]
            followup.update(process=process, status="failed")
            if process["returncode"] == 0 and not process["timed_out"] and followup_path.is_file():
                try:
                    data = json.loads(followup_path.read_text(encoding="utf-8"))
                    followup.update(status=data.get("status", "missing_status"),
                                    failed_cases=data.get("failed_cases"))
                except (OSError, ValueError) as exc:
                    followup["read_error"] = str(exc)
            print("Device diagnostics: " + followup["status"], flush=True)
            if followup.get("failed_cases"):
                print("Failed cases: " + ", ".join(followup["failed_cases"]), flush=True)
        save()
        print("status: " + report["status"], flush=True)
        print("confirmed variants: " + (", ".join(report["confirmed_variants"]) or "none"), flush=True)
        print("report: " + str(args.output.resolve()), flush=True)
        # Exit status describes completion of diagnosis; JSON describes GPU results.
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

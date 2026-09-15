"""Record accelerator state and verify the pinned Megatron-LM checkout.

This script is read-only: it does not install packages or modify Megatron-LM.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_MEGATRON_COMMIT = "5be9626709af2722333bf54797c954c09edeada3"


def git_value(repository, *arguments):
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def inspect_megatron(repository):
    repository = repository.expanduser().resolve()
    result = {
        "root": str(repository),
        "exists": repository.is_dir(),
        "expected_commit": EXPECTED_MEGATRON_COMMIT,
    }
    if not result["exists"]:
        result["error"] = "Megatron root does not exist"
        result["commit_matches"] = False
        return result

    head = git_value(repository, "rev-parse", "HEAD")
    status = git_value(repository, "status", "--short")
    top_level = git_value(repository, "rev-parse", "--show-toplevel")
    result.update(head=head, status=status, top_level=top_level)
    result["commit_matches"] = head["ok"] and head["stdout"] == EXPECTED_MEGATRON_COMMIT
    result["working_tree_dirty"] = status["ok"] and bool(status["stdout"])

    package_dir = repository / "megatron"
    result["package_directory_exists"] = package_dir.is_dir()
    sys.path.insert(0, str(repository))
    try:
        spec = importlib.util.find_spec("megatron")
        result["python_package_found"] = spec is not None
        result["python_package_origin"] = getattr(spec, "origin", None) if spec else None
    except Exception as exc:
        result["python_package_found"] = False
        result["python_package_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if sys.path and sys.path[0] == str(repository):
            sys.path.pop(0)
    return result


def inspect_accelerator(device):
    result = {"requested_device": device, "gpu_ready": False}
    try:
        torch = importlib.import_module("torch")
        result["torch"] = {
            "imported": True,
            "version": getattr(torch, "__version__", None),
            "module_file": getattr(torch, "__file__", None),
            "framework_cuda_version": getattr(torch.version, "cuda", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        }
        if result["torch"]["cuda_available"] and 0 <= device < result["torch"]["device_count"]:
            properties = torch.cuda.get_device_properties(device)
            result["torch"].update(
                device_name=properties.name,
                total_memory_bytes=int(properties.total_memory),
            )
            try:
                result["torch"]["device_capability"] = list(torch.cuda.get_device_capability(device))
            except Exception as exc:
                result["torch"]["device_capability_error"] = f"{type(exc).__name__}: {exc}"
            result["gpu_ready"] = True
    except Exception as exc:
        result["torch"] = {"imported": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        triton = importlib.import_module("triton")
        result["triton"] = {
            "imported": True,
            "version": getattr(triton, "__version__", None),
            "module_file": getattr(triton, "__file__", None),
        }
        try:
            result["triton"]["target"] = str(triton.runtime.driver.active.get_current_target())
        except Exception as exc:
            result["triton"]["target_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        result["triton"] = {"imported": False, "error": f"{type(exc).__name__}: {exc}"}
    return result


def inspect_compiler_selector():
    executable = shutil.which("compiler")
    result = {"executable": executable}
    if not executable:
        result["available"] = False
        return result
    completed = subprocess.run(
        [executable], check=False, capture_output=True, text=True, timeout=20
    )
    result.update(
        available=True,
        returncode=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("--device must be non-negative")
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite evidence")

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "IX_VISIBLE_DEVICES": os.environ.get("IX_VISIBLE_DEVICES"),
            "COREX_ROOT": os.environ.get("COREX_ROOT"),
            "ILUVATAR_SOFTWARE_ROOT": os.environ.get("ILUVATAR_SOFTWARE_ROOT"),
        },
        "megatron": inspect_megatron(args.megatron_root),
        "accelerator": inspect_accelerator(args.device),
        "compiler_selector": inspect_compiler_selector(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    commit_ok = report["megatron"]["commit_matches"]
    package_ok = report["megatron"].get("package_directory_exists", False)
    gpu_ok = report["accelerator"]["gpu_ready"]
    triton_ok = report["accelerator"].get("triton", {}).get("imported", False)
    if not commit_ok:
        report["status"] = "megatron_commit_mismatch"
        code = 2
    elif not package_ok:
        report["status"] = "megatron_package_missing"
        code = 2
    elif not gpu_ok or not triton_ok:
        report["status"] = "accelerator_not_ready"
        code = 3
    else:
        report["status"] = "ready_for_first_validation"
        code = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print("status:", report["status"])
    print("expected Megatron commit:", EXPECTED_MEGATRON_COMMIT)
    print("actual Megatron commit:", report["megatron"].get("head", {}).get("stdout"))
    print("report:", args.output.resolve())
    return code


if __name__ == "__main__":
    raise SystemExit(main())

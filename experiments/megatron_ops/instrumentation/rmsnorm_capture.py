"""Low-intrusion runtime capture for Megatron's fused-residual RMSNorm module.

This module is loaded through ``sitecustomize.py`` before a user-supplied
Megatron launch command.  It does not import Megatron and does not modify its
checkout.  When enabled, it temporarily wraps ``torch.nn.Module._call_impl``
and records only modules whose concrete class name is explicitly selected.

The default target, ``TEFusedResidualRMSNorm``, is the class selected by
Megatron-LM core_v0.19.0 when RMSNorm, Transformer Engine, and
``fused_residual_rmsnorm`` are all enabled.  The module's contract is
``(normalized_output, residual_output)``.  Capturing no calls is useful
evidence that the current training configuration did not exercise that path.
"""

import atexit
import hashlib
import json
import os
import platform
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_TARGETS = ("TEFusedResidualRMSNorm",)
ENV_ENABLE = "CAKE_RMSNORM_CAPTURE"
ENV_OUTPUT_DIR = "CAKE_RMSNORM_CAPTURE_OUTPUT_DIR"
ENV_TARGETS = "CAKE_RMSNORM_CAPTURE_CLASSES"
ENV_MAX_RECORDS = "CAKE_RMSNORM_CAPTURE_MAX_RECORDS"


def _truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(value, default):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def target_names_from_environment(environment=None):
    environment = os.environ if environment is None else environment
    configured = environment.get(ENV_TARGETS, "")
    names = tuple(name.strip() for name in configured.split(",") if name.strip())
    return names or DEFAULT_TARGETS


def _to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def tensor_metadata(tensor):
    """Describe a torch-like tensor without retaining it or reading its data."""
    if tensor is None:
        return None
    required = ("shape", "dtype", "device")
    if not all(hasattr(tensor, item) for item in required):
        return {"kind": type(tensor).__name__, "is_tensor": False}
    try:
        shape = [_to_int(dimension) for dimension in tensor.shape]
    except Exception:
        shape = None
    try:
        stride = [_to_int(dimension) for dimension in tensor.stride()]
    except Exception:
        stride = None
    try:
        contiguous = bool(tensor.is_contiguous())
    except Exception:
        contiguous = None
    metadata = {
        "kind": type(tensor).__name__,
        "is_tensor": True,
        "shape": shape,
        "stride": stride,
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(getattr(tensor, "requires_grad", False)),
        "contiguous": contiguous,
    }
    if shape and len(shape) >= 1:
        metadata["hidden_size"] = shape[-1]
        try:
            metadata["flattened_rows"] = math_product(shape[:-1])
        except (TypeError, ValueError):
            metadata["flattened_rows"] = None
    return metadata


def math_product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def tensor_aliases(left, right):
    if not (getattr(left, "is_cuda", False) or hasattr(left, "data_ptr")):
        return None
    try:
        return bool(left.data_ptr() == right.data_ptr())
    except Exception:
        return None


def output_contract_metadata(output, input_tensor):
    if not isinstance(output, tuple):
        return {"passed": False, "reason": "output_is_not_tuple", "output_type": type(output).__name__}
    if len(output) != 2:
        return {"passed": False, "reason": "tuple_arity_not_two", "tuple_arity": len(output)}
    normalized, residual = output
    normalized_info = tensor_metadata(normalized)
    residual_info = tensor_metadata(residual)
    valid = normalized_info.get("is_tensor") and residual_info.get("is_tensor")
    return {
        "passed": bool(valid),
        "normalized_output": normalized_info,
        "residual_output": residual_info,
        "residual_aliases_input": tensor_aliases(residual, input_tensor) if valid else None,
    }


def _safe_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:80] or "unknown"


class RMSNormCapture:
    """Collect a bounded number of module-call records and serialize at exit."""

    def __init__(self, torch, output_dir, target_names, max_records):
        self.torch = torch
        self.output_dir = Path(output_dir)
        self.target_names = tuple(target_names)
        self.max_records = max_records
        self.records = []
        self.target_call_count = 0
        self._pending_events = []
        self._lock = threading.Lock()
        self._flushed = False
        self.created_at = datetime.now(timezone.utc)

    def is_target(self, module):
        return type(module).__name__ in self.target_names

    def should_capture(self, module):
        if not self.is_target(module):
            return False
        with self._lock:
            self.target_call_count += 1
            return len(self.records) < self.max_records

    def invoke(self, original_call, module, *args, **kwargs):
        input_tensor = args[0] if args else kwargs.get("hidden_states")
        record = {
            "target_call_index": self.target_call_count,
            "module": {"class_name": type(module).__name__, "module_path": type(module).__module__},
            "input": tensor_metadata(input_tensor),
            "status": "started",
        }
        start_event = end_event = None
        try:
            if getattr(input_tensor, "is_cuda", False):
                start_event = self.torch.cuda.Event(enable_timing=True)
                end_event = self.torch.cuda.Event(enable_timing=True)
                start_event.record()
            output = original_call(module, *args, **kwargs)
            if end_event is not None:
                end_event.record()
            record["contract"] = output_contract_metadata(output, input_tensor)
            record["status"] = "captured" if record["contract"]["passed"] else "contract_mismatch"
            with self._lock:
                self.records.append(record)
                if start_event is not None:
                    self._pending_events.append((record, start_event, end_event))
            return output
        except Exception as exc:
            record.update(status="execution_error", error_type=type(exc).__name__, error=str(exc))
            with self._lock:
                self.records.append(record)
            raise

    def _environment(self):
        return {
            "python": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "rank": os.environ.get("RANK"),
            "local_rank": os.environ.get("LOCAL_RANK"),
            "megatron_root": os.environ.get("MEGATRON_ROOT"),
            "target_classes": list(self.target_names),
            "max_records": self.max_records,
        }

    def flush(self):
        with self._lock:
            if self._flushed:
                return None
            self._flushed = True
            pending = list(self._pending_events)
        for record, start_event, end_event in pending:
            try:
                end_event.synchronize()
                record["sampled_forward_event_ms"] = float(start_event.elapsed_time(end_event))
            except Exception as exc:
                record["event_timing_error"] = f"{type(exc).__name__}: {exc}"
        if any(record["status"] == "captured" for record in self.records):
            status = "captured"
        elif self.records:
            status = "captured_with_contract_mismatch"
        else:
            status = "no_target_module_calls"
        report = {
            "schema_version": 1,
            "created_at_utc": self.created_at.isoformat(),
            "operator": "residual_add_rmsnorm_capture",
            "status": status,
            "environment": self._environment(),
            "target_call_count": self.target_call_count,
            "records": self.records,
            "measurement_scope": "sampled forward module-call GPU events for shape discovery; not a training-performance benchmark",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.created_at.strftime("%Y%m%dT%H%M%S%fZ")
        rank = _safe_component(os.environ.get("RANK", "unknown"))
        destination = self.output_dir / f"rmsnorm-capture-{stamp}-rank{rank}-pid{os.getpid()}.json"
        temporary = destination.with_suffix(".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        temporary.replace(destination)
        print(f"CAKE_RMSNORM_CAPTURE_REPORT={destination}", file=sys.stderr)
        return destination


def install_from_environment(environment=None):
    """Install capture only when explicitly enabled; safe to call repeatedly."""
    environment = os.environ if environment is None else environment
    if not _truthy(environment.get(ENV_ENABLE, "")):
        return None
    try:
        import torch
    except Exception as exc:
        print(f"CAKE_RMSNORM_CAPTURE_INSTALL_ERROR=torch_import:{type(exc).__name__}:{exc}", file=sys.stderr)
        return None
    if getattr(torch.nn.Module, "_cake_rmsnorm_capture_installed", False):
        return getattr(torch.nn.Module, "_cake_rmsnorm_capture", None)
    output_dir = environment.get(ENV_OUTPUT_DIR)
    if not output_dir:
        print(f"CAKE_RMSNORM_CAPTURE_INSTALL_ERROR=missing:{ENV_OUTPUT_DIR}", file=sys.stderr)
        return None
    recorder = RMSNormCapture(
        torch=torch,
        output_dir=output_dir,
        target_names=target_names_from_environment(environment),
        max_records=_positive_int(environment.get(ENV_MAX_RECORDS), 64),
    )
    original_call = torch.nn.Module._call_impl

    def wrapped_call(module, *args, **kwargs):
        if recorder.should_capture(module):
            return recorder.invoke(original_call, module, *args, **kwargs)
        return original_call(module, *args, **kwargs)

    torch.nn.Module._call_impl = wrapped_call
    torch.nn.Module._cake_rmsnorm_capture_installed = True
    torch.nn.Module._cake_rmsnorm_capture = recorder
    atexit.register(recorder.flush)
    print(
        "CAKE_RMSNORM_CAPTURE_ENABLED="
        + ",".join(recorder.target_names)
        + f" max_records={recorder.max_records}",
        file=sys.stderr,
    )
    return recorder

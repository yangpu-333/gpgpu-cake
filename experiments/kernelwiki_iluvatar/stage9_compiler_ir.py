"""Retain CoreX GEMM stage IR and introspect relevant Triton APIs."""

import hashlib
import inspect
import json
from pathlib import Path

import torch
import triton
import triton.language as tl

from stage4_gemm_epilogue import gemm_bias


def main():
    directory = Path("stage9-ir-20260928")
    directory.mkdir(exist_ok=False)
    torch.cuda.set_device(0)
    m = n = k = 256
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    bias = torch.randn((n,), device="cuda", dtype=torch.float16)
    output = torch.empty((m, n), device="cuda", dtype=torch.float16)
    summary = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "triton": triton.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "language_symbols": {}, "kernels": [],
    }
    for name in ("swizzle2d", "make_block_ptr", "debug_barrier", "async_copy",
                 "atomic_add", "cumsum", "exp2"):
        value = getattr(tl, name, None)
        entry = {"present": value is not None}
        if value is not None:
            try:
                entry["signature"] = str(inspect.signature(value))
            except Exception as exc:
                entry["signature_error"] = repr(exc)
        summary["language_symbols"][name] = entry
    for stages in (1, 2, 3):
        compiled = gemm_bias[(8, 8)](a, b, bias, output, m, n, k,
                                     32, 32, 32, True,
                                     num_warps=4, num_stages=stages)
        torch.cuda.synchronize()
        entry = {"num_stages": stages, "n_regs": getattr(compiled, "n_regs", None),
                 "n_spills": getattr(compiled, "n_spills", None),
                 "shared": getattr(compiled, "shared", None), "files": []}
        asm = getattr(compiled, "asm", {})
        for kind, value in asm.items():
            if isinstance(value, str):
                payload = value.encode("utf-8")
                suffix = "txt"
            elif isinstance(value, bytes):
                payload = value
                suffix = "bin"
            else:
                entry["files"].append({"kind": kind, "available": False,
                                       "value_type": type(value).__name__})
                continue
            path = directory / f"stages{stages}-{kind}.{suffix}"
            path.write_bytes(payload)
            entry["files"].append({"kind": kind, "available": True,
                                   "path": path.name, "bytes": len(payload),
                                   "sha256": hashlib.sha256(payload).hexdigest()})
        summary["kernels"].append(entry)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("IR_PASS", [(x["num_stages"], x["shared"], x["n_regs"])
                      for x in summary["kernels"]], flush=True)


if __name__ == "__main__":
    main()

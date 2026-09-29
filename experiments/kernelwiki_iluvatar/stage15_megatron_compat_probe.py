"""Probe the pinned Megatron RMSNorm route with process-local CoreX shims.

No installed package or pinned checkout file is changed. The compatibility
mapping is exploratory and does not establish a supported training stack.
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--probe-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()
    root = args.megatron_root.resolve()
    expected = "5be9626709af2722333bf54797c954c09edeada3"
    # Read the exact checkout identity before adding a process-only namespace.
    head_file = root / ".git" / "HEAD"
    if not head_file.is_file() or head_file.read_text().strip() != expected:
        raise RuntimeError("pinned Megatron checkout HEAD differs from expected")

    import transformer_engine as te

    if not hasattr(te, "__version__"):
        te.__version__ = te.te_version()
    print("vendor_te_version", te.__version__)

    namespace = types.ModuleType("megatron")
    namespace.__path__ = [str(root / "megatron")]
    namespace.__package__ = "megatron"
    sys.modules["megatron"] = namespace

    spec = importlib.util.spec_from_file_location("megatron_rmsnorm_route_probe", args.probe_script)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    # The detached HEAD file was checked above. This avoids the old Git build's
    # dubious-ownership check without changing global config or the checkout.
    probe.git_head = lambda repository: {
        "returncode": 0,
        "stdout": head_file.read_text().strip(),
        "stderr": "",
    }
    return probe.main([
        "--megatron-root", str(root),
        "--dtype", args.dtype,
        "--output", str(args.output),
    ])


if __name__ == "__main__":
    sys.exit(main())

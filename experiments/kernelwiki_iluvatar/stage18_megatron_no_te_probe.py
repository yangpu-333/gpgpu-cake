"""Run the pinned Megatron local RMSNorm route with TE masked in this process.

The incompatible CoreX Transformer Engine package is left untouched. This
probes Megatron's local PyTorch path, not a full model training step.
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path


EXPECTED_COMMIT = "5be9626709af2722333bf54797c954c09edeada3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--megatron-root", type=Path, required=True)
    parser.add_argument("--probe-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()
    root = args.megatron_root.resolve()
    head = root / ".git" / "HEAD"
    if not head.is_file() or head.read_text().strip() != EXPECTED_COMMIT:
        raise RuntimeError("pinned Megatron detached HEAD differs from expected")

    # CoreX torch 2.4 keeps the same DTensor class under the older namespace.
    # Exporting that class through the newer import path is process-local.
    import torch.distributed._tensor as legacy_tensor_namespace
    import torch.distributed.tensor as tensor_namespace

    for name in ("DTensor", "DeviceMesh", "Partial", "Placement", "Replicate",
                 "Shard", "distribute_module", "distribute_tensor"):
        if not hasattr(tensor_namespace, name):
            setattr(tensor_namespace, name, getattr(legacy_tensor_namespace, name))
    sys.modules.setdefault(
        "torch.distributed.tensor.placement_types",
        legacy_tensor_namespace.placement_types,
    )
    sys.modules["transformer_engine"] = None
    namespace = types.ModuleType("megatron")
    namespace.__path__ = [str(root / "megatron")]
    namespace.__package__ = "megatron"
    sys.modules["megatron"] = namespace

    spec = importlib.util.spec_from_file_location("megatron_rmsnorm_route_probe", args.probe_script)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    probe.git_head = lambda repository: {
        "returncode": 0,
        "stdout": head.read_text().strip(),
        "stderr": "",
    }
    return probe.main([
        "--megatron-root", str(root),
        "--dtype", args.dtype,
        "--output", str(args.output),
    ])


if __name__ == "__main__":
    sys.exit(main())

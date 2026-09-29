"""Check whether CoreX 4.2 applies Triton warp-specialization option."""

import hashlib
import json
from pathlib import Path

import torch
import triton

from stage4_gemm_epilogue import gemm_bias


def main():
    torch.cuda.set_device(0)
    m = n = k = 256
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
    bias = torch.randn((n,), device="cuda", dtype=torch.float16)
    out = torch.empty((m, n), device="cuda", dtype=torch.float16)
    reference = ((a.double().cpu() @ b.double().cpu()).half().float()
                 + bias.float().cpu()).half()
    result = {"gpu": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "triton": triton.__version__,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "variants": []}
    for name, kwargs in (("default", {}),
                         ("enable_warp_specialization", {"enable_warp_specialization": True}),
                         ("enable_persistent", {"enable_persistent": True})):
        item = {"name": name, "kwargs": kwargs}
        try:
            compiled = gemm_bias[(8, 8)](a, b, bias, out, m, n, k,
                                         32, 32, 32, True,
                                         num_warps=4, num_stages=2, **kwargs)
            item["correct"] = bool(torch.allclose(out.cpu(), reference,
                                                  atol=0.03, rtol=0.03))
            item["n_regs"] = getattr(compiled, "n_regs", None)
            item["shared"] = getattr(compiled, "shared", None)
            item["asm"] = {}
            for key in ("ttgir", "llir"):
                payload = getattr(compiled, "asm", {}).get(key)
                if isinstance(payload, str):
                    path = Path(f"stage9-ws-{name}-{key}.txt")
                    path.write_text(payload, encoding="utf-8")
                    item["asm"][key] = {"path": path.name,
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                        "bytes": path.stat().st_size}
            item["status"] = "compiled" if item["correct"] else "numerical_failure"
        except Exception as exc:
            item["status"] = "error"
            item["error"] = repr(exc)
        result["variants"].append(item)
        print(name, item["status"], flush=True)
    Path("stage9-ws-report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

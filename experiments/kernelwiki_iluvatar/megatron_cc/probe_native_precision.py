"""Diagnose native RMSNorm precision using CPU copies, before candidate creation."""
import json
import math
import torch

torch.manual_seed(23001)
records = []
for dtype in (torch.float32, torch.float16, torch.bfloat16):
    for scale in (1.0, 0.001, 100.0):
        x = (torch.randn(16, 256) * scale).to("cuda", dtype).requires_grad_()
        weight = torch.randn(256, device="cuda", requires_grad=True)
        y = torch.nn.functional.rms_norm(x, (256,), weight, 1e-5)
        dy = torch.randn_like(y)
        gradients = torch.autograd.grad(y, (x, weight), dy)
        facts = {}
        for name, value in zip(("output", "dx", "dw"), (y, *gradients)):
            numbers = value.detach().cpu().double().flatten().tolist()
            facts[name] = {"dtype": str(value.dtype),
                "gpu_isfinite_all": bool(torch.isfinite(value).all().item()),
                "cpu_finite": sum(math.isfinite(v) for v in numbers),
                "cpu_nan": sum(math.isnan(v) for v in numbers),
                "cpu_inf": sum(math.isinf(v) for v in numbers),
                "elements": len(numbers),
                "finite_abs_max": max((abs(v) for v in numbers if math.isfinite(v)), default=None)}
        records.append({"input_dtype": str(dtype), "scale": scale, "facts": facts})
print(json.dumps(records, indent=2))

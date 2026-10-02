"""Check the numerical comparator itself with a deliberately unequal pair."""
import json
import torch

a = torch.tensor([1.0, 2.0], device="cuda")
b = torch.tensor([1.125, 2.25], device="cuda")
result = {}
for name, av, bv in (("gpu_float32", a, b), ("gpu_float64", a.double(), b.double()),
                     ("cpu_float64", a.cpu().double(), b.cpu().double())):
    delta = (av - bv).abs()
    result[name] = {"values": delta.cpu().tolist(), "max": delta.max().item(),
                    "all_close_1e_minus_3": bool((delta <= 0.001).all().item())}
print(json.dumps(result, indent=2))

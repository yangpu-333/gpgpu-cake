"""Characterize compiled native SwiGLU against eager and FP32 formulas."""
import hashlib
import json
from pathlib import Path
import sys
import torch

BASE = Path('/tmp/kda-swiglu-stage29')
sys.path.insert(0, '/tmp/kda-bf16-continuation/megatron_training_entry')
from runtime_compat import prepare_local_runtime
prepare_local_runtime(Path('/tmp/kda-bf16-continuation/megatron-native-5be9626'))
from megatron.core.fusions import fused_bias_swiglu as f

records = []
torch.manual_seed(29111)
for scale in (0., .01, 1., 10.):
    x = (torch.randn(17, 2, 514, device='cuda', dtype=torch.bfloat16) * scale).requires_grad_()
    g = torch.randn(17, 2, 257, device='cuda', dtype=torch.bfloat16)
    y = f.bias_swiglu_impl(x, None, False, False)
    dx = torch.autograd.grad(y, x, g)[0]
    a, b = x.detach().chunk(2, -1)
    y_eager = f.swiglu._torchdynamo_orig_callable(x.detach())
    dx_eager = f.swiglu_back._torchdynamo_orig_callable(g, x.detach())
    af, bf, gf = a.float(), b.float(), g.float()
    sig = 1 / (1 + torch.exp(-af))
    y_fp32 = (af * sig * bf).to(x.dtype)
    dx_fp32 = torch.cat((gf * sig * (1 + af * (1 - sig)) * bf, gf * torch.nn.functional.silu(af)), -1).to(x.dtype)
    record = {'scale': scale, 'comparisons': {}}
    for name, left, right in (('output_vs_eager', y, y_eager), ('dx_vs_eager', dx, dx_eager),
                              ('output_vs_fp32', y, y_fp32), ('dx_vs_fp32', dx, dx_fp32)):
        l, r = left.detach().cpu().double(), right.detach().cpu().double()
        record['comparisons'][name] = {'bitwise_equal': torch.equal(l, r),
            'passed_fixed_tolerance': bool(torch.isfinite(l).all() and torch.isfinite(r).all() and torch.allclose(l, r, atol=3e-4, rtol=1e-3)),
            'max_abs': float((l-r).abs().max()), 'unequal_elements': int((l != r).sum())}
    records.append(record)

out = {'schema': 'native-compiled-swiglu-semantics-v1', 'checks': records,
       'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'note': 'Probe diagnoses arithmetic; candidate must compare to compiled native, not eager or a mathematical substitute.'}
path = BASE/'semantics.json'
assert not path.exists()
path.write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out), flush=True)

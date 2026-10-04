"""Untimed diagnostic: observe compiled closure API calls, not a GPU timeline."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch

BASE = Path('/tmp/kda-swiglu-stage29')
p = argparse.ArgumentParser()
p.add_argument('--version', required=True)
a = p.parse_args()
assert a.version.isdigit()
output = BASE / ('launch-observation-' + a.version + '.json')
assert not output.exists()
sys.path.insert(0, '/tmp/kda-bf16-continuation/megatron_training_entry')
from runtime_compat import prepare_local_runtime
prepare_local_runtime(Path('/tmp/kda-bf16-continuation/megatron-native-5be9626'))
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
path = BASE / ('candidate-' + a.version + '.py')
spec = importlib.util.spec_from_file_location('observed_candidate', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
payload = torch.load(BASE / 'main-native-audit/activation/forward.pt', map_location='cpu', weights_only=False)
x = payload['input'].cuda().requires_grad_()
g = torch.load(BASE / 'main-native-audit/activation/upstream-grad.pt', map_location='cpu', weights_only=False).cuda()
z = module.swiglu(x)
torch.autograd.grad(z, x, g)
counts = {'fwd': 0, 'bwd': 0}
state_before = dict(getattr(module, '_state', {}))
cache = getattr(module, '_launch_cache', {})
for key, closure in list(cache.items()):
    def observed(*args, _closure=closure, _phase=key[0], **kwargs):
        counts[_phase] += 1
        return _closure(*args, **kwargs)
    cache[key] = observed
checks = []
for step in range(8):
    current = (x.detach() * (1.0 + step / 16)).requires_grad_()
    grad = g * (1.0 + step / 32)
    expected = bias_swiglu_impl(current, None, False, False)
    expected_grad = torch.autograd.grad(expected, current, grad)[0]
    actual = module.swiglu(current)
    actual_grad = torch.autograd.grad(actual, current, grad)[0]
    checks.append(all(torch.allclose(left.detach().cpu().double(), right.detach().cpu().double(),
                                    atol=3e-4, rtol=1e-3)
                      for left, right in ((actual, expected), (actual_grad, expected_grad))))
result = {'version': a.version, 'candidate_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
          'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'checks': checks, 'compiled_closure_api_calls': counts,
          'state_before': state_before, 'state_after': dict(getattr(module, '_state', {})),
          'cached_closures': len(cache), 'passed': all(checks), 'timed': False,
          'scope': 'Separate untimed instrumented fresh-input calls. Counts observe host compiled '
                   'closure API invocation; they do not certify a GPU timeline or measure speed.'}
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)

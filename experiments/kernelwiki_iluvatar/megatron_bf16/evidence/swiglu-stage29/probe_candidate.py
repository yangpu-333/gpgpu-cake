"""Fixed native-compiled SwiGLU numerical matrix and CUDA Event pairs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import torch

BASE = Path('/tmp/kda-swiglu-stage29')
p = argparse.ArgumentParser()
p.add_argument('--version', required=True)
a = p.parse_args()
candidate = BASE/('candidate-'+a.version+'.py')
outpath = BASE/('probe-'+a.version+'.json')
assert not outpath.exists()
sys.path.insert(0, '/tmp/kda-bf16-continuation/megatron_training_entry')
from runtime_compat import prepare_local_runtime
prepare_local_runtime(Path('/tmp/kda-bf16-continuation/megatron-native-5be9626'))
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
spec = importlib.util.spec_from_file_location('candidate_swiglu', candidate)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch.manual_seed(29112)
records = []
shapes = ((17, 514), (257, 1538), (128, 2, 4096), (256, 2, 4096), (513, 2, 4096))


def check(name, x, g):
    before = x.detach().cpu().clone()
    y = bias_swiglu_impl(x, None, False, False)
    dx = torch.autograd.grad(y, x, g)[0]
    z = m.swiglu(x)
    dz = torch.autograd.grad(z, x, g)[0]
    row = {'case': name, 'shape': list(x.shape), 'stride': list(x.stride()), 'checks': []}
    for label, left, right in (('output', z, y), ('input_grad', dz, dx)):
        l, r = left.detach().cpu().double(), right.detach().cpu().double()
        valid = (left.shape == right.shape and left.dtype == right.dtype
                 and bool(torch.isfinite(l).all() and torch.isfinite(r).all()
                          and torch.allclose(l, r, atol=3e-4, rtol=1e-3)))
        row['checks'].append({'name': label, 'passed': valid, 'bitwise_equal': torch.equal(l, r),
                              'max_abs': float((l-r).abs().max()), 'unequal_elements': int((l != r).sum())})
    row['input_unchanged'] = torch.equal(before, x.detach().cpu())
    records.append(row)


for shape in shapes:
    for scale in (0., .01, 1., 10.):
        x = (torch.randn(shape, device='cuda', dtype=torch.bfloat16)*scale).requires_grad_()
        g = torch.randn(shape[:-1]+(shape[-1]//2,), device='cuda', dtype=x.dtype)
        check('random-scale-'+str(scale), x, g)
payload = torch.load(BASE/'main-native-audit/activation/forward.pt', map_location='cpu', weights_only=False)
x = payload['input'].cuda().requires_grad_()
g = torch.load(BASE/'main-native-audit/activation/upstream-grad.pt', map_location='cpu', weights_only=False).cuda()
check('actual-native-training-input-and-grad', x, g)
gstrided = torch.randn(17, 514, device='cuda', dtype=torch.bfloat16)[:, ::2]
check('noncontiguous-upstream-gradient', torch.randn(17,514,device='cuda',dtype=torch.bfloat16,requires_grad=True), gstrided)
guards = []
for name, bad in (('cpu', torch.ones(2,514,dtype=torch.bfloat16)),
                  ('FP32', torch.ones(2,514,device='cuda')),
                  ('odd-lastdim', torch.ones(2,515,device='cuda',dtype=torch.bfloat16)),
                  ('noncontiguous-input', torch.ones(514,2,device='cuda',dtype=torch.bfloat16).t())):
    try:
        m.swiglu(bad)
        guards.append({'name': name, 'passed': False})
    except NotImplementedError:
        guards.append({'name': name, 'passed': True})
passed = all(r['input_unchanged'] and all(c['passed'] for c in r['checks']) for r in records) and all(g['passed'] for g in guards)
perf = []
if passed:
    x = payload['input'].cuda().requires_grad_()
    g = torch.randn(x.shape[:-1]+(x.shape[-1]//2,),device='cuda',dtype=x.dtype)
    for backward in (False, True):
        def call(arm):
            y = bias_swiglu_impl(x,None,False,False) if arm == 'native' else m.swiglu(x)
            if backward:
                torch.autograd.grad(y,x,g)
        for arm in ('native','candidate'):
            for _ in range(10):call(arm)
        torch.cuda.synchronize()
        samples = {'native': [], 'candidate': []}
        for index in range(15):
            for arm in (('native','candidate') if index%2 == 0 else ('candidate','native')):
                start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(30):call(arm)
                end.record();end.synchronize()
                samples[arm].append(start.elapsed_time(end)*1000/30)
        perf.append({'backward':backward, 'samples_us':samples,
                     'median_us':{k:statistics.median(v) for k,v in samples.items()}})
out = {'status':'passed' if passed else 'rejected',
       'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
       'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'checks':records,'guard_checks':guards,'performance':perf,
       'timer':'torch.cuda.Event; 10warmup,15paired samples x30calls; forward or forward+backward. Not formal training throughput.'}
outpath.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':out['status'],'failed':[r for r in records if not r['input_unchanged'] or not all(c['passed'] for c in r['checks'])], 'performance':perf}),flush=True)

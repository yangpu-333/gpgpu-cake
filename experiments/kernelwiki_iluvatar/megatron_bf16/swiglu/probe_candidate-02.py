"""Independent native-compiled SwiGLU probes, fresh-data checks, and Event pairs.

The original 22 numerical cases and guard matrix are retained. Extra probes
exercise code-cache reuse, alignment, and the active CUDA stream. Event API
elapsed time may include host submission gaps; formal training uses its own
unchanged GPU-synchronized wall timer.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys
import torch


BASE = Path('/tmp/kda-swiglu-stage29')
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--version', required=True)
a = p.parse_args()
assert a.version.isdigit(), 'simple candidate version required'
candidate = BASE / ('candidate-' + a.version + '.py')
outpath = BASE / ('probe-' + a.version + '-v2.json')
assert not outpath.exists(), 'immutable probe evidence already exists'
sys.path.insert(0, '/tmp/kda-bf16-continuation/megatron_training_entry')
from runtime_compat import prepare_local_runtime
native_root = Path('/tmp/kda-bf16-continuation/megatron-native-5be9626')
runtime = prepare_local_runtime(native_root)
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
spec = importlib.util.spec_from_file_location('candidate_swiglu', candidate)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_extension():
    if not hasattr(m, '_ext'):
        return {'required': False, 'passed': True}
    receipt_path = BASE / ('cpp-build-' + a.version + '-receipt.json')
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'built', 'successful explicit build required'
    name = receipt['name']
    assert re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name), 'invalid module name'
    build_dir = (BASE / ('cpp-build-' + a.version)).resolve()
    expected = build_dir / (name + '.so')
    artifact = Path(m._ext.__file__).resolve()
    assert artifact == expected, 'extension imported outside pinned build location'
    assert Path(receipt['artifact']).resolve() == expected
    assert Path(receipt['build_directory']).resolve() == build_dir
    binary_sha = sha(artifact)
    assert receipt['artifact_sha256'] == binary_sha, 'extension bytes changed after build'
    source_hashes = {filename: sha(BASE / filename) for filename in
                     ('swiglu-' + a.version + '.cpp', 'swiglu-' + a.version + '.cu')}
    assert receipt['sources'] == source_hashes, 'model source differs from explicit build'
    build_script_sha = receipt['build_script_sha256']
    assert re.fullmatch(r'[0-9a-f]{64}', build_script_sha), 'build script hash required'
    script_path = Path(receipt['build_script_path']).resolve()
    assert script_path == (BASE / 'build_cpp.py').resolve(), 'fixed controller build script required'
    assert sha(script_path) == build_script_sha, 'build script differs from receipt'
    return {'required': True, 'passed': True, 'receipt_path': str(receipt_path),
            'receipt_sha256': sha(receipt_path), 'artifact': str(artifact),
            'artifact_sha256': binary_sha, 'sources': source_hashes,
            'build_script_sha256': build_script_sha,
            'build_script_path': str(script_path), 'build_script_bytes_checked': True}


extension = verify_extension()
torch.manual_seed(29112)
records = []
shapes = ((17, 514), (257, 1538), (128, 2, 4096), (256, 2, 4096), (513, 2, 4096))


def check(name, x, g, group='original'):
    before = x.detach().cpu().clone()
    before_g = g.detach().cpu().clone()
    active_stream = torch.cuda.current_stream(x.device)
    y = bias_swiglu_impl(x, None, False, False)
    dx = torch.autograd.grad(y, x, g)[0]
    z = m.swiglu(x)
    dz = torch.autograd.grad(z, x, g)[0]
    row = {'case': name, 'group': group, 'shape': list(x.shape),
           'stride': list(x.stride()), 'input_storage_offset': x.storage_offset(),
           'upstream_shape': list(g.shape), 'upstream_stride': list(g.stride()),
           'upstream_storage_offset': g.storage_offset(),
           'current_stream': int(active_stream.cuda_stream),
           'checks': []}
    for label, left, right in (('output', z, y), ('input_grad', dz, dx)):
        l, r = left.detach().cpu().double(), right.detach().cpu().double()
        shape_ok = left.shape == right.shape
        dtype_ok = left.dtype == right.dtype == x.dtype
        device_ok = left.device == right.device == x.device
        finite = bool(torch.isfinite(l).all() and torch.isfinite(r).all())
        values_ok = shape_ok and bool(torch.allclose(l, r, atol=3e-4, rtol=1e-3))
        row['checks'].append({'name': label,
                              'passed': bool(shape_ok and dtype_ok and device_ok and finite and values_ok),
                              'shape_equal': shape_ok, 'dtype_equal': dtype_ok,
                              'device_equal': device_ok, 'finite': finite,
                              'bitwise_equal': shape_ok and torch.equal(l, r),
                              'max_abs': float((l-r).abs().max()) if shape_ok else None,
                              'unequal_elements': int((l != r).sum()) if shape_ok else None})
    row['input_unchanged'] = torch.equal(before, x.detach().cpu())
    row['upstream_unchanged'] = torch.equal(before_g, g.detach().cpu())
    records.append(row)


# Original fixed numerical matrix: 20 random cases + real training + strided g.
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
check('noncontiguous-upstream-gradient',
      torch.randn(17, 514, device='cuda', dtype=torch.bfloat16, requires_grad=True), gstrided)
assert len(records) == 22

# Fresh inputs and gradients with identical shape exercise steady-state code reuse.
primary_shape = tuple(payload['input'].shape)
for index, scale in enumerate((1., .01, 10., 1.)):
    x = (torch.randn(primary_shape, device='cuda', dtype=torch.bfloat16)*scale).requires_grad_()
    g = torch.randn(primary_shape[:-1]+(primary_shape[-1]//2,), device='cuda', dtype=x.dtype)
    check('same-shape-fresh-input-and-grad-'+str(index), x, g, 'fresh-data-reuse')
x = torch.randn(primary_shape, device='cuda', dtype=torch.bfloat16, requires_grad=True)
for index in range(2):
    g = torch.randn(primary_shape[:-1]+(primary_shape[-1]//2,), device='cuda', dtype=x.dtype)
    check('same-input-fresh-grad-'+str(index), x, g, 'fresh-data-reuse')

# Return to previously compiled sizes rather than running each size only once.
for index, shape in enumerate(((17, 514), primary_shape, (257, 1538), (17, 514), primary_shape)):
    x = torch.randn(shape, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(shape[:-1]+(shape[-1]//2,), device='cuda', dtype=x.dtype)
    check('alternating-size-'+str(index), x, g, 'alternating-sizes')

# Contiguity does not imply the allocator's usual pointer alignment.
for index, shape in enumerate(((17, 514), primary_shape)):
    n_input = 1
    for extent in shape:
        n_input *= extent
    gshape = shape[:-1]+(shape[-1]//2,)
    x_storage = torch.randn(n_input+7, device='cuda', dtype=torch.bfloat16)
    g_storage = torch.randn(n_input//2+11, device='cuda', dtype=torch.bfloat16)
    x = x_storage[3:3+n_input].view(shape).requires_grad_()
    g = g_storage[5:5+n_input//2].view(gshape)
    assert x.is_contiguous() and g.is_contiguous()
    assert x.storage_offset() == 3 and g.storage_offset() == 5
    check('offset-contiguous-input-and-grad-'+str(index), x, g, 'storage-offset')

# Cached launchers must obtain the current stream on every actual invocation.
for index in range(2):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = (torch.randn(primary_shape, device='cuda', dtype=torch.bfloat16)
             * (index+1)).requires_grad_()
        g = torch.randn(primary_shape[:-1]+(primary_shape[-1]//2,), device='cuda', dtype=x.dtype)
        check('nondefault-current-stream-'+str(index), x, g, 'current-stream')
    stream.synchronize()

# Original guard matrix, unchanged.
guards = []
for name, bad in (('cpu', torch.ones(2, 514, dtype=torch.bfloat16)),
                  ('FP32', torch.ones(2, 514, device='cuda')),
                  ('odd-lastdim', torch.ones(2, 515, device='cuda', dtype=torch.bfloat16)),
                  ('noncontiguous-input', torch.ones(514, 2, device='cuda', dtype=torch.bfloat16).t())):
    try:
        m.swiglu(bad)
        guards.append({'name': name, 'passed': False})
    except NotImplementedError:
        guards.append({'name': name, 'passed': True})
passed = (all(r['input_unchanged'] and r['upstream_unchanged']
              and all(c['passed'] for c in r['checks']) for r in records)
          and all(item['passed'] for item in guards))
perf = []
if passed:
    x = payload['input'].cuda().requires_grad_()
    g = torch.randn(x.shape[:-1]+(x.shape[-1]//2,), device='cuda', dtype=x.dtype)
    for backward in (False, True):
        def call(arm):
            y = bias_swiglu_impl(x, None, False, False) if arm == 'native' else m.swiglu(x)
            if backward:
                torch.autograd.grad(y, x, g)
        for arm in ('native', 'candidate'):
            for _ in range(10):
                call(arm)
        torch.cuda.synchronize()
        samples = {'native': [], 'candidate': []}
        for index in range(15):
            for arm in (('native', 'candidate') if index % 2 == 0 else ('candidate', 'native')):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(30):
                    call(arm)
                end.record()
                end.synchronize()
                samples[arm].append(start.elapsed_time(end)*1000/30)
        perf.append({'backward': backward, 'samples_us': samples,
                     'median_us': {k: statistics.median(v) for k, v in samples.items()}})
out = {'schema': 'native-compiled-swiglu-independent-probe-v2',
       'status': 'passed' if passed else 'rejected',
       'candidate_sha256': sha(candidate), 'verifier_sha256': sha(Path(__file__)),
       'extension': extension, 'runtime_compatibility': runtime,
       'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
       'native_source_sha256': {name: sha(native_root/name) for name in
                               ('megatron/core/fusions/fused_bias_swiglu.py', 'megatron/core/jit.py')},
       'original_case_count': 22, 'total_case_count': len(records),
       'checks': records, 'guard_checks': guards, 'performance': perf,
       'timer': 'torch.cuda.Event API elapsed; 10 warmup, 15 paired samples x30 calls; '
                'forward or forward+backward. CPU dispatch and autograd submission gaps '
                'can appear inside the Event interval. Compilation excluded. '
                'Not isolated kernel time or formal training throughput; formal training '
                'retains its separate original GPU-synchronized time.time wall timer.'}
with outpath.open('x') as stream:
    json.dump(out, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({'status': out['status'], 'cases': len(records),
                  'failed': [r for r in records if not r['input_unchanged']
                             or not r['upstream_unchanged'] or not all(c['passed'] for c in r['checks'])],
                  'performance': perf}), flush=True)

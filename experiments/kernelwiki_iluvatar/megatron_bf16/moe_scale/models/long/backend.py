"""Untimed fresh runtime/source/sort admission for both stage35 targets."""
import fcntl
import hashlib
import json
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parent
NATIVE = Path('/tmp/kda-bf16-continuation/megatron-native-5be9626')
FORMAL = NATIVE.parent / 'megatron_training_entry'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
output = ROOT / 'backend-current.json'
assert not output.exists()
manifest = json.loads((FORMAL / 'source_manifest.json').read_text())
assert len(manifest['files']) == 600
assert all(sha(NATIVE / name) == digest for name, digest in manifest['files'].items())
sys.path.insert(0, str(FORMAL))
from runtime_compat import prepare_local_runtime

with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    runtime = prepare_local_runtime(NATIVE)
    import torch
    torch.set_num_threads(4)
    assert torch.cuda.get_device_name(0) in ('BI-V150', 'Iluvatar BI-V150')
    x = torch.ones(8, device='cuda')
    torch.cuda.synchronize()
    assert torch.equal(x.cpu(), torch.ones(8))
    rows = []
    torch.manual_seed(35019)
    for count in (0, 1, 17, 2048, 8192):
        for pattern in ('zero', 'one', 'random', 'skewed'):
            keys = torch.rand(count) < (0.01 if pattern == 'skewed' else 0.5)
            if pattern == 'zero': keys.zero_()
            elif pattern == 'one': keys.fill_(True)
            for descending in (False, True):
                reference = keys.argsort(stable=True, descending=descending)
                actual = keys.cuda().to(torch.uint8).argsort(stable=True, descending=descending).cpu()
                rows.append({'count': count, 'pattern': pattern, 'descending': descending,
                             'bitwise_equal': bool(torch.equal(reference, actual))})
    assert len(rows) == 40 and all(row['bitwise_equal'] for row in rows)
    try:
        torch.tensor([True, False], device='cuda').argsort(stable=True)
        native_bool = {'supported': True}
    except RuntimeError as error:
        native_bool = {'supported': False, 'error': str(error)}
    record = {'schema': 'moe-heldout-stage35-fresh-backend-v1', 'passed': True,
              'hostname': socket.gethostname(), 'gpu': torch.cuda.get_device_name(0),
              'torch': torch.__version__, 'runtime': runtime, 'source_manifest_sha256': sha(FORMAL/'source_manifest.json'),
              'native_sources_sha256': manifest['files'], 'sort_index_probes': rows,
              'native_gpu_bool_sort': native_bool, 'allocation_verified': True,
              'script_sha256': sha(__file__), 'scope': 'Untimed preflight; no training throughput result.'}
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'passed': True, 'gpu': record['gpu'], 'torch': record['torch'],
                      'sort_probes': len(rows), 'native_gpu_bool_sort': native_bool}))

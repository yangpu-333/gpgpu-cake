"""Reproduce only the frozen CPU offload copy/conversion chain; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    active = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                             '--format=csv,noheader'], text=True, capture_output=True, check=True)
    if active.stdout.strip():
        raise RuntimeError('Another GPU compute process is active; skip diagnostic')
    import torch
    root = Path('/home/huids25/kda-jiuge-prep-20261014')
    relative = 'megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py'
    source = root / 'megatron-native-5be9626' / relative
    actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    expected_sha = json.loads((root / 'source_manifest.json').read_text())['files'][relative]
    if actual_sha != expected_sha:
        raise ValueError('Frozen offload source differs')
    original = torch.arange(16, dtype=torch.bfloat16)
    pinned = original.detach().clone().cpu().pin_memory()
    cloned = pinned.detach().clone()
    master = cloned.float()
    repinned = master.pin_memory()
    result = {'schema': 'jiuge-offload-pinning-copy-chain-probe-v1',
              'source_path': relative, 'source_sha256': actual_sha,
              'source_lines': [274, 277, 278], 'torch_version': torch.__version__,
              'initial_dtype': str(original.dtype), 'master_dtype': str(master.dtype),
              'after_cpu_pin_memory_is_pinned': pinned.is_pinned(),
              'after_detach_clone_is_pinned': cloned.is_pinned(),
              'after_float_is_pinned': master.is_pinned(),
              'explicit_repin_is_pinned': repinned.is_pinned(),
              'copied_values_equal': torch.equal(master, original.float()),
              'gpu_tensors_created': False, 'cpu_pinned_allocation': True,
              'native_source_modified': False, 'full_model_optimization_tested': False,
              'formal_performance_passed': False}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

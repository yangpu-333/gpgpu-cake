"""Prepare an isolated FP32 master-pinning candidate from the frozen HDO source."""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path

EXPECTED = '6a7f96e2afd0081991aff86424e5fdd9c07bce8c2636728baa0b8b2c8b4f2be2'
SOURCE = Path('/home/huids25/kda-jiuge-prep-20261014/megatron-native-5be9626/megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py')


def candidate_text(source):
    if hashlib.sha256(source).hexdigest() != EXPECTED:
        raise ValueError('Frozen source SHA256 differs')
    text = source.decode()
    before = '''                    param = param.detach().clone().cpu().pin_memory()
                    offload_params_numel += param.numel()
                    cpu_copy = True
                if self.param_update_in_fp32 and param.dtype != torch.float32:
                    param = param.detach().clone().float()
                    param_to_fp32_param[orig_param] = param
'''
    after = '''                    param = param.detach().clone().cpu()
                    offload_params_numel += param.numel()
                    cpu_copy = True
                if self.param_update_in_fp32 and param.dtype != torch.float32:
                    param = param.detach().clone().float()
                    if cpu_copy and self.pin_cpu_params:
                        param = param.pin_memory()
                    param_to_fp32_param[orig_param] = param
                elif cpu_copy and self.pin_cpu_params:
                    param = param.pin_memory()
'''
    if text.count(before) != 1:
        raise ValueError('Expected exactly one frozen conversion block')
    return text.replace(before, after)


def load_class(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HybridDeviceOptimizer


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    original = SOURCE.read_bytes()
    changed = candidate_text(original)
    (output / 'native_hybrid_optimizer.py').write_bytes(original)
    (output / 'candidate_hybrid_optimizer.py').write_text(changed)
    patch = ''.join(difflib.unified_diff(original.decode().splitlines(True), changed.splitlines(True),
                                       fromfile='native/hybrid_optimizer.py', tofile='candidate/hybrid_optimizer.py'))
    (output / 'candidate.patch').write_text(patch)
    record = {'schema': 'jiuge-master-pinning-candidate-v1', 'native_source_sha256': EXPECTED,
              'candidate_sha256': hashlib.sha256(changed.encode()).hexdigest(),
              'patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
              'native_source_modified': False, 'formal_performance_passed': False}
    (output / 'candidate.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    prepare(parser.parse_args().output)

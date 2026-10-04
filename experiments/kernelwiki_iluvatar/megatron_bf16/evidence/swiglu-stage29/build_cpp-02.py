"""Explicit controller build; generated candidates never compile their own code."""
import argparse
import hashlib
import json
import os
from pathlib import Path

BASE = Path('/tmp/kda-swiglu-stage29')
p = argparse.ArgumentParser()
p.add_argument('--version', required=True)
a = p.parse_args()
assert a.version.isdigit()
build = BASE / ('cpp-build-' + a.version)
receipt_path = BASE / ('cpp-build-' + a.version + '-receipt.json')
assert not receipt_path.exists()
build.mkdir(exist_ok=False)
os.environ['MAX_JOBS'] = '1'
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0'
import torch
from torch.utils.cpp_extension import load
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
sources = [BASE / ('swiglu-' + a.version + suffix) for suffix in ('.cpp', '.cu')]
receipt = {'name': 'kda_swiglu_stage29_' + a.version,
           'torch': torch.__version__, 'cuda': torch.version.cuda,
           'arch': '7.0 SDK compatibility target; not a NVIDIA GPU identity',
           'sources': {path.name: sha(path) for path in sources},
           'build_script_path': str(Path(__file__).resolve()),
           'build_script_sha256': sha(Path(__file__)),
           'extra_cflags': ['-O3', '-ffp-contract=off'],
           'extra_cuda_cflags': ['-O3', '-ffp-contract=off'],
           'build_directory': str(build)}
try:
    module = load(name=receipt['name'], sources=[str(path) for path in sources],
                  build_directory=str(build), extra_cflags=receipt['extra_cflags'],
                  extra_cuda_cflags=receipt['extra_cuda_cflags'], verbose=True)
    artifact = Path(module.__file__).resolve()
    receipt.update(status='built', artifact=str(artifact), artifact_sha256=sha(artifact))
except Exception as error:
    receipt.update(status='failed', error=repr(error))
    raise
finally:
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')

"""Controller-owned explicit build of model-supplied isolated C++/CUDA sources."""
import hashlib,json,os
from pathlib import Path
ROOT=Path('/tmp/kda-rmsnorm-stage27');build=ROOT/'cpp-build-006';build.mkdir(exist_ok=False)
os.environ['MAX_JOBS']='1'
os.environ['TORCH_CUDA_ARCH_LIST']='7.0'
import torch
from torch.utils.cpp_extension import load
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'name':'kda_bf16_affine_stage27_006','torch':torch.__version__,'cuda':torch.version.cuda,'arch':'7.0 CUDA compatibility target',
         'sources':{n:sha(ROOT/n) for n in ('affine-006.cpp','affine-006.cu')},'build_script_sha256':sha(Path(__file__)),
         'extra_cflags':['-O3'],'extra_cuda_cflags':['-O3'],'build_directory':str(build)}
try:
 module=load(name=receipt['name'],sources=[str(ROOT/'affine-006.cpp'),str(ROOT/'affine-006.cu')],build_directory=str(build),
             extra_cflags=receipt['extra_cflags'],extra_cuda_cflags=receipt['extra_cuda_cflags'],verbose=True)
 receipt.update(status='built',artifact_sha256=sha(Path(module.__file__)),artifact=str(module.__file__))
except Exception as e:
 receipt.update(status='failed',error=repr(e));raise
finally:(ROOT/'cpp-build-006-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

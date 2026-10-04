import importlib.util
import torch

_MODULE_NAME = "kda_swiglu_stage29_003"
_SO_PATH = "/tmp/kda-swiglu-stage29/cpp-build-003/kda_swiglu_stage29_003.so"

_ext = None


def _extension():
    global _ext
    if _ext is None:
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SO_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ext = module
    return _ext


def _validate(x):
    if not isinstance(x, torch.Tensor):
        raise NotImplementedError("swiglu expects a torch.Tensor")
    if not x.is_cuda:
        raise NotImplementedError("swiglu requires a CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise NotImplementedError("swiglu requires a BF16 tensor")
    if x.dim() not in (2, 3):
        raise NotImplementedError("swiglu requires a 2D or 3D tensor")
    if x.size(-1) % 2 != 0:
        raise NotImplementedError("swiglu requires an even last dimension")
    if not x.is_contiguous():
        raise NotImplementedError("swiglu requires a contiguous tensor")


def swiglu(x):
    _validate(x)
    return _extension().swiglu(x)

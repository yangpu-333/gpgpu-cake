import importlib.util
import os

import torch
from torch.autograd import Function

_BUILD_DIR = "/tmp/kda-swiglu-stage29/cpp-build-002"
_MODULE_NAME = "kda_swiglu_stage29_002"


def _load_extension():
    so_path = os.path.join(_BUILD_DIR, _MODULE_NAME + ".so")
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, so_path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load extension at " + so_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ext = _load_extension()


def _check_input(x):
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


class _SwiGLU(Function):
    @staticmethod
    def forward(ctx, x):
        out = _ext.forward(x)
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        dx = _ext.backward(x, grad_output)
        return dx


def swiglu(x):
    _check_input(x)
    return _SwiGLU.apply(x)

import torch
import kda_bf16_affine_stage27_006 as _ext


def _supported(x, weight, eps):
    if not (x.is_cuda and weight.is_cuda):
        return False
    if x.device != weight.device:
        return False
    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        return False
    if x.numel() == 0 or weight.numel() == 0:
        return False
    if x.dim() < 1 or x.shape[-1] != 512:
        return False
    if weight.dim() != 1 or weight.numel() != 512:
        return False
    if eps != 1e-5:
        return False
    return True


def rms_norm(x, weight, eps):
    if not _supported(x, weight, eps):
        raise NotImplementedError(
            "fused rms_norm supports only nonempty BF16 CUDA tensors on the same "
            "device with hidden=512, 1D weight=512, and eps==1e-5"
        )
    return _ext.rms_norm(x, weight, eps)

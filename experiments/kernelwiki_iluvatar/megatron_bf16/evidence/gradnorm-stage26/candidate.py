import torch

def _supported(tensor_lists, per_tensor):
    if per_tensor:
        return False
    if not isinstance(tensor_lists, (list, tuple)) or len(tensor_lists) != 1:
        return False
    tl = tensor_lists[0]
    if not isinstance(tl, (list, tuple)) or len(tl) == 0:
        return False
    for t in tl:
        if type(t) is not torch.Tensor:
            return False
        if t.dtype is not torch.float32:
            return False
        if not t.is_cuda or t.device.index not in (0, None):
            return False
        if not t.is_contiguous():
            return False
    return True

def local_multi_tensor_l2_norm(chunk_size, noop_flag, tensor_lists, per_tensor, *args, **kwargs):
    native = kwargs.pop("native", None)
    if not _supported(tensor_lists, per_tensor):
        if native is None:
            raise ValueError(
                "local_multi_tensor_l2_norm candidate supports only per_tensor=False with "
                "one nonempty tensor_list of contiguous float32 CUDA:0 tensors; pass "
                "native=<original fn> to fall back for other inputs."
            )
        return native(chunk_size, noop_flag, tensor_lists, per_tensor, *args)

    tensor_list = tensor_lists[0]
    input_device = tensor_list[0].device
    # Preserve every native per-tensor GPU norm call and its order.
    norms = [torch.norm(tensor) for tensor in tensor_list]
    # Stack scalar GPU norms into the same [1, num_tensors] float32 vector native
    # builds, then perform exactly ONE GPU->CPU copy instead of one copy per norm.
    l2_cpu = torch.stack(norms).reshape(1, len(norms)).cpu()
    l2_reduced = torch.norm(l2_cpu)
    l2_cuda = torch.tensor([float(l2_reduced)], dtype=torch.float, device=input_device)
    return l2_cuda, None

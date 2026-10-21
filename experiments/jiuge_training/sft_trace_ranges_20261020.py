"""Symmetric host annotations; return original results without retaining tensors."""
from functools import wraps
import importlib


def install(torch, hdo=None):
    counts = {}
    module = importlib.import_module('torch.optim.adamw')
    original = module.adamw

    @wraps(original)
    def adamw(*args, **kwargs):
        params = args[0] if args else kwargs['params']
        if not params or params[0].device.type != 'cpu':
            return original(*args, **kwargs)
        name = 'jiuge/cpu_adamw_math'
        counts[name] = counts.get(name, 0) + 1
        with torch.profiler.record_function(name):
            return original(*args, **kwargs)

    module.adamw = adamw
    if hdo is not None:
        for method, name in (
            ('_set_sub_optimizer_grads', 'jiuge/offload_d2h_submit'),
            ('_sync_hdo_param_groups_to_sub_optimizers', 'jiuge/offload_group_sync'),
            ('_sync_sub_optimizers_state_to_hdo', 'jiuge/offload_state_sync'),
        ):
            old = getattr(hdo, method)

            def make_wrapper(old, name):
                @wraps(old)
                def wrapped(*args, **kwargs):
                    counts[name] = counts.get(name, 0) + 1
                    with torch.profiler.record_function(name):
                        return old(*args, **kwargs)
                return wrapped
            setattr(hdo, method, make_wrapper(old, name))
    return counts

"""Explicit process-local CoreX 4.2 import compatibility for the local backend.

No installed package or pinned Megatron source is edited. Both native and
optimized launches use the same mappings and optional-backend exclusions.
"""
import importlib.machinery
import os
from pathlib import Path
import sys
import types


def prepare_local_runtime(megatron_root):
    import torch
    if torch.__version__.split('+')[0] != '2.4.1':
        raise RuntimeError('CoreX 4.2 compatibility is only checked for torch2.4.1')
    root = Path(megatron_root).resolve()
    os.environ['USE_TF'] = '0'
    import typing
    if not hasattr(typing, 'override'):
        from typing_extensions import override
        typing.override = override
    import triton
    original_jit = triton.jit
    ignored_launch_metadata = []
    def jit_without_profiler_metadata(*args, **kwargs):
        if 'launch_metadata' in kwargs:
            callback = kwargs.pop('launch_metadata')
            ignored_launch_metadata.append(getattr(callback, '__name__', str(type(callback))))
        return original_jit(*args, **kwargs)
    triton.jit = jit_without_profiler_metadata
    original_autotune = triton.autotune
    blocked_restore_value_kernels = []
    class UnavailableAutotuner:
        def __init__(self, original, name):
            self.original, self.name = original, name
        def __getattr__(self, name):
            return getattr(self.original, name)
        def __getitem__(self, grid):
            raise RuntimeError('CoreX2.1 cannot execute restore_value autotuner: '+self.name)
        def __call__(self, *args, **kwargs):
            raise RuntimeError('CoreX2.1 cannot execute restore_value autotuner: '+self.name)
    def autotune_with_explicit_unsupported_guard(*args, **kwargs):
        if 'restore_value' not in kwargs:
            return original_autotune(*args, **kwargs)
        kwargs.pop('restore_value')
        decorator = original_autotune(*args, **kwargs)
        def reject_execution(function):
            name = getattr(function, '__name__', getattr(function, 'name', 'unknown'))
            blocked_restore_value_kernels.append(name)
            return UnavailableAutotuner(decorator(function), name)
        return reject_execution
    triton.autotune = autotune_with_explicit_unsupported_guard
    if any(name == 'megatron' or name.startswith('megatron.') for name in sys.modules):
        raise RuntimeError('prepare runtime before importing any Megatron module')
    # The pinned checkout has a namespace package while the vendor installation
    # supplies a regular package. Select the requested checkout explicitly.
    namespace = types.ModuleType('megatron')
    namespace.__path__ = [str(root / 'megatron')]
    namespace.__package__ = 'megatron'
    namespace.__spec__ = importlib.machinery.ModuleSpec('megatron', None, is_package=True)
    namespace.__spec__.submodule_search_locations = namespace.__path__
    sys.modules['megatron'] = namespace
    sys.path.insert(0, str(root))
    import torch.distributed._tensor as old_tensor
    import torch.distributed.tensor as new_tensor
    exported = []
    for name in ('DTensor', 'DeviceMesh', 'Partial', 'Placement', 'Replicate',
                 'Shard', 'distribute_module', 'distribute_tensor'):
        if not hasattr(new_tensor, name):
            setattr(new_tensor, name, getattr(old_tensor, name))
            exported.append(name)
    sys.modules.setdefault('torch.distributed.tensor.placement_types', old_tensor.placement_types)
    # This wrapper deliberately selects upstream's PyTorch fallback paths.
    # TE, Apex and their native binaries remain installed and unchanged.
    sys.modules['transformer_engine'] = None
    sys.modules['apex'] = None
    return {'torch': torch.__version__, 'selected_megatron_source': str(root),
            'dtensor_namespace_exports': exported,
            'typing_override': 'typing_extensions.override',
            'ignored_triton_profiler_launch_metadata': ignored_launch_metadata,
            'unavailable_restore_value_autotuners': blocked_restore_value_kernels,
            'optional_backends_disabled_in_process': ['transformer_engine', 'apex']}

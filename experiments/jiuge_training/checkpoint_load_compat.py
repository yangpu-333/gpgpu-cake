"""Preserve Megatron's explicit no_dist load mode on Torch 2.6.

Torch 2.6 load() has no no_dist keyword; its public load_state_dict()
retains that exact mode and the same reader/planner. Native common-state
and sharded loads both use it. Calls without no_dist use the original API.
"""
import inspect


def install():
    import torch
    import torch.distributed.checkpoint as dcp

    original = dcp.load
    if getattr(original, '_jiuge_no_dist_compat', False):
        raise ValueError('Checkpoint compatibility already installed')
    if 'no_dist' in inspect.signature(original).parameters:
        raise ValueError('This adapter is unnecessary for the installed API')
    legacy = dcp.load_state_dict
    if 'no_dist' not in inspect.signature(legacy).parameters:
        raise ValueError('No public API preserves the requested no_dist mode')
    evidence = {'schema': 'jiuge-torch26-local-checkpoint-load-compat-v2',
                'torch_version': torch.__version__, 'redirected_calls': 0,
                'normal_calls': 0, 'common_state_keys': [],
                'common_load_calls': 0, 'sharded_load_calls': 0, 'loads': [],
                'native_files_modified': False}

    def compatible(state_dict, *args, **kwargs):
        if 'no_dist' not in kwargs:
            evidence['normal_calls'] += 1
            return original(state_dict, *args, **kwargs)
        options = dict(kwargs)
        if (len(args) > 1 or type(options.get('no_dist')) is not bool or
                set(options) - {'no_dist', 'storage_reader', 'planner', 'process_group', 'coordinator_rank'} or
                (args and 'storage_reader' in options)):
            raise ValueError('Unsupported checkpoint load signature')
        reader = args[0] if args else options.pop('storage_reader', None)
        if reader is None:
            raise ValueError('Explicit native storage reader required')
        common = len(state_dict) == 1 and next(iter(state_dict)).startswith('common_state/')
        evidence['redirected_calls'] += 1
        evidence['common_load_calls' if common else 'sharded_load_calls'] += 1
        if common:
            evidence['common_state_keys'].append(next(iter(state_dict)))
        evidence['loads'].append({'keys': len(state_dict), 'no_dist': options['no_dist'],
            'reader_type': type(reader).__name__,
            'planner_type': type(options.get('planner')).__name__, 'common_state': common})
        return legacy(state_dict, storage_reader=reader, **options)

    compatible._jiuge_no_dist_compat = True
    dcp.load = compatible
    return evidence

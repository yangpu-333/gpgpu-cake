"""Observe CoreX's public native no_dist loader without redirecting its calls."""
import inspect


def install():
    import torch
    import torch.distributed.checkpoint as dcp
    original = dcp.load
    if 'no_dist' not in inspect.signature(original).parameters:
        raise ValueError('Require the inspected CoreX native no_dist API')
    if getattr(original, '_jiuge_CoreX_native_no_dist_observer', False):
        raise ValueError('Native load observer already installed')
    evidence = {'schema': 'jiuge-CoreX-native-no_dist-load-observer-v1',
                'torch_version': torch.__version__, 'native_signature': str(inspect.signature(original)),
                'native_no_dist_supported': True, 'redirected_calls': 0,
                'normal_calls': 0, 'common_state_keys': [],
                'common_load_calls': 0, 'sharded_load_calls': 0, 'loads': [],
                'native_files_modified': False}

    def observed(state_dict, *args, **kwargs):
        common = len(state_dict) == 1 and next(iter(state_dict)).startswith('common_state/')
        evidence['normal_calls'] += 1
        evidence['common_load_calls' if common else 'sharded_load_calls'] += 1
        if common:
            evidence['common_state_keys'].append(next(iter(state_dict)))
        evidence['loads'].append({'keys': len(state_dict), 'no_dist': kwargs.get('no_dist'),
                                  'native_API_args_passed_unchanged': True})
        return original(state_dict, *args, **kwargs)

    observed._jiuge_CoreX_native_no_dist_observer = True
    dcp.load = observed
    return evidence

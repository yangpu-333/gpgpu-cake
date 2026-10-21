"""Move native HDO placeholder initialization before sharded-state inspection.

DistributedOptimizer.load_state_dict already calls HDO.dummy_step for an
empty state. Its sharded_state_dict inspects state_dict before reaching
that call on this native Torch/CPU-offload path. Use the same native
initializer earlier, exclusively while building a load template.
"""
import functools
import gc


def install(evidence):
    import torch
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer

    original = DistributedOptimizer.sharded_state_dict
    evidence['hdo_placeholder_initializations'] = []
    evidence['hdo_step_restorations'] = []

    @functools.wraps(original)
    def sharded(outer, *args, **kwargs):
        loading = kwargs.get('is_loading', args[1] if len(args) > 1 else False)
        inner = outer.optimizer
        if loading and isinstance(inner, HybridDeviceOptimizer) and not inner.state:
            # Constructor-created optimizer/hook cycles are obsolete here.
            # Collect only in this untimed loading path, before large states.
            gc.collect()
            expected = sum(len(g['params']) for g in inner.param_groups)
            inner.dummy_step()
            steps = {float(s['step'].item()) for s in inner.state.values()}
            if len(inner.state) != expected or steps != {1.0}:
                raise ValueError('Native HDO dummy initialization did not cover every parameter')
            evidence['hdo_placeholder_initializations'].append({
                'state_parameters': len(inner.state), 'expected_parameters': expected,
                'placeholder_step': 1, 'native_dummy_step_used': True,
                'real_checkpoint_values_still_required': True})
        return original(outer, *args, **kwargs)

    DistributedOptimizer.sharded_state_dict = sharded

    original_load = DistributedOptimizer.load_state_dict

    @functools.wraps(original_load)
    def load(outer, saved):
        inner = outer.optimizer
        groups = saved.get('optimizer', {}).get('param_groups', [])
        source_steps = [float(g['step'].item()) if torch.is_tensor(g['step']) else float(g['step'])
                        for g in groups if 'step' in g]
        result = original_load(outer, saved)
        if isinstance(inner, HybridDeviceOptimizer):
            if (not source_steps or len(set(source_steps)) != 1 or
                    inner.offload_fraction != 1.0):
                raise ValueError('Uniform saved AdamW steps and full CPU offload required')
            # Bucket templates contain LocalNonpersistentObject step placeholders.
            # They must not replace the persistent saved group step. The native
            # no-TE branch also aliases one scalar across every parameter. Each
            # independent CPU AdamW instance needs its own restored scalar.
            for state in inner.state.values():
                state['step'] = torch.tensor(source_steps[0], dtype=torch.float32, device='cpu')
            inner._sync_hdo_state_to_sub_optimizers()
            pointers = {state['step'].data_ptr() for state in inner.state.values()}
            if len(pointers) != len(inner.state):
                raise ValueError('Restored AdamW step scalars still alias')
            evidence['hdo_step_restorations'].append({
                'source_group_steps': source_steps, 'state_parameters': len(inner.state),
                'independent_step_scalars': len(pointers),
                'source': 'actual optimizer param_groups provided by native checkpoint loader',
                'inferred_from_iteration': False})
            gc.collect()
        return result

    DistributedOptimizer.load_state_dict = load

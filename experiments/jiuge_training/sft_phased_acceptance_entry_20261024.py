"""Use the screened placement/math with untimed audit or native formal backends."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import weakref

from sft_phased_acceptance_20261024 import PREP, POLICIES, pinned_backend, verify_sources


def install_steps(torch, hdo, plan, mode, policy, root, rank, formal):
    from sft_phased_placement_20261023 import activate, install_physical_guard
    import sft_local_moments_v4_20261023 as moments
    from sft_moment_memory_guard_20261021 import install as install_guard
    from sft_memory_snapshot_compat_20261023 import install as optional_peak
    optional_peak()
    original = hdo.step
    state = {'steps': 0, 'optimizer': None, 'events': [], 'binding': None, 'pages': None}
    undo = []

    def step(opt, *args, **kwargs):
        if state['steps'] == 0:
            state['binding'] = activate(plan, mode, rank, root)
            if policy == 'local-moments':
                undo.append(install_guard(moments))
                undo.append(install_physical_guard(moments, state['binding']['selected_cpu_nodes'][0]))
                undo.append(moments.install(torch, state['events']))
        value = original(opt, *args, **kwargs)
        state['steps'] += 1
        state['optimizer'] = weakref.ref(opt)
        return value
    hdo.step = step

    def finish():
        # No per-step page walks, tensor reads, file IO or extra sync in formal timing.
        # Initialization/NUMA guard runs in the excluded warmup; post-run proofs follow.
        opt = state['optimizer']() if state['optimizer'] else None
        result = {'policy': policy, 'rank': rank, 'steps': state['steps'],
                  'binding': state['binding'], 'moment_events': state['events'],
                  'formal_extra_per_step_observer': False, 'optimizer_strong_reference': False}
        if state['pages'] is not None:
            result['final_pages'] = state['pages']
        with (root/f'phased-policy-rank-{rank}.json').open('x') as stream:
            json.dump(result, stream, indent=2)
        hdo.step = original
        for function in reversed(undo):
            function()
        return result
    return state, finish


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--phased-plan', type=Path, required=True)
    parser.add_argument('--phased-policy', choices=tuple(POLICIES.values()), required=True)
    args, rest = parser.parse_known_args()
    arm = rest[rest.index('--pinning-arm')+1]
    stage = rest[rest.index('--sft-pinning-mode')+1]
    if POLICIES[arm] != args.phased_policy or stage not in ('audit', 'resume', 'formal'):
        raise ValueError('Exact acceptance role/policy/stage required')
    if '--capture-windows' in rest or '--capture-inputs' in rest:
        raise ValueError('Acceptance entry is not a profiling entry')
    root = Path(rest[rest.index('--jiuge-run-root')+1])
    rank = int(os.environ['RANK'])
    mode = 'unbound' if arm == 'native' else 'gpu-local8'
    from sft_phased_placement_20261023 import prepare
    setup = prepare(args.phased_plan, mode, rank)
    with (root/f'sft-numa-setup-binding-rank-{rank}.json').open('x') as stream:
        json.dump(setup, stream, indent=2)
    files = verify_sources()
    sys.path.insert(0, str(PREP/'megatron-native-5be9626'))
    import torch
    import megatron.core.optimizer.cpu_offloading.hybrid_optimizer as module
    state, finish = install_steps(torch, module.HybridDeviceOptimizer, args.phased_plan,
                              mode, args.phased_policy, root, rank, stage == 'formal')
    import optimizer_evidence
    save_optimizer = optimizer_evidence.save_optimizer_evidence
    def save_after_training(opt, path):
        result = save_optimizer(opt, path)
        from numa_placement import observe
        from pinning_resume_audit import optimizer_leaves
        leaves = [x for x in optimizer_leaves(opt) if isinstance(x, module.HybridDeviceOptimizer)]
        if len(leaves) != 1:
            raise ValueError('Exactly one final HDO object required')
        state['pages'] = observe(leaves[0], 'after-returned-training', torch)
        return result
    # Both frozen backends call this only after pretrain_gpt returns, outside timers.
    optimizer_evidence.save_optimizer_evidence = save_after_training
    original_run = runpy.run_path

    def dispatch(path, init_globals=None, run_name=None):
        name = Path(path).name
        if Path(path) in (PREP/'pinning_entry.py', PREP/'pinning_formal_entry.py'):
            source = PREP/name
            if hashlib.sha256(source.read_bytes()).hexdigest() != files[name]:
                raise ValueError('Untouched pinning backend identity differs')
            text = pinned_backend(source.read_bytes(), name == 'pinning_formal_entry.py')
            namespace = {'__name__': run_name or '<derived>', '__file__': str(source),
                         '__package__': None, **(init_globals or {})}
            exec(compile(text, '<incremental-pinned-'+name+'>', 'exec'), namespace)
            return namespace
        return original_run(path, init_globals=init_globals, run_name=run_name)

    runpy.run_path = dispatch
    sys.argv = [str(PREP/'sft_pinning_entry.py'), *rest]
    try:
        original_run(str(PREP/'sft_pinning_entry.py'), run_name='__main__')
    finally:
        runpy.run_path = original_run
        optimizer_evidence.save_optimizer_evidence = save_optimizer
        finish()


if __name__ == '__main__':
    main()

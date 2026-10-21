"""Apply affinity before Torch; observe actual buffers outside trace windows."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import weakref
from sft_numa_placement_20261021 import bind

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--sft-numa-mode', required=True)
    parser.add_argument('--sft-numa-plan', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if '--capture-windows' not in remaining or '--pinning-arm' not in remaining:
        raise ValueError('Separate SFT profile with accepted pinned repair required')
    if remaining[remaining.index('--pinning-arm')+1] != 'optimized':
        raise ValueError('All placement arms must use the accepted pinned repair')
    root = Path(remaining[remaining.index('--jiuge-run-root')+1])
    rank = int(os.environ['RANK'])
    receipt = bind(args.sft_numa_plan, args.sft_numa_mode, rank)
    with (root/f'sft-numa-binding-rank-{rank}.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    # Importing Torch only after all affinity checks preserves thread inheritance.
    sys.path.insert(0, str(PREP/'megatron-native-5be9626'))
    import torch
    import megatron.core.optimizer.cpu_offloading.hybrid_optimizer as module
    from numa_placement import observe
    old_step = module.HybridDeviceOptimizer.step
    active = None
    steps = 0

    def snapshot(opt, phase):
        value = observe(opt, phase, torch)
        value.update(rank=rank, mode=args.sft_numa_mode,
                     assigned_gpu=receipt['assigned_gpu'],
                     binding_plan_sha256=receipt['plan_sha256'])
        if args.sft_numa_mode != 'unbound':
            expected = set(receipt['allowed_after'])
            if any(not set(cpus).issubset(expected) for cpus in value['thread_affinity'].values()):
                raise ValueError('A worker thread escaped its assigned physical cores')
        with (root/f'sft-numa-{phase}-rank-{rank}.json').open('x') as stream:
            json.dump(value, stream, indent=2)

    def stepped(opt, *values, **keywords):
        nonlocal active, steps
        result = old_step(opt, *values, **keywords)
        active = weakref.ref(opt)  # Never extend lifetime of a replaced optimizer.
        steps += 1
        if steps == 1:
            snapshot(opt, 'after-first-update')
        return result

    module.HybridDeviceOptimizer.step = stepped
    sys.argv = [str(PREP/'sft_trace_entry_20261020.py'), *remaining]
    runpy.run_path(str(PREP/'sft_trace_entry_20261020.py'), run_name='__main__')
    optimizer = active() if active is not None else None
    if optimizer is None or steps != 35:
        raise ValueError('All35 real updates and a live optimizer required')
    snapshot(optimizer, 'after-training')
    with (root/f'sft-numa-observer-rank-{rank}.json').open('x') as stream:
        json.dump({'steps': steps, 'owns_optimizer_strong_reference_during_training': False,
                   'observations_inside_profile_windows': False,
                   'formal_performance_evidence': False}, stream, indent=2)


if __name__ == '__main__':
    main()

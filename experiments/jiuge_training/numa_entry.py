"""Apply one rank-local affinity policy before the unchanged pinning entry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from numa_placement import bind, observe

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--numa-mode', choices=('unbound', 'local8'), required=True)
    parser.add_argument('--numa-root', type=Path, required=True)
    parser.add_argument('--numa-observe', action='store_true')
    parser.add_argument('--numa-backend', choices=('pinning_entry.py', 'pinning_formal_entry.py',
                        'pinning_semantics_entry.py', 'pinning_resume_compat_entry_v2.py'), required=True)
    args, remaining = parser.parse_known_args()
    rank = int(os.environ['RANK'])
    receipt = bind(args.numa_mode, rank)
    receipt['source_sha256'] = {name: hashlib.sha256((PREP / name).read_bytes()).hexdigest()
                                for name in ('numa_entry.py', 'numa_placement.py')}
    receipt_path = args.numa_root / f'numa-binding-rank-{rank}.json'
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
    optimizers = []
    if args.numa_observe:
        sys.path.insert(0, str(PREP / 'megatron-native-5be9626'))
        import torch
        import megatron.core.optimizer.cpu_offloading.hybrid_optimizer as module
        hdo = module.HybridDeviceOptimizer
        original_init, original_step = hdo._init_sub_optimizers, hdo.step
        def snapshot(opt, phase):
            with (args.numa_root / f'numa-{phase}-rank-{rank}.json').open('x') as stream:
                json.dump(observe(opt, phase, torch), stream, indent=2)
        def initialized(opt):
            result = original_init(opt)
            optimizers.append(opt)
            snapshot(opt, f'after-init-{len(optimizers)}')
            return result
        steps = 0
        active_optimizer = None
        def stepped(opt, *values, **keywords):
            nonlocal steps, active_optimizer
            result = original_step(opt, *values, **keywords)
            active_optimizer = opt
            steps += 1
            if steps == 1:
                snapshot(opt, 'after-first-update')
            return result
        hdo._init_sub_optimizers, hdo.step = initialized, stepped
    sys.argv = [str(PREP / args.numa_backend), *remaining]
    runpy.run_path(str(PREP / args.numa_backend), run_name='__main__')
    if args.numa_observe:
        if active_optimizer is None:
            raise ValueError('Actual HDO training updates required for placement observations')
        snapshot(active_optimizer, 'after-training')


if __name__ == '__main__':
    main()

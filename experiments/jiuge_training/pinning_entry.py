"""Isolated pinning experiment; call the unchanged native entry and HDO source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from pinning_candidate import EXPECTED, SOURCE, load_class

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pinning-arm', choices=('native', 'control', 'optimized'), required=True)
    parser.add_argument('--pinning-campaign', type=Path, required=True)
    parser.add_argument('--full-state-audit', action='store_true')
    args, remaining = parser.parse_known_args()
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError('Native source modified')
    sys.path.insert(0, str(PREP / 'megatron-native-5be9626'))
    import torch
    import megatron.core.optimizer.cpu_offloading.hybrid_optimizer as module
    hdo = module.HybridDeviceOptimizer
    if args.pinning_arm in ('control', 'optimized'):
        name = 'candidate_hybrid_optimizer.py' if args.pinning_arm == 'optimized' else 'native_hybrid_optimizer.py'
        expected = json.loads((args.pinning_campaign / 'candidate.json').read_text())['candidate_sha256'] if args.pinning_arm == 'optimized' else EXPECTED
        if hashlib.sha256((args.pinning_campaign / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Isolated HDO source SHA256 differs')
        candidate = load_class(args.pinning_campaign / name, 'jiuge_hdo_injected')
        hdo._get_sub_optimizer_param_groups = candidate._get_sub_optimizer_param_groups
    rank = int(os.environ['RANK'])
    init = hdo._init_sub_optimizers
    init_count = 0
    def checked_init(opt):
        nonlocal init_count
        init_count += 1
        result = init(opt)
        values = list(opt.gpu_params_map_cpu_copy.values())
        receipt = {'rank': rank, 'arm': args.pinning_arm, 'parameter_count': len(values),
                   'elements': sum(p.numel() for p in values),
                   'pinned_count': sum(p.is_pinned() for p in values),
                   'pinned_elements': sum(p.numel() for p in values if p.is_pinned()),
                   'dtype_counts': {str(t): sum(p.dtype == t for p in values) for t in {p.dtype for p in values}},
                   'native_source_sha256': EXPECTED, 'formal_performance_passed': False}
        path = args.pinning_campaign / f'{args.pinning_arm}-rank-{rank}-actual-pinning-init-{init_count}.json'
        with path.open('x') as stream:
            json.dump(receipt, stream, indent=2)
        if args.pinning_arm == 'optimized' and receipt['pinned_count'] != len(values):
            raise AssertionError('Actual CPU master pinning incomplete')
        return result
    hdo._init_sub_optimizers = checked_init
    audit = None
    if args.full_state_audit:
        import megatron.training.training as training
        from pinning_full_audit import install
        audit = install(args.pinning_campaign, args.pinning_arm, rank, hdo, training)
    sys.argv = [str(PREP / 'native_entry.py'), *remaining]
    try:
        runpy.run_path(str(PREP / 'native_entry.py'), run_name='__main__')
    finally:
        if audit:
            audit.close()
    if audit:
        audit.flush()
        fields = {r['field'] for r in audit.rows}
        complete = audit.step == 3 and sum('/gradient' in key for key in fields) == 597
        if audit.reference:
            complete &= fields == set(audit.reference)
        passed = complete and all(row['passed'] for row in audit.rows)
        receipt = {'arm': args.pinning_arm, 'rank': rank, 'steps': audit.step,
                   'complete_fields': complete, 'passed': passed,
                   'local_raw_copy_verified': False, 'checkpoint_resume_tested': False,
                   'formal_performance_passed': False}
        (audit.folder / 'completion.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if not passed:
            raise AssertionError('Full-state numerical audit incomplete or failed')


if __name__ == '__main__':
    main()

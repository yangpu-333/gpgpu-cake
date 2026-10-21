"""Derive an untimed affinity diagnostic from the frozen native case runner."""
import hashlib
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    payload = (PREP / 'run_native_case.py').read_bytes()
    if hashlib.sha256(payload).hexdigest() != '82c444f9ba77e70562430f51a17357179f39acb58d639a4d7b919719930bc345':
        raise ValueError('Frozen runner changed')
    text = payload.decode()
    changes = [
        ("parser.add_argument('--case',required=True)", """parser.add_argument('--case',required=True)
    parser.add_argument('--numa-mode',choices=('unbound','local8'),required=True)
    parser.add_argument('--pinning-campaign',type=Path,required=True)
    parser.add_argument('--full-state-audit',action='store_true')"""),
        ("str(PREP/'native_entry.py'),'--evidence'", "str(PREP/'numa_entry.py'),'--evidence'"),
        ("    if args.reference:\n", """    command += ['--numa-mode',args.numa_mode,'--numa-root',str(folder),'--numa-observe',
                '--numa-backend','pinning_entry.py','--pinning-arm','optimized',
                '--pinning-campaign',str(args.pinning_campaign)]
    if args.full_state_audit:
        command += ['--full-state-audit']
    if args.reference:
"""),
        ("    snapshots=folder/'source'", "    record.update(numa_mode=args.numa_mode,pinning_arm='optimized',pinning_campaign=str(args.pinning_campaign),full_state_audit=args.full_state_audit)\n    snapshots=folder/'source'"),
        ("'optimizer_evidence.py','input_evidence.py'):", "'optimizer_evidence.py','input_evidence.py','pinning_entry.py','pinning_candidate.py','pinning_full_audit.py','numa_placement.py','numa_entry.py','run_numa_case.py'):"),
    ]
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError('Frozen runner derivation no longer exact: ' + before)
        text = text.replace(before, after)
    exec(compile(text, str(PREP / 'run_numa_case.generated.py'), 'exec'),
         {'__name__': '__main__', '__file__': __file__})


if __name__ == '__main__':
    main()

"""Derive an isolated launch from the frozen native case runner, without edits."""
import hashlib
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
EXPECTED_RUNNER = '82c444f9ba77e70562430f51a17357179f39acb58d639a4d7b919719930bc345'


def main():
    source = (PREP / 'run_native_case.py').read_bytes()
    if hashlib.sha256(source).hexdigest() != EXPECTED_RUNNER:
        raise ValueError('Native runner changed')
    text = source.decode()
    changes = [
        ("parser.add_argument('--case',required=True)",
         "parser.add_argument('--case',required=True)\n    parser.add_argument('--pinning-arm',choices=('native','control','optimized'),required=True)\n    parser.add_argument('--pinning-campaign',type=Path,required=True)\n    parser.add_argument('--full-state-audit',action='store_true')"),
        ("str(PREP/'native_entry.py'),'--evidence'", "str(PREP/'pinning_entry.py'),'--evidence'"),
        ("    if args.reference:\n", "    command += ['--pinning-arm',args.pinning_arm,'--pinning-campaign',str(args.pinning_campaign)]\n    if args.full_state_audit:\n        command += ['--full-state-audit']\n    if args.reference:\n"),
        ("    snapshots=folder/'source'", "    record.update(pinning_arm=args.pinning_arm,pinning_campaign=str(args.pinning_campaign),full_state_audit=args.full_state_audit)\n    snapshots=folder/'source'"),
        ("'optimizer_evidence.py','input_evidence.py'):",
         "'optimizer_evidence.py','input_evidence.py','run_pinning_case.py','pinning_entry.py','pinning_candidate.py','pinning_full_audit.py'):"),
    ]
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError('Native runner derivation no longer exact: ' + before)
        text = text.replace(before, after)
    exec(compile(text, str(PREP / 'run_pinning_case.generated.py'), 'exec'),
         {'__name__': '__main__', '__file__': str(PREP / 'run_pinning_case.py')})


if __name__ == '__main__':
    main()

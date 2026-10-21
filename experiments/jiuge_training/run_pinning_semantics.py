"""Derive an untimed semantics/checkpoint launch without editing native files."""
import hashlib
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
EXPECTED_RUNNER = '82c444f9ba77e70562430f51a17357179f39acb58d639a4d7b919719930bc345'


def main():
    original = (PREP / 'run_native_case.py').read_bytes()
    if hashlib.sha256(original).hexdigest() != EXPECTED_RUNNER:
        raise ValueError('Native runner changed')
    text = original.decode()
    changes = [
        ("parser.add_argument('--case',required=True)", """parser.add_argument('--case',required=True)
    parser.add_argument('--pinning-arm',choices=('native','control','optimized'),required=True)
    parser.add_argument('--pinning-campaign',type=Path,required=True)
    parser.add_argument('--semantics-root',type=Path,required=True)
    parser.add_argument('--save-dir',type=Path)
    parser.add_argument('--load-dir',type=Path)
    parser.add_argument('--resume-audit-campaign',type=Path)
    parser.add_argument('--exit-interval',type=int)"""),
        ("str(PREP/'native_entry.py'),'--evidence'", "str(PREP/'pinning_semantics_entry.py'),'--evidence'"),
        ("    if args.reference:\n", """    command += ['--pinning-arm',args.pinning_arm,'--pinning-campaign',str(args.pinning_campaign),
                '--semantics-root',str(args.semantics_root)]
    if args.save_dir:
        argv += ['--save',str(args.save_dir),'--save-interval','2','--ckpt-format','torch_dist']
    if args.load_dir:
        argv += ['--load',str(args.load_dir),'--ckpt-format','torch_dist']
    if args.resume_audit_campaign:
        if not args.load_dir or args.pinning_arm != 'optimized':
            raise ValueError('Full resume audit requires optimized arm and native checkpoint load')
        command += ['--resume-audit-campaign',str(args.resume_audit_campaign)]
    if args.exit_interval:
        if args.exit_interval < 1:
            raise ValueError('Positive exit interval required')
        argv += ['--exit-interval',str(args.exit_interval)]
    if args.reference:
"""),
        ("    snapshots=folder/'source'", "    record.update(pinning_arm=args.pinning_arm,semantics_root=str(args.semantics_root),untimed_semantics_or_checkpoint_run=True)\n    snapshots=folder/'source'"),
        ("'optimizer_evidence.py','input_evidence.py'):", "'optimizer_evidence.py','input_evidence.py','pinning_entry.py','pinning_candidate.py','pinning_semantics_entry.py','pinning_resume_audit.py','pinning_full_audit.py','run_pinning_semantics.py'):"),
    ]
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError('Native launch derivation no longer exact')
        text = text.replace(before, after)
    exec(compile(text, str(PREP / 'run_pinning_semantics.generated.py'), 'exec'),
         {'__name__': '__main__', '__file__': str(PREP / 'run_pinning_semantics.py')})


if __name__ == '__main__':
    main()

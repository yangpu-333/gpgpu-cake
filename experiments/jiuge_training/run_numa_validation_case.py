"""Independent full-state+semantics or checkpoint-resume affinity validation."""
import hashlib
from pathlib import Path

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    payload=(PREP/'run_native_case.py').read_bytes()
    if hashlib.sha256(payload).hexdigest()!='82c444f9ba77e70562430f51a17357179f39acb58d639a4d7b919719930bc345':
        raise ValueError('Native runner changed')
    text=payload.decode()
    changes=[
      ("parser.add_argument('--case',required=True)","""parser.add_argument('--case',required=True)
    parser.add_argument('--numa-mode',choices=('unbound','local8'),required=True)
    parser.add_argument('--validation-stage',choices=('audit','resume'),required=True)
    parser.add_argument('--pinning-campaign',type=Path,required=True)
    parser.add_argument('--semantics-root',type=Path,required=True)
    parser.add_argument('--load-dir',type=Path)
    parser.add_argument('--resume-audit-campaign',type=Path)"""),
      ("str(PREP/'native_entry.py'),'--evidence'","str(PREP/'numa_entry.py'),'--evidence'"),
      ("    if args.reference:\n","""    if args.steps!=3 or args.seq_length!=2048 or args.capture_windows or args.capture_inputs or args.reference:
        raise ValueError('Frozen full-state/semantic configuration required')
    backend='pinning_semantics_entry.py' if args.validation_stage=='audit' else 'pinning_resume_compat_entry_v2.py'
    command += ['--numa-mode',args.numa_mode,'--numa-root',str(folder),'--numa-observe',
                '--numa-backend',backend,'--pinning-arm','optimized','--pinning-campaign',str(args.pinning_campaign),
                '--semantics-root',str(args.semantics_root)]
    if args.validation_stage=='audit':
        if args.load_dir or args.resume_audit_campaign:
            raise ValueError('Three continuous audit steps required')
        command += ['--full-state-audit']
    else:
        if not args.load_dir or not args.resume_audit_campaign:
            raise ValueError('Same actual checkpoint and full resume audit required')
        argv += ['--load',str(args.load_dir),'--ckpt-format','torch_dist']
        command += ['--resume-audit-campaign',str(args.resume_audit_campaign),
                    '--checkpoint-compat-receipt',str(folder/'checkpoint-compat-rank-{rank}.json')]
    if args.reference:
"""),
      ("    snapshots=folder/'source'","    record.update(numa_mode=args.numa_mode,validation_stage=args.validation_stage,pinning_arm='optimized',pinning_campaign=str(args.pinning_campaign),semantics_root=str(args.semantics_root))\n    snapshots=folder/'source'"),
      ("'optimizer_evidence.py','input_evidence.py'):","'optimizer_evidence.py','input_evidence.py','numa_entry.py','numa_placement.py','pinning_entry.py','pinning_candidate.py','pinning_full_audit.py','pinning_semantics_entry.py','pinning_resume_audit.py','pinning_resume_compat_entry_v2.py','checkpoint_load_compat.py','checkpoint_hdo_load_compat.py','run_numa_validation_case.py'):"),
    ]
    for before,after in changes:
        if text.count(before)!=1:
            raise ValueError('Validation derivation no longer exact: '+before)
        text=text.replace(before,after)
    exec(compile(text,str(PREP/'run_numa_validation_case.generated.py'),'exec'),{'__name__':'__main__','__file__':__file__})


if __name__=='__main__':
    main()

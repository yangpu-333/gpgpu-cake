"""Same fixed 120-step timer; all arms keep accepted pinning repair."""
import hashlib
from pathlib import Path
from numa_admission import check

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    payload=(PREP/'run_native_case.py').read_bytes()
    if hashlib.sha256(payload).hexdigest()!='82c444f9ba77e70562430f51a17357179f39acb58d639a4d7b919719930bc345':
        raise ValueError('Native runner changed')
    text=payload.decode()
    changes=[
      ("parser.add_argument('--case',required=True)","""parser.add_argument('--case',required=True)
    parser.add_argument('--numa-arm',choices=('native','control','optimized'),required=True)
    parser.add_argument('--pinning-campaign',type=Path,required=True)
    parser.add_argument('--admission',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=(63100,63101,63102),required=True)"""),
      ("    folder=PREP/'runs'/args.case","""    check(args.admission)
    if args.steps!=120 or args.seq_length!=2048 or args.capture_inputs or args.capture_windows or args.reference:
        raise ValueError('Fixed 120-step formal contract required')
    folder=PREP/'runs'/args.case"""),
      ("str(PREP/'native_entry.py'),'--evidence'","str(PREP/('pinning_formal_entry.py' if args.numa_arm=='native' else 'numa_entry.py')),'--evidence'"),
      ("    if args.reference:\n","""    command += ['--pinning-arm','optimized','--pinning-campaign',str(args.pinning_campaign)]
    if args.numa_arm!='native':
        command += ['--numa-mode','local8' if args.numa_arm=='optimized' else 'unbound',
                    '--numa-root',str(folder),'--numa-backend','pinning_formal_entry.py']
    if args.reference:
"""),
      ("'--seed','63100'","'--seed',str(args.seed)"),
      ("'--log-interval','1'","'--log-interval','10'"),
      ("'formal_performance_run':False","'formal_performance_run':True"),
      ("capture_output=True,text=True).stdout","capture_output=True,text=True,check=True).stdout"),
      ("    snapshots=folder/'source'","    record.update(numa_arm=args.numa_arm,pinning_arm='optimized',seed=args.seed,admission_sha256=hashlib.sha256(args.admission.read_bytes()).hexdigest())\n    snapshots=folder/'source'"),
      ("'optimizer_evidence.py','input_evidence.py'):","'optimizer_evidence.py','input_evidence.py','numa_entry.py','numa_placement.py','numa_admission.py','pinning_formal_entry.py','pinning_candidate.py','pinning_resume_audit.py','pinning_full_audit.py','run_numa_formal.py'):"),
    ]
    for before,after in changes:
        if text.count(before)!=1:
            raise ValueError('Formal derivation no longer exact: '+before)
        text=text.replace(before,after)
    exec(compile(text,str(PREP/'run_numa_formal.generated.py'),'exec'),{'__name__':'__main__','__file__':__file__,'check':check})


if __name__=='__main__':
    main()

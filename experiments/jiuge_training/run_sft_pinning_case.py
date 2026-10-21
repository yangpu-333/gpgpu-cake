"""Isolated SFT cases derived exactly from the frozen native case runner."""
import hashlib
import json
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
RUNNER_SHA = '82c444f9ba77e70562430f51a17357179f39acb58d639a4d7b919719930bc345'


def check(args):
    if args.reference or args.capture_inputs or args.capture_windows or args.seq_length != 2048:
        raise ValueError('Fixed S2048 SFT validation; no CPT reference/profiler/input capture in native entry')
    receipt = json.loads((PREP/'datasets/alpaca-sft-answer-v1-receipt.json').read_text())
    if not receipt['passed'] or not receipt['all_source_records_retained']:
        raise ValueError('Complete verified SFT data required')
    for path, expected in receipt['files_sha256'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('SFT data changed')
    if args.mode == 'formal':
        from run_pinning_formal import check_admission
        admission = check_admission(args.admission)
        if (args.steps != 120 or args.save_dir or args.load_dir or args.resume_audit_campaign
            or admission.get('objective') != 'answer_only_SFT'
            or admission.get('sft_data_receipt_sha256') != hashlib.sha256((PREP/'datasets/alpaca-sft-answer-v1-receipt.json').read_bytes()).hexdigest()):
            raise ValueError('SFT-bound complete admission and uncontaminated 120-step formal run required')
    elif args.steps != 3 or not args.semantics_root:
        raise ValueError('Untimed three-step validation and semantic originals required')
    if args.mode == 'resume' and not args.load_dir:
        raise ValueError('Resume must load the original SFT checkpoint')
    if args.mode == 'checkpoint' and (not args.save_dir or args.pinning_arm != 'native'):
        raise ValueError('Only the original native SFT checkpoint may be created')


def additions(args, folder, argv):
    extra = ['--sft-pinning-mode',args.mode,'--pinning-arm',args.pinning_arm,
             '--pinning-campaign',str(args.pinning_campaign)]
    if args.mode != 'formal':
        extra += ['--semantics-root',str(args.semantics_root)]
    if args.mode == 'audit':
        extra += ['--full-state-audit']
    if args.mode == 'checkpoint':
        argv += ['--save',str(args.save_dir),'--save-interval','2','--ckpt-format','torch_dist','--exit-interval','2']
    if args.mode == 'resume':
        argv += ['--load',str(args.load_dir),'--ckpt-format','torch_dist']
        extra += ['--checkpoint-compat-receipt',str(folder/'rank-{rank}/checkpoint-compat.json')]
        if args.resume_audit_campaign:
            if args.pinning_arm != 'optimized':
                raise ValueError('Numerical resume observer belongs to the optimized arm')
            extra += ['--resume-audit-campaign',str(args.resume_audit_campaign)]
    return extra


def main():
    payload = (PREP/'run_native_case.py').read_bytes()
    if hashlib.sha256(payload).hexdigest() != RUNNER_SHA:
        raise ValueError('Frozen native runner changed')
    text = payload.decode()
    replacements = [
        ("parser.add_argument('--case',required=True)", """parser.add_argument('--case',required=True)
    parser.add_argument('--mode',choices=('audit','checkpoint','resume','formal'),required=True)
    parser.add_argument('--pinning-arm',choices=('native','control','optimized'),required=True)
    parser.add_argument('--pinning-campaign',type=Path,required=True)
    parser.add_argument('--semantics-root',type=Path)
    parser.add_argument('--save-dir',type=Path)
    parser.add_argument('--load-dir',type=Path)
    parser.add_argument('--resume-audit-campaign',type=Path)
    parser.add_argument('--admission',type=Path)
    parser.add_argument('--seed',type=int,choices=(63100,63101,63102),default=63100)"""),
        ("    folder=PREP/'runs'/args.case", "    check(args)\n    folder=PREP/'runs'/args.case"),
        ("str(PREP/'native_entry.py'),'--evidence'", "str(PREP/'sft_pinning_entry.py'),'--evidence'"),
        ("str(PREP/'datasets/alpaca-cpt-v1_text_document')", "str(PREP/'datasets/alpaca-sft-answer-v1')"),
        ("    if args.reference:\n", "    command += additions(args,folder,argv)\n    if args.mode == 'formal':\n        argv[argv.index('--log-interval')+1] = '10'\n    if args.reference:\n"),
        ("'--seed','63100'", "'--seed',str(args.seed)"),
        ("'formal_performance_run':False", "'formal_performance_run':args.mode == 'formal'"),
        ("    snapshots=folder/'source'", "    record.update(objective='answer_only_SFT',pinning_arm=args.pinning_arm,seed=args.seed,mode=args.mode,pinning_campaign=str(args.pinning_campaign))\n    snapshots=folder/'source'"),
        ("    # This survives SIGKILL/OOM", """    for name in ('sft_pinning_entry.py','run_sft_pinning_case.py','sft_dataset_adapter.py',
                 'pinning_semantics_entry.py','pinning_entry.py','pinning_candidate.py',
                 'pinning_full_audit.py','pinning_resume_audit.py','pinning_formal_entry.py',
                 'pinning_resume_compat_entry_v2.py','checkpoint_load_compat.py','checkpoint_hdo_load_compat.py'):
        shutil.copyfile(PREP/name,snapshots/name)
    # This survives SIGKILL/OOM"""),
        ("stdout=stdout,stderr=stderr,timeout=7200", "stdout=stdout,stderr=stderr,timeout=14400 if args.mode == 'audit' else 7200"),
        ("capture_output=True,text=True).stdout", "capture_output=True,text=True,check=True).stdout"),
    ]
    for before, after in replacements:
        if text.count(before) != 1:
            raise ValueError('Exact native derivation failed: '+before)
        text = text.replace(before, after)
    exec(compile(text,str(PREP/'run_sft_pinning_case.generated.py'),'exec'),
         {'__name__':'__main__','__file__':__file__,'check':check,'additions':additions})


if __name__ == '__main__':
    main()

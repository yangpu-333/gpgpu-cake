"""Launch only a correctness-admitted fixed 120-step formal measurement."""
import hashlib
import json
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
EXPECTED_RUNNER = '82c444f9ba77e70562430f51a17357179f39acb58d639a4d7b919719930bc345'


def check_admission(path):
    receipt = json.loads(path.read_text())
    required = ('passed', 'full_numeric_comparison_verified', 'primary_complete_raw_values_verified',
                'full_model_native_checkpoint_resume_verified', 'inputs_masks_rng_scheduler_verified',
                'local_key_results_verified')
    if receipt.get('schema') != 'jiuge7b-pinning-formal-admission-v1' or not all(receipt.get(k) is True for k in required):
        raise ValueError('Complete primary/raw, semantic, resume and local key-result gates required')
    if receipt.get('candidate_sha256') != '8c83ed05b1333b6775abea19bb2ff07f5966938764654ab133613fd52573ca88':
        raise ValueError('Candidate changed after admission')
    if len(receipt.get('verified_receipts', [])) != 3:
        raise ValueError('Three concrete numeric, semantic and local verification receipts required')
    schemas = set()
    for row in receipt['verified_receipts']:
        if hashlib.sha256(Path(row['file']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Admission evidence changed')
        evidence = json.loads(Path(row['file']).read_text())
        if evidence.get('passed') is not True:
            raise ValueError('Underlying admission evidence did not pass')
        schemas.add(evidence['schema'])
    if schemas != {'jiuge7b-pinning-full-archive-verification-v1', 'jiuge7b-pinning-semantics-verification-v1',
                   'jiuge-pinning-local-key-verification-v1'}:
        raise ValueError('Concrete full-value, semantic/resume and local key evidence required')
    return receipt


def main():
    original = (PREP / 'run_native_case.py').read_bytes()
    if hashlib.sha256(original).hexdigest() != EXPECTED_RUNNER:
        raise ValueError('Frozen runner changed')
    text = original.decode()
    changes = [
        ("parser.add_argument('--case',required=True)", """parser.add_argument('--case',required=True)
    parser.add_argument('--pinning-arm',choices=('native','control','optimized'),required=True)
    parser.add_argument('--pinning-campaign',type=Path,required=True)
    parser.add_argument('--admission',type=Path,required=True)
    parser.add_argument('--seed',type=int,choices=(63100,63101,63102),required=True)"""),
        ("    folder=PREP/'runs'/args.case", """    check_admission(args.admission)
    if args.steps != 120 or args.seq_length != 2048 or args.reference or args.capture_inputs or args.capture_windows:
        raise ValueError('Fixed 120-step / S2048 formal configuration required')
    folder=PREP/'runs'/args.case"""),
        ("str(PREP/'native_entry.py'),'--evidence'", "str(PREP/'pinning_formal_entry.py'),'--evidence'"),
        ("    if args.reference:\n", """    command += ['--pinning-arm',args.pinning_arm,'--pinning-campaign',str(args.pinning_campaign)]
    if args.reference:
"""),
        ("'--seed','63100'", "'--seed',str(args.seed)"),
        ("'--log-interval','1'", "'--log-interval','10'"),
        ("'formal_performance_run':False", "'formal_performance_run':True"),
        ("capture_output=True,text=True).stdout", "capture_output=True,text=True,check=True).stdout"),
        ("    snapshots=folder/'source'", """    record.update(pinning_arm=args.pinning_arm,seed=args.seed,admission_sha256=hashlib.sha256(args.admission.read_bytes()).hexdigest())
    snapshots=folder/'source'"""),
        ("'optimizer_evidence.py','input_evidence.py'):", "'optimizer_evidence.py','input_evidence.py','pinning_formal_entry.py','pinning_candidate.py','pinning_resume_audit.py','pinning_full_audit.py','run_pinning_formal.py'):"),
    ]
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError('Formal runner derivation no longer exact: ' + before)
        text = text.replace(before, after)
    exec(compile(text, str(PREP / 'run_pinning_formal.generated.py'), 'exec'),
         {'__name__': '__main__', '__file__': str(PREP / 'run_pinning_formal.py'), 'check_admission': check_admission})


if __name__ == '__main__':
    main()

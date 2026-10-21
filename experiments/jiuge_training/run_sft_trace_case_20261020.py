"""Derive a fixed 35-step paired profile from the unchanged native launch."""
import hashlib
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
EXPECTED_RUNNER = '82c444f9ba77e70562430f51a17357179f39acb58d639a4d7b919719930bc345'


def main():
    source = (PREP/'run_native_case.py').read_bytes()
    if hashlib.sha256(source).hexdigest() != EXPECTED_RUNNER:
        raise ValueError('Frozen native runner changed')
    text = source.decode()
    changes = [
        ("parser.add_argument('--case',required=True)",
         "parser.add_argument('--case',required=True)\n    parser.add_argument('--pinning-arm',choices=('native','optimized'),required=True)\n    parser.add_argument('--pinning-campaign',type=Path,required=True)"),
        ("    folder=PREP/'runs'/args.case", "    if args.steps != 35 or args.seq_length != 2048 or not args.capture_windows or args.reference or args.capture_inputs:\n        raise ValueError('Fixed 35-step S2048 profile only')\n    folder=PREP/'runs'/args.case"),
        ("str(PREP/'native_entry.py'),'--evidence'", "str(PREP/'sft_trace_entry_20261020.py'),'--evidence'"),
        ("    if args.reference:\n", "    command += ['--pinning-arm',args.pinning_arm,'--pinning-campaign',str(args.pinning_campaign)]\n    if args.reference:\n"),
        ('datasets/alpaca-cpt-v1_text_document', 'datasets/alpaca-sft-answer-v1'),
        ("    snapshots=folder/'source'", "    record.update(pinning_arm=args.pinning_arm,pinning_campaign=str(args.pinning_campaign),objective='answer_only_SFT',symmetric_host_annotations=True)\n    snapshots=folder/'source'"),
        ("'optimizer_evidence.py','input_evidence.py'):",
         "'optimizer_evidence.py','input_evidence.py','run_sft_trace_case_20261020.py','sft_trace_entry_20261020.py','sft_trace_ranges_20261020.py','sft_dataset_adapter.py','pinning_entry.py','pinning_candidate.py'):"),
        ("capture_output=True,text=True).stdout", "capture_output=True,text=True,check=True).stdout"),
    ]
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError('Exact native derivation failed: '+before)
        text = text.replace(before, after)
    exec(compile(text, str(PREP/'run_sft_trace_case_20261020.generated.py'), 'exec'),
         {'__name__':'__main__', '__file__':str(PREP/'run_sft_trace_case_20261020.py')})


if __name__ == '__main__':
    main()

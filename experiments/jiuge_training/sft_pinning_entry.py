"""Apply the source-validated SFT objective to the existing pinning harness."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--sft-pinning-mode', choices=('audit','checkpoint','resume','formal'), required=True)
    args, remaining = parser.parse_known_args()
    root = Path(remaining[remaining.index('--jiuge-run-root')+1])
    rank = int(os.environ['RANK'])
    sys.path.insert(0, str(PREP/'megatron-native-5be9626'))
    from sft_dataset_adapter import install
    dataset = install()
    if args.sft_pinning_mode != 'formal':
        os.environ['JIUGE_SFT_ADMISSION_ROOT'] = str(root)
    target = {'audit':'pinning_semantics_entry.py', 'checkpoint':'pinning_semantics_entry.py',
              'resume':'pinning_resume_compat_entry_v2.py', 'formal':'pinning_formal_entry.py'}[args.sft_pinning_mode]
    record = {'schema':'jiuge7b-sft-pinning-objective-v1','rank':rank,'mode':args.sft_pinning_mode,
              'objective':'answer_and_EOD_only_full_parameter_SFT','entry_target':target,
              'adapter_sha256':hashlib.sha256((PREP/'sft_dataset_adapter.py').read_bytes()).hexdigest(),
              'data_receipt_sha256':hashlib.sha256((PREP/'datasets/alpaca-sft-answer-v1-receipt.json').read_bytes()).hexdigest(),
              'run_returned_successfully':False,'GPU_tensor_audit_in_formal':False}
    sys.argv = [str(PREP/target), *remaining]
    try:
        runpy.run_path(str(PREP/target), run_name='__main__')
        record['run_returned_successfully'] = True
    except SystemExit as error:
        record['run_returned_successfully'] = error.code in (None,0)
        raise
    finally:
        record['supervision_rows'] = dataset.sampled_supervision_rows
        with (root/f'sft-objective-rank-{rank}.json').open('x') as stream:
            json.dump(record, stream, indent=2)


if __name__ == '__main__':
    main()

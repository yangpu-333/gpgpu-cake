"""Use native dense Megatron training with explicit answer-only SFT supervision."""
import hashlib
import json
import os
from pathlib import Path
import sys

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    sys.path.insert(0, str(PREP / 'megatron-native-5be9626'))
    from sft_dataset_adapter import install
    from native_entry import main as native_main
    dataset = install()
    root = Path(sys.argv[sys.argv.index('--jiuge-run-root') + 1])
    rank = int(os.environ.get('RANK', '0'))
    if '--capture-inputs' in sys.argv:
        os.environ['JIUGE_SFT_ADMISSION_ROOT'] = str(root)
    path = root / f'sft-adapter-rank-{rank}.json'
    record = {'schema': 'jiuge7b-sft-data-adapter-v1', 'rank': rank,
              'dataset_class': dataset.__name__, 'native_dense_entry_unchanged': True,
              'native_sft_packed_layout_used': False,
              'objective': 'answer_and_EOD_only_loss_full_parameter_SFT',
              'adapter_sha256': hashlib.sha256((PREP / 'sft_dataset_adapter.py').read_bytes()).hexdigest(),
              'entry_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'data_receipt_sha256': hashlib.sha256((PREP / 'datasets/alpaca-sft-answer-v1-receipt.json').read_bytes()).hexdigest()}
    with path.open('x') as stream:
        json.dump(record, stream, indent=2)
    try:
        native_main()
    finally:
        task = {'schema': 'jiuge7b-sft-supervision-counts-v1', 'rank': rank,
                'rows': dataset.sampled_supervision_rows,
                'TP_rank0_builds_inputs_other_rank_receives_native_broadcast': True,
                'counts_are_input_slots_and_supervised_targets_separately': True}
        with (root/f'sft-supervision-rank-{rank}.json').open('x') as stream:
            json.dump(task, stream, indent=2)


if __name__ == '__main__':
    main()

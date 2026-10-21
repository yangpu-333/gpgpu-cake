"""Answer-only SFT, exact accepted pinning patch and symmetric host ranges."""
import hashlib
import json
import os
from pathlib import Path
import sys

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    sys.path.insert(0, str(PREP/'megatron-native-5be9626'))
    import torch
    import megatron.core.optimizer.cpu_offloading.hybrid_optimizer as module
    from sft_dataset_adapter import install as install_sft
    from sft_trace_ranges_20261020 import install as install_ranges
    from pinning_entry import main as pinning_main
    root = Path(sys.argv[sys.argv.index('--jiuge-run-root') + 1])
    rank = int(os.environ['RANK'])
    arm = sys.argv[sys.argv.index('--pinning-arm') + 1]
    if '--full-state-audit' in sys.argv or '--capture-inputs' in sys.argv:
        raise ValueError('Profiling must not enable intrusive numerical observers')
    dataset = install_sft()
    counts = install_ranges(torch, module.HybridDeviceOptimizer)
    returned = False
    try:
        pinning_main()
        returned = True
    finally:
        record = {'schema': 'jiuge7b-sft-paired-trace-entry-v1', 'rank': rank, 'arm': arm,
                  'run_returned': returned, 'objective': 'answer_and_EOD_only_full_parameter_SFT',
                  'rows': dataset.sampled_supervision_rows, 'range_call_counts': counts,
                  'data_receipt_sha256': hashlib.sha256((PREP/'datasets/alpaca-sft-answer-v1-receipt.json').read_bytes()).hexdigest(),
                  'host_ranges_are_symmetric': True, 'extra_per_step_synchronization': False,
                  'formal_performance_evidence': False}
        with (root/f'sft-trace-objective-rank-{rank}.json').open('x') as stream:
            json.dump(record, stream, indent=2)
        with (root/f'sft-supervision-rank-{rank}.json').open('x') as stream:
            json.dump({'rows': dataset.sampled_supervision_rows}, stream, indent=2)


if __name__ == '__main__':
    main()

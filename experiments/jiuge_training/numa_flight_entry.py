"""Independent diagnostic only: retain native NCCL recorder after real training."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
ENV = {'TORCH_NCCL_TRACE_BUFFER_SIZE': '100000', 'TORCH_NCCL_ENABLE_TIMING': '1',
       'TORCH_NCCL_TRACE_CPP_STACK': '0'}


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--flight-root', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if any(os.environ.get(name) != value for name, value in ENV.items()):
        raise ValueError('Explicit isolated diagnostic recorder environment required')
    if '--numa-observe' not in remaining or '--capture-windows' not in remaining:
        raise ValueError('Use the separate whole-step diagnostic capture only')
    rank = int(os.environ['RANK'])
    start = {'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns()}
    # CPU affinity is applied by the unchanged NUMA logic before Torch imports.
    sys.argv = [str(PREP / 'numa_observation_v2_entry.py'), *remaining]
    runpy.run_path(str(PREP / 'numa_observation_v2_entry.py'), run_name='__main__')
    import torch
    torch.cuda.synchronize()
    before = {'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns()}
    data = torch._C._distributed_c10d._dump_nccl_trace_json(True, False)
    after = {'wall_ns': time.time_ns(), 'monotonic_ns': time.monotonic_ns()}
    value = json.loads(data)
    path = args.flight_root / f'nccl-flight-rank-{rank}.json'
    with path.open('x') as stream:
        stream.write(data); stream.write('\n')
    entries = value.get('entries', [])
    receipt = {'schema': 'jiuge7b-numa-native-flight-capture-v1', 'rank': rank,
               'torch_version': torch.__version__, 'raw_file': str(path), 'raw_bytes': path.stat().st_size,
               'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'environment': ENV,
               'top_level_keys': sorted(value), 'entry_count': len(entries),
               'entry_field_counts': dict(Counter(key for row in entries for key in row)),
               'clock_anchors': {'start': start, 'before_dump': before, 'after_dump': after},
               'formal_throughput_evidence': False, 'cross_rank_native_sequence_verified': False,
               'source_sha256': {name: hashlib.sha256((PREP / name).read_bytes()).hexdigest()
                                  for name in ('numa_flight_entry.py', 'numa_observation_v2_entry.py', 'numa_placement.py')}}
    with (args.flight_root / f'nccl-flight-receipt-rank-{rank}.json').open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    if not entries:
        raise ValueError('Actual native training recorder entries missing; preserve raw failure')


if __name__ == '__main__':
    main()

"""CPU-only coverage review of closed NCCL traces, without order-based proof."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import socket
from analyze_trace import load_trace

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
CASES = ('cpu-numa-unbound-01', 'cpu-numa-local8-02-v2', 'cpu-numa-unbound-03-v2')


def main():
    if socket.gethostname() != 'sg-huids25-261016-13441-default0-0':
        raise ValueError('Use the allocated successor; CPU-only closed-trace analysis')
    root = PREP / 'cpu-numa-collective-sequence-coverage-20261016-01'
    root.mkdir(exist_ok=False)
    (root / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    rows = []
    for name in CASES:
        case = PREP / 'runs' / name
        if json.loads((case / 'case.json').read_text())['exit_code'] != 0:
            raise ValueError('Completed capture required')
        windows = sorted(case.glob('window-*'))
        if len(windows) != 2:
            raise ValueError('Both original windows required')
        for window in windows:
            for rank in (0, 1):
                path = window / f'rank-{rank}.json.gz'
                capture = json.loads((window / f'rank-{rank}-capture.json').read_text())
                if hashlib.sha256(path.read_bytes()).hexdigest() != capture['trace_sha256']:
                    raise ValueError('Compressed original differs from capture receipt')
                trace, source = load_trace(path, max_bytes=512 * 1024**2)
                if source['decompressed_json_sha256'] != capture['decompressed_json_sha256']:
                    raise ValueError('Uncompressed original differs from capture receipt')
                outer, = [e for e in trace['traceEvents'] if e.get('ph') == 'X'
                          and e.get('cat') == 'user_annotation' and e.get('name') == capture['window_name']]
                left, right = outer['ts'], outer['ts'] + outer['dur']
                kernels = [e for e in trace['traceEvents'] if e.get('ph') == 'X'
                           and e.get('cat') == 'kernel' and 'Collective name' in e.get('args', {})
                           and e['ts'] < right and e['ts'] + e['dur'] > left]
                if not kernels:
                    raise ValueError('No actual communication kernel coverage')
                keys = Counter(key for event in kernels for key in event['args'])
                seq_keys = {key: value for key, value in keys.items()
                            if re.search(r'seq|op.?count', key, flags=re.I)}
                signatures = Counter((e['args'].get('Process Group Name'), e['args'].get('Collective name'),
                                      e['args'].get('dtype'), e['args'].get('In msg nelems')) for e in kernels)
                rows.append({'case': name, 'window': window.name, 'rank': rank,
                             'compressed_sha256': capture['trace_sha256'], 'source': source,
                             'communication_kernel_count': len(kernels),
                             'kernel_argument_key_counts': dict(keys),
                             'potential_sequence_key_counts': seq_keys,
                             'signatures': [{'process_group': k[0], 'operation': k[1], 'dtype': k[2],
                                             'input_elements': k[3], 'count': v} for k, v in signatures.items()],
                             'native_collective_sequence_verified': False})
                del trace, kernels
    result = {'schema': 'jiuge7b-collective-sequence-coverage-v1',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'rows': rows,
              'checked_windows': len(rows), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'all_original_capture_sha_verified': True,
              'any_potential_sequence_key': any(row['potential_sequence_key_counts'] for row in rows),
              'native_collective_sequence_verified': False,
              'limitations': ['Autograd Sequence number and External id are not native cross-rank collective sequence proof.',
                             'Matching by signature and order remains a candidate association.',
                             'No new GPU workload, complete critical path, zero-loss or formal speedup claim.']}
    (root / 'coverage.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'checked_windows': len(rows), 'counts': [row['communication_kernel_count'] for row in rows],
                      'any_potential_sequence_key': result['any_potential_sequence_key'],
                      'native_collective_sequence_verified': False}), flush=True)
    from pinning_key_results import export
    export(PREP, [root], PREP / 'cpu-numa-collective-sequence-key-results-20261016.zip')


if __name__ == '__main__':
    main()

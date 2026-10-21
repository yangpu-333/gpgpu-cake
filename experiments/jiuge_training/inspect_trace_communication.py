"""Read recorded communication and memcpy metadata without inferring missing sizes."""
import argparse
from collections import Counter
import json
from pathlib import Path

from analyze_trace import load_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports = []
    paired_windows = []
    dtype_bytes = {'BFloat16': 2, 'Half': 2, 'Float': 4, 'Double': 8,
                   'Long': 8, 'Int': 4, 'Bool': 1, 'Byte': 1, 'Char': 1}
    for folder in sorted(args.case_root.glob('window-*')):
        rank_groups = []
        for rank in range(2):
            trace, _ = load_trace(folder / f'rank-{rank}.json.gz', max_bytes=512 * 1024 * 1024)
            kinds, examples, seen = Counter(), [], set()
            memcpy_bytes = Counter()
            groups, messages = {}, {}
            for event in trace['traceEvents']:
                name, category = str(event.get('name', '')), event.get('cat')
                if 'nccl' in name.lower() or 'c10d::' in name:
                    key = (category, name)
                    kinds[str(key)] += 1
                    if key not in seen and len(examples) < 16:
                        seen.add(key)
                        examples.append({'category': category, 'name': name, 'args': event.get('args', {})})
                if category == 'gpu_memcpy':
                    size = event.get('args', {}).get('bytes')
                    if isinstance(size, (int, float)):
                        memcpy_bytes[name] += size
                fields = event.get('args', {})
                if category == 'kernel' and 'nccl' in name.lower():
                    group = str(fields.get('Process Group Name'))
                    signature = (fields.get('Collective name'), fields.get('dtype'),
                                 fields.get('In msg nelems'), fields.get('Out msg nelems'))
                    record = {'signature': signature, 'start_us': event['ts'],
                              'end_us': event['ts'] + event['dur'], 'duration_us': event['dur'],
                              'base_time_nanoseconds': trace.get('baseTimeNanoseconds'),
                              'stream': fields.get('stream'), 'correlation': fields.get('correlation')}
                    groups.setdefault(group, []).append(record)
                    key = str((group, *signature))
                    stats = messages.setdefault(key, {'group': group,
                        'group_description': fields.get('Process Group Description'),
                        'group_ranks': fields.get('Process Group Ranks'),
                        'group_size': fields.get('Group size'), 'collective': signature[0],
                        'dtype': signature[1], 'input_elements': signature[2],
                        'output_elements': signature[3], 'count': 0,
                        'logical_input_bytes_sum': 0, 'logical_bytes_known': True})
                    stats['count'] += 1
                    width = dtype_bytes.get(signature[1])
                    if width is not None and isinstance(signature[2], int):
                        stats['logical_input_bytes_sum'] += signature[2] * width
                    else:
                        stats['logical_bytes_known'] = False
            reports.append({'window': folder.name, 'rank': rank,
                            'communication_name_counts': dict(kinds), 'examples': examples,
                            'recorded_memcpy_bytes_by_name': dict(memcpy_bytes),
                            'device_message_groups': list(messages.values()),
                            'network_bytes_measured': False})
            rank_groups.append(groups)
            del trace
        candidates, checks = [], []
        for group in sorted(set(rank_groups[0]) | set(rank_groups[1])):
            first = sorted(rank_groups[0].get(group, []), key=lambda x: x['start_us'])
            second = sorted(rank_groups[1].get(group, []), key=lambda x: x['start_us'])
            matched = len(first) == len(second) and all(
                a['signature'] == b['signature'] for a, b in zip(first, second))
            checks.append({'group': group, 'rank_event_counts': [len(first), len(second)],
                           'ordered_type_dtype_length_signatures_match': matched})
            if not matched:
                continue
            for ordinal, (a, b) in enumerate(zip(first, second), 1):
                if not isinstance(a['base_time_nanoseconds'], int) or not isinstance(b['base_time_nanoseconds'], int):
                    continue
                a0, b0 = a['base_time_nanoseconds'] / 1000, b['base_time_nanoseconds'] / 1000
                candidates.append({'group': group, 'ordinal_in_captured_group': ordinal,
                    'signature': a['signature'],
                    'rank_start_absolute_us': [a0 + a['start_us'], b0 + b['start_us']],
                    'rank_duration_us': [a['duration_us'], b['duration_us']],
                    'start_skew_us': abs((a0 - b0) + (a['start_us'] - b['start_us'])),
                    'end_skew_us': abs((a0 - b0) + (a['end_us'] - b['end_us']))})
        paired_windows.append({'window': folder.name, 'group_checks': checks,
            'matching_method': 'same-group ordered kernel type/dtype/message-length signatures',
            'native_collective_sequence_numbers_verified': False,
            'largest_start_skew_candidates': sorted(candidates, key=lambda x: x['start_skew_us'], reverse=True)[:8],
            'critical_path_proven': False})
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump({'schema': 'jiuge-recorded-communication-metadata-v2', 'ranks': reports,
                   'cross_rank_pair_candidates': paired_windows,
                   'logical_input_bytes_are_not_measured_wire_bytes': True}, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'first_rank_message_groups': reports[0]['device_message_groups'],
                      'first_rank_memcpy_bytes': reports[0]['recorded_memcpy_bytes_by_name'],
                      'pair_candidates': [item['largest_start_skew_candidates'][:2] for item in paired_windows]}, ensure_ascii=False))


if __name__ == '__main__':
    main()

"""Match recorded GPU memcpy events to unique CUPTI runtime correlation IDs.

CPU API spans include driver work, staging and possible blocking. They are not
an independently measured exposed wait or a formal throughput measurement.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from analyze_trace import interval_length, load_trace


def stats(values):
    values = sorted(values)
    return {'count': len(values), 'sum_us': sum(values),
            'p50_us': values[math.ceil(.5 * len(values)) - 1] if values else None,
            'p95_us': values[math.ceil(.95 * len(values)) - 1] if values else None,
            'max_us': values[-1] if values else None}


def inspect(trace, window_name):
    events = trace['traceEvents']
    outer = [e for e in events if e.get('ph') == 'X' and
             e.get('cat') == 'user_annotation' and e.get('name') == window_name]
    if len(outer) != 1:
        raise ValueError('Exactly one actual enclosing capture annotation required')
    start, end = outer[0]['ts'], outer[0]['ts'] + outer[0]['dur']
    if not math.isfinite(end - start) or end <= start:
        raise ValueError('Invalid enclosing capture interval')

    def clipped(event):
        if event.get('ph') != 'X':
            return None
        t, d = event.get('ts'), event.get('dur')
        if not isinstance(t, (int, float)) or not isinstance(d, (int, float)):
            return None
        if not math.isfinite(t) or not math.isfinite(d) or d < 0:
            raise ValueError('Nonfinite/negative recorded event interval')
        a, b = max(start, t), min(end, t + d)
        return (a, b) if b > a else None

    by_correlation = defaultdict(list)
    runtime, sync = {}, defaultdict(list)
    for index, event in enumerate(events):
        if event.get('cat') != 'cuda_runtime' or event.get('ph') != 'X':
            continue
        runtime[index] = event
        correlation = event.get('args', {}).get('correlation')
        if isinstance(correlation, int) and correlation > 0:
            by_correlation[correlation].append(index)
        if 'Synchronize' in str(event.get('name')) and clipped(event):
            sync[str(event['name'])].append(event)
    groups = {}
    api_copy_kinds = defaultdict(set)
    for event in events:
        if event.get('cat') != 'gpu_memcpy' or not clipped(event):
            continue
        name = str(event.get('name'))
        group = groups.setdefault(name, {'events': [], 'api_ids': set(), 'unmatched': 0, 'ambiguous': 0})
        group['events'].append(event)
        correlation = event.get('args', {}).get('correlation')
        candidates = by_correlation.get(correlation, []) if isinstance(correlation, int) and correlation > 0 else []
        if len(candidates) == 1:
            index = candidates[0]
            group['api_ids'].add(index)
            api_copy_kinds[index].add(name)
        elif not candidates:
            group['unmatched'] += 1
        else:
            group['ambiguous'] += 1
    result = []
    for name, group in sorted(groups.items()):
        gpu_intervals = defaultdict(list)
        sizes = []
        for event in group['events']:
            gpu_intervals[str(event.get('args', {}).get('device'))].append(clipped(event))
            size = event.get('args', {}).get('bytes')
            if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
                sizes.append(size)
        api = [runtime[index] for index in sorted(group['api_ids'])]
        api_intervals = [value for event in api if (value := clipped(event)) is not None]
        result.append({'copy_name': name, 'gpu_copy_event_count': len(group['events']),
            'recorded_bytes': sum(sizes), 'copy_events_without_recorded_size': len(group['events']) - len(sizes),
            'gpu_duration': stats([e['dur'] for e in group['events']]),
            'gpu_union_us_by_device': {device: interval_length(values) for device, values in gpu_intervals.items()},
            'unique_matched_cpu_api_count': len(api), 'unmatched_gpu_copy_events': group['unmatched'],
            'ambiguous_gpu_copy_events': group['ambiguous'],
            'cpu_api_duration': stats([e['dur'] for e in api]),
            'cpu_api_union_clipped_us': interval_length(api_intervals),
            'cpu_api_clipped_sum_us': sum(b-a for a, b in api_intervals),
            'cpu_api_events_outside_window': sum(clipped(e) is None for e in api),
            'cpu_api_names': sorted({str(e['name']) for e in api}),
            'api_correlations_shared_with_other_copy_names': sum(len(api_copy_kinds[index]) > 1 for index in group['api_ids'])})
    return {'window_us': end - start, 'copy_groups': result,
        'cpu_synchronization_api_spans': {name: {**stats([e['dur'] for e in series]),
            'union_clipped_us': interval_length([clipped(e) for e in series])} for name, series in sync.items()},
        'matching': 'unique cuda_runtime event with the same positive CUPTI correlation ID',
        'cpu_api_spans_are_exposed_wait': False, 'physical_wire_bytes_measured': False,
        'categories_are_additive_wall_time': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case = json.loads((args.case_root / 'case.json').read_text(encoding='utf-8'))
    if case['exit_code'] != 0:
        raise ValueError('Completed native training capture required')
    rows = []
    for folder in sorted(args.case_root.glob('window-*')):
        for rank in (0, 1):
            receipt = json.loads((folder / f'rank-{rank}-capture.json').read_text(encoding='utf-8'))
            path = folder / f'rank-{rank}.json.gz'
            if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['trace_sha256']:
                raise ValueError('Compressed trace SHA differs from the actual capture receipt')
            trace, source = load_trace(path, max_bytes=512 * 1024**2)
            if source['decompressed_json_sha256'] != receipt['decompressed_json_sha256']:
                raise ValueError('Decompressed trace SHA differs')
            rows.append({'window': folder.name, 'rank': rank, 'steps': receipt['stable_step_ids'],
                         'source': source, 'compressed_trace_sha256': receipt['trace_sha256'],
                         **inspect(trace, receipt['window_name'])})
            del trace
    if len(rows) != 4 or any(len(row['steps']) != 5 for row in rows):
        raise ValueError('Both actual five-step windows and both ranks required')
    result = {'schema': 'jiuge-cupti-copy-api-review-v1', 'case': case['case'], 'windows': rows,
              'source_sha256': {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ('inspect_copy_api.py', 'analyze_trace.py')},
              'formal_performance_evidence': False, 'critical_path_proven': False,
              'collector_drop_count_verified': False}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'case': case['case'], 'windows_reviewed': len(rows),
                     'output': str(args.output), 'formal_performance_evidence': False}))


if __name__ == '__main__':
    main()

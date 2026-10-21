"""Review real native capture receipts, clocks, CPU phases and device overlap."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from analyze_trace import analyze_rank_trace, load_trace


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case = read(args.case_root / 'case.json')
    if case['exit_code'] != 0:
        raise ValueError('A completed native training case is required')
    entries = [read(args.case_root / f'rank-{rank}/entry.json') for rank in range(2)]
    if any(entry['status'] != 'completed' or any(s['skipped'] for s in entry['completed_native_steps'])
           for entry in entries):
        raise ValueError('Both ranks must complete real optimizer updates without skipped steps')
    reports = []
    for folder in sorted(args.case_root.glob('window-*')):
        for rank in range(2):
            receipt = read(folder / f'rank-{rank}-capture.json')
            path = folder / f'rank-{rank}.json.gz'
            if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['trace_sha256']:
                raise ValueError('Raw compressed trace SHA256 mismatch')
            trace, source = load_trace(path, max_bytes=512 * 1024 * 1024)
            if source['decompressed_json_sha256'] != receipt['decompressed_json_sha256']:
                raise ValueError('Decompressed trace SHA256 mismatch')
            metadata = {'world_size': 2, 'stable_step_ids': receipt['stable_step_ids'],
                        'gpu_activity_enabled': True,
                        'window_boundaries_include_full_gpu_work': receipt['whole_window_boundary_drained'],
                        'window_complete_training_steps_verified': all(
                            any(s['step'] == step and s['skipped'] == 0 for s in entries[rank]['completed_native_steps'])
                            for step in receipt['stable_step_ids']),
                        'collective_gpu_capture_verified': receipt['gpu_collective_kernel_events'] > 0,
                        'expected_collective_activity': True,
                        'dropped_events': receipt['reported_collector_drop_count'],
                        'capture_errors': []}
            analysis = analyze_rank_trace(trace, rank, receipt['stable_step_ids'], metadata,
                                          max_events=2_000_000, window={'name': receipt['window_name']})
            outer = [e for e in trace['traceEvents'] if e.get('name') == receipt['window_name']
                     and e.get('ph') == 'X' and e.get('cat') == 'user_annotation']
            if len(outer) != 1:
                raise ValueError('Exactly one full capture annotation required')
            start, end = outer[0]['ts'], outer[0]['ts'] + outer[0]['dur']
            base = trace.get('baseTimeNanoseconds')
            clock = {'base_time_nanoseconds': base, 'single_host': True,
                     'shared_kineto_export_time_domain': True,
                     'independent_physical_clock_calibration_performed': False}
            if isinstance(base, int):
                left, right = base + round(start * 1000), base + round(end * 1000)
                clock.update(annotation_wall_start_ns=left, annotation_wall_end_ns=right,
                             annotation_inside_recorded_host_bounds=(
                                 receipt['window_start_wall_ns'] <= left <= right <= receipt['window_end_wall_ns']))
            cpu = {}
            for name in ('jiuge/forward_backward', 'jiuge/optimizer_update'):
                series = [e['dur'] for e in trace['traceEvents'] if e.get('ph') == 'X'
                          and e.get('cat') == 'user_annotation'
                          and e.get('name') == name and start <= e.get('ts', -1) < end]
                cpu[name] = {'count': len(series), 'host_range_sum_us': sum(series),
                             'host_window_fraction': sum(series) / (end - start)}
            sync = Counter()
            collectives = []
            for event in trace['traceEvents']:
                if event.get('ph') != 'X' or not start <= event.get('ts', -1) < end:
                    continue
                name = str(event.get('name', ''))
                if event.get('cat') == 'cuda_runtime' and 'Synchronize' in name:
                    sync[name] += event.get('dur', 0)
                if event.get('cat') == 'cpu_op' and ('c10d::' in name or 'nccl:' in name):
                    if len(collectives) < 12:
                        collectives.append({'name': name, 'args': event.get('args', {})})
            reports.append({'window': folder.name, 'rank': rank, 'clock_review': clock,
                            'cpu_phase_ranges': cpu,
                            'cpu_cuda_sync_work_sum_us_not_exposed_wait': dict(sync),
                            'collective_cpu_metadata_examples': collectives, 'analysis': analysis})
            del trace
    rows = []
    log = (args.case_root / 'stdout.log').read_text(encoding='utf-8')
    for match in re.finditer(r'iteration\s+(\d+)/\s*(\d+).*?elapsed time per iteration \(ms\):\s*([\d.]+).*?lm loss:\s*([\d.E+\-]+)', log):
        rows.append({'step': int(match[1]), 'iterations': int(match[2]),
                     'native_log_elapsed_ms': float(match[3]), 'lm_loss': float(match[4])})
    result = {'schema': 'jiuge7b-native-trace-review-v1', 'case': case['case'],
              'real_native_case_exit_code': case['exit_code'],
              'steps_per_rank': [len(entry['completed_native_steps']) for entry in entries],
              'windows': reports, 'native_iteration_log': rows,
              'formal_performance_passed': False, 'critical_path_proven': False,
              'complete_zero_drop_coverage_claimed': False,
              'limitations': ['Collector drop count is unavailable in this PyTorch API.',
                              'CPU phase ranges and GPU activity are separate accounts, not additive.',
                              'GPU idle and nonoverlap communication are descriptive proxies.',
                              'Two ranks share one host; this does not establish multi-node behavior.',
                              'Only selected optimizer state samples were archived; no full training resume test.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    for report in reports:
        device = report['analysis']['capture_window']['devices'][0]
        print(json.dumps({'window': report['window'], 'rank': report['rank'],
                          'window_seconds': device['window_us'] / 1e6,
                          'gpu': device, 'cpu_phases': report['cpu_phase_ranges']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

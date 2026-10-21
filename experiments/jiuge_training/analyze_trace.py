"""Bounded, CPU-only analysis of per-rank Kineto Chrome JSON traces.

Example (explicitly selected stable step numbers):
  python analyze_trace.py --trace 0=rank0.json --trace 1=rank1.json.gz \
      --steps 20,21,22 --capture-metadata capture.json --output analysis.json

Optional capture metadata must describe the actual capture, not an acceptance
claim. Fields: world_size, stable_step_ids, gpu_activity_enabled,
step_boundaries_include_full_gpu_work, collective_gpu_capture_verified,
expected_collective_activity (bool), dropped_events (integer), capture_errors
(list). These declarations are reported as such. Complete descriptive coverage
does not prove a real model, numerical correctness, throughput, or critical path.
ProfilerStep boundaries alone do not prove completion of asynchronous GPU work.
Use --window-name with an enclosing user annotation, or --window-range with an
explicit ts range, for whole-window accounting. Only window endpoints need GPU
draining; internal training steps retain their asynchronous schedule. Metadata
then requires window_boundaries_include_full_gpu_work and
window_complete_training_steps_verified. Per-step attribution can remain
incomplete while the enclosing window has complete descriptive coverage.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re


DEFAULT_MAX_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_EVENTS = 500_000
STEP_NAME = re.compile(r'^(?:ProfilerStep#|jiuge/step/|jiuge/train_step/)(\d+)$')
COLLECTIVE_NAME = re.compile(
    r'(?:nccl|rccl|xccl|hccl|mccl|ixccl|ixcl|all[_ ]?reduce|'
    r'all[_ ]?gather|reduce[_ ]?scatter|all[_ ]?to[_ ]?all|'
    r'collective|broadcast|send[_ ]?recv)', re.IGNORECASE)
DEVICE_LABEL = re.compile(r'\bGPU\s+(\d+)\b', re.IGNORECASE)


def union_intervals(intervals):
    """Return disjoint half-open intervals; zero/negative widths are ignored."""
    merged = []
    for start, end in sorted((a, b) for a, b in intervals if b > a):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def interval_length(intervals):
    return sum(end - start for start, end in union_intervals(intervals))


def intersection_length(first, second):
    first, second = union_intervals(first), union_intervals(second)
    result, i, j = 0.0, 0, 0
    while i < len(first) and j < len(second):
        result += max(0.0, min(first[i][1], second[j][1]) - max(first[i][0], second[j][0]))
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1
    return result


def load_trace(path, max_bytes=DEFAULT_MAX_BYTES):
    """Bound both compressed input size and decompressed JSON bytes."""
    path = Path(path)
    if max_bytes <= 0 or path.stat().st_size > max_bytes:
        raise ValueError('trace exceeds the input byte limit')
    opener = gzip.open if path.name.lower().endswith('.gz') else open
    with opener(path, 'rb') as stream:
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError('decompressed trace exceeds the input byte limit')
    value = json.loads(payload)
    if not isinstance(value, dict) or not isinstance(value.get('traceEvents'), list):
        raise ValueError('expected a Kineto object with a traceEvents array')
    return value, {'path': str(path.resolve()), 'compressed': path.name.lower().endswith('.gz'),
                   'json_bytes': len(payload),
                   'decompressed_json_sha256': hashlib.sha256(payload).hexdigest()}


def _number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


def _event_interval(event):
    ts, dur = event.get('ts'), event.get('dur')
    if not _number(ts) or not _number(dur) or dur < 0 or not _number(ts + dur):
        return None
    return float(ts), float(ts + dur)


def _device(event, device_labels):
    args = event.get('args') if isinstance(event.get('args'), dict) else {}
    device = args.get('device')
    if isinstance(device, int) and not isinstance(device, bool) and device >= 0:
        return str(device)
    if isinstance(device, str) and device.isdigit():
        return str(int(device))
    pid = event.get('pid')
    return device_labels.get(pid) if isinstance(pid, (str, int)) else None


def _gpu_kind(event):
    cat = str(event.get('cat', '')).lower()
    # Kineto emits nested GPU-side ranges for record_function annotations.
    # These describe enclosing spans, not continuously executing device work.
    if cat == 'gpu_user_annotation':
        return None
    if cat in ('kernel', 'gpu_kernel', 'cuda_kernel'):
        return 'collective' if COLLECTIVE_NAME.search(str(event.get('name', ''))) else 'compute'
    if cat == 'gpu_memcpy':
        return 'memcpy'
    if cat == 'gpu_memset':
        return 'memset'
    if cat.startswith('gpu_'):
        return 'unknown_gpu'
    return None


def _metadata_issues(metadata, steps, whole_window=False):
    issues = []
    if metadata.get('gpu_activity_enabled') is not True:
        issues.append('GPU activity capture is not declared enabled')
    if whole_window:
        if metadata.get('window_boundaries_include_full_gpu_work') is not True:
            issues.append('complete asynchronous GPU work inside capture-window boundaries is unverified')
        if metadata.get('window_complete_training_steps_verified') is not True:
            issues.append('complete training steps inside the capture window are unverified')
    elif metadata.get('step_boundaries_include_full_gpu_work') is not True:
        issues.append('complete asynchronous GPU work inside step boundaries is unverified')
    if metadata.get('collective_gpu_capture_verified') is not True:
        issues.append('GPU collective coverage/recognition is unverified')
    declared = metadata.get('stable_step_ids')
    if not isinstance(declared, list) or not set(steps).issubset(
            {x for x in declared if isinstance(x, int) and not isinstance(x, bool)}):
        issues.append('selected steps are not declared stable after warmup')
    if not isinstance(metadata.get('expected_collective_activity'), bool):
        issues.append('expected device collective activity is not fixed')
    dropped = metadata.get('dropped_events')
    if not isinstance(dropped, int) or isinstance(dropped, bool) or dropped != 0:
        issues.append('zero dropped events is not established')
    if metadata.get('capture_errors') != []:
        issues.append('capture errors are present or not reported')
    return issues


def analyze_rank_trace(trace, rank, steps, metadata=None, max_events=DEFAULT_MAX_EVENTS, top_k=20, window=None):
    """Analyze one rank, keeping each physical device separate."""
    metadata = metadata or {}
    events = trace.get('traceEvents')
    if not isinstance(events, list) or len(events) > max_events:
        raise ValueError('traceEvents is missing or exceeds the event limit')
    if not steps or len(steps) > 256 or any(
            not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in steps):
        raise ValueError('explicit nonnegative stable step IDs are required')
    if len(set(steps)) != len(steps):
        raise ValueError('duplicate selected step ID')
    if not 1 <= top_k <= 100:
        raise ValueError('top_k must be between 1 and 100')
    if window is not None and not isinstance(window, dict):
        raise ValueError('window must specify a name or explicit start_ts_us/end_ts_us')
    issues = _metadata_issues(metadata, steps, whole_window=window is not None)
    distribution = trace.get('distributedInfo') or {}
    if not isinstance(distribution, dict):
        issues.append('distributedInfo is not an object')
        distribution = {}
    if distribution.get('rank') is not None and distribution.get('rank') != rank:
        issues.append('trace distributedInfo rank does not match supplied rank')
    device_labels = {}
    boundaries, enclosing = defaultdict(list), []
    gpu_events, cpu_collectives = [], []
    invalid_gpu = missing_device = unknown_gpu = 0
    signatures, duplicates = set(), 0
    for event in events:
        if not isinstance(event, dict):
            issues.append('non-object trace event')
            continue
        if event.get('ph') == 'M' and event.get('name') in ('process_labels', 'process_name'):
            args = event.get('args') if isinstance(event.get('args'), dict) else {}
            label = DEVICE_LABEL.search(str(args.get('labels', args.get('name', ''))))
            if label and isinstance(event.get('pid'), (str, int)):
                device_labels[event.get('pid')] = label.group(1)
    for event in events:
        if not isinstance(event, dict) or event.get('ph') != 'X':
            continue
        name = str(event.get('name', ''))
        marker = STEP_NAME.fullmatch(name)
        interval = _event_interval(event)
        if window and window.get('name') == name and str(event.get('cat', '')).lower() == 'user_annotation':
            if interval and interval[1] > interval[0]:
                enclosing.append(interval)
        if marker and str(event.get('cat', '')).lower() == 'user_annotation':
            step = int(marker.group(1))
            if step in steps:
                if interval is None or interval[1] <= interval[0]:
                    issues.append('invalid selected step boundary: ' + str(step))
                else:
                    boundaries[step].append(interval)
        kind = _gpu_kind(event)
        if kind:
            if interval is None:
                invalid_gpu += 1
                continue
            device = _device(event, device_labels)
            if device is None:
                missing_device += 1
                continue
            args = event.get('args') if isinstance(event.get('args'), dict) else {}
            signature = (kind, name, device, str(args.get('stream', event.get('tid'))),
                         interval, str(args.get('correlation')))
            if signature in signatures:
                duplicates += 1
            signatures.add(signature)
            gpu_events.append((interval, device, kind, name))
            unknown_gpu += kind == 'unknown_gpu'
        elif interval and str(event.get('cat', '')).lower() in ('cpu_op', 'user_annotation'):
            if COLLECTIVE_NAME.search(name):
                cpu_collectives.append(interval)
    if invalid_gpu:
        issues.append(str(invalid_gpu) + ' GPU events have invalid timestamps/durations')
    if missing_device:
        issues.append(str(missing_device) + ' GPU events lack a physical-device identity')
    if unknown_gpu:
        issues.append(str(unknown_gpu) + ' GPU activities have unknown categories')
    if duplicates:
        issues.append(str(duplicates) + ' duplicate GPU events; work totals retain these records')
    windows = []
    for step in steps:
        found = boundaries[step]
        if len(found) != 1:
            issues.append('selected step must have exactly one boundary: ' + str(step))
        else:
            windows.append((found[0][0], found[0][1], step))
    windows.sort()
    if any(windows[i][0] < windows[i - 1][1] for i in range(1, len(windows))):
        issues.append('selected step boundaries overlap; no cross-step total is reported')
    whole_bounds = None
    if window is not None:
        if window.get('name'):
            if len(enclosing) != 1:
                issues.append('capture window must have exactly one valid annotation boundary')
            else:
                whole_bounds = enclosing[0]
        else:
            start, end = window.get('start_ts_us'), window.get('end_ts_us')
            if not _number(start) or not _number(end) or end <= start:
                issues.append('explicit capture-window range is invalid')
            else:
                whole_bounds = (float(start), float(end))
        if whole_bounds:
            if any(left < whole_bounds[0] or right > whole_bounds[1] for left, right, _ in windows):
                issues.append('selected stable CPU steps are outside the enclosing capture window')
            windows.append((whole_bounds[0], whole_bounds[1], None))
    reports, kernel_totals = [], defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    whole_report = None
    for start, end, step in windows:
        local_issues, buckets = [], defaultdict(lambda: defaultdict(list))
        if step is not None and metadata.get('step_boundaries_include_full_gpu_work') is not True:
            local_issues.append('per-step asynchronous GPU attribution is unverified')
        sums = defaultdict(lambda: defaultdict(float))
        counts = defaultdict(lambda: defaultdict(int))
        for (left, right), device, kind, name in gpu_events:
            clipped = (max(start, left), min(end, right))
            if clipped[1] <= clipped[0]:
                continue
            if left < start or right > end:
                local_issues.append('GPU activity crosses a complete capture-window boundary' if step is None
                                    else 'GPU activity crosses a selected step boundary')
            buckets[device][kind].append(clipped)
            sums[device][kind] += clipped[1] - clipped[0]
            counts[device][kind] += 1
            if kind == 'compute' and (window is None or step is None):
                item = kernel_totals[device][name]
                item[0] += clipped[1] - clipped[0]
                item[1] += 1
        collective_count = sum(values['collective'] for values in counts.values())
        host_collective = any(left < end and right > start for left, right in cpu_collectives)
        if not buckets:
            local_issues.append('selected step has no captured device activity')
        if host_collective and not collective_count:
            local_issues.append('CPU collective annotation exists without a recognized device collective')
        if metadata.get('expected_collective_activity') is True and not collective_count:
            local_issues.append('expected device collective missing in selected step')
        if metadata.get('expected_collective_activity') is False and collective_count:
            local_issues.append('device collective activity contradicts capture metadata')
        devices = []
        for device in sorted(buckets):
            values = buckets[device]
            compute, comm = interval_length(values['compute']), interval_length(values['collective'])
            overlap = intersection_length(values['compute'], values['collective'])
            busy = interval_length([item for series in values.values() for item in series])
            width = end - start
            devices.append({
                'device': device, 'window_us': width,
                'interval_union_us': {kind: interval_length(values[kind]) for kind in
                                      ('compute', 'collective', 'memcpy', 'memset', 'unknown_gpu')},
                'gpu_busy_union_us': busy,
                'kernel_work_sum_us': {'compute': sums[device]['compute'],
                                       'collective': sums[device]['collective'],
                                       'all_kernels': sums[device]['compute'] + sums[device]['collective']},
                'activity_record_counts': dict(counts[device]),
                'compute_collective_overlap_us': overlap,
                'collective_overlap_fraction': overlap / comm if comm else None,
                'collective_nonoverlap_with_compute_proxy_us': max(0.0, comm - overlap),
                'gpu_idle_proxy_us': max(0.0, width - busy),
                'gpu_idle_proxy_fraction': max(0.0, width - busy) / width,
            })
        local_issues = sorted(set(local_issues))
        report = {'step': step, 'start_ts_us': start, 'end_ts_us': end,
                        'status': 'incomplete' if issues or local_issues else 'descriptive',
                        'issues': local_issues, 'devices': devices}
        if step is None:
            whole_report = report
            whole_report['selector'] = window
        else:
            reports.append(report)
        if window is None or step is None:
            issues.extend(('capture window: ' if step is None else 'step ' + str(step) + ': ') +
                          issue for issue in local_issues)
    top = []
    for device in sorted(kernel_totals):
        ordered = sorted(kernel_totals[device].items(), key=lambda pair: pair[1][0], reverse=True)
        top.append({'device': device, 'ranking': 'clipped device kernel work sum, not critical path',
                    'kernels': [{'name': name, 'kernel_work_sum_us': values[0],
                                 'event_intersections': values[1]} for name, values in ordered[:top_k]]})
    issues = sorted(set(issues))
    return {'rank': rank, 'trace_world_size': distribution.get('world_size'),
            'status': 'incomplete' if issues else 'complete_descriptive', 'issues': issues,
            'event_count': len(events), 'steps': reports, 'top_compute_kernels_by_device': top,
            'analysis_scope': 'complete_capture_window' if window is not None else 'selected_step_ranges',
            'capture_window': whole_report,
            'per_step_attribution_complete': bool(reports) and all(x['status'] == 'descriptive' for x in reports),
            'capture_metadata_declarations': metadata,
            'critical_path_proven': False, 'formal_performance_passed': False}


def _combine_reports(reports, steps, metadata):
    issues = []
    world = metadata.get('world_size')
    if not isinstance(world, int) or isinstance(world, bool) or not 2 <= world <= 4096:
        issues.append('at-least-two-rank capture world_size is not established')
    else:
        if {report['rank'] for report in reports} != set(range(world)):
            issues.append('not all expected ranks have trace files')
        for report in reports:
            if report['trace_world_size'] is not None and report['trace_world_size'] != world:
                issues.append('trace distributedInfo world_size disagrees with metadata')
    if not reports:
        issues.append('no traces supplied')
    if any(report['status'] == 'incomplete' for report in reports):
        issues.append('one or more ranks have incomplete descriptive coverage')
    return {'schema': 'jiuge-kineto-analysis-v1',
            'status': 'incomplete' if issues else 'complete_descriptive', 'issues': issues,
            'timestamp_unit': 'microseconds (Kineto ts/dur; displayTimeUnit is display only)',
            'selected_step_ids': steps, 'ranks': reports,
            'analysis_scope': 'complete_capture_window' if reports and all(
                x.get('analysis_scope') == 'complete_capture_window' for x in reports) else 'selected_step_ranges',
            'cross_rank_or_cross_device_time_sum_reported': False,
            'critical_path_proven': False, 'formal_performance_passed': False,
            'real_jiuge_training_verified': False,
            'limitations': [
                'Capture metadata is a declaration, not independently verified execution evidence.',
                'ProfilerStep CPU ranges need verified GPU-completion boundaries for full-step attribution.',
                'Complete enclosing-window coverage does not imply complete attribution to individual asynchronous steps.',
                'Collectives are identified by device-kernel naming; CPU annotations are never GPU duration.',
                'Collective non-overlap is a proxy, not proven exposed communication on the critical path.',
                'GPU idle is a captured-activity coverage proxy; it is not measured hardware utilization.',
                'Kernel work sums can exceed wall time because device streams overlap.',
                'No rank clock alignment, stream dependency reconstruction, or network-byte measurement is performed.',
                'This profiled capture is separate from the frozen unprofiled throughput/correctness acceptance.',
            ]}


def analyze_traces(traces, steps, metadata=None, max_events=DEFAULT_MAX_EVENTS, top_k=20, window=None):
    """Analyze rank files independently; never sum rank/device wall times."""
    metadata = metadata or {}
    reports = [analyze_rank_trace(value, rank, steps, metadata, max_events, top_k, window=window)
               for rank, value in sorted(traces.items())]
    return _combine_reports(reports, steps, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', action='append', required=True, metavar='RANK=PATH')
    parser.add_argument('--steps', required=True, help='Comma-separated explicitly selected stable step IDs')
    parser.add_argument('--capture-metadata', type=Path)
    windows = parser.add_mutually_exclusive_group()
    windows.add_argument('--window-name', help='Exact enclosing user_annotation name on every rank')
    windows.add_argument('--window-range', help='Explicit microsecond START_TS:END_TS, not display units')
    parser.add_argument('--max-json-mib', type=int, default=128)
    parser.add_argument('--max-events', type=int, default=DEFAULT_MAX_EVENTS)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if not 1 <= args.max_json_mib <= 512 or not 1 <= args.max_events <= 2_000_000:
            raise ValueError('byte/event resource limits exceed the supported bounds')
        steps = [int(value) for value in args.steps.split(',')]
        window = {'name': args.window_name} if args.window_name else None
        if args.window_range:
            start, end = args.window_range.split(':')
            window = {'start_ts_us': float(start), 'end_ts_us': float(end)}
        metadata = {}
        if args.capture_metadata:
            if args.capture_metadata.stat().st_size > 1024 * 1024:
                raise ValueError('capture metadata exceeds 1 MiB')
            metadata = json.loads(args.capture_metadata.read_text(encoding='utf-8'))
            if not isinstance(metadata, dict):
                raise ValueError('capture metadata must be an object')
        ranks, sources = set(), []
        # Process one file at a time to avoid retaining every large parsed trace.
        reports = []
        for specification in args.trace:
            rank_text, separator, path = specification.partition('=')
            if not separator or not rank_text.isdigit() or int(rank_text) in ranks or not 0 <= int(rank_text) < 4096:
                raise ValueError('trace must use a unique nonnegative RANK=PATH')
            rank = int(rank_text)
            document, source = load_trace(path, args.max_json_mib * 1024 * 1024)
            report = analyze_rank_trace(document, rank, steps, metadata, args.max_events, args.top_k, window=window)
            reports.append(report)
            ranks.add(rank)
            source['rank'] = rank
            sources.append(source)
            del document
        result = _combine_reports(sorted(reports, key=lambda item: item['rank']), steps, metadata)
        result['sources'] = sources
    except (ValueError, OSError, TypeError, json.JSONDecodeError) as error:
        result = {'schema': 'jiuge-kineto-analysis-v1', 'status': 'input_error',
                  'issues': [str(error)], 'formal_performance_passed': False,
                  'critical_path_proven': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output.resolve()),
                      'issues': result['issues']}, ensure_ascii=False))
    return 0 if result['status'] == 'complete_descriptive' else 2


if __name__ == '__main__':
    raise SystemExit(main())

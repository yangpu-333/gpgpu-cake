"""Review actual CPU AdamW scopes and synchronization in closed ACT traces."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
from analyze_trace import load_trace, interval_length

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    if socket.gethostname() != 'sg-huids25-261016-24e4c-default0-0':
        raise ValueError('Wrong CPU analysis allocation')
    root = PREP / 'cpu-adamw-trace-review-20261016-01'
    root.mkdir(exist_ok=False)
    rows = []
    for arm in ('native', 'optimized'):
        case = PREP / 'runs' / f'pinning-{arm}-profile-01'
        if json.loads((case / 'case.json').read_text())['exit_code'] != 0:
            raise ValueError('Completed capture required')
        for folder in sorted(case.glob('window-*')):
            for rank in (0, 1):
                receipt = json.loads((folder / f'rank-{rank}-capture.json').read_text())
                path = folder / f'rank-{rank}.json.gz'
                if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['trace_sha256']:
                    raise ValueError('Capture receipt compressed SHA differs')
                trace, source = load_trace(path, max_bytes=512 * 1024**2)
                if source['decompressed_json_sha256'] != receipt['decompressed_json_sha256']:
                    raise ValueError('Decompressed SHA differs')
                outer, = [e for e in trace['traceEvents'] if e.get('ph') == 'X'
                          and e.get('cat') == 'user_annotation' and e.get('name') == receipt['window_name']]
                left, right = outer['ts'], outer['ts'] + outer['dur']
                def clip(e):
                    a, b = max(left, e.get('ts', 0)), min(right, e.get('ts', 0) + e.get('dur', 0))
                    return (a, b) if b > a else None
                events = [e for e in trace['traceEvents'] if e.get('ph') == 'X' and clip(e)]
                adam = [e for e in events if 'AdamW.step' in str(e.get('name'))]
                names = Counter(e['name'] for e in events if 'adam' in str(e.get('name')).lower())
                by_thread = defaultdict(list)
                for e in adam:
                    by_thread[(e.get('pid'), e.get('tid'))].append(clip(e))
                for spans in by_thread.values():
                    spans.sort()
                operations, cuda = defaultdict(list), defaultdict(list)
                for event in events:
                    a, b = clip(event)
                    thread = (event.get('pid'), event.get('tid'))
                    if not any(x <= a and b <= y for x, y in by_thread.get(thread, ())):
                        continue
                    if event.get('cat') == 'cpu_op' and str(event['name']).startswith('aten::'):
                        operations[event['name']].append((a, b))
                    if event.get('cat') == 'cuda_runtime':
                        cuda[event['name']].append((a, b))
                sync = defaultdict(list)
                for event in events:
                    if event.get('cat') == 'cuda_runtime' and 'Synchronize' in str(event['name']):
                        sync[event['name']].append(clip(event))
                phase_rows = []
                for phase in sorted([e for e in events if e.get('name') == 'jiuge/optimizer_update'], key=lambda e:e['ts']):
                    a, b = clip(phase)
                    selected = [e for e in adam if a <= e['ts'] < b]
                    phase_rows.append({'start_wall_ns': trace['baseTimeNanoseconds'] + round(a * 1000),
                                       'end_wall_ns': trace['baseTimeNanoseconds'] + round(b * 1000),
                                       'host_s': (b - a) / 1e6, 'adam_scope_count': len(selected),
                                       'adam_scope_sum_s': sum(clip(e)[1] - clip(e)[0] for e in selected) / 1e6,
                                       'last_adam_end_wall_ns': trace['baseTimeNanoseconds'] + round(max(
                                           (e['ts'] + e['dur'] for e in selected), default=a) * 1000)})
                row = {'arm': arm, 'window': folder.name, 'rank': rank, 'source': source,
                       'compressed_sha256': receipt['trace_sha256'], 'steps': receipt['stable_step_ids'],
                       'adam_scope_names': dict(names), 'adam_scope_count': len(adam),
                       'adam_host_union_s': interval_length([clip(e) for e in adam]) / 1e6,
                       'adam_host_sum_s': sum(clip(e)[1] - clip(e)[0] for e in adam) / 1e6,
                       'cuda_api_within_adam_s': {n: interval_length(v) / 1e6 for n,v in cuda.items()},
                       'synchronize_api_s': {n: interval_length(v) / 1e6 for n,v in sync.items()},
                       'aten_scopes_within_adam': sorted([{'name': n,'count': len(v),
                           'host_sum_s': sum(b-a for a,b in v)/1e6} for n,v in operations.items()],
                           key=lambda v:v['host_sum_s'], reverse=True)[:25],
                       'optimizer_phases': phase_rows}
                rows.append(row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('source','aten_scopes_within_adam','optimizer_phases')}), flush=True)
                del events, trace
    result = {'schema':'jiuge7b-cpu-adamw-host-trace-review-v1','utc':datetime.now(timezone.utc).isoformat(),
              'rows':rows, 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limitations': ['AdamW scopes include hooks and Python overhead, not isolated CPU FLOPs.',
                             'Nested ATen scopes are not additive.', 'Synchronization spans are not proven exposed critical-path wait.',
                             'No new GPU workload, formal speedup, zero-drop or multi-node claim.']}
    (root / 'cpu-adamw-review.json').write_text(json.dumps(result,indent=2)+'\n')
    (root / 'inspect_cpu_adamw.py').write_bytes(Path(__file__).read_bytes())
    from pinning_key_results import export
    export(PREP,[root],PREP/'cpu-adamw-trace-key-results-20261016.zip')


if __name__ == '__main__':
    main()

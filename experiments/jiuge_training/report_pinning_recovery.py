"""Small read-only untimed progress report; no Torch or GPU API calls."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    state = json.loads((PREP / 'pinning-recovery-20261016-04/controller.json').read_text())
    result = {'time_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'phase': state['status'], 'completed_stages': len(state['runs']), 'ranks': [],
              'formal_performance_started': False}
    for name in ('memory.current', 'memory.peak'):
        path = Path('/sys/fs/cgroup') / name
        if path.exists():
            result[name] = int(path.read_text())
    stats = dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    result['memory_types'] = {key: int(stats[key]) for key in ('anon', 'file', 'shmem') if key in stats}
    result['memory_events'] = dict(line.split() for line in Path('/sys/fs/cgroup/memory.events').read_text().splitlines())
    result['worker_io'] = []
    controller = Path('/proc') / str(state.get('pid', 0))
    if (controller / 'io').exists():
        metrics = dict(line.split(': ', 1) for line in (controller / 'io').read_text().splitlines())
        result['controller_rchar_bytes'] = int(metrics['rchar'])
    pending, seen = [state.get('active_child_pid')], set()
    while pending:
        pid = pending.pop()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        process = Path('/proc') / str(pid)
        try:
            pending.extend(map(int, (process / f'task/{pid}/children').read_text().split()))
            argv = (process / 'cmdline').read_bytes().split(b'\0')
            if any(part.endswith((b'/pinning_resume_compat_entry_v2.py',
                                  b'/verify_pinning_archive.py')) for part in argv):
                io = dict(line.split(': ', 1) for line in (process / 'io').read_text().splitlines())
                result['worker_io'].append({'pid': pid, 'rchar': int(io['rchar'])})
        except (FileNotFoundError, ProcessLookupError):
            pass
    if state['status'].startswith('pinning-'):
        log = PREP / 'runs' / state['status'] / 'stdout.log'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 4000))
                lines = stream.read().decode(errors='replace').splitlines()
            selected = [s.strip() for s in lines if any(key in s for key in
                ('loading distributed checkpoint', 'successfully loaded', 'native_step_complete',
                 'native_step_start', 'iteration ', 'datetime:'))]
            result['latest_training_markers'] = selected[-3:]
    for rank in (0, 1):
        for arm in ('resume-restored', 'resume-next-step'):
            journal = PREP / 'pinning-validation-20261015-04' / arm / f'rank-{rank}/tensor-journal.jsonl'
            if not journal.exists():
                continue
            rows = []
            for line in journal.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # The last active journal line may be incomplete.
            if rows:
                result['ranks'].append({'rank': rank, 'stage': arm, 'fields': len(rows),
                    'failed_fields': sum(not r['passed'] for r in rows),
                    'max_abs': max(r['max_abs'] for r in rows), 'latest_field': rows[-1]['field']})
    if state['status'] == 'stopped':
        result['error'] = state.get('error')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

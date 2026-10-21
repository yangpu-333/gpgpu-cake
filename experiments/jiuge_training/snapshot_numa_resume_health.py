"""Owned untimed-resume process/cgroup counters; no tensor or GPU operations."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
CASE = 'cpu-numa-local8-full-resume-02'


def main():
    folder = PREP / 'runs' / CASE
    control = json.loads((PREP / 'cpu-numa-validation-20261016-02/controller.json').read_text())
    if control['status'] != CASE:
        raise ValueError('Only the currently running untimed resume may be sampled')
    rows = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
            if not any(item.endswith(b'/numa_observation_v2_entry.py') for item in argv[1:3]) or not any(CASE.encode() in item for item in argv):
                continue
            status = dict(line.split(':', 1) for line in (proc / 'status').read_text().splitlines() if ':' in line)
            io = dict(line.split(':', 1) for line in (proc / 'io').read_text().splitlines())
            stats = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
            rows.append({'pid': int(proc.name), 'state': stats[0], 'cpu_user_ticks': int(stats[11]),
                         'cpu_system_ticks': int(stats[12]), 'rss_pages': int(stats[21]),
                         'status': {key: status[key].strip() for key in ('VmRSS', 'RssAnon', 'RssFile', 'RssShmem', 'VmLck', 'Threads', 'Cpus_allowed_list')},
                         'io': {key: int(value) for key, value in io.items()}, 'wchan': (proc / 'wchan').read_text().strip()})
        except (FileNotFoundError, ProcessLookupError):
            continue
    counters = Path('/sys/fs/cgroup')
    memory = {line.split()[0]: int(line.split()[1]) for line in (counters / 'memory.stat').read_text().splitlines()}
    progress = []
    for arm, expected in (('resume-restored', 995), ('resume-next-step', 1195)):
        for rank in (0, 1):
            journal = PREP / 'cpu-numa-validation-20261016-02' / arm / f'rank-{rank}/tensor-journal.jsonl'
            count, failed, maximum, last = 0, 0, 0., None
            if journal.is_file():
                with journal.open() as stream:
                    for line in stream:
                        if not line.endswith('\n'):
                            continue
                        value = json.loads(line)
                        count += 1; failed += int(not value['passed'])
                        maximum = max(maximum, value['max_abs']); last = value['field']
            progress.append({'arm': arm, 'rank': rank, 'fields_recorded': count, 'expected_fields': expected,
                             'failed_recorded_fields': failed, 'max_abs_so_far': maximum,
                             'last_complete_field': last, 'partial_progress_is_completion_proof': False})
    result = {'schema': 'jiuge7b-untimed-resume-health-v1', 'utc': datetime.now(timezone.utc).isoformat(),
              'case': CASE, 'worker_count': len(rows), 'owned_workers': rows,
              'cgroup_memory_current_bytes': int((counters / 'memory.current').read_text()),
              'cgroup_memory_stat': {key: memory[key] for key in ('anon', 'file', 'shmem', 'file_mapped', 'file_dirty', 'file_writeback', 'inactive_file', 'active_file')},
              'memory_events': (counters / 'memory.events').read_text(),
              'cpu_stat': (counters / 'cpu.stat').read_text(), 'gpu_or_tensor_operations': False,
              'audit_field_progress': progress,
              'formal_performance_evidence': False}
    target = folder / ('health-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '.json')
    with target.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

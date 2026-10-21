"""After formal timing, capture independent native/candidate mechanism timelines."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from continue_pinning_validation import remaining_seconds
from continue_pinning_formal import resources

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    performance = json.loads((PREP / 'pinning-formal-20261016-01/controller.json').read_text())
    if performance['status'] not in ('completed_promoted', 'completed_not_promoted'):
        raise ValueError('Complete all required formal rounds before independent profiling')
    if remaining_seconds() < 3600:
        raise TimeoutError('Insufficient remaining free allocation for both diagnostic captures')
    root = PREP / 'pinning-profile-review-20261016-01'
    root.mkdir(exist_ok=False)
    state = {'schema': 'jiuge7b-pinning-post-formal-profile-v1', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'execution_hostname': socket.gethostname(),
             'status': 'initializing', 'formal_controller_sha256': hashlib.sha256(
                 (PREP / 'pinning-formal-20261016-01/controller.json').read_bytes()).hexdigest(),
             'formal_performance_passed': performance['formal_performance_passed'],
             'profile_durations_are_formal_performance': False, 'cases': []}
    def save():
        temp = root / 'controller.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(root / 'controller.json')
    def run(name, argv):
        if remaining_seconds() < 600:
            raise TimeoutError('Free allocation reserve reached')
        with (root / f'{name}.log').open('x') as stream:
            result = subprocess.run([sys.executable, *map(str, argv)], cwd=PREP, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=max(1, remaining_seconds()-600))
        if result.returncode:
            raise ValueError(f'{name} failed; preserve original logs')
    save()
    try:
        state['resource_gate_before'] = resources()
        source = root / 'source'
        source.mkdir()
        state['source_sha256'] = {}
        for name in ('run_pinning_profiles.py', 'run_pinning_case.py', 'pinning_entry.py',
                     'pinning_candidate.py', 'review_native_trace.py', 'analyze_trace.py',
                     'inspect_trace_communication.py', 'archive_case.py'):
            payload = (PREP / name).read_bytes()
            (source / name).write_bytes(payload)
            state['source_sha256'][name] = hashlib.sha256(payload).hexdigest()
        save()
        campaign = PREP / 'pinning-validation-20261016-09'
        run('prepare', [PREP / 'pinning_candidate.py', '--output', campaign])
        summaries = []
        for arm in ('native', 'optimized'):
            if any(hashlib.sha256((PREP / name).read_bytes()).hexdigest() != digest
                   for name, digest in state['source_sha256'].items()):
                raise ValueError('Frozen diagnostic source changed')
            resources()
            state['status'] = 'capturing_' + arm
            save()
            case = f'pinning-{arm}-profile-01'
            case_root = PREP / 'runs' / case
            run(case, [PREP / 'run_pinning_case.py', '--case', case, '--steps', '35', '--seq-length', '2048',
                       '--pinning-arm', arm, '--pinning-campaign', campaign, '--capture-windows'])
            state['cases'].append(case)
            save()
            run(case + '-trace-analysis', [PREP / 'review_native_trace.py', '--case-root', case_root,
                                           '--output', case_root / 'trace-analysis.json'])
            run(case + '-communication-analysis', [PREP / 'inspect_trace_communication.py', '--case-root', case_root,
                                                   '--output', case_root / 'communication-analysis.json'])
            run(case + '-archive', [PREP / 'archive_case.py', '--case-root', case_root, '--create', '--snapshot-analysis'])
            run(case + '-verify-primary-archive', [PREP / 'archive_case.py', '--case-root', case_root,
                '--verify', '--receipt', root / f'{case}-primary-archive-verification.json'])
            communication = json.loads((case_root / 'communication-analysis.json').read_text())
            trace = json.loads((case_root / 'trace-analysis.json').read_text())
            for row in communication['ranks']:
                window = next(item for item in trace['windows'] if item['rank'] == row['rank'] and item['window'] == row['window'])
                summaries.append({'arm': arm, 'rank': row['rank'], 'window': row['window'],
                                  'recorded_memcpy_bytes_by_name': row['recorded_memcpy_bytes_by_name'],
                                  'cpu_phase_ranges': window['cpu_phase_ranges'],
                                  'analysis': window['analysis'], 'clock_review': window['clock_review']})
        with (root / 'mechanism-summary.json').open('x') as stream:
            json.dump({'schema': 'jiuge7b-pinning-mechanism-comparison-v1', 'windows': summaries,
                       'not_a_formal_speedup_measurement': True, 'collector_drop_count_verified': False,
                       'critical_path_proven': False}, stream, indent=2)
        state.update(status='completed', complete_original_traces_location=str(PREP / 'runs'),
                     local_full_trace_copy_required=False, resource_gate_after=resources())
    except BaseException as exc:
        state.update(status='stopped', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save()
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()

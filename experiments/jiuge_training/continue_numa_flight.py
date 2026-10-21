"""After formal closure, capture separate real-training timelines/native sequences."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from continue_pinning_formal import resources
from continue_pinning_validation import remaining_seconds
from numa_allocation_v2 import check as allocation_check
from numa_admission import check as admission_check
from pinning_key_results import export

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation', type=Path, required=True)
    args = parser.parse_args()
    allocation_check(args.allocation)
    formal = json.loads((PREP / 'cpu-numa-formal-20261016-02/controller.json').read_text())
    if formal['status'] not in ('completed_promoted', 'completed_not_promoted'):
        raise ValueError('All formal primary/required-repeat intervals must be closed first')
    admission = admission_check(PREP / 'cpu-numa-validation-20261016-02/formal-admission.json')
    root = PREP / 'cpu-numa-flight-20261016-01'; root.mkdir(exist_ok=False)
    source = root / 'source'; source.mkdir()
    expected = json.loads((PREP / 'cpu-numa-flight-source-manifest.json').read_text())
    names = tuple(dict.fromkeys((*expected, 'continue_numa_flight.py', 'review_native_trace.py',
                  'inspect_trace_communication.py', 'archive_case.py', 'analyze_trace.py')))
    hashes = {}
    for name in names:
        data = (PREP / name).read_bytes(); digest = hashlib.sha256(data).hexdigest()
        if name in expected and expected[name] != digest:
            raise ValueError('Frozen diagnostic source differs: ' + name)
        (source / name).write_bytes(data); hashes[name] = digest
    state = {'schema': 'jiuge7b-numa-native-flight-controller-v1', 'status': 'initializing',
             'started_utc': datetime.now(timezone.utc).isoformat(), 'cases': [], 'source_sha256': hashes,
             'formal_closed_status': formal['status'], 'formal_performance_passed': False,
             'diagnostic_only': True, 'native_sequence_analysis_pending': True,
             'admitted_frozen_source_sha256': admission['frozen_source_sha256']}
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = root / 'controller.tmp'; temp.write_text(json.dumps(state, indent=2) + '\n'); temp.replace(root / 'controller.json')
    def run(stage, argv):
        if remaining_seconds() < 1800:
            raise TimeoutError('Free allocation reserve reached')
        if any(hashlib.sha256((PREP / name).read_bytes()).hexdigest() != digest for name, digest in hashes.items()):
            raise ValueError('Diagnostic source changed')
        state['status'] = stage; save(); print(json.dumps({'event': 'flight_stage_start', 'stage': stage}), flush=True)
        with (root / (stage + '.log')).open('x') as stream:
            subprocess.run([sys.executable, *map(str, argv)], cwd=PREP, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=max(1, remaining_seconds() - 600))
    save()
    try:
        if remaining_seconds() < 4000:
            raise TimeoutError('Reserve for both independent diagnostic cases required')
        for mode in ('unbound', 'local8'):
            gate = resources()
            case = 'cpu-numa-flight-' + mode + '-01'; folder = PREP / 'runs' / case
            campaign = root / ('candidate-' + mode)
            run('prepare-' + mode, [PREP / 'pinning_candidate.py', '--output', campaign])
            run(case, [PREP / 'run_numa_flight_case.py', '--case', case, '--steps', '35', '--seq-length', '2048',
                       '--numa-mode', mode, '--pinning-campaign', campaign, '--capture-windows'])
            after = resources()
            for rank in (0, 1):
                entry = json.loads((folder / f'rank-{rank}/entry.json').read_text())
                states = entry['actual_optimizer_state']['optimizers']
                if (entry['status'] != 'completed' or len(states) != 1 or states[0]['adam_state_entries'] != 199 or
                        states[0]['min_step'] != 35 or states[0]['max_step'] != 35 or
                        [row['step'] for row in entry['completed_native_steps']] != list(range(1, 36)) or
                        any(row['skipped'] for row in entry['completed_native_steps'])):
                    raise ValueError('Full 35 actual unskipped training updates required')
                receipt = json.loads((folder / f'nccl-flight-receipt-rank-{rank}.json').read_text())
                if receipt['entry_count'] <= 0:
                    raise ValueError('Actual native flight entries missing')
            run(case + '-trace', [PREP / 'review_native_trace.py', '--case-root', folder, '--output', folder / 'trace-analysis.json'])
            run(case + '-communication', [PREP / 'inspect_trace_communication.py', '--case-root', folder,
                                         '--output', folder / 'communication-analysis.json'])
            run(case + '-archive', [PREP / 'archive_case.py', '--case-root', folder, '--create', '--snapshot-analysis'])
            run(case + '-verify', [PREP / 'archive_case.py', '--case-root', folder, '--verify',
                                  '--receipt', root / (case + '-primary-verification.json')])
            for pattern in ('case*.json', 'stdout.log', 'stderr.log', 'archive-manifest.json', '*-analysis.json',
                            'numa-binding-rank-*.json', 'nccl-flight-receipt-rank-*.json', 'numa-observer-lifetime-rank-*.json',
                            'rank-*/entry.json', 'rank-*/optimizer-state.json', 'window-*/rank-*-capture.json'):
                for path in folder.glob(pattern):
                    target = root / 'keys' / path.relative_to(PREP)
                    target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
            state['cases'].append({'case': case, 'mode': mode, 'resource_before': gate, 'resource_after': after}); save()
        state['status'] = 'completed_capture_and_primary_archive'; save()
        bundle = PREP / 'cpu-numa-flight-key-results-20261016.zip'
        catalog = export(PREP, [root], bundle)
        print(json.dumps({'status': state['status'], 'bundle': str(bundle),
                          'key_uncompressed_bytes': catalog['key_uncompressed_bytes']}), flush=True)
    except BaseException as exc:
        state.update(status='stopped', error_type=type(exc).__name__, error=str(exc)); save(); raise


if __name__ == '__main__':
    main()

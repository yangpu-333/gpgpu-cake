"""Continue fixed correctness/resume gates after full numerical campaign 04.

Stops before formal timing. A local key-result SHA receipt is needed next.
Does not extend a resource allocation or copy large originals to Windows.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
CODE = ('pinning_semantics_entry.py', 'pinning_resume_audit.py', 'run_pinning_semantics.py',
        'verify_pinning_semantics.py', 'verify_pinning_archive.py', 'pinning_key_results.py',
        'pinning_entry.py', 'pinning_candidate.py', 'pinning_full_audit.py', 'input_evidence.py',
        'jiuge7b_bridge.py', 'run_native_case.py', 'native_entry.py', 'continue_pinning_validation.py')


def remaining_seconds():
    stat = Path('/proc/1/stat').read_text().rsplit(')', 1)[1].split()
    if Path('/proc/1/cmdline').read_bytes().split(b'\0')[:2] != [b'sleep', b'39600']:
        raise ValueError('Unexpected resource lifetime; no extension authorized')
    elapsed = float(Path('/proc/uptime').read_text().split()[0]) - float(stat[19]) / os.sysconf('SC_CLK_TCK')
    return 39600 - elapsed


def main():
    if socket.gethostname() != 'sg-huids25-261015-0d9b8-default0-0':
        raise ValueError('Wrong allocated host')
    root = PREP / 'pinning-followup-20261015-02'
    root.mkdir(exist_ok=False)
    source = root / 'source'
    source.mkdir()
    hashes = {}
    for name in CODE:
        payload = (PREP / name).read_bytes()
        hashes[name] = hashlib.sha256(payload).hexdigest()
        (source / name).write_bytes(payload)
    state = {'schema': 'jiuge7b-pinning-followup-controller-v1', 'pid': os.getpid(),
             'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'waiting_full_numeric_04',
             'source_sha256': hashes, 'runs': [], 'formal_performance_started': False,
             'complete_raw_originals_location': str(PREP), 'local_full_raw_copy_planned': False}
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = root / 'controller.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(root / 'controller.json')
    def command(name, argv, reserve=1800):
        if remaining_seconds() < reserve:
            raise TimeoutError('Insufficient remaining free allocation; stopping before next case')
        if any(hashlib.sha256((PREP / key).read_bytes()).hexdigest() != digest for key, digest in hashes.items()):
            raise ValueError('Frozen followup source changed')
        state['status'] = name
        save()
        with (root / f'{name}.log').open('x') as stream:
            process = subprocess.Popen([sys.executable, *map(str, argv)], cwd=PREP, stdout=stream, stderr=subprocess.STDOUT)
            state['active_child_pid'] = process.pid
            save()
            code = process.wait(timeout=max(1, remaining_seconds() - 600))
        state.pop('active_child_pid', None)
        state['runs'].append({'name': name, 'exit_code': code})
        save()
        if code:
            raise ValueError(f'{name} failed with exit code {code}')
    save()
    try:
        audit = PREP / 'pinning-validation-20261015-04'
        while True:
            status = json.loads((audit / 'controller.json').read_text())
            if status['status'] == 'stopped':
                raise ValueError('Full numeric campaign stopped: ' + status.get('error', 'unknown'))
            if status['status'] == 'numeric_runs_completed_pending_archive_and_resume':
                break
            if remaining_seconds() < 1800:
                raise TimeoutError('Free allocation reserve reached while waiting for audit')
            time.sleep(20)
        semantics = PREP / 'pinning-semantics-20261015-02'
        semantics.mkdir(exist_ok=False)
        candidates = PREP / 'pinning-validation-20261015-05'
        checkpoint_candidates = PREP / 'pinning-validation-20261015-06'
        native_resume_candidates = PREP / 'pinning-validation-20261015-07'
        for number, folder in (('05', candidates), ('06', checkpoint_candidates), ('07', native_resume_candidates)):
            command('prepare_' + number, [PREP / 'pinning_candidate.py', '--output', folder])
        cases = []
        for arm in ('native', 'control', 'optimized'):
            name = f'pinning-semantics-{arm}-02'
            cases.append(name)
            command(name, [PREP / 'run_pinning_semantics.py', '--case', name, '--steps', '3',
                          '--seq-length', '2048', '--pinning-arm', arm, '--pinning-campaign', candidates,
                          '--semantics-root', semantics / f'semantics-{arm}'], reserve=1800)
        name = 'pinning-native-checkpoint-02'
        cases.append(name)
        checkpoint = semantics / 'checkpoints/native-step2'
        command(name, [PREP / 'run_pinning_semantics.py', '--case', name, '--steps', '3', '--seq-length', '2048',
                      '--pinning-arm', 'native', '--pinning-campaign', checkpoint_candidates,
                      '--semantics-root', semantics / 'checkpoint-native', '--save-dir', checkpoint,
                      '--exit-interval', '2'], reserve=4200)
        if (checkpoint / 'latest_checkpointed_iteration.txt').read_text().strip() != '2':
            raise ValueError('Native checkpoint is not at iteration 2')
        name = 'pinning-native-resume-02'
        cases.append(name)
        command(name, [PREP / 'run_pinning_semantics.py', '--case', name, '--steps', '3', '--seq-length', '2048',
                      '--pinning-arm', 'native', '--pinning-campaign', native_resume_candidates,
                      '--semantics-root', semantics / 'resume-native', '--load-dir', checkpoint], reserve=1800)
        name = 'pinning-optimized-resume-02'
        cases.append(name)
        command(name, [PREP / 'run_pinning_semantics.py', '--case', name, '--steps', '3', '--seq-length', '2048',
                      '--pinning-arm', 'optimized', '--pinning-campaign', checkpoint_candidates,
                      '--semantics-root', semantics / 'resume-optimized', '--load-dir', checkpoint,
                      '--resume-audit-campaign', audit], reserve=7800)
        command('verify_semantics', [PREP / 'verify_pinning_semantics.py', '--root', semantics,
                                    '--resume-campaign', audit, '--output', semantics / 'semantics-verification.json'])
        command('verify_primary_raw', [PREP / 'verify_pinning_archive.py', '--root', audit,
                                      '--include-resume', '--output', audit / 'primary-archive-verification.json'], reserve=4200)
        summary = root / 'summary'
        summary.mkdir()
        (summary / 'numeric-completion-before-key-export.json').write_text(json.dumps({
            'full_correctness_resume_and_primary_archive_passed': True,
            'local_key_results_verified': False, 'formal_performance_started': False,
            'created_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
        bundle = PREP / 'pinning-complete-key-results-20261015-v2.zip'
        # Do not bundle the controller/log that is actively writing this export.
        folders = [audit, semantics, candidates, checkpoint_candidates, native_resume_candidates, source, summary]
        folders += [PREP / 'runs' / f'pinning-{arm}-audit-04' for arm in ('native', 'control', 'optimized')]
        folders += [PREP / 'runs' / case for case in cases]
        export = [PREP / 'pinning_key_results.py', '--root', PREP, '--output', bundle]
        for folder in folders:
            export += ['--folder', folder]
        command('export_key_results', export)
        state.update(status='ready_for_local_key_verification', key_results_bundle=str(bundle),
                     full_correctness_resume_and_primary_archive_passed=True,
                     next_gate='Download only key bundle; verify locally before formal timing')
    except BaseException as exc:
        state.update(status='stopped', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save()
        print(json.dumps({k: v for k, v in state.items() if k != 'source_sha256'}), flush=True)


if __name__ == '__main__':
    main()

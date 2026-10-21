"""One bounded experiment coordinator; transfer only small verified key bundles.

Continues the already-running ACT experiment. No resource extension, scheduler,
external model calls, bulk tensor download, or deletion is performed here.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from pinning_key_results import verify

BASE = Path(__file__).resolve().parent
PREP = '/home/huids25/kda-jiuge-prep-20261014'
HOST = 'huids25@192.168.5.194'
KEY = str(Path.home() / '.ssh/id_ed25519')
SSH = ['ssh', '-i', KEY, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=12', '-o', 'StrictHostKeyChecking=yes', '-p', '32405', HOST]
SCP = ['scp', '-i', KEY, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=12', '-o', 'StrictHostKeyChecking=yes', '-P', '32405']
MAX_BUNDLE = 68 * 1024**2


def remote(command, timeout=45):
    result = subprocess.run([*SSH, command], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'SSH command failed ({result.returncode}): {result.stderr[-3000:]}')
    return result.stdout


def main():
    folder = BASE / 'results/pinning-driver-20261015-02'
    folder.mkdir(exist_ok=False)
    shutil.copyfile(__file__, folder / Path(__file__).name)
    state = {'schema': 'jiuge7b-pinning-local-coordinator-v1', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'status': 'checking', 'local_large_raw_download_enabled': False,
             'key_bundles': [], 'formal_performance_passed': False}
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = folder / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        temp.replace(folder / 'status.json')
    def fetch_bundle(remote_path, name):
        if remote_path != f'{PREP}/{name}.zip':
            raise ValueError('Unexpected key bundle path')
        size = int(remote(f'stat -c %s {remote_path}').strip())
        if size <= 0 or size > MAX_BUNDLE or len(state['key_bundles']) >= 2:
            raise ValueError('Local key-result size/count cap exceeded')
        target = folder / f'{name}.zip'
        subprocess.run([*SCP, f'{HOST}:{remote_path}', str(target)], check=True, timeout=600)
        if target.stat().st_size != size:
            raise ValueError('Incomplete key bundle transfer')
        receipt = folder / f'{name}-verification.json'
        proof = verify(target, receipt)
        state['key_bundles'].append({'file': target.name, 'compressed_bytes': size,
                                    'verified_key_bytes': proof['verified_key_bytes'],
                                    'sha256': proof['bundle_sha256']})
        save()
        return receipt
    save()
    try:
        if remote('hostname').strip() != 'sg-huids25-261015-0d9b8-default0-0':
            raise ValueError('SSH points to a different allocation')
        seconds = float(remote(f"cd {PREP} && .venv/bin/python -c 'from continue_pinning_validation import remaining_seconds; print(remaining_seconds())'"))
        deadline = time.monotonic() + max(0, seconds - 600)
        followup_path = f'{PREP}/pinning-followup-20261015-02/controller.json'
        previous = None
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Existing free allocation reserve reached; no extension')
            try:
                followup = json.loads(remote(f'cat {followup_path}'))
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                state.update(status='waiting_network', last_network_error=str(exc))
                save()
                time.sleep(30)
                continue
            phase = followup['status']
            if phase != previous:
                print(json.dumps({'phase': phase, 'utc': datetime.now(timezone.utc).isoformat()}), flush=True)
                previous = phase
            state.update(status='waiting_remote_validation', remote_validation_status=phase)
            save()
            if phase == 'stopped':
                (folder / 'remote-validation-stopped.json').write_text(json.dumps(followup, indent=2) + '\n')
                raise ValueError('Remote validation stopped: ' + followup.get('error', 'unknown'))
            if phase == 'ready_for_local_key_verification':
                break
            time.sleep(30)
        name = 'pinning-complete-key-results-20261015-v2'
        receipt = fetch_bundle(followup['key_results_bundle'], name)
        remote_receipt = f'{PREP}/local-pinning-complete-key-verification-20261015.json'
        remote(f'test ! -e {remote_receipt}')
        subprocess.run([*SCP, str(receipt), f'{HOST}:{remote_receipt}'], check=True, timeout=60)
        admission = f'{PREP}/pinning-formal-admission-20261015.json'
        command = (f'cd {PREP} && .venv/bin/python admit_pinning_formal.py '
                   f'--primary {PREP}/pinning-validation-20261015-04/primary-archive-verification.json '
                   f'--semantics {PREP}/pinning-semantics-20261015-02/semantics-verification.json '
                   f'--key-bundle {followup["key_results_bundle"]} --local-key-receipt {remote_receipt} --output {admission}')
        (folder / 'formal-admission.json').write_text(remote(command), encoding='utf-8')
        state.update(status='running_formal', all_correctness_and_local_key_gates_passed=True)
        save()
        # Only the formal controller runs now. No polling/capture/backup during timing.
        with (folder / 'formal-controller-stdout.log').open('x') as stdout, (folder / 'formal-controller-stderr.log').open('x') as stderr:
            result = subprocess.run([*SSH, f'cd {PREP} && .venv/bin/python -u continue_pinning_formal.py --admission {admission}'],
                                    stdout=stdout, stderr=stderr, timeout=max(1, deadline-time.monotonic()))
        formal = json.loads(remote(f'cat {PREP}/pinning-formal-20261015-01/controller.json'))
        (folder / 'formal-controller-final.json').write_text(json.dumps(formal, indent=2) + '\n', encoding='utf-8')
        state.update(remote_formal_status=formal['status'], formal_exit_code=result.returncode,
                     formal_performance_passed=formal['formal_performance_passed'])
        save()
        # The controller has returned; take closed logs and completed cases only.
        export = (f'cd {PREP} && .venv/bin/python pinning_key_results.py --root {PREP} '
                  f'--folder {PREP}/pinning-formal-20261015-01 '
                  f'--output {PREP}/pinning-formal-key-results-20261015.zip')
        for case in formal['completed_cases']:
            if not case.startswith('pinning-formal-') or not case.replace('-', '').isalnum():
                raise ValueError('Unexpected completed formal case name')
            export += f' --folder {PREP}/runs/{case}'
        remote(export, timeout=300)
        fetch_bundle(f'{PREP}/pinning-formal-key-results-20261015.zip', 'pinning-formal-key-results-20261015')
        if result.returncode or not formal['status'].startswith('completed_'):
            raise ValueError('Formal run incomplete; original timings retained without promotion')
        state['status'] = formal['status']
    except BaseException as exc:
        state.update(status='stopped', error_type=type(exc).__name__, error=str(exc), formal_performance_passed=False)
        raise
    finally:
        save()
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()

"""Continue independent mechanism tracing after the complete formal campaign.

Waits on a local receipt only during formal timing. Transfers one bounded key
bundle; original trace bodies stay on ACT. No allocation extension or deletion.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from pinning_key_results import verify

BASE = Path(__file__).resolve().parent
PREP = '/home/huids25/kda-jiuge-prep-20261014'
HOST = 'huids25@192.168.5.194'
FOLDER = BASE / 'results/pinning-profile-driver-20261016-01'
FORMAL_STATUS = BASE / 'results/pinning-driver-20261016-05/status.json'
PROFILE_ROOT = PREP + '/pinning-profile-review-20261016-01'
OPTIONS = ['-i', str(Path.home() / '.ssh/id_ed25519'),
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
           '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
           '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6',
           '-o', 'UserKnownHostsFile=' + str(BASE / 'results/act-known-hosts-20261016')]
SSH = ['ssh', *OPTIONS, '-p', '31457', HOST]
SCP = ['scp', *OPTIONS, '-P', '31457']


def remote(command, timeout=60):
    result = subprocess.run([*SSH, command], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError('ACT command failed: ' + result.stderr[-2500:])
    return result.stdout


def main():
    FOLDER.mkdir(exist_ok=False)
    shutil.copyfile(__file__, FOLDER / Path(__file__).name)
    payload = (BASE / 'run_pinning_profiles.py').read_bytes()
    (FOLDER / 'expected-profile-source.py').write_bytes(payload)
    expected_sha = hashlib.sha256(payload).hexdigest()
    state = {'schema': 'jiuge7b-post-formal-profile-local-driver-v1',
             'started_utc': datetime.now(timezone.utc).isoformat(),
             'status': 'waiting_local_formal_completion',
             'remote_profile_source_sha256': expected_sha,
             'remote_probes_during_formal': False, 'full_raw_download_enabled': False,
             'formal_performance_measurement': False}

    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = FOLDER / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        temp.replace(FOLDER / 'status.json')

    save()
    try:
        # No ACT connection until the original coordinator has closed formal IO.
        deadline = datetime.fromisoformat('2026-10-16T11:10:00+00:00')
        while True:
            if datetime.now(timezone.utc) >= deadline:
                raise TimeoutError('Free allocation reserve reached; no extension')
            formal = json.loads(FORMAL_STATUS.read_text(encoding='utf-8'))
            if formal['status'] in ('completed_promoted', 'completed_not_promoted'):
                break
            if formal['status'] == 'stopped':
                raise ValueError('Formal coordinator stopped; no independent capture started')
            time.sleep(30)
        state.update(status='checking_remote', formal_status=formal['status'])
        save()
        if remote('hostname').strip() != 'sg-huids25-261016-24e4c-default0-0':
            raise ValueError('Wrong ACT allocation')
        actual_sha = remote('sha256sum ' + PREP + '/run_pinning_profiles.py').split()[0]
        if actual_sha != expected_sha:
            raise ValueError('Staged profile controller differs from the recorded source')
        state['status'] = 'running_independent_profiles'
        save()
        with (FOLDER / 'profile-controller-stdout.log').open('x') as stdout, \
                (FOLDER / 'profile-controller-stderr.log').open('x') as stderr:
            result = subprocess.run([*SSH, 'cd ' + PREP + ' && .venv/bin/python -u run_pinning_profiles.py'],
                                    stdout=stdout, stderr=stderr,
                                    timeout=max(1, (deadline - datetime.now(timezone.utc)).total_seconds()))
        controller = json.loads(remote('cat ' + PROFILE_ROOT + '/controller.json'))
        (FOLDER / 'remote-controller-final.json').write_text(json.dumps(controller, indent=2) + '\n', encoding='utf-8')
        if result.returncode or controller['status'] != 'completed':
            raise ValueError('Independent capture incomplete; preserve its original evidence')
        name = 'pinning-profile-key-results-20261016'
        bundle = PREP + '/' + name + '.zip'
        command = ('cd ' + PREP + ' && .venv/bin/python pinning_key_results.py --root ' + PREP
                   + ' --folder ' + PROFILE_ROOT + ' --output ' + bundle)
        if set(controller['cases']) != {'pinning-native-profile-01', 'pinning-optimized-profile-01'}:
            raise ValueError('Unexpected independent profile cases')
        for case in controller['cases']:
            command += ' --folder ' + PREP + '/runs/' + case
        remote(command, timeout=300)
        size = int(remote('stat -c %s ' + bundle).strip())
        if not 0 < size <= 68 * 1024**2:
            raise ValueError('Key bundle exceeds the local budget')
        target = FOLDER / (name + '.zip')
        subprocess.run([*SCP, HOST + ':' + bundle, str(target)], check=True, timeout=600)
        if target.stat().st_size != size:
            raise ValueError('Incomplete key bundle download')
        receipt = FOLDER / 'key-bundle-verification.json'
        proof = verify(target, receipt)
        remote('test ! -e ' + PROFILE_ROOT + '/local-key-verification.json')
        subprocess.run([*SCP, str(receipt), HOST + ':' + PROFILE_ROOT + '/local-key-verification.json'],
                       check=True, timeout=60)
        state.update(status='completed', key_bundle_sha256=proof['bundle_sha256'],
                     verified_key_files=proof['verified_key_files'],
                     verified_key_bytes=proof['verified_key_bytes'],
                     original_traces_location=PREP + '/runs', local_full_trace_copy_verified=False)
    except BaseException as exc:
        state.update(status='stopped', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save()
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()

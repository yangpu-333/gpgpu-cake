"""Restart-safe local key handoff and post-review; no GPU or raw-data access."""
from datetime import datetime, timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time

from drive_sft_pinning_validation import PREP, ROOT, SSH, SCP, remote_json, run
from pinning_key_results import verify

BASE = Path(__file__).resolve().parent
LOCAL = BASE / 'results/sft-pinning-driver-20261020-02'
DEADLINE = datetime.fromisoformat('2026-10-20T20:00:00+08:00').timestamp()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def download(remote, name):
    allowed = {PREP + '/sft-pinning-correctness-key-results-20261020.zip',
               PREP + '/sft-pinning-complete-key-results-20261020.zip'}
    if remote not in allowed:
        raise ValueError('Only the two closed, bounded key bundles are eligible')
    target = LOCAL / name
    if target.exists():
        return target
    temporary = LOCAL / (name + '.partial-' + str(time.time_ns()))
    run([*SCP, 'huids25@192.168.5.200:' + remote, str(temporary)])
    temporary.rename(target)
    return target


def checked_keys(bundle, name):
    receipt = LOCAL / name
    if not receipt.exists():
        return verify(bundle, receipt, 64 * 1024**2)
    # Recheck the actual ZIP after a restart without overwriting the old proof.
    fresh = LOCAL / (receipt.stem + '-recheck-' + str(time.time_ns()) + '.json')
    checked = verify(bundle, fresh, 64 * 1024**2)
    if checked != read(receipt):
        raise ValueError('Existing local key-verification receipt differs')
    return checked


def main():
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL / 'driver.lock').open('a+b') as lock:
        lock.seek(0, 2)
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('The existing local driver holds the lock; no duplicate started.', flush=True)
            return 0
        return observe()


def observe():
    status_path = LOCAL / 'status.json'
    state = read(status_path) if status_path.exists() else {
        'schema': 'jiuge7b-sft-pinning-local-driver-v2',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'local_large_originals_downloaded': False,
        'recovery_of': 'sft-pinning-driver-20261020-01',
    }
    if state.get('status') == 'completed' and state.get('independent_review_completed'):
        print('Closed independent review already exists; no repeated work.', flush=True)
        return 0

    def save(status, **values):
        state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(),
                     pid=os.getpid(), **values)
        atomic_json(status_path, state)
        print(json.dumps(state, ensure_ascii=True), flush=True)

    save('resumed-observation')
    while time.time() < DEADLINE:
        try:
            controller = remote_json(ROOT + '/controller.json')
            if controller is None:
                time.sleep(20)
                continue
            atomic_json(LOCAL / 'remote-controller-latest.json', controller)
            phase = controller['status']
            save('observing', remote_status=phase)
            if phase == 'ready-for-local-key-verification':
                bundle = download(controller['key_results_bundle'], 'sft-correctness-key-results.zip')
                checked = checked_keys(bundle, 'local-key-verification.json')
                existing = remote_json(ROOT + '/local-key-verification.json')
                if existing is None:
                    run([*SCP, str(LOCAL / 'local-key-verification.json'),
                         'huids25@192.168.5.200:' + ROOT + '/local-key-verification.json.uploading'])
                    run([*SSH, 'mv ' + ROOT + '/local-key-verification.json.uploading ' +
                         ROOT + '/local-key-verification.json'])
                elif existing != checked:
                    raise ValueError('ACT admission receipt differs from the independently checked local keys')
                save('local-correctness-keys-verified-gate-sent',
                     key_files=checked['verified_key_files'],
                     key_uncompressed_bytes=checked['verified_key_bytes'],
                     bundle_sha256=checked['bundle_sha256'])
            elif phase == 'stopped':
                save('remote-stopped', error=controller.get('error'), formal_performance_passed=False)
                return 2
            elif phase == 'completed':
                final = remote_json(ROOT + '/final-key-bundle.json')
                if final is None:
                    time.sleep(5)
                    continue
                bundle = download(final['path'], 'sft-complete-key-results.zip')
                if (bundle.stat().st_size != final['bytes'] or
                        hashlib.sha256(bundle.read_bytes()).hexdigest() != final['sha256']):
                    raise ValueError('ACT/local final key bundle size or SHA differs')
                checked = checked_keys(bundle, 'final-key-verification.json')
                # The reviewer requires status=completed; promotion remains unaccepted locally
                # until its independent timing, source and actual logged-loss checks close.
                save('completed', independent_review_completed=False,
                     formal_performance_passed=controller['formal_performance_passed'],
                     final_key_files=checked['verified_key_files'],
                     final_key_bytes=checked['verified_key_bytes'])
                review = BASE / 'results/sft-pinning-independent-review-20261020.json'
                if not review.exists():
                    code = ('from pathlib import Path; import review_sft_pinning_results as r; '
                            'r.DRIVER=Path(__import__("sys").argv[1]); r.main()')
                    env = dict(os.environ, PYTHONIOENCODING='utf-8')
                    with (LOCAL / 'independent-review.stdout.log').open('a', encoding='utf-8') as out, \
                            (LOCAL / 'independent-review.stderr.log').open('a', encoding='utf-8') as err:
                        result = subprocess.run([sys.executable, '-u', '-c', code, str(LOCAL)],
                                                cwd=BASE, env=env, stdout=out, stderr=err,
                                                timeout=1800, check=False)
                    if result.returncode:
                        save('independent-review-failed', exit_code=result.returncode)
                        return result.returncode
                result = read(review)
                save('completed', independent_review_completed=True,
                     formal_performance_passed=result['formal_performance_passed'],
                     independent_review=str(review))
                return 0
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            save('bridge-or-transfer-temporarily-unavailable', error=str(error))
        except BaseException as error:
            save('local-validation-failed', error_type=type(error).__name__, error=str(error))
            raise
        time.sleep(30)
    save('local-observation-window-ended')
    return 3


if __name__ == '__main__':
    raise SystemExit(main())

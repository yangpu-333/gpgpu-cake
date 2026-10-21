"""Key-only local handoff after a completed SFT audit hits its time reserve."""
from datetime import datetime, timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time

from drive_sft_pinning_validation import PREP, SSH, SCP, remote_json, run
from pinning_key_results import verify

BASE = Path(__file__).resolve().parent
LOCAL = BASE/'results/sft-pinning-formal-driver-20261020-03'
PARENT = PREP+'/sft-pinning-validation-20261020-01'
ROOT = PREP+'/sft-pinning-formal-continuation-20261020-01'
PARENT_BUNDLE = PREP+'/sft-pinning-correctness-key-results-20261020.zip'
FINAL_BUNDLE = PREP+'/sft-pinning-complete-key-results-20261020-continuation.zip'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def atomic_json(path, data):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def get_keys(remote, name, receipt_name):
    if remote not in (PARENT_BUNDLE, FINAL_BUNDLE):
        raise ValueError('Only the exact two bounded key bundles are eligible')
    path = LOCAL/name
    if not path.exists():
        partial = LOCAL/(name+'.partial-'+str(time.time_ns()))
        run([*SCP,'huids25@192.168.5.200:'+remote,str(partial)])
        partial.rename(path)
    receipt = LOCAL/receipt_name
    destination = receipt if not receipt.exists() else LOCAL/(receipt.stem+'-recheck-'+str(time.time_ns())+'.json')
    checked = verify(path,destination,64*1024**2)
    if destination != receipt and checked != read(receipt):
        raise ValueError('Closed key-verification receipt changed')
    return path, checked


def main():
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL/'driver.lock').open('a+b') as lock:
        lock.seek(0,2)
        if not lock.tell():
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            print('Existing continuation coordinator holds the lock.',flush=True)
            return 0
        return observe()


def observe():
    state_path = LOCAL/'status.json'
    state = read(state_path) if state_path.exists() else {
        'schema':'jiuge7b-sft-formal-local-continuation-v1',
        'started_utc':datetime.now(timezone.utc).isoformat(),
        'local_large_originals_downloaded':False}

    def save(status,**values):
        state.update(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),**values)
        if status!='bridge-temporarily-unavailable':
            state.pop('transient_error',None)
        atomic_json(state_path,state)
        print(json.dumps(state,ensure_ascii=True),flush=True)

    if state.get('status')=='completed' and state.get('independent_review_completed'):
        return 0
    save('waiting-for-closed-parent-correctness')
    parent_key_sent = state.get('parent_local_key_sent',False)
    until = time.monotonic()+18*3600
    while time.monotonic()<until:
        try:
            if not parent_key_sent:
                parent = remote_json(PARENT+'/controller.json')
                if parent is None:
                    time.sleep(30)
                    continue
                atomic_json(LOCAL/'parent-controller-latest.json',parent)
                if parent['formal_performance_started']:
                    save('continuation-not-needed-original-formal-started')
                    return 0
                if parent['status']!='stopped':
                    save('waiting-for-closed-parent-correctness',parent_status=parent['status'])
                    time.sleep(30)
                    continue
                if (parent.get('error_type')!='TimeoutError' or parent.get('error') not in {
                        'Local key gate not closed with three-hour formal reserve',
                        'Insufficient authorized free time for next stage'}
                    or not parent.get('full_correctness_resume_and_primary_archive_passed')
                    or any(row['exit_code'] for row in parent['commands'])):
                    save('parent-validation-not-eligible',parent_error=parent.get('error'))
                    return 2
                _, checked = get_keys(PARENT_BUNDLE,'sft-correctness-key-results.zip','parent-local-key-verification.json')
                old_receipt = remote_json(PARENT+'/local-key-verification.json')
                if old_receipt is None:
                    receipt = LOCAL/'parent-local-key-verification.json'
                    run([*SCP,str(receipt),'huids25@192.168.5.200:'+PARENT+'/local-key-verification.json.uploading'])
                    run([*SSH,'mv '+PARENT+'/local-key-verification.json.uploading '+PARENT+'/local-key-verification.json'])
                elif (not old_receipt.get('passed') or
                      old_receipt.get('bundle_sha256')!=checked['bundle_sha256'] or
                      old_receipt.get('verified_key_files')!=checked['verified_key_files']):
                    raise ValueError('Existing parent admission receipt differs from actual closed keys')
                parent_key_sent = True
                save('parent-correctness-keys-verified-waiting-resource',parent_local_key_sent=True,
                     parent_key_files=checked['verified_key_files'],parent_key_bytes=checked['verified_key_bytes'],
                     parent_key_bundle_sha256=checked['bundle_sha256'])
            controller = remote_json(ROOT+'/controller.json')
            if controller is None:
                time.sleep(30)
                continue
            atomic_json(LOCAL/'remote-controller-latest.json',controller)
            phase = controller['status']
            save('observing',remote_status=phase)
            if phase=='stopped':
                save('continuation-stopped',error=controller.get('error'),formal_performance_passed=False)
                return 2
            if phase=='completed':
                final = remote_json(ROOT+'/final-key-bundle.json')
                if final is None:
                    time.sleep(5)
                    continue
                if final['path']!=FINAL_BUNDLE:
                    raise ValueError('Final bounded bundle identity differs')
                bundle, checked = get_keys(FINAL_BUNDLE,'sft-complete-key-results.zip','final-key-verification.json')
                if bundle.stat().st_size!=final['bytes'] or hashlib.sha256(bundle.read_bytes()).hexdigest()!=final['sha256']:
                    raise ValueError('ACT/local final key bundle bytes differ')
                save('completed',independent_review_completed=False,
                     formal_performance_passed=controller['formal_performance_passed'],
                     final_key_files=checked['verified_key_files'],final_key_bytes=checked['verified_key_bytes'])
                review = BASE/'results/sft-pinning-independent-review-20261020-continuation.json'
                if not review.exists():
                    with (LOCAL/'independent-review.stdout.log').open('a',encoding='utf-8') as out, \
                            (LOCAL/'independent-review.stderr.log').open('a',encoding='utf-8') as err:
                        result = subprocess.run([sys.executable,'-u',str(BASE/'review_sft_pinning_continuation_20261020.py')],
                                                cwd=BASE,env=dict(os.environ,PYTHONIOENCODING='utf-8'),
                                                stdout=out,stderr=err,timeout=1800,check=False)
                    if result.returncode:
                        save('independent-review-failed',exit_code=result.returncode)
                        return result.returncode
                checked_review = read(review)
                save('completed',independent_review_completed=True,
                     formal_performance_passed=checked_review['formal_performance_passed'],
                     independent_review=str(review))
                return 0
        except (RuntimeError,subprocess.TimeoutExpired,json.JSONDecodeError) as error:
            save('bridge-temporarily-unavailable',transient_error=str(error))
        except BaseException as error:
            save('local-validation-failed',error_type=type(error).__name__,error=str(error))
            raise
        time.sleep(30)
    save('local-observation-window-ended')
    return 3


if __name__=='__main__':
    raise SystemExit(main())

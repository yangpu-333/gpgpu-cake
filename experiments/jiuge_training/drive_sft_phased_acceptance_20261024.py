"""One bounded-key Windows observer; current GPU process is never relaunched here."""
from datetime import datetime, timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import time

from drive_sft_pinning_validation import PREP, SSH, SCP, run, remote_json
from pinning_key_results import verify

BASE = Path(__file__).resolve().parent
LOCAL = BASE/'results/sft-phased-acceptance-driver-20261024-01'
ROOT = PREP+'/sft-phased-acceptance-20261024-01'


def main():
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL/'driver.lock').open('a+b') as lock:
        lock.seek(0); lock.write(b'1'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        status_path = LOCAL/'status.json'
        state = json.loads(status_path.read_text()) if status_path.exists() else {}
        if state.get('status') in ('completed','remote-stopped'):
            raise ValueError('Immutable closed driver; do not restart it')
        state.update(pid=os.getpid(),raw_originals_downloaded=False)
        def save(status, **extra):
            state.update(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
            temp = LOCAL/'status.json.tmp'
            temp.write_text(json.dumps(state,indent=2)+'\n')
            temp.replace(status_path)
        def receive(remote, name, receipt_name):
            destination = LOCAL/name
            receipt = LOCAL/receipt_name
            if not destination.exists():
                run([*SCP,'huids25@192.168.5.200:'+remote,str(destination)])
            if not receipt.exists():
                result = verify(destination,receipt,64*1024**2)
            else:
                result = json.loads(receipt.read_text())
                if hashlib.sha256(destination.read_bytes()).hexdigest()!=result['bundle_sha256']:
                    raise ValueError('Closed local bundle differs')
            return destination, receipt, result
        limit = time.monotonic()+12*3600
        while time.monotonic()<limit:
            try:
                controller = remote_json(ROOT+'/controller.json')
                if controller is None:
                    save('waiting-for-controller')
                    time.sleep(30)
                    continue
                (LOCAL/'remote-controller-latest.json').write_text(json.dumps(controller,indent=2)+'\n')
                save('observing', remote_status=controller['status'],completed_cases=controller['completed_cases'])
                if controller['status']=='ready-for-local-key-verification':
                    _,receipt,result = receive(controller['key_results_bundle'],'correctness-key-results.zip','local-key-verification.json')
                    remote_receipt = remote_json(ROOT+'/local-key-verification.json')
                    if remote_receipt is None:
                        run([*SCP,str(receipt),'huids25@192.168.5.200:'+ROOT+'/local-key-verification.json'])
                    elif remote_receipt!=result:
                        raise ValueError('Existing ACT gate differs; do not overwrite')
                    save('correctness-key-gate-closed',key_files=result['verified_key_files'])
                if controller['status']=='stopped':
                    save('remote-stopped',error=controller.get('error'),formal_performance_passed=False)
                    return 2
                if controller['status']=='completed':
                    bundle = remote_json(ROOT+'/final-key-bundle.json')
                    if bundle is None:
                        time.sleep(10)
                        continue
                    path,_,result = receive(bundle['path'],'final-key-results.zip','final-key-verification.json')
                    if path.stat().st_size!=bundle['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=bundle['sha256']:
                        raise ValueError('ACT/local final bundle SHA differs')
                    save('completed',formal_performance_passed=controller['formal_performance_passed'],
                         key_files=result['verified_key_files'],key_bytes=result['verified_key_bytes'])
                    return 0
            except (subprocess.TimeoutExpired,RuntimeError) as error:
                save('bridge-temporarily-unavailable',error=str(error)[-1200:])
            time.sleep(30)
        save('observer-window-ended')
        return 3


if __name__=='__main__':
    raise SystemExit(main())

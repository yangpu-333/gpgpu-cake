"""Observe shared ACT status; download and verify only a bounded key bundle."""
from datetime import datetime,timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path,PurePosixPath
import subprocess
import time
import zipfile
from drive_sft_pinning_validation import SSH,SCP,PREP,run,remote_json
from pinning_key_results import verify
import pinning_key_results
pinning_key_results.KEY_SUFFIXES = pinning_key_results.KEY_SUFFIXES | {'.sh'}

BASE=Path(__file__).resolve().parent
LOCAL=BASE/'results/sft-moment-screen-driver-20261021-01'
ROOT=PREP+'/sft-moment-screen-20261021-01'


def main():
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL/'driver.lock').open('a+b') as lock:
        if lock.tell()==0:
            lock.write(b'1')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        state={'schema':'jiuge7b-sft-moment-local-driver-v1','pid':os.getpid(),
               'status':'observing','raw_originals_downloaded':False,'local_key_limit_bytes':32*1024**2}
        def save(status,**extra):
            state.update(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
            tmp=LOCAL/'status.tmp'
            tmp.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
            tmp.replace(LOCAL/'status.json')
        save('observing')
        deadline=time.monotonic()+4*3600
        while time.monotonic()<deadline:
            try:
                controller=remote_json(ROOT+'/controller.json')
                if controller is None:
                    save('waiting-for-controller')
                    time.sleep(30)
                    continue
                (LOCAL/'remote-controller-latest.json').write_text(json.dumps(controller,indent=2)+'\n',encoding='utf-8')
                save('observing',remote_status=controller['status'],completed_cases=len(controller['cases']))
                if controller['status']=='stopped':
                    save('remote-stopped',error=controller.get('error'))
                    return 2
                if controller['status']=='completed':
                    receipt=remote_json(ROOT+'/final-key-bundle.json')
                    if receipt is None:
                        time.sleep(10)
                        continue
                    if not 0<receipt['bytes']<=32*1024**2:
                        raise ValueError('Local key bundle cap exceeded')
                    bundle=LOCAL/'sft-moment-screen-key-results.zip'
                    if not bundle.exists():
                        partial=LOCAL/'sft-moment-screen-key-results.zip.partial'
                        run([*SCP,'huids25@192.168.5.200:'+receipt['path'],str(partial)])
                        if partial.stat().st_size!=receipt['bytes'] or hashlib.sha256(partial.read_bytes()).hexdigest()!=receipt['sha256']:
                            raise ValueError('ACT/local key bundle SHA differs')
                        partial.replace(bundle)
                    if hashlib.sha256(bundle.read_bytes()).hexdigest()!=receipt['sha256']:
                        raise ValueError('Existing closed key bundle differs')
                    verification=LOCAL/'local-key-verification.json'
                    if verification.exists():
                        checked=json.loads(verification.read_text(encoding='utf-8'))
                        assert checked['bundle_sha256']==receipt['sha256'] and checked['passed']
                    else:
                        checked=verify(bundle,verification,32*1024**2)
                    extracted=LOCAL/'keys'
                    with zipfile.ZipFile(bundle) as archive:
                        catalog=json.loads(archive.read('key-results-manifest.json'))
                        for name in catalog['files']:
                            relative=PurePosixPath(name)
                            if relative.is_absolute() or '..' in relative.parts:
                                raise ValueError('Unsafe key path')
                            target=extracted.joinpath(*relative.parts)
                            payload=archive.read(name)
                            if target.exists():
                                if target.read_bytes()!=payload: raise ValueError('Closed extracted key differs')
                            else:
                                target.parent.mkdir(parents=True,exist_ok=True)
                                with target.open('xb') as stream: stream.write(payload)
                    save('completed',verified_key_files=checked['verified_key_files'],verified_key_bytes=checked['verified_key_bytes'],bundle_sha256=receipt['sha256'])
                    return 0
            except (RuntimeError,subprocess.TimeoutExpired) as exc:
                save('bridge-temporarily-unavailable',error=str(exc))
            time.sleep(60)
        save('observation-deadline-reached')
        return 3


if __name__=='__main__':
    raise SystemExit(main())

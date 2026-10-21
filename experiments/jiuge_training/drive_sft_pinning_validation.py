"""Local bounded-key handoff through the retained V100 shared-storage bridge."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

from pinning_key_results import verify

BASE = Path(__file__).resolve().parent
LOCAL = BASE/'results/sft-pinning-driver-20261020-01'
PREP = '/home/huids25/kda-jiuge-prep-20261014'
ROOT = PREP+'/sft-pinning-validation-20261020-01'
SSH = ['ssh','-i','C:/Users/lenovo/.ssh/id_ed25519','-o','StrictHostKeyChecking=yes',
       '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=10',
       '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6','-p','30353','huids25@192.168.5.200']
SCP = ['scp','-i','C:/Users/lenovo/.ssh/id_ed25519','-o','StrictHostKeyChecking=yes',
       '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=10','-P','30353']


def run(argv, **kw):
    result = subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180,**kw)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8',errors='replace')[-1500:])
    return result.stdout


def remote_json(path):
    code = 'from pathlib import Path; p=Path('+repr(path)+'); print(p.read_text() if p.is_file() else "null")'
    # Single-quoted shell code is not constructed from user or model text.
    value = run([*SSH,PREP+'/.venv/bin/python -'],input=code.encode('utf-8'))
    return json.loads(value)


def download(remote,name):
    target = LOCAL/name
    if target.exists():
        raise ValueError('Preserve immutable existing key bundle: '+str(target))
    run([*SCP,'huids25@192.168.5.200:'+remote,str(target)])
    return target


def main():
    LOCAL.mkdir(exist_ok=False)
    state = {'schema':'jiuge7b-sft-pinning-local-driver-v1','started_utc':datetime.now(timezone.utc).isoformat(),
             'status':'waiting-for-remote-controller','local_large_originals_downloaded':False}
    def save(status,**values):
        state.update(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),**values)
        temp = LOCAL/'status.json.tmp'
        temp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        temp.replace(LOCAL/'status.json')
        print(json.dumps({'status':status,'utc':state['updated_utc']},ensure_ascii=False),flush=True)
    save(state['status'])
    gate_sent = False
    deadline = time.monotonic()+12*3600
    while time.monotonic()<deadline:
        try:
            controller = remote_json(ROOT+'/controller.json')
        except (RuntimeError,subprocess.TimeoutExpired) as error:
            save('bridge-temporarily-unavailable',error=str(error))
            time.sleep(30)
            continue
        if controller is None:
            time.sleep(20)
            continue
        (LOCAL/'remote-controller-latest.json').write_text(json.dumps(controller,indent=2)+'\n',encoding='utf-8')
        phase = controller['status']
        if state.get('remote_status') != phase:
            save('observing',remote_status=phase)
        if phase == 'ready-for-local-key-verification' and not gate_sent:
            bundle = download(controller['key_results_bundle'],'sft-correctness-key-results.zip')
            receipt = LOCAL/'local-key-verification.json'
            checked = verify(bundle,receipt,64*1024**2)
            run([*SCP,str(receipt),'huids25@192.168.5.200:'+ROOT+'/local-key-verification.json.uploading'])
            run([*SSH,'mv '+ROOT+'/local-key-verification.json.uploading '+ROOT+'/local-key-verification.json'])
            gate_sent = True
            save('local-correctness-keys-verified-gate-sent',key_files=checked['verified_key_files'],
                 key_uncompressed_bytes=checked['verified_key_bytes'],bundle_sha256=checked['bundle_sha256'])
        if phase == 'stopped':
            save('remote-stopped',error=controller.get('error'),formal_performance_passed=False)
            return 2
        if phase == 'completed':
            final = remote_json(ROOT+'/final-key-bundle.json')
            if final is None:
                time.sleep(5)
                continue
            bundle = download(final['path'],'sft-complete-key-results.zip')
            if bundle.stat().st_size != final['bytes'] or hashlib.sha256(bundle.read_bytes()).hexdigest() != final['sha256']:
                raise ValueError('ACT/local final bundle SHA differs')
            checked = verify(bundle,LOCAL/'final-key-verification.json',64*1024**2)
            save('completed',formal_performance_passed=controller['formal_performance_passed'],
                 final_key_files=checked['verified_key_files'],final_key_bytes=checked['verified_key_bytes'])
            return 0
        time.sleep(30)
    save('authorized-campaign-window-ended')
    return 3


if __name__ == '__main__':
    raise SystemExit(main())

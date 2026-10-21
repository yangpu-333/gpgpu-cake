"""After closed GPU capture, review bounded keys and run ACT CPU-only delivery."""
from datetime import datetime,timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
from drive_sft_pinning_validation import SSH,SCP,PREP,run

BASE=Path(__file__).resolve().parent
FOLDER=BASE/'results/sft-paired-trace-postprocess-20261020-01'
DRIVER=BASE/'results/sft-paired-trace-driver-20261020-01'
REMOTE_SOURCES=('export_sft_trace_lanes_20261020.py','package_sft_paired_trace_20261020.py')


def main():
    FOLDER.mkdir(exist_ok=True)
    with (FOLDER/'postprocess.lock').open('a+b') as lock:
        if lock.tell()==0: lock.write(b'1'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        state={'schema':'jiuge7b-sft-paired-trace-postprocess-v1','pid':os.getpid(),'cpu_only':True,
               'raw_trace_bodies_downloaded':False}
        def save(status,**extra):
            state.update(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),**extra)
            temp=FOLDER/'status.tmp'
            temp.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
            temp.replace(FOLDER/'status.json')
        def local(script,name):
            with (FOLDER/(name+'.stdout.log')).open('xb') as out,(FOLDER/(name+'.stderr.log')).open('xb') as err:
                subprocess.run([sys.executable,str(BASE/script)],stdout=out,stderr=err,check=True,timeout=600)
        def download(path,target,max_bytes):
            code='from pathlib import Path; import hashlib,json; p=Path('+repr(path)+'); print(json.dumps({"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}))'
            proof=json.loads(run([*SSH,PREP+'/.venv/bin/python -'],input=code.encode()).decode())
            if proof['bytes']>max_bytes: raise ValueError('Small key export limit exceeded')
            if not target.exists():
                partial=target.with_suffix(target.suffix+'.partial')
                run([*SCP,'huids25@192.168.5.200:'+path,str(partial)])
                if partial.stat().st_size!=proof['bytes'] or hashlib.sha256(partial.read_bytes()).hexdigest()!=proof['sha256']:
                    raise ValueError('ACT/local small export SHA differs')
                partial.replace(target)
            if hashlib.sha256(target.read_bytes()).hexdigest()!=proof['sha256']:
                raise ValueError('Existing closed local export differs')
            return proof
        try:
            save('waiting-for-closed-key-bundle')
            deadline=time.monotonic()+4*3600
            while time.monotonic()<deadline:
                if (DRIVER/'status.json').exists():
                    current=json.loads((DRIVER/'status.json').read_text(encoding='utf-8'))
                    if current['status']=='completed': break
                    if current['status']=='remote-stopped': raise ValueError('GPU capture failed: '+current.get('error',''))
                save('waiting-for-closed-key-bundle')
                time.sleep(30)
            else: raise TimeoutError('Local wait cap reached')
            save('independent-key-review')
            local('review_sft_paired_trace_20261020.py','independent-review')
            # Preserve immutable postprocessing source identities before transfer.
            identities={n:hashlib.sha256((BASE/n).read_bytes()).hexdigest() for n in REMOTE_SOURCES}
            for name,sha in identities.items():
                code='from pathlib import Path; import hashlib,json; p=Path('+repr(PREP+'/'+name)+'); print(json.dumps({"exists":p.exists(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None}))'
                existing=json.loads(run([*SSH,PREP+'/.venv/bin/python -'],input=code.encode()).decode())
                if existing['exists'] and existing['sha256']!=sha: raise ValueError('ACT existing CPU source differs')
                if not existing['exists']: run([*SCP,str(BASE/name),'huids25@192.168.5.200:'+PREP+'/'])
            save('ACT-CPU-only-lanes-and-raw-archive',postprocessing_sources=identities)
            code='''from pathlib import Path
import hashlib,json,os,subprocess,sys
p=Path('''+repr(PREP)+''')
expected='''+repr(identities)+'''
for name,sha in expected.items(): assert hashlib.sha256((p/name).read_bytes()).hexdigest()==sha
env={**os.environ,'CUDA_VISIBLE_DEVICES':''}
for script,output in [('export_sft_trace_lanes_20261020.py','sft-paired-trace-lanes-20261020.json'),('package_sft_paired_trace_20261020.py','sft-paired-trace-delivery-20261020-01/verification.json')]:
    if not (p/output).exists(): subprocess.run([sys.executable,str(p/script)],env=env,cwd=p,check=True,timeout=600)
print('ACT-CPU-postprocessing-complete')
'''
            # Existing bridge helper has a bounded180s per-call cap; launch each
            # sequential CPU stage separately, both normally finish well below it.
            result=run([*SSH,PREP+'/.venv/bin/python -'],input=code.encode())
            with (FOLDER/'ACT-CPU-postprocess.stdout.log').open('xb') as stream: stream.write(result)
            lanes=download(PREP+'/sft-paired-trace-lanes-20261020.json',BASE/'results/sft-paired-trace-lanes-20261020.json',8*1024**2)
            archive=download(PREP+'/sft-paired-trace-delivery-20261020-01/verification.json',BASE/'results/sft-paired-trace-delivery-verification-20261020.json',2*1024**2)
            save('plotting-bounded-results',lanes_SHA=lanes,ACT_raw_archive_receipt_SHA=archive)
            local('plot_sft_paired_trace_20261020.py','plots')
            save('completed-scientific-postprocessing',independent_review_passed=True,
                 reports_and_final_ACT_review_archive_pending=True)
        except BaseException as exc:
            save('stopped',error_type=type(exc).__name__,error=str(exc))
            raise


if __name__=='__main__': main()

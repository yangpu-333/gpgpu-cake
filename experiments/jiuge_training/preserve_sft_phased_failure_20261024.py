"""Preserve bounded failed-attempt keys; all original value packs stay on ACT."""
import hashlib
import json
from pathlib import Path
import subprocess
from drive_sft_pinning_validation import PREP,SSH,SCP
from pinning_key_results import verify

BASE=Path(__file__).resolve().parent


def main():
    code='''from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,'''+repr(PREP)+''')
from pinning_key_results import export,verify
p=Path('''+repr(PREP)+''');root=p/'sft-phased-acceptance-20261024-01'
r=json.loads((root/'controller.json').read_text());assert r['status']=='stopped' and not r['formal_performance_started']
target=p/'sft-phased-acceptance-failure-20261024-01';target.mkdir(exist_ok=False)
cases=[p/'runs'/('phased-audit-'+arm+'-20261024-01') for arm in ('native','control','optimized')]
bundle=target/'failed-acceptance-key-results.zip'
export(p,[root,*cases],bundle,64*1024**2)
v=verify(bundle,target/'key-verification.json',64*1024**2)
failure={'status':r['status'],'error':r['error'],'completed_cases':r['completed_cases'],'formal_started':False,
 'candidate_steps_completed':0,'candidate_moment_objects_created':0,
 'actual_guard':'8GiB plus2x allocation; unchanged','raw_tensor_packs_remain_ACT':True,
 'full_raw_SHA_gate_not_yet_completed':True,'failed_attempt_sources_preserved':True,
 'bundle':str(bundle),'bytes':bundle.stat().st_size,'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'ACT_key_verification':v}
with (target/'failure.json').open('x') as f:json.dump(failure,f,indent=2)
print(json.dumps(failure))
'''
    result=subprocess.run([*SSH,PREP+'/.venv/bin/python -'],input=code,text=True,capture_output=True,check=True,timeout=180)
    receipt=json.loads(result.stdout)
    folder=BASE/'results/sft-phased-acceptance-failure-20261024-01';folder.mkdir(exist_ok=False)
    bundle=folder/'failed-acceptance-key-results.zip'
    subprocess.run([*SCP,'huids25@192.168.5.200:'+receipt['bundle'],str(bundle)],capture_output=True,check=True,timeout=180)
    if bundle.stat().st_size!=receipt['bytes'] or hashlib.sha256(bundle.read_bytes()).hexdigest()!=receipt['sha256']:
        raise ValueError('ACT/local failure keys differ')
    local=verify(bundle,folder/'local-key-verification.json',64*1024**2)
    receipt.update(local_key_verification=local)
    with (BASE/'results/sft-phased-acceptance-failure-preservation-20261024-01.json').open('x',encoding='utf-8') as f:
        json.dump(receipt,f,indent=2)
    print(json.dumps({'passed':True,'bytes':receipt['bytes'],'key_files':local['verified_key_files'],'full_raw_downloaded':False}))


if __name__=='__main__':main()

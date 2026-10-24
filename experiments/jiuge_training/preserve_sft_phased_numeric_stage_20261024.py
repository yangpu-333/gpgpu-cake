"""Close only the completed three-step numeric stage; no resume/formal promotion."""
from pathlib import Path
import hashlib
import json
import subprocess
from drive_sft_pinning_validation import SSH,SCP,PREP
from pinning_key_results import verify

BASE=Path(__file__).resolve().parent


def main():
    code='''from pathlib import Path
import sys,json,hashlib
p=Path('''+repr(PREP)+''');sys.path.insert(0,str(p))
from pinning_key_results import export,verify
root=p/'sft-phased-acceptance-20261024-02';case=p/'runs/phased-audit-optimized-20261024-02'
record=json.loads((case/'case.json').read_text());assert record['exit_code']==0
for rank in (0,1):
 done=json.loads((root/'numeric/optimized'/f'rank-{rank}/completion.json').read_text())
 assert done['passed'] and done['complete_fields'] and done['steps']==3
target=p/'sft-phased-numeric-stage-delivery-20261024-02';target.mkdir(exist_ok=False)
bundle=target/'completed-numeric-key-results.zip'
export(p,[root/'numeric',root/'semantics',root/'source',case],bundle,64*1024**2)
v=verify(bundle,target/'key-verification.json',64*1024**2)
r={'passed':True,'stage':'candidate own three-step full values only','resume_passed':False,'formal_passed':False,'full_raw_SHA_gate_not_yet_closed':True,'bundle':str(bundle),'bytes':bundle.stat().st_size,'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'ACT_key_verification':v}
with (target/'stage-preservation.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
'''
    x=subprocess.run([*SSH,PREP+'/.venv/bin/python -'],input=code,text=True,capture_output=True,timeout=120,check=True)
    result=json.loads(x.stdout)
    folder=BASE/'results/sft-phased-numeric-stage-20261024-02';folder.mkdir(exist_ok=False)
    bundle=folder/'completed-numeric-key-results.zip'
    subprocess.run([*SCP,'huids25@192.168.5.200:'+result['bundle'],str(bundle)],capture_output=True,timeout=120,check=True)
    assert bundle.stat().st_size==result['bytes'] and hashlib.sha256(bundle.read_bytes()).hexdigest()==result['sha256']
    result['local_key_verification']=verify(bundle,folder/'local-key-verification.json',64*1024**2)
    with (BASE/'results/sft-phased-numeric-stage-preservation-20261024-02.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({'passed':True,'bytes':result['bytes'],'key_files':result['local_key_verification']['verified_key_files'],'stage':'three-step numeric only','raw_downloaded':False}))


if __name__=='__main__':main()

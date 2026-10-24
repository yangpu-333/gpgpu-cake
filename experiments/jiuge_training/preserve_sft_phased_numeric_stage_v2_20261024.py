"""Use an explicit immutable-key list after recursive inventory exceeded its timeout."""
from pathlib import Path
import hashlib
import json
import subprocess
from drive_sft_pinning_validation import SSH,SCP,PREP
from pinning_key_results import verify

BASE=Path(__file__).resolve().parent


def main():
    code='''from pathlib import Path
import sys,json,hashlib,zipfile,datetime
p=Path('''+repr(PREP)+''');sys.path.insert(0,str(p))
from pinning_key_results import verify
root=p/'sft-phased-acceptance-20261024-02';case=p/'runs/phased-audit-optimized-20261024-02'
record=json.loads((case/'case.json').read_text());assert record['exit_code']==0
paths=[]
for arm in ('native','control','optimized'):
 for rank in (0,1):
  d=root/'numeric'/arm/f'rank-{rank}'
  done=json.loads((d/'completion.json').read_text());assert done['passed'] and done['steps']==3
  paths.extend([d/'all-state-manifest.json',d/'completion.json'])
paths.extend([root/'reference-origins.json',p/'sft-phased-acceptance-source-manifest-20261024-02.json',case/'case.json',case/'stdout.log',case/'stderr.log'])
for rank in (0,1):
 paths.extend([case/f'audit-cache-rank-{rank}.json',case/f'phased-policy-rank-{rank}.json',case/f'rank-{rank}/entry.json'])
assert sum(f.stat().st_size for f in paths)<64*1024**2
target=p/'sft-phased-numeric-stage-delivery-20261024-02-v2';target.mkdir(exist_ok=False)
bundle=target/'completed-numeric-key-results.zip';files={}
with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for f in paths:
  assert f.is_file() and not f.is_symlink() and f.resolve().is_relative_to(p)
  name=f.relative_to(p).as_posix();b=f.read_bytes();files[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};z.writestr(name,b)
 catalog={'schema':'jiuge-pinning-key-results-v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'primary_root':str(p),'files':files,'excluded_originals':[],'key_file_count':len(files),'key_uncompressed_bytes':sum(x['bytes'] for x in files.values()),'budget_bytes':64*1024**2,'full_raw_originals_included':False,'full_raw_local_copy_verified':False,'primary_originals_deleted':False,'selection':'explicit completed manifests/logs only; no recursive original inventory'}
 z.writestr('key-results-manifest.json',json.dumps(catalog,indent=2))
v=verify(bundle,target/'key-verification.json',64*1024**2)
r={'passed':True,'stage':'candidate own three-step full values only','resume_passed':False,'formal_passed':False,'full_raw_SHA_gate_not_yet_closed':True,'bundle':str(bundle),'bytes':bundle.stat().st_size,'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'ACT_key_verification':v,'previous_recursive_inventory_timeout_preserved':True}
with (target/'stage-preservation.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
'''
    x=subprocess.run([*SSH,PREP+'/.venv/bin/python -'],input=code,text=True,capture_output=True,timeout=120,check=True)
    result=json.loads(x.stdout)
    folder=BASE/'results/sft-phased-numeric-stage-20261024-02-v2';folder.mkdir(exist_ok=False)
    bundle=folder/'completed-numeric-key-results.zip'
    subprocess.run([*SCP,'huids25@192.168.5.200:'+result['bundle'],str(bundle)],capture_output=True,timeout=120,check=True)
    assert bundle.stat().st_size==result['bytes'] and hashlib.sha256(bundle.read_bytes()).hexdigest()==result['sha256']
    result['local_key_verification']=verify(bundle,folder/'local-key-verification.json',64*1024**2)
    with (BASE/'results/sft-phased-numeric-stage-preservation-20261024-02-v2.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps({'passed':True,'bytes':result['bytes'],'key_files':result['local_key_verification']['verified_key_files'],'stage':'three-step numeric only','raw_downloaded':False}))


if __name__=='__main__':main()

"""Preserve bounded recovery sources/results/reports on ACT without raw downloads."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile
from drive_sft_pinning_validation import SSH,SCP,PREP

BASE=Path(__file__).resolve().parent


def main():
    source=json.loads((BASE/'results/sft-phased-acceptance-source-manifest-20261024-02.json').read_text(encoding='utf-8'))['files']
    files={('source/'+n):BASE/n for n in source}
    for n,h in source.items():
        assert hashlib.sha256((BASE/n).read_bytes()).hexdigest()==h,n
    extras=[
        'README.md','BRIEF.md','review_sft_phased_acceptance_v2_20261024.py',
        'finish_sft_phased_acceptance_v2_20261024.py','preserve_sft_phased_failure_20261024.py',
        'preserve_sft_phased_recovery_materials_20261024.py',
        'results/sft-phased-acceptance-source-manifest-20261024-02.json',
        'results/sft-phased-recovery-CPU-preparation-20261024-02.json',
        'results/sft-phased-acceptance-failure-preservation-20261024-01.json',
        'results/sft-phased-acceptance-failure-20261024-01/failed-acceptance-key-results.zip',
        'results/sft-phased-acceptance-recovery-admission-20261024-02.json',
        'results/sft-phased-acceptance-allocation-approved-20261024-02.json',
        'results/sft-phased-acceptance-postprocess-sources-20261024-02.json',
        'results/sft-phased-acceptance-postprocess-sources-20261024-02-v2.json',
        'results/sft-phased-acceptance-failed-0points-20261024-01.jpg',
        'results/sft-phased-acceptance-recovery-running-20261024-02.jpg',
    ]
    files.update({n:BASE/n for n in extras})
    assert all(p.is_file() for p in files.values())
    assert sum(p.stat().st_size for p in files.values())<64*1024**2
    identities={n:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for n,p in files.items()}
    local=BASE/'results/sft-phased-acceptance-recovery-materials-20261024-02.zip'
    with zipfile.ZipFile(local,'x',zipfile.ZIP_DEFLATED) as z:
        for n,p in files.items():z.write(p,n)
        z.writestr('preservation-manifest.json',json.dumps(identities,indent=2))
    digest=hashlib.sha256(local.read_bytes()).hexdigest()
    remote=PREP+'/'+local.name
    subprocess.run([*SCP,str(local),'huids25@192.168.5.200:'+remote],check=True,capture_output=True,timeout=90)
    code='''from pathlib import Path
import hashlib,json,zipfile
p=Path('''+repr(PREP)+''');bundle=p/'''+repr(local.name)+'''
assert hashlib.sha256(bundle.read_bytes()).hexdigest()=='''+repr(digest)+'''
dest=p/'sft-phased-acceptance-recovery-delivery-20261024-02';dest.mkdir(exist_ok=False)
with zipfile.ZipFile(bundle) as z:
 manifest=json.loads(z.read('preservation-manifest.json'))
 for n,row in manifest.items():
  assert not Path(n).is_absolute() and '..' not in Path(n).parts
  b=z.read(n);assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
 z.extractall(dest)
for n,row in manifest.items():
 f=dest/n;assert f.stat().st_size==row['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==row['sha256']
r={'passed':True,'folder':str(dest),'files':len(manifest),'bytes':sum(x['bytes'] for x in manifest.values()),'bundle_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'raw_tensors_downloaded':False,'original158_sources_modified':False,'candidate_full_acceptance_not_yet_passed':True,'same_campaign_deadline_epoch':1792853953,'max_points':0}
with (dest/'ACT-verification.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
'''
    result=subprocess.run([*SSH,PREP+'/.venv/bin/python -'],input=code,check=True,capture_output=True,text=True,timeout=90)
    with (BASE/'results/sft-phased-acceptance-recovery-ACT-preservation-20261024-02.json').open('x',encoding='utf-8') as f:f.write(result.stdout)
    print(result.stdout)


if __name__=='__main__':main()

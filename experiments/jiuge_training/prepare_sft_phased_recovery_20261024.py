"""CPU-check observation recovery; original158 identities remain frozen."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from drive_sft_pinning_validation import SSH,SCP,PREP

BASE=Path(__file__).resolve().parent
FILES=('sft_phased_audit_cache_20261024.py','sft_phased_acceptance_recovery_20261024.py',
       'sft_phased_acceptance_entry_v2_20261024.py','run_sft_phased_acceptance_v2_20261024.py',
       'continue_sft_phased_acceptance_v2_20261024.py','drive_sft_phased_acceptance_v2_20261024.py',
       'check_sft_audit_cache_20261024.py','bootstrap_sft_phased_acceptance_v2_20261024.sh',
       'probe_sft_phased_acceptance_v2_20261024.py','approve_sft_phased_acceptance_v2_20261024.py',
       'prepare_sft_phased_recovery_20261024.py')


def main():
    old=json.loads((BASE/'results/sft-phased-acceptance-source-manifest-20261024-01.json').read_text(encoding='utf-8'))['files']
    files=dict(old)
    for n,h in old.items():
        if hashlib.sha256((BASE/n).read_bytes()).hexdigest()!=h:raise ValueError('Original158 changed: '+n)
    for n in FILES:
        data=(BASE/n).read_bytes()
        if n.endswith('.py'):ast.parse(data)
        files[n]=hashlib.sha256(data).hexdigest()
    output=BASE/'results/sft-phased-acceptance-source-manifest-20261024-02.json'
    with output.open('x',encoding='utf-8') as f:json.dump({'files':files,'old158_unchanged':True,
       'change':'untimed observer cache policy only;formal math/timer/candidate unchanged',
       'absolute_stop_epoch':1792853953,'max_points':0,'references':'same-campaign completed native/control;not prior performance'},f,indent=2)
    bundle=BASE/'results/sft-phased-recovery-source-20261024-02.zip'
    with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
        for n in FILES:z.write(BASE/n,n)
        z.write(output,output.name)
    subprocess.run([*SCP,str(bundle),'huids25@192.168.5.200:'+PREP+'/'+bundle.name],capture_output=True,check=True,timeout=90)
    code='''from pathlib import Path
import hashlib,json,subprocess,zipfile,sys,os
p=Path('''+repr(PREP)+''')
sys.path.insert(0,str(p));os.chdir(p)
with zipfile.ZipFile(p/'''+repr(bundle.name)+''') as z:
 for n in z.namelist():
  assert Path(n).name==n
  f=p/n;assert not f.exists(),n
  with f.open('xb') as out:out.write(z.read(n))
m=json.loads((p/'''+repr(output.name)+''').read_text())
for n,h in m['files'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
subprocess.run(['bash','-n',str(p/'bootstrap_sft_phased_acceptance_v2_20261024.sh')],check=True)
subprocess.run([str(p/'.venv/bin/python'),str(p/'check_sft_audit_cache_20261024.py')],cwd=p,check=True,timeout=120)
from continue_sft_phased_acceptance_v2_20261024 import generated
compile(generated(),'<same-controller-with-owned-audit-cache-recovery>','exec')
print(json.dumps({'passed':True,'source_files':len(m['files']),'old158_unchanged':True,'GPU_used':False}))
'''
    result=subprocess.run([*SSH,PREP+'/.venv/bin/python -'],input=code,text=True,capture_output=True,timeout=180)
    r={'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
       'source_manifest_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'GPU_experiment_launched':False}
    with (BASE/'results/sft-phased-recovery-CPU-preparation-20261024-02.json').open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
    if result.returncode:raise RuntimeError(result.stderr[-4000:])
    print(result.stdout)


if __name__=='__main__':main()

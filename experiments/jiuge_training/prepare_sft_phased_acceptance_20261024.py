"""Prepare and CPU-check a new proposal without allocating or launching GPUs."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from drive_sft_pinning_validation import PREP, SSH, SCP

BASE = Path(__file__).resolve().parent
FILES = ('sft_phased_acceptance_20261024.py','sft_phased_acceptance_entry_20261024.py',
         'run_sft_phased_acceptance_20261024.py','sft_phased_acceptance_20261024.contract.json',
         'check_sft_phased_acceptance_20261024.py','analyze_sft_phased_formal_20261024.py',
         'continue_sft_phased_acceptance_20261024.py','probe_sft_phased_acceptance_20261024.py',
         'bootstrap_sft_phased_acceptance_20261024.sh','drive_sft_phased_acceptance_20261024.py',
         'prepare_sft_phased_acceptance_20261024.py','approve_sft_phased_acceptance_20261024.py')


def main():
    old = json.loads((BASE/'results/sft-phased-moments-source-manifest-20261023-07.json').read_text())['files']
    hashes = dict(old)
    for name,digest in old.items():
        if hashlib.sha256((BASE/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Prior146 source bytes changed: '+name)
    for name in FILES:
        data = (BASE/name).read_bytes()
        if name.endswith('.py'):
            ast.parse(data)
        if name.endswith('.json'):
            json.loads(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = {'schema':'jiuge7b-incremental-SFT-acceptance-source-v1','utc':datetime.now(timezone.utc).isoformat(),
                'old146_unchanged':True,'files':hashes,'proposal_not_GPU_admission':True}
    output = BASE/'results/sft-phased-acceptance-source-manifest-20261024-01.json'
    with output.open('x',encoding='utf-8') as stream:
        json.dump(manifest,stream,indent=2);stream.write('\n')
    for name in FILES:
        subprocess.run([*SCP,str(BASE/name),'huids25@192.168.5.200:'+PREP+'/'+name],check=True,timeout=60)
    subprocess.run([*SCP,str(output),'huids25@192.168.5.200:'+PREP+'/'+output.name],check=True,timeout=60)
    code = '''from pathlib import Path
import hashlib,json,subprocess,sys
p=Path('''+repr(PREP)+''')
m=json.loads((p/'''+repr(output.name)+''').read_text())
for n,h in m['files'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
subprocess.run(['bash','-n',str(p/'bootstrap_sft_phased_acceptance_20261024.sh')],check=True)
r=subprocess.run([str(p/'.venv/bin/python'),str(p/'check_sft_phased_acceptance_20261024.py')],cwd=p,capture_output=True,text=True,timeout=120)
with (p/'sft-phased-acceptance-CPU-preparation-20261024-01.log').open('x') as f:f.write(r.stdout+'\\n'+r.stderr)
if r.returncode:raise RuntimeError(r.stderr[-3000:])
proof=json.loads(r.stdout.strip().splitlines()[-1]);assert proof['passed'] and not proof['GPU_used']
print(json.dumps({'passed':True,'source_files':len(m['files']),'old146_unchanged':True,'CPU_checks':proof}))
'''
    result = subprocess.run([*SSH,PREP+'/.venv/bin/python -'],input=code,text=True,capture_output=True,timeout=180)
    receipt = {'schema':'jiuge7b-phased-acceptance-preparation-v1','exit_code':result.returncode,
               'stdout':result.stdout,'stderr':result.stderr,'source_manifest_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
               'GPU_experiment_launched':False,'resource_submitted':False}
    with (BASE/'results/sft-phased-acceptance-CPU-preparation-20261024-01.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2)
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])
    print(result.stdout)


if __name__=='__main__':
    main()

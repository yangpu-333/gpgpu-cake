"""Freeze the new proposal; CPU preparation only, no resource submission."""
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from drive_sft_pinning_validation import SSH,PREP,run

BASE=Path(__file__).resolve().parent
NEW=('probe_sft_page_policy_20261021.py','sft_local_moments_20261021.py',
     'check_sft_local_moments_20261021.py','sft_moment_entry_20261021.py',
     'run_sft_moment_case_20261021.py','continue_sft_moment_screen_20261021.py',
     'bootstrap_sft_moment_screen_20261021.sh','sft_moment_screen_20261021.contract.json',
     'review_sft_numa_v2_20261021.py','review_sft_moment_screen_20261021.py',
     'check_sft_moment_preparation_20261021.py','probe_sft_moment_runtime_20261021.py',
     'prepare_sft_moment_screen_20261021.py')


def main():
    previous=json.loads((BASE/'results/sft-numa-source-manifest-v2-20261021.json').read_text())['files']
    cpu=json.loads((BASE/'results/sft-local-moments-CPU-regression-20261021.json').read_text())
    if not cpu['passed'] or cpu['GPU_used']:raise ValueError('Closed CPU regression gate required')
    files=dict(previous)
    for name in (*previous,*NEW):
        data=(BASE/name).read_bytes();digest=hashlib.sha256(data).hexdigest()
        if name in previous and previous[name]!=digest:raise ValueError('Prior frozen source changed: '+name)
        if name in cpu['source_sha256'] and cpu['source_sha256'][name]!=digest:raise ValueError('CPU-tested source changed: '+name)
        if name.endswith('.py'):compile(data,name,'exec')
        files[name]=digest
    probe=json.loads(subprocess.check_output([sys.executable,str(BASE/'check_sft_moment_preparation_20261021.py')]))
    if not probe['passed'] or probe['GPU_used']:raise ValueError('CPU launch/gate preparation failed')
    manifest={'schema':'jiuge7b-sft-moment-screen-frozen-proposal-v1','files':files,
              'prior58_sources_unchanged':True,'CPU_regression_receipt':cpu,
              'CPU_generated_program_probe':probe,'GPU_experiment_started':False,
              'new_allocation_not_authorized_or_submitted':True,'created_utc':datetime.now(timezone.utc).isoformat()}
    local=BASE/'results/sft-moment-screen-source-manifest-20261021.json'
    if local.exists():
        existing=json.loads(local.read_text())
        if existing['files']!=files:raise ValueError('Do not replace a frozen proposal')
        manifest=existing
    else:
        with local.open('x') as stream:json.dump(manifest,stream,indent=2)
    payloads={n:base64.b64encode((BASE/n).read_bytes()).decode() for n in NEW}
    encoded=base64.b64encode(local.read_bytes()).decode()
    code='''from pathlib import Path
import base64,hashlib,json,subprocess,sys,shutil
p=Path('''+repr(PREP)+''');files='''+repr(files)+''';payloads='''+repr(payloads)+'''
for name,sha in files.items():
 f=p/name
 if f.exists():assert hashlib.sha256(f.read_bytes()).hexdigest()==sha,name
 else:
  assert name in payloads,name
  with f.open('xb') as stream:stream.write(base64.b64decode(payloads[name]))
m=p/'sft-moment-screen-source-manifest-20261021.json';data=base64.b64decode('''+repr(encoded)+''')
if m.exists():assert m.read_bytes()==data
else:
 with m.open('xb') as stream:stream.write(data)
snapshot=p/'sft-moment-placement-preparation-20261021-01/source-proposal'
snapshot.mkdir(exist_ok=False)
for name,sha in files.items():
 f=p/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==sha,name
 shutil.copyfile(f,snapshot/name)
shutil.copyfile(m,snapshot/m.name)
subprocess.run(['bash','-n',str(p/'bootstrap_sft_moment_screen_20261021.sh')],check=True)
probe=json.loads(subprocess.check_output([sys.executable,str(p/'check_sft_moment_preparation_20261021.py')]))
assert probe['passed'] and not probe['GPU_used']
cpu=json.loads((p/'sft-moment-placement-preparation-20261021-01/cpu-regression/summary.json').read_text())
assert cpu['passed'] and not cpu['GPU_used']
result={'passed':True,'ACT_source_snapshot':str(snapshot),'frozen_files':len(files),
        'prior58_unchanged':True,'manifest_sha256':hashlib.sha256(data).hexdigest(),
        'CPU_preparation_probe':probe,'CPU_regression_passed':True,'GPU_experiment_started':False,
        'new_allocation_submitted':False}
with (snapshot.parent/'preparation-verification.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result))
'''
    result=json.loads(run([*SSH,PREP+'/.venv/bin/python -'],input=code.encode()))
    if result['manifest_sha256']!=hashlib.sha256(local.read_bytes()).hexdigest():raise ValueError('ACT/local proposal manifest SHA differs')
    with (BASE/'results/sft-moment-screen-preparation-20261021.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

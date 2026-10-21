"""Freeze a reviewable new screen after CPU evidence; submit no resource."""
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from drive_sft_pinning_validation import SSH,PREP,run

BASE=Path(__file__).resolve().parent
NEW=(
    'sft_local_moments_v3_20261021.py','check_sft_local_moments_v3_20261021.py',
    'probe_sft_moment_peak_v3_20261021.py','prepare_sft_moment_direct_zero_cpu_20261021.py',
    'sft_moment_memory_guard_20261021.py','sft_moment_direct_zero_entry_20261021.py',
    'run_sft_moment_direct_zero_case_20261021.py','continue_sft_moment_direct_zero_20261021.py',
    'check_sft_moment_direct_zero_guard_20261021.py','check_sft_moment_direct_zero_guard_v2_20261021.py',
    'probe_sft_moment_direct_zero_runtime_20261021.py','review_sft_moment_direct_zero_20261021.py',
    'bootstrap_sft_moment_direct_zero_20261021.sh','sft_moment_direct_zero_v3_20261021.contract.json',
    'drive_sft_moment_direct_zero_20261021.py','package_sft_moment_direct_zero_20261021.py',
    'plot_sft_moment_direct_zero_20261021.py','finish_sft_moment_direct_zero_20261021.py',
    'check_sft_moment_direct_zero_preparation_20261021.py','prepare_sft_moment_direct_zero_screen_20261021.py',
)


def main():
    previous=json.loads((BASE/'results/sft-moment-screen-source-manifest-v2-20261021.json').read_text())['files']
    files=dict(previous)
    for name in (*previous,*NEW):
        data=(BASE/name).read_bytes();sha=hashlib.sha256(data).hexdigest()
        if name in previous and previous[name]!=sha:raise ValueError('Frozen prior source changed: '+name)
        if name.endswith('.py'):compile(data,name,'exec')
        files[name]=sha
    probe=json.loads(subprocess.check_output([sys.executable,str(BASE/'check_sft_moment_direct_zero_preparation_20261021.py')]))
    assert probe['passed'] and not probe['GPU_used']
    data={'schema':'jiuge7b-sft-direct-zero-frozen-preparation-v3','files':files,
          'original73_unchanged':True,'CPU_generated_program_probe':probe,
          'new_GPU_allocation_submitted':False,'user_requested_pause_after_preparation':True,
          'raw_backup_before_job_reset_requested':True,'utc':datetime.now(timezone.utc).isoformat()}
    manifest=BASE/'results/sft-moment-direct-zero-source-manifest-v3-20261021.json'
    with manifest.open('x') as stream:json.dump(data,stream,indent=2)
    payloads={name:base64.b64encode((BASE/name).read_bytes()).decode() for name in NEW}
    encoded=base64.b64encode(manifest.read_bytes()).decode()
    code='''from pathlib import Path
import base64,hashlib,json,subprocess,sys,shutil
p=Path('''+repr(PREP)+''');files='''+repr(files)+''';payloads='''+repr(payloads)+'''
for name,expected in files.items():
 f=p/name
 if f.exists():assert hashlib.sha256(f.read_bytes()).hexdigest()==expected,name
 else:
  with f.open('xb') as stream:stream.write(base64.b64decode(payloads[name]))
root=p/'sft-moment-direct-zero-preparation-20261021-01'
cpu=json.loads((root/'summary.json').read_text())
guard=json.loads((root/'guarded-regression-v2/summary.json').read_text())
assert cpu['passed'] and guard['passed'] and not cpu['GPU_used'] and not guard['GPU_used']
tested=json.loads((p/'sft-moment-direct-zero-CPU-launch-20261021.json').read_text())['source_sha256']
for name,expected in tested.items():assert files[name]==expected
tested_guard=json.loads((root/'guarded-source-sha.json').read_text())
for name,expected in tested_guard.items():assert files[name]==expected
m=p/'sft-moment-direct-zero-source-manifest-v3-20261021.json';data=base64.b64decode('''+repr(encoded)+''')
with m.open('xb') as stream:stream.write(data)
snapshot=root/'source-proposal';snapshot.mkdir(exist_ok=False)
for name,sha in files.items():
 assert hashlib.sha256((p/name).read_bytes()).hexdigest()==sha,name
 shutil.copyfile(p/name,snapshot/name)
shutil.copyfile(m,snapshot/m.name)
subprocess.run(['bash','-n',str(p/'bootstrap_sft_moment_direct_zero_20261021.sh')],check=True)
probe=json.loads(subprocess.check_output([sys.executable,str(p/'check_sft_moment_direct_zero_preparation_20261021.py')]))
assert probe['passed'] and not probe['GPU_used']
result={'schema':'jiuge7b-sft-direct-zero-proposal-admission-v3','passed':True,
 'frozen_sources':len(files),'original73_unchanged':True,'CPU_regression':cpu,
 'guarded_CPU_regression':guard,'CPU_guard_fixture_reserve_bytes':512*1024**2,
 'SFT_production_reserve_bytes':8*1024**3,'production8GiB_guard_correct_refusal_in_CPU16GiB_fixture_preserved':True,
 'generated_program_probe':probe,'ACT_source_snapshot':str(snapshot),
 'manifest_sha256':hashlib.sha256(data).hexdigest(),'new_GPU_allocation_submitted':False,
 'new_7B_OOM_fixed':False,'formal_performance_evidence':False}
with (root/'screen-preparation-verification.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result))
'''
    result=json.loads(run([*SSH,PREP+'/.venv/bin/python -'],input=code.encode()))
    assert result['manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
    with (BASE/'results/sft-moment-direct-zero-preparation-v3-20261021.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('CPU_regression','guarded_CPU_regression')},indent=2))


if __name__=='__main__':main()

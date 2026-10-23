"""Admit the exact browser-confirmed allocation; never discover or launch jobs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from drive_sft_pinning_validation import SSH, PREP

BASE = Path(__file__).resolve().parent
JOB = 'sg-huids25-261023-3e329'
HOST = JOB+'-default0-0'


def main():
    manifest_path = BASE/'results/sft-phased-moments-resume-source-manifest-20261023-04.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    authority = json.loads((BASE/'results/sft-phased-moments-resume-authorization-20261023-04.json').read_text(encoding='utf-8'))
    budget = json.loads((BASE/'results/sft-phased-moments-resume-budget-20261023-04.json').read_text(encoding='utf-8'))
    assert authority['authorized'] and authority['max_charge_points'] == 0
    assert authority['source_manifest_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert len(manifest['files']) == 131
    for name, sha in manifest['files'].items():
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest() == sha, name
    payload = {'source_files_sha256': manifest['files'], 'budget': budget}
    code = '''from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,time
p=Path('''+repr(PREP)+''')
payload=json.loads('''+repr(json.dumps(payload))+''')
host='''+repr(HOST)+'''
probe_path=p/'sft-direct-zero-allocation-20261023'/(host+'.json')
if not probe_path.exists():raise SystemExit('Allocation probe not ready; no gate written')
probe=json.loads(probe_path.read_text());now=time.time()
assert probe['hostname']==host
assert 0<=now-datetime.fromisoformat(probe['utc']).timestamp()<120,'Stale allocation probe'
assert not probe['compute_apps'].strip(),'Foreign compute processes present'
rows=[[x.strip() for x in r.split(',')] for r in probe['cards'].strip().splitlines()]
assert len(rows)==2 and all('A100' in r[1] and r[4]=='0 MiB' and r[5]=='0 %' for r in rows)
gpus=probe['topology']['gpus'];uuids=[g['uuid'] for g in gpus]
assert [r[2] for r in rows]==uuids and len(set(uuids))==2
assert len({g['node'] for g in gpus})==1 and 'NODE' in probe['gpu_topology']
assert probe['cgroup']['cpu.max']=='1601000 100000'
assert int(probe['cgroup']['memory.max'])==322123595776
assert probe['persistent_free_bytes']>200*1024**3
budget=payload['budget'];start=budget['first_start_epoch'];stop=budget['stop_before_epoch']
assert start==1792761041 and stop==1792771841 and stop==start+10800
assert int((p/'sft-phased-moments-bootstrap-20261023-04/campaign-first-start.epoch').read_text())==start
assert now<stop-2400,'Insufficient unchanged campaign window'
for name,sha in payload['source_files_sha256'].items():
 assert Path(name).name==name and hashlib.sha256((p/name).read_bytes()).hexdigest()==sha,name
gate={'schema':'jiuge7b-phased-moments-exact-allocation-approval-v2','job':host.split('-default')[0],'hostname':host,'node':'inspur-gpu-04','max_charge_points':0,'startup_limit_seconds':10800,'first_start_epoch':start,'stop_before_epoch':stop,'ordered_gpu_uuids':uuids,'cpu_max':probe['cgroup']['cpu.max'],'memory_max_bytes':int(probe['cgroup']['memory.max']),'source_files_sha256':payload['source_files_sha256'],'actual_GPU_users_gate':probe['compute_apps'],'frozen_topology_before_any_training':gpus,'accepted_baseline':'accepted pinned repair;unbound setup and update','control':'same phased setup/update affinity as candidate;native moment allocation','candidate':'phased affinity plus original direct-zero local moments and physical capacity guard','operative_contract':'sft_phased_moments_resume_20261023_04.contract.json','separate_one_update_before_six_fresh_cases':True,'historical_controls_reused':False,'raw_storage':'ACT Ceph','local_key_limit_mib':32,'keep_old_V100':True,'other_processes_stopped':False,'new_login_identity_created':False,'training_uid_gid':[1001,1001],'ROOT_installation_explicit_user_authorization':True,'new_resource_user_authorization':'2026-10-23: authorized new3hours,0points;root dependencies thenUID1001','approved_utc':datetime.now(timezone.utc).isoformat(),'previous127_sources_unchanged':True,'resource_budget_not_reset':True,'first_attempt_this_window':'sg-huids25-261023-f1dfb:obsolete morning command,failed0points before GPU training'}
target=p/'sft-phased-moments-approved-20261023-03.json'
if target.exists():
 existing=json.loads(target.read_text());assert existing['hostname']==host and existing['source_files_sha256']==gate['source_files_sha256'] and existing['stop_before_epoch']==stop
 gate=existing
else:
 temporary=p/'sft-phased-moments-approved-20261023-03.json.pending'
 with temporary.open('x') as stream:json.dump(gate,stream,indent=2)
 assert not target.exists();temporary.rename(target)
print(json.dumps(gate),flush=True)
'''
    result = subprocess.run([*SSH, PREP+'/.venv/bin/python -'], input=code.encode(),
                            capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace'))
    gate = json.loads(result.stdout)
    target = BASE/'results/sft-phased-moments-allocation-approved-20261023-04.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(gate, stream, indent=2)
    print(json.dumps({k: v for k, v in gate.items() if k != 'source_files_sha256'}), flush=True)


if __name__ == '__main__':
    main()

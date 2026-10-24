"""Admit only the inspected new allocation; no training or credential changes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from drive_sft_pinning_validation import PREP, SSH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job',required=True)
    args = parser.parse_args()
    if not args.job.startswith('sg-huids25-261024-') or not args.job[19:].replace('-','').isalnum():
        raise ValueError('Current user new dated ACT job required')
    host = args.job+'-default0-0'
    code = '''from pathlib import Path
import hashlib,json,time
p=Path('''+repr(PREP)+''')
host='''+repr(host)+'''
probe=json.loads((p/'sft-direct-zero-allocation-20261023'/(host+'.json')).read_text())
assert probe['hostname']==host and not probe['compute_apps'].strip()
assert int(probe['cgroup']['memory.max'])>=300*1024**3
q,period=map(int,probe['cgroup']['cpu.max'].split());assert 16<=q/period<17
assert probe['persistent_free_bytes']>=1024**4
rows=[[x.strip() for x in row.split(',')] for row in probe['cards'].strip().splitlines()]
assert len(rows)==2 and all('A100' in r[1] and r[3].startswith('40960') and r[4].startswith('0 ') and r[5].startswith('0 ') for r in rows),rows
nodes={g['node'] for g in probe['topology']['gpus']};assert len(nodes)==1
assert 'NODE' in probe['gpu_topology'] and 'SYS' not in probe['gpu_topology'].split('Legend:')[0]
m=json.loads((p/'sft-phased-acceptance-source-manifest-20261024-01.json').read_text())
for n,h in m['files'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
start=int((p/'sft-phased-acceptance-bootstrap-20261024-01/campaign-first-start.epoch').read_text())
assert 0<time.time()-start<1800
plan=p/'sft-phased-acceptance-plan-20261024-01.json'
with plan.open('x') as f:json.dump(probe['plan'],f,indent=2)
r={'schema':'jiuge7b-phased-full-acceptance-approved-v1','hostname':host,'max_charge_points':0,
   'startup_limit_seconds':39600,'stop_before_epoch':start+39600,'source_files_sha256':m['files'],
   'plan_file':str(plan),'ordered_gpu_uuids':[g['uuid'] for g in probe['topology']['gpus']],
   'no_other_compute_processes_at_probe':True,'old_V100_retained':True,'no_new_login_identity':True}
target=p/'sft-phased-acceptance-approved-20261024-01.json'
with target.open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
'''
    result = subprocess.run([*SSH,PREP+'/.venv/bin/python -'],input=code,text=True,capture_output=True,check=True,timeout=60)
    root = Path(__file__).resolve().parent
    with (root/'results/sft-phased-acceptance-allocation-approved-20261024-01.json').open('x',encoding='utf-8') as stream:
        stream.write(result.stdout)
    print(result.stdout)


if __name__=='__main__':
    main()

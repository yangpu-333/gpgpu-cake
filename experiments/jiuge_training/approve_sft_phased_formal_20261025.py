"""Admit only the idle original physical pair, within one non-reset six-hour budget."""
import argparse
import json
from pathlib import Path
import re
import subprocess
from drive_sft_pinning_validation import PREP, SSH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'sg-huids25-261025-[a-z0-9]+', args.job)
    host = args.job+'-default0-0'
    code = '''from pathlib import Path
from datetime import datetime
import hashlib,json,sys,time
p=Path('''+repr(PREP)+''');sys.path.insert(0,str(p))
from sft_phased_formal_20261025 import PAIR,MANIFEST,APPROVAL,evidence_gate
host='''+repr(host)+'''
probe=json.loads((p/'sft-direct-zero-allocation-20261023'/(host+'.json')).read_text())
assert probe['hostname']==host and not probe['compute_apps'].strip()
assert 0<=time.time()-datetime.fromisoformat(probe['utc']).timestamp()<1800
assert int(probe['cgroup']['memory.max'])>=300*1024**3
q,period=map(int,probe['cgroup']['cpu.max'].split());assert 16<=q/period<17
rows=[[v.strip() for v in x.split(',')] for x in probe['cards'].strip().splitlines()]
assert len(rows)==2 and all('A100' in x[1] and x[4].startswith('0 ') and x[5].startswith('0 ') for x in rows)
assert [x['uuid'] for x in probe['topology']['gpus']]==PAIR,'Physical GPU pair mismatch: stop this job, no training'
assert len({x['node'] for x in probe['topology']['gpus']})==1
assert probe['persistent_free_bytes']>128*1024**3
m=json.loads((p/MANIFEST).read_text())
for n,h in m['files'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
evidence_gate()
start=int((p/'sft-phased-formal-bootstrap-20261025-01/campaign-first-start.epoch').read_text())
stop=start+21600
assert start<=time.time()<stop-10800,'Keep original3h formal reserve'
plan=p/'sft-phased-formal-plan-20261025-01.json'
with plan.open('x') as f:json.dump(probe['plan'],f,indent=2)
r={'hostname':host,'max_charge_points':0,'startup_limit_seconds':21600,'stop_before_epoch':stop,
   'source_files_sha256':m['files'],'plan_file':str(plan),'ordered_gpu_uuids':PAIR,
   'campaign_start_epoch':start,'old_V100_retained':True,'no_new_identity':True,
   'correctness_closed_on_Oct24':True,'old_performance_reused':False}
with APPROVAL.open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
'''
    result = subprocess.run([*SSH, PREP+'/.venv/bin/python -'], input=code,
                            text=True, capture_output=True, check=True, timeout=120)
    output = Path(__file__).resolve().parent/'results/sft-phased-formal-allocation-approved-20261025-01.json'
    with output.open('x', encoding='utf-8') as stream:
        stream.write(result.stdout)
    record = json.loads(result.stdout)
    print(json.dumps({k:record[k] for k in ('hostname','campaign_start_epoch','stop_before_epoch','max_charge_points')}))


if __name__ == '__main__':
    main()

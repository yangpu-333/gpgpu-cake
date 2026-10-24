"""Approve a necessary retry within the original, never-reset11h budget."""
import argparse
import json
from pathlib import Path
import re
import subprocess
from drive_sft_pinning_validation import SSH,PREP


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);args=parser.parse_args()
    if not re.fullmatch(r'sg-huids25-261024-[a-z0-9]+',args.job):
        raise ValueError('Exact current-user ACT job required')
    host=args.job+'-default0-0'
    code='''from pathlib import Path
from datetime import datetime
import hashlib,json,time
p=Path('''+repr(PREP)+''');host='''+repr(host)+'''
probe=json.loads((p/'sft-direct-zero-allocation-20261023'/(host+'.json')).read_text())
assert probe['hostname']==host and not probe['compute_apps'].strip()
assert 0<=time.time()-datetime.fromisoformat(probe['utc']).timestamp()<1800
assert int(probe['cgroup']['memory.max'])>=300*1024**3
q,period=map(int,probe['cgroup']['cpu.max'].split());assert 16<=q/period<17
rows=[[v.strip() for v in x.split(',')] for x in probe['cards'].strip().splitlines()]
assert len(rows)==2 and all('A100' in x[1] and x[4].startswith('0 ') and x[5].startswith('0 ') for x in rows)
old=json.loads((p/'sft-phased-acceptance-approved-20261024-01.json').read_text())
assert [x['uuid'] for x in probe['topology']['gpus']]==old['ordered_gpu_uuids']
assert len({x['node'] for x in probe['topology']['gpus']})==1
assert probe['persistent_free_bytes']>1024**4
m=json.loads((p/'sft-phased-acceptance-source-manifest-20261024-02.json').read_text())
for n,h in m['files'].items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
start=int((p/'sft-phased-acceptance-bootstrap-20261024-01/campaign-first-start.epoch').read_text())
assert start+39600==1792853953 and time.time()<1792853953-1200
plan=p/'sft-phased-acceptance-plan-20261024-02.json'
with plan.open('x') as f:json.dump(probe['plan'],f,indent=2)
r={'hostname':host,'max_charge_points':0,'startup_limit_seconds':39600,'stop_before_epoch':1792853953,
 'source_files_sha256':m['files'],'plan_file':str(plan),'ordered_gpu_uuids':old['ordered_gpu_uuids'],
 'same_original_campaign_start_epoch':start,'old_V100_retained':True,'no_new_identity':True,
 'same_native_and_control_references_reused_with_provenance':True}
with (p/'sft-phased-acceptance-approved-20261024-02.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
'''
    result=subprocess.run([*SSH,PREP+'/.venv/bin/python -'],input=code,text=True,capture_output=True,check=True,timeout=90)
    base=Path(__file__).resolve().parent
    with (base/'results/sft-phased-acceptance-allocation-approved-20261024-02.json').open('x',encoding='utf-8') as f:f.write(result.stdout)
    r=json.loads(result.stdout);print(json.dumps({'hostname':r['hostname'],'sources':len(r['source_files_sha256']),'stop_before_epoch':r['stop_before_epoch']}))


if __name__=='__main__':main()

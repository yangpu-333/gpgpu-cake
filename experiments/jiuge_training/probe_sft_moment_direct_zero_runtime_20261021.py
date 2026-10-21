"""Repeat live guarded CPU equality/resume before the new GPU screen."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from check_sft_moment_direct_zero_guard_20261021 import generated

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    gate=json.loads((PREP/'sft-moment-direct-zero-approved-20261021.json').read_text())
    host=socket.gethostname()
    if gate['hostname']!=host or gate['max_charge_points']!=0 or time.time()>=gate['stop_before_epoch']-600:
        raise ValueError('Exact authorized active allocation gate required')
    for name,expected in gate['source_files_sha256'].items():
        if hashlib.sha256((PREP/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen new proposal differs: '+name)
    root=PREP/('sft-moment-direct-zero-runtime-'+host)
    policy=(PREP/'probe_sft_page_policy_20261021.py').read_text()
    before="ROOT = PREP/'sft-moment-placement-preparation-20261021-01'"
    if policy.count(before)!=1:raise ValueError('Exact policy probe required')
    path=PREP/('sft-direct-zero-policy-runtime-'+host+'.py')
    with path.open('x') as stream:stream.write(policy.replace(before,'ROOT = Path('+repr(str(root))+')'))
    subprocess.run([sys.executable,str(path)],check=True,stdout=subprocess.DEVNULL,timeout=60)
    text=generated()
    for before,after in [("ROOT = PREP/'sft-moment-direct-zero-preparation-20261021-01/guarded-regression'",
                          'ROOT = Path('+repr(str(root/'cpu-regression'))+')'),
                         ("socket.gethostname()=='sg-huids25-260917-94135-default0-0'",'socket.gethostname()=='+repr(host))]:
        if text.count(before)!=1:raise ValueError('Exact guarded live CPU fixture required')
        text=text.replace(before,after)
    path=PREP/('sft-direct-zero-CPU-runtime-'+host+'.py')
    with path.open('x') as stream:stream.write(text)
    subprocess.run([sys.executable,str(path)],check=True,stdout=subprocess.DEVNULL,timeout=180)
    regression=json.loads((root/'cpu-regression/summary.json').read_text())
    if not regression['passed'] or regression['GPU_used']:raise ValueError('Live guarded CPU state/resume gate failed')
    result={'passed':True,'hostname':host,'ACT_root':str(root),'GPU_used':False,
            'same_cpu_regression_on_new_allocation':regression,'allocation_guard_active':True,
            'new_7B_SFT_acceptance':False}
    with (root/'summary.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()

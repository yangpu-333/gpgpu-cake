"""Repeat existing CPU fixtures inside the exact new authorized container."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    gate=json.loads((PREP/'sft-moment-screen-approved-20261021.json').read_text())
    host=socket.gethostname()
    if gate['hostname']!=host or gate['max_charge_points']!=0 or time.time()>=gate['stop_before_epoch']-600:
        raise ValueError('Exact active authorized container gate required')
    root=PREP/('sft-moment-runtime-'+host)
    probe_source=(PREP/'probe_sft_page_policy_20261021.py').read_bytes()
    check_source=(PREP/'check_sft_local_moments_20261021.py').read_bytes()
    for name,data in (('probe_sft_page_policy_20261021.py',probe_source),('check_sft_local_moments_20261021.py',check_source)):
        if hashlib.sha256(data).hexdigest()!=gate['source_files_sha256'][name]:raise ValueError('Frozen CPU fixture source changed')
    before="ROOT = PREP/'sft-moment-placement-preparation-20261021-01'"
    text=probe_source.decode()
    if text.count(before)!=1:raise ValueError('Exact private-page runtime probe derivation required')
    text=text.replace(before,'ROOT = Path('+repr(str(root))+')')
    # The generated source is stored separately and never replaces its reference.
    generated=PREP/('sft-moment-policy-runtime-'+host+'.py')
    with generated.open('x') as stream:stream.write(text)
    subprocess.run([sys.executable,str(generated)],check=True,stdout=subprocess.DEVNULL,timeout=60)
    text=check_source.decode()
    for before,after in (("ROOT = PREP/'sft-moment-placement-preparation-20261021-01/cpu-regression'",'ROOT = Path('+repr(str(root/'cpu-regression'))+')'),
                         ("socket.gethostname()=='sg-huids25-260917-94135-default0-0'",'socket.gethostname()=='+repr(host))):
        if text.count(before)!=1:raise ValueError('Exact CPU regression runtime derivation required')
        text=text.replace(before,after)
    generated=PREP/('sft-moment-CPU-runtime-'+host+'.py')
    with generated.open('x') as stream:stream.write(text)
    subprocess.run([sys.executable,str(generated)],check=True,stdout=subprocess.DEVNULL,timeout=150)
    value=json.loads((root/'cpu-regression/summary.json').read_text())
    if not value['passed'] or value['GPU_used']:raise ValueError('Live CPU placement/resume gate failed')
    result={'passed':True,'hostname':host,'ACT_root':str(root),'GPU_used':False,
            'same_cpu_regression_on_new_allocation':value,'new_7B_SFT_acceptance':False}
    with (root/'summary.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()

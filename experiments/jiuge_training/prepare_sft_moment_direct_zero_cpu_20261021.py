"""Detached bounded CPU preparation; never allocates a new ACT/GPU resource."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT=PREP/'sft-moment-direct-zero-preparation-20261021-01'
NEW=('sft_local_moments_v3_20261021.py','check_sft_local_moments_v3_20261021.py',
     'probe_sft_moment_peak_v3_20261021.py','prepare_sft_moment_direct_zero_cpu_20261021.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert socket.gethostname()=='sg-huids25-260917-94135-default0-0'
    assert os.getuid()==1001 and not ROOT.exists()
    prior=json.loads((PREP/'sft-moment-screen-source-manifest-v2-20261021.json').read_text())['files']
    assert len(prior)==73
    for name,expected in prior.items():assert sha(PREP/name)==expected,name
    source=(PREP/'probe_sft_page_policy_20261021.py').read_text()
    before='sft-moment-placement-preparation-20261021-01'
    assert source.count(before)==1
    generated=PREP/'probe_sft_page_policy_direct_zero_20261021.generated.py'
    with generated.open('x') as stream:stream.write(source.replace(before,ROOT.name))
    subprocess.run([sys.executable,str(generated)],check=True,stdout=subprocess.DEVNULL,timeout=60)
    frozen=ROOT/'source';frozen.mkdir()
    hashes={**prior,**{name:sha(PREP/name) for name in NEW}}
    for name,expected in hashes.items():
        assert sha(PREP/name)==expected
        shutil.copyfile(PREP/name,frozen/name)
    with (ROOT/'source-manifest.json').open('x') as stream:
        json.dump({'files':hashes,'original73_unchanged':True,'GPU_started':False},stream,indent=2)

    def status(value,**extra):
        d={'status':value,'utc':datetime.now(timezone.utc).isoformat(),
           'GPU_used':False,'raw_originals_remain_ACT':True,**extra}
        tmp=ROOT/'status.tmp';tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(ROOT/'status.json')

    try:
        for name,stage in [('check_sft_local_moments_v3_20261021.py','CPU_full_state_resume'),
                           ('probe_sft_moment_peak_v3_20261021.py','isolated_CPU_memory_peak')]:
            status(stage)
            with (ROOT/(stage+'.log')).open('x') as stream:
                subprocess.run([sys.executable,str(PREP/name)],stdout=stream,
                               stderr=subprocess.STDOUT,check=True,timeout=1000)
        regression=json.loads((ROOT/'cpu-regression/summary.json').read_text())
        memory=json.loads((ROOT/'memory-probe/summary.json').read_text())
        assert regression['passed'] and not regression['GPU_used'] and not memory['GPU_used']
        for name,expected in hashes.items():assert sha(PREP/name)==expected,name
        files={str(f.relative_to(ROOT)):{'bytes':f.stat().st_size,'sha256':sha(f)}
               for f in ROOT.rglob('*') if f.is_file() and f.name!='status.json'}
        with (ROOT/'manifest.json').open('x') as stream:json.dump({'files':files},stream,indent=2)
        for name,expected in files.items():assert sha(ROOT/name)==expected['sha256'],name
        summary={'schema':'jiuge7b-sft-direct-zero-CPU-preparation-v3',
                 'passed':regression['passed'] and memory['measured_peak_reduction_passed'],
                 'ACT_root':str(ROOT),'CPU_regression':regression,
                 'memory_comparisons':memory['comparisons'],'GPU_used':False,
                 'original73_sources_unchanged':True,'new_candidate_7B_OOM_fixed':False,
                 'new_SFT_acceptance':False,'formal_performance_evidence':False,
                 'ACT_verified_files':len(files),'ACT_bytes':sum(f['bytes'] for f in files.values()),
                 'manifest_sha256':sha(ROOT/'manifest.json')}
        with (ROOT/'summary.json').open('x') as stream:json.dump(summary,stream,indent=2)
        status('completed_CPU_preparation',passed=summary['passed'],new_GPU_allocation_created=False)
        print(json.dumps(summary),flush=True)
    except BaseException as exc:
        status('failed',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__=='__main__':main()

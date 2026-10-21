"""Fixed fresh three-arm affinity benchmark and the existing CV repeat rule."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from analyze_numa_formal import analyze
from analyze_pinning_formal import SEEDS
from continue_pinning_formal import resources
from continue_pinning_validation import remaining_seconds
from numa_admission import check

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')
ORDERS=(('native','control','optimized'),('optimized','control','native'),('control','native','optimized'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission',type=Path,required=True)
    args=parser.parse_args()
    if socket.gethostname()!='sg-huids25-261016-13441-default0-0':
        raise ValueError('Only the authorized zero-cost successor allocation may run this formal campaign')
    admitted=check(args.admission)
    root=PREP/'cpu-numa-formal-20261016-01'; root.mkdir(exist_ok=False)
    source=root/'source'; source.mkdir()
    names=tuple(dict.fromkeys((*admitted['frozen_source_sha256'],'continue_numa_formal.py','run_numa_formal.py',
                              'analyze_numa_formal.py','analyze_pinning_formal.py','numa_admission.py','pinning_formal_entry.py')))
    hashes={}
    for name in names:
        data=(PREP/name).read_bytes(); (source/name).write_bytes(data); hashes[name]=hashlib.sha256(data).hexdigest()
    state={'schema':'jiuge7b-cpu-numa-formal-controller-v1','status':'initializing',
           'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':hashes,'completed_cases':[],
           'rounds':[],'formal_performance_passed':False,'host':socket.gethostname(),
           'baseline':'Previously accepted pinned repair; original unbound placement',
           'prior_unpinned_or_pinned_timing_reused':False,'remote_probes_inside_timed_cases':False}
    def save():
        temp=root/'controller.tmp'; temp.write_text(json.dumps(state,indent=2)+'\n'); temp.replace(root/'controller.json')
    def round_run(label):
        if remaining_seconds()<10100:
            raise TimeoutError('Require complete conservative formal-round free-time reserve')
        before=resources(); state.update(status='running_'+label); save()
        for seed,order in zip(SEEDS,ORDERS):
            for arm in order:
                if remaining_seconds()<1800:
                    raise TimeoutError('Free allocation reserve reached')
                if any(hashlib.sha256((PREP/name).read_bytes()).hexdigest()!=digest for name,digest in hashes.items()):
                    raise ValueError('Frozen formal source changed; no retuning or silent replacement')
                gate=resources()
                case=f'cpu-numa-formal-{label}-{seed}-{arm}'
                print(json.dumps({'event':'case_start','case':case}),flush=True)
                with (root/f'{case}.log').open('x') as stream:
                    subprocess.run([sys.executable,PREP/'run_numa_formal.py','--case',case,'--steps','120',
                                    '--seq-length','2048','--seed',str(seed),'--numa-arm',arm,
                                    '--pinning-campaign',admitted['validation_root'],'--admission',args.admission],
                                   cwd=PREP,stdout=stream,stderr=subprocess.STDOUT,check=True,
                                   timeout=max(1,remaining_seconds()-600))
                time.sleep(3); after=resources()
                metadata={'resource_before':gate,'resource_after':after,
                          'cpu_quota':Path('/sys/fs/cgroup/cpu.max').read_text(),
                          'memory_limit':Path('/sys/fs/cgroup/memory.max').read_text(),
                          'cpu_stat_after':Path('/sys/fs/cgroup/cpu.stat').read_text(),
                          'host_loadavg_after':Path('/proc/loadavg').read_text()}
                (root/f'{case}-resources.json').write_text(json.dumps(metadata,indent=2)+'\n')
                state['completed_cases'].append(case); save()
                print(json.dumps({'event':'case_complete','case':case}),flush=True)
        result=analyze(PREP,label,admitted); result.update(resource_gate_before=before,resource_gate_after=resources())
        (root/f'{label}-result.json').write_text(json.dumps(result,indent=2)+'\n')
        state['rounds'].append({name:result[name] for name in ('round','round_promotion_passed','repeat_required','seed_mean_cv','vs_native','vs_control')}); save()
        print(json.dumps({'event':'round_complete',**state['rounds'][-1]}),flush=True)
        return result
    save()
    try:
        primary=round_run('primary')
        confirmation=round_run('confirmation') if primary['repeat_required'] else None
        passed=primary['round_promotion_passed'] and (confirmation is None or confirmation['round_promotion_passed'])
        state.update(status='completed_promoted' if passed else 'completed_not_promoted',
                     formal_performance_passed=passed,repeat_performed=confirmation is not None)
    except BaseException as exc:
        state.update(status='stopped_incomplete',error_type=type(exc).__name__,error=str(exc)); raise
    finally:
        state['updated_utc']=datetime.now(timezone.utc).isoformat(); save()
        print(json.dumps({name:value for name,value in state.items() if name!='source_sha256'}),flush=True)


if __name__=='__main__':
    main()

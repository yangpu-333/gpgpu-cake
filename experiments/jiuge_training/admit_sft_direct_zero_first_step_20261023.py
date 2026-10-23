"""Run a separate memory admission; only its success releases the six cases."""
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from sft_numa_placement_20261021 import discover, build_plan
from continue_sft_moment_direct_zero_20261021 import moment_gate

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT = PREP/'sft-direct-zero-first-step-20261023-01'


def main():
    gate_path = PREP/'sft-direct-zero-first-step-approved-20261023.json'
    gate = json.loads(gate_path.read_text())
    if (gate['hostname'] != socket.gethostname() or gate['max_charge_points'] != 0
            or time.time() >= gate['stop_before_epoch']-600):
        raise ValueError('Exact unexpired zero-cost allocation required')
    if gate_path.read_bytes() != (PREP/'sft-moment-direct-zero-approved-20261021.json').read_bytes():
        raise ValueError('Single-update and six-case allocation gates must agree')
    ROOT.mkdir(exist_ok=False)
    source = ROOT/'source'
    source.mkdir()
    for name, sha in gate['source_files_sha256'].items():
        if Path(name).name != name or hashlib.sha256((PREP/name).read_bytes()).hexdigest() != sha:
            raise ValueError('Frozen approved source differs: '+name)
        shutil.copyfile(PREP/name, source/name)
    shutil.copyfile(gate_path, ROOT/'approval.json')
    plan = build_plan(discover())
    if ([g['uuid'] for g in plan['topology']['gpus']] != gate['ordered_gpu_uuids']
            or len({g['node'] for g in plan['topology']['gpus']}) != 1):
        raise ValueError('Exact ordered same-NUMA card pair required')
    plan_path = ROOT/'affinity-plan.json'
    with plan_path.open('x') as stream:
        json.dump(plan, stream, indent=2)
    candidate = ROOT/'candidate'
    def run(name, args):
        left = min(1900, int(gate['stop_before_epoch']-time.time()-180))
        if left < 120:
            raise TimeoutError('Preserve independent shutdown reserve')
        with (ROOT/(name+'.log')).open('x') as stream:
            subprocess.run([sys.executable,*map(str,args)],cwd=PREP,stdout=stream,
                           stderr=subprocess.STDOUT,timeout=left,check=True)
    state = {'status':'preparing', 'hostname':socket.gethostname(),
             'stop_before_epoch':gate['stop_before_epoch'], 'max_charge_points':0,
             'full_SFT_correctness_acceptance':False, 'formal_performance_evidence':False}
    def save():
        temp = ROOT/'status.tmp'
        temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(ROOT/'status.json')
    save()
    try:
        run('candidate-prepare',[PREP/'pinning_candidate.py','--output',candidate])
        case = 'sft-direct-zero-first-step-20261023-01'
        state.update(status='running_one_update',case=case)
        save()
        run('first-step',[PREP/'run_sft_direct_zero_first_step_20261023.py','--case',case,
                          '--steps','1','--seq-length','2048','--pinning-arm','optimized',
                          '--pinning-campaign',candidate,'--sft-numa-mode','gpu-local8',
                          '--sft-numa-plan',plan_path,'--sft-moment-arm','local-moments'])
        folder = PREP/'runs'/case
        result = json.loads((folder/'first-step-memory-admission.json').read_text())
        if not result['passed']:
            raise ValueError('First-update memory admission failed')
        for rank in (0,1):
            binding = json.loads((folder/f'sft-numa-binding-rank-{rank}.json').read_text())
            moment_gate(folder,rank,'local-moments',binding)
        state.update(status='passed_first_update_memory_admission',
                     first_step_receipt=str(folder/'first-step-memory-admission.json'),
                     actual_moment_page_gate_passed=True, six_fresh_cases_released=True)
        save()
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),six_fresh_cases_released=False)
        save()
        raise
    print(json.dumps(state),flush=True)


if __name__ == '__main__':
    main()

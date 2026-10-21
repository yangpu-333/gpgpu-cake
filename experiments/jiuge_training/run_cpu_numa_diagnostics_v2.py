"""Continue diagnostics with unique immutable pinning receipts per case."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from continue_pinning_validation import remaining_seconds
from continue_pinning_formal import resources
from pinning_key_results import export

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
CODE = ('cpu_numa.contract.json','run_cpu_numa_diagnostics_v2.py','run_numa_case.py','numa_entry.py','numa_placement.py',
        'pinning_entry.py','pinning_candidate.py','native_entry.py','run_native_case.py','review_native_trace.py',
        'inspect_trace_communication.py','analyze_trace.py','archive_case.py')


def main():
    if socket.gethostname() != 'sg-huids25-261016-24e4c-default0-0':
        raise ValueError('Wrong allocation')
    old = PREP/'cpu-numa-diagnostics-20261016-01'
    root = PREP/'cpu-numa-diagnostics-20261016-02'
    root.mkdir(exist_ok=False)
    (root/'source').mkdir()
    hashes = {}
    for name in CODE:
        data=(PREP/name).read_bytes(); (root/'source'/name).write_bytes(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
    state={'schema':'jiuge7b-cpu-numa-diagnostic-controller-v2','status':'waiting_original_controller',
           'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':hashes,'cases':[],
           'formal_performance_passed':False,'prior_failure_preserved':str(old),
           'orchestration_fix':'Unique pinning campaign per case; no immutable receipt overwrite or optimizer change'}
    def save():
        temp=root/'controller.tmp'; temp.write_text(json.dumps(state,indent=2)+'\n'); temp.replace(root/'controller.json')
    def run(name,argv):
        if remaining_seconds()<900:
            raise TimeoutError('Preserve free allocation reserve')
        with (root/f'{name}.log').open('x') as stream:
            subprocess.run([sys.executable,*map(str,argv)],cwd=PREP,stdout=stream,stderr=subprocess.STDOUT,
                           check=True,timeout=max(1,remaining_seconds()-600))
    save()
    try:
        while True:
            previous=json.loads((old/'controller.json').read_text())
            if previous['status']=='stopped':
                break
            if previous['status']=='completed_diagnostic_only':
                raise ValueError('Original diagnostics completed; no duplicate workload required')
            if remaining_seconds()<3000:
                raise TimeoutError('Insufficient time to finish diagnostics on this allocation')
            time.sleep(20)
        if [row['case'] for row in previous['cases']]!=['cpu-numa-unbound-01']:
            raise ValueError('Expected exactly the completed first unbound reference; inspect earlier failure')
        if 'CalledProcessError'!=previous.get('error_type'):
            raise ValueError('Unexpected original failure')
        error=(PREP/'runs/cpu-numa-local8-02/stderr.log').read_text()
        if 'FileExistsError' not in error or 'actual-pinning-init-1.json' not in error:
            raise ValueError('Only the diagnosed immutable receipt collision can be continued automatically')
        proof=json.loads((old/'cpu-numa-unbound-01-primary-verification.json').read_text())
        if not proof['passed']:
            raise ValueError('First reference archive did not pass')
        state['cases'].append({'case':'cpu-numa-unbound-01','mode':'unbound','reused_completed_reference':True,
                               'archive_proof_sha256':hashlib.sha256((old/'cpu-numa-unbound-01-primary-verification.json').read_bytes()).hexdigest()})
        for index,mode in ((2,'local8'),(3,'unbound')):
            if any(hashlib.sha256((PREP/name).read_bytes()).hexdigest()!=digest for name,digest in hashes.items()):
                raise ValueError('Frozen diagnostics source changed')
            before=resources()
            case=f'cpu-numa-{mode}-{index:02d}-v2'; case_root=PREP/'runs'/case
            campaign=root/f'candidate-{index:02d}'
            run('prepare-'+str(index),[PREP/'pinning_candidate.py','--output',campaign])
            state.update(status='running_'+case,current_resource_gate=before); save()
            print(json.dumps({'event':'case_start','case':case}),flush=True)
            run(case,[PREP/'run_numa_case.py','--case',case,'--steps','35','--seq-length','2048',
                      '--numa-mode',mode,'--pinning-campaign',campaign,'--capture-windows'])
            resources()
            run(case+'-trace',[PREP/'review_native_trace.py','--case-root',case_root,'--output',case_root/'trace-analysis.json'])
            run(case+'-comm',[PREP/'inspect_trace_communication.py','--case-root',case_root,'--output',case_root/'communication-analysis.json'])
            run(case+'-archive',[PREP/'archive_case.py','--case-root',case_root,'--create','--snapshot-analysis'])
            run(case+'-verify',[PREP/'archive_case.py','--case-root',case_root,'--verify','--receipt',root/f'{case}-primary-verification.json'])
            state['cases'].append({'case':case,'mode':mode,'resource_gate_passed':True}); save()
            print(json.dumps({'event':'case_complete','case':case}),flush=True)
        state.update(status='completed_diagnostic_only',resource_after=resources(),finished_utc=datetime.now(timezone.utc).isoformat()); save()
        export(PREP,[root,old,*[PREP/'runs'/row['case'] for row in state['cases']],PREP/'runs/cpu-numa-local8-02'],
               PREP/'cpu-numa-diagnostic-key-results-20261016-v2.zip')
        print(json.dumps({'status':state['status'],'cases':state['cases']}),flush=True)
    except BaseException as exc:
        state.update(status='stopped',error_type=type(exc).__name__,error=str(exc)); save(); raise


if __name__=='__main__':
    main()

"""Frozen unbound/local8/unbound diagnostic; never a formal promotion."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from continue_pinning_validation import remaining_seconds
from continue_pinning_formal import resources
from pinning_key_results import export

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
CODE = ('cpu_numa.contract.json','run_cpu_numa_diagnostics.py','run_numa_case.py','numa_entry.py','numa_placement.py',
        'pinning_entry.py','pinning_candidate.py','native_entry.py','run_native_case.py','review_native_trace.py',
        'inspect_trace_communication.py','analyze_trace.py','archive_case.py')


def main():
    if socket.gethostname() != 'sg-huids25-261016-24e4c-default0-0':
        raise ValueError('Wrong resource allocation')
    if remaining_seconds() < 3600:
        raise TimeoutError('Require enough existing free time for all three diagnostics')
    root = PREP / 'cpu-numa-diagnostics-20261016-01'
    root.mkdir(exist_ok=False)
    (root / 'source').mkdir()
    hashes = {}
    for name in CODE:
        data = (PREP / name).read_bytes()
        (root / 'source' / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    state = {'schema':'jiuge7b-cpu-numa-diagnostic-controller-v1','started_utc':datetime.now(timezone.utc).isoformat(),
             'status':'initializing','source_sha256':hashes,'cases':[], 'formal_performance_passed':False,
             'originals_location':str(root),'raw_download_enabled':False,'resource_before':resources()}
    def save():
        temp = root / 'controller.tmp'
        temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(root/'controller.json')
    def run(name, command):
        if remaining_seconds() < 900:
            raise TimeoutError('Preserve free allocation reserve')
        with (root / f'{name}.log').open('x') as stream:
            subprocess.run([sys.executable,*map(str,command)],cwd=PREP,stdout=stream,stderr=subprocess.STDOUT,
                           check=True,timeout=max(1,remaining_seconds()-600))
    save()
    try:
        # CPU-only query probe; no allocation or migration of another process.
        import torch
        from numa_placement import sampled_page_nodes
        sample = sampled_page_nodes([('own_cpu_probe',torch.zeros(8192))])
        (root/'own-page-query-probe.json').write_text(json.dumps(sample,indent=2)+'\n')
        campaign = PREP / 'pinning-validation-20261016-10'
        run('prepare',[PREP/'pinning_candidate.py','--output',campaign])
        for index,mode in enumerate(('unbound','local8','unbound'),1):
            if any(hashlib.sha256((PREP/name).read_bytes()).hexdigest()!=digest for name,digest in hashes.items()):
                raise ValueError('Frozen diagnostic sources changed')
            before = resources()
            case = f'cpu-numa-{mode}-{index:02d}'
            case_root = PREP/'runs'/case
            state.update(status='running_'+case,current_resource_gate=before)
            save()
            print(json.dumps({'event':'case_start','case':case}),flush=True)
            run(case,[PREP/'run_numa_case.py','--case',case,'--steps','35','--seq-length','2048',
                      '--numa-mode',mode,'--pinning-campaign',campaign,'--capture-windows'])
            resources()
            run(case+'-trace',[PREP/'review_native_trace.py','--case-root',case_root,'--output',case_root/'trace-analysis.json'])
            run(case+'-comm',[PREP/'inspect_trace_communication.py','--case-root',case_root,'--output',case_root/'communication-analysis.json'])
            run(case+'-archive',[PREP/'archive_case.py','--case-root',case_root,'--create','--snapshot-analysis'])
            run(case+'-verify',[PREP/'archive_case.py','--case-root',case_root,'--verify','--receipt',root/f'{case}-primary-verification.json'])
            state['cases'].append({'case':case,'mode':mode,'resource_gate_passed':True})
            save()
            print(json.dumps({'event':'case_complete','case':case}),flush=True)
        state.update(status='completed_diagnostic_only',resource_after=resources(),finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        bundle = PREP/'cpu-numa-diagnostic-key-results-20261016.zip'
        export(PREP,[root,campaign,*[PREP/'runs'/row['case'] for row in state['cases']]],bundle)
        print(json.dumps({'status':state['status'],'key_bundle':str(bundle)}),flush=True)
    except BaseException as exc:
        state.update(status='stopped',error_type=type(exc).__name__,error=str(exc))
        save()
        raise


if __name__=='__main__':
    main()

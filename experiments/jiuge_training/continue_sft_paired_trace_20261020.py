"""Capture ABBA on one allocated pair; raw traces and logs stay on ACT."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from pinning_key_results import export

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT = PREP/'sft-paired-tracing-20261020-01'
GATE = PREP/'sft-paired-trace-approved-20261020.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024*1024):
            h.update(block)
    return h.hexdigest()


def main():
    approval = json.loads(GATE.read_text())
    if approval['hostname'] != socket.gethostname() or approval['max_charge_points'] != 0:
        raise ValueError('Wrong allocated job/budget')
    ROOT.mkdir(exist_ok=False)
    source = ROOT/'source'
    source.mkdir()
    for name,sha in approval['source_files_sha256'].items():
        if digest(PREP/name) != sha:
            raise ValueError('Frozen source changed: '+name)
        shutil.copyfile(PREP/name,source/name)
    shutil.copyfile(GATE,ROOT/'approval.json')
    state = {'schema':'jiuge7b-sft-paired-tracing-controller-v1','status':'initializing',
             'started_utc':datetime.now(timezone.utc).isoformat(), 'hostname':socket.gethostname(),
             'cases':[], 'max_charge_points':0,'stop_before_epoch':approval['stop_before_epoch'],
             'formal_performance_evidence':False,'raw_traces_local_downloaded':False}

    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = ROOT/'controller.tmp'
        temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(ROOT/'controller.json')

    def run(name,args):
        left = int(approval['stop_before_epoch']-time.time()-180)
        if left < 120:
            raise TimeoutError('Independent zero-cost job reserve reached')
        with (ROOT/(name+'.log')).open('x') as stream:
            subprocess.run([sys.executable,*map(str,args)],cwd=PREP,stdout=stream,
                           stderr=subprocess.STDOUT,check=True,timeout=left)

    def resources():
        cards = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
        uuids = [line.split(',')[0].strip() for line in cards.strip().splitlines()]
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
        if uuids != approval['ordered_gpu_uuids'] or len(uuids) != 2 or apps.strip():
            raise ValueError('Assigned pair changed or another GPU process is present')
        return {'cards':cards,'compute_apps':apps,'utc':datetime.now(timezone.utc).isoformat()}

    save()
    try:
        state['resource_gate_before'] = resources()
        state['cpu_max'] = Path('/sys/fs/cgroup/cpu.max').read_text().strip()
        state['memory_max'] = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        state['gpu_topology'] = subprocess.check_output(['nvidia-smi','topo','-m'],text=True)
        campaign = ROOT/'candidate'
        run('candidate-prepare',[PREP/'pinning_candidate.py','--output',campaign])
        order = [('native','a1'),('optimized','b1'),('optimized','b2'),('native','a2')]
        for arm,tag in order:
            if approval['stop_before_epoch']-time.time() < 1200:
                raise TimeoutError('Insufficient reserve for next complete capture')
            for name,sha in approval['source_files_sha256'].items():
                if digest(PREP/name) != sha:
                    raise ValueError('Frozen source changed before capture: '+name)
            gate = resources()
            case = f'sft-paired-trace-20261020-{tag}-{arm}'
            folder = PREP/'runs'/case
            # Per-case candidate directory avoids receipt filename collisions.
            isolated = ROOT/('candidate-'+tag)
            isolated.mkdir()
            for name in ('native_hybrid_optimizer.py','candidate_hybrid_optimizer.py','candidate.patch','candidate.json'):
                shutil.copyfile(campaign/name,isolated/name)
            state.update(status='capturing_'+tag+'_'+arm,active_case=case)
            save()
            run(case,[PREP/'run_sft_trace_case_20261020.py','--case',case,'--steps','35',
                      '--seq-length','2048','--pinning-arm',arm,'--pinning-campaign',isolated,'--capture-windows'])
            run(case+'-completion',[PREP/'verify_trace_case_completion.py','--case-root',folder,'--steps','35','--sft','--profile'])
            objective = [json.loads((folder/f'sft-trace-objective-rank-{r}.json').read_text()) for r in (0,1)]
            for rank,item in enumerate(objective):
                pin = json.loads((isolated/f'{arm}-rank-{rank}-actual-pinning-init-1.json').read_text())
                if (not item['run_returned'] or pin['parameter_count'] != 199
                    or pin['pinned_count'] != (199 if arm == 'optimized' else 0)
                    or item['range_call_counts']['jiuge/cpu_adamw_math'] != 35*199
                    or item['data_receipt_sha256'] != '1e30f6af3c21d31ee6fae117bfec9c25120bb1b14fe86d815d4f059fdc7a8d08'):
                    raise ValueError('Actual pinning, AdamW calls or SFT objective differs')
            state['cases'].append({'case':case,'arm':arm,'tag':tag,'gate_before':gate,'gate_after':resources()})
            save()
        state.update(status='analyzing',active_case=None)
        save()
        reference = None
        folders = []
        for item in state['cases']:
            folder = PREP/'runs'/item['case']
            rows = json.loads((folder/'sft-supervision-rank-0.json').read_text())['rows']
            losses = json.loads((folder/'completion-verification.json').read_text())['native_losses']
            if reference is None:
                reference = (rows,losses)
            elif (rows,losses) != reference:
                raise ValueError('Paired samples/masks or all35 native logged losses differ')
            for script,name in [('review_native_trace.py','trace-analysis.json'),
                                ('inspect_trace_communication.py','communication-analysis.json'),
                                ('analyze_sft_paired_trace_20261020.py','symmetric-mechanism.json')]:
                run(item['case']+'-'+name,[PREP/script,'--case-root',folder,'--output',folder/name])
            run(item['case']+'-archive',[PREP/'archive_case.py','--case-root',folder,'--create','--snapshot-analysis'])
            run(item['case']+'-archive-verify',[PREP/'archive_case.py','--case-root',folder,'--verify',
                                             '--receipt',ROOT/(item['case']+'-archive-verification.json')])
            folders.append(folder)
        state.update(status='completed',paired_samples_masks_and_all35_losses_equal=True,
                     verified_raw_traces=16,resource_gate_after=resources(),
                     finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        bundle = PREP/'sft-paired-trace-key-results-20261020.zip'
        catalog = export(PREP,[ROOT,*folders],bundle,32*1024**2)
        receipt = {'path':str(bundle),'bytes':bundle.stat().st_size,'sha256':digest(bundle),
                   'key_files':catalog['key_file_count'],'key_uncompressed_bytes':catalog['key_uncompressed_bytes']}
        with (ROOT/'final-key-bundle.json').open('x') as stream:
            json.dump(receipt,stream,indent=2)
        print(json.dumps(receipt),flush=True)
    except BaseException as exc:
        state.update(status='stopped',error_type=type(exc).__name__,error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()

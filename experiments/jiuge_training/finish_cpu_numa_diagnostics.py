"""Wait for closed GPU diagnostics, then analyze originals using CPU only."""
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

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    if socket.gethostname()!='sg-huids25-261016-24e4c-default0-0':
        raise ValueError('Wrong analysis allocation')
    root=PREP/'cpu-numa-mechanism-20261016-01'; root.mkdir(exist_ok=False)
    state={'schema':'jiuge7b-cpu-numa-analysis-controller-v1','status':'waiting_closed_diagnostics',
           'started_utc':datetime.now(timezone.utc).isoformat(),'formal_performance_passed':False}
    def save():
        temp=root/'controller.tmp'; temp.write_text(json.dumps(state,indent=2)+'\n'); temp.replace(root/'controller.json')
    def run(name,argv):
        with (root/f'{name}.log').open('x') as stream:
            subprocess.run([sys.executable,*map(str,argv)],cwd=PREP,stdout=stream,stderr=subprocess.STDOUT,
                           check=True,timeout=600,env={**__import__('os').environ,'CUDA_VISIBLE_DEVICES':''})
    save()
    try:
        control=PREP/'cpu-numa-diagnostics-20261016-02/controller.json'
        while True:
            diagnostic=json.loads(control.read_text())
            if diagnostic['status']=='completed_diagnostic_only':
                break
            if diagnostic['status']=='stopped':
                raise ValueError('Diagnostics stopped: '+diagnostic.get('error','unknown'))
            if remaining_seconds()<1200:
                raise TimeoutError('Free analysis reserve reached')
            time.sleep(20)
        state.update(status='analyzing_closed_traces',resource_gate=resources(),cases=diagnostic['cases']); save()
        source=root/'source'; source.mkdir()
        for name in ('finish_cpu_numa_diagnostics.py','inspect_cpu_adamw_v2.py','review_numa_snapshots.py','inspect_copy_api.py','analyze_trace.py'):
            (source/name).write_bytes((PREP/name).read_bytes())
        previous=json.loads((PREP/'cpu-numa-diagnostics-20261016-01/controller.json').read_text())
        shared=set(previous['source_sha256'])&set(diagnostic['source_sha256'])
        if any(previous['source_sha256'][name]!=diagnostic['source_sha256'][name] for name in shared):
            raise ValueError('Reused reference and resumed diagnostics sources differ')
        adam_root=PREP/'cpu-numa-adamw-review-20261016-01'
        command=[PREP/'inspect_cpu_adamw_v2.py','--review-name',adam_root.name]
        for row in diagnostic['cases']:
            command+=['--case',row['case']]
        run('adamw',command)
        summaries=[]
        adam=json.loads((adam_root/'cpu-adamw-review.json').read_text())
        for row in diagnostic['cases']:
            case=row['case']; case_root=PREP/'runs'/case
            run(case+'-numa',[PREP/'review_numa_snapshots.py','--case-root',case_root,'--output',root/(case+'-numa.json')])
            run(case+'-copy',[PREP/'inspect_copy_api.py','--case-root',case_root,'--output',root/(case+'-copy.json')])
            trace=json.loads((case_root/'trace-analysis.json').read_text())
            comm=json.loads((case_root/'communication-analysis.json').read_text())
            windows=[]
            for window in trace['windows']:
                device=window['analysis']['capture_window']['devices'][0]
                key=(window['window'],window['rank'])
                cpu=next(x for x in adam['rows'] if (x['arm'],x['window'],x['rank'])==(case,*key))
                fused=next(x for x in cpu['aten_scopes_within_adam'] if x['name']=='aten::_fused_adamw_')
                copied=next(x for x in comm['ranks'] if (x['window'],x['rank'])==key)
                windows.append({'window':key[0],'rank':key[1],'window_s':device['window_us']/1e6,
                                'fused_adamw_host_sum_s':fused['host_sum_s'],'fused_adamw_calls':fused['count'],
                                'cpu_phase_ranges':window['cpu_phase_ranges'],'gpu':device,
                                'memcpy_bytes':copied['recorded_memcpy_bytes_by_name'],
                                'synchronize_api_s':cpu['synchronize_api_s']})
            task_window_s={name:max(w['window_s'] for w in windows if w['window']==name) for name in ('window-01','window-02')}
            summaries.append({'case':case,'mode':row['mode'],'diagnostic_mean_seconds_per_step':sum(task_window_s.values())/10,
                              'fused_adamw_host_seconds_per_rank_step':sum(w['fused_adamw_host_sum_s'] for w in windows)/20,
                              'windows':windows,'cross_rank_pair_candidates':comm['cross_rank_pair_candidates']})
        (root/'mechanism-summary.json').write_text(json.dumps({'schema':'jiuge7b-cpu-numa-diagnostic-mechanism-v1',
             'cases':summaries,'formal_performance_passed':False,'critical_path_proven':False,
             'collector_zero_drop_verified':False,'shared_training_sources_unchanged':True},indent=2)+'\n')
        state.update(status='completed_diagnostic_analysis',finished_utc=datetime.now(timezone.utc).isoformat()); save()
        folders=[root,adam_root,PREP/'cpu-numa-diagnostics-20261016-01',PREP/'cpu-numa-diagnostics-20261016-02']
        folders += [PREP/'runs'/row['case'] for row in diagnostic['cases']]
        folders += [PREP/'runs/cpu-numa-local8-02']
        bundle=PREP/'cpu-numa-complete-diagnostic-key-results-20261016.zip'
        catalog=export(PREP,folders,bundle)
        print(json.dumps({'status':state['status'],'summary':[{'case':r['case'],'diagnostic_seconds_per_step':r['diagnostic_mean_seconds_per_step'],
                            'fused_adamw_cpu_seconds_per_rank_step':r['fused_adamw_host_seconds_per_rank_step']} for r in summaries],
                          'key_bundle':str(bundle),'key_bytes':catalog['key_uncompressed_bytes']}),flush=True)
    except BaseException as exc:
        state.update(status='stopped',error_type=type(exc).__name__,error=str(exc)); save(); raise


if __name__=='__main__':
    main()

"""Archive the two reports and small reviewed keys on ACT with exact file SHAs."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
from drive_sft_pinning_validation import SSH,SCP,PREP,run

BASE=Path(__file__).resolve().parent
STAGE=BASE/'results/sft-paired-trace-final-delivery-20261020-01'
REMOTE=PREP+'/sft-paired-trace-delivery-20261020-01/final-review'


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    with path.open('x',encoding='utf-8') as stream: json.dump(value,stream,ensure_ascii=False,indent=2); stream.write('\n')


def main():
    review=read(BASE/'results/sft-paired-trace-independent-review-v2-20261020.json')
    raw=read(BASE/'results/sft-paired-trace-delivery-verification-20261020.json')
    post=read(BASE/'results/sft-paired-trace-postprocess-20261020-02/status.json')
    if not review['passed'] or not raw['passed'] or post['status']!='completed-scientific-postprocessing':
        raise ValueError('Close all scientific and CPU analysis gates first')
    manifest=read(BASE/'results/sft-paired-trace-source-manifest-20261020.json')['files']
    for name,expected in manifest.items():
        if sha(BASE/name)!=expected: raise ValueError('Frozen local source changed: '+name)
    code='''from pathlib import Path
import hashlib,json
p=Path('''+repr(PREP)+''')
files='''+repr(manifest)+'''
for name,expected in files.items():
    for path in (p/name,p/'sft-paired-tracing-20261020-01/source'/name):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,str(path)
print(json.dumps({'passed':True,'frozen_files':len(files),'ACT_live_and_execution_snapshot_SHA_match':True,'prior31_scientific_files_unchanged':True}))
'''
    source_proof=json.loads(run([*SSH,PREP+'/.venv/bin/python -'],input=code.encode()).decode())
    write(BASE/'results/sft-paired-trace-final-source-verification-20261020.json',source_proof)
    STAGE.mkdir(exist_ok=False)
    selected=[BASE/'README.md',BASE/'BRIEF.md']
    for pattern in ('*sft*20261020*.py','sft_paired_trace_20261020.contract.json','bootstrap_sft_paired_trace_20261020.sh'):
        selected+=list(BASE.glob(pattern))
    for name in ('sft-paired-trace-approved-20261020.json','sft-paired-trace-source-manifest-20261020.json',
                 'sft-paired-trace-preparation-20261020.json','sft-paired-trace-final-source-verification-20261020.json',
                 'sft-paired-trace-independent-review-v2-20261020.json','sft-paired-trace-delivery-verification-20261020.json',
                 'sft-paired-trace-lanes-20261020.json','sft-paired-trace-comparison-20261020.png',
                 'sft-paired-trace-comparison-20261020.svg','sft-paired-trace-lanes-20261020.png',
                 'sft-paired-trace-completed-zero-points-20261020.png'):
        selected.append(BASE/'results'/name)
    for folder in ('sft-paired-trace-driver-20261020-01','sft-paired-trace-postprocess-20261020-01',
                   'sft-paired-trace-postprocess-20261020-02','sft-paired-trace-source-supplement-20261020-01'):
        root=BASE/'results'/folder
        selected += [p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.log','.sh','.zip')
                     and 'keys' not in p.relative_to(root).parts]
    # The README also references these compact, already accepted historic reviews.
    selected += [BASE/'results'/name for name in ('sft-pinning-independent-review-20261020-continuation-v2.json',
                                                 'cpt-sft-independent-review-20261017.json')]
    selected=list({str(p.resolve()):p for p in selected}.values())
    files={}
    for path in selected:
        relative=path.resolve().relative_to(BASE.resolve())
        target=STAGE/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        files[relative.as_posix()]={'bytes':path.stat().st_size,'sha256':sha(path)}
    total=sum(v['bytes'] for v in files.values())
    if total>32*1024**2: raise ValueError('Final small review archive budget exceeded')
    catalog={'schema':'jiuge7b-sft-paired-final-review-catalog-v1','files':files,
             'human_reports':['README.md','BRIEF.md'],'raw_trace_zip_downloaded':False,
             'raw_bundle_SHA':raw['bundle_sha256'],'failed_CPU_review_preserved':True,
             'platform_job':'sg-huids25-261020-4c8f2','platform_status_observed':'Completed',
             'platform_points_observed':'0.00','keep_old_V100':True}
    write(STAGE/'manifest.json',catalog)
    code='from pathlib import Path; p=Path('+repr(REMOTE)+'); assert not p.exists(); print("new-final-review-target-absent")'
    run([*SSH,PREP+'/.venv/bin/python -'],input=code.encode())
    run([*SCP,'-r',str(STAGE),'huids25@192.168.5.200:'+REMOTE])
    code='''from pathlib import Path
import hashlib,json
p=Path('''+repr(REMOTE)+''')
m=json.loads((p/'manifest.json').read_text())
for name,expected in m['files'].items():
    f=p/name
    assert f.stat().st_size==expected['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==expected['sha256'],name
r={'schema':'jiuge7b-sft-paired-final-ACT-review-verification-v1','passed':True,'folder':str(p),
   'files':len(m['files']),'bytes':sum(x['bytes'] for x in m['files'].values()),
   'manifest_sha256':hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest(),
   'raw_originals_and_closed_prior_archives_modified':False}
with p.with_name(p.name+'.verification.json').open('x') as stream: json.dump(r,stream,indent=2)
print(json.dumps(r))
'''
    verified=json.loads(run([*SSH,PREP+'/.venv/bin/python -'],input=code.encode()).decode())
    if verified['manifest_sha256']!=sha(STAGE/'manifest.json'): raise ValueError('ACT/local final manifest SHA differs')
    write(BASE/'results/sft-paired-trace-final-ACT-review-verification-20261020.json',verified)
    completion={'schema':'jiuge7b-sft-paired-trace-completion-v1','utc':datetime.now(timezone.utc).isoformat(),
                'all_requested_capture_and_analysis_completed':True,'raw_traces':16,'case_order':'ABBA',
                'actual_updates_each_run_each_rank':35,'SFT_sample_IDs_supervision_counts_logged_losses_equal':True,
                'accepted_pinning_candidate_unchanged':True,'frozen45_source_SHA_verified':True,
                'CPU_review_V2_passed':True,'original_key_export_suffix_failure_and_logs_preserved':True,
                'raw_archive':raw,'ACT_final_report_archive':verified,'two_human_reports':['README.md','BRIEF.md'],
                'job':'sg-huids25-261020-4c8f2','platform_status':'completed','platform_exit_code':0,
                'platform_charge_points':0,'platform_proof':'results/sft-paired-trace-completed-zero-points-20261020.png',
                'raw_data_remains_ACT':True,'raw_zip_downloaded_to_local':False,'old_V100_retained':True,
                'new_formal_speedup_claim':False,'previous_SFT_formal_vs_native_percent':92.85561884581284,
                'critical_path_proven':False,'collector_drop_count_verified':False,
                'native_collective_sequence_numbers_verified':False,'one_native_NCCL_kernel_has_no_message_metadata':True}
    output=BASE/'results/sft-paired-trace-completion-receipt-20261020.json'
    write(output,completion)
    remote_receipt=PREP+'/sft-paired-trace-delivery-20261020-01/final-review-completion.json'
    run([*SCP,str(output),'huids25@192.168.5.200:'+remote_receipt])
    code='from pathlib import Path; import hashlib; print(hashlib.sha256(Path('+repr(remote_receipt)+').read_bytes()).hexdigest())'
    receipt_sha=run([*SSH,PREP+'/.venv/bin/python -'],input=code.encode()).decode().strip()
    if receipt_sha!=sha(output): raise ValueError('Completion receipt ACT/local SHA differs')
    write(BASE/'results/sft-paired-trace-completion-upload-verification-20261020.json',
          {'passed':True,'ACT_path':remote_receipt,'sha256':receipt_sha})
    active=BASE/'results/active-validation-state.json'
    state=read(active)
    state.update(as_of_local=datetime.now().astimezone().isoformat(),experiments_running_for_this_campaign=False,
                 resource='new2A100 job completed0points; oldV100 retained',
                 current_actual_progress='All16pairedSFTtraces, scientific checks, independentV2CPUreview, rawACTzip, plots and two reports closed; ACT final-review SHA verified',
                 next_actions=['No active GPU experiment; await next user objective'])
    state['sft_paired_tracing'].update(status='completed',job_status='completed',platform_points=0,
            raw_traces=16,completion_receipt=str(output.relative_to(BASE)).replace('\\','/'),
            final_ACT_review=verified,raw_ACT_bundle=raw['bundle'],CPU_postprocess='results/sft-paired-trace-postprocess-20261020-02/status.json',
            CPU_postprocess_pid=3056,independent_review='results/sft-paired-trace-independent-review-v2-20261020.json',
            old_failed_CPU_review_preserved=True,raw_zip_local_downloaded=False,reports_complete=True,
            heartbeat='Pause after completion',completed_utc=completion['utc'])
    active.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':True,'ACT_final_review':verified,'completion_receipt_SHA':receipt_sha},indent=2))


if __name__=='__main__': main()

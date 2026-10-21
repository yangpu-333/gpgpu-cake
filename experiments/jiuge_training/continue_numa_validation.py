"""New complete affinity audit/resume before any fresh formal timing."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from continue_pinning_validation import remaining_seconds
from continue_pinning_formal import resources
from pinning_key_results import export

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')
CODE=('continue_numa_validation.py','run_numa_validation_case.py','verify_numa_semantics.py','verify_numa_archive.py',
      'cpu_numa.contract.json','numa_entry.py','numa_placement.py','numa_admission.py','run_numa_formal.py',
      'analyze_numa_formal.py','continue_numa_formal.py','analyze_pinning_formal.py','pinning_formal_entry.py',
      'pinning_entry.py','pinning_candidate.py','pinning_full_audit.py','pinning_resume_audit.py',
      'pinning_semantics_entry.py','pinning_resume_compat_entry_v2.py','checkpoint_load_compat.py',
      'checkpoint_hdo_load_compat.py','verify_pinning_archive.py','verify_pinning_semantics.py',
      'input_evidence.py','optimizer_evidence.py','jiuge7b_bridge.py','native_entry.py','run_native_case.py')


def main():
    if socket.gethostname()!='sg-huids25-261016-13441-default0-0':
        raise ValueError('Use the authorized zero-cost successor, with a complete audit/formal time reserve')
    if remaining_seconds()<20000:
        raise TimeoutError('Insufficient free time for full verification and subsequent benchmark')
    diagnostic=json.loads((PREP/'cpu-numa-mechanism-20261016-01/mechanism-summary.json').read_text())
    local=[row for row in diagnostic['cases'] if row['mode']=='local8']
    native=[row for row in diagnostic['cases'] if row['mode']=='unbound']
    if len(local)!=1 or len(native)!=2 or not all(local[0]['diagnostic_mean_seconds_per_step']<row['diagnostic_mean_seconds_per_step'] for row in native):
        raise ValueError('No consistent diagnostic candidate; inspect before committing a full campaign')
    root=PREP/'cpu-numa-validation-20261016-01'
    subprocess.run([sys.executable,PREP/'pinning_candidate.py','--output',root],check=True)
    source=root/'source'; source.mkdir()
    hashes={}
    for name in CODE:
        data=(PREP/name).read_bytes(); (source/name).write_bytes(data); hashes[name]=hashlib.sha256(data).hexdigest()
    state={'schema':'jiuge7b-cpu-numa-validation-controller-v1','status':'initializing',
           'started_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':hashes,'runs':[],
           'hostname':socket.gethostname(),'formal_performance_passed':False,'complete_raw_local_download':False,
           'diagnostic_summary_sha256':hashlib.sha256((PREP/'cpu-numa-mechanism-20261016-01/mechanism-summary.json').read_bytes()).hexdigest()}
    def save():
        temp=root/'controller.tmp'; temp.write_text(json.dumps(state,indent=2)+'\n'); temp.replace(root/'controller.json')
    def run(name,argv,reserve=3600):
        if remaining_seconds()<reserve:
            raise TimeoutError('Insufficient existing free allocation reserve')
        if any(hashlib.sha256((PREP/name).read_bytes()).hexdigest()!=digest for name,digest in hashes.items()):
            raise ValueError('Frozen validation source changed')
        state.update(status=name); save(); print(json.dumps({'event':'validation_stage_start','stage':name}),flush=True)
        with (root/f'{name}.log').open('x') as stream:
            subprocess.run([sys.executable,*map(str,argv)],cwd=PREP,stdout=stream,stderr=subprocess.STDOUT,
                           check=True,timeout=max(1,remaining_seconds()-1800))
        state['runs'].append(name); save(); print(json.dumps({'event':'validation_stage_complete','stage':name}),flush=True)
    save()
    try:
        state['resource_gate_before']=resources(); save()
        old=PREP/'pinning-validation-20261015-04'
        old_proof=json.loads((old/'primary-archive-verification.json').read_text())
        if not old_proof['passed'] or not old_proof['checkpoint_resume_tested']:
            raise ValueError('Complete previous native/pinned audit, raw and restore proof required')
        if any(row['max_abs_error']!=0 for row in old_proof['runs']):
            raise ValueError('Expected immutable zero-difference original/pinned reference')
        origins=[]
        for rank in (0,1):
            original=old/'native'/f'rank-{rank}'
            target=root/'native'/f'rank-{rank}'; target.mkdir(parents=True)
            manifest=json.loads((original/'all-state-manifest.json').read_text())
            for relative in sorted({chunk['file'] for row in manifest['tensors'] for chunk in row['chunks']}):
                source_path=(old/relative).resolve(strict=True)
                target_path=root/relative
                if not source_path.is_relative_to(old.resolve(strict=True)) or not target_path.resolve().is_relative_to(root.resolve()):
                    raise ValueError('Reference raw path escaped experiment archive')
                target_path.parent.mkdir(parents=True,exist_ok=True)
                if not target_path.exists():
                    os.link(source_path,target_path)
                if source_path.stat().st_ino!=target_path.stat().st_ino:
                    raise ValueError('Reference alias is not the same immutable original')
                origins.append({'source':str(source_path),'alias':str(target_path),'bytes':source_path.stat().st_size,
                                'same_inode':True,'body_copied':False})
            for name in ('all-state-manifest.json','completion.json'):
                shutil.copyfile(original/name,target/name)
        (root/'reference-origins.json').write_text(json.dumps({'method':'Hardlink same native reference bodies; new candidate compares every field in CPU float64',
             'prior_baseline_equivalence_proof_sha256':hashlib.sha256((old/'primary-archive-verification.json').read_bytes()).hexdigest(),
             'files':origins},indent=2)+'\n')
        resources()
        run('cpu-numa-local8-full-audit-01',[PREP/'run_numa_validation_case.py','--case','cpu-numa-local8-full-audit-01',
             '--steps','3','--seq-length','2048','--numa-mode','local8','--validation-stage','audit',
             '--pinning-campaign',root,'--semantics-root',root/'semantics-optimized'],reserve=10000)
        resources()
        resume_candidate=root/'resume-candidate'
        run('prepare-resume',[PREP/'pinning_candidate.py','--output',resume_candidate])
        checkpoint=PREP/'pinning-semantics-20261015-02/checkpoints/native-step2'
        if (checkpoint/'latest_checkpointed_iteration.txt').read_text().strip()!='2':
            raise ValueError('Same actual iteration2 native checkpoint required')
        run('cpu-numa-local8-full-resume-01',[PREP/'run_numa_validation_case.py','--case','cpu-numa-local8-full-resume-01',
             '--steps','3','--seq-length','2048','--numa-mode','local8','--validation-stage','resume',
             '--pinning-campaign',resume_candidate,'--semantics-root',root/'resume-optimized',
             '--load-dir',checkpoint,'--resume-audit-campaign',root],reserve=10000)
        resources()
        for case in ('cpu-numa-local8-full-audit-01','cpu-numa-local8-full-resume-01'):
            for rank in (0,1):
                optimizers=json.loads((PREP/'runs'/case/f'rank-{rank}/optimizer-state.json').read_text())['optimizers']
                if (len(optimizers)!=1 or optimizers[0]['adam_state_entries']!=199 or
                    optimizers[0]['min_step']!=3 or optimizers[0]['max_step']!=3 or
                    optimizers[0]['elements_with_both_adam_moments']!=3834518016):
                    raise ValueError('Actual updated/restored AdamW step or complete coverage differs')
        run('verify-semantics',[PREP/'verify_numa_semantics.py','--root',root,'--output',root/'semantics-verification.json'])
        run('verify-full-primary-raw',[PREP/'verify_numa_archive.py','--root',root,'--output',root/'full-archive-verification.json'],reserve=10000)
        state.update(status='ready_for_local_key_verification',finished_utc=datetime.now(timezone.utc).isoformat(),resource_gate_after=resources()); save()
        # Restrict export below to summary metadata/source; raw tensors/checkpoint are never eligible.
        export(PREP,[root,PREP/'runs/cpu-numa-local8-full-audit-01',PREP/'runs/cpu-numa-local8-full-resume-01'],
               PREP/'cpu-numa-validation-key-results-20261016.zip')
        print(json.dumps({'status':state['status'],'key_bundle':str(PREP/'cpu-numa-validation-key-results-20261016.zip')}),flush=True)
    except BaseException as exc:
        state.update(status='stopped',error_type=type(exc).__name__,error=str(exc)); save(); raise


if __name__=='__main__':
    main()

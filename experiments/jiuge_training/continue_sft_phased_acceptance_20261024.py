"""Fresh three-arm SFT states, unmodified native checkpoint, then fixed formal gates."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

from sft_phased_acceptance_20261024 import PREP, ROOT, APPROVAL, read, sha, verify_sources


def main():
    approval = read(APPROVAL)
    if (socket.gethostname() != approval['hostname'] or os.getuid()!=1001
            or approval['max_charge_points'] != 0):
        raise ValueError('Exact authorized container and non-root UID required')
    ROOT.mkdir(exist_ok=False)
    lock = (PREP/'sft-phased-acceptance-20261024.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources = verify_sources()
    if sources != approval['source_files_sha256']:
        raise ValueError('Exact approved source set required')
    snapshot = ROOT/'source'
    snapshot.mkdir()
    for name in sources:
        shutil.copyfile(PREP/name, snapshot/name)
    numeric, semantics = ROOT/'numeric', ROOT/'semantics'
    cases = []
    state = {'schema':'jiuge7b-incremental-SFT-acceptance-controller-v1', 'status':'initializing',
             'started_utc':datetime.now(timezone.utc).isoformat(), 'sources':sources,
             'hostname':socket.gethostname(), 'approval':approval, 'commands':[], 'completed_cases':[],
             'formal_performance_started':False, 'formal_performance_passed':False,
             'raw_originals':'ACT Ceph only','local_full_raw_copy':False}

    def update(status, **extra):
        state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        temp = ROOT/'controller.json.tmp'
        temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(ROOT/'controller.json')
        try:
            print(json.dumps({'status':status,'utc':state['updated_utc']}),flush=True)
        except BrokenPipeError:
            pass

    def remaining():
        return approval['stop_before_epoch']-time.time()

    def resources():
        raw = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True)
        rows = [[x.strip() for x in row.split(',')] for row in raw.strip().splitlines()]
        if (len(rows)!=2 or [r[0] for r in rows]!=approval['ordered_gpu_uuids']
                or any('A100' not in r[1] or int(r[2])>256 or int(r[3]) for r in rows) or apps.strip()):
            raise ValueError('Actual fixed two GPUs must be idle; never stop another process')
        return {'gpu':raw,'compute_apps':apps}

    def command(name, argv, reserve=900):
        if remaining()<reserve:
            raise TimeoutError('Insufficient remaining authorized window; preserve completed stage')
        if verify_sources()!=sources:
            raise ValueError('Frozen sources differ')
        update(name)
        with (ROOT/(name+'.stdout.log')).open('x') as out,(ROOT/(name+'.stderr.log')).open('x') as err:
            child = subprocess.Popen([sys.executable,*map(str,argv)],cwd=PREP,stdout=out,stderr=err,start_new_session=True)
            update(name, active_child_pid=child.pid)
            try:
                code = child.wait(timeout=max(1,remaining()-180))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait(timeout=10)
                raise
        state.pop('active_child_pid',None)
        state['commands'].append({'name':name,'argv':[str(x) for x in argv],'exit_code':code})
        update(name)
        if code:
            raise RuntimeError(name+' failed; no subsequent GPU stage admitted')

    def prepare(name):
        path = ROOT/'candidates'/name
        command('prepare-'+name,[PREP/'pinning_candidate.py','--output',path])
        return path

    def case(name, stage, arm, campaign, sem=None, extra=(), original=False, seed=63100):
        resources()
        runner = 'run_sft_pinning_case.py' if original else 'run_sft_phased_acceptance_20261024.py'
        argv = [PREP/runner,'--case',name,'--mode',stage,'--pinning-arm',arm,
                '--pinning-campaign',campaign,'--seq-length','2048','--steps','120' if stage=='formal' else '3',
                '--seed',str(seed)]
        if sem:
            argv += ['--semantics-root',sem]
        if not original:
            argv += ['--phased-plan',approval['plan_file']]
        command(name,[*argv,*extra],reserve=1200)
        resources()
        folder = PREP/'runs'/name
        record = read(folder/'case.json')
        if record['exit_code'] or record['resource_gate_before'].strip() or record['resource_gate_after'].strip():
            raise ValueError('Case exit/occupancy failed')
        for rank in (0,1):
            entry = read(folder/f'rank-{rank}/entry.json')
            if entry['status']!='completed' or entry['native_files_checked']!=600:
                raise ValueError('Complete frozen native execution required')
            if not entry['weight_loading']['all_loaded_shard_bytes_verified']:
                raise ValueError('Loaded official weight byte verification required')
            objective = read(folder/f'sft-objective-rank-{rank}.json')
            expected = {'audit':3,'checkpoint':2,'resume':1,'formal':120}[stage]
            if not objective['run_returned_successfully'] or len(objective['supervision_rows'])!=(expected if rank==0 else 0):
                raise ValueError('TP reader/broadcast SFT sample role differs')
            if stage!='checkpoint':
                leaves = entry['actual_optimizer_state']['optimizers']
                steps = 120 if stage=='formal' else 3
                if sum(x['adam_state_entries'] for x in leaves)!=199 or any(x['min_step']!=steps or x['max_step']!=steps for x in leaves if x['adam_state_entries']):
                    raise ValueError('199 complete AdamW groups at required update count')
            if not original:
                policy = read(folder/f'phased-policy-rank-{rank}.json')
                if policy['steps']!=expected or not policy.get('final_pages'):
                    raise ValueError('Actual policy/update/page proof incomplete')
                if arm=='optimized' and len(policy['moment_events'])!=398:
                    raise ValueError('All398 current candidate moment objects required')
                if arm=='optimized':
                    query = policy['final_pages']['page_queries']
                    node = policy['binding']['selected_cpu_nodes'][0]
                    if query['return_code']!=0 or query['sample_nodes'] is None:
                        raise ValueError('Actual candidate page query must succeed')
                    for field in ('exp_avg','exp_avg_sq'):
                        rows = [x for x in query['tensors'] if x['field'].endswith('/'+field)]
                        if len(rows)!=199 or sum(x['numel'] for x in rows)!=3834518016:
                            raise ValueError('Full candidate moment object sampling coverage required')
                        if any(n!=node for x in rows for n in query['sample_nodes'][x['sample_start']:x['sample_start']+x['sample_count']]):
                            raise ValueError('Sampled moment pages must match actual local node')
        cases.append(folder)
        state['completed_cases'].append(name)
        update('completed-'+name)
        return folder

    try:
        update('preflight', initial_resource_gate=resources())
        if int(Path('/sys/fs/cgroup/memory.max').read_text()) < 300*1024**3:
            raise ValueError('At least300GiB required without weakening memory guards')
        if shutil.disk_usage(PREP).free < 1024**4:
            raise ValueError('At least1TiB primary storage reserve before full raw audit')
        import torch
        if torch.__version__!='2.6.0+cu118' or torch.cuda.is_initialized():
            raise ValueError('Frozen non-CUDA parent runtime required')
        command('prepare-numeric',[PREP/'pinning_candidate.py','--output',numeric])
        semantics.mkdir()
        for arm in ('native','control','optimized'):
            sem = semantics/('semantics-'+arm)
            folder = case('phased-audit-'+arm+'-20261024-01','audit',arm,numeric,sem)
            for rank in (0,1):
                for path in (sem/f'rank-{rank}').glob('actual-input*'):
                    os.link(path,folder/f'rank-{rank}'/path.name)
            command('verify-source-'+arm,[PREP/'verify_sft_admission.py','--case-root',folder])
        checkpoint = ROOT/'checkpoints/native-step2'
        case('phased-native-checkpoint-20261024-01','checkpoint','native',prepare('checkpoint'),
             semantics/'checkpoint-native',extra=('--save-dir',checkpoint),original=True)
        if (checkpoint/'latest_checkpointed_iteration.txt').read_text().strip()!='2':
            raise ValueError('Unmodified native SFT iteration2 checkpoint required')
        for arm in ('native','optimized'):
            extra = ['--load-dir',checkpoint]
            if arm=='optimized':
                extra += ['--resume-audit-campaign',numeric]
            folder = case('phased-resume-'+arm+'-20261024-01','resume',arm,prepare('resume-'+arm),
                          semantics/('resume-'+arm),extra=extra,original=arm=='native')
            for rank in (0,1):
                proof = read(folder/f'rank-{rank}/checkpoint-compat.json')
                if not proof['run_returned_successfully'] or proof['common_load_calls']<1 or proof['sharded_load_calls']<1:
                    raise ValueError('Original checkpoint compatibility path not completed')
        semantic_receipt = semantics/'semantics-verification.json'
        command('verify-semantics',[PREP/'verify_sft_pinning_semantics.py','--root',semantics,
                                    '--resume-campaign',numeric,'--output',semantic_receipt])
        checkpoint_rows = [{'file':str(p.relative_to(checkpoint)),'bytes':p.stat().st_size,'sha256':sha(p)}
                           for p in sorted(checkpoint.rglob('*')) if p.is_file()]
        (ROOT/'checkpoint-sha256.json').write_text(json.dumps({'checkpoint':str(checkpoint),'files':checkpoint_rows},indent=2)+'\n')
        primary = numeric/'primary-archive-verification.json'
        command('verify-all-originals',[PREP/'verify_pinning_archive.py','--root',numeric,'--include-resume','--output',primary])
        from pinning_key_results import export
        bundle = PREP/'sft-phased-acceptance-correctness-20261024-01.zip'
        export(PREP,[ROOT,*cases],bundle,64*1024**2)
        update('ready-for-local-key-verification', key_results_bundle=str(bundle),
               full_correctness_resume_and_primary_archive_passed=True)
        local = ROOT/'local-key-verification.json'
        while not local.exists():
            if remaining()<10800:
                raise TimeoutError('Correctness preserved; require3h formal reserve and closed key gate')
            time.sleep(5)
        from admit_pinning_formal import admit
        admission = admit(primary,semantic_receipt,bundle,local)
        admission.update(objective='answer_only_SFT',sft_data_receipt_sha256=sha(PREP/'datasets/alpaca-sft-answer-v1-receipt.json'),
                         incremental_phased_candidate_accepted=True,incremental_source_sha256=sources,
                         incremental_baseline='accepted pinned, unbound')
        admitted = ROOT/'formal-admission.json'
        admitted.write_text(json.dumps(admission,indent=2)+'\n')
        if remaining()<10800:
            raise TimeoutError('Formal requires original3h reserve; no partial promotion')
        update('starting-formal',formal_performance_started=True)
        from analyze_sft_phased_formal_20261024 import analyze
        rounds = []
        for round_name in ('primary','confirmation'):
            label = 'phased20261024'+round_name
            for seed,order in zip((63100,63101,63102),
                    (('native','control','optimized'),('optimized','control','native'),('control','native','optimized'))):
                for arm in order:
                    name = f'pinning-formal-{label}-{seed}-{arm}'
                    case(name,'formal',arm,prepare(name),extra=('--admission',admitted),seed=seed)
            result = analyze(label)
            with (ROOT/('formal-'+round_name+'.json')).open('x') as stream:
                json.dump(result,stream,indent=2)
            rounds.append(result)
            update('completed-formal-'+round_name)
            if round_name=='primary' and not result['repeat_required']:
                break
        update('completed',formal_performance_passed=all(x['round_promotion_passed'] for x in rounds),
               repeat_performed=len(rounds)==2,resource_gate_final=resources())
        final = PREP/'sft-phased-acceptance-final-20261024-01.zip'
        export(PREP,[ROOT,*cases],final,64*1024**2)
        (ROOT/'final-key-bundle.json').write_text(json.dumps({'path':str(final),'bytes':final.stat().st_size,'sha256':sha(final)},indent=2)+'\n')
    except BaseException as error:
        update('stopped',error_type=type(error).__name__,error=str(error))
        raise


if __name__ == '__main__':
    main()

"""Fresh SFT numerical, native checkpoint and unchanged-standard formal gates."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT = PREP/'sft-pinning-validation-20261020-01'
NUMERIC = ROOT/'numeric'
SEMANTICS = ROOT/'semantics'
CODE = ('continue_sft_pinning_validation.py','run_sft_pinning_case.py','sft_pinning_entry.py',
        'verify_sft_pinning_semantics.py','bootstrap_sft_pinning_20261020.sh','sft_pinning.contract.json',
        'run_native_case.py','native_entry.py','jiuge7b_bridge.py','sft_dataset_adapter.py',
        'input_evidence.py','optimizer_evidence.py','trace_hooks.py','verify_sft_admission.py',
        'pinning_entry.py','pinning_candidate.py','pinning_full_audit.py','pinning_resume_audit.py',
        'pinning_semantics_entry.py','pinning_resume_compat_entry_v2.py','checkpoint_load_compat.py',
        'checkpoint_hdo_load_compat.py','pinning_formal_entry.py','run_pinning_formal.py',
        'verify_pinning_semantics.py','verify_pinning_archive.py','pinning_key_results.py',
        'admit_pinning_formal.py','analyze_pinning_formal.py')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8*1024**2):
            h.update(block)
    return h.hexdigest()


def main():
    approval = read(PREP/'sft-pinning-launch-approved-20261020.json')
    if socket.gethostname() != approval['hostname'] or os.getuid() != 1001:
        raise ValueError('Exact approved new allocation and numeric UID required')
    ROOT.mkdir(exist_ok=False)
    snapshot = ROOT/'source'
    snapshot.mkdir()
    hashes = {name:sha(PREP/name) for name in CODE}
    if hashes != approval['source_files_sha256']:
        raise ValueError('Approved sources changed')
    for name in CODE:
        shutil.copyfile(PREP/name,snapshot/name)
    state = {'schema':'jiuge7b-sft-pinning-validation-controller-v1','status':'initializing',
             'objective':'answer_only_SFT','started_utc':datetime.now(timezone.utc).isoformat(),
             'source_sha256':hashes,'commands':[],'completed_cases':[],
             'formal_performance_started':False,'formal_performance_passed':False,
             'raw_storage':'ACT persistent shared directory','local_large_raw_copy_enabled':False}
    cases = []
    def update(status):
        state.update(status=status,updated_utc=datetime.now(timezone.utc).isoformat())
        temp = ROOT/'controller.json.tmp'
        temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(ROOT/'controller.json')
        try:
            print(json.dumps({'status':status,'utc':state['updated_utc']}),flush=True)
        except BrokenPipeError:
            pass  # A lost progress viewer cannot interrupt numerical work.
    def remaining():
        return approval['stop_before_epoch']-time.time()
    def frozen(reserve=600):
        if remaining() < reserve:
            raise TimeoutError('Insufficient authorized free time for next stage')
        if any(sha(PREP/name) != value for name,value in hashes.items()):
            raise ValueError('Source changed during SFT campaign')
    expected_uuids = None
    def resources():
        nonlocal expected_uuids
        raw = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
        rows = [[x.strip() for x in line.split(',')] for line in raw.strip().splitlines()]
        uuids = [row[0] for row in rows]
        if len(rows) != 2 or any('A100' not in row[1] or int(row[2])>256 or int(row[3]) for row in rows) or apps.strip():
            raise ValueError('Assigned two A100s must be idle with no other compute process')
        if expected_uuids is not None and uuids != expected_uuids:
            raise ValueError('Physical GPU allocation changed')
        expected_uuids = uuids
        return {'gpu':raw,'compute_apps':apps,'ordered_gpu_uuids':uuids,
                'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text(),
                'memory_max':Path('/sys/fs/cgroup/memory.max').read_text()}
    def command(name,argv,reserve=600):
        frozen(reserve)
        update(name)
        with (ROOT/(name+'.stdout.log')).open('x') as out,(ROOT/(name+'.stderr.log')).open('x') as err:
            child = subprocess.Popen([sys.executable,*map(str,argv)],cwd=PREP,stdout=out,stderr=err,start_new_session=True)
            state['active_child_pid'] = child.pid
            update(name)
            try:
                code = child.wait(timeout=max(1,remaining()-300))
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
        if code:
            raise RuntimeError(name+' failed; preserve actual originals/logs')
    def candidate(name):
        path = ROOT/'candidates'/name
        command('prepare-'+name,[PREP/'pinning_candidate.py','--output',path])
        return path
    def case(name,mode,arm,campaign,semantics=None,extra=(),steps=3):
        resources()
        argv = [PREP/'run_sft_pinning_case.py','--case',name,'--mode',mode,'--pinning-arm',arm,
                '--pinning-campaign',campaign,'--seq-length','2048','--steps',str(steps)]
        if semantics:
            argv += ['--semantics-root',semantics]
        command(name,[*argv,*extra],reserve=1200 if mode != 'formal' else 1500)
        resources()
        folder = PREP/'runs'/name
        record = read(folder/'case.json')
        if record['exit_code'] or record['resource_gate_before'].strip() or record['resource_gate_after'].strip():
            raise ValueError('Case exit/occupancy gate failed')
        for rank in (0,1):
            entry = read(folder/f'rank-{rank}/entry.json')
            if entry['status'] != 'completed' or entry['native_files_checked'] != 600:
                raise ValueError('Frozen native entry failed')
            if mode != 'checkpoint':
                leaves = entry['actual_optimizer_state']['optimizers']
                if (sum(x['adam_state_entries'] for x in leaves)!=199 or
                    any(x['min_step']!=steps or x['max_step']!=steps for x in leaves if x['adam_state_entries'])):
                    raise ValueError('Actual complete optimizer update coverage differs')
            objective = read(folder/f'sft-objective-rank-{rank}.json')
            if not objective['run_returned_successfully'] or objective['objective'] != 'answer_and_EOD_only_full_parameter_SFT':
                raise ValueError('Actual SFT objective not installed')
            if rank == 0:
                expected_count = {'audit':3,'checkpoint':2,'resume':1,'formal':120}[mode]
                rows = objective['supervision_rows']
                if len(rows) != expected_count or any(not 0<x['supervised_answer_and_EOD_targets']<=2048 for x in rows):
                    raise ValueError('Actual answer supervision coverage differs')
        cases.append(folder)
        state['completed_cases'].append(name)
        update('completed-'+name)
        return folder
    try:
        initial = resources()
        if int(initial['memory_max'].strip()) < 300*1024**3:
            raise ValueError('Full native checkpoint restore requires at least authorized300GiB')
        cpu_quota,cpu_period = map(int,initial['cpu_max'].split())
        if not 16 <= cpu_quota/cpu_period < 17:
            raise ValueError('Fixed16CPU quota required')
        import torch
        if sys.version_info[:2] != (3,12) or torch.__version__ != '2.6.0+cu118' or torch.cuda.is_initialized():
            raise ValueError('Frozen CPU-side runtime differs')
        initial.update(hostname=socket.gethostname(),runtime=sys.version,torch=torch.__version__,
                       topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True))
        (ROOT/'allocation.json').write_text(json.dumps(initial,indent=2)+'\n')
        command('prepare-numeric',[PREP/'pinning_candidate.py','--output',NUMERIC])
        SEMANTICS.mkdir()
        for arm in ('native','control','optimized'):
            sem = SEMANTICS/('semantics-'+arm)
            folder = case('sft-pinning-audit-'+arm+'-20261020-01','audit',arm,NUMERIC,sem)
            # Semantic observers own these untimed actual inputs. Reuse bytes,
            # not values from a CPT run, for independent source reconstruction.
            for rank in (0,1):
                for path in (sem/f'rank-{rank}').glob('actual-input*'):
                    os.link(path,folder/f'rank-{rank}'/path.name)
            command('verify-SFT-source-'+arm,[PREP/'verify_sft_admission.py','--case-root',folder])
        checkpoint = ROOT/'checkpoints/native-step2'
        case('sft-pinning-native-checkpoint-20261020-01','checkpoint','native',candidate('checkpoint-native'),
             SEMANTICS/'checkpoint-native',extra=('--save-dir',checkpoint))
        if (checkpoint/'latest_checkpointed_iteration.txt').read_text().strip() != '2':
            raise ValueError('The immutable SFT checkpoint must be native iteration2')
        for arm in ('native','optimized'):
            extra = ['--load-dir',checkpoint]
            if arm == 'optimized':
                extra += ['--resume-audit-campaign',NUMERIC]
            folder = case('sft-pinning-'+arm+'-resume-20261020-01','resume',arm,candidate('resume-'+arm),
                          SEMANTICS/('resume-'+arm),extra=extra)
            for rank in (0,1):
                proof = read(folder/f'rank-{rank}/checkpoint-compat.json')
                if (not proof['run_returned_successfully'] or proof['common_load_calls']<1 or
                    proof['sharded_load_calls']<1 or len(proof['hdo_placeholder_initializations'])!=1):
                    raise ValueError('Actual native checkpoint load compatibility gate failed')
        semantic_receipt = SEMANTICS/'semantics-verification.json'
        command('verify-semantics',[PREP/'verify_sft_pinning_semantics.py','--root',SEMANTICS,
                                    '--resume-campaign',NUMERIC,'--output',semantic_receipt])
        update('hash-native-SFT-checkpoint')
        checkpoint_files = [{'file':str(p.relative_to(checkpoint)),'bytes':p.stat().st_size,'sha256':sha(p)}
                            for p in sorted(checkpoint.rglob('*')) if p.is_file()]
        (ROOT/'checkpoint-sha256.json').write_text(json.dumps({'checkpoint':str(checkpoint),'files':checkpoint_files},indent=2)+'\n')
        primary = NUMERIC/'primary-archive-verification.json'
        command('verify-all-original-values',[PREP/'verify_pinning_archive.py','--root',NUMERIC,
                                              '--include-resume','--output',primary],reserve=1800)
        from pinning_key_results import export
        bundle = PREP/'sft-pinning-correctness-key-results-20261020.zip'
        export(PREP,[ROOT,*cases],bundle,64*1024**2)
        state.update(key_results_bundle=str(bundle),full_correctness_resume_and_primary_archive_passed=True)
        update('ready-for-local-key-verification')
        local_receipt = ROOT/'local-key-verification.json'
        while not local_receipt.exists():
            if remaining()<10800:
                raise TimeoutError('Local key gate not closed with three-hour formal reserve')
            time.sleep(5)
        from admit_pinning_formal import admit
        admission = admit(primary,semantic_receipt,bundle,local_receipt)
        admission.update(objective='answer_only_SFT',
                         sft_data_receipt_sha256=sha(PREP/'datasets/alpaca-sft-answer-v1-receipt.json'))
        admission_path = ROOT/'formal-admission.json'
        admission_path.write_text(json.dumps(admission,indent=2)+'\n')
        frozen(10800)
        state['formal_performance_started'] = True
        from analyze_pinning_formal import analyze
        rounds = []
        for round_name in ('primary','confirmation'):
            label = 'sft20261020'+round_name
            for seed,order in zip((63100,63101,63102),
                                  (('native','control','optimized'),('optimized','control','native'),('control','native','optimized'))):
                for arm in order:
                    name = f'pinning-formal-{label}-{seed}-{arm}'
                    case(name,'formal',arm,candidate(name),extra=('--admission',admission_path,'--seed',str(seed)),steps=120)
            result = analyze(PREP,label)
            (ROOT/('formal-'+round_name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
            rounds.append(result)
            update('completed-formal-'+round_name)
            if round_name == 'primary' and not result['repeat_required']:
                break
        state.update(formal_performance_passed=all(row['round_promotion_passed'] for row in rounds),
                     repeat_performed=len(rounds)==2,formal_rounds=[{k:v for k,v in row.items() if k!='cases'} for row in rounds],
                     finished_utc=datetime.now(timezone.utc).isoformat())
        state['resource_gate_final'] = resources()
        update('completed')
        final = PREP/'sft-pinning-complete-key-results-20261020.zip'
        export(PREP,[ROOT,*cases],final,64*1024**2)
        (ROOT/'final-key-bundle.json').write_text(json.dumps({'path':str(final),'sha256':sha(final),'bytes':final.stat().st_size},indent=2)+'\n')
    except BaseException as error:
        state.update(error_type=type(error).__name__,error=str(error))
        update('stopped')
        raise


if __name__ == '__main__':
    main()

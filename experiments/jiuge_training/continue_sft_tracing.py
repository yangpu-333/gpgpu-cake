"""Native SFT admission/capture and same-allocation CPT tracing, no promotion run."""
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
ROOT = PREP/'sft-tracing-20261017-01'
NAMES = ('sft-native-admission-20261017-01', 'sft-native-profile-20261017-01',
         'cpt-native-admission-20261017-01', 'cpt-native-profile-20261017-01')
CODE = ('continue_sft_tracing.py','preflight_sft_allocation.py','prepare_alpaca_sft.py',
        'sft_dataset_adapter.py','sft_entry.py','run_sft_case.py','verify_sft_admission.py',
        'verify_trace_case_completion.py',
        'verify_cpt_sft_pair.py','sft_tracing.contract.json',
        'summarize_cpt_sft_tracing.py',
        'probe_distributed_profiler.py',
        'bootstrap_sft_ubuntu.sh',
        'native_entry.py','jiuge7b_bridge.py','run_native_case.py','trace_hooks.py',
        'optimizer_evidence.py','input_evidence.py','monitor_case_resources.py',
        'archive_case.py','review_native_trace.py','analyze_trace.py','inspect_trace_communication.py')


def main():
    if socket.gethostname() != 'sg-huids25-261017-e2bcb-default0-0':
        raise ValueError('Authorized new SFT allocation only')
    ROOT.mkdir(exist_ok=False)
    source = ROOT/'source'
    source.mkdir()
    hashes = {}
    for name in CODE:
        hashes[name] = hashlib.sha256((PREP/name).read_bytes()).hexdigest()
        shutil.copyfile(PREP/name,source/name)
    (ROOT/'source-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    status = {'schema':'jiuge7b-cpt-sft-tracing-controller-v1',
              'started_utc':datetime.now(timezone.utc).isoformat(), 'status':'initializing',
              'formal_performance_campaign':False, 'new_optimizer_candidate':False,
              'raw_storage':'ACTpersistentCeph', 'completed_cases':[], 'commands':[]}
    def update(state):
        status.update(status=state,updated_utc=datetime.now(timezone.utc).isoformat())
        temporary = ROOT/'controller.json.tmp'
        temporary.write_text(json.dumps(status,indent=2)+'\n')
        temporary.replace(ROOT/'controller.json')
        print(json.dumps({'state':state,'utc':status['updated_utc']}),flush=True)
    def run(name,argv,timeout=5400):
        update(name)
        with (ROOT/(name+'.stdout.log')).open('x') as out, (ROOT/(name+'.stderr.log')).open('x') as err:
            result = subprocess.run([sys.executable,*argv],stdout=out,stderr=err,cwd=PREP,timeout=timeout)
        status['commands'].append({'name':name,'argv':argv,'exit_code':result.returncode})
        if result.returncode:
            raise RuntimeError(name+' failed; see preserved ACT logs')
    def lifetime_gate():
        elapsed = int(subprocess.check_output(['ps','-o','etimes=','-p','1'],text=True).strip())
        if elapsed > 18000:
            raise RuntimeError('Less than one hour remains in authorized six-hour startup cap')
        for name, expected in hashes.items():
            if hashlib.sha256((PREP/name).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Source changed during campaign: '+name)
    def case(name,sft,profile):
        lifetime_gate()
        # Each case runner separately locks and checks assigned GPUs for other compute.
        update('running_'+name)
        argv = [sys.executable,str(PREP/('run_sft_case.py' if sft else 'run_native_case.py')),
                '--case',name,'--seq-length','2048','--steps','35' if profile else '3',
                '--capture-windows' if profile else '--capture-inputs']
        case_root = PREP/'runs'/name
        monitor = None
        with (ROOT/(name+'.launcher.stdout.log')).open('x') as out, (ROOT/(name+'.launcher.stderr.log')).open('x') as err:
            process = subprocess.Popen(argv,stdout=out,stderr=err,cwd=PREP,start_new_session=True)
            deadline = time.monotonic()+30
            while not case_root.is_dir() and process.poll() is None and time.monotonic()<deadline:
                time.sleep(0.2)
            if case_root.is_dir():
                with (ROOT/(name+'.monitor.log')).open('x') as log:
                    monitor = subprocess.Popen([sys.executable,str(PREP/'monitor_case_resources.py'),
                        '--case-root',str(case_root),'--seconds','2','--timeout','5400'],stdout=log,stderr=log)
            try:
                code = process.wait(timeout=5500)
            except subprocess.TimeoutExpired:
                # This new session contains only the case launched immediately above.
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait(timeout=10)
                if monitor and monitor.poll() is None:
                    monitor.terminate()
                    monitor.wait(timeout=10)
                raise
        if code:
            raise RuntimeError(name+' native training failed with '+str(code))
        if monitor:
            if monitor.wait(timeout=15):
                raise RuntimeError('Memory observer failed')
        record = json.loads((case_root/'case.json').read_text())
        if record['exit_code'] != 0:
            raise RuntimeError('Native case did not complete')
        if sft and not profile:
            run('verify-'+name,[str(PREP/'verify_sft_admission.py'),'--case-root',str(case_root)])
        run('complete-'+name,[str(PREP/'verify_trace_case_completion.py'),'--case-root',str(case_root),
                              '--steps','35' if profile else '3',
                              *(['--sft'] if sft else []), *(['--profile'] if profile else [])])
        if profile:
            run('analyze-'+name,[str(PREP/'review_native_trace.py'),'--case-root',str(case_root),
                                  '--output',str(case_root/'trace-analysis.json')])
            run('communication-'+name,[str(PREP/'inspect_trace_communication.py'),'--case-root',str(case_root),
                                       '--output',str(case_root/'communication-analysis.json')])
        run('archive-'+name,[str(PREP/'archive_case.py'),'--case-root',str(case_root),'--create',
                             *(['--snapshot-analysis'] if profile else [])])
        run('verify-archive-'+name,[str(PREP/'archive_case.py'),'--case-root',str(case_root),'--verify',
                                  '--receipt',str(ROOT/(name+'-archive-verification.json'))])
        status['completed_cases'].append(name)
        update('completed_'+name)
    try:
        run('preflight',[str(PREP/'preflight_sft_allocation.py'),'--output',str(ROOT/'allocation.json')])
        run('two-rank-profiler-capability',['-m','torch.distributed.run','--standalone','--nproc_per_node=2',
            str(PREP/'probe_distributed_profiler.py'),'--output',str(PREP/'sft-profiler-probe-20261017-01')])
        case(NAMES[0],True,False)
        case(NAMES[1],True,True)
        case(NAMES[2],False,False)
        run('compare-real-CPT-SFT-inputs',[str(PREP/'verify_cpt_sft_pair.py')])
        case(NAMES[3],False,True)
        run('summarize-CPT-SFT-traces',[str(PREP/'summarize_cpt_sft_tracing.py')])
        status.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                      archive_primary_verification_passed=True,
                      stable_trace_windows=8,actual_full_parameter_SFT_verified=True,
                      raw_downloaded_to_local=False,
                      comparison_scope='same allocation/settings, distinct full-text versus answer-only objectives; profiling, no formal speedup claim')
        update('completed')
    except BaseException as error:
        status.update(error_type=type(error).__name__,error=str(error))
        update('stopped')
        raise


if __name__ == '__main__':
    main()

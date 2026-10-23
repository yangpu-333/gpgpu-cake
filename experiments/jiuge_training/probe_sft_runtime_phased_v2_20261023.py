"""Unchanged8GiB CPU admission plus exact threaded phase-switch state/resume."""
from pathlib import Path
import json,os,socket,subprocess,sys
PREP=Path('/home/huids25/kda-jiuge-prep-20261014')
if __name__=='__main__':
    text=(PREP/'probe_sft_moment_direct_zero_runtime_20261021.py').read_text()
    before='sft-moment-direct-zero-approved-20261021.json'
    assert text.count(before)==1
    text=text.replace(before,'sft-phased-moments-approved-20261023-07.json')
    scope={'__name__':'CPU_admission','__file__':__file__}
    exec(compile(text,'<unchanged-live-CPU-admission>','exec'),scope)
    scope['main']()
    root=PREP/('sft-moment-direct-zero-runtime-'+socket.gethostname())
    results=[]
    for mode in ('0','1'):
        env=os.environ.copy();env.update(OMP_NUM_THREADS='8',JIUGE_FIXTURE_FUSED=mode,JIUGE_PHASE_FIXTURE_ROOT=str(root/('phased-wrapped-fused-'+mode)),JIUGE_PHASE_FIXTURE_HOST=socket.gethostname())
        with (root/('phased-wrapped-fused-'+mode+'.log')).open('x') as stream:
            subprocess.run([sys.executable,str(PREP/'check_sft_phased_wrapped_initializer_20261023.py')],check=True,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=260)
        result=json.loads((root/('phased-wrapped-fused-'+mode)/'summary.json').read_text())
        assert result['passed'] and not result['GPU_used']
        assert all(n['actual_thread_phase_switch_verified'] and n['native_disabled_initializer_exercised'] and n['fused']==(mode=='1') for n in result['nodes'])
        results.append(result)
    with (root/'phased-admission.json').open('x') as stream:json.dump({'passed':True,'GPU_used':False,'wrapped_fused_and_nonfused_fixtures':results},stream,indent=2)
    print(json.dumps({'passed':True,'actual_thread_phase_switch_verified':True,'wrapped_fused_and_nonfused_passed':True,'GPU_used':False}),flush=True)

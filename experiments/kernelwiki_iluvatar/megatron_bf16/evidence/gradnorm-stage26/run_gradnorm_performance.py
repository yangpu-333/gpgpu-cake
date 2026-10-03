"""Three-arm native/CE0008/CE0008+gradnorm formal main-scope experiment."""
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from datetime import datetime,timezone

BASE=Path('/tmp/kda-gradnorm-stage26')
BF=Path('/tmp/kda-bf16-continuation/megatron_bf16')
NATIVE=BF.parent/'megatron-native-5be9626'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    audit=json.loads((BASE/'comparison-03.json').read_text())
    probe=json.loads((BASE/'probe-001.json').read_text())
    assert audit['passed'] and probe['status']=='passed'
    assert audit['gradnorm_candidate_sha256']==probe['candidate_sha256']==sha(BASE/'candidate.py')
    contract=BF/'contracts/0008-main.json'; c=json.loads(contract.read_text()); parent=BF/'candidates/0008/candidate.py'
    spec=importlib.util.spec_from_file_location('sealed_bf16_performance',BF/'summarize_performance.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report={'scope':'Formal original BF16 main model. Three seeds; native/CE0008/CE0008+gradnorm fresh process arms. Separate three-step tensor audit; no audit/profile/checkpoint/eval during timing.',
            'runner_sha256':sha(Path(__file__)),'candidate_sha256':sha(BASE/'candidate.py'),
            'audit_sha256':sha(BASE/'comparison-03.json'),'pairs':[]}
    env=os.environ.copy()
    for name in list(env):
        if 'API_KEY' in name or 'AUTH_TOKEN' in name:env.pop(name)
    env.update(CUDA_VISIBLE_DEVICES='0',MASTER_ADDR='127.0.0.1',MASTER_PORT='29931',RANK='0',WORLD_SIZE='1',LOCAL_RANK='0')
    orders=(('native','control','optimized'),('optimized','control','native'),('control','native','optimized'))
    for seed,order in zip((26102,26103,26104),orders):
        arms={}
        for arm in order:
            folder=BASE/(arm+'-perf-'+str(seed));folder.mkdir(exist_ok=False)
            cmd=['/root/miniconda3/bin/python',str(BASE/'launch_gradnorm-02.py'),'--bf16-root',str(BF),
                 '--gradnorm-receipt',str(folder/'gradnorm.json'),'--megatron-root',str(NATIVE),
                 '--contract',str(contract),'--entry-evidence',str(folder/'entry.json')]
            if arm!='native':cmd+=['--candidate',str(parent)]
            if arm=='optimized':cmd+=['--gradnorm-candidate',str(BASE/'candidate.py')]
            cmd+=c['training_argv']+['--train-iters','120','--seed',str(seed),'--log-interval','10']
            record={'case':folder.name,'mode':'native' if arm=='native' else 'optimized','steps':120,
                    'seed':seed,'audit':False,'profile':False,'contract_sha256':sha(contract),
                    'command':cmd,'started_utc':datetime.now(timezone.utc).isoformat()}
            with open('/tmp/kda-biv150-gpu0.lock','a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
                    record['exit_code']=subprocess.run(cmd,env=env,cwd=NATIVE,stdout=out,stderr=err,timeout=180).returncode
            record['finished_utc']=datetime.now(timezone.utc).isoformat()
            (folder/'command.json').write_text(json.dumps(record,indent=2)+'\n')
            assert record['exit_code']==0
            arms[arm]=module.validate_case(folder,record['mode'],seed,c,sha(contract),sha(parent))
            hook=json.loads((folder/'gradnorm.json').read_text())
            assert hook['enabled']==(arm=='optimized') and hook['calls']==(120 if arm=='optimized' else 0)
            assert hook['wrapper_sha256']==sha(BASE/'launch_gradnorm-02.py')
            if arm=='optimized':assert hook['candidate_sha256']==sha(BASE/'candidate.py')
            print(json.dumps({'seed':seed,'arm':arm,'iteration_ms':arms[arm]['mean_iteration_ms']}),flush=True)
        report['pairs'].append({'seed':seed,'order':order,'arms':arms,
                               'vs_native':arms['native']['mean_iteration_ms']/arms['optimized']['mean_iteration_ms'],
                               'vs_control':arms['control']['mean_iteration_ms']/arms['optimized']['mean_iteration_ms']})
    for metric in ('vs_native','vs_control'):
        ratio=math.exp(statistics.mean(math.log(r[metric]) for r in report['pairs']))
        report[metric]={'ratio':ratio,'gain_percent':(ratio-1)*100,'all_seeds_faster':all(r[metric]>1 for r in report['pairs'])}
    report['status']='validated'
    (BASE/'performance-01.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','vs_native','vs_control')}),flush=True)


if __name__=='__main__':main()

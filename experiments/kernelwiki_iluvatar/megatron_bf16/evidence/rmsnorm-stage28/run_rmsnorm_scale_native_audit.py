"""End-to-end native baseline audit against both held-out optimized arms."""
import argparse,fcntl,hashlib,importlib.util,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import torch
BASE=Path('/tmp/kda-rmsnorm-stage28');SEALED=Path('/tmp/kda-rmsnorm-stage27');BF=Path('/tmp/kda-bf16-continuation/megatron_bf16');NATIVE=BF.parent/'megatron-native-5be9626';GRAD=Path('/tmp/kda-gradnorm-stage26')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--model',choices=('wide','deep'),required=True);a=p.parse_args()
contract=BF/('contracts/0008-'+a.model+'.json');c=json.loads(contract.read_text());parent=BF/'candidates/0008/candidate.py'
folder=BASE/('native-audit-'+a.model);folder.mkdir(exist_ok=False)
cmd=['/root/miniconda3/bin/python',str(SEALED/'launch_rmsnorm-02.py'),'--gradnorm-launcher',str(GRAD/'launch_gradnorm-04.py'),'--rmsnorm-receipt',str(folder/'rmsnorm.json'),'--bf16-root',str(BF),'--gradnorm-receipt',str(folder/'gradnorm.json'),'--megatron-root',str(NATIVE),'--contract',str(contract),'--entry-evidence',str(folder/'entry.json'),'--audit-dir',str(folder/'snapshots')]
cmd+=c['training_argv']+['--train-iters','3','--seed','28101','--log-interval','1','--save',str(folder/'snapshots'),'--save-interval','3','--save-wgrads-interval','1','--save-params-interval','1']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',MASTER_ADDR='127.0.0.1',MASTER_PORT='29953',RANK='0',WORLD_SIZE='1',LOCAL_RANK='0')
for k in list(env):
 if 'API_KEY' in k or 'AUTH_TOKEN' in k:env.pop(k)
record={'case':folder.name,'command':cmd,'mode':'native','steps':3,'seed':28101,'audit':True,'profile':False,'contract_sha256':sha(contract),'started_utc':datetime.now(timezone.utc).isoformat()};start=time.perf_counter()
with open('/tmp/kda-biv150-gpu0.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
  record['exit_code']=subprocess.run(cmd,env=env,cwd=NATIVE,stdout=out,stderr=err,timeout=180).returncode
record.update(finished_utc=datetime.now(timezone.utc).isoformat(),whole_process_elapsed_seconds=time.perf_counter()-start)
(folder/'command.json').write_text(json.dumps(record,indent=2)+'\n');assert record['exit_code']==0
spec=importlib.util.spec_from_file_location('sealed_compare',BF/'compare_training.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for arm in ('control','optimized'):
 target=BASE/(arm+'-audit-'+a.model+'-006-complete')
 report=m.Comparison(folder,target,torch,contract,parent).run()
 report.update(verifier_sha256=sha(Path(__file__)))
 (BASE/('native-comparison-'+a.model+'-'+arm+'.json')).write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'model':a.model,'arm':arm,'passed':report['passed'],'checks':report['check_summary']}),flush=True)

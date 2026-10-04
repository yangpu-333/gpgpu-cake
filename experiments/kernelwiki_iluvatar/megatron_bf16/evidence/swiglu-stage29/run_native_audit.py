"""Immutable original pretrain_gpt audit invocation for native SwiGLU."""
import fcntl,json,hashlib,os,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
BASE=Path('/tmp/kda-swiglu-stage29');BF=Path('/tmp/kda-bf16-continuation/megatron_bf16');NATIVE=BF.parent/'megatron-native-5be9626';GRAD=Path('/tmp/kda-gradnorm-stage26')
folder=BASE/'main-native-audit';folder.mkdir(exist_ok=False);contract=BASE/'contracts/main.json';c=json.loads(contract.read_text())
cmd=['/root/miniconda3/bin/python',str(BASE/'launch_swiglu.py'),'--swiglu-receipt',str(folder/'swiglu.json'),'--capture-swiglu',str(folder/'activation'),'--gradnorm-launcher',str(GRAD/'launch_gradnorm-04.py'),'--bf16-root',str(BF),'--gradnorm-receipt',str(folder/'gradnorm.json'),'--megatron-root',str(NATIVE),'--contract',str(contract),'--entry-evidence',str(folder/'entry.json'),'--audit-dir',str(folder/'snapshots')]
cmd+=c['training_argv']+['--train-iters','3','--seed','29101','--log-interval','1','--save',str(folder/'snapshots'),'--save-interval','3','--save-wgrads-interval','1','--save-params-interval','1']
env=os.environ.copy()
for k in list(env):
 if 'API_KEY' in k or 'AUTH_TOKEN' in k:env.pop(k)
env.update(CUDA_VISIBLE_DEVICES='0',MASTER_ADDR='127.0.0.1',MASTER_PORT='29961',RANK='0',WORLD_SIZE='1',LOCAL_RANK='0')
r={'case':folder.name,'command':cmd,'mode':'native','steps':3,'seed':29101,'audit':True,'profile':False,'contract_sha256':hashlib.sha256(contract.read_bytes()).hexdigest(),'started_utc':datetime.now(timezone.utc).isoformat()};start=time.perf_counter()
with open('/tmp/kda-biv150-gpu0.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
  r['exit_code']=subprocess.run(cmd,env=env,cwd=NATIVE,stdout=out,stderr=err,timeout=240).returncode
r.update(finished_utc=datetime.now(timezone.utc).isoformat(),whole_process_elapsed_seconds=time.perf_counter()-start);(folder/'command.json').write_text(json.dumps(r,indent=2)+'\n');assert r['exit_code']==0
entry=json.loads((folder/'entry.json').read_text());receipt=json.loads((folder/'swiglu.json').read_text())
print(json.dumps({'model':entry['models'][0],'swiglu':receipt}),flush=True)

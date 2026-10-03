"""Separate immutable three-step CE+gradnorm-controlled RMSNorm audit."""
import argparse,fcntl,hashlib,importlib.util,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import torch
BASE=Path('/tmp/kda-rmsnorm-stage27');BF=Path('/tmp/kda-bf16-continuation/megatron_bf16');NATIVE=BF.parent/'megatron-native-5be9626'
GRAD=Path('/tmp/kda-gradnorm-stage26');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--version',default='002');p.add_argument('--case-suffix',default='complete');a=p.parse_args()
candidate=BASE/('candidate-'+a.version+'.py');suffix=a.version+'-'+a.case_suffix
contract=BF/'contracts/0008-main.json';parent=BF/'candidates/0008/candidate.py';c=json.loads(contract.read_text())
probe=json.loads((BASE/('probe-'+a.version+'.json')).read_text());assert probe['status']=='passed' and probe['candidate_sha256']==sha(candidate)
env=os.environ.copy()
for name in list(env):
 if 'API_KEY' in name or 'AUTH_TOKEN' in name:env.pop(name)
env.update(CUDA_VISIBLE_DEVICES='0',MASTER_ADDR='127.0.0.1',MASTER_PORT='29941',RANK='0',WORLD_SIZE='1',LOCAL_RANK='0')
for arm in ('control','optimized'):
 folder=BASE/(arm+'-audit-'+suffix);folder.mkdir(exist_ok=False)
 cmd=['/root/miniconda3/bin/python',str(BASE/'launch_rmsnorm.py'),'--gradnorm-launcher',str(GRAD/'launch_gradnorm-04.py'),
      '--rmsnorm-receipt',str(folder/'rmsnorm.json'),'--bf16-root',str(BF),'--gradnorm-receipt',str(folder/'gradnorm.json'),
      '--gradnorm-candidate',str(GRAD/'candidate.py'),'--megatron-root',str(NATIVE),'--contract',str(contract),
      '--entry-evidence',str(folder/'entry.json'),'--candidate',str(parent),'--audit-dir',str(folder/'snapshots')]
 if arm=='optimized':cmd+=['--rmsnorm-candidate',str(candidate)]
 cmd+=c['training_argv']+['--train-iters','3','--seed','27101','--log-interval','1','--save',str(folder/'snapshots'),
                         '--save-interval','3','--save-wgrads-interval','1','--save-params-interval','1']
 record={'command':cmd,'mode':'optimized','steps':3,'seed':27101,'audit':True,'profile':False,'contract_sha256':sha(contract),'started_utc':datetime.now(timezone.utc).isoformat()}
 start=time.perf_counter()
 with open('/tmp/kda-biv150-gpu0.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
   record['exit_code']=subprocess.run(cmd,env=env,cwd=NATIVE,stdout=out,stderr=err,timeout=180).returncode
 record.update(finished_utc=datetime.now(timezone.utc).isoformat(),whole_process_elapsed_seconds=time.perf_counter()-start)
 (folder/'command.json').write_text(json.dumps(record,indent=2)+'\n');assert record['exit_code']==0,arm
 print('Audit process completed: '+arm,flush=True)
spec=importlib.util.spec_from_file_location('sealed_compare',BF/'compare_training.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
control=BASE/('control-audit-'+suffix);optimized=BASE/('optimized-audit-'+suffix)
class Controlled(m.Comparison):
 def require(self,condition,name,**details):
  if name=='native has no candidate installation':
   e=json.loads((control/'entry.json').read_text());n=json.loads((control/'rmsnorm.json').read_text());o=json.loads((optimized/'rmsnorm.json').read_text())
   condition=e['iluvatar_kernels'] and e['kernel_adapters'][0]['candidate_sha256']==sha(parent)
   condition=condition and not n['enabled'] and n['calls']==0 and o['enabled'] and o['calls']==27 and o['fallback_calls']==0
   condition=condition and o['candidate_sha256']==sha(candidate) and n['wrapper_sha256']==o['wrapper_sha256']==sha(BASE/'launch_rmsnorm.py')
   for folder in (control,optimized):
    g=json.loads((folder/'gradnorm.json').read_text());condition=condition and g['calls']==3 and g['candidate_sha256']==sha(GRAD/'candidate.py') and g['fallback_calls']==0
   name='Identical CE0008+gradnorm baseline; only explicit RMSNorm candidate differs'
  elif name=='native run mode':condition=True;name='Controlled baseline explicitly optimized CE+gradnorm'
  return super().require(condition,name,**details)
report=Controlled(control,optimized,torch,contract,parent).run()
report.update(schema='CE0008-gradnorm-controlled-rmsnorm-audit-v1',rmsnorm_candidate_sha256=sha(candidate),verifier_sha256=sha(Path(__file__)))
(BASE/('comparison-'+suffix+'.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'checks':report['check_summary']}),flush=True)

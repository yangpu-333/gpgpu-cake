"""Formal native/control/RMSNorm candidate benchmark after full audit passes."""
import argparse,fcntl,hashlib,importlib.util,json,math,os,statistics,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
SEALED=Path('/tmp/kda-rmsnorm-stage27');BASE=Path('/tmp/kda-rmsnorm-stage28');BF=Path('/tmp/kda-bf16-continuation/megatron_bf16');NATIVE=BF.parent/'megatron-native-5be9626';GRAD=Path('/tmp/kda-gradnorm-stage26')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--version',default='002');p.add_argument('--model',choices=('wide','deep'),required=True);a=p.parse_args();candidate=SEALED/('candidate-'+a.version+'.py')
audit=BASE/('comparison-'+a.model+'-'+a.version+'-complete.json');assert json.loads(audit.read_text())['passed']
contract=BF/('contracts/0008-'+a.model+'.json');c=json.loads(contract.read_text());parent=BF/'candidates/0008/candidate.py'
spec=importlib.util.spec_from_file_location('sealed_perf',BF/'summarize_performance.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
source=(BF/'summarize_performance.py').read_text();start=source.index("    invocation=command['command']");end=source.index("    runtime=entry['runtime_compatibility']",start)
replacement="""    invocation=command['command']
    require(invocation==EXPECTED_COMMANDS[str(folder)],'exact controlled RMSNorm wrapper invocation')
    require(entry['formal_entry']==str(Path('/tmp/kda-bf16-continuation/megatron-native-5be9626/pretrain_gpt.py')),'original formal entry')
"""
adapted=source[:start]+replacement+source[end:];exec(compile(adapted,str(BF/'summarize_performance.py'),'exec'),m.__dict__);m.EXPECTED_COMMANDS={}
env=os.environ.copy()
for name in list(env):
 if 'API_KEY' in name or 'AUTH_TOKEN' in name:env.pop(name)
env.update(CUDA_VISIBLE_DEVICES='0',MASTER_ADDR='127.0.0.1',MASTER_PORT='29952',RANK='0',WORLD_SIZE='1',LOCAL_RANK='0')
report={'candidate_sha256':sha(candidate),'audit_sha256':sha(audit),'runner_sha256':sha(Path(__file__)),
        'adapted_validator_sha256':hashlib.sha256(adapted.encode()).hexdigest(),'scope':'Same formal synchronized native time.time timer, held-out BF16 localTorch model, seeds rotate three-arm order; separate 3-step audit, no profile/audit/eval/checkpoints during timing. Only outer argv validator adapted.','pairs':[]}
orders=(('native','control','optimized'),('optimized','control','native'),('control','native','optimized'))
for seed,order in zip((28202,28203,28204),orders):
 arms={}
 for arm in order:
  folder=BASE/(arm+'-'+a.model+'-perf-'+a.version+'-'+str(seed));folder.mkdir(exist_ok=False)
  cmd=['/root/miniconda3/bin/python',str(SEALED/'launch_rmsnorm-02.py'),'--gradnorm-launcher',str(GRAD/'launch_gradnorm-04.py'),
       '--rmsnorm-receipt',str(folder/'rmsnorm.json'),'--bf16-root',str(BF),'--gradnorm-receipt',str(folder/'gradnorm.json'),
       '--megatron-root',str(NATIVE),'--contract',str(contract),'--entry-evidence',str(folder/'entry.json')]
  if arm!='native':cmd+=['--candidate',str(parent),'--gradnorm-candidate',str(GRAD/'candidate.py')]
  if arm=='optimized':cmd+=['--rmsnorm-candidate',str(candidate)]
  cmd+=c['training_argv']+['--train-iters','120','--seed',str(seed),'--log-interval','10'];m.EXPECTED_COMMANDS[str(folder)]=cmd
  record={'case':folder.name,'command':cmd,'mode':'native' if arm=='native' else 'optimized','steps':120,'seed':seed,'audit':False,'profile':False,
          'contract_sha256':sha(contract),'started_utc':datetime.now(timezone.utc).isoformat()};start=time.perf_counter()
  with open('/tmp/kda-biv150-gpu0.lock','a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
    record['exit_code']=subprocess.run(cmd,env=env,cwd=NATIVE,stdout=out,stderr=err,timeout=180).returncode
  record.update(finished_utc=datetime.now(timezone.utc).isoformat(),whole_process_elapsed_seconds=time.perf_counter()-start)
  (folder/'command.json').write_text(json.dumps(record,indent=2)+'\n');assert record['exit_code']==0
  arms[arm]=m.validate_case(folder,record['mode'],seed,c,sha(contract),sha(parent))
  r=json.loads((folder/'rmsnorm.json').read_text());g=json.loads((folder/'gradnorm.json').read_text())
  assert r['enabled']==(arm=='optimized') and r['calls']==((2*int(c['training_argv'][c['training_argv'].index('--num-layers')+1])+1)*120 if arm=='optimized' else 0) and r['fallback_calls']==0
  assert g['enabled']==(arm!='native') and g['calls']==(120 if arm!='native' else 0) and g['fallback_calls']==0
  print(json.dumps({'arm':arm,'seed':seed,'iteration_ms':arms[arm]['mean_iteration_ms']}),flush=True)
 report['pairs'].append({'seed':seed,'order':order,'arms':arms,'vs_native':arms['native']['mean_iteration_ms']/arms['optimized']['mean_iteration_ms'],
                         'vs_control':arms['control']['mean_iteration_ms']/arms['optimized']['mean_iteration_ms']})
for metric in ('vs_native','vs_control'):
 ratio=math.exp(statistics.mean(math.log(r[metric]) for r in report['pairs']));report[metric]={'ratio':ratio,'gain_percent':(ratio-1)*100,'all_seeds_faster':all(r[metric]>1 for r in report['pairs'])}
report['status']='validated';(BASE/('performance-'+a.model+'-'+a.version+'.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('status','vs_native','vs_control')}),flush=True)

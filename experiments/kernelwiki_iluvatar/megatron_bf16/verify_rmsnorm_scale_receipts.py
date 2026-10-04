"""Recompute held-out training results from archived raw logs; no GPU claim."""
import hashlib,importlib.util,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'evidence/rmsnorm-stage28'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def verify(model):
 audit=read(BASE/('comparison-'+model+'-006-complete.json'))
 assert audit['passed'] and audit['check_summary']['failed']==0
 assert all(x['passed'] for x in audit['checks'])
 assert all(x['passed'] and x['all_elements_finite'] for x in audit['tensor_comparisons'])
 assert audit['rmsnorm_candidate_sha256']==sha(ROOT/'evidence/rmsnorm-stage27/candidate-006.py')
 assert audit['verifier_sha256']==sha(BASE/'run_rmsnorm_scale_audit-03.py')
 for arm in ('control','optimized'):
  native_audit=read(BASE/('native-comparison-'+model+'-'+arm+'.json'))
  assert native_audit['passed'] and native_audit['check_summary']['failed']==0
  assert all(x['passed'] and x['all_elements_finite'] for x in native_audit['tensor_comparisons'])
  assert native_audit['verifier_sha256']==sha(BASE/'run_rmsnorm_scale_native_audit.py')
 perf=read(BASE/('performance-'+model+'-006.json'))
 assert perf['audit_sha256']==sha(BASE/('comparison-'+model+'-006-complete.json'))
 assert perf['runner_sha256']==sha(BASE/'run_rmsnorm_scale_performance.py')
 assert perf['candidate_sha256']==audit['rmsnorm_candidate_sha256']
 spec=importlib.util.spec_from_file_location('parser',ROOT.parent/'megatron_training_entry/summarize_performance.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 ratios={'vs_native':[],'vs_control':[]}
 contract=read(ROOT/('contracts/0008-'+model+'.json'))
 manifest=read(ROOT.parent/'megatron_training_entry/source_manifest.json')
 hashes={n:sha(ROOT/n) for n in ('launch_training.py','kernel_adapter.py','run_case.py')}
 hashes.update({'formal/'+n:sha(ROOT.parent/'megatron_training_entry'/n) for n in ('runtime_compat.py','source_manifest.json')})
 build=read(ROOT/'evidence/rmsnorm-stage27/cpp-build-006-receipt.json')
 layers=int(contract['training_argv'][contract['training_argv'].index('--num-layers')+1])
 for pair,seed in zip(perf['pairs'],(28202,28203,28204)):
  assert pair['seed']==seed
  means={}
  for arm in pair['order']:
   f=BASE/(arm+'-'+model+'-perf-006-'+str(seed))
   cmd=read(f/'command.json');r=read(f/'rmsnorm.json');g=read(f/'gradnorm.json')
   assert cmd['exit_code']==0 and cmd['steps']==120 and not cmd['audit'] and not cmd['profile']
   assert cmd['contract_sha256']==sha(ROOT/('contracts/0008-'+model+'.json'))
   assert cmd['command'][-len(contract['training_argv'])-6:]==contract['training_argv']+['--train-iters','120','--seed',str(seed),'--log-interval','10']
   remote='/tmp/kda-rmsnorm-stage28';sealed='/tmp/kda-rmsnorm-stage27';bf='/tmp/kda-bf16-continuation/megatron_bf16';native='/tmp/kda-bf16-continuation/megatron-native-5be9626';grad='/tmp/kda-gradnorm-stage26';name=f.name
   expected=['/root/miniconda3/bin/python',sealed+'/launch_rmsnorm-02.py','--gradnorm-launcher',grad+'/launch_gradnorm-04.py','--rmsnorm-receipt',remote+'/'+name+'/rmsnorm.json','--bf16-root',bf,'--gradnorm-receipt',remote+'/'+name+'/gradnorm.json','--megatron-root',native,'--contract',bf+'/contracts/0008-'+model+'.json','--entry-evidence',remote+'/'+name+'/entry.json']
   if arm!='native':expected+=['--candidate',bf+'/candidates/0008/candidate.py','--gradnorm-candidate',grad+'/candidate.py']
   if arm=='optimized':expected+=['--rmsnorm-candidate',sealed+'/candidate-006.py']
   assert cmd['command']==expected+contract['training_argv']+['--train-iters','120','--seed',str(seed),'--log-interval','10']

   assert r['calls']==((2*layers+1)*120 if arm=='optimized' else 0) and r['fallback_calls']==0
   assert g['calls']==(120 if arm!='native' else 0) and g['fallback_calls']==0
   entry=read(f/'entry.json')
   assert entry['status']=='completed' and entry['native_sources_sha256']==manifest['files'] and entry['megatron_commit']==manifest['commit']
   assert entry['integration_sha256']==hashes and entry['profile_hooks']==[] and not entry.get('audit')
   assert entry['runtime_compatibility']['optional_backends_disabled_in_process']==['transformer_engine','apex']
   assert entry['models'][0]['layers']==layers and entry['models'][0]['bf16'] is True
   assert entry['models'][0]['parameters']==(16912896 if model=='wide' else 33825280)
   assert entry['iluvatar_kernels']==(arm!='native')
   if arm!='native':
    h=entry['kernel_adapters'][0]
    assert h['candidate_sha256']==sha(ROOT/'candidates/0008/candidate.py')
    assert h['counts']=={'ce_candidate_calls':120,'ce_native_calls':0,'residual_native_calls':layers*120,'fallback_reasons':{}}
    assert all(o['autocast'] is False and o['logits_dtype']=='torch.bfloat16' for o in h['observations'])
    assert g['candidate_sha256']==sha(ROOT/'gradnorm_candidates/001/candidate.py')
   else:assert entry['kernel_adapters']==[]
   assert r['wrapper_sha256']==sha(ROOT/'evidence/rmsnorm-stage27/launch_rmsnorm-02.py')
   if arm=='optimized':
    assert r['extension']['sha256']==build['artifact_sha256'] and r['extension']['sources']==build['sources']
    assert r['candidate_sha256']==perf['candidate_sha256']
   # Archived means are separately checked against all native interval logs.
   rows=m.parse_intervals((f/'stdout.log').read_text())
   selected=[x for x in rows if 30<=x['iteration']<=120]
   means[arm]=statistics.mean(x['mean_iteration_ms'] for x in selected)
   assert math.isclose(means[arm],pair['arms'][arm]['mean_iteration_ms'],abs_tol=1e-9)
  for key,arm in (('vs_native','native'),('vs_control','control')):
   ratios[key].append(means[arm]/means['optimized'])
 for key,values in ratios.items():
  ratio=math.exp(statistics.mean(map(math.log,values)))
  assert math.isclose(ratio,perf[key]['ratio'],abs_tol=1e-12)
 return {'model':model,'checks':audit['check_summary'],'vs_native':perf['vs_native'],'vs_control':perf['vs_control']}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--model',choices=('wide','deep'),required=True)
 print(json.dumps(verify(p.parse_args().model)))

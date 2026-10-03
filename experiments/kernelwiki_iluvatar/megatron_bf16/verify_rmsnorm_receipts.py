"""CPU-only gate binding native sources, C++ build, numeric audits and raw timers."""
import hashlib,importlib.util,json,math,statistics
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT/'evidence/rmsnorm-stage27'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))

def verify():
 decision=read(BASE/'decision.json');perf=read(BASE/'performance-006.json');audit=read(BASE/'comparison-006-complete.json')
 build=read(BASE/'cpp-build-006-receipt.json');candidate=BASE/'candidate-006.py'
 assert decision['status']=='accepted_for_tested_main_scope'
 assert build['status']=='built' and build['torch']=='2.4.1' and build['cuda']=='10.2'
 assert build['sources']==decision['compiled_sources_sha256']=={n:sha(BASE/n) for n in ('affine-006.cpp','affine-006.cu')}
 assert build['artifact_sha256']==decision['artifact_sha256']==read(BASE/'artifacts-006/receipt.json')['artifact_sha256']
 assert build['build_script_sha256']==sha(BASE/'build_rmsnorm_cpp.py')
 assert sha(candidate)==decision['candidate_sha256']==perf['candidate_sha256']==audit['rmsnorm_candidate_sha256']
 # Windows model-output serialization and tested Linux files differ in EOL only.
 for original,actual in (('candidate.py','candidate-006.py'),('affine.cpp','affine-006.cpp'),('affine.cu','affine-006.cu')):
  assert (ROOT/'rmsnorm_candidates/006'/original).read_text()==(BASE/actual).read_text()
 for name in ('probe-006.json','layouts-006.json'):
  proof=read(BASE/name);assert proof['status']=='passed' and proof['candidate_sha256']==sha(candidate)
  checks=proof['checks'] if name.startswith('layouts') else [c for r in proof['checks'] for c in r['checks']]
  assert len(checks)==36 and all(c['passed'] and c['bitwise_equal'] for c in checks)
 assert audit['passed'] and audit['check_summary']=={'total':1475,'passed':1475,'failed':0}
 assert all(c['passed'] for c in audit['checks']) and all(t['passed'] and t['exact_equal'] and t['all_elements_finite'] for t in audit['tensor_comparisons'])
 assert sha(BASE/'comparison-006-complete.json')==perf['audit_sha256']
 assert sha(BASE/'run_rmsnorm_performance-04.py')==perf['runner_sha256']
 assert audit['verifier_sha256']==sha(BASE/'run_rmsnorm_audit-03.py')
 old=read(BASE/'comparison-005-complete.json');assert not old['passed'] and old['check_summary']['failed']==8
 rejected=read(BASE/'performance-003.json');assert rejected['vs_control']['ratio']<1
 spec=importlib.util.spec_from_file_location('raw_parser',ROOT.parent/'megatron_training_entry/summarize_performance.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
 manifest=read(ROOT.parent/'megatron_training_entry/source_manifest.json');contract=read(ROOT/'contracts/0008-main.json')
 hashes={n:sha(ROOT/n) for n in ('launch_training.py','kernel_adapter.py','run_case.py')}
 hashes.update({'formal/'+n:sha(ROOT.parent/'megatron_training_entry'/n) for n in ('runtime_compat.py','source_manifest.json')})
 remote='/tmp/kda-rmsnorm-stage27';bf='/tmp/kda-bf16-continuation/megatron_bf16';native='/tmp/kda-bf16-continuation/megatron-native-5be9626';grad='/tmp/kda-gradnorm-stage26'
 ratios={'vs_native':[],'vs_control':[]};assert len(perf['pairs'])==3
 orders=(['native','control','optimized'],['optimized','control','native'],['control','native','optimized'])
 for pair,seed,order in zip(perf['pairs'],(27202,27203,27204),orders):
  assert pair['seed']==seed and pair['order']==order;means={};commands={}
  for arm in order:
   name=arm+'-perf03-006-'+str(seed);folder=BASE/name;cmd=read(folder/'command.json');entry=read(folder/'entry.json');r=read(folder/'rmsnorm.json');g=read(folder/'gradnorm.json');commands[arm]=cmd
   assert cmd['case']==name and cmd['exit_code']==0 and cmd['steps']==120 and cmd['seed']==seed and cmd['audit'] is False and cmd['profile'] is False
   argv=contract['training_argv']+['--train-iters','120','--seed',str(seed),'--log-interval','10']
   expected=['/root/miniconda3/bin/python',remote+'/launch_rmsnorm-02.py','--gradnorm-launcher',grad+'/launch_gradnorm-04.py',
             '--rmsnorm-receipt',remote+'/'+name+'/rmsnorm.json','--bf16-root',bf,'--gradnorm-receipt',remote+'/'+name+'/gradnorm.json',
             '--megatron-root',native,'--contract',bf+'/contracts/0008-main.json','--entry-evidence',remote+'/'+name+'/entry.json']
   if arm!='native':expected+=['--candidate',bf+'/candidates/0008/candidate.py','--gradnorm-candidate',grad+'/candidate.py']
   if arm=='optimized':expected+=['--rmsnorm-candidate',remote+'/candidate-006.py']
   assert cmd['command']==expected+argv and entry['training_argv']==argv and entry['formal_entry']==native+'/pretrain_gpt.py'
   assert entry['native_sources_sha256']==manifest['files'] and entry['megatron_commit']==manifest['commit'] and entry['integration_sha256']==hashes
   assert entry['contract_sha256']==cmd['contract_sha256']==sha(ROOT/'contracts/0008-main.json')
   assert entry['profile_hooks']==[] and not entry.get('audit')
   model=entry['models'][0];assert model['parameters']==13701632 and model['layers']==4 and model['bf16'] is True
   assert entry['runtime_compatibility']['optional_backends_disabled_in_process']==['transformer_engine','apex']
   assert entry['iluvatar_kernels']==(arm!='native')
   if arm!='native':
    handle=entry['kernel_adapters'][0];assert handle['candidate_sha256']==sha(ROOT/'candidates/0008/candidate.py')
    assert handle['counts']=={'ce_candidate_calls':120,'ce_native_calls':0,'residual_native_calls':480,'fallback_reasons':{}}
    assert all(o['autocast'] is False and o['logits_dtype']=='torch.bfloat16' for o in handle['observations'])
   else:assert entry['kernel_adapters']==[]
   assert r['wrapper_sha256']==sha(BASE/'launch_rmsnorm-02.py') and r['calls']==(1080 if arm=='optimized' else 0) and r['fallback_calls']==0
   assert g['calls']==(120 if arm!='native' else 0) and g['fallback_calls']==0
   if arm!='native':assert g['candidate_sha256']==sha(ROOT/'gradnorm_candidates/001/candidate.py')
   if arm=='optimized':
    assert r['candidate_sha256']==sha(candidate) and r['extension']['sha256']==build['artifact_sha256'] and r['extension']['sources']==build['sources']
   intervals=parser.parse_intervals((folder/'stdout.log').read_text());means[arm]=statistics.mean(x['mean_iteration_ms'] for x in intervals if x['retained'])
   assert means[arm]==pair['arms'][arm]['mean_iteration_ms']
  for x,y in zip(order,order[1:]):assert datetime.fromisoformat(commands[x]['finished_utc'])<=datetime.fromisoformat(commands[y]['started_utc'])
  for metric,base in (('vs_native','native'),('vs_control','control')):
   ratio=means[base]/means['optimized'];assert ratio==pair[metric];ratios[metric].append(ratio)
 for metric,values in ratios.items():
  ratio=math.exp(statistics.mean(math.log(x) for x in values));assert ratio==perf[metric]['ratio']==decision[metric]['ratio'] and all(x>1 for x in values)
 assert decision['vs_control']['gain_percent']>1
 return {'status':'verified','checks':1475,'timed_processes':9,'vs_control':decision['vs_control'],'vs_native':decision['vs_native']}

if __name__=='__main__':print(json.dumps(verify()))

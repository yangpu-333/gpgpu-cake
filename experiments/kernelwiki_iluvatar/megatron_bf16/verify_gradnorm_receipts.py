"""CPU-only recheck of stage26 correctness, raw timing and source provenance."""
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'evidence/gradnorm-stage26'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))


def verify():
    decision=read(BASE/'decision.json');audit=read(BASE/'comparison-04.json');performance=read(BASE/'performance-01.json')
    candidate=ROOT/'gradnorm_candidates/001/candidate.py';contract_path=ROOT/'contracts/0008-main.json';contract=read(contract_path)
    assert decision['candidate_sha256']==audit['gradnorm_candidate_sha256']==performance['candidate_sha256']==sha(candidate)
    assert decision['receipt_sha256']=={'audit':sha(BASE/'comparison-04.json'),'performance':sha(BASE/'performance-01.json')}
    assert audit['passed'] and audit['check_summary']=={'total':1475,'passed':1475,'failed':0}
    assert all(t['passed'] and t['exact_equal'] and t['all_elements_finite'] for t in audit['tensor_comparisons'])
    norm=read(BASE/'optimized-audit-03/gradnorm.json')
    assert len(norm['norm_audit'])==3 and all(r['bitwise_equal'] and r['native_norm']==r['candidate_norm'] for r in norm['norm_audit'])
    coverage=read(BASE/'coverage-120-01/gradnorm.json')
    assert coverage['calls']==120 and coverage['fallback_calls']==0 and coverage['candidate_sha256']==sha(candidate)
    assert coverage['wrapper_sha256']==sha(BASE/'launch_gradnorm-04.py')
    assert performance['runner_sha256']==sha(BASE/'run_gradnorm_performance-03.py')
    assert performance['audit_sha256']==sha(BASE/'comparison-03.json')
    formal=ROOT.parent/'megatron_training_entry'
    spec=importlib.util.spec_from_file_location('original_log_parser',formal/'summarize_performance.py')
    parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    manifest=read(formal/'source_manifest.json')
    hashes={name:sha(ROOT/name) for name in ('launch_training.py','kernel_adapter.py','run_case.py')}
    hashes.update({'formal/'+name:sha(formal/name) for name in ('runtime_compat.py','source_manifest.json')})
    assert len(performance['pairs'])==3
    ratios={'vs_native':[],'vs_control':[]}
    remote='/tmp/kda-gradnorm-stage26';bf='/tmp/kda-bf16-continuation/megatron_bf16';native='/tmp/kda-bf16-continuation/megatron-native-5be9626'
    for pair,seed,order in zip(performance['pairs'],(26102,26103,26104),
                             (['native','control','optimized'],['optimized','control','native'],['control','native','optimized'])):
        assert pair['seed']==seed and pair['order']==order
        means={};commands={}
        for arm in order:
            name=arm+'-perf03-'+str(seed);folder=BASE/name;cmd=read(folder/'command.json');entry=read(folder/'entry.json');hook=read(folder/'gradnorm.json')
            commands[arm]=cmd
            assert cmd['case']==name and cmd['steps']==120 and cmd['seed']==seed and cmd['exit_code']==0
            assert cmd['audit'] is False and cmd['profile'] is False and entry['profile_hooks']==[]
            assert cmd['mode']==('native' if arm=='native' else 'optimized') and entry['status']=='completed'
            assert entry['native_sources_sha256']==manifest['files'] and entry['megatron_commit']==manifest['commit']
            assert entry['integration_sha256']==hashes and entry['launcher_sha256']==hashes['launch_training.py']
            assert entry['contract_sha256']==cmd['contract_sha256']==sha(contract_path)
            argv=contract['training_argv']+['--train-iters','120','--seed',str(seed),'--log-interval','10']
            expected=['/root/miniconda3/bin/python',remote+'/launch_gradnorm-02.py','--bf16-root',bf,
                      '--gradnorm-receipt',remote+'/'+name+'/gradnorm.json','--megatron-root',native,
                      '--contract',bf+'/contracts/0008-main.json','--entry-evidence',remote+'/'+name+'/entry.json']
            if arm!='native':expected+=['--candidate',bf+'/candidates/0008/candidate.py']
            if arm=='optimized':expected+=['--gradnorm-candidate',remote+'/candidate.py']
            assert cmd['command']==expected+argv and entry['training_argv']==argv
            assert entry['formal_entry']==native+'/pretrain_gpt.py' and len(entry['models'])==1
            model=entry['models'][0]
            assert model['bf16'] is True and model['fp16'] is False and model['parameters']==13701632 and model['layers']==4
            assert entry['runtime_compatibility']['optional_backends_disabled_in_process']==['transformer_engine','apex']
            assert entry['iluvatar_kernels']==(arm!='native')
            if arm=='native':assert entry['kernel_adapters']==[]
            else:
                handle=entry['kernel_adapters'][0]
                assert handle['candidate_sha256']==sha(ROOT/'candidates/0008/candidate.py')
                assert handle['counts']=={'ce_candidate_calls':120,'ce_native_calls':0,'residual_native_calls':480,'fallback_reasons':{}}
                assert all(o['autocast'] is False and o['logits_dtype']=='torch.bfloat16' for o in handle['observations'])
            assert hook['wrapper_sha256']==sha(BASE/'launch_gradnorm-02.py')
            assert hook['calls']==(120 if arm=='optimized' else 0) and hook['enabled']==(arm=='optimized')
            if arm=='optimized':assert hook['candidate_sha256']==sha(candidate)
            intervals=parser.parse_intervals((folder/'stdout.log').read_text())
            means[arm]=statistics.mean(r['mean_iteration_ms'] for r in intervals if r['retained'])
            assert means[arm]==pair['arms'][arm]['mean_iteration_ms']
        for before,after in zip(order,order[1:]):assert datetime.fromisoformat(commands[before]['finished_utc'])<=datetime.fromisoformat(commands[after]['started_utc'])
        for metric,base in (('vs_native','native'),('vs_control','control')):
            ratio=means[base]/means['optimized'];assert ratio==pair[metric];ratios[metric].append(ratio)
    for metric,values in ratios.items():
        ratio=math.exp(statistics.mean(math.log(v) for v in values))
        assert ratio==performance[metric]['ratio']==decision[metric]['ratio'] and all(v>1 for v in values)
    assert decision['vs_control']['gain_percent']>1
    return {'status':'verified','checks':1475,'timed_processes':9,'vs_native':decision['vs_native'],'vs_control':decision['vs_control']}


if __name__=='__main__':print(json.dumps(verify()))

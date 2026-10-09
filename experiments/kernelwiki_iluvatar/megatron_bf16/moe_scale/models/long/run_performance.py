"""Fixed stage34 three-seed rotated native/control/weighted-candidate formal timing."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BF = Path('/tmp/kda-bf16-continuation/megatron_bf16')
FORMAL = BF.parent/'megatron_training_entry'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())
ORDERS = (('native','control','optimized'),('optimized','control','native'),('control','native','optimized'))


def validator():
    path=BF/'summarize_performance.py'
    assert sha(path)=='6130e7cd9a2ffa26b42db42142694781bd45f9bdaa83336d4f1944a48368e5b7'
    spec=importlib.util.spec_from_file_location('stage34_native_timing',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    source=path.read_text()
    start=source.index("    invocation=command['command']")
    end=source.index("    runtime=entry['runtime_compatibility']",start)
    replacement="""    invocation=command['command']
    require(invocation==EXPECTED_COMMANDS[str(folder)],'exact fixed MoE outer invocation')
    require(entry['formal_entry']==EXPECTED_FORMAL_ENTRY,'original native entry')
"""
    adapted=source[:start]+replacement+source[end:]
    exec(compile(adapted,str(path),'exec'),m.__dict__)
    m.EXPECTED_COMMANDS={}
    m.EXPECTED_FORMAL_ENTRY=str(BF.parent/'megatron-native-5be9626/pretrain_gpt.py')
    return m,{'original_sha256':sha(path),'adapted_sha256':hashlib.sha256(adapted.encode()).hexdigest(),
              'replacement':'Only direct-launch argv gate replaced by exact outer command equality; all source/model/precision/native interval checks retained.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',required=True);p.add_argument('--audit',required=True)
    a=p.parse_args();assert a.version.isdigit()
    output=ROOT/('performance-'+a.version+'.json');assert not output.exists()
    candidate=ROOT/'candidates'/a.version/'candidate.py'
    contract=read(ROOT/'contract.json');audit_path=ROOT/a.audit;audit=read(audit_path)
    assert audit['passed'] is True and audit['candidate_sha256']==sha(candidate)
    assert audit['contract_sha256']==sha(ROOT/'contract.json')
    probe=read(ROOT/('probe-'+a.version+'.json'))
    assert probe['status']=='passed' and probe['candidate_sha256']==sha(candidate) and probe['verifier_sha256']==sha(ROOT/'probe.py')
    for row in audit['comparisons'].values():
        path=ROOT/row['path'];assert sha(path)==row['sha256'] and read(path)['passed'] is True
        for item in read(path)['input_files']+read(path)['route_input_files']:
            path=Path(item['path']);assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
    offset=read(ROOT/('offset-'+a.version+'.json'))
    assert offset['passed'] is True and offset['candidate_sha256']==sha(candidate) and offset['verifier_sha256']==sha(ROOT/'probe_offset.py')
    assert audit['controller_sha256']=={n:sha(ROOT/n) for n in ('launch_weighted.py','run_case.py','compare.py','probe.py','probe_offset.py','seal_audit.py','run_performance.py')}
    m,identity=validator()
    report={'schema':'moe-weighted-stage34-three-arm-performance-v1','status':'running','version':a.version,
        'candidate_sha256':sha(candidate),'contract_sha256':sha(ROOT/'contract.json'),
        'audit_path':str(audit_path),'audit_sha256':sha(audit_path),'probe_sha256':sha(ROOT/('probe-'+a.version+'.json')),
        'runner_sha256':sha(__file__),'validator':identity,'seeds':contract['performance']['independent_seeds'],'pairs':[],
        'completed_cases':[],'promotion_rule':contract['promotion_rule'],
        'metric':'Unchanged native GPU-synchronized time.time120 steps/first20 excluded/ten retained interval10 means',
        'scope':contract['scope'],'model':contract['model'],'started_utc':datetime.now(timezone.utc).isoformat()}
    try:
        for seed,order in zip(report['seeds'],ORDERS):
            pair={'seed':seed,'order':list(order),'cases':{}}
            for arm in order:
                name=arm+'-perf-'+a.version+'-'+str(seed)
                command=[sys.executable,str(ROOT/'run_case.py'),'--case',name,'--arm',arm,'--steps','120','--seed',str(seed)]
                if arm=='optimized':command+=['--candidate',str(candidate)]
                subprocess.run(command,check=True)
                folder=ROOT/name;record=read(folder/'command.json')
                expected=[sys.executable,'-u',str(ROOT/'launch_weighted.py'),
                    '--weighted-receipt',str(folder/'weighted.json'),'--bf16-root',str(BF),
                    '--gradnorm-receipt',str(folder/'gradnorm.json'),'--megatron-root',str(BF.parent/'megatron-native-5be9626'),
                    '--contract',str(ROOT/'contract.json'),'--entry-evidence',str(folder/'entry.json')]
                if arm!='native':expected+=['--candidate',str(BF/'candidates/0008/candidate.py'),'--gradnorm-candidate','/tmp/kda-gradnorm-stage26/candidate.py']
                if arm=='optimized':expected+=['--weighted-candidate',str(candidate)]
                expected+=contract['training_argv']+['--train-iters','120','--seed',str(seed),'--log-interval','10']
                assert record['wrapper_sha256']==sha(ROOT/'launch_weighted.py') and record['runner_sha256']==sha(ROOT/'run_case.py')
                m.EXPECTED_COMMANDS[str(folder)]=expected
                checked=m.validate_case(folder,'native' if arm=='native' else 'optimized',seed,contract,sha(ROOT/'contract.json'),contract['candidate_sha256'])
                weighted,grad=read(folder/'weighted.json'),read(folder/'gradnorm.json')
                assert not any((folder/d).exists() for d in ('snapshots','activation','routing','profile'))
                assert weighted['wrapper_sha256']==sha(ROOT/'launch_weighted.py')
                assert weighted['calls']==contract['model']['num_layers']*contract['model']['num_moe_experts']*120 and weighted['sort_compat_calls']==contract['model']['num_layers']*120
                assert weighted['router_calls']==0 and weighted['expert_calls']==weighted['configurations']==[]
                assert weighted['enabled'] is (arm=='optimized')
                assert weighted['candidate_calls']==(contract['model']['num_layers']*contract['model']['num_moe_experts']*120 if arm=='optimized' else 0)
                assert weighted['native_calls']==(0 if arm=='optimized' else contract['model']['num_layers']*contract['model']['num_moe_experts']*120) and weighted['fallback_calls']==0
                assert weighted['candidate_sha256']==(sha(candidate) if arm=='optimized' else None)
                assert weighted['native_compiled_callables']=={'weighted_swiglu':True,'weighted_swiglu_back':True}
                assert grad['enabled'] is (arm!='native') and grad['calls']==(120 if arm!='native' else 0)
                assert grad['fallback_calls']==0 and grad['norm_audit']==[]
                checked['stage34_arm']=arm
                checked['dispatch_sha256']={n:sha(folder/n) for n in ('weighted.json','gradnorm.json')}
                pair['cases'][arm]=checked
                row={'arm':arm,'seed':seed,'case':name,'mean_iteration_ms':checked['mean_iteration_ms']}
                report['completed_cases'].append(row);print(json.dumps(row),flush=True)
            for prev,nxt in zip(order,order[1:]):
                assert datetime.fromisoformat(pair['cases'][prev]['finished_utc'])<=datetime.fromisoformat(pair['cases'][nxt]['started_utc'])
            assert all(pair['cases'][arm]['model']==pair['cases']['native']['model'] for arm in order)
            report['pairs'].append(pair)
        for metric,left,right in (('vs_native','native','optimized'),('vs_control','control','optimized'),
                                  ('control_vs_native','native','control')):
            ratios=[pair['cases'][left]['mean_iteration_ms']/pair['cases'][right]['mean_iteration_ms'] for pair in report['pairs']]
            ratio=math.exp(statistics.mean(math.log(r) for r in ratios))
            report[metric]={'ratio':ratio,'gain_percent':100*(ratio-1),'all_seeds_faster':all(r>1 for r in ratios)}
        promoted=report['vs_control']['ratio']>1.01 and report['vs_control']['all_seeds_faster']
        report.update(status='validated',promoted=promoted,decision='promoted_tested_model_only' if promoted else 'not_promoted')
    except Exception as e:
        report.update(status='rejected',error_type=type(e).__name__,error=str(e))
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','decision','vs_control','vs_native','error')}))
    return 0 if report['status']=='validated' else 1


if __name__=='__main__':
    raise SystemExit(main())

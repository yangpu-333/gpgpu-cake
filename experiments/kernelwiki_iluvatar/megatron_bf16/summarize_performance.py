"""Validate and summarize three independent native BF16 formal-loop pairs.

Native GPU-synchronized interval time.time is the primary training metric.
Imports the unchanged FP32 log parser only for the common 120-step log grammar.
It never imports candidate code or changes tolerances or training behavior.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parent
FORMAL=ROOT.parent/'megatron_training_entry'
sys.path.insert(0,str(FORMAL))
from summarize_performance import parse_intervals as parse_native_intervals


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value,message):
    if not value:
        raise ValueError(message)


def integration():
    result={name:sha(ROOT/name) for name in ('launch_training.py','kernel_adapter.py','run_case.py')}
    result.update({'formal/'+name:sha(FORMAL/name) for name in ('runtime_compat.py','source_manifest.json')})
    return result


def validate_case(folder,arm,seed,contract,contract_sha,candidate_sha):
    command=json.loads((folder/'command.json').read_text())
    entry=json.loads((folder/'entry.json').read_text())
    require(command['mode']==arm and command['seed']==seed and command['steps']==120,'arm/seed/steps differ')
    require(command['case']==folder.name,'case name differs')
    require(command['audit'] is False and command['profile'] is False,'diagnostic instrumentation in performance case')
    require(command['exit_code']==0 and entry['status']=='completed','process did not finish')
    require(command['contract_sha256']==contract_sha==entry['contract_sha256'],'contract hash differs')
    manifest=json.loads((FORMAL/'source_manifest.json').read_text())
    require(entry['native_sources_sha256']==manifest['files'],'native source differs')
    require(entry['megatron_commit']==contract['megatron_commit']==manifest['commit'],'native commit differs')
    hashes=integration()
    require(entry['integration_sha256']==hashes and entry['launcher_sha256']==hashes['launch_training.py'],'integration source differs')
    argv=contract['training_argv']+['--train-iters','120','--seed',str(seed),'--log-interval','10']
    require(entry['training_argv']==argv,'training argv differs')
    invocation=command['command']
    require(len(invocation)>9 and invocation[1]=='-u' and invocation[2].endswith('/launch_training.py'),'wrong launcher')
    prefix=['--megatron-root',invocation[4],'--contract',invocation[6],'--entry-evidence',invocation[8]]
    require(invocation[8].endswith('/'+folder.name+'/entry.json'),'wrong evidence path')
    if arm=='optimized':
        require(len(invocation)>10,'candidate missing')
        prefix+=['--candidate',invocation[10]]
    require(invocation[3:]==prefix+argv,'unexpected invocation flags')
    require(entry['formal_entry']==invocation[4]+'/pretrain_gpt.py','not original pretrain entry')
    runtime=entry['runtime_compatibility']
    require(runtime['torch']=='2.4.1' and runtime['optional_backends_disabled_in_process']==['transformer_engine','apex'],'wrong runtime')
    require(entry['profile_hooks']==[] and 'profiler_activities' not in entry,'profiler active')
    model=contract['model']
    require(len(entry['models'])==1,'model chunk count differs')
    actual=entry['models'][0]
    expected={'class':'megatron.core.models.gpt.gpt_model.GPTModel','parameters':model['parameters'],
              'layers':model['num_layers'],'hidden_size':model['hidden_size'],'vocab_size':model['vocab_size'],
              'bf16':True,'fp16':False}
    require(all(actual.get(k)==v for k,v in expected.items()),'constructed model differs')
    adapters=entry['kernel_adapters']
    require(entry['iluvatar_kernels'] is (arm=='optimized'),'opt-in differs')
    if arm=='native':
        require(adapters==[],'native has candidate adapter')
    else:
        require(len(adapters)==1,'candidate adapter count differs')
        handle=adapters[0]
        require(all(handle[k] is True for k in ('active','enabled','provenance_verified')),'adapter inactive')
        require(handle['candidate_sha256']==handle['candidate_expected_sha256']==candidate_sha,'candidate hash differs')
        require(handle['runtime_world_size']==1 and handle['verified_devices'] in ({'0':'BI-V150'},{'0':'Iluvatar BI-V150'}),'wrong GPU/TP scope')
        require(handle['counts']=={'ce_candidate_calls':120,'ce_native_calls':0,
                                  'residual_native_calls':model['num_layers']*120,'fallback_reasons':{}},'candidate coverage differs')
        cfg=handle['config_at_installation']
        require(cfg['bf16'] is True and cfg['fp16'] is False and cfg['normalization']=='RMSNorm','wrong precision config')
        for observation in handle['observations']:
            require(observation['logits_dtype']=='torch.bfloat16' and observation['loss_dtype']=='torch.float32'
                    and observation['autocast'] is False and observation['logits_shape']==[model['sequence_length'],model['micro_batch_size'],model['vocab_size']],
                    'wrong observed CE dtype/shape')
        require(len(handle['observations'])==4,'missing observed CE calls')
    records=parse_native_intervals((folder/'stdout.log').read_text())
    mean=statistics.mean(row['mean_iteration_ms'] for row in records if row['retained'])
    return {'case':folder.name,'mode':arm,'seed':seed,'mean_iteration_ms':mean,
            'tokens_per_second':model['sequence_length']*model['global_batch_size']*1000/mean,
            'intervals':records,'whole_process_seconds':command['whole_process_elapsed_seconds'],
            'started_utc':command['started_utc'],'finished_utc':command['finished_utc'],
            'model':actual,'adapter':adapters[0] if adapters else None,
            'raw_sha256':{name:sha(folder/name) for name in ('command.json','entry.json','stdout.log','stderr.log')}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--prefix',required=True,help='case names are ARM-PREFIX-perf-SEED')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('immutable output exists')
    report={'status':'initializing','harness_sha256':sha(__file__),
            'log_parser_sha256':sha(FORMAL/'summarize_performance.py'),'pairs':[],
            'metric':'Native GPU-synchronized time.time training intervals; mean of ten log-interval10 records after first20steps. No tensor audit/profile/checkpoint/eval.',
            'scope':'True BF16 native MockGPTDataset/local Torch/single BI-V150, CE-only candidate. Three-step tensor correctness is a separate gate; interval scalar logs do not prove 120-step tensor equivalence.'}
    try:
        contract=json.loads(args.contract.read_text())
        candidate_sha=sha(args.candidate)
        require(contract['candidate_sha256']==candidate_sha,'candidate contract differs')
        require(contract['performance']['steps']==120,'unsupported steps')
        require(contract['model']['micro_batch_size']==2 and contract['model']['global_batch_size']==2,'fixed parser batch2 required')
        seeds=contract['performance']['independent_seeds']
        require(len(seeds)==3 and len(set(seeds))==3,'three independent seeds required')
        report.update(candidate_sha256=candidate_sha,contract_sha256=sha(args.contract),model=contract['model'])
        for index,seed in enumerate(seeds):
            arms={arm:validate_case(ROOT/'evidence'/('%s-%s-perf-%d'%(arm,args.prefix,seed)),arm,seed,
                  contract,sha(args.contract),candidate_sha) for arm in ('native','optimized')}
            n,o=arms['native'],arms['optimized']
            require(n['model']==o['model'],'pair constructed models differ')
            first,second=(n,o) if index%2==0 else (o,n)
            require(datetime.fromisoformat(first['finished_utc'])<=datetime.fromisoformat(second['started_utc']),
                    'pair overlapped or expected alternation differs')
            diffs=[abs(a['lm_loss']-b['lm_loss']) for a,b in zip(n['intervals'],o['intervals'])]
            report['pairs'].append({'seed':seed,'arms':arms,'throughput_ratio':n['mean_iteration_ms']/o['mean_iteration_ms'],
                                   'maximum_interval_scalar_loss_difference':max(diffs)})
        ratio=math.exp(statistics.mean(math.log(pair['throughput_ratio']) for pair in report['pairs']))
        report.update(status='validated',geomean_throughput_ratio=ratio,throughput_gain_percent=100*(ratio-1),
                      all_pairs_faster=all(p['throughput_ratio']>1 for p in report['pairs']))
    except Exception as error:
        report.update(status='rejected',error=str(error))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report.get(k) for k in ('status','geomean_throughput_ratio','throughput_gain_percent','error')}))
    return 0 if report['status']=='validated' else 1


if __name__=='__main__':
    raise SystemExit(main())

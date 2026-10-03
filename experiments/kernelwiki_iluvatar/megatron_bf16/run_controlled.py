"""Three-arm formal BF16 trial: native, accepted0008 control, candidate0009.

Numerical gates and all timed training code remain unchanged. Controls use the
same seed/model/argv on the same GPU in sequential fresh processes.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=('main','wide','deep'),required=True)
    parser.add_argument('--megatron-root',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--operator',type=Path,required=True)
    args=parser.parse_args()
    scope=args.scope
    candidate=ROOT/'candidates/0009/candidate.py'
    control=ROOT/'candidates/0008/candidate.py'
    contract_path=ROOT/'contracts'/('0009-'+scope+'.json')
    control_contract_path=ROOT/'contracts'/('0008-'+scope+'.json')
    contract=json.loads(contract_path.read_text())
    control_contract=json.loads(control_contract_path.read_text())
    audit=json.loads(args.audit.read_text())
    operator=json.loads(args.operator.read_text())
    assert contract['candidate_sha256']==sha(candidate) and control_contract['candidate_sha256']==sha(control)
    assert contract['training_argv']==control_contract['training_argv'] and contract['model']==control_contract['model']
    assert contract['performance']['independent_seeds']==control_contract['performance']['independent_seeds']
    assert audit['passed'] is True and audit['check_summary']['failed']==0
    assert audit['comparator_sha256']==sha(ROOT/'compare_training.py')
    assert any(row['sha256']==sha(contract_path) for row in audit['input_files'])
    assert any(row['sha256']==sha(candidate) for row in audit['input_files'])
    assert operator['status']=='passed' and operator['candidate_sha256']==sha(candidate) and operator['passed_cases']==504
    assert operator['harness_sha256']==sha(ROOT/'verify_operator.py')
    previous=json.loads((ROOT/'evidence'/('0008-'+scope+'-performance.json')).read_text())
    assert previous['status']=='validated' and previous['candidate_sha256']==sha(control)
    orders=(('native','control','optimized'),('optimized','control','native'),('control','native','optimized'))
    prefix='0009-'+scope
    for index,seed in enumerate(contract['performance']['independent_seeds']):
        for arm in orders[index]:
            name=('optimized-'+prefix+'-control-perf-'+str(seed)) if arm=='control' else arm+'-'+prefix+'-perf-'+str(seed)
            path=control_contract_path if arm=='control' else contract_path
            command=[sys.executable,str(ROOT/'run_case.py'),name,'--mode','native' if arm=='native' else 'optimized',
                '--contract',str(path),'--megatron-root',str(args.megatron_root.resolve()),'--steps','120','--seed',str(seed)]
            if arm!='native':
                command+=['--candidate',str(control if arm=='control' else candidate)]
            subprocess.run(command,check=True)
    subprocess.run([sys.executable,str(ROOT/'summarize_performance.py'),'--contract',str(contract_path),
        '--candidate',str(candidate),'--prefix',prefix,'--output',str(ROOT/'evidence'/(prefix+'-performance.json'))],check=True)
    spec=importlib.util.spec_from_file_location('bf16_three_arm_reader',ROOT/'summarize_performance.py')
    reader=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    summary=json.loads((ROOT/'evidence'/(prefix+'-performance.json')).read_text())
    controls=[]
    for pair in summary['pairs']:
        seed=pair['seed']
        folder=ROOT/'evidence'/('optimized-'+prefix+'-control-perf-'+str(seed))
        observed=reader.validate_case(folder,'optimized',seed,control_contract,sha(control_contract_path),sha(control))
        optimized=pair['arms']['optimized']
        require_order=orders[len(controls)]
        times={'native':pair['arms']['native'],'control':observed,'optimized':optimized}
        for a,b in zip(require_order,require_order[1:]):
            assert times[a]['finished_utc']<=times[b]['started_utc']
        controls.append({'seed':seed,'control':observed,
            'candidate_vs_control_throughput_ratio':observed['mean_iteration_ms']/optimized['mean_iteration_ms'],
            'control_vs_same_native_ratio':pair['arms']['native']['mean_iteration_ms']/observed['mean_iteration_ms']})
    ratio=math.exp(statistics.mean(math.log(row['candidate_vs_control_throughput_ratio']) for row in controls))
    result={'status':'validated','candidate_sha256':sha(candidate),'control_sha256':sha(control),
            'harness_sha256':sha(__file__),'scope':contract['model'],'orders':orders,'controls':controls,
            'candidate_vs_control_geomean':ratio,'all_seeds_faster_than_control':all(r['candidate_vs_control_throughput_ratio']>1 for r in controls),
            'native_geomean':summary['geomean_throughput_ratio'],
            'metric':'Three sequential fresh process arms per seed; same formal native interval timer, model, data, optimizer and source hashes. Not confidence interval evidence.'}
    output=ROOT/'evidence'/(prefix+'-controlled.json')
    assert not output.exists()
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','candidate_vs_control_geomean','all_seeds_faster_than_control','native_geomean')}))


if __name__=='__main__':
    main()

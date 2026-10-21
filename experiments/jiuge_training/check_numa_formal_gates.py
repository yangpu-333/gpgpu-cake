"""Synthetic admission-gate regression only; never measured performance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
from analyze_numa_formal import analyze

BASE=Path(__file__).resolve().parent


def main():
    results=BASE/'results'; checks=[]
    with zipfile.ZipFile(results/'pinning-driver-20261016-05/pinning-formal-key-results-20261016.zip') as archive:
        template_case=json.loads(archive.read('runs/pinning-formal-primary-63100-optimized/case.json'))
        entries=[json.loads(archive.read(f'runs/pinning-formal-primary-63100-optimized/rank-{rank}/entry.json')) for rank in (0,1)]
    hashes={name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in ('numa_entry.py','numa_placement.py')}
    admitted={'frozen_source_sha256':hashes}
    with tempfile.TemporaryDirectory(prefix='numa-formal-gate-',dir=results) as directory:
        root=Path(directory).resolve()
        if not root.is_relative_to(results.resolve()):
            raise ValueError('Temporary fixture cleanup target escapes workspace results')
        for seed in (63100,63101,63102):
            for arm in ('native','control','optimized'):
                folder=root/'runs'/f'cpu-numa-formal-primary-{seed}-{arm}'; folder.mkdir(parents=True)
                case=deepcopy(template_case); case.update(case=folder.name,numa_arm=arm,pinning_arm='optimized',seed=seed)
                (folder/'case.json').write_text(json.dumps(case),encoding='utf-8')
                for rank,template in enumerate(entries):
                    data=deepcopy(template)
                    for interval in data['raw_interval_times']:
                        interval['seconds']=32.0 if arm=='optimized' else 40.0
                    (folder/f'rank-{rank}').mkdir(); (folder/f'rank-{rank}/entry.json').write_text(json.dumps(data),encoding='utf-8')
                    if arm!='native':
                        cpus=list(range(8)) if rank==0 else list(range(16,24))
                        receipt={'mode':'local8' if arm=='optimized' else 'unbound','source_sha256':hashes,
                                 'allowed_before':list(range(64)),'allowed_after':cpus if arm=='optimized' else list(range(64)),
                                 'selected_local_physical_core_cpus':cpus}
                        (folder/f'numa-binding-rank-{rank}.json').write_text(json.dumps(receipt),encoding='utf-8')
        valid=analyze(root,'primary',admitted)
        assert abs(valid['vs_control']['geomean_throughput_ratio']-1.25)<1e-12 and valid['round_promotion_passed']
        checks.append('Complete pinned baseline/control/local8 fixture accepted with the unchanged statistics')
        entry=root/'runs/cpu-numa-formal-primary-63100-control/rank-0/entry.json'
        original=json.loads(entry.read_text(encoding='utf-8'))
        for name,mutate in (
            ('A pageable control is rejected',lambda x:x['actual_final_cpu_masters'].update(pinned_groups=0)),
            ('A hidden profiler is rejected',lambda x:x.update(profiler_enabled=True)),
            ('An actual optimizer at step119 is rejected',lambda x:x['actual_optimizer_state']['optimizers'][0].update(min_step=119)),
            ('Missing native timer intervals are rejected',lambda x:x['raw_interval_times'].pop()),
        ):
            data=deepcopy(original); mutate(data); entry.write_text(json.dumps(data),encoding='utf-8')
            try:
                analyze(root,'primary',admitted)
            except ValueError:
                checks.append(name)
            else:
                raise AssertionError(name)
            entry.write_text(json.dumps(original),encoding='utf-8')
        binding=root/'runs/cpu-numa-formal-primary-63100-control/numa-binding-rank-0.json'
        data=json.loads(binding.read_text(encoding='utf-8')); data['allowed_after']=[0]
        binding.write_text(json.dumps(data),encoding='utf-8')
        try:
            analyze(root,'primary',admitted)
        except ValueError:
            checks.append('A control that changes affinity is rejected')
        else:
            raise AssertionError('Changed control affinity accepted')
    output={'schema':'jiuge7b-cpu-numa-synthetic-gate-preflight-v1','passed':True,'checks':checks,
            'synthetic_timing_fixture':True,'formal_performance_evidence':False}
    (results/'cpu-numa-formal-gate-preflight-20261016.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()

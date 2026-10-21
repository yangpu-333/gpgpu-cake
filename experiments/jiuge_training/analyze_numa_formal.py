"""Apply unchanged timing/statistics rules to fresh pinned-baseline arms."""
import hashlib
import json
from pathlib import Path
from analyze_pinning_formal import summarize,evaluate,SEEDS,ARMS


def analyze(root,round_name,admission):
    cases=[]; means={seed:{} for seed in SEEDS}
    for seed in SEEDS:
        for arm in ARMS:
            folder=root/'runs'/f'cpu-numa-formal-{round_name}-{seed}-{arm}'
            case=json.loads((folder/'case.json').read_text())
            if (case['exit_code'] or case['resource_gate_before'].strip() or case['resource_gate_after'].strip() or
                not case['formal_performance_run'] or case['seed']!=seed or case['numa_arm']!=arm or case['pinning_arm']!='optimized'):
                raise ValueError('Formal case identity, success or GPU occupancy differs')
            if case['environment']['OMP_NUM_THREADS']!='8':
                raise ValueError('CPU thread allocation changed')
            ranks=[json.loads((folder/f'rank-{rank}/entry.json').read_text()) for rank in (0,1)]
            for record in ranks:
                loaded=record['weight_loading']; master=record['actual_final_cpu_masters']
                if not loaded['all_source_tensors_and_native_parameters_covered'] or not loaded['all_loaded_shard_bytes_verified']:
                    raise ValueError('Actual 7B weight loading incomplete')
                if master!={'groups':199,'elements':3834518016,'pinned_groups':199}:
                    raise ValueError('All new arms must preserve accepted pinning repair')
                actual=record['actual_optimizer_state']['optimizers']
                if len(actual)!=1 or actual[0]['adam_state_entries']!=199 or actual[0]['min_step']!=120 or actual[0]['max_step']!=120:
                    raise ValueError('Actual AdamW state coverage or step differs')
                binding_path=folder/f"numa-binding-rank-{record['rank']}.json"
                if arm=='native':
                    if binding_path.exists():
                        raise ValueError('Native pinned baseline must bypass affinity wrapper')
                else:
                    binding=json.loads(binding_path.read_text())
                    if binding['mode']!=('local8' if arm=='optimized' else 'unbound'):
                        raise ValueError('Actual affinity dispatch differs')
                    for name,digest in binding['source_sha256'].items():
                        if admission['frozen_source_sha256'][name]!=digest:
                            raise ValueError('Formal affinity differs from admitted implementation')
                    if arm=='optimized' and binding['allowed_after']!=binding['selected_local_physical_core_cpus']:
                        raise ValueError('Actual local8 affinity incomplete')
                    if arm=='control' and binding['allowed_before']!=binding['allowed_after']:
                        raise ValueError('No-op control changed affinity')
            summary=summarize(ranks); means[seed][arm]=summary['mean_seconds_per_step']
            cases.append({'case':case['case'],'seed':seed,'arm':arm,
                          'case_sha256':hashlib.sha256((folder/'case.json').read_bytes()).hexdigest(),**summary})
    return {'schema':'jiuge7b-cpu-numa-formal-round-v1','round':round_name,'cases':cases,**evaluate(means),
            'baseline':'Accepted pinned optimizer with original unbound affinity; all new measurements',
            'formal_performance_passed':False,'all_expected_cases_verified':True}

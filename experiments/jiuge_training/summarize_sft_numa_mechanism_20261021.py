"""Summarize closed trace-analysis keys only; never reopen raw GPU traces."""
import hashlib
import json
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent
RESULTS = BASE/'results'
KEYS = RESULTS/'sft-numa-driver-20261021-02/keys'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    review = read(RESULTS/'sft-numa-independent-review-v2-20261021.json')
    if not review['passed'] or review['formal_performance_passed']:
        raise ValueError('Closed diagnostic review required')
    rows, sources = [], {}
    for case in review['cases']:
        root = KEYS/'runs'/case['case']
        path = root/'symmetric-mechanism.json'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        archive = read(root/'archive-manifest.json')
        if digest != archive['files'][path.name]['sha256']:
            raise ValueError('Closed mechanism key SHA differs')
        sources[case['case']] = digest
        for window in read(path)['windows']:
            copies = window['copy_api_review']['copy_groups']
            selected = {}
            for label, name in (('HtoD', 'Memcpy HtoD (Pinned -> Device)'),
                                ('DtoH', 'Memcpy DtoH (Device -> Pinned)')):
                matching = [x for x in copies if x['copy_name'] == name]
                if len(matching) != 1:
                    raise ValueError('Exactly one pinned bulk copy group required')
                copy = matching[0]
                if (copy['unmatched_gpu_copy_events'] or copy['ambiguous_gpu_copy_events']
                        or copy['copy_events_without_recorded_size']
                        or len(copy['gpu_union_us_by_device']) != 1):
                    raise ValueError('Single-device correlated copy coverage incomplete')
                selected.update({label+'_GPU_seconds_per_rank_step':
                                    next(iter(copy['gpu_union_us_by_device'].values()))/5e6,
                                 label+'_CPU_API_seconds_per_rank_step':copy['cpu_api_union_clipped_us']/5e6,
                                 label+'_recorded_GiB_per_rank_step':copy['recorded_bytes']/5/1024**3})
            collective = window['gpu_union_us']['collective']
            rows.append({'case':case['case'], 'policy':case['policy'], 'rank':window['rank'],
                         'window':window['window'], **selected,
                         'NCCL_overlap_compute_fraction':window['compute_collective_overlap_us']/collective,
                         'NCCL_overlap_memcpy_seconds_per_rank_step':window['collective_memcpy_overlap_us']/5e6})
    if len(rows) != 24:
        raise ValueError('All24 rank-window keys required')
    metrics = [k for k in rows[0] if k not in ('case', 'policy', 'rank', 'window')]
    policies = {}
    for policy in review['policies']:
        observations = [r for r in rows if r['policy'] == policy]
        if len(observations) != 8:
            raise ValueError('Exactly eight correlated observations per policy required')
        policies[policy] = {metric:{'mean':statistics.mean(r[metric] for r in observations),
                                   'min':min(r[metric] for r in observations),
                                   'max':max(r[metric] for r in observations)} for metric in metrics}
    value = {'schema':'jiuge7b-sft-numa-mechanism-key-summary-v1', 'passed':True,
             'GPU_used':False, 'raw_traces_downloaded_or_reparsed':False,
             'formal_performance_evidence':False, 'native_collective_sequence_verified':False,
             'critical_path_proven':False, 'physical_wire_bytes_measured':False,
             'CPU_API_is_exposed_wait':False, 'categories_are_additive':False,
             'scope':'Pinned bulk HtoD/DtoH copy groups; 2runs/policy, one seed, eight correlated rank-window observations',
             'source_mechanism_SHA256':sources, 'policies':policies, 'windows':rows,
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (RESULTS/'sft-numa-mechanism-summary-20261021.json').open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2)
    print(json.dumps({'passed':True,'policies':policies},indent=2))


if __name__ == '__main__':
    main()

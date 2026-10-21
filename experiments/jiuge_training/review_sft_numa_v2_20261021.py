"""Independent bounded-key review; all numerical originals stay on ACT."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
from sft_tp_reader_gate_20261021 import check_rank_rows

BASE = Path(__file__).resolve().parent
DRIVER = BASE/'results/sft-numa-driver-20261021-02'
KEYS = DRIVER/'keys'
CAMPAIGN = KEYS/'sft-numa-diagnostic-20261021-02'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def pages(data, binding):
    query = data['page_queries']
    if query.get('return_code') != 0 or data['torch_intraop_threads'] != 8:
        raise ValueError('Actual page query or unchanged CPU thread count failed')
    groups = {}
    for tensor in query['tensors']:
        name = tensor['field']
        label = 'master' if name.startswith('master/') else name.rsplit('/', 1)[-1]
        group = groups.setdefault(label, {'tensors': 0, 'elements': 0, 'pinned_tensors': 0,
                                          'sample_nodes': Counter(), 'gpu_local_weighted_sum': 0.,
                                          'elements_with_valid_samples': 0})
        group['tensors'] += 1
        group['elements'] += tensor['numel']
        group['pinned_tensors'] += int(tensor['is_pinned'])
        start, count = tensor['sample_start'], tensor['sample_count']
        locations = query['sample_nodes'][start:start+count]
        if len(locations) != count or not count or any(n < 0 for n in locations):
            raise ValueError('Missing or invalid actual tensor page samples')
        group['sample_nodes'].update(map(str, locations))
        group['elements_with_valid_samples'] += tensor['numel']
        group['gpu_local_weighted_sum'] += tensor['numel']*sum(
            n == binding['assigned_gpu']['node'] for n in locations)/count
    for label, group in groups.items():
        group['sample_nodes'] = dict(group['sample_nodes'])
        group['element_weighted_GPU_local_sample_fraction'] = group.pop('gpu_local_weighted_sum')/group['elements_with_valid_samples']
        if label in ('master', 'gradient', 'exp_avg', 'exp_avg_sq'):
            if (group['tensors'] != 199 or group['elements'] != 3834518016
                    or group['elements_with_valid_samples'] != 3834518016):
                raise ValueError('Actual master/gradient/Adam moment sample coverage incomplete')
    if not {'master', 'gradient', 'exp_avg', 'exp_avg_sq'}.issubset(groups):
        raise ValueError('Required optimizer buffers absent')
    return groups


def main():
    checked = read(DRIVER/'local-key-verification.json')
    if not checked['passed'] or checked['verified_key_bytes'] > 32*1024**2:
        raise ValueError('Bounded local key SHA gate missing')
    bundle = DRIVER/'sft-numa-key-results-v2.zip'
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != checked['bundle_sha256']:
        raise ValueError('Closed local key bundle differs')
    state = read(CAMPAIGN/'controller.json')
    approval = read(CAMPAIGN/'approval.json')
    plan = read(CAMPAIGN/'affinity-plan.json')
    contract = read(BASE/'sft_numa_v2_20261021.contract.json')
    plan_sha = hashlib.sha256((CAMPAIGN/'affinity-plan.json').read_bytes()).hexdigest()
    expected_order = contract['order_with_two_distinct_candidates' if 'socket-split8' in plan['policies']
                              else 'order_with_one_distinct_candidate']
    actual_order = [x['policy']+'-'+x['tag'] for x in state['cases']]
    if (state['status'] != 'completed' or actual_order != expected_order
            or state['verified_raw_traces'] != 4*len(expected_order)
            or approval['max_charge_points'] != 0 or approval['startup_limit_seconds'] != 10800):
        raise ValueError('Complete ordered, capped diagnostic required')
    manifest = read(BASE/'results/sft-numa-source-manifest-v2-20261021.json')['files']
    if manifest != approval['source_files_sha256']:
        raise ValueError('Exact frozen source manifest differs')
    for name, sha in manifest.items():
        if hashlib.sha256((CAMPAIGN/'source'/name).read_bytes()).hexdigest() != sha:
            raise ValueError('Incomplete frozen source bundle: '+name)
    reference = None
    cases, windows, snapshots, originals = [], [], [], []
    for item in state['cases']:
        root = KEYS/'runs'/item['case']
        for label in ('gate_before', 'gate_after'):
            gate = item[label]
            if (gate['compute_apps'].strip() or
                    [r.split(',')[0].strip() for r in gate['cards'].strip().splitlines()] != approval['ordered_gpu_uuids']):
                raise ValueError('Assigned pair or other-compute gate failed')
        case = read(root/'case.json')
        completion = read(root/'completion-verification.json')
        archive = read(CAMPAIGN/(item['case']+'-archive-verification.json'))
        archive_payload = (root/'archive-manifest.json').read_bytes()
        if (case['exit_code'] or not completion['passed'] or not archive['passed']
                or hashlib.sha256(archive_payload).hexdigest() != archive['manifest_sha256']):
            raise ValueError('Training/ACT raw archive verification incomplete')
        archive_manifest = json.loads(archive_payload)
        samples = read(root/'sft-supervision-rank-0.json')['rows']
        losses = completion['native_losses']
        if len(samples) != 35 or len(losses) != 35:
            raise ValueError('All35 samples and logged losses required')
        if reference is None:
            reference = (samples, losses)
        elif reference != (samples, losses):
            raise ValueError('Paired SFT samples/mask counts/logged losses differ')
        for rank in (0, 1):
            binding = read(root/f'sft-numa-binding-rank-{rank}.json')
            pin = read(CAMPAIGN/('candidate-'+item['tag'])/f'optimized-rank-{rank}-actual-pinning-init-1.json')
            entry = read(root/f'rank-{rank}/entry.json')
            objective = read(root/f'sft-trace-objective-rank-{rank}.json')
            check_rank_rows(rank, read(root/f'sft-supervision-rank-{rank}.json')['rows'], samples)
            if (binding['allowed_after'] != plan['policies'][item['policy']][rank]
                    or binding['mode'] != item['policy'] or binding['plan_sha256'] != plan_sha
                    or pin['parameter_count'] != 199 or pin['pinned_count'] != 199
                    or entry['native_files_checked'] != 600
                    or [x['step'] for x in entry['completed_native_steps']] != list(range(1, 36))
                    or any(x['skipped'] for x in entry['completed_native_steps'])
                    or not objective['run_returned']
                    or objective['range_call_counts']['jiuge/cpu_adamw_math'] != 6965):

                raise ValueError('Actual affinity/pinning/native/SFT/update/range coverage differs')
            for phase in ('after-first-update', 'after-training'):
                path = root/f'sft-numa-{phase}-rank-{rank}.json'
                if hashlib.sha256(path.read_bytes()).hexdigest() != archive_manifest['files'][path.name]['sha256']:
                    raise ValueError('NUMA sample is not the ACT archived original')
                snapshot = read(path)
                snapshots.append({'case': item['case'], 'policy': item['policy'], 'rank': rank,
                                  'phase': phase, 'CPU_affinity': binding['allowed_after'],
                                  'assigned_gpu_node': binding['assigned_gpu']['node'],
                                  'categories': pages(snapshot, binding)})
        mechanism = read(root/'symmetric-mechanism.json')
        trace = read(root/'trace-analysis.json')
        comm = read(root/'communication-analysis.json')
        for row in mechanism['windows']:
            review = next(x for x in trace['windows'] if x['rank'] == row['rank'] and x['window'] == row['window'])
            issues = set(review['analysis']['issues'])
            if issues-{'zero dropped events is not established'}:
                raise ValueError('Additional trace coverage issues: '+repr(issues))
            if len(review['analysis']['capture_window']['devices']) != 1:
                raise ValueError('Exactly one observed device per rank required')
            host, gpu = row['host_ranges'], row['gpu_union_us']
            if host['jiuge/cpu_adamw_math']['count'] != 995:
                raise ValueError('Complete995 AdamW ranges per rank/window required')
            pair = next(x for x in comm['cross_rank_pair_candidates'] if x['window'] == row['window'])
            windows.append({'case': item['case'], 'policy': item['policy'], 'rank': row['rank'],
                            'window': row['window'], 'profiled_seconds_per_rank_step': row['window_seconds']/5,
                            'CPU_AdamW_seconds_per_rank_step': host['jiuge/cpu_adamw_math']['seconds_per_rank_step'],
                            'CPU_optimizer_seconds_per_rank_step': host['jiuge/optimizer_update']['seconds_per_rank_step'],
                            'GPU_compute_seconds_per_rank_step': gpu['compute']/5e6,
                            'GPU_memcpy_seconds_per_rank_step': gpu['memcpy']/5e6,
                            'GPU_NCCL_seconds_per_rank_step': gpu['collective']/5e6,
                            'CPU_AdamW_overlap_HtoD_seconds_per_rank_step': row['cpu_adamw_gpu_overlap_us']['HtoD']/5e6,
                            'GPU_idle_proxy_fraction': row['gpu_idle_proxy_us']/(row['window_seconds']*1e6),
                            'maximum_collective_start_skew_candidate_ms': max(
                                [x['start_skew_us']/1000 for x in pair['largest_start_skew_candidates']], default=0),
                            'native_collective_sequence_verified': False})
            originals.append({'case': item['case'], 'window': row['window'], 'rank': row['rank'],
                              'trace_sha256': row['trace_sha256'], 'decompressed_sha256': row['decompressed_sha256']})
        rows = [x for x in windows if x['case'] == item['case']]
        slow_rank_windows = [max(x['profiled_seconds_per_rank_step'] for x in rows if x['window'] == name)
                             for name in sorted({x['window'] for x in rows})]
        cases.append({'case': item['case'], 'policy': item['policy'],
                      'diagnostic_slow_rank_mean_seconds_per_step': statistics.mean(slow_rank_windows),
                      'raw_archive_verified_files': archive['verified_file_count'],
                      'raw_archive_verified_bytes': archive['total_expected_bytes']})
    policies = {}
    metrics = [k for k, v in windows[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k != 'rank']
    for policy in plan['policies']:
        rows = [x for x in windows if x['policy'] == policy]
        if len(rows) != 8:
            raise ValueError('Two runs, both ranks and both windows per policy required')
        policies[policy] = {key: {'mean': statistics.mean(x[key] for x in rows),
                                 'min': min(x[key] for x in rows), 'max': max(x[key] for x in rows)} for key in metrics}
    baseline = [x['diagnostic_slow_rank_mean_seconds_per_step'] for x in cases if x['policy'] == 'unbound']
    screens = {}
    for policy in set(plan['policies'])-{'unbound'}:
        candidate = [x['diagnostic_slow_rank_mean_seconds_per_step'] for x in cases if x['policy'] == policy]
        screens[policy] = {'both_candidate_runs_faster_than_both_unbound_runs': max(candidate) < min(baseline),
                           'descriptive_ratio_of_means': statistics.mean(baseline)/statistics.mean(candidate),
                           'formal_performance_evidence': False}
    result = {'schema': 'jiuge7b-sft-numa-independent-diagnostic-review-v2', 'passed': True,
              'utc': datetime.now(timezone.utc).isoformat(), 'baseline': 'accepted pinned repair; fresh unbound SFT runs',
              'cases': cases, 'policies': policies, 'windows': windows, 'page_snapshots': snapshots,
              'candidate_screens': screens, 'raw_trace_index': originals,
              'paired_all35_sample_IDs_mask_counts_and_logged_losses_equal': True,
              'formal_performance_passed': False, 'new_full_state_and_resume_required': True,
              'collector_drop_count_verified': False, 'critical_path_proven': False,
              'native_collective_sequence_verified': False,
              'sampled_page_locations_are_complete_inventory': False,
              'CPU_and_GPU_categories_additive': False,
              'statistical_scope': 'Two independent runs/policy, one seed, two windows/rank; eight correlated observations, not eight seeds',
              'local_key_bundle_sha256': checked['bundle_sha256']}
    output = BASE/'results/sft-numa-independent-review-v2-20261021.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'passed': True, 'cases': cases, 'candidate_screens': screens}, indent=2))


if __name__ == '__main__':
    main()

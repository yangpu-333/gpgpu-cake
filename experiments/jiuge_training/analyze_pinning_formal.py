"""Apply the existing ten-window, three-seed promotion and CV repeat rules."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ARMS = ('native', 'control', 'optimized')
SEEDS = (63100, 63101, 63102)


def summarize(rank_records):
    times = []
    for record in rank_records:
        if record['status'] != 'completed' or record['completed_steps'] != 120 or record['skipped_steps']:
            raise ValueError('Incomplete formal training')
        rows = record['raw_interval_times']
        if [r['end_iteration'] for r in rows] != list(range(10, 121, 10)):
            raise ValueError('Exactly twelve original timer intervals required')
        if any(r['steps'] != 10 or not r['native_reset'] or not r['native_barrier'] or
               not math.isfinite(r['seconds']) or r['seconds'] <= 0 for r in rows):
            raise ValueError('Invalid original timer interval')
        if record['profiler_enabled'] or record['full_state_capture_enabled'] or record['checkpoint_io_enabled']:
            raise ValueError('Formal timing contaminated by capture/checkpoint IO')
        times.append(rows)
    if len(times) != 2 or {r['rank'] for r in rank_records} != {0, 1}:
        raise ValueError('Both actual TP ranks required')
    # Same effective work on both TP ranks. Slowest rank/window defines task time;
    # TP ranks do not multiply the effective token count.
    retained = [max(rank[index]['seconds'] for rank in times) / 10 for index in range(2, 12)]
    mean = statistics.mean(retained)
    return {'retained_interval_seconds_per_step': retained, 'mean_seconds_per_step': mean,
            'effective_tokens_per_second': 2048 / mean, 'retained_steps': 100,
            'rank_aggregation': 'max seconds per aligned 10-step window',
            'interval_cv': statistics.stdev(retained) / mean}


def evaluate(seed_means):
    if set(seed_means) != set(SEEDS) or any(set(seed_means[s]) != set(ARMS) for s in SEEDS):
        raise ValueError('All three fixed seeds and arms required')
    if any(not math.isfinite(v) or v <= 0 for row in seed_means.values() for v in row.values()):
        raise ValueError('Nonpositive/nonfinite formal result')
    ratios = {arm: [seed_means[s][arm] / seed_means[s]['optimized'] for s in SEEDS]
              for arm in ('native', 'control')}
    cvs = {arm: statistics.stdev(seed_means[s][arm] for s in SEEDS) /
                 statistics.mean(seed_means[s][arm] for s in SEEDS) for arm in ARMS}
    comparisons = {arm: {'seed_throughput_ratios': values, 'geomean_throughput_ratio': math.exp(statistics.mean(map(math.log, values))),
                         'all_seeds_faster': all(ratio > 1 for ratio in values)} for arm, values in ratios.items()}
    gain = comparisons['control']['geomean_throughput_ratio'] - 1
    return {'seed_mean_cv': cvs, 'vs_native': comparisons['native'], 'vs_control': comparisons['control'],
            'gain_vs_control': gain, 'round_promotion_passed': comparisons['control']['geomean_throughput_ratio'] > 1.01 and comparisons['control']['all_seeds_faster'],
            'repeat_required': any(cvs[arm] >= .1 for arm in ('native', 'control'))}


def analyze(root, round_name):
    cases = []
    means = {seed: {} for seed in SEEDS}
    for seed in SEEDS:
        for arm in ARMS:
            folder = root / 'runs' / f'pinning-formal-{round_name}-{seed}-{arm}'
            case = json.loads((folder / 'case.json').read_text())
            if case['exit_code'] != 0 or case['resource_gate_before'].strip() or case['resource_gate_after'].strip():
                raise ValueError('Failed case or unresolved GPU occupancy')
            if not case['formal_performance_run'] or case['seed'] != seed or case['pinning_arm'] != arm:
                raise ValueError('Wrong formal case identity')
            ranks = [json.loads((folder / f'rank-{rank}/entry.json').read_text()) for rank in (0, 1)]
            for record in ranks:
                loaded = record['weight_loading']
                if not loaded['all_source_tensors_and_native_parameters_covered'] or not loaded['all_loaded_shard_bytes_verified']:
                    raise ValueError('Actual original 7B weight loading incomplete')
                master = record['actual_final_cpu_masters']
                if master['groups'] != 199 or master['elements'] != 3834518016 or master['pinned_groups'] != (199 if arm == 'optimized' else 0):
                    raise ValueError('Actual optimized/control master dispatch differs')
                optimizer = record['actual_optimizer_state']['optimizers']
                if len(optimizer) != 1 or optimizer[0]['adam_state_entries'] != 199 or optimizer[0]['min_step'] != 120 or optimizer[0]['max_step'] != 120:
                    raise ValueError('All actual AdamW states must reach native step120')
            summary = summarize(ranks)
            means[seed][arm] = summary['mean_seconds_per_step']
            cases.append({'seed': seed, 'arm': arm, 'case': case['case'],
                          'case_sha256': hashlib.sha256((folder / 'case.json').read_bytes()).hexdigest(), **summary})
    return {'schema': 'jiuge7b-pinning-formal-round-v1', 'round': round_name, 'cases': cases,
            **evaluate(means), 'all_expected_cases_verified': True,
            'formal_performance_passed': False,
            'note': 'Round result only. Final promotion also requires any predeclared independent repeat.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--round', choices=('primary', 'confirmation'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.round)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}), flush=True)

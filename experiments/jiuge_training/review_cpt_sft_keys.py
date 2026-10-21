"""Review closed CPT/SFT key evidence and plot descriptive trace ranges."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
KEYS = BASE/'results/cpt-sft-tracing-20261017'
OUT = BASE/'results/cpt-sft-independent-review-20261017.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    local = read(BASE/'results/cpt-sft-key-verification-20261017.json')
    require(local['passed'] and not local['full_raw_local_copy_verified'], 'Key-only bundle must pass SHA checks')
    manifest = read(BASE/'results/sft-source-manifest-v4-20261017.json')['files']
    campaign = KEYS/'sft-tracing-20261017-01'
    require(read(campaign/'source-sha256.json') == manifest, 'Frozen source manifest differs')
    for name, sha in manifest.items():
        path = (BASE/'results/cpt-sft-bootstrap-source-20261017.sh' if name == 'bootstrap_sft_ubuntu.sh'
                else campaign/'source'/name)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == sha, 'Frozen source bytes differ')
    controller = read(campaign/'controller.json')
    names = ('sft-native-admission-20261017-01', 'sft-native-profile-20261017-01',
             'cpt-native-admission-20261017-01', 'cpt-native-profile-20261017-01')
    require(controller['status'] == 'completed' and controller['completed_cases'] == list(names), 'Campaign incomplete')
    require(all(row['exit_code'] == 0 for row in controller['commands']), 'A controller stage failed')
    summary = read(campaign/'tracing-summary.json')
    require(summary['passed'] and not summary['formal_speedup_evidence'], 'Diagnostic summary required')
    require(read(campaign/'cpt-sft-input-pair-verification.json')['passed'], 'Actual paired inputs differ')
    case_rows = []
    for name in names:
        folder = KEYS/'runs'/name
        case, close = read(folder/'case.json'), read(folder/'completion-verification.json')
        steps = 35 if 'profile' in name else 3
        require(case['exit_code'] == 0 and close['passed'], 'Case failed')
        require(not case['resource_gate_before'].strip() and not case['resource_gate_after'].strip(), 'Assigned GPU occupancy gate failed')
        require(close['actual_non_skipped_updates_each_rank'] == close['all_adam_states_step'] == steps, 'Actual optimizer updates differ')
        require(close['adam_state_entries_each_rank'] == 199, 'Adam state coverage differs')
        archive = read(campaign/(name+'-archive-verification.json'))
        require(archive['passed'], 'ACT primary archive failed')
        require(hashlib.sha256((folder/'archive-manifest.json').read_bytes()).hexdigest() == archive['manifest_sha256'], 'Archive receipt differs')
        for rank in range(2):
            entry = read(folder/f'rank-{rank}/entry.json')
            require(entry['status'] == 'completed' and entry['native_files_checked'] == 600, 'Native source coverage differs')
            weights = entry['weight_loading']
            require(weights['source_tensor_count'] == 339 and weights['source_parameter_elements'] == 7668831744
                    and weights['all_loaded_shard_bytes_verified'], 'Official weight coverage differs')
        case_rows.append({'case': name, 'updates_each_rank': steps,
                          'ACT_primary_verified_files': archive['verified_file_count'],
                          'ACT_primary_verified_bytes': archive['total_expected_bytes']})
    tasks = []
    for task in summary['tasks']:
        folder = KEYS/'runs'/task['case']
        trace = read(folder/'trace-analysis.json')
        require(len(trace['windows']) == len(task['windows']) == 4, 'Two windows on both ranks required')
        for row in task['windows']:
            original = next(x for x in trace['windows'] if x['window'] == row['window'] and x['rank'] == row['rank'])
            capture = read(folder/row['window']/f"rank-{row['rank']}-capture.json")
            gpu = original['analysis']['capture_window']['devices'][0]
            require(capture['trace_sha256'] == row['compressed_sha256'], 'Trace SHA linkage differs')
            require(capture['decompressed_json_sha256'] == row['decompressed_sha256'], 'Decompressed SHA linkage differs')
            require(capture['stable_step_ids'] == row['step_ids'] and len(row['step_ids']) == 5, 'Stable steps differ')
            require(capture['whole_window_boundary_drained'] and capture['gpu_collective_kernel_events'] == 922, 'Whole-step NCCL capture gate failed')
            require(abs(gpu['window_us']/1e6-row['profiled_five_step_window_seconds']) < 1e-9, 'Window duration differs')
            require(abs(gpu['compute_collective_overlap_us']/5e6-row['compute_NCCL_overlap_seconds_per_step']) < 1e-9, 'Overlap accounting differs')
            require(0 <= gpu['compute_collective_overlap_us'] <= min(gpu['interval_union_us']['compute'], gpu['interval_union_us']['collective']), 'Invalid overlap')
        def bounds(key):
            values = [row[key] for row in task['windows']]
            return {'min': min(values), 'max': max(values)}
        ranges = {key: bounds(key) for key in (
            'host_optimizer_offload_seconds_per_step', 'host_optimizer_offload_window_fraction',
            'GPU_compute_seconds_per_step', 'GPU_compute_window_fraction', 'GPU_memcpy_seconds_per_step',
            'GPU_NCCL_seconds_per_step', 'NCCL_overlap_with_compute_fraction', 'GPU_idle_proxy_fraction')}
        comm = read(folder/'communication-analysis.json')
        require(all(check['ordered_type_dtype_length_signatures_match'] for window in comm['cross_rank_pair_candidates']
                    for check in window['group_checks']), 'Ordered communication signatures differ')
        skew = max(item['start_skew_us'] for window in comm['cross_rank_pair_candidates'] for item in window['largest_start_skew_candidates'])
        copies = comm['ranks'][0]['recorded_memcpy_bytes_by_name']
        tasks.append({'objective': task['objective'], 'ranges_across_two_windows_two_ranks': ranges,
                      'largest_ordered_signature_start_skew_candidate_seconds': skew/1e6,
                      'native_collective_sequence_verified': False,
                      'recorded_HtoD_Pageable_GiB_each_rank_step_window01': copies['Memcpy HtoD (Pageable -> Device)']/5/1024**3,
                      'input_slots_per_window_not_summed_across_TP': 10240,
                      'supervised_targets_per_window': [row['supervised_targets_in_window'] for row in task['windows'] if row['rank'] == 0],
                      'sampled_memory_peak_GiB': read(folder/'host-memory-summary.json')['sampled_peak_bytes']/1024**3})
    closed = read(KEYS/'tracing-delivery-20261017-02/closed-review.json')
    require(closed['passed'] and closed['trace_windows'] == 8 and closed['source_files_verified'] == 24, 'ACT delivery review failed')
    result = {'schema': 'jiuge7b-CPT-SFT-independent-key-review-v1', 'utc': datetime.now(timezone.utc).isoformat(),
              'passed': True, 'local_bundle_sha256': local['bundle_sha256'], 'cases': case_rows, 'tasks': tasks,
              'source_files_verified': 24, 'raw_traces_verified_on_ACT': 8,
              'ACT_primary_verified_files': sum(x['ACT_primary_verified_files'] for x in case_rows),
              'ACT_primary_verified_bytes': sum(x['ACT_primary_verified_bytes'] for x in case_rows),
              'full_raw_originals_downloaded': False, 'full_gradient_optimizer_equivalence_claimed': False,
              'formal_speedup_evidence': False, 'causal_critical_path_proven': False,
              'raw_bundle': closed['original_bundle'], 'raw_bundle_sha256': closed['original_bundle_sha256']}
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.1), layout='constrained')
    items = [('host_optimizer_offload_window_fraction', 'Host optimizer/offload'),
             ('GPU_compute_window_fraction', 'GPU compute'),
             ('GPU_idle_proxy_fraction', 'GPU idle proxy')]
    for ax, task in zip(axes, tasks):
        for index, (key, label) in enumerate(items):
            value = task['ranges_across_two_windows_two_ranks'][key]
            low, high = value['min']*100, value['max']*100
            ax.plot([low, high], [index, index], linewidth=9, solid_capstyle='round', color=('#2962a3','#0f8877','#8b97a5')[index])
            ax.text(high+2, index, f'{low:.0f}–{high:.0f}%', va='center', fontsize=10)
        ax.set(yticks=range(3), yticklabels=[x[1] for x in items], xlim=(0, 90),
               ylim=(2.6,-.6), title=task['objective'], xlabel='Fraction of stable capture window (%)')
        ax.grid(axis='x', alpha=.2)
        ax.spines[['top','right','left']].set_visible(False)
    fig.suptitle('Jiuge 7B native CPT and SFT | 2 × A100 | TP2 | S2048', fontsize=13)
    fig.supxlabel('Ranges across 2 windows × 2 ranks; host/GPU accounts overlap and must not be added.\nDiagnostic profiling only; no formal speedup comparison.', fontsize=9)
    fig.savefig(BASE/'results/cpt-sft-profile-ranges-20261017.png', dpi=160)
    plt.close(fig)
    print(json.dumps({'passed': True, 'raw_traces': 8, 'tasks': tasks}, ensure_ascii=False))


if __name__ == '__main__':
    main()

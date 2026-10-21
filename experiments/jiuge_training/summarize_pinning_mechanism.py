"""Review bounded ACT trace summaries without downloading raw timelines."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

BASE = Path(__file__).resolve().parent
RESULTS = BASE / 'results'
PROFILE = RESULTS / 'pinning-profile-driver-20261016-01'


def load_bundle(path, receipt_path):
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if not receipt['passed'] or hashlib.sha256(path.read_bytes()).hexdigest() != receipt['bundle_sha256']:
        raise ValueError('Bundle differs from its verified receipt')
    with zipfile.ZipFile(path) as archive:
        catalog = json.loads(archive.read('key-results-manifest.json'))
        if set(archive.namelist()) != set(catalog['files']) | {'key-results-manifest.json'}:
            raise ValueError('Unexpected key members')
        objects = {}
        for name, expected in catalog['files'].items():
            data = archive.read(name)
            if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                raise ValueError('Key member checksum differs: ' + name)
            if name.endswith('.json'):
                objects[name] = json.loads(data)
    return objects, receipt


def main():
    profiles, profile_receipt = load_bundle(PROFILE / 'pinning-profile-key-results-20261016.zip',
                                            PROFILE / 'key-bundle-verification.json')
    copies, copy_receipt = load_bundle(RESULTS / 'pinning-copy-api-key-results-20261016.zip',
                                      RESULTS / 'pinning-copy-api-key-results-20261016-verification.json')
    controller = profiles['pinning-profile-review-20261016-01/controller.json']
    copy_controller = copies['pinning-copy-api-review-20261016-01/controller.json']
    if controller['status'] != 'completed' or copy_controller['status'] != 'completed':
        raise ValueError('Both ACT analysis stages must be complete')
    rows, arm_stats, skews = [], {}, {}
    native_messages, native_bytes = {}, {}
    for arm in ('native', 'optimized'):
        case = f'pinning-{arm}-profile-01'
        trace = profiles[f'runs/{case}/trace-analysis.json']
        communication = profiles[f'runs/{case}/communication-analysis.json']
        copy = copies[f'pinning-copy-api-review-20261016-01/{case}.json']
        proof = profiles[f'pinning-profile-review-20261016-01/{case}-primary-archive-verification.json']
        if trace['real_native_case_exit_code'] or not proof['passed'] or proof['verified_file_count'] != 35:
            raise ValueError('Training/primary archive verification incomplete')
        if any(not check['ordered_type_dtype_length_signatures_match']
               for window in communication['cross_rank_pair_candidates'] for check in window['group_checks']):
            raise ValueError('Rank communication signatures differ')
        skews[arm] = [{**candidate, 'window': window['window']}
                      for window in communication['cross_rank_pair_candidates']
                      for candidate in window['largest_start_skew_candidates']
                      if candidate['signature'] == ['allreduce', 'Int', 1, 1]]
        arm_rows = []
        if len(copy['windows']) != 4 or len(trace['windows']) != 4:
            raise ValueError('Both windows and ranks required')
        for window in trace['windows']:
            name, rank = window['window'], window['rank']
            expected_steps = list(range(21, 26)) if name == 'window-01' else list(range(31, 36))
            receipt = profiles[f'runs/{case}/{name}/rank-{rank}-capture.json']
            matching = [item for item in copy['windows'] if (item['window'], item['rank']) == (name, rank)]
            recorded = next(item for item in communication['ranks']
                            if (item['window'], item['rank']) == (name, rank))
            if len(matching) != 1:
                raise ValueError('Copy window missing/duplicated')
            matched = matching[0]
            if matched['steps'] != expected_steps or receipt['stable_step_ids'] != expected_steps:
                raise ValueError('Wrong stable training steps')
            if (matched['compressed_trace_sha256'] != receipt['trace_sha256'] or
                    matched['source']['decompressed_json_sha256'] != receipt['decompressed_json_sha256']):
                raise ValueError('Copy analysis is from a different trace')
            groups = matched['copy_groups']
            if any(item['unmatched_gpu_copy_events'] or item['ambiguous_gpu_copy_events'] or
                   item['copy_events_without_recorded_size'] or
                   item['api_correlations_shared_with_other_copy_names'] for item in groups):
                raise ValueError('Memcpy/API mapping or recorded byte coverage incomplete')
            actual_bytes = {item['copy_name']: item['recorded_bytes'] for item in groups}
            if actual_bytes != recorded['recorded_memcpy_bytes_by_name']:
                raise ValueError('Independent copy analyzers disagree on byte counts')
            device, = window['analysis']['capture_window']['devices']
            if abs(device['window_us'] - matched['window_us']) > .001:
                raise ValueError('Analysis capture intervals differ')
            clock = window['clock_review']
            if not clock['shared_kineto_export_time_domain'] or not clock['annotation_inside_recorded_host_bounds']:
                raise ValueError('Cross-rank clock metadata incomplete')
            key = (name, rank)
            # Group summary order follows first appearance on different streams;
            # compare the same per-group counts/metadata, independent of that order.
            messages = sorted(json.dumps(item, sort_keys=True) for item in recorded['device_message_groups'])
            h2d = [item for item in groups if 'HtoD' in item['copy_name']]
            dtoh = [item for item in groups if 'DtoH' in item['copy_name']]
            if arm == 'native':
                native_messages[key] = messages
                native_bytes[key] = sum(item['recorded_bytes'] for item in h2d)
            elif messages != native_messages[key] or sum(item['recorded_bytes'] for item in h2d) != native_bytes[key]:
                raise ValueError('Copy or communication work changed across arms')
            row = {'arm': arm, 'window': name, 'rank': rank, 'steps': expected_steps,
                   'raw_compressed_sha256': receipt['trace_sha256'], 'window_s': device['window_us'] / 1e6,
                   'recorded_memcpy_bytes': actual_bytes,
                   'h2d_cpu_api_clipped_sum_s': sum(item['cpu_api_clipped_sum_us'] for item in h2d) / 1e6,
                   'h2d_gpu_copy_work_sum_s': sum(item['gpu_duration']['sum_us'] for item in h2d) / 1e6,
                   'dtoh_gpu_copy_work_sum_s': sum(item['gpu_duration']['sum_us'] for item in dtoh) / 1e6,
                   'gpu_memcpy_union_s': device['interval_union_us']['memcpy'] / 1e6,
                   'gpu_compute_union_s': device['interval_union_us']['compute'] / 1e6,
                   'gpu_collective_union_s': device['interval_union_us']['collective'] / 1e6,
                   'compute_collective_overlap_s': device['compute_collective_overlap_us'] / 1e6,
                   'gpu_idle_proxy_s': device['gpu_idle_proxy_us'] / 1e6,
                   'optimizer_host_range_s': window['cpu_phase_ranges']['jiuge/optimizer_update']['host_range_sum_us'] / 1e6,
                   'forward_backward_host_range_s': window['cpu_phase_ranges']['jiuge/forward_backward']['host_range_sum_us'] / 1e6,
                   'gpu_collective_events': device['activity_record_counts']['collective']}
            arm_rows.append(row)
            rows.append(row)
        rank_steps = sum(len(row['steps']) for row in arm_rows)
        wall = sum(row['window_s'] for row in arm_rows)
        totals = {name: sum(row[name] for row in arm_rows) for name in (
            'h2d_cpu_api_clipped_sum_s', 'h2d_gpu_copy_work_sum_s', 'dtoh_gpu_copy_work_sum_s',
            'gpu_memcpy_union_s', 'gpu_compute_union_s', 'gpu_collective_union_s',
            'compute_collective_overlap_s', 'gpu_idle_proxy_s', 'optimizer_host_range_s',
            'forward_backward_host_range_s')}
        byte_names = {name for row in arm_rows for name in row['recorded_memcpy_bytes']}
        arm_stats[arm] = {'rank_steps': rank_steps,
                         'per_rank_step_seconds': {name: value / rank_steps for name, value in totals.items()},
                         'per_rank_step_recorded_bytes': {name: sum(row['recorded_memcpy_bytes'].get(name, 0)
                                                                  for row in arm_rows) / rank_steps for name in sorted(byte_names)},
                         'gpu_compute_window_fraction': totals['gpu_compute_union_s'] / wall,
                         'gpu_idle_proxy_window_fraction': totals['gpu_idle_proxy_s'] / wall,
                         'optimizer_host_range_window_fraction': totals['optimizer_host_range_s'] / wall,
                         'collective_overlap_with_compute_fraction': totals['compute_collective_overlap_s'] / totals['gpu_collective_union_s']}
    output = {'schema': 'jiuge7b-independent-pinning-mechanism-review-v1',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'key_bundle_receipts': [profile_receipt, copy_receipt],
              'verified_raw_trace_windows': len(rows), 'closed_original_archives_passed': True,
              'copy_volume_and_communication_signatures_unchanged': True,
              'rows': rows, 'arm_statistics': arm_stats, 'small_collective_skew_candidates': skews,
              'limitations': {'formal_throughput_evidence': False, 'critical_path_proven': False,
                              'collector_zero_drop_verified': False, 'cpu_api_spans_are_exposed_wait': False,
                              'categories_are_additive_wall_time': False, 'physical_wire_bytes_measured': False,
                              'native_collective_sequence_numbers_verified': False,
                              'per_step_asynchronous_gpu_attribution_verified': False}}
    path = RESULTS / 'pinning-mechanism-review-20261016.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(path), 'arm_statistics': arm_stats}, indent=2))


if __name__ == '__main__':
    main()

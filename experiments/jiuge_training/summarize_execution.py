"""Create a factual execution receipt from saved training and archive evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    names = ('native-smoke-256g-01', 'native-profile-256g-01',
             'native-smoke-s2048-01', 'native-profile-s2048-01')
    archives = [read(args.results / f'archive-verification-{name}.json') for name in names]
    numeric = read(args.results / 'local-evidence-review-20261015.json')
    probe = read(args.results / 'offload-pinning-probe-20261015.json')
    assert all(item['passed'] and not item['mismatches'] for item in archives)
    assert numeric['passed'] and numeric['trace_count'] == 8
    assert probe['after_cpu_pin_memory_is_pinned'] and not probe['after_float_is_pinned']
    rows = []
    for sequence, name in ((128, names[1]), (2048, names[3])):
        analysis = read(args.results / name / 'trace-analysis.json')
        for row in analysis['windows']:
            device = row['analysis']['capture_window']['devices'][0]
            rows.append({'sequence_length': sequence, 'window': row['window'], 'rank': row['rank'],
                         'window_seconds': device['window_us'] / 1e6,
                         'optimizer_host_range_fraction': row['cpu_phase_ranges'][
                             'jiuge/optimizer_update']['host_window_fraction'],
                         'device_compute_fraction': device['interval_union_us']['compute'] / device['window_us'],
                         'compute_collective_overlap_us': device['compute_collective_overlap_us']})
    source_manifest = read(args.results / 'native-inputs-20261015/source_manifest.json')
    source_copies = {}
    for path in (args.results / 'native-inputs-20261015/sources').glob('*.py'):
        relative = 'megatron/core/optimizer/' + (
            'cpu_offloading/' if path.name == 'hybrid_optimizer.py' else '') + path.name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == source_manifest['files'][relative]
        source_copies[relative] = digest
    formal = read(base / 'tracing.contract.template.json')['frozen_formal_benchmark_reference']
    formal_sha = hashlib.sha256((base / formal['path']).read_bytes()).hexdigest()
    assert formal_sha == formal['sha256']
    result = {'schema': 'jiuge7b-cpt-execution-summary-v1',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'personal_scope': 'Jiuge 7B full-parameter continued pretraining',
              'hardware': 'ACT, two NVIDIA A100-PCIE-40GB, 16CPU, 256GiB host',
              'topology': {'tp': 2, 'pp': 1, 'cp': 1, 'dp': 1, 'interconnect': 'SYS, no NVLink'},
              'native_backend': 'pinned Megatron local Torch + native CPU optimizer offload',
              'native_training_cases': numeric['cases'], 'timeline_files_verified': numeric['trace_count'],
              'actual_cpu_input_batches_verified': len(numeric['actual_inputs']),
              'verified_archive_file_count': sum(item['verified_file_count'] for item in archives),
              'verified_archive_bytes': sum(item['total_expected_bytes'] for item in archives),
              'archive_receipts': archives, 'profiled_window_rows': rows,
              'source_copies_sha256': source_copies,
              'frozen_formal_benchmark_sha256': formal_sha,
              'first_optimization_candidate': 'CPU master parameter pinning lost after BF16 clone/FP32 conversion',
              'minimal_copy_chain_reproduced': True, 'native_baseline_source_modified': False,
              'raw_timelines_and_samples_saved_on_act_and_local': True,
              'model_weights_and_indexed_dataset_primary_copy': 'ACT persistent shared home',
              'old_failed_dual_gpu_job_deleted_with_confirmation': True,
              'old_v100_retained_per_user_request': True,
              'remaining_limits': {'collector_zero_drop_count_verified': False,
                  'native_collective_sequence_numbers_verified': False,
                  'complete_critical_path_proven': False,
                  'full_optimizer_update_equivalence_tested': False,
                  'complete_checkpoint_resume_tested': False,
                  'multi_node_or_dp2_validated': False},
              'formal_performance_passed': False, 'iluvatar_performance_evidence': False,
              'new_external_model_api_calls': False,
              'code_files_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in sorted(base.glob('*.py'))}}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'verified_archive_files': result['verified_archive_file_count'],
                      'verified_archive_bytes': result['verified_archive_bytes'],
                      'timeline_files': result['timeline_files_verified'],
                      'frozen_benchmark_unchanged': True}))


if __name__ == '__main__':
    main()

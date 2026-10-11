"""Verify accepted fresh rounds and exclude the resource-rejected diagnosis.

CPU checks of recorded GPU evidence only. Default CI exposes missing large
originals; --require-raw is the local publication gate, not a GPU rerun.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
BACKUP_SHA = '188cda588c5bb6ded8cea0033ec587a54b80292c0cb601bf00bf11ddfcfdedcd'
WRAPPER_SHA = '0fa42f0d9c3982eb0efb5000a5585912e4ad86a234c0a2373b1f89b943c00dab'
CANDIDATE_SHA = 'b9a2ca76c294481e38e2422332ccb821224bd64b8bb7cd61d972a5e2a4a9133e'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))


def verify(require_raw=False, long_only=False):
    import_path = HERE / 'evidence' / ('completed-formal-long-import-retry01-02.json' if long_only else 'completed-formal-import-retry01-02.json')
    imported = read(import_path)
    assert imported['schema'] == ('retry01-partial-formal-local-import-v2' if long_only else 'retry01-completed-formal-local-import-v2')
    assert imported['status'] == 'exact_metadata_and_full_raw_verified'
    assert imported['original_files_overwritten'] is False and imported['gpu_work_executed'] is False
    assert imported['diagnostic_round_excluded_from_promotion'] is True and imported['diagnostic_processes'] == 9
    backup_path = HERE / 'evidence/raw-http-backup-retry01-01-receipt.json'
    assert sha(backup_path) == BACKUP_SHA == imported['raw_backup_receipt_sha256']
    backup = read(backup_path)
    assert backup['complete_raw_backup'] and backup['formal_timing_barrier_released']
    assert backup['status'] == 'all_384_originals_byte_verified'
    assert len(backup['files']) == 384 and backup['original_bytes'] == 22291589014
    assert backup['unique_objects'] == 227 and backup['unique_bytes'] == 9978190961
    reader_path = HERE / 'verify_receipts_retry01_mount.py'
    assert sha(reader_path) == imported['reader_sha256']
    spec = importlib.util.spec_from_file_location('completed_fresh_receipt_reader', reader_path)
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)

    for row in imported['files']:
        path = HERE / row['destination_relative']
        assert path.resolve().is_relative_to(HERE.resolve()) and not path.is_symlink()
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['destination_relative']
    assert len(imported['files']) == imported['file_count']
    old_path = HERE / 'evidence/retry01-metadata/root/browser-formal-long-retry01-primary-01.json'
    assert sha(old_path) == imported['original_failed_wrapper_sha256']
    old = read(old_path)
    assert old['status'] == 'failed' and old['exit_code'] == 0
    assert old['error'] == 'Post-run resources busy; timing needs further examination'
    assert old['occupancy_before']['passed'] and old['occupancy_after']['both_gpus_idle']
    diagnostic_path = HERE / 'evidence/long-primary-diagnostic-import-01.json'
    assert sha(diagnostic_path) == imported['diagnostic_import_sha256']
    assert read(diagnostic_path)['accepted_for_skill_release'] is False
    remote_wrapper = HERE / 'evidence/retry01-metadata/root/browser_formal_retry01_v2.py'
    assert sha(remote_wrapper) == WRAPPER_SHA

    models = {}
    accepted = 0
    assert set(imported['selected']) == ({'long'} if long_only else {'long', 'experts'})
    for model, selection in imported['selected'].items():
        expected_first = 'resource_confirmation' if model == 'long' else 'primary'
        expected_directory = 'confirmation' if model == 'long' else ''
        assert selection['primary_round'] == expected_first
        assert selection['primary_timing_directory'] == expected_directory
        assert len(selection['rounds']) == (2 if selection['repeat_required'] else 1)
        rounds = []
        for index, row in enumerate(selection['rounds']):
            if index == 0:
                assert row['round'] == expected_first and row['timing_directory'] == expected_directory
            else:
                assert row['round'] == ('confirmation2' if model == 'long' else 'confirmation')
            wrapper_path = HERE / row['wrapper_relative']
            perf_path = HERE / row['performance_relative']
            assert sha(wrapper_path) == row['wrapper_sha256'] and sha(perf_path) == row['performance_sha256']
            wrapper, perf = read(wrapper_path), read(perf_path)
            assert wrapper['schema'] == 'retry01-formal-wrapper-v2'
            assert wrapper['status'] == 'passed' and wrapper['exit_code'] == 0
            assert wrapper['wrapper_sha256'] == WRAPPER_SHA
            assert wrapper['complete_local_raw_proof_sha256'] == BACKUP_SHA
            assert wrapper['model'] == model and wrapper['round'] == row['round']
            assert wrapper['performance_sha256'] == sha(perf_path)
            for phase in ('before', 'after'):
                boundary = wrapper['occupancy_' + phase]
                assert boundary['passed'] and boundary['both_gpus_idle'] and boundary['training_port_free']
                assert boundary['foreign_compute_processes'] == []
                assert wrapper['cpu_' + phase]['outside_formal_timing'] is True
            assert wrapper['post_training_settle_seconds'] == 5
            if model == 'long' and index == 0:
                assert wrapper['resource_repeat_predecessor_sha256'] == sha(old_path)
                assert wrapper['primary_performance_sha256'] == sha(HERE / 'models-retry01/long/evidence/performance-004.json')
            cv = {arm: statistics.stdev(p['cases'][arm]['mean_iteration_ms'] for p in perf['pairs']) /
                       statistics.mean(p['cases'][arm]['mean_iteration_ms'] for p in perf['pairs'])
                  for arm in ('native', 'control', 'optimized')}
            assert cv == row['seed_mean_cv'] == wrapper['seed_mean_cv']
            if index == 0:
                assert selection['repeat_required'] is any(cv[a] >= .1 for a in ('native', 'control'))
                assert wrapper['repeat_required'] == selection['repeat_required']
            assert row['timing_directory'] in ('', 'confirmation'), 'Explicit confirmation2 reader required'
            verified = reader.verify(model, require_raw=require_raw, confirmation=bool(row['timing_directory']))
            assert all(r['native_control_equal'] and r['native_optimized_equal'] for r in verified['loss_print_comparisons'])
            assert perf['candidate_sha256'] == CANDIDATE_SHA
            assert verified['vs_control'] == wrapper['vs_control'] and verified['vs_native'] == wrapper['vs_native']
            rounds.append({'round': row['round'], 'performance_relative': row['performance_relative'],
                'wrapper_relative': row['wrapper_relative'], 'performance_sha256': sha(perf_path),
                'wrapper_sha256': sha(wrapper_path), 'seed_mean_cv': cv, **verified})
            accepted += 9
        promoted = all(r['promoted'] for r in rounds)
        assert selection['promoted_with_repeat_policy'] == promoted
        models[model] = {'primary': rounds[0], 'rounds': rounds, 'repeat_required': selection['repeat_required'],
                          'promoted': promoted, 'configuration': read(HERE / 'models-retry01' / model / 'contract.json')['model']}
    assert accepted == imported['accepted_formal_processes']
    return {'schema': 'fresh-heldout-completion-verified-v2', 'status': 'receipt_verified',
        'candidate_sha256': CANDIDATE_SHA, 'retuned': False, 'models': models, 'long_only': long_only,
        'accepted_formal_processes': accepted, 'accepted_timed_steps': accepted * 120,
        'accepted_retained_steps': accepted * 100, 'diagnostic_processes': 9,
        'all_timing_processes': accepted + 9, 'diagnostic_round_excluded_from_promotion': True,
        'raw_original_file_count': 384, 'raw_original_bytes': 22291589014,
        'raw_unique_objects': 227, 'raw_unique_bytes': 9978190961,
        'original_byte_coverage_complete': all(r['original_byte_coverage_complete'] for m in models.values() for r in m['rounds']),
        'require_raw': require_raw, 'raw_backup_receipt_sha256': BACKUP_SHA,
        'formal_import_sha256': sha(import_path), 'completion_reader_sha256': sha(__file__),
        'numerical_reader_sha256': sha(reader_path), 'old_collection_or_inventory_overwritten': False,
        'gpu_work_executed': False,
        'scope': 'Fresh replacement-Pod collection. Single BI-V150; four-layer BF16 MoE with FP32 SGD masters, mock data, dropout0, TP/PP/CP/EP1. CPU checks recorded GPU receipts and logs, not GPU arithmetic. Printed loss equality is not a convergence or 120-step all-tensor proof; no production, AdamW, GroupedGEMM, multiGPU, SFT/RL claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-raw', action='store_true')
    parser.add_argument('--long-only', action='store_true', help='Verify accepted long round; does not establish two-model completion')
    parser.add_argument('--seal', action='store_true', help='Write immutable decision only after full local raw verification')
    args = parser.parse_args()
    if args.seal:
        assert args.require_raw and not args.long_only, 'Publication requires both models and actual original bytes'
    result = verify(args.require_raw, args.long_only)
    if args.seal:
        assert result['original_byte_coverage_complete'] and all(m['promoted'] for m in result['models'].values())
        target = HERE / 'evidence/decision-retry01-02.json'
        assert not target.exists()
        result.update(status='complete', sealed_utc=datetime.now(timezone.utc).isoformat())
        with target.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

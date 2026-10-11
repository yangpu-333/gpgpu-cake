"""Collect accepted resource-gated rounds; preserve the first diagnostic round.

Inbound, credential-free GETs of our own JSON/log originals only. No Jupyter
control API, Torch execution, candidate execution, or GPU operation is used.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent.parent
BASE = 'http://share.ai.ic.h3i.buaa.edu.cn/ai-exp-sy2506311-kda1-service-qpia/files/private/kda-biv150-stage35-recovery01/'
WRAPPER_SHA = '0fa42f0d9c3982eb0efb5000a5585912e4ad86a234c0a2373b1f89b943c00dab'
PARENT_SHA = '5a59d06701b6e57672e09d5da7968676949007c4cafd05449667aefe8a05113f'
BACKUP_SHA = '188cda588c5bb6ded8cea0033ec587a54b80292c0cb601bf00bf11ddfcfdedcd'
sha = lambda data: hashlib.sha256(data).hexdigest()
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
DOWNLOADS = ROOT / 'results/formal-v2-originals'


def fetch(origin, expected=None):
    relative = PurePosixPath(origin)
    assert not relative.is_absolute() and '..' not in relative.parts
    assert relative.suffix in ('.json', '.log', '.py')
    assert not any(x in relative.name.lower() for x in ('ssh', 'token', 'password'))
    target = DOWNLOADS.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(['curl.exe', '-q', '--fail', '--silent', '--show-error',
        '--connect-timeout', '8', '--max-time', '20', '--noproxy', 'share.ai.ic.h3i.buaa.edu.cn',
        '--proto', '=http', '--output', str(target), BASE + origin],
        capture_output=True, text=True, timeout=25)
    assert process.returncode == 0, (origin, process.stderr)
    data = target.read_bytes()
    assert expected is None or sha(data) == expected, origin
    return data


def wrapper(model, round_name):
    name = f'browser-formal-{model}-retry01-{round_name}-02.json'
    data = fetch(name)
    value = json.loads(data)
    assert value['schema'] == 'retry01-formal-wrapper-v2'
    assert value['model'] == model and value['round'] == round_name
    assert value['status'] == 'passed' and value['exit_code'] == 0
    assert value['wrapper_sha256'] == WRAPPER_SHA and value['parent_wrapper_sha256'] == PARENT_SHA
    assert value['complete_local_raw_proof_sha256'] == BACKUP_SHA
    assert value['occupancy_before']['passed'] and value['occupancy_after']['passed']
    assert not value['occupancy_before']['foreign_compute_processes']
    assert not value['occupancy_after']['foreign_compute_processes']
    assert value['post_training_settle_seconds'] == 5
    return name, data, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--long-only', action='store_true', help='Archive the completed long round without sealing either-model completion')
    args = parser.parse_args()
    record_name = 'completed-formal-long-import-retry01-02.json' if args.long_only else 'completed-formal-import-retry01-02.json'
    record_path = ROOT / 'evidence' / record_name
    assert not record_path.exists(), 'Immutable completion receipt already exists'
    assert sha((ROOT / 'evidence/raw-http-backup-retry01-01-receipt.json').read_bytes()) == BACKUP_SHA
    diagnostic_path = ROOT / 'evidence/long-primary-diagnostic-import-01.json'
    diagnostic = read(diagnostic_path)
    assert diagnostic['accepted_for_skill_release'] is False
    old_wrapper = ROOT / 'evidence/retry01-metadata/root/browser-formal-long-retry01-primary-01.json'
    old_data = old_wrapper.read_bytes()
    old = json.loads(old_data)
    assert old['status'] == 'failed' and old['exit_code'] == 0
    assert old['error'] == 'Post-run resources busy; timing needs further examination'

    selected = {}
    cached = {}
    tasks = {}

    def add(origin, destination, expected=None, data=None):
        assert origin not in tasks
        relative = PurePosixPath(destination)
        assert not relative.is_absolute() and '..' not in relative.parts
        if data is not None:
            assert expected is None or sha(data) == expected
            cached[origin] = data
        tasks[origin] = {'origin_relative': origin, 'destination_relative': destination,
                         'expected_sha256': expected}

    add('browser_formal_retry01_v2.py', 'evidence/retry01-metadata/root/browser_formal_retry01_v2.py', WRAPPER_SHA)
    for model, first_round, timing in (('long', 'resource_confirmation', 'confirmation'),
                                        ('experts', 'primary', '')):
        if args.long_only and model != 'long':
            continue
        name, data, w = wrapper(model, first_round)
        if model == 'long':
            assert w['resource_repeat_predecessor_sha256'] == sha(old_data)
            assert w['primary_performance_sha256'] == sha((ROOT / 'models-retry01/long/evidence/performance-004.json').read_bytes())
        selected[model] = {'primary_round': first_round, 'primary_timing_directory': timing,
                           'repeat_required': w['repeat_required'], 'rounds': []}
        rounds = [(first_round, timing, name, data, w)]
        if w['repeat_required']:
            next_round, next_timing = ('confirmation2', 'confirmation2') if model == 'long' else ('confirmation', 'confirmation')
            name2, data2, w2 = wrapper(model, next_round)
            rounds.append((next_round, next_timing, name2, data2, w2))
        for round_name, timing_directory, name, data, w in rounds:
            remote = 'models-retry01/' + model + ('/' + timing_directory if timing_directory else '')
            dest = 'models-retry01/' + model + '/evidence' + ('/' + timing_directory if timing_directory else '')
            perf_data = fetch(remote + '/performance-004.json', w['performance_sha256'])
            perf = json.loads(perf_data)
            assert perf['status'] == 'validated' and len(perf['completed_cases']) == 9
            cv = {arm: statistics.stdev(p['cases'][arm]['mean_iteration_ms'] for p in perf['pairs']) /
                       statistics.mean(p['cases'][arm]['mean_iteration_ms'] for p in perf['pairs'])
                  for arm in ('native', 'control', 'optimized')}
            assert cv == w['seed_mean_cv']
            if round_name == first_round:
                assert w['repeat_required'] is any(cv[a] >= .1 for a in ('native', 'control'))
            assert perf['vs_control'] == w['vs_control'] and perf['vs_native'] == w['vs_native']
            add(remote + '/performance-004.json', dest + '/performance-004.json', w['performance_sha256'], perf_data)
            for pair in perf['pairs']:
                for arm, case in pair['cases'].items():
                    assert case['case'] == f'{arm}-perf-004-{pair["seed"]}'
                    assert pair['seed'] in (35102, 35103, 35104)
                    digests = {**case['raw_sha256'], **case['dispatch_sha256']}
                    assert set(digests) == {'command.json', 'entry.json', 'stdout.log', 'stderr.log', 'weighted.json', 'gradnorm.json'}
                    for filename, digest in digests.items():
                        add(remote + '/' + case['case'] + '/' + filename,
                            dest + '/' + case['case'] + '/' + filename, digest)
            add(name, 'evidence/retry01-metadata/root/' + name, sha(data), data)
            for suffix in ('.stdout.log', '.stderr.log'):
                log = name[:-5] + suffix
                add(log, 'evidence/retry01-metadata/root/' + log)
            selected[model]['rounds'].append({'round': round_name, 'timing_directory': timing_directory,
                'wrapper_relative': 'evidence/retry01-metadata/root/' + name,
                'performance_relative': dest + '/performance-004.json',
                'wrapper_sha256': sha(data), 'performance_sha256': w['performance_sha256'], 'seed_mean_cv': cv})

    def download(row):
        origin = row['origin_relative']
        data = cached.get(origin)
        if data is None:
            data = fetch(origin, row['expected_sha256'])
        return row, data

    with ThreadPoolExecutor(max_workers=4) as pool:
        originals = list(pool.map(download, tasks.values()))
    payload, rows = {}, []
    for row, data in originals:
        destination = ROOT / row['destination_relative']
        assert destination.resolve().is_relative_to(ROOT.resolve()) and not destination.is_symlink()
        if destination.exists():
            assert destination.read_bytes() == data, 'Existing original differs: ' + str(destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(data)
        rows.append({**row, 'bytes': len(data), 'sha256': sha(data)})
        payload['payload/' + row['origin_relative']] = data

    spec = importlib.util.spec_from_file_location('retry01_accepted_receipt_reader', ROOT / 'verify_receipts_retry01_mount.py')
    reader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reader)
    for model, selection in selected.items():
        for round_row in selection['rounds']:
            directory = round_row['timing_directory']
            assert directory in ('', 'confirmation'), 'Explicit confirmation2 receipt reader needed; never relabel its originals'
            result = reader.verify(model, require_raw=True, confirmation=bool(directory))
            assert result['original_byte_coverage_complete']
            assert all(r['native_control_equal'] and r['native_optimized_equal'] for r in result['loss_print_comparisons'])
            round_row['cpu_receipt_validation'] = result
        selection['promoted_with_repeat_policy'] = all(r['cpu_receipt_validation']['promoted'] for r in selection['rounds'])
    accepted_count = sum(9 * len(s['rounds']) for s in selected.values())
    record = {'schema': 'retry01-partial-formal-local-import-v2' if args.long_only else 'retry01-completed-formal-local-import-v2', 'status': 'exact_metadata_and_full_raw_verified',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'files': rows, 'file_count': len(rows),
        'selected': selected, 'accepted_formal_processes': accepted_count, 'diagnostic_processes': 9,
        'original_failed_wrapper_sha256': sha(old_data), 'diagnostic_import_sha256': sha(diagnostic_path.read_bytes()),
        'diagnostic_round_excluded_from_promotion': True, 'original_files_overwritten': False,
        'collector_sha256': sha(Path(__file__).read_bytes()), 'raw_backup_receipt_sha256': BACKUP_SHA,
        'reader_sha256': sha((ROOT / 'verify_receipts_retry01_mount.py').read_bytes()),
        'gpu_work_executed': False, 'tensor_payloads_included_in_archive': False}
    archive_path = ROOT / 'results' / ('metadata-retry01-accepted-long-02.zip' if args.long_only else 'metadata-retry01-accepted-formal-02.zip')
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            archive.writestr(name, data)
        archive.writestr('manifest.json', json.dumps(record, indent=2) + '\n')
    record['archive_sha256'] = sha(archive_path.read_bytes())
    with record_path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': record['status'], 'files': len(rows), 'accepted_formal_processes': accepted_count,
                     'models': {m: {'promoted': s['promoted_with_repeat_policy'],
                         'rounds': [{k: r['cpu_receipt_validation'][k] for k in ('vs_control', 'vs_native')} for r in s['rounds']]}
                         for m, s in selected.items()}}, indent=2))


if __name__ == '__main__':
    main()

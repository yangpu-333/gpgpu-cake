"""Check final evidence provenance without running or changing GPU experiments."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-checkout', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite evidence')
    root = Path(__file__).resolve().parent
    parent = root.parent
    decision = json.loads((root / 'evidence/decision-0006.json').read_text(encoding='utf-8'))
    final = root / 'evidence/0006-final'
    reports = {name: json.loads((final / (name + '.json')).read_text(encoding='utf-8'))
               for name in decision['report_sha256']}
    native_commit = '5be9626709af2722333bf54797c954c09edeada3'
    expected_native = {}
    for report in reports.values():
        for path, digest in report.get('native_sources_sha256', {}).items():
            if path in expected_native and expected_native[path] != digest:
                raise ValueError('native source changed between processes: ' + path)
            expected_native[path] = digest
    native_checks = {}
    prefix = ['git', '-c', 'safe.directory=' + args.native_checkout.resolve().as_posix(),
              '-C', str(args.native_checkout.resolve())]
    for path, digest in expected_native.items():
        payload = subprocess.check_output(prefix + ['show', native_commit + ':' + path])
        native_checks[path] = hashlib.sha256(payload).hexdigest() == digest
    parent_source = (parent / 'candidates/0003/candidate.py').read_text(encoding='utf-8').strip()
    candidate_prefix = {i: (root / ('candidates/' + i + '/candidate.py')).read_text(
        encoding='utf-8').startswith(parent_source) for i in ('0004', '0005', '0006')}
    receipts = {}
    for i in ('0004', '0005', '0006'):
        candidate = root / ('candidates/' + i + '/candidate.py')
        receipt = json.loads(candidate.with_suffix('.py.receipt.json').read_text(encoding='utf-8'))
        response = root / 'evidence' / receipt['response'].replace('\\', '/').split('/')[-1]
        receipts[i] = receipt['output_sha256'] == sha(candidate) and receipt['response_sha256'] == sha(response)
    checks = {
        'all_native_sources_match_pinned_git_objects': bool(native_checks) and all(native_checks.values()),
        'all_processes_same_recorded_environment': all(
            r.get('environment', {}).get('megatron_commit') == native_commit
            and r.get('environment', {}).get('gpu') == 'Iluvatar BI-V150'
            and r.get('environment', {}).get('torch') == '2.4.1'
            and r.get('environment', {}).get('triton') == '2.1.0' for r in reports.values()),
        'all_candidate_children_retain_normalized_parent_source': all(candidate_prefix.values()),
        'all_candidates_match_original_cli_response_receipts': all(receipts.values()),
        'all_final_report_bytes_match_decision': all(sha(final / (n + '.json')) == digest
            for n, digest in decision['report_sha256'].items()),
    }
    payload = {'candidate_id': '0006', 'checks': checks, 'native_source_checks': native_checks,
               'normalized_parent_prefix': candidate_prefix, 'cli_receipts': receipts,
               'native_commit': native_commit, 'passed': all(checks.values()),
               'scope': 'Delivery provenance only; GPU correctness and timing remain in the frozen formal reports.'}
    args.output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2))
    return 0 if payload['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

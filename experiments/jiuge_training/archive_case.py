"""SHA256 manifests for completed native cases and their local copies."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--create', action='store_true')
    mode.add_argument('--verify', action='store_true')
    parser.add_argument('--snapshot-analysis', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    root = args.case_root.resolve()
    case = json.loads((root / 'case.json').read_text(encoding='utf-8'))
    if case['exit_code'] != 0:
        raise ValueError('Only completed native training cases are admitted here')
    manifest_path = root / 'archive-manifest.json'
    if args.create:
        if args.snapshot_analysis:
            if not (root / 'trace-analysis.json').is_file():
                raise ValueError('Completed analysis required before snapshotting its source')
            folder = root / 'analysis-source'
            folder.mkdir(exist_ok=True)
            for name in ('analyze_trace.py', 'review_native_trace.py', 'inspect_trace_communication.py'):
                payload = (Path(__file__).resolve().parent / name).read_bytes()
                target = folder / name
                if target.exists() and target.read_bytes() != payload:
                    raise ValueError('Existing immutable analysis source differs')
                if not target.exists():
                    with target.open('xb') as stream:
                        stream.write(payload)
        files = {str(path.relative_to(root)).replace('\\', '/'): {
                    'bytes': path.stat().st_size, 'sha256': digest(path)}
                 for path in sorted(root.rglob('*')) if path.is_file() and path != manifest_path}
        result = {'schema': 'jiuge-completed-case-archive-v1', 'case': case['case'],
                  'created_utc': datetime.now(timezone.utc).isoformat(), 'files': files,
                  'formal_performance_evidence': False}
        with manifest_path.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    else:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        mismatches = []
        for name, expected in manifest['files'].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                mismatches.append(name + ': missing or unsafe path')
            elif path.stat().st_size != expected['bytes'] or digest(path) != expected['sha256']:
                mismatches.append(name + ': size/SHA256 mismatch')
        result = {'schema': 'jiuge-local-case-archive-verification-v1', 'case': case['case'],
                  'manifest_sha256': digest(manifest_path),
                  'verified_file_count': len(manifest['files']) - len(mismatches),
                  'total_expected_bytes': sum(value['bytes'] for value in manifest['files'].values()),
                  'mismatches': mismatches, 'passed': not mismatches,
                  'full_optimizer_checkpoint_archived': False}
        if args.receipt:
            with args.receipt.open('x', encoding='utf-8') as stream:
                json.dump(result, stream, indent=2)
                stream.write('\n')
        if mismatches:
            raise ValueError(json.dumps(result))
    print(json.dumps({'case': case['case'], 'file_count': len(result.get('files', {}))}
                     if args.create else result), flush=True)


if __name__ == '__main__':
    main()

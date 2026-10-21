"""Fetch the pinned native framework into a new preparation directory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = '5be9626709af2722333bf54797c954c09edeada3'
SOURCE = 'https://github.com/NVIDIA/Megatron-LM.git'


def run(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    print(result.stdout, end='', flush=True)
    print(result.stderr, end='', flush=True)
    result.check_returncode()
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest['commit'] != COMMIT:
        raise ValueError('Unexpected source manifest commit')
    if args.root.exists():
        if not (args.root / '.git').is_dir():
            raise ValueError('Existing directory is not a Git checkout; refusing to replace it')
    else:
        args.root.mkdir(parents=True)
        run(['git', 'init', str(args.root)])
        run(['git', '-C', str(args.root), 'remote', 'add', 'origin', SOURCE])
        run(['git', '-C', str(args.root), 'fetch', '--depth=1', 'origin', COMMIT])
        run(['git', '-C', str(args.root), 'checkout', '--detach', 'FETCH_HEAD'])
    actual = run(['git', '-C', str(args.root), 'rev-parse', 'HEAD'])
    if actual != COMMIT:
        raise ValueError('Existing checkout is not the requested pinned commit')
    dirty = run(['git', '-C', str(args.root), 'status', '--porcelain'])
    errors = []
    for name, expected in manifest['files'].items():
        path = (args.root / name).resolve()
        if not path.is_relative_to(args.root.resolve()):
            raise ValueError('Manifest path escaped source directory')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(name)
    record = {'schema': 'jiuge-native-source-staging-v1',
              'utc': datetime.now(timezone.utc).isoformat(), 'commit': actual,
              'repository': SOURCE, 'checkout_path': str(args.root),
              'source_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'files_checked': len(manifest['files']), 'mismatches': errors,
              'working_tree_clean': not dirty,
              'verified': not errors and not dirty, 'framework_imported': False}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record, indent=2))
    return 0 if record['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

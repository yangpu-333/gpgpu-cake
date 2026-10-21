"""Inventory ACT partials; remove only verified bulky local duplicates on request."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

NAMES = ('pinning-validation-20261015-01', 'pinning-validation-20261015-02')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(8388608):
            digest.update(data)
    return digest.hexdigest()


def inventory(root):
    root = root.resolve(strict=True)
    files = {}
    for name in NAMES:
        folder = (root / name).resolve(strict=True)
        if not folder.is_relative_to(root):
            raise ValueError('Archive path escapes root')
        for path in sorted(folder.rglob('*')):
            if path.is_file():
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    raise ValueError('Unexpected link in partial archive')
                files[path.relative_to(root).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    return {'schema': 'jiuge-pinning-partial-primary-inventory-v1', 'root': str(root), 'files': files,
            'created_utc': datetime.now(timezone.utc).isoformat(), 'primary_raw_deleted': False}


def cleanup(root, catalog):
    root = root.resolve(strict=True)
    workspace = Path('E:/code/GPGPU').resolve(strict=True)
    expected_root = workspace / 'experiments/jiuge_training/results'
    if root != expected_root or not root.is_relative_to(workspace):
        raise ValueError('Only the exact experiment local results root is allowed')
    manifest = json.loads(catalog.read_text())
    mismatches = []
    for name, expected in manifest['files'].items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            mismatches.append(name)
    if mismatches:
        raise ValueError('Local/ACT partials differ: ' + str(mismatches[:10]))
    targets = [root / NAMES[0] / 'raw-objects']
    targets.extend(root / NAMES[1] / 'native' / f'rank-{rank}' / 'raw.zlibpack' for rank in (0, 1))
    for target in targets:
        resolved = target.resolve(strict=True)
        if not resolved.is_relative_to(root) or target.is_symlink():
            raise ValueError('Unsafe local deletion target')
        for path in ([resolved] if resolved.is_file() else resolved.rglob('*')):
            if path.is_file() and path.relative_to(root).as_posix() not in manifest['files']:
                raise ValueError('Uncatalogued file in local raw-only target')
    deleted = []
    for target in targets:
        resolved = target.resolve(strict=True)
        if not resolved.is_relative_to(root) or target.is_symlink():
            raise ValueError('Unsafe local deletion target')
        if resolved.is_dir():
            # Absolute target has been checked against the exact workspace root.
            count = sum(p.stat().st_size for p in resolved.rglob('*') if p.is_file())
            shutil.rmtree(resolved)
        else:
            count = resolved.stat().st_size
            resolved.unlink()
        deleted.append({'path': str(resolved), 'bytes': count})
    return {'schema': 'jiuge-verified-local-bulky-copy-cleanup-v1',
            'primary_inventory_sha256': sha(catalog), 'verified_primary_file_count': len(manifest['files']),
            'deleted_local_duplicates': deleted, 'freed_bytes': sum(item['bytes'] for item in deleted),
            'act_primary_raw_deleted': False, 'local_key_results_retained': True,
            'reason': 'User requested full originals on ACT and only key results on the local computer'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--delete-local-large-raw', action='store_true')
    args = parser.parse_args()
    result = cleanup(args.root, args.catalog) if args.delete_local_large_raw else inventory(args.root)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}), flush=True)

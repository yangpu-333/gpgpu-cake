"""Export bounded key results; large numeric originals remain on ACT.

The exported manifest explicitly distinguishes key-file verification from a
complete raw-value archive verification. No raw tensor, checkpoint, or trace
body is eligible for this bundle.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

KEY_SUFFIXES = {'.json', '.jsonl', '.log', '.txt', '.py', '.patch'}
MANIFEST = 'key-results-manifest.json'


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def select(root, folders, limit):
    root = root.resolve(strict=True)
    selected, excluded = {}, []
    total = 0
    for folder in folders:
        folder = folder.resolve(strict=True)
        if not folder.is_relative_to(root) or folder == root:
            raise ValueError('Select named result folders inside the persistent experiment root')
        for path in sorted(folder.rglob('*')):
            if path.is_symlink():
                raise ValueError('Links are not eligible for key-result export')
            if not path.is_file():
                continue
            if not path.resolve(strict=True).is_relative_to(root):
                raise ValueError('File escapes the persistent experiment root')
            name = path.relative_to(root).as_posix()
            if name in selected:
                continue
            if (path.suffix not in KEY_SUFFIXES or
                    any(part in {'raw-objects', 'checkpoints', 'huggingface-cache'} for part in path.parts) or
                    path.name == 'tensor-journal.jsonl'):
                excluded.append({'file': name, 'bytes': path.stat().st_size})
                continue
            size = path.stat().st_size
            total += size
            if total > limit:
                raise ValueError('Key-result budget exceeded; select fewer completed runs')
            selected[name] = {'bytes': size, 'sha256': digest(path.read_bytes())}
    if not selected:
        raise ValueError('No key results selected')
    return selected, excluded


def export(root, folders, output, limit=64 * 1024**2):
    root = root.resolve(strict=True)
    if limit <= 0:
        raise ValueError('Positive local key-result budget required')
    # Inventory before creating the output, so an old/new bundle cannot include itself.
    files, excluded = select(root, folders, limit)
    if output.resolve() in {(root / name).resolve() for name in files}:
        raise ValueError('Bundle would overwrite a selected source')
    result = {'schema': 'jiuge-pinning-key-results-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'primary_root': str(root), 'files': files, 'excluded_originals': excluded,
              'key_file_count': len(files), 'key_uncompressed_bytes': sum(r['bytes'] for r in files.values()),
              'budget_bytes': limit, 'full_raw_originals_included': False,
              'full_raw_local_copy_verified': False, 'primary_originals_deleted': False}
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for name, expected in files.items():
            payload = (root / name).read_bytes()
            if len(payload) != expected['bytes'] or digest(payload) != expected['sha256']:
                raise ValueError('Source changed during key export: ' + name)
            archive.writestr(name, payload)
        archive.writestr(MANIFEST, json.dumps(result, indent=2) + '\n')
    return result


def verify(bundle, output, limit=64 * 1024**2):
    with zipfile.ZipFile(bundle) as archive:
        info = archive.infolist()
        names = [entry.filename for entry in info]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate bundle members')
        if sum(entry.file_size for entry in info) > limit + 4 * 1024**2:
            raise ValueError('Key bundle exceeds local size budget')
        catalog = json.loads(archive.read(MANIFEST))
        if catalog['full_raw_originals_included'] or catalog['full_raw_local_copy_verified']:
            raise ValueError('Incorrect complete-raw-copy claim')
        if set(names) != set(catalog['files']) | {MANIFEST}:
            raise ValueError('Bundle/catalog membership differs')
        total = 0
        for name, expected in catalog['files'].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe bundle path')
            if relative.suffix not in KEY_SUFFIXES or relative.name == 'tensor-journal.jsonl':
                raise ValueError('Numeric original in key-only bundle')
            payload = archive.read(name)
            total += len(payload)
            if len(payload) != expected['bytes'] or digest(payload) != expected['sha256']:
                raise ValueError('Key result differs: ' + name)
        if total > limit or total != catalog['key_uncompressed_bytes']:
            raise ValueError('Key-result total differs or exceeds budget')
    receipt = {'schema': 'jiuge-pinning-local-key-verification-v1', 'bundle': str(bundle.resolve()),
               'bundle_sha256': digest(bundle.read_bytes()), 'primary_root': catalog['primary_root'],
               'verified_key_files': len(catalog['files']), 'verified_key_bytes': total,
               'passed': True, 'full_raw_local_copy_verified': False,
               'originals_location': 'ACT persistent directory; excluded from this local bundle'}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--folder', type=Path, action='append')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-bundle', type=Path)
    parser.add_argument('--max-mib', type=int, default=64)
    args = parser.parse_args()
    limit = args.max_mib * 1024**2
    if args.verify_bundle:
        result = verify(args.verify_bundle, args.output, limit)
    else:
        if not args.root or not args.folder:
            parser.error('Export requires --root and named --folder arguments')
        result = export(args.root, args.folder, args.output, limit)
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'excluded_originals')}))


if __name__ == '__main__':
    main()

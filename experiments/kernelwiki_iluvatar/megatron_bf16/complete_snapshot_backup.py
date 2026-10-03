"""Finish snapshot backup using hash deduplication and resumable file chunks.

Reuses complete, SHA-verified members of interrupted archives. Identical raw
snapshots are hard-linked locally; every original filename retains exact bytes.
Remote originals and interrupted archives are retained.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
from backup_snapshots import sha

ROOT = Path(__file__).resolve().parent
BACKUP = ROOT / 'evidence/raw-backup'
CHUNK = 16 * 1024 * 1024


def main():
    inventories = {label: json.loads((BACKUP / (label + '-snapshot-inventory.json')).read_text())
                   for label in ('nfs', 'pod-local')}
    by_sha = {}
    rows = []
    for label, inventory in inventories.items():
        expected = {r['path']: r for r in inventory['files']}
        for row in inventory['files']:
            item = dict(row, source_label=label,
                        remote_path=inventory['remote_root'] + '/' + row['path'])
            rows.append(item)
            dest = ROOT / 'evidence' / row['path']
            if dest.exists() and dest.stat().st_size == row['bytes'] and sha(dest) == row['sha256']:
                by_sha[row['sha256']] = dest
        archive = BACKUP / (label + '-snapshots.tar.gz')
        if not archive.exists():
            continue
        try:
            with tarfile.open(archive, 'r|gz') as packed:
                for member in packed:
                    if member.isdir():
                        continue
                    assert member.isfile() and member.name in expected
                    row = expected[member.name]
                    assert member.size == row['bytes']
                    if row['sha256'] in by_sha:
                        continue
                    dest = ROOT / 'evidence' / member.name
                    assert dest.resolve().is_relative_to((ROOT / 'evidence').resolve())
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    partial = dest.with_suffix(dest.suffix + '.partial')
                    digest = hashlib.sha256()
                    stream = packed.extractfile(member)
                    with partial.open('wb') as out:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                            out.write(chunk)
                            digest.update(chunk)
                    assert partial.stat().st_size == row['bytes'] and digest.hexdigest() == row['sha256']
                    partial.replace(dest)
                    by_sha[row['sha256']] = dest
        except (EOFError, tarfile.ReadError, gzip.BadGzipFile, OSError) as exc:
            print(json.dumps({'phase': 'partial-archive-reused', 'source_label': label,
                              'unique_verified': len(by_sha), 'end': type(exc).__name__}), flush=True)
    unique = {r['sha256']: r for r in rows}
    print(json.dumps({'phase': 'deduplicated-inventory', 'unique': len(unique),
                      'already_verified': len(by_sha),
                      'remaining_raw_bytes': sum(r['bytes'] for h, r in unique.items() if h not in by_sha)}), flush=True)
    for index, (digest, row) in enumerate(unique.items(), 1):
        if digest in by_sha:
            continue
        dest = ROOT / 'evidence' / row['path']
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_suffix(dest.suffix + '.chunk-partial')
        offset = partial.stat().st_size if partial.exists() else 0
        assert offset <= row['bytes'] and (offset % CHUNK == 0 or offset == row['bytes'])
        with partial.open('ab') as out:
            while offset < row['bytes']:
                # Pipefail rejects failed reads. Fixed chunks limit SSH failure loss.
                command = 'set -o pipefail; dd if=' + shlex.quote(row['remote_path'])
                command += f' bs={CHUNK} skip={offset // CHUNK} count=1 status=none | gzip -1'
                result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                                         'bi-v150', command], capture_output=True, timeout=180)
                assert result.returncode == 0, result.stderr.decode(errors='replace')
                data = gzip.decompress(result.stdout)
                assert len(data) == min(CHUNK, row['bytes'] - offset)
                out.write(data)
                out.flush()
                offset += len(data)
        assert partial.stat().st_size == row['bytes'] and sha(partial) == digest
        partial.replace(dest)
        by_sha[digest] = dest
        print(json.dumps({'phase': 'unique-verified', 'index': index, 'unique_verified': len(by_sha),
                          'total_unique': len(unique)}), flush=True)
    for row in rows:
        dest = ROOT / 'evidence' / row['path']
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(by_sha[row['sha256']], dest)
        assert dest.stat().st_size == row['bytes'] and sha(dest) == row['sha256']
    for label, inventory in inventories.items():
        receipt = BACKUP / (label + '-snapshot-receipt.json')
        result = {'status': 'verified', 'source_label': label, 'remote_deleted': False,
                  'remote_root': inventory['remote_root'], 'files': inventory['files'],
                  'verified_files': len(inventory['files']),
                  'raw_bytes': sum(r['bytes'] for r in inventory['files']),
                  'method': 'SHA-verified complete archive members plus resumable16MiB gzip chunks; identical hashes hard-linked locally.',
                  'scope': 'All original snapshot filenames and exact bytes, including failed partial files; remote originals retained; interrupted archives not claimed complete.'}
        assert not receipt.exists()
        receipt.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': 'verified', 'files': len(rows), 'unique_hashes': len(unique),
                      'raw_bytes': sum(r['bytes'] for r in rows)}), flush=True)


if __name__ == '__main__':
    main()

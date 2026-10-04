"""Stream raw stage29 files without retaining a redundant transfer archive."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

BASE = Path(__file__).resolve().parent.parent / 'evidence/swiglu-stage29'
ROOT = BASE / 'raw-backup'
REMOTE = '/tmp/kda-swiglu-stage29'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    inventory = BASE / 'raw-inventory.json'
    receipt_path = BASE / 'raw-backup-receipt.json'
    assert not inventory.exists() and not receipt_path.exists()
    program = '''import hashlib,json
from pathlib import Path
root=Path('/tmp/kda-swiglu-stage29');rows=[]
for path in sorted(root.rglob('*')):
 if not path.is_file() or not ('snapshots' in path.parts or 'activation' in path.parts or path.suffix=='.so'):continue
 assert path.resolve().is_relative_to(root)
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 rows.append({'path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps(rows))
'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'bi-v150',
                             '/root/miniconda3/bin/python -'],
                            input=program.encode(), capture_output=True, timeout=180)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    rows = json.loads(result.stdout)
    assert rows
    inventory.write_text(json.dumps(rows, indent=2) + '\n', encoding='utf-8', newline='\n')
    unique = {}
    for row in rows:
        unique.setdefault(row['sha256'], row)
    print(json.dumps({'phase': 'inventory', 'files': len(rows),
                      'raw_bytes': sum(row['bytes'] for row in rows),
                      'unique_files': len(unique),
                      'unique_bytes': sum(row['bytes'] for row in unique.values())}), flush=True)
    expected = {row['path']: row for row in unique.values()}
    command = 'tar -cf - -C ' + REMOTE + ' ' + ' '.join(shlex.quote(name) for name in expected)
    process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', 'bi-v150', command],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    seen = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as packed:
            for member in packed:
                assert member.isfile() and member.name in expected
                assert member.name not in seen
                seen.add(member.name)
                dest = ROOT / member.name
                assert dest.resolve().is_relative_to(ROOT.resolve())
                assert not dest.exists()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open('xb') as output:
                    stream = packed.extractfile(member)
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        output.write(chunk)
                row = expected[member.name]
                assert dest.stat().st_size == row['bytes'] and sha(dest) == row['sha256']
        assert process.wait(timeout=20) == 0 and seen == set(expected)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    for row in rows:
        dest = ROOT / row['path']
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(ROOT / unique[row['sha256']]['path'], dest)
        assert dest.stat().st_size == row['bytes'] and sha(dest) == row['sha256']
    receipt = {'status': 'verified', 'remote_root': REMOTE, 'remote_deleted': False,
               'inventory_sha256': sha(inventory), 'files': rows,
               'verified_files': len(rows), 'raw_bytes': sum(row['bytes'] for row in rows),
               'unique_files': len(unique),
               'unique_bytes': sum(row['bytes'] for row in unique.values()),
               'transfer': 'Streaming uncompressed tar; no duplicate archive retained; '
                           'same-byte files linked only after SHA256 verification'}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({key: receipt[key] for key in
                      ('status', 'verified_files', 'raw_bytes', 'unique_files', 'unique_bytes')}), flush=True)


if __name__ == '__main__':
    main()

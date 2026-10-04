"""Back up one stage31 Attention candidate's three audit arms after GPU timing has finished.

Raw checkpoints and tensor snapshots stay in a local Git-ignored directory.
Unique contents are streamed once into shared SHA256 objects; original paths
are hardlinked, including equal contents shared between candidate versions.
No transfer tar is retained and no remote evidence is deleted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile


SOURCE = Path(__file__).resolve().parent
REMOTE = '/tmp/kda-attention-stage31'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--base', type=Path, default=SOURCE / 'evidence')
    parser.add_argument('--host', default='bi-v150')
    parser.add_argument('--audit-suffix', default='complete')
    parser.add_argument('--timing-finished', action='store_true',
                        help='Required acknowledgement that all remote GPU timing has finished.')
    args = parser.parse_args()
    require(args.version.isdigit() and len(args.version) == 3, 'three-digit version required')
    require(args.timing_finished, 'defer backup I/O until all GPU timing finishes; pass --timing-finished')
    require(args.audit_suffix.replace('-', '').isalnum(), 'simple immutable audit suffix required')
    base = args.base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    root = base / 'raw-backup'
    root.mkdir(exist_ok=True)
    # The ignore rule lives inside the raw-only directory, leaving every sealed
    # project source, contract and experiment controller unchanged.
    ignore = root / '.gitignore'
    if not ignore.exists():
        with ignore.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write('*\n')
    require(ignore.read_text(encoding='utf-8') == '*\n', 'raw backup directory lacks exact Git ignore rule')
    objects = root / 'objects'
    objects.mkdir(exist_ok=True)
    inventory_path = base / ('raw-inventory-' + args.version + '.json')
    receipt_path = base / ('raw-backup-' + args.version + '-receipt.json')
    require(not receipt_path.exists(), 'immutable completed backup receipt already exists')
    program = '''import hashlib,json
from pathlib import Path
root=Path(REMOTE);version=VERSION;suffix=SUFFIX;rows=[]
for arm in ('native','control','optimized'):
 folder=root/('main-'+arm+'-audit-'+version+'-'+suffix)/'snapshots'
 if not folder.is_dir():raise RuntimeError('missing audit snapshots: '+str(folder))
 for p in sorted(folder.rglob('*')):
  if not p.is_file():continue
  if p.is_symlink() or not p.resolve().is_relative_to(root):raise RuntimeError('unsafe raw file')
  h=hashlib.sha256()
  with p.open('rb') as stream:
   for block in iter(lambda:stream.read(1048576),b''):h.update(block)
  rows.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
capture=root/'native-default-capture-02/attention'
if not capture.is_dir():raise RuntimeError('missing original native capture')
for p in sorted(capture.glob('*.pt')):
 if p.is_symlink() or not p.resolve().is_relative_to(root):raise RuntimeError('unsafe capture')
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 rows.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps(rows))
'''.replace('REMOTE', repr(REMOTE)).replace('VERSION', repr(args.version)).replace('SUFFIX', repr(args.audit_suffix))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', args.host,
                             '/root/miniconda3/bin/python -'], input=program.encode(),
                            capture_output=True, timeout=600)
    require(result.returncode == 0, result.stderr.decode(errors='replace'))
    rows = json.loads(result.stdout)
    require(isinstance(rows, list) and rows, 'nonempty raw inventory required')
    expected, unique = {}, {}
    for row in rows:
        rel = PurePosixPath(row['path'])
        require(not rel.is_absolute() and '..' not in rel.parts
                and ((rel.parts[0] in {'main-' + arm + '-audit-' + args.version + '-' + args.audit_suffix
                                      for arm in ('native', 'control', 'optimized')}
                      and rel.parts[1] == 'snapshots') or
                     (rel.parts[0] == 'native-default-capture-02' and rel.parts[1] == 'attention'
                      and rel.suffix == '.pt')) and row['bytes'] >= 0
                and len(row['sha256']) == 64 and all(c in '0123456789abcdef' for c in row['sha256'])
                and row['path'] not in expected, 'unsafe raw inventory row')
        expected[row['path']] = row
        previous = unique.setdefault(row['sha256'], row)
        require(previous['bytes'] == row['bytes'], 'same SHA has conflicting byte count')
    if inventory_path.exists():
        require(json.loads(inventory_path.read_text(encoding='utf-8')) == rows,
                'immutable remote inventory changed on backup retry')
    else:
        write_new(inventory_path, rows)
    needed = {}
    for digest, row in unique.items():
        object_path = objects / digest
        if object_path.exists():
            require(object_path.is_file() and object_path.stat().st_size == row['bytes']
                    and sha(object_path) == digest, 'existing deduplicated raw object differs')
        else:
            needed[row['path']] = row
    print(json.dumps({'phase': 'inventory', 'version': args.version, 'files': len(rows),
                      'raw_bytes': sum(row['bytes'] for row in rows), 'unique_files': len(unique),
                      'unique_bytes': sum(row['bytes'] for row in unique.values()),
                      'objects_to_transfer': len(needed)}), flush=True)
    seen = set()
    if needed:
        command = 'tar -cf - -C ' + shlex.quote(REMOTE) + ' -- ' + ' '.join(shlex.quote(name) for name in needed)
        process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', args.host, command],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        try:
            with tarfile.open(fileobj=process.stdout, mode='r|') as packed:
                for member in packed:
                    require(member.isfile() and member.name in needed and member.name not in seen,
                            'unexpected, linked or duplicate raw transfer member')
                    row = needed[member.name]
                    require(member.size == row['bytes'], 'transferred raw byte count differs')
                    seen.add(member.name)
                    destination = objects / row['sha256']
                    partial = objects / (row['sha256'] + '.partial')
                    require(not destination.exists() and not partial.exists(),
                            'refusing to overwrite raw object or interrupted partial file')
                    h = hashlib.sha256()
                    with partial.open('xb') as out:
                        stream = packed.extractfile(member)
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            h.update(block)
                            out.write(block)
                    require(partial.stat().st_size == row['bytes'] and h.hexdigest() == row['sha256'],
                            'raw transfer SHA256 differs')
                    partial.rename(destination)
            require(process.wait(timeout=20) == 0 and seen == set(needed), 'raw transfer incomplete')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    verified_objects = {}
    for row in rows:
        rel = PurePosixPath(row['path'])
        destination = root.joinpath(*rel.parts)
        require(destination.resolve().is_relative_to(root.resolve()), 'raw destination escaped backup root')
        object_path = objects / row['sha256']
        if row['sha256'] not in verified_objects:
            require(object_path.stat().st_size == row['bytes'] and sha(object_path) == row['sha256'],
                    'final raw object verification failed')
            verified_objects[row['sha256']] = row['bytes']
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(object_path, destination)
        require(destination.is_file() and os.path.samefile(object_path, destination),
                'original raw path must be a hardlink to verified object')
    receipt = {'status': 'verified', 'version': args.version, 'audit_suffix': args.audit_suffix,
               'remote_root': REMOTE, 'remote_deleted': False, 'inventory_sha256': sha(inventory_path),
               'files': rows, 'verified_files': len(rows), 'raw_bytes': sum(row['bytes'] for row in rows),
               'unique_files': len(unique), 'unique_bytes': sum(row['bytes'] for row in unique.values()),
               'objects_transferred': len(needed), 'deduplication': 'shared SHA256 objects and hardlinks',
               'transfer_archive_retained': False, 'backup_script_sha256': sha(__file__),
               'gpu_timing_finished_acknowledged': True,
               'scope': 'Local raw tensor/checkpoint backup only; no tensor arithmetic replay or remote deletion.'}
    write_new(receipt_path, receipt)
    print(json.dumps({key: receipt[key] for key in ('status', 'version', 'verified_files',
                                                   'raw_bytes', 'unique_files', 'unique_bytes')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

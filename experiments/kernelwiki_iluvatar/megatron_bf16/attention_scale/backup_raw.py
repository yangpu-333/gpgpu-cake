"""Back up stage32 frozen Attention003 audit snapshots after all GPU timing ends.

One immutable inventory/receipt belongs to each model and audit suffix. Raw
files stay in a Git-ignored local directory; identical contents across models
and rounds share SHA256 objects and hardlinks. No transfer archive is retained,
and no remote file is changed or deleted. This script does not import Torch,
Megatron, Triton, a candidate, or any GPU experiment controller.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile


SOURCE = Path(__file__).resolve().parent
REMOTE = '/tmp/kda-attention-scale-stage32'
VERSION = '003'
ARMS = ('native', 'control', 'optimized')


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


def validate_inventory(model, suffix, rows):
    require(model in ('wide', 'deep') and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]*', suffix),
            'declared model and simple immutable audit suffix required')
    require(isinstance(rows, list) and rows, 'nonempty complete original raw inventory required')
    folders = {model + '-' + arm + '-audit-' + VERSION + '-' + suffix for arm in ARMS}
    expected, unique, seen_folders = {}, {}, set()
    for row in rows:
        rel = PurePosixPath(row['path'])
        require(not rel.is_absolute() and '..' not in rel.parts and len(rel.parts) >= 3
                and '\\' not in row['path'] and ':' not in row['path']
                and rel.parts[0] in folders and rel.parts[1] == 'snapshots'
                and type(row['bytes']) is int and row['bytes'] >= 0
                and type(row['sha256']) is str and re.fullmatch(r'[0-9a-f]{64}', row['sha256'])
                and row['path'] not in expected, 'unsafe, duplicate or out-of-scope raw inventory row')
        expected[row['path']] = row
        seen_folders.add(rel.parts[0])
        previous = unique.setdefault(row['sha256'], row)
        require(previous['bytes'] == row['bytes'], 'same raw SHA has conflicting byte count')
    require(seen_folders == folders, 'original snapshot inventory must cover all three audited arms')
    return expected, unique


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('wide', 'deep'), required=True)
    parser.add_argument('--base', type=Path, default=SOURCE / 'evidence')
    parser.add_argument('--host', default='bi-v150')
    parser.add_argument('--audit-suffix', default='complete')
    parser.add_argument('--timing-finished', action='store_true',
                        help='Required acknowledgement that every remote GPU timing job has finished.')
    args = parser.parse_args()
    require(args.timing_finished,
            'defer remote hashing/large-file I/O until all GPU timing finishes; pass --timing-finished')
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]*', args.audit_suffix),
            'simple immutable audit suffix required')
    base = args.base.resolve()
    inventory_path = base / ('raw-inventory-' + args.model + '-' + args.audit_suffix + '.json')
    receipt_path = base / ('raw-backup-' + args.model + '-' + args.audit_suffix + '-receipt.json')
    require(not receipt_path.exists(), 'refusing to overwrite an immutable completed backup receipt')
    base.mkdir(parents=True, exist_ok=True)
    root = base / 'raw-backup'
    root.mkdir(exist_ok=True)
    ignore = root / '.gitignore'
    if not ignore.exists():
        with ignore.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write('*\n')
    require(ignore.read_text(encoding='utf-8') == '*\n', 'exact raw-only Git ignore rule required')
    objects = root / 'objects'
    objects.mkdir(exist_ok=True)
    program = '''import hashlib,json
from pathlib import Path
root=Path(REMOTE);model=MODEL;suffix=SUFFIX;version=VERSION;rows=[]
for arm in ('native','control','optimized'):
 folder=root/(model+'-'+arm+'-audit-'+version+'-'+suffix)/'snapshots'
 if not folder.is_dir():raise RuntimeError('missing original audit snapshots: '+str(folder))
 count=0
 for p in sorted(folder.rglob('*')):
  if not p.is_file():continue
  if p.is_symlink() or not p.resolve().is_relative_to(root):raise RuntimeError('unsafe original raw file')
  h=hashlib.sha256()
  before=p.stat()
  with p.open('rb') as stream:
   for block in iter(lambda:stream.read(1048576),b''):h.update(block)
  after=p.stat()
  if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('raw file changed while hashing')
  rows.append({'path':p.relative_to(root).as_posix(),'bytes':after.st_size,'sha256':h.hexdigest()})
  count+=1
 if count==0:raise RuntimeError('empty original audit arm snapshots: '+str(folder))
print(json.dumps(rows))
'''.replace('REMOTE', repr(REMOTE)).replace('MODEL', repr(args.model)).replace('SUFFIX', repr(args.audit_suffix)).replace('VERSION', repr(VERSION))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', args.host,
                             '/root/miniconda3/bin/python -'], input=program.encode(),
                            capture_output=True, timeout=600)
    require(result.returncode == 0, result.stderr.decode(errors='replace'))
    rows = json.loads(result.stdout)
    expected, unique = validate_inventory(args.model, args.audit_suffix, rows)
    if inventory_path.exists():
        require(json.loads(inventory_path.read_text(encoding='utf-8')) == rows,
                'refusing to replace changed immutable remote inventory on retry')
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
    print(json.dumps({'phase': 'inventory', 'model_name': args.model, 'version': VERSION,
                      'audit_suffix': args.audit_suffix, 'files': len(rows),
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
                    require(member.size == row['bytes'], 'transferred original raw byte count differs')
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
                            'original raw transfer SHA256 differs')
                    partial.rename(destination)
            require(process.wait(timeout=20) == 0 and seen == set(needed), 'original raw transfer incomplete')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    verified_objects = set()
    for row in rows:
        rel = PurePosixPath(row['path'])
        destination = root.joinpath(*rel.parts)
        require(destination.resolve().is_relative_to(root.resolve()), 'raw destination escaped backup root')
        object_path = objects / row['sha256']
        if row['sha256'] not in verified_objects:
            require(object_path.stat().st_size == row['bytes'] and sha(object_path) == row['sha256'],
                    'final original raw object verification failed')
            verified_objects.add(row['sha256'])
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(object_path, destination)
        require(destination.is_file() and os.path.samefile(object_path, destination),
                'original raw path must be a hardlink to its verified SHA256 object')
    receipt = {'schema': 'frozen-attention-stage32-raw-backup-v1', 'status': 'verified',
               'model_name': args.model, 'version': VERSION, 'audit_suffix': args.audit_suffix,
               'remote_root': REMOTE, 'remote_deleted': False, 'inventory_sha256': sha(inventory_path),
               'files': rows, 'verified_files': len(rows), 'raw_bytes': sum(row['bytes'] for row in rows),
               'unique_files': len(unique), 'unique_bytes': sum(row['bytes'] for row in unique.values()),
               'objects_transferred': len(needed), 'deduplication': 'shared SHA256 objects and hardlinks',
               'transfer_archive_retained': False, 'backup_script_sha256': sha(__file__),
               'gpu_timing_finished_acknowledged': True,
               'scope': 'Complete original three-arm audit raw tensor/checkpoint/metadata backup; '
                        'no Torch import, tensor arithmetic replay, GPU work or remote deletion.'}
    write_new(receipt_path, receipt)
    print(json.dumps({key: receipt[key] for key in ('status', 'model_name', 'version', 'audit_suffix',
                                                   'verified_files', 'raw_bytes', 'unique_files', 'unique_bytes')}),
          flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

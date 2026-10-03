"""Copy immutable GPU snapshots to desktop and verify every remote SHA256.

No remote deletion. Archives and extracted tensors are Git-ignored. Includes
failed/partial snapshots as exact bytes without claiming they are valid tensors.
Run when GPU experiments have finished so backup I/O cannot distort timings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='bi-v150')
    parser.add_argument('--remote-root',required=True)
    parser.add_argument('--label',choices=('nfs','pod-local'),required=True)
    args=parser.parse_args()
    allowed={'/private/kda-biv150-stage3/megatron_bf16','/tmp/kda-bf16-continuation/megatron_bf16'}
    assert args.remote_root in allowed
    backup=ROOT/'evidence/raw-backup'
    backup.mkdir(exist_ok=True)
    receipt=backup/(args.label+'-snapshot-receipt.json')
    archive=backup/(args.label+'-snapshots.tar.gz')
    assert not receipt.exists()
    program='''from pathlib import Path
import hashlib,json
root=Path(%r).resolve()/'evidence'
files=[];folders=[]
for folder in sorted(root.glob('*/snapshots')):
 assert folder.resolve().is_relative_to(root)
 folders.append(folder.relative_to(root).as_posix())
 for path in sorted(folder.rglob('*')):
  if not path.is_file():continue
  assert path.resolve().is_relative_to(root)
  h=hashlib.sha256()
  with path.open('rb') as stream:
   for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
  files.append({'path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps({'remote_root':str(root),'folders':folders,'files':files}))
'''%args.remote_root
    remote=subprocess.run(['ssh','-o','BatchMode=yes',args.host,'/root/miniconda3/bin/python -'],
                          input=program.encode(),capture_output=True,timeout=900)
    assert remote.returncode==0,remote.stderr.decode(errors='replace')
    inventory=json.loads(remote.stdout)
    assert inventory['files'] and inventory['folders']
    print(json.dumps({'phase':'inventory','source_label':args.label,
        'files':len(inventory['files']),'raw_bytes':sum(r['bytes'] for r in inventory['files'])}),flush=True)
    inventory_path=backup/(args.label+'-snapshot-inventory.json')
    if inventory_path.exists():
        assert json.loads(inventory_path.read_text())==inventory
    else:
        inventory_path.write_text(json.dumps(inventory,indent=2)+'\n')
    command='set -o pipefail; tar -I '+shlex.quote('gzip -1')+' -cf - -C '+shlex.quote(inventory['remote_root'])+' '+ ' '.join(shlex.quote(p) for p in inventory['folders'])
    offset=archive.stat().st_size if archive.exists() else 0
    if offset:
        command+=' | tail -c +'+str(offset+1)
    print(json.dumps({'phase':'streaming','source_label':args.label,'resume_bytes':offset}),flush=True)
    with archive.open('ab') as out:
        copied=subprocess.run(['ssh','-o','BatchMode=yes',args.host,command],stdout=out,stderr=subprocess.PIPE,timeout=14400)
    assert copied.returncode==0,copied.stderr.decode(errors='replace')
    print(json.dumps({'phase':'archive-copied','source_label':args.label,'bytes':archive.stat().st_size}),flush=True)
    expected={row['path']:row for row in inventory['files']}
    observed=set()
    with tarfile.open(archive,'r:gz') as packed:
        for member in packed:
            if member.isdir():continue
            assert member.isfile() and member.name in expected
            dest=ROOT/'evidence'/member.name
            assert dest.resolve().is_relative_to((ROOT/'evidence').resolve())
            row=expected[member.name]
            assert member.size==row['bytes']
            if dest.exists():
                assert dest.stat().st_size==row['bytes'] and sha(dest)==row['sha256']
            else:
                dest.parent.mkdir(parents=True,exist_ok=True)
                digest=hashlib.sha256()
                source=packed.extractfile(member)
                with dest.open('xb') as target:
                    for chunk in iter(lambda:source.read(1024*1024),b''):
                        target.write(chunk);digest.update(chunk)
                assert digest.hexdigest()==row['sha256']
            observed.add(member.name)
            if len(observed)%16==0:
                print(json.dumps({'phase':'verified','source_label':args.label,'files':len(observed)}),flush=True)
    assert observed==set(expected)
    result={'status':'verified','source_label':args.label,'remote_deleted':False,
            'remote_root':inventory['remote_root'],'archive':str(archive.resolve()),
            'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),
            'files':inventory['files'],'verified_files':len(observed),
            'raw_bytes':sum(row['bytes'] for row in inventory['files']),
            'scope':'Byte-exact local backup of all snapshots including partial failures; no claim every file is valid tensor data. Remote originals retained.'}
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','source_label','verified_files','raw_bytes','archive_bytes')}),flush=True)


if __name__=='__main__':
    main()

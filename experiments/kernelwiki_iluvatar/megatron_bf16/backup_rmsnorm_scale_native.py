"""Preserve all stage28 audit snapshots by exact hash, retaining remote originals."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parent/'evidence/rmsnorm-stage28/raw-backup-native'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--only-version');args=parser.parse_args()
    if args.only_version:
        assert args.only_version.isdigit()
        ROOT=ROOT/('raw-backup-'+args.only_version)
    ROOT.mkdir(parents=True,exist_ok=True)
    program='''import json,hashlib
from pathlib import Path
root=Path('/tmp/kda-rmsnorm-stage28');rows=[]
for folder in sorted(root.glob('native-audit*/snapshots')):
 for p in sorted(folder.rglob('*')):
  if not p.is_file():continue
  assert p.resolve().is_relative_to(root)
  h=hashlib.sha256()
  with p.open('rb') as s:
   for chunk in iter(lambda:s.read(1024*1024),b''):h.update(chunk)
  rows.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps(rows))
'''
    if args.only_version:
        program=program.replace("*audit*/snapshots",'*audit*'+args.only_version+'*/snapshots')
    r=subprocess.run(['ssh','-o','BatchMode=yes','bi-v150','/root/miniconda3/bin/python -'],input=program.encode(),capture_output=True,timeout=120)
    assert r.returncode==0,r.stderr.decode()
    rows=json.loads(r.stdout);assert rows
    inventory=ROOT/'snapshot-inventory.json';assert not inventory.exists()
    inventory.write_text(json.dumps(rows,indent=2)+'\n')
    unique={}
    for row in rows:unique.setdefault(row['sha256'],row)
    print(json.dumps({'phase':'inventory','files':len(rows),'raw_bytes':sum(r['bytes'] for r in rows),'unique_files':len(unique),'unique_bytes':sum(r['bytes'] for r in unique.values())}),flush=True)
    archive=ROOT/'raw-snapshots.tar.gz';assert not archive.exists()
    cmd='tar -czf - -C /tmp/kda-rmsnorm-stage28 '+' '.join(shlex.quote(r['path']) for r in unique.values())
    with archive.open('xb') as out:
        r=subprocess.run(['ssh','-o','BatchMode=yes','bi-v150',cmd],stdout=out,stderr=subprocess.PIPE,timeout=900)
    assert r.returncode==0,r.stderr.decode()
    expected={r['path']:r for r in unique.values()}
    with tarfile.open(archive,'r:gz') as packed:
        for member in packed:
            assert member.isfile() and member.name in expected
            dest=ROOT/member.name;assert dest.resolve().is_relative_to(ROOT.resolve())
            dest.parent.mkdir(parents=True,exist_ok=True)
            assert not dest.exists()
            with dest.open('xb') as out:
                stream=packed.extractfile(member)
                for chunk in iter(lambda:stream.read(1024*1024),b''):out.write(chunk)
            row=expected[member.name];assert dest.stat().st_size==row['bytes'] and sha(dest)==row['sha256']
    for row in rows:
        dest=ROOT/row['path']
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            os.link(ROOT/unique[row['sha256']]['path'],dest)
        assert dest.stat().st_size==row['bytes'] and sha(dest)==row['sha256']
    receipt={'status':'verified','remote_root':'/tmp/kda-rmsnorm-stage28','remote_deleted':False,
             'files':rows,'verified_files':len(rows),'raw_bytes':sum(r['bytes'] for r in rows),
             'archive_sha256':sha(archive),'unique_files':len(unique)}
    (ROOT/'snapshot-backup-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','verified_files','raw_bytes','unique_files')}),flush=True)


if __name__=='__main__':main()

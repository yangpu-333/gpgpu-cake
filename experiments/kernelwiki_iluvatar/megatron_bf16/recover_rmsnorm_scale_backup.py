"""Resume verified files from a truncated archive, then fetch only missing hashes."""
import argparse,gzip,hashlib,json,os,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'evidence/rmsnorm-stage28'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--native',action='store_true');a=p.parse_args();root=ROOT/'raw-backup-native' if a.native else ROOT
 rows=json.loads((root/'snapshot-inventory.json').read_text());unique={}
 for r in rows:unique.setdefault(r['sha256'],r)
 if a.native:
  shared=json.loads((ROOT/'snapshot-inventory.json').read_text());by_hash={r['sha256']:r for r in shared}
  for r in rows:
   if r['sha256'] not in by_hash:continue
   source=ROOT/by_hash[r['sha256']]['path'];dest=root/r['path']
   if source.exists() and source.stat().st_size==r['bytes'] and sha(source)==r['sha256'] and not dest.exists():
    dest.parent.mkdir(parents=True,exist_ok=True);os.link(source,dest)
 archive=root/'raw-snapshots.tar.gz';expected={r['path']:r for r in unique.values()}
 if archive.exists():
  try:
   with tarfile.open(archive,mode='r|gz') as packed:
    for member in packed:
     assert member.isfile() and member.name in expected
     dest=root/member.name;assert dest.resolve().is_relative_to(root.resolve());r=expected[member.name]
     if dest.exists() and sha(dest)==r['sha256']:continue
     dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_suffix(dest.suffix+'.recovering')
     with temp.open('wb') as f:
      stream=packed.extractfile(member)
      for chunk in iter(lambda:stream.read(1024*1024),b''):f.write(chunk)
     if temp.stat().st_size==r['bytes'] and sha(temp)==r['sha256']:temp.replace(dest)
  except (EOFError,tarfile.ReadError,OSError) as e:print('Retained verified prefix; archive incomplete: '+type(e).__name__,flush=True)
 for digest,r in unique.items():
  dest=root/r['path'];dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists() and dest.stat().st_size==r['bytes'] and sha(dest)==digest:continue
  compressed=dest.with_suffix(dest.suffix+'.transfer.gz');temp=dest.with_suffix(dest.suffix+'.recovering')
  cmd='gzip -1 -c -- '+shlex.quote('/tmp/kda-rmsnorm-stage28/'+r['path'])
  with compressed.open('wb') as f:
   result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','bi-v150',cmd],stdout=f,stderr=subprocess.PIPE,timeout=900)
  assert result.returncode==0,result.stderr.decode()
  with gzip.open(compressed,'rb') as source,temp.open('wb') as f:
   for chunk in iter(lambda:source.read(1024*1024),b''):f.write(chunk)
  assert temp.stat().st_size==r['bytes'] and sha(temp)==digest
  temp.replace(dest);print(json.dumps({'verified':r['path'],'bytes':r['bytes']}),flush=True)
 for r in rows:
  dest=root/r['path'];dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():os.link(root/unique[r['sha256']]['path'],dest)
  assert dest.stat().st_size==r['bytes'] and sha(dest)==r['sha256']
 receipt={'status':'verified','remote_root':'/tmp/kda-rmsnorm-stage28','remote_deleted':False,'files':rows,'verified_files':len(rows),'raw_bytes':sum(r['bytes'] for r in rows),'unique_files':len(unique),'partial_archive_retained':True,'archive_sha256':sha(archive),'method':'Verified complete prefix files plus per-file gzip resumption; incomplete archive is not claimed complete'}
 (root/'snapshot-backup-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({k:receipt[k] for k in ('status','verified_files','raw_bytes','unique_files')}),flush=True)
if __name__=='__main__':main()

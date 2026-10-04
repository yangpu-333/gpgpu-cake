"""Fetch immutable small stage28 evidence; snapshot backup handled separately."""
import hashlib,io,subprocess,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent/'evidence/rmsnorm-stage28';root.mkdir(exist_ok=True)
incomplete={p.name for p in root.iterdir() if p.is_dir() and not (p/'command.json').exists()}
r=subprocess.run(['ssh','bi-v150','tar -cf - --exclude=__pycache__ --exclude=snapshots --exclude=deps --exclude=cpp-build-006 -C /tmp/kda-rmsnorm-stage28 .'],capture_output=True,timeout=120)
assert r.returncode==0,r.stderr.decode();count=0
with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:') as packed:
 for member in packed:
  if not member.isfile():continue
  relative=member.name.removeprefix('./');p=root/relative;assert p.resolve().is_relative_to(root.resolve())
  content=packed.extractfile(member).read()
  if p.exists() and Path(relative).parts[0] not in incomplete:assert p.read_bytes()==content,relative
  elif p.exists():p.write_bytes(content)
  else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content)
  count+=1
print('Small immutable files fetched:',count)

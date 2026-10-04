"""Fetch small immutable stage29 files; raw checkpoints backed up separately."""
import io
from pathlib import Path
import subprocess
import tarfile

root = Path(__file__).resolve().parent.parent / 'evidence/swiglu-stage29'
root.mkdir(exist_ok=True)
command = ['ssh', 'bi-v150', 'tar -cf - --exclude=__pycache__ --exclude=snapshots '
           '--exclude=activation --exclude=*.so --exclude=*.o --exclude=build.ninja '
           '--exclude=.ninja_* -C /tmp/kda-swiglu-stage29 .']
response = subprocess.run(command, capture_output=True, timeout=180)
assert response.returncode == 0, response.stderr.decode(errors='replace')
count = 0
with tarfile.open(fileobj=io.BytesIO(response.stdout), mode='r:') as packed:
    for member in packed:
        if not member.isfile():
            continue
        relative = member.name.removeprefix('./')
        path = root / relative
        assert path.resolve().is_relative_to(root.resolve()), relative
        content = packed.extractfile(member).read()
        if path.exists():
            assert path.read_bytes() == content, 'Immutable evidence changed: ' + relative
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        count += 1
print('Small immutable stage29 files fetched:', count)

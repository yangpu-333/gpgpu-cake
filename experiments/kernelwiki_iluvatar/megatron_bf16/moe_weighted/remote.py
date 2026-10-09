"""Small immutable deploy/read/run helper; no raw checkpoint transfer or GPU imports locally."""
import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
BASE = '/tmp/kda-moe-weighted-stage34'
ENV = 'env LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/openmpi/lib PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages '


def python(program):
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'bi-v150',
                        '/root/miniconda3/bin/python -'], input=program.encode(), capture_output=True, timeout=120)
    if r.returncode: raise RuntimeError(r.stderr.decode(errors='replace'))
    return json.loads(r.stdout)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('deploy', 'run', 'fetch'))
    p.add_argument('names', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.mode == 'deploy':
        payload = {}
        for name in a.names:
            rel = PurePosixPath(name)
            assert not rel.is_absolute() and '..' not in rel.parts
            data = (HERE / name).read_bytes()
            payload[name] = base64.b64encode(data).decode()
        program = """import json,base64,hashlib
from pathlib import Path
root=Path(BASE);rows=[]
for name,encoded in PAYLOAD.items():
 p=root/name;data=base64.b64decode(encoded)
 if p.exists():assert p.read_bytes()==data,str(p)
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 rows.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
print(json.dumps(rows))
""".replace('BASE', repr(BASE)).replace('PAYLOAD', repr(payload))
        print(json.dumps(python(program)))
    elif a.mode == 'run':
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'bi-v150',
                   ENV + '/root/miniconda3/bin/python -u ' + shlex.quote(BASE + '/' + a.names[0]) + ' ' +
                   ' '.join(shlex.quote(x) for x in a.names[1:])])
        return result.returncode
    else:
        program = """import json,base64
from pathlib import Path
root=Path(BASE);rows={}
for name in NAMES:
 p=root/name
 if p.is_file():
  assert p.stat().st_size<16*1024*1024 and not p.is_symlink()
  rows[name]=base64.b64encode(p.read_bytes()).decode()
print(json.dumps(rows))
""".replace('BASE', repr(BASE)).replace('NAMES', repr(a.names))
        values = python(program)
        for name, value in values.items():
            rel = PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
            path = HERE / 'evidence' / name;data=base64.b64decode(value)
            if path.exists():assert path.read_bytes()==data,str(path)
            else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        print(json.dumps({'fetched': len(values), 'bytes':sum(len(base64.b64decode(x)) for x in values.values())}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

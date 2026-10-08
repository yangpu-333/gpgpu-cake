"""Capture after-timing remote hashes for the three restored native dataset build files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


BASE = Path(__file__).resolve().parent / 'evidence'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='bi-v150')
    parser.add_argument('--timing-finished', action='store_true', required=True)
    args = parser.parse_args()
    before = json.loads((BASE / 'runtime-extra-before-perf.json').read_text(encoding='utf-8-sig'))
    program = """import fcntl,hashlib,json
from pathlib import Path
base=Path('/tmp/kda-attention-scale-stage32')
native=Path('/tmp/kda-bf16-continuation/megatron-native-5be9626')
with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for model in ('wide','deep'):
  value=json.loads((base/('performance-'+model+'-003-complete02.json')).read_text())
  assert value['status']=='validated' and len(value['completed_cases'])==9
 files={}
 for name in NAMES:
  p=native/name; content=p.read_bytes()
  files[name]={'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}
 value={'phase':'after-performance','megatron_commit':COMMIT,'files':files,
        'gpu_timing_finished_acknowledged':True,
        'scope':'Extra native dataset build files outside600-file manifest; remote reports, not local binary verification.'}
 print(json.dumps(value))
""".replace('NAMES', repr(sorted(before['files']))).replace('COMMIT', repr(before['megatron_commit']))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', args.host,
                             '/root/miniconda3/bin/python -'], input=program.encode(),
                            capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    after = json.loads(result.stdout)
    if after['files'] != before['files']:
        raise ValueError('extra runtime bytes changed')
    content = (json.dumps(after, indent=2) + '\n').encode()
    path = BASE / 'runtime-extra-after-perf.json'
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('immutable runtime receipt changed')
    else:
        path.write_bytes(content)
    print(json.dumps({'status': 'verified', 'path': str(path),
                      'sha256': hashlib.sha256(content).hexdigest(), 'files': len(after['files'])}))


if __name__ == '__main__':
    main()

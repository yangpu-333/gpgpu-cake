"""Freeze immutable sources and stage new files through the retained ACT bridge."""
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from drive_sft_pinning_validation import SSH, PREP, run

BASE = Path(__file__).resolve().parent
NEW = ('sft_numa_placement_20261021.py', 'sft_numa_entry_20261021.py',
       'run_sft_numa_case_20261021.py', 'probe_sft_numa_20261021.py',
       'continue_sft_numa_20261021.py', 'bootstrap_sft_numa_20261021.sh',
       'sft_numa_20261021.contract.json')


def main():
    prior = json.loads((BASE/'results/sft-paired-trace-source-manifest-20261020.json').read_text(encoding='utf-8'))['files']
    files = dict(prior)
    for name in (*prior, *NEW, 'numa_placement.py'):
        data = (BASE/name).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if name in prior and prior[name] != sha:
            raise ValueError('Previously frozen source changed: '+name)
        if name.endswith('.py'):
            compile(data, name, 'exec')
        files[name] = sha
    probe = subprocess.check_output([sys.executable, str(BASE/'probe_sft_numa_20261021.py')])
    result = json.loads(probe)
    if not result['passed'] or result['GPU_used']:
        raise ValueError('CPU-only topology/derivation gate failed')
    manifest = {'schema': 'jiuge7b-sft-numa-source-manifest-v1', 'files': files,
                'prior45_sources_unchanged': True, 'new_files': list(NEW),
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'CPU_preparation_probe': result, 'GPU_experiment_started': False}
    target = BASE/'results/sft-numa-source-manifest-20261021.json'
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8'))['files'] != files:
            raise ValueError('Never replace an existing frozen manifest')
    else:
        with target.open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, indent=2)
    payloads = {name: base64.b64encode((BASE/name).read_bytes()).decode() for name in NEW}
    code = '''import base64,hashlib,json,subprocess,sys
from pathlib import Path
p=Path('''+repr(PREP)+''')
files='''+repr(files)+'''
payloads='''+repr(payloads)+'''
for name,sha in files.items():
    path=p/name
    if path.exists():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha, name
    else:
        assert name in payloads,name
for name,b64 in payloads.items():
    path=p/name
    if not path.exists():
        with path.open('xb') as stream: stream.write(base64.b64decode(b64))
for name,sha in files.items():
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==sha,name
subprocess.run(['bash','-n',str(p/'bootstrap_sft_numa_20261021.sh')],check=True)
probe=json.loads(subprocess.check_output([sys.executable,str(p/'probe_sft_numa_20261021.py')]))
assert probe['passed'] and not probe['GPU_used']
print(json.dumps({'passed':True,'source_files_verified':len(files),'prior45_unchanged':True,
                  'CPU_preparation_probe':probe,'GPU_experiment_started':False}))
'''
    receipt = json.loads(run([*SSH, PREP+'/.venv/bin/python -'], input=code.encode()))
    receipt['utc'] = datetime.now(timezone.utc).isoformat()
    path = BASE/'results/sft-numa-preparation-20261021.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

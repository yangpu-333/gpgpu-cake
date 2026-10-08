"""Upload the six explicit stage32 controller/contract files without overwrite."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parent
EXPECTED = {
    'common.py': '02d3949c524ba451687bc6655bce09f664447456446b9f1c6eb44e55ed0abd0c',
    'launch_attention_scale.py': '1f528939a0da292910f264670c550a03f3eb7caa087209a8abb1d1046cb1fe2e',
    'run_audit.py': 'd22dfd86f48e40d1fa21c939dc888e90a111f9aa590fe9d330ab5f7a15f046d1',
    'run_performance.py': '99dd90aabd494a00f34f9e0ba3a0ef7c80920419cd099e4e7c2b3723b43e694e',
    'contracts/wide.json': 'd89816a795d1c9e7d1b46137186b34265b0256b9051d5d26466733bcd44c5b4f',
    'contracts/deep.json': 'a473489c036e86a9269b9d91b9e8c0223889fb464adf17ff1a14b4d3422d6162',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='bi-v150')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'immutable deployment receipt already exists'
    payload = {}
    for name, digest in EXPECTED.items():
        data = (ROOT / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, 'controller changed: ' + name
        payload[name] = base64.b64encode(data).decode()
    program = '''import sys,json,base64,hashlib
from pathlib import Path,PurePosixPath
root=Path('/tmp/kda-attention-scale-stage32');rows={}
for name,value in json.load(sys.stdin).items():
 rel=PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
 p=root.joinpath(*rel.parts);assert p.resolve().is_relative_to(root.resolve())
 data=base64.b64decode(value,validate=True)
 if p.exists():assert not p.is_symlink() and p.read_bytes()==data
 else:
  p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as stream:stream.write(data)
 rows[name]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps({'status':'frozen_bytes_deployed','files':rows}))
'''
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', args.host,
                             '/root/miniconda3/bin/python -c ' + shlex.quote(program)],
                            input=json.dumps(payload).encode(), capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    receipt = json.loads(result.stdout)
    assert receipt['files'] == EXPECTED
    receipt.update(candidate_version='003',
                   candidate_sha256='4881f91c230324f258cf6faf7b1d63a493b043b960df4efdca0dad708195db48',
                   candidate_retuned=False, swiglu_candidate_enabled=False,
                   deployment_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

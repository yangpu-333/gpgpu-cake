"""Restore only pinned experiment dependencies and exact native Git blobs.

No credentials, caches, prior training checkpoints or installed packages are
copied. The in-memory transfer archive is never retained on either host.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


SOURCE = Path(__file__).resolve().parent
BF = SOURCE.parent
ROOT = BF.parent
COMMIT = '5be9626709af2722333bf54797c954c09edeada3'
TARGETS = {
    'native': '/tmp/kda-bf16-continuation/megatron-native-5be9626',
    'bf16': '/tmp/kda-bf16-continuation/megatron_bf16',
    'formal': '/tmp/kda-bf16-continuation/megatron_training_entry',
    'gradnorm': '/tmp/kda-gradnorm-stage26',
    'stage31': '/tmp/kda-attention-stage31',
    'stage29': '/tmp/kda-swiglu-stage29',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def collect(git_source):
    manifest = json.loads((ROOT / 'megatron_training_entry/source_manifest.json').read_text())
    assert manifest['commit'] == COMMIT and len(manifest['files']) == 600
    names = list(manifest['files'])
    batch = subprocess.run(['git', '-C', str(git_source), 'cat-file', '--batch'],
                           input=''.join(COMMIT + ':' + name + '\n' for name in names).encode(),
                           check=True, capture_output=True)
    stream = io.BytesIO(batch.stdout)
    payload = {}
    for name in names:
        fields = stream.readline().split()
        assert len(fields) == 3 and fields[1] == b'blob', name
        data = stream.read(int(fields[2]))
        assert stream.read(1) == b'\n' and sha(data) == manifest['files'][name], name
        payload['native/' + name] = data
    assert not stream.read()
    payload['native/.git/HEAD'] = (COMMIT + '\n').encode()
    selected = {
        'bf16': (BF, ['compare_training.py', 'summarize_performance.py',
                      'launch_training.py', 'kernel_adapter.py', 'run_case.py',
                      'candidates/0008/candidate.py']),
        'formal': (ROOT / 'megatron_training_entry',
                   ['runtime_compat.py', 'source_manifest.json', 'summarize_performance.py']),
        'gradnorm': (BF / 'evidence/gradnorm-stage26', ['launch_gradnorm-04.py']),
        'stage31': (BF / 'attention', ['common.py', 'launch_attention.py', 'run_audit.py',
                                      'run_performance.py', 'probe_full.py', 'probe_native_backend.py',
                                      'launch_capture.py', 'run_capture_default.py', 'contracts/main.json']),
        'stage29': (BF / 'evidence/swiglu-stage29',
                    ['candidate-004.py', 'launch_swiglu-03.py', 'probe-004-v3.json',
                     'decision.json', 'contracts/main.json']),
    }
    for group, (folder, relatives) in selected.items():
        for relative in relatives:
            payload[group + '/' + relative] = (folder / relative).read_bytes()
    payload['gradnorm/candidate.py'] = (BF / 'gradnorm_candidates/001/candidate.py').read_bytes()
    payload['stage31/candidate-003.py'] = (BF / 'attention/candidates/003/candidate.py').read_bytes()
    payload['stage31/generation-003.json'] = (BF / 'attention/candidates/003/generation.json').read_bytes()
    archived = BF / 'attention/evidence/round003'
    for relative in ('backend-default.json', 'probe-full-003.json',
                     'native-default-capture-02/attention.json',
                     'audit-summary-main-003-complete.json', 'performance-main-003-complete.json'):
        payload['stage31/' + relative] = (archived / relative).read_bytes()
    for model in ('wide', 'deep'):
        relative = 'contracts/' + model + '.json'
        payload['stage29/' + relative] = (BF / 'evidence/swiglu-stage29' / relative).read_bytes()
    capture = json.loads(payload['stage31/native-default-capture-02/attention.json'])
    for name, row in capture['raw_files'].items():
        relative = 'native-default-capture-02/attention/' + name
        data = (archived / 'raw-backup' / relative).read_bytes()
        assert sha(data) == row['sha256'] and len(data) == row['bytes']
        payload['stage31/' + relative] = data
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='bi-v150')
    parser.add_argument('--git-source', type=Path,
                        default=ROOT.parents[1] / 'research/code/megatron-lm-core-v0.19.0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'immutable restoration receipt already exists'
    payload = collect(args.git_source)
    packed = io.BytesIO()
    with tarfile.open(fileobj=packed, mode='w:gz') as archive:
        for name, data in sorted(payload.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    program = '''import sys,tarfile,json,hashlib
from pathlib import Path,PurePosixPath
targets=TARGETS
rows=[]
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as packed:
 for member in packed:
  rel=PurePosixPath(member.name)
  assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts and rel.parts[0] in targets
  root=Path(targets[rel.parts[0]]);p=root.joinpath(*rel.parts[1:]);assert p.resolve().is_relative_to(root.resolve())
  data=packed.extractfile(member).read();assert len(data)==member.size
  if p.exists():assert p.is_file() and not p.is_symlink() and p.read_bytes()==data, str(p)
  else:
   p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('xb') as stream:stream.write(data)
  rows.append({'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
print(json.dumps({'status':'restored_exact_bytes','files':rows}))
'''.replace('TARGETS', repr(TARGETS))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', args.host,
                             '/root/miniconda3/bin/python -c ' + shlex.quote(program)],
                            input=packed.getvalue(), capture_output=True, timeout=180)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    receipt = json.loads(result.stdout)
    expected = {TARGETS[group] + '/' + relative: {'bytes': len(data), 'sha256': sha(data)}
                for name, data in payload.items() for group, relative in [name.split('/', 1)]}
    assert {row['path']: {'bytes': row['bytes'], 'sha256': row['sha256']}
            for row in receipt['files']} == expected
    receipt.update(megatron_commit=COMMIT, native_source_files=600,
                   native_snapshot_mode='Exact cat-file Git blobs with commit HEAD marker; no Git objects copied',
                   historical_stage31_probe_and_capture_restored=True,
                   fresh_stage32_training_not_started=True, transfer_archive_retained=False,
                   restored_package_installations=False,
                   script_sha256=sha(Path(__file__).read_bytes()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': receipt['status'], 'files': len(receipt['files']),
                      'native_sources': 600, 'transfer_archive_retained': False}))


if __name__ == '__main__':
    main()

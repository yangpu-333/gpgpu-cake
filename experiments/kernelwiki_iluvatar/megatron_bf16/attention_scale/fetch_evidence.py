"""Fetch immutable small stage32 receipts only when explicitly executed.

No GPU work or model API calls. Tensor/checkpoint/binary bytes are handled by
backup_raw.py. Archives may contain only the selected model/round/suffix and
the common frozen controller files; different bytes never overwrite a file.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile


SOURCE = Path(__file__).resolve().parent
REMOTE = '/tmp/kda-attention-scale-stage32'
VERSION = '003'
LIMIT = 16 * 1024 * 1024
FIXED = {'common.py', 'launch_attention_scale.py', 'run_audit.py', 'run_performance.py',
         'contracts/wide.json', 'contracts/deep.json',
         'backend-current-20261008-01.json', 'probe-full-0032026100801.json',
         'probe-fresh-20261008-01.stdout.log', 'probe-fresh-20261008-01.stderr.log',
         'probe-fresh-20261008-02.stdout.log', 'probe-fresh-20261008-02.stderr.log'}
CASE_FILES = {'command.json', 'entry.json', 'attention.json', 'gradnorm.json',
              'swiglu.json', 'stdout.log', 'stderr.log'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def allowed_names(model, audit_suffix, run_suffix, seeds):
    names = set(FIXED)
    names |= {f'audit-summary-{model}-003-{audit_suffix}.json',
              f'comparison-{model}-003-{audit_suffix}.json',
              f'comparison-native-optimized-{model}-003-{audit_suffix}.json',
              f'comparison-native-control-{model}-003-{audit_suffix}.json',
              f'performance-{model}-003-{run_suffix}.json',
              f'admission-{model}-003-{audit_suffix}.json'}
    for arm in ('native', 'control', 'optimized'):
        cases = [f'{model}-{arm}-audit-003-{audit_suffix}']
        cases += [f'{model}-{arm}-perf-003-{run_suffix}-{seed}' for seed in seeds]
        names |= {case + '/' + name for case in cases for name in CASE_FILES}
    return names


def validate_row(row, allowed):
    name = row['path']
    rel = PurePosixPath(name)
    require(not rel.is_absolute() and '..' not in rel.parts and name in allowed,
            'archive member escaped the selected model/round: ' + name)
    require(type(row['bytes']) is int and 0 <= row['bytes'] <= LIMIT
            and isinstance(row['sha256'], str)
            and re.fullmatch(r'[0-9a-f]{64}', row['sha256']) is not None,
            'invalid small-file byte/hash inventory')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('wide', 'deep'), required=True)
    parser.add_argument('--version', choices=(VERSION,), default=VERSION)
    parser.add_argument('--audit-suffix', default='complete')
    parser.add_argument('--run-suffix', default='complete')
    parser.add_argument('--base', type=Path, default=SOURCE / 'evidence')
    parser.add_argument('--host', default='bi-v150')
    parser.add_argument('--timing-finished', action='store_true', required=True,
                        help='Confirm no GPU performance process is active during transfer.')
    args = parser.parse_args()
    require(all(re.fullmatch(r'[A-Za-z0-9-]+', value)
                for value in (args.audit_suffix, args.run_suffix)), 'simple immutable suffix required')
    # These must agree with the sealed model contracts. Read JSON, never import runners.
    contract = json.loads((SOURCE / 'contracts' / (args.model + '.json')).read_text(encoding='utf-8'))
    seeds = contract['performance']['independent_seeds']
    require(len(seeds) == 3 and len(set(seeds)) == 3 and all(type(seed) is int and seed > 0 for seed in seeds),
            'three sealed independent seeds required')
    allowed = allowed_names(args.model, args.audit_suffix, args.run_suffix, seeds)
    base = args.base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    program = '''import hashlib,json
from pathlib import Path
root=Path(REMOTE);allowed=ALLOWED;rows=[]
for name in sorted(allowed):
 p=root/name
 if not p.is_file():continue
 if p.is_symlink() or not p.resolve().is_relative_to(root):raise RuntimeError('unsafe source')
 before=p.stat()
 if before.st_size>LIMIT:raise RuntimeError('small file exceeds limit: '+name)
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 after=p.stat()
 if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('source changed during inventory')
 rows.append({'path':name,'bytes':after.st_size,'sha256':h.hexdigest()})
print(json.dumps(rows))
'''.replace('REMOTE', repr(REMOTE)).replace('ALLOWED', repr(sorted(allowed))).replace('LIMIT', str(LIMIT))
    inventory_process = subprocess.run(['ssh', '-o', 'BatchMode=yes', args.host,
                                        '/root/miniconda3/bin/python -'], input=program.encode(),
                                       capture_output=True, timeout=120)
    require(inventory_process.returncode == 0, inventory_process.stderr.decode(errors='replace'))
    rows = json.loads(inventory_process.stdout)
    require(isinstance(rows, list) and rows, 'nonempty selected small-file inventory required')
    inventory = {}
    for row in rows:
        validate_row(row, allowed)
        require(row['path'] not in inventory, 'duplicate small-file inventory member')
        inventory[row['path']] = row
    command = 'tar -cf - -C ' + shlex.quote(REMOTE) + ' -- ' + ' '.join(shlex.quote(name) for name in inventory)
    process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', args.host, command],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    seen = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as packed:
            for member in packed:
                require(member.isfile() and member.name in inventory and member.name not in seen,
                        'unexpected, linked or duplicate transfer member')
                row = inventory[member.name]
                require(member.size == row['bytes'], 'transfer byte size differs from remote inventory')
                content = packed.extractfile(member).read()
                require(len(content) == row['bytes'] and hashlib.sha256(content).hexdigest() == row['sha256'],
                        'transferred bytes differ from immutable remote inventory')
                destination = base.joinpath(*PurePosixPath(member.name).parts)
                require(destination.resolve().is_relative_to(base), 'transfer path escaped destination')
                if destination.exists():
                    require(destination.is_file() and destination.read_bytes() == content,
                            'immutable local evidence changed: ' + member.name)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open('xb') as stream:
                        stream.write(content)
                seen.add(member.name)
        require(process.wait(timeout=20) == 0 and seen == set(inventory), 'small evidence transfer incomplete')
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    result = {'schema': 'frozen-attention-stage32-small-fetch-v1', 'status': 'fetched',
              'model_name': args.model, 'version': VERSION, 'audit_suffix': args.audit_suffix,
              'run_suffix': args.run_suffix, 'remote_root': REMOTE,
              'files': sorted(rows, key=lambda row: row['path']), 'fetch_script_sha256': sha(__file__),
              'raw_snapshots_fetched': False, 'transfer_archive_retained': False, 'remote_deleted': False,
              'gpu_timing_finished_acknowledged': True}
    digest = hashlib.sha256(json.dumps(result['files'], sort_keys=True).encode()).hexdigest()
    receipt = base / f'small-fetch-{args.model}-003-{args.audit_suffix}-{args.run_suffix}-{digest[:16]}-receipt.json'
    if receipt.exists():
        require(json.loads(receipt.read_text(encoding='utf-8')) == result, 'immutable fetch receipt changed')
    else:
        with receipt.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps({'status': 'fetched', 'model_name': args.model, 'version': VERSION,
                      'files': len(rows), 'receipt': str(receipt)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

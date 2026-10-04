"""Fetch small immutable stage30 evidence; raw tensors are backed up separately.

This script makes SSH calls only when explicitly executed. It streams a tar
transfer without retaining a duplicate archive, refuses changed local evidence,
and never fetches credential directories, candidate caches or tensor snapshots.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile


SOURCE = Path(__file__).resolve().parent
REMOTE = '/tmp/kda-swiglu-scale-stage30'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('wide', 'deep'), required=True)
    parser.add_argument('--base', type=Path, default=SOURCE / 'evidence')
    parser.add_argument('--host', default='bi-v150')
    parser.add_argument('--audit-suffix', default='complete')
    parser.add_argument('--run-suffix', default='complete')
    args = parser.parse_args()
    require(all(value.replace('-', '').isalnum() for value in (args.audit_suffix, args.run_suffix)),
            'simple immutable audit/run suffixes required')
    base = args.base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    # Only experiment output and the explicitly frozen controller/contract set
    # can enter this transfer. Dependencies in other roots remain local pins.
    program = '''import json
from pathlib import Path
root=Path(REMOTE); model=MODEL; audit_suffix=AUDIT_SUFFIX;run_suffix=RUN_SUFFIX;rows=[]
fixed={'common.py','run_audit.py','run_performance.py','adaptation-record.json',
       'contracts/wide.json','contracts/deep.json','controller-repair.json',
       'audit-summary-'+model+'-004-'+audit_suffix+'.json',
       'comparison-'+model+'-004-'+audit_suffix+'.json',
       'comparison-native-optimized-'+model+'-004-'+audit_suffix+'.json',
       'comparison-native-control-'+model+'-004-'+audit_suffix+'.json',
       'performance-'+model+'-004-'+run_suffix+'.json'}
case_names={model+'-'+arm+'-audit-004-'+audit_suffix for arm in ('native','control','optimized')}
case_prefixes=tuple(model+'-'+arm+'-perf-004-'+run_suffix+'-' for arm in ('native','control','optimized'))
for p in sorted(root.rglob('*')):
 if not p.is_file() or p.is_symlink(): continue
 rel=p.relative_to(root); name=rel.as_posix()
 if any(x in rel.parts for x in ('snapshots','activation','__pycache__','raw-backup')): continue
 if p.suffix in ('.pt','.pth','.so','.o','.a','.tar','.gz','.zip','.pyc'): continue
 if name not in fixed and not (rel.parts[0] in case_names or rel.parts[0].startswith(case_prefixes)): continue
 if p.stat().st_size>16777216: raise RuntimeError('small file exceeds 16MiB: '+name)
 rows.append(name)
print(json.dumps(rows))
'''.replace('REMOTE', repr(REMOTE)).replace('MODEL', repr(args.model)).replace('AUDIT_SUFFIX', repr(args.audit_suffix)).replace('RUN_SUFFIX', repr(args.run_suffix))
    listed = subprocess.run(['ssh', '-o', 'BatchMode=yes', args.host,
                             '/root/miniconda3/bin/python -'], input=program.encode(),
                            capture_output=True, timeout=120)
    require(listed.returncode == 0, listed.stderr.decode(errors='replace'))
    names = json.loads(listed.stdout)
    require(isinstance(names, list) and names and len(names) == len(set(names)),
            'nonempty unique small-evidence inventory required')
    for name in names:
        rel = PurePosixPath(name)
        require(not rel.is_absolute() and '..' not in rel.parts, 'unsafe transfer member')
    command = 'tar -cf - -C ' + shlex.quote(REMOTE) + ' -- ' + ' '.join(shlex.quote(name) for name in names)
    process = subprocess.Popen(['ssh', '-o', 'BatchMode=yes', args.host, command],
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    seen, files = set(), []
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as packed:
            for member in packed:
                require(member.isfile() and member.name in names and member.name not in seen,
                        'unexpected, linked or duplicate transfer member: ' + member.name)
                require(member.size <= 16 * 1024 * 1024, 'unexpected large transfer member')
                seen.add(member.name)
                rel = PurePosixPath(member.name)
                destination = base.joinpath(*rel.parts)
                require(destination.resolve().is_relative_to(base), 'transfer path escaped destination')
                content = packed.extractfile(member).read()
                require(len(content) == member.size, 'incomplete transferred file')
                if destination.exists():
                    require(destination.is_file() and destination.read_bytes() == content,
                            'immutable local evidence changed: ' + member.name)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open('xb') as stream:
                        stream.write(content)
                files.append({'path': member.name, 'bytes': len(content),
                              'sha256': hashlib.sha256(content).hexdigest()})
        require(process.wait(timeout=20) == 0 and seen == set(names), 'small evidence transfer incomplete')
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    result = {'status': 'fetched', 'model_name': args.model, 'remote_root': REMOTE,
              'audit_suffix': args.audit_suffix, 'run_suffix': args.run_suffix,
              'files': sorted(files, key=lambda row: row['path']), 'fetch_script_sha256': sha(__file__),
              'raw_snapshots_fetched': False, 'transfer_archive_retained': False,
              'remote_deleted': False}
    receipt = base / ('small-fetch-' + args.model + '-receipt.json')
    if receipt.exists():
        require(json.loads(receipt.read_text(encoding='utf-8')) == result,
                'refusing to replace a different immutable fetch receipt')
    else:
        with receipt.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps({'status': 'fetched', 'model_name': args.model, 'files': len(files),
                      'receipt': str(receipt)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

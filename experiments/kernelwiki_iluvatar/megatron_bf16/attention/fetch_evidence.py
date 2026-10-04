"""Stream only small immutable stage31 evidence when explicitly executed.

No tensor/checkpoint/binary or credential directories enter this transfer.
Repeated fetches add immutable receipts for each distinct small-file inventory.
This script performs no GPU work or model API calls.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile


SOURCE = Path(__file__).resolve().parent
REMOTE = '/tmp/kda-attention-stage31'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--base', type=Path, default=SOURCE / 'evidence')
    parser.add_argument('--host', default='bi-v150')
    args = parser.parse_args()
    require(args.version.isdigit() and len(args.version) == 3, 'three-digit candidate version required')
    base = args.base.resolve()
    base.mkdir(parents=True, exist_ok=True)
    program = '''import json
from pathlib import Path
root=Path(REMOTE);version=VERSION;rows=[]
fixed={'common.py','launch_attention.py','run_audit.py','run_performance.py',
       'controller-repair.json','controller_v1/run_performance.py',
       'probe_full.py','probe_native_backend.py','contracts/main.json',
       'contracts/native-default-preflight.json','backend-default.json',
       'launch_capture.py','run_capture_default.py',
       'candidate-'+version+'.py','generation-'+version+'.json'}
for p in sorted(root.rglob('*')):
 if not p.is_file() or p.is_symlink():continue
 rel=p.relative_to(root);name=rel.as_posix()
 if any(x in rel.parts for x in ('snapshots','attention','activation','profile','__pycache__','raw-backup')):continue
 if p.suffix in ('.pt','.pth','.so','.o','.a','.tar','.gz','.zip','.pyc'):continue
 top=rel.parts[0]
 selected=name in fixed or top=='native-default-capture-02' or (
  top.startswith('main-') and ('-'+version+'-') in top) or (
  len(rel.parts)==1 and ((name.startswith('probe-full-'+version) and p.suffix in ('.json','.log')) or
  (name.startswith(('audit-summary-','comparison-','performance-','admission-','decision-')) and
   ('-'+version+'-') in name)))
 if not selected:continue
 if p.stat().st_size>16777216:raise RuntimeError('small file exceeds16MiB: '+name)
 rows.append(name)
print(json.dumps(rows))
'''.replace('REMOTE', repr(REMOTE)).replace('VERSION', repr(args.version))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', args.host,
                             '/root/miniconda3/bin/python -'], input=program.encode(),
                            capture_output=True, timeout=120)
    require(result.returncode == 0, result.stderr.decode(errors='replace'))
    names = json.loads(result.stdout)
    require(isinstance(names, list) and names and len(names) == len(set(names)),
            'nonempty unique small inventory required')
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
                        'unexpected, linked or duplicate small member')
                require(member.size <= 16 * 1024 * 1024, 'unexpected large member')
                seen.add(member.name)
                destination = base.joinpath(*PurePosixPath(member.name).parts)
                require(destination.resolve().is_relative_to(base), 'transfer path escaped root')
                content = packed.extractfile(member).read()
                require(len(content) == member.size, 'incomplete small file')
                if destination.exists():
                    require(destination.is_file() and destination.read_bytes() == content,
                            'immutable local evidence changed: ' + member.name)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open('xb') as stream:
                        stream.write(content)
                files.append({'path': member.name, 'bytes': len(content),
                              'sha256': hashlib.sha256(content).hexdigest()})
        require(process.wait(timeout=20) == 0 and seen == set(names), 'small transfer incomplete')
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    result = {'status': 'fetched', 'version': args.version, 'remote_root': REMOTE,
              'files': sorted(files, key=lambda row: row['path']), 'fetch_script_sha256': sha(__file__),
              'raw_snapshots_fetched': False, 'transfer_archive_retained': False, 'remote_deleted': False}
    digest = hashlib.sha256(json.dumps(result['files'], sort_keys=True).encode()).hexdigest()
    receipt = base / ('small-fetch-' + args.version + '-' + digest[:16] + '-receipt.json')
    if receipt.exists():
        require(json.loads(receipt.read_text(encoding='utf-8')) == result, 'immutable fetch receipt changed')
    else:
        with receipt.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    print(json.dumps({'status': 'fetched', 'version': args.version,
                      'files': len(files), 'receipt': str(receipt)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

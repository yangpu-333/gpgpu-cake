"""Run a fresh immutable heldout model audit with the frozen stage34 candidate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parent.parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('long', 'experts'), required=True)
    args = parser.parse_args()
    root = BASE / args.model
    contract = read(root/'contract.json')
    frozen = read(BASE/'controllers.json')
    assert all(sha(BASE/name) == digest for name, digest in frozen['files'].items())
    candidate = root/'candidates/004/candidate.py'
    assert sha(candidate) == 'b9a2ca76c294481e38e2422332ccb821224bd64b8bb7cd61d972a5e2a4a9133e'
    backend = BASE/'long/backend-current.json'
    assert read(backend)['passed'] and len(read(backend)['sort_index_probes']) == 40
    assert read(backend)['script_sha256']==sha(BASE/'long/backend.py')
    output = root/'audit-orchestration.json'
    assert not output.exists()
    record = {'schema': 'moe-heldout-stage35-audit-orchestration-v1', 'model': args.model,
              'status': 'running', 'candidate_sha256': sha(candidate),
              'contract_sha256': sha(root/'contract.json'), 'controllers_sha256': sha(BASE/'controllers.json'),
              'backend_sha256': sha(backend), 'completed': [],
              'started_utc': datetime.now(timezone.utc).isoformat()}
    def execute(name, arguments):
        process = subprocess.run([sys.executable, '-u', str(root/name), *arguments])
        record['completed'].append({'script': name, 'arguments': arguments, 'exit_code': process.returncode})
        if process.returncode:
            raise RuntimeError(name + ' failed with exit ' + str(process.returncode))
    try:
        seed = str(contract['audit_seed'])
        calls = contract['model']['num_layers'] * 3 * contract['model']['num_moe_experts']
        execute('run_case.py', ['--case', 'native-admission01', '--arm', 'native', '--seed', seed,
                                '--capture-limit', str(calls)])
        execute('run_case.py', ['--case', 'control-admission01', '--arm', 'control', '--seed', seed])
        execute('compare.py', ['--left', 'native-admission01', '--right', 'control-admission01',
                               '--left-arm', 'native', '--right-arm', 'control', '--output', 'comparison-admission01.json'])
        execute('probe.py', ['--version', '004'])
        execute('probe_offset.py', ['--version', '004'])
        execute('replay.py', ['--version', '004'])
        execute('run_case.py', ['--case', 'optimized-audit004', '--arm', 'optimized', '--seed', seed,
                                '--candidate', str(candidate)])
        for name, left, arm in (('comparison-native-004.json', 'native-admission01', 'native'),
                                ('comparison-control-004.json', 'control-admission01', 'control')):
            execute('compare.py', ['--left', left, '--right', 'optimized-audit004',
                                   '--left-arm', arm, '--right-arm', 'optimized', '--output', name])
        execute('seal_audit.py', ['--version', '004'])
        record['status'] = 'passed'
    except Exception as error:
        record.update(status='rejected', error_type=type(error).__name__, error=str(error))
    record['finished_utc'] = datetime.now(timezone.utc).isoformat()
    output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'model': args.model, 'status': record['status'], 'error': record.get('error')}), flush=True)
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

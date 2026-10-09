"""Capture complete SSH orchestration stdout/stderr without changing GPU programs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=('long', 'experts'), required=True)
    p.add_argument('--phase', choices=('audit', 'performance'), required=True)
    a = p.parse_args()
    folder = HERE/'models'/a.model/'evidence'
    folder.mkdir(exist_ok=True)
    output = folder/(a.phase+'-controller.stdout.log')
    errors = folder/(a.phase+'-controller.stderr.log')
    receipt = folder/(a.phase+'-controller.json')
    assert not any(path.exists() for path in (output, errors, receipt))
    arguments = ['run_audit.py', '--model', a.model] if a.phase == 'audit' else [
        'run_performance.py', '--version', '004', '--audit', 'audit-004.json']
    command = [sys.executable, str(HERE/'remote.py'), a.model, 'run', *arguments]
    with output.open('xb') as stdout, errors.open('xb') as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    record = {'schema': 'stage35-local-ssh-console-v1', 'model': a.model, 'phase': a.phase,
              'exit_code': result.returncode, 'command': command,
              'stdout_sha256': sha(output), 'stderr_sha256': sha(errors),
              'capture_script_sha256': sha(Path(__file__))}
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))
    print(output.read_text(encoding='utf-8', errors='replace')[-1800:])
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())

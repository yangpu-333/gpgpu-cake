"""Run one frozen recovery step after a fresh, conservative occupancy check."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path('/private/kda-biv150-stage35-recovery01')
PYTHON = '/root/miniconda3/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=('backend', 'long', 'experts'))
    args = parser.parse_args()
    prefix = ROOT / ('browser-' + args.step + '-retry01-job-01')
    receipt = Path(str(prefix) + '.json')
    assert ROOT.is_dir() and ROOT.resolve() == Path('/mnt/kda-biv150-stage35-recovery01')
    assert not receipt.exists()
    record = {'schema': 'browser-recovery-step-v1', 'step': args.step, 'own_pid': os.getpid(),
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'controller_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'status': 'checking_occupancy', 'gpu_started': False}
    def save():
        receipt.write_text(json.dumps(record, indent=2) + '\n')
    save()
    try:
        gpu = subprocess.run(['/usr/local/corex/bin/ixsmi'], text=True,
                             capture_output=True, timeout=20)
        ps = subprocess.run(['ps', '-eo', 'pid,ppid,user,etimes,pcpu,pmem,comm'],
                            text=True, capture_output=True, timeout=20)
        record['occupancy'] = {'utc': datetime.now(timezone.utc).isoformat(),
                               'gpu_returncode': gpu.returncode, 'gpu_stdout': gpu.stdout,
                               'gpu_stderr': gpu.stderr, 'process_returncode': ps.returncode,
                               'process_stdout': ps.stdout}
        assert gpu.returncode == ps.returncode == 0
        devices = re.findall(r'(\d+)MiB\s*/\s*32768MiB\s*\|\s*(\d+)%', gpu.stdout)
        assert len(devices) == 2 and all(int(mem) <= 256 and int(util) == 0 for mem, util in devices), 'GPU not idle'
        assert 'No running processes found' in gpu.stdout, 'Foreign GPU process found'
        foreign = []
        for line in ps.stdout.splitlines()[1:]:
            fields = line.split()
            if len(fields) < 7 or int(fields[0]) == os.getpid():
                continue
            if fields[6].startswith(('python', 'torchrun', 'triton')) or float(fields[4]) >= 10:
                foreign.append({'pid': int(fields[0]), 'command': fields[6], 'cpu_percent': float(fields[4])})
        record['foreign_compute_processes'] = foreign
        assert not foreign, 'Another compute process is active; leaving it untouched'
        if args.step == 'backend':
            script = ROOT / 'models-retry01/long/backend.py'
            output = ROOT / 'models-retry01/long/backend-current.json'
            argv = ['timeout', '180', PYTHON, '-u', str(script)]
        else:
            script = ROOT / ('models-retry01/' + args.step + '/run_audit.py')
            output = ROOT / ('models-retry01/' + args.step + '/audit-orchestration.json')
            backend = json.loads((ROOT / 'models-retry01/long/backend-current.json').read_text())
            assert backend['passed'] and backend['torch'].startswith('2.4.1')
            assert len(backend['sort_index_probes']) == 40
            argv = ['timeout', '3600', PYTHON, '-u', str(script), '--model', args.step]
        assert not output.exists(), 'Existing result must not be overwritten'
        record['argv'] = argv
        record['status'] = 'running'
        record['gpu_started'] = True
        save()
        env = os.environ.copy()
        env.update(LD_LIBRARY_PATH='/usr/local/corex/lib64:/usr/local/openmpi/lib',
                   PYTHONPATH='/usr/local/corex/lib64/python3/dist-packages',
                   CUDA_VISIBLE_DEVICES='0')
        with Path(str(prefix) + '.stdout.log').open('x') as stdout, Path(str(prefix) + '.stderr.log').open('x') as stderr:
            process = subprocess.run(argv, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr)
        record['exit_code'] = process.returncode
        assert process.returncode == 0, 'Frozen controller failed; inspect owned logs'
        result = json.loads(output.read_text())
        assert result.get('passed') is True or result.get('status') == 'passed'
        record['result_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
        record['status'] = 'passed'
    except Exception as error:
        record['status'] = 'failed' if record['gpu_started'] else 'blocked_before_gpu'
        record['error'] = str(error)
    finally:
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({'step': args.step, 'status': record['status'], 'error': record.get('error')}), flush=True)
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

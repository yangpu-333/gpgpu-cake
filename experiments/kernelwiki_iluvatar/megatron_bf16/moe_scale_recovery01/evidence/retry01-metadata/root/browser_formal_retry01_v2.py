"""External gates around the byte-identical three-arm formal controller."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import time

ROOT = Path('/private/kda-biv150-stage35-recovery01')
PYTHON = '/root/miniconda3/bin/python'
MANIFEST_SHA = '533af8c2a305d0c92f03c585505e60d16a47c6998f9f5c8b618d368cb222ec10'
PLAN_SHA = '93366e057a1814c484a6bc0f7baa3bb05b5e21c8fb59faa5a2c58e8fbd3ea6cb'
SOURCE_SHA = 'a72ded960488d399ded2255ef55f357257e03532b8fa21a06d958170c5c98c2d'
utc = lambda: datetime.now(timezone.utc).isoformat()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())


def occupancy():
    gpu = subprocess.run(['/usr/local/corex/bin/ixsmi'], capture_output=True, text=True, timeout=20)
    processes = subprocess.run(['ps', '-eo', 'pid,ppid,user,etimes,pcpu,pmem,stat,comm'], capture_output=True, text=True, timeout=20)
    devices = re.findall(r'(\d+)MiB\s*/\s*32768MiB\s*\|\s*(\d+)%', gpu.stdout)
    foreign = []
    for line in processes.stdout.splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 8 and not fields[6].startswith('Z') and int(fields[0]) != os.getpid() and (
                fields[7].startswith(('python', 'torchrun', 'triton')) or float(fields[4]) >= 10):
            foreign.append({'pid': int(fields[0]), 'command': fields[7], 'process_state': fields[6], 'cpu_percent': float(fields[4])})
    with socket.socket() as port:
        port.settimeout(1)
        port_free = port.connect_ex(('127.0.0.1', 29941)) != 0
    idle = gpu.returncode == processes.returncode == 0 and len(devices) == 2 and all(
        int(memory) <= 256 and int(utilization) == 0 for memory, utilization in devices)
    idle = idle and 'No running processes found' in gpu.stdout
    return {'utc': utc(), 'gpu_stdout': gpu.stdout, 'process_stdout': processes.stdout,
            'both_gpus_idle': idle, 'foreign_compute_processes': foreign,
            'training_port_free': port_free, 'passed': idle and not foreign and port_free}


def cpu_boundary_sample():
    # Outside retained timing only. No profiler, tensor dump or background polling.
    before = Path('/proc/stat').read_text().splitlines()[0]
    time.sleep(2)
    after = Path('/proc/stat').read_text().splitlines()[0]
    return {'utc': utc(), 'before': before, 'after': after, 'outside_formal_timing': True}


def cvs(performance):
    return {arm: statistics.stdev([p['cases'][arm]['mean_iteration_ms'] for p in performance['pairs']]) /
            statistics.mean([p['cases'][arm]['mean_iteration_ms'] for p in performance['pairs']])
            for arm in ('native', 'control', 'optimized')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('long', 'experts'), required=True)
    parser.add_argument('--backup-sha256', required=True)
    parser.add_argument('--round', choices=('primary', 'confirmation', 'resource_confirmation', 'confirmation2'), default='primary')
    args = parser.parse_args()
    prefix = ROOT / ('browser-formal-' + args.model + '-retry01-' + args.round + '-02')
    receipt = Path(str(prefix) + '.json')
    assert not receipt.exists(), 'Existing wrapper receipt is immutable'
    record = {'schema': 'retry01-formal-wrapper-v2', 'parent_wrapper_sha256': '5a59d06701b6e57672e09d5da7968676949007c4cafd05449667aefe8a05113f', 'model': args.model, 'round': args.round,
              'own_pid': os.getpid(), 'wrapper_sha256': sha(__file__), 'started_utc': utc(),
              'status': 'checking', 'gpu_started': False}
    def save():
        receipt.write_text(json.dumps(record, indent=2) + '\n')
    save()
    try:
        assert ROOT.resolve() == Path('/mnt/kda-biv150-stage35-recovery01')
        assert sha(ROOT / 'browser_formal_retry01.py') == '5a59d06701b6e57672e09d5da7968676949007c4cafd05449667aefe8a05113f'
        proof = ROOT / 'local-raw-backup-retry01-01-receipt.json'
        assert sha(proof) == args.backup_sha256
        backup = read(proof)
        assert backup['schema'] == 'stage35-recovery01-retry01-exact-raw-http-backup-receipt-v1'
        assert backup['status'] == 'all_384_originals_byte_verified'
        assert backup['complete_raw_backup'] is True and backup['formal_timing_barrier_released'] is True
        assert backup['plan_sha256'] == PLAN_SHA
        assert len(backup['files']) == backup['original_file_count'] == 384
        assert sum(row['bytes'] for row in backup['files']) == backup['original_bytes'] == 22291589014
        assert len(backup['objects']) == backup['unique_objects'] == 227
        assert all(backup['cpu_numerical_receipt_gates'][model]['original_byte_coverage_complete'] is True
                   for model in ('long', 'experts'))
        record['complete_local_raw_proof_sha256'] = args.backup_sha256
        controllers = ROOT / 'models-retry01/controllers.json'
        assert sha(controllers) == MANIFEST_SHA
        frozen = read(controllers)
        for name, digest in frozen['files'].items():
            assert sha(ROOT / 'models-retry01' / name) == digest, name
        assert sha(ROOT / 'plan.json') == frozen['plan_sha256']
        assert sha(ROOT / 'provenance.json') == frozen['provenance_sha256']
        source_manifest = Path('/tmp/kda-bf16-continuation/megatron_training_entry/source_manifest.json')
        assert sha(source_manifest) == SOURCE_SHA
        native = Path('/tmp/kda-bf16-continuation/megatron-native-5be9626')
        for name, digest in read(source_manifest)['files'].items():
            assert sha(native / name) == digest, name
        model_root = ROOT / 'models-retry01' / args.model
        backend = ROOT / 'models-retry01/long/backend-current.json'
        assert read(backend)['passed'] is True
        audit = model_root / 'audit-004.json'
        assert read(audit)['passed'] is True
        record['audit_sha256'] = sha(audit)
        record['backend_sha256'] = sha(backend)
        if args.round != 'primary':
            assert args.round != 'confirmation2' or args.model == 'long'
            prior_base = model_root / 'confirmation' if args.round == 'confirmation2' else model_root
            primary = read(prior_base / 'performance-004.json')
            assert primary['status'] == 'validated'
            record['primary_performance_sha256'] = sha(model_root / 'performance-004.json')
            if args.round == 'resource_confirmation':
                assert args.model == 'long'
                original_path = ROOT / 'browser-formal-long-retry01-primary-01.json'
                original = read(original_path)
                assert original['status'] == 'failed' and original['exit_code'] == 0
                assert original['error'] == 'Post-run resources busy; timing needs further examination'
                assert original['occupancy_before']['passed'] is True
                assert original['occupancy_after']['both_gpus_idle'] is True
                record['resource_repeat_predecessor_sha256'] = sha(original_path)
                record['repeat_reason'] = 'First complete timing retained as diagnosis because post-run process ownership was unresolved'
            else:
                assert any(cvs(primary)[arm] >= .1 for arm in ('native', 'control')), 'Repeat policy not triggered'
                record['repeat_reason'] = 'Predeclared native/control CV >= 10%'
            target = model_root / ('confirmation2' if args.round == 'confirmation2' else 'confirmation')
            target.mkdir()
            for name, digest in frozen['files'].items():
                relative = Path(name)
                if relative.parts[0] == args.model:
                    destination = target.joinpath(*relative.parts[1:])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / 'models-retry01' / name, destination)
                    assert sha(destination) == digest
            for name in ('audit-004.json', 'probe-004.json', 'offset-004.json',
                         'comparison-admission01.json', 'comparison-native-004.json', 'comparison-control-004.json'):
                shutil.copyfile(model_root / name, target / name)
                assert sha(model_root / name) == sha(target / name)
            model_root = target
        assert not (model_root / 'performance-004.json').exists()
        record['cpu_before'] = cpu_boundary_sample()
        record['occupancy_before'] = occupancy()
        assert record['occupancy_before']['passed'], 'Other computation active; leave it untouched'
        env = {k: v for k, v in os.environ.items() if 'API_KEY' not in k and 'AUTH_TOKEN' not in k}
        env.update(LD_LIBRARY_PATH='/usr/local/corex/lib64:/usr/local/openmpi/lib',
                   PYTHONPATH='/usr/local/corex/lib64/python3/dist-packages', CUDA_VISIBLE_DEVICES='0')
        argv = [PYTHON, '-u', str(model_root / 'run_performance.py'), '--version', '004', '--audit', 'audit-004.json']
        record.update(status='running', gpu_started=True, argv=argv)
        save()
        with Path(str(prefix) + '.stdout.log').open('x') as stdout, Path(str(prefix) + '.stderr.log').open('x') as stderr:
            # The unchanged run_case controller enforces its own600s child group timeout.
            process = subprocess.run(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        record['exit_code'] = process.returncode
        record['post_training_settle_seconds'] = 5
        record['cpu_after'] = cpu_boundary_sample()
        time.sleep(3)
        record['occupancy_after'] = occupancy()
        assert process.returncode == 0, 'Frozen performance controller failed; original logs preserved'
        assert record['occupancy_after']['passed'], 'Post-run resources busy; timing needs further examination'
        performance = read(model_root / 'performance-004.json')
        assert performance['status'] == 'validated' and len(performance['completed_cases']) == 9
        record['performance_sha256'] = sha(model_root / 'performance-004.json')
        record['seed_mean_cv'] = cvs(performance)
        record['repeat_required'] = args.round in ('primary', 'resource_confirmation') and any(record['seed_mean_cv'][arm] >= .1 for arm in ('native', 'control'))
        record['vs_native'] = performance['vs_native']
        record['vs_control'] = performance['vs_control']
        record['promoted'] = performance['promoted']
        record['status'] = 'passed'
    except Exception as error:
        record.update(status='failed' if record['gpu_started'] else 'blocked_before_gpu', error=str(error))
    finally:
        record['finished_utc'] = utc()
        save()
        print(json.dumps({key: record.get(key) for key in ('model', 'round', 'status', 'vs_native', 'vs_control', 'repeat_required', 'error')}), flush=True)
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())

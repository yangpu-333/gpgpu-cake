"""Read-only Linux Pod observation. No Torch import or GPU allocation."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess


def run(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {'argv': argv, 'returncode': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'argv': argv, 'returncode': None, 'error_type': type(error).__name__}


def resource_gate(gpu_result, process_result, own_pid):
    gpu_text = gpu_result.get('stdout', '')
    devices = re.findall(r'(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%', gpu_text)
    foreign = []
    for line in process_result.get('stdout', '').splitlines()[1:]:
        fields = line.split()
        if len(fields) < 8:
            continue
        pid, command, state = int(fields[0]), fields[7], fields[6]
        if pid != own_pid and not state.startswith('Z') and (
                command.startswith(('python', 'torchrun', 'triton')) or float(fields[4]) >= 10):
            foreign.append({'pid': pid, 'command': command, 'state': state,
                            'cpu_percent': float(fields[4])})
    idle = gpu_result.get('returncode') == process_result.get('returncode') == 0
    idle = idle and bool(devices) and all(int(mem) <= 256 and int(util) == 0
                                        for mem, _, util in devices)
    idle = idle and 'No running processes found' in gpu_text
    return {'device_count': len(devices), 'devices': [
        {'used_mib': int(mem), 'total_mib': int(total), 'util_percent': int(util)}
        for mem, total, util in devices], 'all_observed_gpus_idle': idle,
        'unresolved_compute_processes': foreign, 'passed': idle and not foreign,
        'processes_stopped': False, 'observation_only_not_formal_run_admission': True}


def collect():
    gpu = run(['/usr/local/corex/bin/ixsmi'])
    processes = run(['ps', '-eo', 'pid,ppid,user,etimes,pcpu,pmem,stat,comm'])
    versions = {}
    for package in ('torch', 'triton', 'transformer-engine', 'transformers'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    mount = Path('/private')
    storage = {'path': str(mount), 'exists': mount.exists(), 'resolved': str(mount.resolve())}
    if mount.exists():
        total, used, free = shutil.disk_usage(mount)
        storage.update(total_bytes=total, used_bytes=used, free_bytes=free)
    return {'schema': 'jiuge-resource-observation-v1',
            'utc': datetime.now(timezone.utc).isoformat(), 'hostname': socket.gethostname(),
            'python': os.sys.version, 'package_versions': versions,
            'gpu': gpu, 'processes': processes, 'storage': storage,
            'mount_type': run(['df', '-T', '/private']),
            'memory_info': run(['free', '-b']),
            'gate': resource_gate(gpu, processes, os.getpid()),
            'gpu_allocation_or_training_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    record = collect()
    text = json.dumps(record, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(text, end='')
    return 0 if record['gate']['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

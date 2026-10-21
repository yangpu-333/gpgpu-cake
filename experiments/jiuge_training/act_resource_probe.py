"""Read-only observation of the user-selected ACT NVIDIA job."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess


def run(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    gpu = run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu,compute_cap',
               '--format=csv,noheader,nounits'])
    active = run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                  '--format=csv,noheader,nounits'])
    processes = run(['ps', '-eo', 'pid,ppid,user,pcpu,stat,comm'])
    unresolved = []
    for row in processes['stdout'].splitlines()[1:]:
        fields = row.split()
        if len(fields) != 6 or int(fields[0]) == os.getpid() or int(fields[1]) == os.getpid() or fields[4].startswith('Z'):
            continue
        if fields[5].startswith(('python', 'torchrun', 'triton')) or float(fields[3]) >= 10:
            unresolved.append({'pid': int(fields[0]), 'command': fields[5], 'state': fields[4]})
    versions = {}
    for package in ('torch', 'triton', 'transformers', 'safetensors'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    home = Path.home()
    total, used, free = shutil.disk_usage(home)
    record = {'schema': 'jiuge-act-resource-observation-v1',
              'utc': datetime.now(timezone.utc).isoformat(), 'hostname': socket.gethostname(),
              'gpu': gpu, 'gpu_compute_processes': active, 'unresolved_compute_processes': unresolved,
              'package_versions': versions,
              'cgroup_memory': {name: Path(name).read_text().strip() for name in
                  ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current') if Path(name).exists()},
              'storage': {'path': str(home), 'total_bytes': total, 'used_bytes': used, 'free_bytes': free},
              'public_models_mounted': any(Path(name).is_dir() for name in
                  ('/public/Models', '/home/huids25/public/Models', '/data/public/Models')),
              'gpu_allocation_or_training_started': False, 'other_processes_stopped': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()

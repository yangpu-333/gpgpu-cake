"""Verify the specifically authorized SFT container before using its GPUs."""
from datetime import datetime, timezone
import argparse
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    expected = 'sg-huids25-261017-e2bcb-default0-0'
    if socket.gethostname() != expected or (os.getuid(), os.getgid()) != (1001,1001):
        raise ValueError('Unexpected allocation/user identity')
    def command(argv):
        return subprocess.run(argv, text=True, capture_output=True, check=True).stdout
    gpu_text = command(['nvidia-smi','--query-gpu=index,name,uuid,pci.bus_id,memory.total,memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'])
    gpus = [[item.strip() for item in row] for row in csv.reader(gpu_text.splitlines())]
    compute = command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'])
    if len(gpus) != 2 or any('A100' not in row[1] for row in gpus) or compute.strip():
        raise ValueError('Exactly two assigned idle A100s without other compute processes required')
    if any(int(row[5]) > 64 or int(row[6]) != 0 for row in gpus):
        raise ValueError('Assigned GPU is busy or has unexplained resident memory')
    memory = int(Path('/sys/fs/cgroup/memory.max').read_text())
    cpu = Path('/sys/fs/cgroup/cpu.max').read_text().strip()
    quota, period = (int(x) for x in cpu.split())
    if not 16 <= quota / period < 17:
        raise ValueError('Authorized 16-core quota required')
    if memory < 224*1024**3 or memory > 301*1024**3:
        raise ValueError('Unexpected memory quota')
    pid1 = Path('/proc/1/cmdline').read_bytes().split(b'\0')
    actual_pid1=[x.decode() for x in pid1 if x]
    expected_pid1=['timeout','--signal=KILL','21600','bash',
                   '/home/huids25/kda-jiuge-prep-20261014/bootstrap_sft_ubuntu.sh']
    if actual_pid1 != expected_pid1:
        raise ValueError('Authorized six-hour independent container timeout required')
    # The container image is just the OS/driver environment. The scientific
    # runtime must still be the isolated, previously pinned preparation venv.
    import torch
    prep=Path('/home/huids25/kda-jiuge-prep-20261014')
    if (sys.version_info[:3] != (3,12,14) or torch.__version__ != '2.6.0+cu118'
        or not Path(torch.__file__).resolve().is_relative_to((prep/'.venv').resolve())):
        raise ValueError('Actual imported Python/Torch differs from the pinned preparation runtime')
    disks = os.statvfs('/home/huids25')
    record = {'schema': 'jiuge7b-authorized-sft-allocation-v1',
              'utc': datetime.now(timezone.utc).isoformat(), 'hostname': expected,
              'job': 'sg-huids25-261017-e2bcb', 'node': 'xfusion-2', 'uid_gid': [1001,1001],
              'max_charge_points': 0, 'free_minutes': 720, 'startup_sleep_seconds':21600,
              'cpu_max': cpu, 'memory_max_bytes': memory, 'gpus_csv': gpu_text,
              'gpu_rows': gpus, 'compute_processes': compute,
              'gpu_topology': command(['nvidia-smi','topo','-m']),
              'startup_elapsed_seconds': int(command(['ps','-o','etimes=','-p','1']).strip()),
              'persistent_available_bytes': disks.f_bavail*disks.f_frsize,
              'runtime':{'python':sys.version,'executable':sys.executable,'torch':torch.__version__,
                         'torch_file':torch.__file__,'torch_cuda_build':torch.version.cuda,
                         'CUDA_initialized_by_CPU_preflight':torch.cuda.is_initialized(),
                         'PYTHONPATH':os.environ.get('PYTHONPATH'),
                         'LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH'),
                         'container_image':os.environ.get('JIUGE_BASE_IMAGE')},
              'actual_pid1':actual_pid1,
              'same_hardware_as_old_inspur_gpu04': False, 'keep_old_V100': True,
              'other_users_processes_modified': False, 'passed': True}
    with args.output.open('x') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()

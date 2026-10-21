"""Read-only exact-job resource gate and CPU-only real-affinity probe."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from sft_numa_placement_20261021 import discover, build_plan

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
EXPECTED_HOST = 'sg-huids25-261021-28f8b-default0-0'
OUTPUT = PREP/'sft-numa-allocation-preflight-20261021.json'


def main():
    if socket.gethostname() != EXPECTED_HOST or os.getuid() != 1001:
        raise ValueError('Exact new job and existing numeric UID1001 required')
    apps = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_memory', '--format=csv,noheader'],
        text=True, timeout=20)
    if apps.strip():
        raise ValueError('Assigned GPU has other compute; do not launch')
    topology = discover()
    plan = build_plan(topology)
    plan_path = PREP/'sft-numa-CPU-probe-plan-20261021.json'
    with plan_path.open('x') as stream:
        json.dump(plan, stream, indent=2)
    probes = []
    for mode in plan['policies']:
        for rank in (0, 1):
            code = '''import json
from sft_numa_placement_20261021 import bind
r=bind('''+repr(str(plan_path))+','+repr(mode)+','+str(rank)+''')
import torch
from pathlib import Path
r.update(torch_version=torch.__version__,torch_intraop_threads=torch.get_num_threads(),
         cuda_initialized=torch.cuda.is_initialized())
x=torch.arange(16384,dtype=torch.float32).reshape(128,128)
y=x.square().sum(dim=1)
assert y.isfinite().all().item()
assert not torch.cuda.is_initialized() and torch.get_num_threads()==8
r['CPU_tensor_probe_passed']=True
print(json.dumps(r))
'''
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = '8'
            row = json.loads(subprocess.check_output([sys.executable, '-c', code], cwd=PREP, env=env, timeout=60))
            probes.append(row)
    start_path = PREP/'sft-numa-bootstrap-20261021-01/first-start-sg-huids25-261021-28f8b.epoch'
    result = {'schema': 'jiuge7b-sft-numa-allocation-preflight-v1',
              'hostname': socket.gethostname(), 'uid_gid': [os.getuid(), os.getgid()],
              'utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
              'assigned_gpu_compute_apps': apps, 'topology': topology, 'plan': plan,
              'CPU_affinity_runtime_probes': probes,
              'cpu_max': Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
              'memory_max_bytes': int(Path('/sys/fs/cgroup/memory.max').read_text()),
              'first_start_epoch': int(start_path.read_text()),
              'GPU_experiment_started': False,
              'preflight_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with OUTPUT.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'passed': True, 'CPU_probes': len(probes), 'output': str(OUTPUT)}), flush=True)


if __name__ == '__main__':
    main()

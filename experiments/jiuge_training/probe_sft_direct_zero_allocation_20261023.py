"""Read allocation identities before approval, importing no Torch or CUDA."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import socket
import subprocess

from sft_numa_placement_20261021 import discover, build_plan

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    root = PREP/'sft-direct-zero-allocation-20261023'
    root.mkdir(exist_ok=True)
    topology = discover()
    plan = build_plan(topology)
    def command(args):
        return subprocess.check_output(args, text=True, timeout=20)
    result = {'hostname':socket.gethostname(), 'utc':datetime.now(timezone.utc).isoformat(),
              'topology':topology, 'plan':plan,
              'cards':command(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader']),
              'gpu_topology':command(['nvidia-smi','topo','-m']),
              'compute_apps':command(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader']),
              'cgroup':{n:(Path('/sys/fs/cgroup')/n).read_text().strip() for n in
                        ('cpu.max','memory.max','memory.current','memory.events','cpuset.mems.effective')},
              'persistent_free_bytes':shutil.disk_usage(PREP).free,
              'GPU_initialized':False, 'other_processes_stopped':False}
    with (root/(socket.gethostname()+'.json')).open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

"""Read-only ACT CPU/NUMA admission; no CUDA context or memory policy change."""
from datetime import datetime, timezone
import ctypes
import ctypes.util
import json
import mmap
import os
from pathlib import Path
import socket
import subprocess
import sys
from continue_pinning_validation import remaining_seconds

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    if socket.gethostname() != 'sg-huids25-261016-24e4c-default0-0':
        raise ValueError('Wrong allocated host')
    result = {'schema': 'jiuge7b-cpu-numa-environment-v1', 'utc': datetime.now(timezone.utc).isoformat(),
              'hostname': socket.gethostname(), 'remaining_free_seconds': remaining_seconds(),
              'originals_location': str(PREP), 'cuda_initialized': False, 'policy_changed': False}
    for label, command in {
        'gpu': ['nvidia-smi', '--query-gpu=uuid,name,memory.used,utilization.gpu,pci.bus_id', '--format=csv,noheader,nounits'],
        'compute_apps': ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
        'topology': ['nvidia-smi', 'topo', '-m'],
        'cpu': ['lscpu', '-e=CPU,NODE,SOCKET,CORE,ONLINE'],
        'filesystem': ['df', '-B1', str(PREP)],
    }.items():
        result[label] = subprocess.check_output(command, text=True, timeout=20)
    result['cgroup'] = {name: (Path('/sys/fs/cgroup') / name).read_text().strip() for name in (
        'cpu.max', 'cpu.stat', 'cpuset.cpus.effective', 'cpuset.mems.effective', 'memory.current',
        'memory.max', 'memory.numa_stat') if (Path('/sys/fs/cgroup') / name).is_file()}
    result['loadavg'] = Path('/proc/loadavg').read_text().strip()
    result['allowed_cpu_ids'] = sorted(os.sched_getaffinity(0))
    result['numa_nodes'] = {node.name: {name: (node / name).read_text().strip() for name in ('cpulist', 'meminfo', 'distance')}
                            for node in sorted(Path('/sys/devices/system/node').glob('node[0-9]*'))}
    result['cpu_topology'] = {str(cpu): {name: (Path(f'/sys/devices/system/cpu/cpu{cpu}/topology') / name).read_text().strip()
                                       for name in ('physical_package_id', 'core_id', 'thread_siblings_list')}
                              for cpu in result['allowed_cpu_ids']}
    # Query only our own already-faulted pages. nodes=NULL requests location;
    # no migration, mbind, set_mempolicy, affinity or shared settings are changed.
    library = ctypes.util.find_library('numa')
    query = {'libnuma': library, 'query_only': True}
    if library:
        lib = ctypes.CDLL(library, use_errno=True)
        buffer = mmap.mmap(-1, 8192)
        buffer[:] = b'x' * 8192
        address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
        pages = (ctypes.c_void_p * 2)(address, address + 4096)
        status = (ctypes.c_int * 2)(-999, -999)
        lib.numa_move_pages.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        code = lib.numa_move_pages(0, 2, pages, None, status, 0)
        query.update(return_code=code, errno=ctypes.get_errno(), sampled_page_nodes=list(status))
        buffer.close()
    result['actual_page_location_query_probe'] = query
    import torch
    result['torch_cpu'] = {'version': torch.__version__, 'num_threads': torch.get_num_threads(),
                           'interop_threads': torch.get_num_interop_threads(),
                           'parallel_info': torch.__config__.parallel_info(),
                           'cuda_initialized': torch.cuda.is_initialized()}
    if result['torch_cpu']['cuda_initialized']:
        raise ValueError('Read-only CPU probe unexpectedly initialized CUDA')
    path = PREP / 'cpu-numa-environment-20261016-01.json'
    with path.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

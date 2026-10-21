"""Per-worker CPU affinity and read-only NUMA observations; no shared policy."""
import ctypes
import errno
import json
import os
from pathlib import Path
import platform
import re


def cpulist(text):
    values = set()
    for part in text.strip().split(','):
        if not part:
            continue
        bounds = part.split('-')
        values.update(range(int(bounds[0]), int(bounds[-1]) + 1))
    return values


def bind(mode, rank):
    if mode not in ('unbound', 'local8') or rank not in (0, 1):
        raise ValueError('Frozen two-rank placement modes required')
    before = sorted(os.sched_getaffinity(0))
    threads = list(Path('/proc/self/task').iterdir())
    if len(threads) != 1:
        raise ValueError('Apply worker affinity before importing Torch or creating threads')
    node = rank
    allowed = cpulist(Path(f'/sys/devices/system/node/node{node}/cpulist').read_text()) & set(before)
    cores = {}
    for cpu in sorted(allowed):
        root = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((root / 'physical_package_id').read_text().strip(), (root / 'core_id').read_text().strip())
        cores.setdefault(key, cpu)
    selected = sorted(cores.values())[:8]
    if len(selected) != 8 or os.environ.get('OMP_NUM_THREADS') != '8':
        raise ValueError('Keep the existing 8 OpenMP threads per rank and require 8 distinct local cores')
    if mode == 'local8':
        os.sched_setaffinity(0, selected)
    return {'rank': rank, 'mode': mode, 'allowed_before': before, 'allowed_after': sorted(os.sched_getaffinity(0)),
            'gpu_local_numa_node': node, 'selected_local_physical_core_cpus': selected,
            'OMP_NUM_THREADS': os.environ['OMP_NUM_THREADS'], 'memory_policy_syscall_used': False,
            'placement': 'CPU affinity only; subsequent allocations use the existing memory policy',
            'process_only': True}


def sampled_page_nodes(tensors):
    if platform.machine() != 'x86_64':
        return {'supported': False, 'reason': 'Query syscall ABI not validated on this architecture'}
    syscall_number = 279
    header = Path('/usr/include/x86_64-linux-gnu/asm/unistd_64.h')
    if header.is_file():
        match = re.search(r'#define\s+__NR_move_pages\s+(\d+)', header.read_text())
        if match is None or int(match[1]) != syscall_number:
            raise ValueError('Unexpected move_pages query ABI')
    addresses, descriptions = [], []
    page_size = os.sysconf('SC_PAGE_SIZE')
    for label, tensor in tensors:
        if tensor.device.type != 'cpu' or not tensor.is_contiguous():
            continue
        first = tensor.data_ptr() // page_size * page_size
        last = (tensor.data_ptr() + tensor.numel() * tensor.element_size() - 1) // page_size * page_size
        pages = max(1, (last - first) // page_size + 1)
        chosen = sorted({first + round(index * (pages - 1) / 15) * page_size for index in range(16)})
        start = len(addresses)
        addresses.extend(chosen)
        descriptions.append({'field': label, 'numel': tensor.numel(), 'dtype': str(tensor.dtype),
                             'is_pinned': tensor.is_pinned(), 'sample_start': start, 'sample_count': len(chosen)})
    if not addresses:
        return {'supported': False, 'reason': 'No actual CPU tensors available'}
    locations = (ctypes.c_void_p * len(addresses))(*addresses)
    status = (ctypes.c_int * len(addresses))(*([-999] * len(addresses)))
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    # nodes=NULL and flags=0: location query only, never migrate any page.
    code = libc.syscall(ctypes.c_long(syscall_number), ctypes.c_int(0), ctypes.c_ulong(len(addresses)),
                        locations, ctypes.c_void_p(), status, ctypes.c_int(0))
    error = ctypes.get_errno() if code < 0 else 0
    result = {'query_only': True, 'return_code': code, 'errno': error,
              'error_name': errno.errorcode.get(error), 'sample_count': len(addresses),
              'page_size': page_size, 'not_a_complete_page_inventory': True,
              'tensors': descriptions, 'sample_nodes': list(status) if code >= 0 else None}
    return result


def observe(optimizer, label, torch):
    tensors = [(f'master/{index}', value) for index, value in enumerate(optimizer.gpu_params_map_cpu_copy.values())]
    for index, cpu_optimizer in enumerate(optimizer.cpu_optimizers):
        for parameter_index, group in enumerate(cpu_optimizer.param_groups):
            for parameter in group['params']:
                if parameter.grad is not None:
                    tensors.append((f'cpu_optimizer/{index}/{parameter_index}/gradient', parameter.grad))
                for name, value in cpu_optimizer.state.get(parameter, {}).items():
                    if torch.is_tensor(value):
                        tensors.append((f'cpu_optimizer/{index}/{parameter_index}/{name}', value))
    return {'phase': label, 'pid': os.getpid(), 'torch_intraop_threads': torch.get_num_threads(),
            'thread_affinity': {path.name: sorted(os.sched_getaffinity(int(path.name)))
                                for path in Path('/proc/self/task').iterdir()},
            'page_queries': sampled_page_nodes(tensors),
            'process_numa_maps': Path('/proc/self/numa_maps').read_text(),
            'process_status': Path('/proc/self/status').read_text(),
            'cgroup_cpu_stat': Path('/sys/fs/cgroup/cpu.stat').read_text(),
            'cgroup_memory_numa_stat': Path('/sys/fs/cgroup/memory.numa_stat').read_text(),
            'limitations': 'numa_maps is process/VMA accounting; sampled own-page queries have explicit coverage or permission status.'}

"""CPU data-allocation limit with conservative RSS and live host reserves.

RLIMIT_DATA does not cap all mappings or a container's aggregate memory. The
separate training supervisor must also watch its OWN process-tree RSS. This
module never represents that sampled supervisor as a kernel container limit.
"""
from pathlib import Path
import resource

RANK_DATA_LIMIT_BYTES = 144 * 1024**3
RESERVE_BYTES = 8 * 1024**3


def process_values():
    return {line.split(':', 1)[0]: int(line.split()[1])*1024
            for line in Path('/proc/self/status').read_text().splitlines()
            if line.startswith(('VmSize:', 'VmData:', 'VmRSS:', 'VmHWM:', 'VmPeak:'))}


def host_available():
    return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith('MemAvailable:'))


def enforce_rank_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
    if hard != resource.RLIM_INFINITY and hard < RANK_DATA_LIMIT_BYTES:
        raise ValueError('A smaller inherited hard limit requires explicit review')
    if max(process_values()['VmData'], process_values()['VmRSS'])+RESERVE_BYTES >= RANK_DATA_LIMIT_BYTES:
        raise MemoryError('Insufficient conservative process data/RSS reserve')
    resource.setrlimit(resource.RLIMIT_DATA, (RANK_DATA_LIMIT_BYTES, RANK_DATA_LIMIT_BYTES))
    check_live(1024**2)
    return snapshot()


def check_live(allocation_bytes):
    from sft_moment_memory_guard_20261021 import check_values
    if resource.getrlimit(resource.RLIMIT_DATA) != (RANK_DATA_LIMIT_BYTES, RANK_DATA_LIMIT_BYTES):
        raise ValueError('Verified per-rank data limit is not active')
    values = process_values()
    # The actual kernel RLIMIT_DATA remains144GiB. Do not impose an
    # additional arbitrary8GiB+2*allocation preemption below that limit.
    if allocation_bytes<=0:raise ValueError('Positive allocation required')
    required = allocation_bytes
    check_values(0, host_available(), allocation_bytes)
    return required


def snapshot():
    base = Path('/sys/fs/cgroup/memory')
    return {'process_status': Path('/proc/self/status').read_text(),
            'actual_kernel_process_RLIMIT_DATA': resource.getrlimit(resource.RLIMIT_DATA),
            'actual_kernel_process_RLIMIT_AS': resource.getrlimit(resource.RLIMIT_AS),
            'process_values_bytes': process_values(), 'host_MemAvailable_bytes': host_available(),
            'reserve_bytes': RESERVE_BYTES,
            'scope': 'per-process CPU data allocations; NOT container/aggregate RSS/all-mapping cap',
            'required_additional_own_tree_RSS_supervisor': True,
            'cgroup_v1_actual': {n: (base/n).read_text().strip() for n in
                                 ('memory.limit_in_bytes', 'memory.usage_in_bytes',
                                  'memory.max_usage_in_bytes', 'memory.stat')},
            'container_limit_changed': False}

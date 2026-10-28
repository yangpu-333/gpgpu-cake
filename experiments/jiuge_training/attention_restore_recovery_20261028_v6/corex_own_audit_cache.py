"""Use the frozen private audit-cache observer with actual cgroup-v1 readings."""
from pathlib import Path
import time
from sft_phased_audit_cache_20261024 import OwnAuditCache

class OwnCoreXAuditCache(OwnAuditCache):
    def memory(self):
        base = Path('/sys/fs/cgroup/memory')
        return {'epoch': time.time(), 'cgroup_version': 1,
                'cgroup': {name: (base/name).read_text() for name in
                           ('memory.stat', 'memory.numa_stat', 'memory.usage_in_bytes',
                            'memory.limit_in_bytes', 'memory.failcnt', 'memory.oom_control')},
                'host_meminfo': Path('/proc/meminfo').read_text(),
                'NUMA_meminfo': {path.parent.name: path.read_text() for path in
                                Path('/sys/devices/system/node').glob('node[0-9]*/meminfo')}}

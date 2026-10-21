"""Read-only allocation guards and bounded initialization telemetry."""
import json
from pathlib import Path

RESERVE_BYTES = 8*1024**3


def snapshot():
    root=Path('/sys/fs/cgroup')
    return {'process_status':Path('/proc/self/status').read_text(),
            'cgroup':{name:(root/name).read_text().strip() for name in
                      ('memory.current','memory.peak','memory.max','memory.events','memory.numa_stat')},
            'peak_is_cumulative_container_peak_not_per_case':True}


def check_values(current, limit, allocation_bytes):
    if not 0<=current<=limit or allocation_bytes<=0:
        raise ValueError('Invalid actual memory budget')
    required=RESERVE_BYTES+2*allocation_bytes
    if limit-current<required:
        raise MemoryError('Stop before moment allocation: insufficient cgroup headroom; '
                          f'free={limit-current},required={required};no resource extension or fallback')
    return required


def check_live(allocation_bytes):
    root=Path('/sys/fs/cgroup')
    limit=(root/'memory.max').read_text().strip()
    if limit=='max':raise ValueError('A finite container memory limit is required')
    return check_values(int((root/'memory.current').read_text()),int(limit),allocation_bytes)


def install(module):
    old_zero,old_copy=module.create_local_zero,module.relocate_moment
    def zero(parameter,node,torch):
        check_live(parameter.numel()*parameter.element_size())
        return old_zero(parameter,node,torch)
    def copy(value,node,torch):
        check_live(value.numel()*value.element_size())
        return old_copy(value,node,torch)
    module.create_local_zero,module.relocate_moment=zero,copy
    def uninstall():
        if module.create_local_zero is not zero or module.relocate_moment is not copy:
            raise ValueError('Another memory guard replaced the active one')
        module.create_local_zero,module.relocate_moment=old_zero,old_copy
    return uninstall


class MeteredMomentEvents(list):
    def __init__(self,root,rank):
        super().__init__()
        self.path=root/f'moment-initialization-memory-rank-{rank}.jsonl'
        with self.path.open('x') as stream:
            stream.write(json.dumps({'phase':'before-optimizer-initialization',**snapshot()})+'\n')

    def append(self,receipt):
        super().append(receipt)
        if len(self)==1 or len(self)%50==0 or len(self)==398:
            with self.path.open('a') as stream:
                stream.write(json.dumps({'created_moment_objects':len(self),
                    'created_moment_bytes':sum(x['bytes'] for x in self),
                    'copied_source_moment_bytes':sum(x['copied_source_moment_bytes'] for x in self),
                    **snapshot()})+'\n')

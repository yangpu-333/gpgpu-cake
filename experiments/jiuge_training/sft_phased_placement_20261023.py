"""Separate setup affinity from update affinity, without migrating any page."""
import hashlib
import json
import os
from pathlib import Path

NODE_RESERVE_BYTES = 16*1024**3
BOTH_RANK_MOMENT_BYTES = 2*2*3834518016*4


def node_memory(node):
    values = {}
    for line in Path(f'/sys/devices/system/node/node{node}/meminfo').read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4:
            values[parts[2].rstrip(':')] = int(parts[3])*1024
    return values


def check_node_values(free_bytes, allocation_bytes):
    if free_bytes < 0 or allocation_bytes <= 0:
        raise ValueError('Actual positive node memory budget required')
    required = NODE_RESERVE_BYTES+2*allocation_bytes
    if free_bytes < required:
        raise MemoryError(f'Insufficient physical NUMA reserve: free={free_bytes},required={required}')
    return required


def prepare(plan_path, mode, rank):
    payload = Path(plan_path).read_bytes()
    plan = json.loads(payload)
    if rank not in (0,1) or mode not in ('unbound','gpu-local8'):
        raise ValueError('Fixed two-rank policies required')
    if len(list(Path('/proc/self/task').iterdir())) != 1:
        raise ValueError('Setup affinity must precede Torch and worker threads')
    if os.environ.get('OMP_NUM_THREADS') != '8':
        raise ValueError('Keep eight OpenMP threads per rank')
    for name in ('GOMP_CPU_AFFINITY','KMP_AFFINITY','OMP_PLACES','OMP_PROC_BIND'):
        if os.environ.get(name):
            raise ValueError('Uncontrolled thread-affinity environment: '+name)
    before = sorted(os.sched_getaffinity(0))
    if before != plan['topology']['allowed_cpus']:
        raise ValueError('Actual allowed CPU set differs')
    selected = before
    if mode != 'unbound':
        gpu_nodes = {g['node'] for g in plan['topology']['gpus']}
        if len(gpu_nodes) != 1:
            raise ValueError('Two assigned GPUs must share one NUMA node')
        available = {}
        for cpu in before:
            t = plan['topology']['cpu_topology'][str(cpu)]
            if t['node'] not in gpu_nodes:
                available.setdefault(t['node'],{}).setdefault((t['package'],t['core']),cpu)
        candidates = [(node,sorted(cores.values())) for node,cores in available.items() if len(cores)>=16]
        if not candidates:
            raise ValueError('Opposite NUMA node needs16 distinct physical setup cores')
        node, cpus = sorted(candidates)[0]
        selected = cpus[rank*8:(rank+1)*8]
        os.sched_setaffinity(0,selected)
    after = sorted(os.sched_getaffinity(0))
    if after != selected:
        raise ValueError('Actual setup affinity differs')
    return {'schema':'jiuge7b-two-phase-setup-affinity-v1','rank':rank,'mode':mode,
            'allowed_before':before,'allowed_after':after,'plan_sha256':hashlib.sha256(payload).hexdigest(),
            'assigned_gpu':plan['topology']['gpus'][rank],
            'selected_cpu_nodes':sorted({plan['topology']['cpu_topology'][str(c)]['node'] for c in after}),
            'memory_policy_changed':False,'page_migration_used':False,
            'setup_affinity_is_not_proof_of_actual_buffer_pages':True}


def activate(plan_path, mode, rank, root):
    payload = Path(plan_path).read_bytes()
    plan = json.loads(payload)
    selected = plan['policies'][mode][rank]
    before = sorted(os.sched_getaffinity(0))
    for _ in range(5):
        tasks = list(Path('/proc/self/task').iterdir())
        for task in tasks:
            try:
                os.sched_setaffinity(int(task.name),selected)
            except ProcessLookupError:
                pass
        actual = {}
        for task in Path('/proc/self/task').iterdir():
            try:
                actual[task.name] = sorted(os.sched_getaffinity(int(task.name)))
            except ProcessLookupError:
                pass
        if actual and all(cpus == selected for cpus in actual.values()):
            break
    else:
        raise ValueError('Could not bind every thread of this rank before update')
    nodes = sorted({plan['topology']['cpu_topology'][str(c)]['node'] for c in selected})
    result = {'schema':'jiuge7b-two-phase-update-affinity-v1','rank':rank,'mode':mode,
              'pid':os.getpid(),'allowed_before':before,'allowed_after':sorted(os.sched_getaffinity(0)),
              'thread_affinity_after':actual,'selected_cpu_nodes':nodes,
              'assigned_gpu':plan['topology']['gpus'][rank],'plan_sha256':hashlib.sha256(payload).hexdigest(),
              'memory_policy_changed':False,'page_migration_used':False,
              'physical_node_memory_before_update':{str(n):node_memory(n) for n in sorted({x['node'] for x in plan['topology']['cpu_topology'].values()})}}
    with (Path(root)/f'sft-numa-binding-rank-{rank}.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    return result


def install_physical_guard(module, node):
    memory = node_memory(node)
    if memory['MemFree'] < BOTH_RANK_MOMENT_BYTES+NODE_RESERVE_BYTES:
        raise MemoryError('Local node cannot admit both complete moment sets; do not start allocation')
    original = module.create_local_zero
    def guarded(parameter, target, torch):
        if target != node:
            raise ValueError('Moment target changed')
        check_node_values(node_memory(node)['MemFree'],parameter.numel()*parameter.element_size())
        return original(parameter,target,torch)
    module.create_local_zero = guarded
    def uninstall():
        if module.create_local_zero is not guarded:
            raise ValueError('Physical guard was replaced')
        module.create_local_zero = original
    return uninstall

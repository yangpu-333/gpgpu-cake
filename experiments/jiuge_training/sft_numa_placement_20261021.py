"""Process-only affinity plans from actual CPU cores and assigned GPU topology.

This module never imports Torch, migrates pages, or changes memory policy.
NUMA page locations must be observed separately; affinity is not proof of them.
"""
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess


def cpulist(value):
    result = set()
    for item in value.strip().split(','):
        if item:
            ends = item.split('-')
            result.update(range(int(ends[0]), int(ends[-1]) + 1))
    return result


def discover():
    allowed = sorted(os.sched_getaffinity(0))
    nodes = {int(p.name[4:]): cpulist((p/'cpulist').read_text())
             for p in Path('/sys/devices/system/node').glob('node[0-9]*')}
    cpus = {}
    for cpu in allowed:
        membership = [n for n, values in nodes.items() if cpu in values]
        if len(membership) != 1:
            raise ValueError('Each allowed CPU must have exactly one known NUMA node')
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        cpus[str(cpu)] = {'node': membership[0],
                         'package': (base/'physical_package_id').read_text().strip(),
                         'core': (base/'core_id').read_text().strip()}
    cards = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,pci.bus_id', '--format=csv,noheader'],
        text=True, timeout=20)
    gpus = []
    for row in cards.strip().splitlines():
        uuid, address = map(str.strip, row.split(','))
        domain, bus, device = address.lower().split(':')
        pci = f'{domain[-4:]}:{bus}:{device}'
        node = int(Path(f'/sys/bus/pci/devices/{pci}/numa_node').read_text())
        if node < 0:
            raise ValueError('Unknown GPU NUMA node; do not infer it from rank')
        gpus.append({'uuid': uuid, 'pci': pci, 'node': node})
    if len(gpus) != 2 or len({g['uuid'] for g in gpus}) != 2:
        raise ValueError('Exactly two assigned, distinct GPUs required')
    return {'allowed_cpus': allowed, 'cpu_topology': cpus, 'gpus': gpus}


def build_plan(topology):
    cores = {}
    for cpu in sorted(topology['allowed_cpus']):
        item = topology['cpu_topology'][str(cpu)]
        key = (item['package'], item['core'])
        previous = cores.get(key)
        if previous is not None and previous['node'] != item['node']:
            raise ValueError('One physical core cannot span NUMA nodes')
        cores.setdefault(key, {'cpu': cpu, 'node': item['node']})
    by_node = {}
    for item in cores.values():
        by_node.setdefault(item['node'], []).append(item['cpu'])
    for values in by_node.values():
        values.sort()
    gpu_nodes = [g['node'] for g in topology['gpus']]
    if len(gpu_nodes) != 2 or any(n not in by_node for n in gpu_nodes):
        raise ValueError('Two known GPU nodes with allowed CPU cores required')

    def select(nodes):
        used, chosen = set(), []
        for node in nodes:
            values = [c for c in by_node[node] if c not in used][:8]
            if len(values) != 8:
                raise ValueError('Require eight disjoint physical cores per rank')
            used.update(values)
            chosen.append(values)
        return chosen

    policies = {'unbound': [topology['allowed_cpus'][:] for _ in (0, 1)],
                'gpu-local8': select(gpu_nodes)}
    candidates = [(sum(a != b for a, b in zip(pair, gpu_nodes)), pair)
                  for pair in itertools.permutations(sorted(by_node), 2)
                  if all(len(by_node[n]) >= 8 for n in pair)]
    if candidates:
        split_nodes = min(candidates)[1]
        split = select(split_nodes)
        if split != policies['gpu-local8']:
            policies['socket-split8'] = split
    return {'schema': 'jiuge7b-sft-numa-affinity-plan-v1', 'topology': topology,
            'policies': policies, 'threads_per_rank': 8,
            'memory_policy_changed': False, 'page_migration_used': False,
            'actual_page_location_inferred_from_affinity': False}


def bind(plan_path, mode, rank):
    payload = Path(plan_path).read_bytes()
    plan = json.loads(payload)
    if rank not in (0, 1) or mode not in plan['policies']:
        raise ValueError('Unknown two-rank placement')
    if len(list(Path('/proc/self/task').iterdir())) != 1:
        raise ValueError('Bind before importing Torch or creating worker threads')
    if os.environ.get('OMP_NUM_THREADS') != '8':
        raise ValueError('Retain the accepted eight OpenMP threads per rank')
    for name in ('GOMP_CPU_AFFINITY', 'KMP_AFFINITY', 'OMP_PLACES', 'OMP_PROC_BIND'):
        if os.environ.get(name):
            raise ValueError('Uncontrolled inherited thread placement: '+name)
    before = sorted(os.sched_getaffinity(0))
    if before != plan['topology']['allowed_cpus']:
        raise ValueError('Assigned CPU set differs from the frozen resource plan')
    selected = plan['policies'][mode][rank]
    if mode != 'unbound':
        if len(selected) != 8 or not set(selected).issubset(before):
            raise ValueError('Eight allowed physical cores required')
        os.sched_setaffinity(0, selected)
    after = sorted(os.sched_getaffinity(0))
    if after != selected:
        raise ValueError('Actual process affinity differs from plan')
    return {'schema': 'jiuge7b-sft-numa-worker-binding-v1', 'rank': rank,
            'mode': mode, 'pid': os.getpid(), 'plan_sha256': hashlib.sha256(payload).hexdigest(),
            'allowed_before': before, 'allowed_after': after,
            'assigned_gpu': plan['topology']['gpus'][rank],
            'selected_cpu_nodes': sorted({plan['topology']['cpu_topology'][str(c)]['node'] for c in after}),
            'memory_policy_changed': False, 'page_migration_used': False}

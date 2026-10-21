"""Read-only CPU placement observation during the untimed numeric audit."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    workers = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            argv = [os.fsdecode(x) for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
            # torchrun invokes workers as python -u script.py.
            if not any(Path(item).name == 'pinning_entry.py' for item in argv[1:3]) or '--full-state-audit' not in argv:
                continue
            if str(PREP / 'runs/pinning-native-audit-04') not in argv:
                continue
            status = (proc / 'status').read_text()
            affinity = {line.split(':', 1)[0]: line.split(':', 1)[1].strip() for line in status.splitlines()
                        if line.startswith(('Cpus_allowed_list:', 'Mems_allowed_list:'))}
            nodes, anon_nodes = {}, {}
            for line in (proc / 'numa_maps').read_text().splitlines():
                width = re.search(r'kernelpagesize_kB=(\d+)', line)
                page_bytes = int(width[1]) * 1024 if width else 4096
                for node, count in re.findall(r'\bN(\d+)=(\d+)\b', line):
                    nodes[node] = nodes.get(node, 0) + int(count) * page_bytes
                    if re.search(r'\banon=\d+', line):
                        anon_nodes[node] = anon_nodes.get(node, 0) + int(count) * page_bytes
            workers.append({'pid': int(proc.name), 'argv': argv, 'affinity': affinity,
                            'resident_mapping_bytes_by_numa_node': nodes,
                            'anonymous_mapping_bytes_by_numa_node': anon_nodes})
        except (FileNotFoundError, ProcessLookupError):
            continue
    if len(workers) != 2:
        raise ValueError('Exactly two owned live untimed audit workers required')
    topology = subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True, text=True, check=True, timeout=20).stdout
    counters = {}
    for name in ('cpu.max', 'cpuset.cpus.effective', 'cpuset.mems.effective', 'memory.current', 'memory.numa_stat'):
        path = Path('/sys/fs/cgroup') / name
        if path.is_file():
            counters[name] = path.read_text().strip()
    result = {'schema': 'jiuge7b-untimed-audit-numa-observation-v1', 'utc': datetime.now(timezone.utc).isoformat(),
              'workers': workers, 'gpu_topology': topology, 'cgroup': counters,
              'training_configuration_modified': False, 'affinity_modified': False,
              'formal_timing_active': False,
              'limitations': 'Single during-audit snapshot, not a pinning or causal performance measurement; per-process mappings may share pages'}
    with (PREP / 'numa-readonly-during-audit-20261015.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'workers': [{k: v for k, v in row.items() if k != 'argv'} for row in workers],
                      'gpu_topology': topology, 'cgroup_cpu_quota': counters.get('cpu.max')}), flush=True)


if __name__ == '__main__':
    main()

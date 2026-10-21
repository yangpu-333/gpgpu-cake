"""Require the explicitly approved allocation, exact topology and free lifetime."""
import hashlib
import json
from pathlib import Path
import socket
from continue_pinning_validation import remaining_seconds
from continue_pinning_formal import resources

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def check(path):
    value = json.loads(Path(path).read_text())
    if (value['schema'] != 'jiuge7b-authorized-numa-allocation-v2' or
            value['hostname'] != socket.gethostname() or value['max_charge_points'] != 0 or
            not value['user_authorized_create_restore_stop'] or
            value['cpu_max'] != Path('/sys/fs/cgroup/cpu.max').read_text().strip() or
            value['memory_max_bytes'] != int(Path('/sys/fs/cgroup/memory.max').read_text()) or
            value['memory_max_bytes'] != 322123595776):
        raise ValueError('Approved same-spec zero-cost allocation required')
    if remaining_seconds() < 1800:
        raise TimeoutError('Preserve free allocation shutdown reserve')
    for rank, expected in enumerate((0, 1)):
        if int(Path(f'/sys/bus/pci/devices/{value["gpu_pci_lower"][rank]}/numa_node').read_text()) != expected:
            raise ValueError('Ordered GPU NUMA topology differs')
    gate = resources()
    if [line.split(',')[0].strip() for line in gate['gpu_stdout'].strip().splitlines()] != value['ordered_gpu_uuids']:
        raise ValueError('GPU order differs from frozen affinity experiment')
    return value

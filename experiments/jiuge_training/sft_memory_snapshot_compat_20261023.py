"""Optional kernel peak telemetry; mandatory live allocation guards unchanged."""
import hashlib
from pathlib import Path


def snapshot(root=Path('/sys/fs/cgroup'), status=Path('/proc/self/status')):
    values = {name:(root/name).read_text().strip() for name in
              ('memory.current','memory.max','memory.events','memory.numa_stat')}
    try:
        values['memory.peak'] = (root/'memory.peak').read_text().strip()
    except FileNotFoundError:
        values['memory.peak'] = None
    return {'process_status':status.read_text(), 'cgroup':values,
            'kernel_cgroup_peak_available':values['memory.peak'] is not None,
            'missing_kernel_peak_is_not_zero_or_a_sampled_peak':True,
            'process_VmHWM_is_per_process_not_total_container_peak':True,
            'peak_is_cumulative_container_peak_not_per_case':True}


def install():
    import sft_moment_memory_guard_20261021 as guard
    # Change only the observer. check_live, its8GiB reserve, factories and math stay native.
    # The actual original digest is read from the unchanged frozen93 manifest.
    import json
    manifest = json.loads((Path(__file__).parent/'sft-moment-direct-zero-source-manifest-v3-20261021.json').read_text())
    expected = manifest['files']['sft_moment_memory_guard_20261021.py']
    if hashlib.sha256(Path(guard.__file__).read_bytes()).hexdigest() != expected:
        raise ValueError('Frozen production memory guard differs')
    if guard.RESERVE_BYTES != 8*1024**3:
        raise ValueError('Production8GiB allocation reserve must remain unchanged')
    guard.snapshot = snapshot

"""Read-only source, allocation and host preflight before full resume."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from numa_allocation_v2 import check
from continue_pinning_validation import remaining_seconds

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    allocation = check(PREP / 'cpu-numa-allocation-20261016-02.json')
    if os.getuid() != 1001 or os.getgid() != 1001:
        raise ValueError('Existing same UID/GID login required')
    actual = json.loads((PREP / 'cpu-numa-validation-20261016-01/controller.json').read_text())['source_sha256']
    expected = json.loads((PREP / 'cpu-numa-v2-source-manifest.json').read_text())
    for name, digest in {**actual, **expected}.items():
        if hashlib.sha256((PREP / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen source mismatch: ' + name)
    if any((PREP / path).exists() for path in ('cpu-numa-validation-20261016-02',
                                             'runs/cpu-numa-local8-full-resume-02',
                                             'cpu-numa-formal-20261016-02')):
        raise ValueError('New versioned outputs already exist; inspect before launch')
    result = {'schema': 'jiuge7b-numa-successor-preflight-v2', 'passed': True,
              'utc': datetime.now(timezone.utc).isoformat(), 'allocation': allocation,
              'source_files_sha256': {**actual, **expected}, 'remaining_seconds': remaining_seconds(),
              'memory_current_bytes': int(Path('/sys/fs/cgroup/memory.current').read_text()),
              'memory_events': Path('/sys/fs/cgroup/memory.events').read_text(),
              'gpu_topology': subprocess.run(['nvidia-smi', 'topo', '-m'], text=True, capture_output=True, check=True).stdout,
              'processes': subprocess.run(['ps', '-u', '1001', '-o', 'pid,ppid,etimes,comm,args'],
                                          text=True, capture_output=True, check=True).stdout,
              'ceph_available_bytes': os.statvfs(PREP).f_bavail * os.statvfs(PREP).f_frsize}
    with (PREP / 'cpu-numa-successor-preflight-20261016-02.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

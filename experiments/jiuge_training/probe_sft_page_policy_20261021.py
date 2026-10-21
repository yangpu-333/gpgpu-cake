"""Tiny CPU-only NUMA capability probe on new private mappings; no migration."""
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import json
import mmap
import os
from pathlib import Path
import platform
import re
import socket
import subprocess

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT = PREP/'sft-moment-placement-preparation-20261021-01'


def parse_list(text):
    values = set()
    for part in text.strip().split(','):
        if part:
            bounds = list(map(int, part.split('-')))
            values.update(range(bounds[0], bounds[-1]+1))
    return values


def main():
    ROOT.mkdir(exist_ok=False)
    (ROOT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    if platform.machine() != 'x86_64':
        raise ValueError('Only validated Linux x86_64 syscall ABI supported')
    abi = {'mbind':237, 'get_mempolicy':239, 'move_pages':279}
    header = Path('/usr/include/x86_64-linux-gnu/asm/unistd_64.h')
    if header.is_file():
        text = header.read_text()
        for name, number in abi.items():
            match = re.search(r'#define\s+__NR_'+name+r'\s+(\d+)', text)
            if match is None or int(match[1]) != number:
                raise ValueError('Live header does not match syscall '+name)
    original = sorted(os.sched_getaffinity(0))
    allowed_memory = parse_list(Path('/sys/fs/cgroup/cpuset.mems.effective').read_text())
    current = int(Path('/sys/fs/cgroup/memory.current').read_text())
    limit_text = Path('/sys/fs/cgroup/memory.max').read_text().strip()
    if limit_text != 'max' and int(limit_text)-current < 512*1024**2:
        raise ValueError('Preserve512MiB container CPU memory reserve')
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory',
                                    '--format=csv,noheader'],text=True,timeout=15)
    if apps.strip():
        raise ValueError('Other GPU computation present; defer shared-container probe')
    nodes = {}
    for path in Path('/sys/devices/system/node').glob('node[0-9]*'):
        node = int(path.name[4:])
        available = sorted(parse_list((path/'cpulist').read_text()) & set(original))
        if available and node in allowed_memory:
            nodes[node] = available[0]
    libc = ctypes.CDLL(None,use_errno=True)
    libc.syscall.restype = ctypes.c_long
    page = os.sysconf('SC_PAGE_SIZE')
    rows = []
    try:
        for node, cpu in sorted(nodes.items()):
            os.sched_setaffinity(0,{cpu})
            for mode in ('first-touch','mbind-before-touch'):
                memory = mmap.mmap(-1,64*page,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS,
                                   prot=mmap.PROT_READ|mmap.PROT_WRITE)
                address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
                result = {'node':node,'cpu':cpu,'mode':mode,'bytes':64*page,
                          'private_anonymous_new_mapping':True,'page_migration_flags':0}
                if mode == 'mbind-before-touch':
                    words = (max(nodes)+64)//64
                    mask = (ctypes.c_ulong*words)()
                    mask[node//64] |= 1 << (node%64)
                    ctypes.set_errno(0)
                    code = libc.syscall(ctypes.c_long(abi['mbind']),ctypes.c_void_p(address),
                                        ctypes.c_ulong(64*page),ctypes.c_int(2),mask,
                                        ctypes.c_ulong(words*64),ctypes.c_uint(0))
                    error = ctypes.get_errno() if code < 0 else 0
                    result.update(mbind_return_code=code,errno=error,error_name=errno.errorcode.get(error))
                    if code:
                        result['passed'] = False
                        rows.append(result);memory.close();continue
                ctypes.memset(address,7,64*page)
                pointers = (ctypes.c_void_p*64)(*[address+i*page for i in range(64)])
                locations = (ctypes.c_int*64)(*([-999]*64))
                ctypes.set_errno(0)
                code = libc.syscall(ctypes.c_long(abi['move_pages']),ctypes.c_int(0),ctypes.c_ulong(64),
                                    pointers,ctypes.c_void_p(),locations,ctypes.c_int(0))
                error = ctypes.get_errno() if code < 0 else 0
                result.update(query_return_code=code,query_errno=error,sampled_nodes=list(locations),
                              passed=code==0 and all(n==node for n in locations),
                              contents_preserved=memory[0]==7 and memory[-1]==7)
                rows.append(result)
                memory.close()
    finally:
        os.sched_setaffinity(0,original)
    value = {'schema':'jiuge7b-sft-private-page-policy-capability-v1',
             'utc':datetime.now(timezone.utc).isoformat(),'hostname':socket.gethostname(),
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'GPU_used':False,'torch_imported':False,'other_compute_apps':apps,
             'original_CPU_affinity_restored':sorted(os.sched_getaffinity(0))==original,
             'global_or_task_memory_policy_changed':False,'migrated_existing_pages':False,
             'nodes_and_probe_CPUs':nodes,'ABI':abi,'live_ABI_header_checked':header.is_file(),
             'rows':rows,'first_touch_all_nodes_passed':all(x['passed'] for x in rows if x['mode']=='first-touch'),
             'mbind_all_nodes_passed':all(x['passed'] for x in rows if x['mode']=='mbind-before-touch'),
             'formal_performance_evidence':False,'same_as_A100_training_environment':False}
    with (ROOT/'page-policy-probe.json').open('x') as stream:json.dump(value,stream,indent=2)
    print(json.dumps(value))


if __name__ == '__main__':
    main()

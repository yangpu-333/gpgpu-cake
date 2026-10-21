"""Experimental CPU AdamW moment placement on fresh private anonymous mappings.

Only exp_avg/exp_avg_sq storage changes. Native AdamW math, parameter/gradient
storage, step counters and GPU transfers are untouched. Not SFT-accepted yet.
"""
import ctypes
import hashlib
import inspect
import mmap
import os
from pathlib import Path
import platform
import re
import weakref

TORCH_ADAMW_SHA256 = 'e680a255db9c53c4248a55f414b7af9aab90bb672312ac960bb459903a161c1d'


def _abi():
    if platform.machine() != 'x86_64' or os.name != 'posix':
        raise ValueError('Validated Linux x86_64 only')
    header = Path('/usr/include/x86_64-linux-gnu/asm/unistd_64.h')
    if not header.is_file():
        raise ValueError('Live syscall header required')
    for name, number in (('mbind',237),('move_pages',279)):
        match = re.search(r'#define\s+__NR_'+name+r'\s+(\d+)',header.read_text())
        if match is None or int(match[1]) != number:
            raise ValueError('Unsupported syscall ABI: '+name)
    libc = ctypes.CDLL(None,use_errno=True)
    libc.syscall.restype = ctypes.c_long
    return libc


def node_for_current_cpus():
    allowed = set(os.sched_getaffinity(0))
    locations = []
    for path in Path('/sys/devices/system/node').glob('node[0-9]*'):
        cpus = set()
        for part in (path/'cpulist').read_text().strip().split(','):
            ends = list(map(int,part.split('-')))
            cpus.update(range(ends[0],ends[-1]+1))
        if allowed & cpus:
            locations.append(int(path.name[4:]))
    if len(locations) != 1:
        raise ValueError('CPU execution must already be bound to exactly one NUMA node')
    return locations[0]


def relocate_moment(tensor,node,torch):
    if (tensor.device.type != 'cpu' or tensor.dtype != torch.float32 or not tensor.is_contiguous()
            or tensor.requires_grad or tensor.numel() == 0 or tensor.is_pinned()):
        raise ValueError('Only nonempty contiguous unpinned FP32 CPU moments are supported')
    if node_for_current_cpus() != node:
        raise ValueError('Moment target must equal the actual bound CPU node')
    libc = _abi()
    page = os.sysconf('SC_PAGE_SIZE')
    size = tensor.numel()*tensor.element_size()
    rounded = (size+page-1)//page*page
    memory = mmap.mmap(-1,rounded,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS,
                       prot=mmap.PROT_READ|mmap.PROT_WRITE)
    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
    mask = (ctypes.c_ulong*((node+64)//64))()
    mask[node//64] = 1 << (node%64)
    ctypes.set_errno(0)
    code = libc.syscall(ctypes.c_long(237),ctypes.c_void_p(address),ctypes.c_ulong(rounded),
                        ctypes.c_int(2),mask,ctypes.c_ulong(len(mask)*64),ctypes.c_uint(0))
    if code != 0:
        error = ctypes.get_errno();memory.close()
        raise OSError(error,'Private moment mbind failed; no fallback to inferred placement')
    # frombuffer retains the mmap owner for the storage lifetime. Bind before
    # first write; never use MPOL_MF_MOVE or change a task/global memory policy.
    placed = torch.frombuffer(memory,dtype=tensor.dtype,count=tensor.numel()).reshape(tensor.shape)
    placed.copy_(tensor)
    count = min(16,(size+page-1)//page)
    chosen = sorted({round(i*((size-1)//page)/max(1,count-1)) for i in range(count)})
    addresses = (ctypes.c_void_p*len(chosen))(*[address+i*page for i in chosen])
    locations = (ctypes.c_int*len(chosen))(*([-999]*len(chosen)))
    code = libc.syscall(ctypes.c_long(279),ctypes.c_int(0),ctypes.c_ulong(len(chosen)),addresses,
                        ctypes.c_void_p(),locations,ctypes.c_int(0))
    if code != 0 or any(n != node for n in locations):
        raise ValueError('Actual sampled moment pages do not match the bound CPU node')
    if placed.is_pinned() or placed.dtype != tensor.dtype or placed.shape != tensor.shape:
        raise ValueError('Moment storage properties changed')
    return placed, {'target_node':node,'bytes':size,'mapping_bytes':rounded,
                    'sampled_nodes':list(locations),'sampled_pages':len(chosen),
                    'sample_is_full_inventory':False,'private_mapping':True,
                    'migrated_existing_pages':False,'mbind_flags':0,'query_only_move_pages':True,
                    'source_and_target_dtype':'torch.float32','source_and_target_pinned':False}


def install(torch,events):
    import torch.optim.adamw as module
    path = Path(inspect.getsourcefile(module.AdamW))
    if hashlib.sha256(path.read_bytes()).hexdigest() != TORCH_ADAMW_SHA256:
        raise ValueError('Actual Torch AdamW source differs from the reviewed2.6.0 source')
    original = module.AdamW._init_group
    if getattr(original,'_jiuge_local_moments',False):
        raise ValueError('Do not install the candidate twice')
    node = node_for_current_cpus()
    tracked = {}

    def remember(value):
        identity = id(value)
        def discard(reference):
            if tracked.get(identity) is reference:
                tracked.pop(identity,None)
        tracked[identity] = weakref.ref(value,discard)

    def wrapped(opt,group,params_with_grad,grads,amsgrad,exp_avgs,exp_avg_sqs,max_exp_avg_sqs,state_steps):
        start = len(params_with_grad)
        result = original(opt,group,params_with_grad,grads,amsgrad,exp_avgs,exp_avg_sqs,max_exp_avg_sqs,state_steps)
        for index in range(start,len(params_with_grad)):
            parameter = params_with_grad[index]
            if parameter.device.type != 'cpu':
                continue
            if amsgrad:
                raise ValueError('The frozen SFT contract does not use AMSGrad')
            state = opt.state[parameter]
            for name,passed_list in (('exp_avg',exp_avgs),('exp_avg_sq',exp_avg_sqs)):
                value = state[name]
                reference = tracked.get(id(value))
                if reference is not None and reference() is value:
                    continue
                placed,receipt = relocate_moment(value,node,torch)
                state[name] = placed
                passed_list[index] = placed
                remember(placed)
                events.append({'field':name,'elements':placed.numel(),**receipt})
        return result

    wrapped._jiuge_local_moments = True
    module.AdamW._init_group = wrapped

    def uninstall():
        if module.AdamW._init_group is not wrapped:
            raise ValueError('Another patch replaced the active candidate')
        module.AdamW._init_group = original

    return uninstall

"""Create fresh CPU AdamW moments directly in their final NUMA mappings.

Only the two zeros_like factory expressions in the hash-checked native
_init_group are replaced. Native AdamW update mathematics and step creation
stay unchanged. Native loaded states use the separately tested copy path.
This candidate has not passed full7B SFT acceptance.
"""
import ast
import ctypes
import hashlib
import inspect
import mmap
import os
from pathlib import Path
import textwrap
import weakref

from sft_local_moments_20261021 import (
    TORCH_ADAMW_SHA256, _abi, node_for_current_cpus, relocate_moment,
)


def create_local_zero(parameter, node, torch):
    if (parameter.device.type != 'cpu' or parameter.dtype != torch.float32
            or not parameter.is_contiguous() or parameter.numel() == 0):
        raise ValueError('Only nonempty contiguous FP32 CPU parameters are supported')
    if node_for_current_cpus() != node:
        raise ValueError('Target differs from the actual single bound CPU node')
    libc = _abi()
    page = os.sysconf('SC_PAGE_SIZE')
    size = parameter.numel()*parameter.element_size()
    rounded = (size+page-1)//page*page
    memory = mmap.mmap(-1, rounded, flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS,
                       prot=mmap.PROT_READ|mmap.PROT_WRITE)
    address = ctypes.addressof(ctypes.c_char.from_buffer(memory))
    mask = (ctypes.c_ulong*((node+64)//64))()
    mask[node//64] = 1 << (node%64)
    ctypes.set_errno(0)
    code = libc.syscall(ctypes.c_long(237), ctypes.c_void_p(address), ctypes.c_ulong(rounded),
                        ctypes.c_int(2), mask, ctypes.c_ulong(len(mask)*64), ctypes.c_uint(0))
    if code != 0:
        error = ctypes.get_errno()
        memory.close()
        raise OSError(error, 'Fresh local-zero moment mbind failed')
    placed = torch.frombuffer(memory, dtype=torch.float32,
                              count=parameter.numel()).reshape(parameter.shape)
    # This is the first write. There is no preceding native zeros_like tensor.
    placed.zero_()
    count = min(16, (size+page-1)//page)
    chosen = sorted({round(i*((size-1)//page)/max(1,count-1)) for i in range(count)})
    addresses = (ctypes.c_void_p*len(chosen))(*[address+i*page for i in chosen])
    locations = (ctypes.c_int*len(chosen))(*([-999]*len(chosen)))
    code = libc.syscall(ctypes.c_long(279), ctypes.c_int(0), ctypes.c_ulong(len(chosen)),
                        addresses, ctypes.c_void_p(), locations, ctypes.c_int(0))
    if code != 0 or any(n != node for n in locations):
        raise ValueError('Actual fresh moment samples differ from the bound node')
    if placed.is_pinned() or placed.requires_grad:
        raise ValueError('Final moment must remain unpinned and without a gradient')
    return placed, {'target_node':node, 'bytes':size, 'mapping_bytes':rounded,
                    'sampled_nodes':list(locations), 'sampled_pages':len(chosen),
                    'sample_is_full_inventory':False, 'private_mapping':True,
                    'migrated_existing_pages':False, 'mbind_flags':0,
                    'query_only_move_pages':True, 'initialization_path':'direct_local_zero',
                    'native_source_moment_created':False, 'copied_source_moment_bytes':0,
                    'moment_dtype':'torch.float32', 'moment_pinned':False}


def derive_initializer(original, factory):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('Exactly one native initializer function is required')
    method = tree.body[0]
    if method.name != '_init_group' or method.decorator_list:
        raise ValueError('Unsupported native initializer or wrapper')
    changed = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name)
                or target.value.id != 'state' or not isinstance(target.slice, ast.Constant)
                or target.slice.value not in ('exp_avg','exp_avg_sq')):
            continue
        expected = "torch.zeros_like(p, memory_format=torch.preserve_format)"
        if ast.dump(node.value, include_attributes=False) != ast.dump(ast.parse(expected, mode='eval').body,
                                                                   include_attributes=False):
            raise ValueError('Native moment factory differs')
        field = target.slice.value
        node.value = ast.copy_location(ast.Call(func=ast.Name(id='_jiuge_local_zero',ctx=ast.Load()),
                                                args=[ast.Name(id='p',ctx=ast.Load()),ast.Constant(field)],
                                                keywords=[]), node.value)
        changed.append(field)
    if sorted(changed) != ['exp_avg','exp_avg_sq']:
        raise ValueError('Exactly two moment factory expressions must change')
    ast.fix_missing_locations(tree)
    # Isolate the function globals; never replace torch.zeros_like globally.
    namespace = dict(original.__globals__)
    namespace['_jiuge_local_zero'] = factory
    exec(compile(tree, '<frozen-native-AdamW-two-factory-replacement>', 'exec'), namespace)
    return namespace['_init_group']


def install(torch, events):
    import torch.optim.adamw as module
    source = Path(inspect.getsourcefile(module.AdamW))
    if hashlib.sha256(source.read_bytes()).hexdigest() != TORCH_ADAMW_SHA256:
        raise ValueError('Actual native AdamW source differs from the reviewed2.6.0 file')
    original = module.AdamW._init_group
    if getattr(original, '_jiuge_local_moments', False):
        raise ValueError('Another moment-placement patch is active')
    node = node_for_current_cpus()
    tracked = {}

    def known(value):
        reference = tracked.get(id(value))
        return reference is not None and reference() is value

    def remember(value):
        identity = id(value)
        def discard(reference):
            if tracked.get(identity) is reference:
                tracked.pop(identity,None)
        tracked[identity] = weakref.ref(value,discard)

    def factory(parameter, field):
        if parameter.device.type != 'cpu':
            return torch.zeros_like(parameter,memory_format=torch.preserve_format)
        placed, receipt = create_local_zero(parameter,node,torch)
        remember(placed)
        events.append({'field':field,'elements':placed.numel(),**receipt})
        return placed

    direct_initializer = derive_initializer(original, factory)

    def wrapped(opt, group, params_with_grad, grads, amsgrad,
                exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps):
        for parameter in group['params']:
            if parameter.grad is None or parameter.device.type != 'cpu':
                continue
            if amsgrad or group['amsgrad']:
                raise ValueError('The SFT candidate contract excludes AMSGrad')
            state = opt.state[parameter]
            if not state:
                continue  # The native function will create step and both local-zero moments.
            for field in ('exp_avg','exp_avg_sq'):
                value = state[field]
                if known(value):
                    continue
                placed, receipt = relocate_moment(value,node,torch)
                state[field] = placed
                remember(placed)
                events.append({'field':field,'elements':placed.numel(),**receipt,
                               'initialization_path':'native_resume_copy',
                               'native_source_moment_created':True,
                               'copied_source_moment_bytes':placed.numel()*placed.element_size()})
        # The original list append, checks, step handling and native update implementation remain intact.
        return direct_initializer(opt,group,params_with_grad,grads,amsgrad,
                                  exp_avgs,exp_avg_sqs,max_exp_avg_sqs,state_steps)

    wrapped._jiuge_local_moments = True
    wrapped._jiuge_local_moments_direct_zero_v3 = True
    module.AdamW._init_group = wrapped

    def uninstall():
        if module.AdamW._init_group is not wrapped:
            raise ValueError('Another patch replaced the current initializer')
        module.AdamW._init_group = original

    return uninstall

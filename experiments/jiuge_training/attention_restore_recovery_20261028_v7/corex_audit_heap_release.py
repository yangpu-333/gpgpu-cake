"""Untimed, own-process allocator housekeeping. Never changes live tensors."""
import ctypes
import gc
import os
from pathlib import Path
import platform

def release(label):
    assert os.environ.get('JIUGE_DT_UNTIMED_AUDIT')=='1'
    assert platform.system()=='Linux' and platform.machine()=='x86_64'
    from corex_process_guard_data import process_values,host_available
    before=process_values()
    import torch
    initialized=torch.cuda.is_initialized()
    stats_before=torch.cuda.host_memory_stats()
    if initialized:torch.cuda.synchronize()
    torch._C._host_emptyCache()
    stats_after=torch.cuda.host_memory_stats()
    if initialized:
        assert stats_after['reserved_bytes.current']<=stats_before['reserved_bytes.current']
        assert stats_after['segment.current']<=stats_before['segment.current']
    assert torch.cuda.is_initialized()==initialized
    collected=gc.collect()
    libc=ctypes.CDLL(None,use_errno=True)
    trim=libc.malloc_trim;trim.argtypes=[ctypes.c_size_t];trim.restype=ctypes.c_int
    result=trim(0)
    return {'label':label,'PID':os.getpid(),'before_process_bytes':before,'after_process_bytes':process_values(),
            'unreachable_objects_collected':collected,'malloc_trim_return_code':result,
            'host_available_bytes':host_available(),'live_tensor_or_math_modified':False,
            'global_allocator_policy_modified':False,'other_process_or_file_modified':False,
            'native_host_cache_empty_called':True,'host_stats_before':stats_before,'host_stats_after':stats_after,
            'CUDA_initialized_before':initialized,'CUDA_initialized_after':torch.cuda.is_initialized(),
            'host_allocated_bytes_counter_not_used_for_acceptance':True}

def install(records):
    import pinning_full_audit
    original=pinning_full_audit.FullAudit.save_hdo
    def observed(audit,opt,phase):
        result=original(audit,opt,phase)
        records.append(release(audit.arm+':'+phase))
        return result
    pinning_full_audit.FullAudit.save_hdo=observed
    def undo():
        assert pinning_full_audit.FullAudit.save_hdo is observed
        pinning_full_audit.FullAudit.save_hdo=original
    return undo


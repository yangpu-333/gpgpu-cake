"""CoreX API/placement observers for a genuine native save and two restores.

The checkpoint is saved by the original native HDO. Candidate resume alone
uses the accepted pinned-master derivation and the two-factory CoreX port.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

root = Path(os.environ['JIUGE_DT_ROOT'])
control = Path(os.environ['JIUGE_DT_CONTROL'])
sys.path[:0] = [str(control), str(root), str(root/'megatron-native-5be9626')]
import corex_process_guard_data as guard
sys.modules['corex_process_guard'] = guard
guard.enforce_rank_limit()
rank = int(os.environ['RANK'])
arm = os.environ['JIUGE_DT_ARM']
stage = os.environ['JIUGE_DT_RESTORE_STAGE']
assert (stage, arm) in (('checkpoint', 'native'), ('resume', 'native'), ('resume', 'optimized'))
run_root = Path(sys.argv[sys.argv.index('--jiuge-run-root')+1])
from sft_phased_placement_20261023 import prepare, activate, install_physical_guard
mode = 'gpu-local8' if arm == 'optimized' else 'unbound'
setup = prepare(control/'numa-plan.json', 'gpu-local8', rank)
with (run_root/f'CoreX-setup-affinity-rank-{rank}.json').open('x') as stream:
    json.dump(setup, stream, indent=2)

import torch
torch.use_deterministic_algorithms(False)
import megatron.core.optimizer.cpu_offloading.hybrid_optimizer as module
from megatron.core.transformer.torch_norm import WrappedTorchNorm
import megatron.core.models.gpt.gpt_layer_specs as gpt_specs
import megatron.core.transformer.transformer_block as block
gpt_specs.LNImpl = WrappedTorchNorm
block.LayerNormImpl = WrappedTorchNorm

# CoreX already has the public no_dist loader. Do not install the old redirect.
import checkpoint_load_compat
import corex_native_checkpoint_load
native_install = corex_native_checkpoint_load.install
def install_native_observer():
    evidence = native_install()
    evidence.update(legacy_redirect_not_used=True,
                    actual_CoreX_observer_SHA256=hashlib.sha256(
                        (control/'corex_native_checkpoint_load.py').read_bytes()).hexdigest())
    import torch.distributed.checkpoint as dcp
    native_observed=dcp.load
    def resource_scoped_load(state_dict,*args,**kwargs):
        own_memory('before-native-DCP-load');drop_owned_gradient_scratch('before-native-DCP-load')
        result=native_observed(state_dict,*args,**kwargs);own_memory('after-native-DCP-load');return result
    dcp.load=resource_scoped_load
    evidence['native_API_and_reader_planner_unchanged']=True
    evidence['only_unserialized_gradient_scratch_lifetime_changed']=True
    return evidence
checkpoint_load_compat.install = install_native_observer

state = {'arm': arm, 'stage': stage, 'rank': rank, 'HDO_update_calls': 0,
         'events': [], 'binding': None, 'final_moment_page_query': None,
         'native_checkpoint_save_uses_original_HDO': stage == 'checkpoint',
         'legacy_load_redirect_used': False,
         'kernel_guard_before_optimizer': guard.snapshot()}
undo = []
allocation_receipt={'setup':setup,'common_repair_all_arms':True,'stage':stage}
from corex_setup_gradients_v3 import install as install_setup_gradients
undo.append(install_setup_gradients(torch,module.HybridDeviceOptimizer,allocation_receipt))
state.update(common_setup_node0=True,gradient_preallocation=allocation_receipt,strict_deterministic_algorithms=False)
from corex_own_audit_cache import OwnCoreXAuditCache
cache = OwnCoreXAuditCache(root/'CoreX-attention-fusion-20261028-03/numeric', root/'CoreX-attention-fusion-20261028-03/checkpoints-rs04')
undo.append(cache.install())
heap_records=[]
from corex_audit_heap_release import release,install as install_heap_release
undo.append(install_heap_release(heap_records))
from node0_optimizer_state import install as install_node0,bind_untimed_audit
state['CPU_state_layout']={}
scope,undo_node0=install_node0(torch,state['CPU_state_layout']);undo.append(undo_node0)
state['events']=state['CPU_state_layout']['factory_events']
import pinning_full_audit
undo.append(bind_untimed_audit(pinning_full_audit,scope))
from attention_fusion_adapter import install as install_attention
state['attention']={};undo.append(install_attention('native-softmax' if arm=='optimized' else 'native',state['attention']))
# Drop only the dummy-update gradient scratch after the original dummy step.
# All master/moment/step state and original native load/update functions remain live.
def own_memory(label):
    import time
    values={x.split(':',1)[0]:int(x.split()[1])*1024 for x in Path('/proc/self/status').read_text().splitlines() if x.startswith(('VmData:','VmRSS:','VmHWM:'))}
    row={'label':label,'epoch':time.time(),'PID':os.getpid(),'values':values}
    with (run_root/f'load-lifetime-rank-{rank}.jsonl').open('a')as f:f.write(json.dumps(row)+'\n');f.flush()
    return row
import weakref
owned_optimizer_refs=[]
original_init_sub=module.HybridDeviceOptimizer._init_sub_optimizers
def track_initialized(opt,*args,**kwargs):
    value=original_init_sub(opt,*args,**kwargs);owned_optimizer_refs.append(weakref.ref(opt));return value
module.HybridDeviceOptimizer._init_sub_optimizers=track_initialized
undo.append(lambda:setattr(module.HybridDeviceOptimizer,'_init_sub_optimizers',original_init_sub))
def drop_owned_gradient_scratch(label):
    if stage!='resume':return
    torch.cuda.synchronize();total=0;groups=0;seen=set()
    for ref in owned_optimizer_refs:
        opt=ref()
        if opt is None or id(opt)in seen:continue
        seen.add(id(opt));items=list(opt.cpu_copy_map_grad.items())
        for parameter,buffer in items:
            assert parameter.grad is None or parameter.grad is buffer
            total+=buffer.numel()*buffer.element_size();groups+=1;parameter.grad=None
        opt.cpu_copy_map_grad.clear();items.clear()
    if groups:
        heap_records.append(release(label));own_memory(label+'-after-release')
        state.setdefault('pre_load_scratch_release',[]).append({'label':label,'groups':groups,'bytes':total,'no_master_moment_step_or_RNG_modified':True})

original_dummy=module.HybridDeviceOptimizer.dummy_step
state['dummy_gradient_scratch_released']=[];state['gradient_scratch_recreated']=[]
def scoped_dummy(opt,*args,**kwargs):
    result=original_dummy(opt,*args,**kwargs)
    if stage=='resume':
        torch.cuda.synchronize();before=own_memory('dummy-complete-before-scratch-release');items=list(opt.cpu_copy_map_grad.items());assert len(items)==199
        amount=sum(v.numel()*v.element_size()for _,v in items)
        for parameter,buffer in items:
            assert parameter.grad is None or parameter.grad is buffer
            parameter.grad=None
        opt.cpu_copy_map_grad.clear();items.clear()
        heap_records.append(release('after-native-dummy-gradient-scratch-release'))
        state['dummy_gradient_scratch_released'].append({'groups':199,'bytes':amount,'before':before,'after':own_memory('dummy-gradient-scratch-released'),'master_moments_steps_not_removed':True})
    return result
module.HybridDeviceOptimizer.dummy_step=scoped_dummy
undo.append(lambda:setattr(module.HybridDeviceOptimizer,'dummy_step',original_dummy))

step = module.HybridDeviceOptimizer.step

def guarded_step(opt, *args, **kwargs):
    heap_records.append(release('before-HDO-call-'+str(state['HDO_update_calls'])))
    guard.check_live(2*1024**3)
    if state['HDO_update_calls'] == 0:
        cache.first_update()
        state['binding'] = activate(control/'numa-plan.json', mode, rank, run_root)
    if stage=='resume' and not opt.cpu_copy_map_grad:
        with scope():
            masters=list(opt.gpu_params_map_cpu_copy.values());assert len(masters)==199
            for parameter in masters:
                buffer=torch.empty(parameter.shape,dtype=parameter.dtype,device='cpu',pin_memory=opt.pin_cpu_grads);buffer.zero_();opt.cpu_copy_map_grad[parameter]=buffer;parameter.grad=buffer
            from numa_placement import sampled_page_nodes
            query=sampled_page_nodes([(str(i),opt.cpu_copy_map_grad[p])for i,p in enumerate(masters)])
            assert query['return_code']==0 and set(query['sample_nodes'])=={0}
            state['gradient_scratch_recreated'].append({'groups':199,'page_query':query,'before_original_full_D2H_overwrite':True})
    result = step(opt, *args, **kwargs)
    state['HDO_update_calls'] += 1
    return result
module.HybridDeviceOptimizer.step = guarded_step

import optimizer_evidence
original_save = optimizer_evidence.save_optimizer_evidence
def save_with_pages(opt, path):
    result = original_save(opt, path)
    from pinning_resume_audit import optimizer_leaves
    from numa_placement import sampled_page_nodes
    leaves = [value for value in optimizer_leaves(opt)
              if isinstance(value, module.HybridDeviceOptimizer)]
    assert len(leaves) == 1
    tensors = []
    for i, cpu in enumerate(leaves[0].cpu_optimizers):
        for j, (parameter, value) in enumerate(cpu.state.items()):
            for name in ('exp_avg', 'exp_avg_sq'):
                tensors.append((f'{i}/{j}/{name}', value[name]))
    assert len(tensors) == 398
    state['final_moment_page_query'] = sampled_page_nodes(tensors)
    if stage == 'resume':
        query = state['final_moment_page_query']
        node = 0
        assert query['return_code'] == 0 and set(query['sample_nodes']) == {node}
        events = state['events']
        assert len(events) == 796
        assert all(e['initialization_path'] == 'direct_local_zero' for e in events[:398])
        assert all(e['initialization_path'] == 'native_resume_copy' for e in events[398:])
    state['kernel_guard_after_training'] = guard.snapshot()
    return result
optimizer_evidence.save_optimizer_evidence = save_with_pages

# Apply the exact accepted pinned-master derivation to every repaired arm.
# Native save/load APIs and optimizer math remain the original methods.
original_run = runpy.run_path
from sft_phased_acceptance_20261024 import pinned_backend
def dispatch(path, init_globals=None, run_name=None):
    if Path(path) == root/'pinning_entry.py':
        source=root/'pinning_entry.py'
        text=pinned_backend(source.read_bytes(),formal=False)
        namespace={'__name__':run_name or '<strict-common-pinned-native-save-resume>',
                   '__file__':str(source),'__package__':None,**(init_globals or {})}
        exec(compile(text,'<strict-common-pinned-native-save-resume>','exec'),namespace)
        return namespace
    return original_run(path,init_globals=init_globals,run_name=run_name)
runpy.run_path=dispatch
try:
    runpy.run_path(str(root/'sft_pinning_entry.py'), run_name='__main__')
    assert stage == 'resume' and state['HDO_update_calls'] == 2
    state['status'] = 'passed'
except SystemExit as error:
    if stage == 'checkpoint' and error.code in (None, 0):
        assert state['HDO_update_calls'] == 2
        state.update(status='passed', native_exit_interval_2=True)
    else:
        state.update(status='failed', error=repr(error))
    raise
except BaseException as error:
    import traceback
    state.update(status='failed', error=str(error), traceback=traceback.format_exc())
    raise
finally:
    runpy.run_path = original_run
    optimizer_evidence.save_optimizer_evidence = original_save
    module.HybridDeviceOptimizer.step = step
    for function in reversed(undo):
        function()
    cache.save(run_root/f'CoreX-own-cache-rank-{rank}.json')
    with (run_root/f'CoreX-own-heap-release-rank-{rank}.json').open('x') as f:json.dump({'records':heap_records,'no_tensor_math_or_guard_modified':True,'installed_in_formal':False},f,indent=2)
    with (run_root/f'CoreX-policy-rank-{rank}.json').open('x') as stream:
        json.dump(state, stream, indent=2)

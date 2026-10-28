"""Real7B three updates; original full-state comparator and old native compatibility gate retained."""
import hashlib,json,os,runpy,sys
from pathlib import Path
R=Path(os.environ['JIUGE_DT_ROOT']);C=Path(os.environ['JIUGE_DT_CONTROL']);CAMP=Path(os.environ['JIUGE_ATTENTION_CAMPAIGN']);sys.path[:0]=[str(C),str(R),str(R/'megatron-native-5be9626')]
import corex_process_guard_data as guard
sys.modules['corex_process_guard']=guard;guard.enforce_rank_limit();rank=int(os.environ['RANK']);arm=os.environ['JIUGE_DT_ARM'];assert arm in('native','control','optimized')
run_root=Path(sys.argv[sys.argv.index('--jiuge-run-root')+1]);state={'arm':arm,'rank':rank,'HDO_update_calls':0,'passed':False,'candidate':'native-softmax with fused scale/mask/casts only','old_reference_gate_retained':True,'CPU_state_layout_common_all_arms':True}
from sft_phased_placement_20261023 import prepare,activate
setup=prepare(C/'numa-plan.json','gpu-local8',rank);(run_root/f'CoreX-setup-affinity-rank-{rank}.json').write_text(json.dumps(setup,indent=2))
import torch
# Preserve the already accepted original GPU execution policy; do not silently
# substitute the separate strict-mode diagnostic baseline for the old reference.
torch.use_deterministic_algorithms(False);state['deterministic_algorithms']=False
import megatron.core.optimizer.cpu_offloading.hybrid_optimizer as hdo
from megatron.core.transformer.torch_norm import WrappedTorchNorm
import megatron.core.models.gpt.gpt_layer_specs as specs
import megatron.core.transformer.transformer_block as block
specs.LNImpl=WrappedTorchNorm;block.LayerNormImpl=WrappedTorchNorm
undo=[]
from corex_setup_gradients_v3 import install as gradients
state['gradient_preallocation']={'setup':setup};undo.append(gradients(torch,hdo.HybridDeviceOptimizer,state['gradient_preallocation']))
import pinning_full_audit
from corex_fullstate_reference_v6 import install as reference
undo.append(reference(pinning_full_audit,CAMP/'numeric',rank))
from node0_optimizer_state import install as node0,bind_untimed_audit
state['CPU_state_layout']={};scope,undo_node0=node0(torch,state['CPU_state_layout']);undo.append(undo_node0);undo.append(bind_untimed_audit(pinning_full_audit,scope))
from corex_audit_heap_release import release,install as heap
heap_records=[];undo.append(heap(heap_records))
from corex_own_audit_cache import OwnCoreXAuditCache
cache=OwnCoreXAuditCache(CAMP/'numeric',CAMP/'checkpoints');undo.append(cache.install())
from attention_fusion_adapter import install as attention
state['attention']={};undo.append(attention('native-softmax'if arm=='optimized'else'native',state['attention']))
step=hdo.HybridDeviceOptimizer.step
def guarded(opt,*args,**kwargs):
 heap_records.append(release('before-HDO-'+str(state['HDO_update_calls'])));guard.check_live(2*1024**3)
 if state['HDO_update_calls']==0:
  cache.first_update();state['binding']=activate(C/'numa-plan.json','unbound'if arm=='native'else'gpu-local8',rank,run_root)
 value=step(opt,*args,**kwargs);state['HDO_update_calls']+=1;return value
hdo.HybridDeviceOptimizer.step=guarded
original_run=runpy.run_path
from sft_phased_acceptance_20261024 import pinned_backend
def dispatch(path,init_globals=None,run_name=None):
 if Path(path)==R/'pinning_entry.py':
  source=R/'pinning_entry.py';text=pinned_backend(source.read_bytes(),formal=False);namespace={'__name__':run_name or '<same-pinned-attention-audit>','__file__':str(source),'__package__':None,**(init_globals or {})};exec(compile(text,'<same-pinned-attention-audit>','exec'),namespace);return namespace
 return original_run(path,init_globals=init_globals,run_name=run_name)
runpy.run_path=dispatch
try:
 original_run(str(R/'sft_pinning_entry.py'),run_name='__main__');assert state['HDO_update_calls']==3
 from pinning_resume_audit import optimizer_leaves
 # FullAudit already covers all199 groups and all398 moments numerically.
 assert len(state['CPU_state_layout']['events'])==199 and sum(x['elements']for x in state['CPU_state_layout']['events'])==3834518016
 assert len(state['CPU_state_layout']['factory_events'])==398 and all(x['initialization_path']=='direct_local_zero'for x in state['CPU_state_layout']['factory_events'])
 assert state['attention']['forward_calls']>=28*3;state['passed']=True
except BaseException as e:
 import traceback
 state.update(error=repr(e),traceback=traceback.format_exc());raise
finally:
 runpy.run_path=original_run;hdo.HybridDeviceOptimizer.step=step
 for fn in reversed(undo):fn()
 cache.save(run_root/f'CoreX-own-cache-rank-{rank}.json');(run_root/f'CoreX-own-heap-release-rank-{rank}.json').write_text(json.dumps({'records':heap_records,'installed_in_performance':False},indent=2));(run_root/f'attention-policy-rank-{rank}.json').write_text(json.dumps(state,indent=2))

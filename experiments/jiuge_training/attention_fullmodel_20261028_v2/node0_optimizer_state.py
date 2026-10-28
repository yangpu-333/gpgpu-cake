"""Common placement only: original AdamW init/update math, private audit allocations on node0."""
import contextlib,json,os
from pathlib import Path

def install(torch,receipt):
 from corex_process_guard_data import check_live
 from sft_phased_placement_20261023 import check_node_values,node_memory
 from numa_placement import sampled_page_nodes
 control=Path(os.environ['JIUGE_DT_CONTROL']);plan=json.loads((control/'numa-plan.json').read_text());rank=int(os.environ['RANK']);cores={}
 for cpu in plan['topology']['allowed_cpus']:
  row=plan['topology']['cpu_topology'][str(cpu)]
  if row['node']==0:cores.setdefault((row['package'],row['core']),cpu)
 selected=sorted(cores.values())[rank*8:(rank+1)*8];assert len(selected)==8
 @contextlib.contextmanager
 def scope():
  prior={int(p.name):set(os.sched_getaffinity(int(p.name)))for p in Path('/proc/self/task').iterdir()};default=prior[os.getpid()]
  try:
   for p in Path('/proc/self/task').iterdir():
    try:os.sched_setaffinity(int(p.name),selected)
    except ProcessLookupError:pass
   yield
  finally:
   for p in Path('/proc/self/task').iterdir():
    try:os.sched_setaffinity(int(p.name),prior.get(int(p.name),default))
    except ProcessLookupError:pass
 import sft_local_moments_v4_20261023 as frozen
 old_zero,old_copy=frozen.create_local_zero,frozen.relocate_moment
 def guarded_zero(parameter,node,runtime):
  assert node==0;check_node_values(node_memory(0)['MemFree'],parameter.numel()*parameter.element_size());return old_zero(parameter,node,runtime)
 def guarded_copy(value,node,runtime):
  assert node==0;check_node_values(node_memory(0)['MemFree'],value.numel()*value.element_size());return old_copy(value,node,runtime)
 frozen.create_local_zero,frozen.relocate_moment=guarded_zero,guarded_copy
 receipt.update(common_all_arms=True,CPU_state_node=0,update_formula_unchanged=True,events=[],factory_events=[],only_two_zero_factories_replaced=True,uses_previously_validated_direct_local_zero=True)
 from corex_adamw_local_moments import install as factories
 with scope():undo_factories=factories(torch,receipt['factory_events'])
 original=torch.optim.AdamW._init_group
 def initialize(opt,group,*args,**kwargs):
  new=[p for p in group['params']if p.grad is not None and not opt.state[p]]
  with scope():
   for p in new:
    amount=p.numel()*p.element_size();check_live(amount);check_node_values(node_memory(0)['MemFree'],amount)
   value=original(opt,group,*args,**kwargs)
   for p in new:
    state=opt.state[p];query=sampled_page_nodes([(n,state[n])for n in('exp_avg','exp_avg_sq')]);assert query['return_code']==0 and set(query['sample_nodes'])=={0}
    receipt['events'].append({'elements':p.numel(),'dtype':str(p.dtype),'moment_page_query':query,'original_init_no_extra_update':True})
   return value
 torch.optim.AdamW._init_group=initialize
 def undo():
  assert torch.optim.AdamW._init_group is initialize;torch.optim.AdamW._init_group=original;undo_factories()
  assert frozen.create_local_zero is guarded_zero and frozen.relocate_moment is guarded_copy
  frozen.create_local_zero,frozen.relocate_moment=old_zero,old_copy
 return scope,undo

def bind_untimed_audit(module,scope):
 old=module.FullAudit.tensor
 def tensor(self,*args,**kwargs):
  with scope():return old(self,*args,**kwargs)
 module.FullAudit.tensor=tensor
 return lambda:setattr(module.FullAudit,'tensor',old)

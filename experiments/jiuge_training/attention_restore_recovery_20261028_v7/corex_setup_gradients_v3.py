"""One-time allocation of the native pinned gradient buffers on setup node0.

No transfer/update method or live parameter value is changed. Update affinity is
restored before returning. Original allocation and physical reserves remain.
"""
import os
from pathlib import Path

def install(torch,hdo,receipt):
 from corex_process_guard_data import check_live
 from sft_phased_placement_20261023 import node_memory,check_node_values
 from numa_placement import sampled_page_nodes
 original=hdo._init_sub_optimizers
 def initialize(opt):
  prior={int(p.name):sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()}
  default=prior[os.getpid()]
  # Use the exact existing two-phase setup assignment, not an invented CPU set.
  plan=receipt['setup'];selected=plan.get('allowed_after') if plan.get('selected_cpu_nodes')==[0] else None
  if selected is None:
   import json
   payload=json.loads(Path(os.environ['JIUGE_DT_CONTROL'],'numa-plan.json').read_text())
   topology=payload['topology'];cores={}
   for cpu in topology['allowed_cpus']:
    t=topology['cpu_topology'][str(cpu)]
    if t['node']==0:cores.setdefault((t['package'],t['core']),cpu)
   cpus=sorted(cores.values());rank=int(os.environ['RANK'])
   selected=cpus[rank*8:(rank+1)*8]
  assert len(selected)==8 and all(x in default for x in selected)
  changed=[]
  try:
   for p in Path('/proc/self/task').iterdir():
    try:os.sched_setaffinity(int(p.name),selected);changed.append(int(p.name))
    except ProcessLookupError:pass
   value=original(opt)
   masters=list(opt.gpu_params_map_cpu_copy.values())
   assert len(masters)==199 and sum(x.numel() for x in masters)==3834518016
   assert not opt.cpu_copy_map_grad
   for p in masters:
    amount=p.numel()*p.element_size();check_live(amount)
    buf=torch.empty(p.shape,dtype=p.dtype,device='cpu',pin_memory=opt.pin_cpu_grads)
    buf.zero_();opt.cpu_copy_map_grad[p]=buf;p.grad=buf
   master_pages=sampled_page_nodes([(str(i),x) for i,x in enumerate(masters)])
   grad_pages=sampled_page_nodes([(str(i),opt.cpu_copy_map_grad[x]) for i,x in enumerate(masters)])
   for query in (master_pages,grad_pages):assert query['return_code']==0 and set(query['sample_nodes'])=={0}
   receipt.setdefault('initializations',[]).append({'groups':199,'elements':3834518016,'setup_CPUs':selected,'master_pages':master_pages,'gradient_pages':grad_pages,'buffer_created_once_before_training':True,'update_math_or_transfer_changed':False})
   return value
  finally:
   for p in Path('/proc/self/task').iterdir():
    try:os.sched_setaffinity(int(p.name),prior.get(int(p.name),default))
    except ProcessLookupError:pass
 hdo._init_sub_optimizers=initialize
 return lambda:setattr(hdo,'_init_sub_optimizers',original)

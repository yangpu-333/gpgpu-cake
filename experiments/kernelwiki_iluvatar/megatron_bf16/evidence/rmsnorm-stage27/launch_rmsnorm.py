"""Explicit process-only RMSNorm hook around unchanged CE+gradnorm launcher."""
import argparse,hashlib,importlib.util,json,runpy,sys
from pathlib import Path

p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
p.add_argument('--rmsnorm-candidate',type=Path);p.add_argument('--rmsnorm-receipt',type=Path,required=True)
p.add_argument('--gradnorm-launcher',type=Path,required=True)
a,rest=p.parse_known_args();assert not a.rmsnorm_receipt.exists()
bf=Path(rest[rest.index('--bf16-root')+1]);sys.path.insert(0,str(bf.parent/'megatron_training_entry'))
import runtime_compat
original_prepare=runtime_compat.prepare_local_runtime
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'enabled':a.rmsnorm_candidate is not None,'calls':0,'fallback_calls':0,'wrapper_sha256':sha(Path(__file__)),
         'candidate_sha256':sha(a.rmsnorm_candidate) if a.rmsnorm_candidate else None,'shapes':{}}
def prepare(root):
 result=original_prepare(root)
 if a.rmsnorm_candidate:
  import torch
  spec=importlib.util.spec_from_file_location('candidate_rmsnorm',a.rmsnorm_candidate);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  native=torch.nn.RMSNorm.forward
  def forward(self,x):
   if self.normalized_shape!=(512,) or self.weight is None:
    receipt['fallback_calls']+=1;return native(self,x)
   receipt['calls']+=1
   key=str((tuple(x.shape),tuple(x.stride()),str(x.dtype),str(self.weight.dtype)))
   receipt['shapes'][key]=receipt['shapes'].get(key,0)+1
   try:return m.rms_norm(x,self.weight,self.eps)
   except NotImplementedError:
    receipt['fallback_calls']+=1;return native(self,x)
  torch.nn.RMSNorm.forward=forward
 return result
runtime_compat.prepare_local_runtime=prepare
sys.argv=[str(a.gradnorm_launcher)]+rest
try:runpy.run_path(str(a.gradnorm_launcher),run_name='__main__')
finally:a.rmsnorm_receipt.write_text(json.dumps(receipt,indent=2)+'\n')

"""Probe pinned native SwiGLU default fusion; never used as optimized baseline code."""
import json,sys,hashlib,traceback
from pathlib import Path
import torch
sys.path.insert(0,'/tmp/kda-bf16-continuation/megatron_training_entry')
from runtime_compat import prepare_local_runtime
root=Path('/tmp/kda-bf16-continuation/megatron-native-5be9626')
receipt={'runtime':prepare_local_runtime(root),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)}
from megatron.core.fusions import fused_bias_swiglu as f
x=torch.randn(17,2,514,device='cuda',dtype=torch.bfloat16,requires_grad=True)
try:
 y=f.bias_swiglu_impl(x,None,False,False);g=torch.autograd.grad(y,x,torch.ones_like(y))[0];torch.cuda.synchronize()
 receipt.update(status='passed',shape=list(y.shape),finite_output=bool(torch.isfinite(y).all()),finite_grad=bool(torch.isfinite(g).all()))
except Exception as e:receipt.update(status='failed',exception=type(e).__name__,message=str(e),traceback=traceback.format_exc())
path=Path('/tmp/kda-swiglu-stage29/native-default-probe.json');assert not path.exists();path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('runtime','traceback')}),flush=True)

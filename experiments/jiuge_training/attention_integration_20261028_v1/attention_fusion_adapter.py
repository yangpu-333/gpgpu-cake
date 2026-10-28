"""Replace only MCore fallback scale/mask/softmax, preserving QK and BF16 boundaries."""
import hashlib,math
from pathlib import Path

EXPECTED_SOFTMAX='a75dc9d23380ff5d2ccdbfaa335c050bc4d16039fe35797f96dc108bad011e68'

def install(mode,receipt):
 import torch
 import megatron.core.fusions.fused_softmax as module
 from megatron.core.transformer.enums import AttnMaskType
 from fused_scale_mask_softmax import fused_scale_mask_softmax
 assert mode in('native','fused')
 assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==EXPECTED_SOFTMAX
 original=module.FusedScaleMaskSoftmax.forward
 receipt.update(mode=mode,forward_calls=0,configurations={},QK_baddbmm_alpha_unchanged=True,BF16_output_boundary_preserved=True,dropout_RNG_and_recomputation_outside_replacement=True)
 def forward(self,input,mask,softmax_offset=None):
  assert input.is_cuda and input.dtype==torch.bfloat16 and input.is_contiguous()
  assert input.ndim==4 and input.shape[0]==1 and input.shape[1]==14 and input.shape[-1]==input.shape[-2]and input.shape[-1]<=4096
  assert self.input_in_bf16 and not self.input_in_fp16 and self.softmax_in_fp32
  assert self.attn_mask_type==AttnMaskType.causal and self.window_size is None and softmax_offset is None
  assert not self.is_kernel_available(mask,*input.shape),'Baseline must be the actual unmodified torch fallback'
  scale=1.0 if self.scale is None else float(self.scale);assert math.isfinite(scale)
  key=str((tuple(input.shape),str(input.dtype),self.scale,mask is None));receipt['configurations'][key]=receipt['configurations'].get(key,0)+1;receipt['forward_calls']+=1
  if mode=='native':return original(self,input,mask,softmax_offset)
  # QK alpha was already applied and rounded by the original BF16 baddbmm.
  # self.scale is the *additional* layer coefficient, never 1/sqrt(head_dim).
  probabilities=fused_scale_mask_softmax(input,scale,mask)
  return probabilities.bfloat16()
 module.FusedScaleMaskSoftmax.forward=forward
 def undo():
  assert module.FusedScaleMaskSoftmax.forward is forward
  module.FusedScaleMaskSoftmax.forward=original
 return undo

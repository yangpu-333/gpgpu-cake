"""Actual MCore TP2 attention with QK/PV and recomputation; no model/optimizer claim."""
import hashlib,json,os,sys,time,traceback
from pathlib import Path
R=Path(os.environ['JIUGE_DT_ROOT']);C=Path(os.environ['JIUGE_DT_CONTROL']);sys.path[:0]=[str(C),str(R),str(R/'megatron-native-5be9626')]
rank=int(os.environ['RANK']);cpus=[int(x)for x in os.environ['JIUGE_PROBE_NODE0_CPUS'].split(',')];os.sched_setaffinity(0,cpus[rank*8:(rank+1)*8]);out=C/'actual-attention-probe'/f'rank-{rank}';out.mkdir(parents=True,exist_ok=False)
r={'rank':rank,'scope':'actual MCore TP2 DotProductAttention, synthetic Q/K/V at real7B shape; not full7B weights or optimizer updates','cases':[],'passed':False,'full_model_accepted':False}
try:
 from corex_process_guard_data import enforce_rank_limit,check_live
 enforce_rank_limit()
 import torch
 import torch.distributed as dist
 from torch.utils.checkpoint import checkpoint
 torch.cuda.set_device(rank);torch.set_num_threads(8);torch.use_deterministic_algorithms(True,warn_only=False)
 dist.init_process_group('nccl')
 from megatron.core import parallel_state,tensor_parallel
 parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
 tensor_parallel.model_parallel_cuda_manual_seed(63100)
 from megatron.core.transformer.transformer_config import TransformerConfig
 from megatron.core.transformer.dot_product_attention import DotProductAttention
 from megatron.core.transformer.enums import AttnMaskType
 from attention_fusion_adapter import install
 atol=3e-4;rtol=1e-3
 def compare(a,b):
  assert a.shape==b.shape and a.dtype==b.dtype
  aa=a.detach().reshape(-1);bb=b.detach().reshape(-1);bad=0;mx=0.;count=0;nonfinite=0;hs=[hashlib.sha256(),hashlib.sha256()]
  for start in range(0,aa.numel(),1048576):
   check_live(128*1024**2);x=aa[start:start+1048576].cpu();y=bb[start:start+1048576].cpu()
   for h,t in zip(hs,(x,y)):h.update(memoryview(t.contiguous().view(torch.uint8).numpy()).cast('B'))
   x=x.double();y=y.double();err=(x-y).abs();bad+=int((err>atol+rtol*y.abs()).sum());mx=max(mx,float(err.max()));nonfinite+=int((~torch.isfinite(x)).sum()+(~torch.isfinite(y)).sum());count+=x.numel()
  return {'elements':count,'max_abs':mx,'bad_original_tolerance':bad,'nonfinite':nonfinite,'native_SHA256':hs[0].hexdigest(),'fused_SHA256':hs[1].hexdigest(),'passed':not bad and not nonfinite}
 for layering in(False,True):
  for layer in(1,14,28):
   for recompute in(False,True):
    cfg=TransformerConfig(num_layers=28,hidden_size=3584,num_attention_heads=28,num_query_groups=28,kv_channels=128,bf16=True,params_dtype=torch.bfloat16,attention_softmax_in_fp32=True,apply_query_key_layer_scaling=layering,attention_dropout=0.0,hidden_dropout=0.0,masked_softmax_fusion=False)
    attention=DotProductAttention(cfg,layer,AttnMaskType.causal,'self')
    torch.manual_seed(63100+layer);q,k,v=[torch.randn((2048,1,14,128),device='cuda',dtype=torch.bfloat16,requires_grad=True)for _ in range(3)];g=torch.randn((2048,1,1792),device='cuda',dtype=torch.bfloat16)
    def call(q,k,v):return attention(q,k,v,None)
    outputs=[];grads=[];policies=[];rng=[]
    initial_cuda=torch.cuda.get_rng_state();initial_tp=tensor_parallel.get_cuda_rng_tracker().get_states()
    for mode in('native','fused'):
     torch.cuda.set_rng_state(initial_cuda);tensor_parallel.get_cuda_rng_tracker().set_states(initial_tp);policy={};undo=install(mode,policy)
     try:
      for x in(q,k,v):x.grad=None
      y=checkpoint(call,q,k,v,use_reentrant=True,preserve_rng_state=True)if recompute else call(q,k,v)
      y.backward(g);d=tuple(x.grad.detach().clone()for x in(q,k,v));torch.cuda.synchronize();outputs.append(y.detach());grads.append(d);policies.append(policy)
      rng.append({'CUDA':hashlib.sha256(torch.cuda.get_rng_state().numpy().tobytes()).hexdigest(),'TP':{n:hashlib.sha256(x.numpy().tobytes()).hexdigest()for n,x in tensor_parallel.get_cuda_rng_tracker().get_states().items()}})
     finally:undo()
    metrics={'output':compare(outputs[1],outputs[0]),**{n:compare(a,b)for n,a,b in zip(('dq','dk','dv'),grads[1],grads[0])}}
    row={'layer_scaling':layering,'layer':layer,'recompute':recompute,'actual_QK_alpha':attention.softmax_scale,'additional_softmax_scale':attention.scale_mask_softmax.scale,'metrics':metrics,'policies':policies,'RNG_equal':rng[0]==rng[1]};row['passed']=all(x['passed']for x in metrics.values())and row['RNG_equal'];r['cases'].append(row)
    with(out/'journal.jsonl').open('a')as f:f.write(json.dumps(row)+'\n');f.flush()
    print(json.dumps({'layer':layer,'layer_scaling':layering,'recompute':recompute,'passed':row['passed'],'max_abs':{k:x['max_abs']for k,x in metrics.items()}}),flush=True)
    assert row['passed'],'Real attention QKV/output original numerical gate rejected'
    del q,k,v,g,y,d,outputs,grads,attention
 r['passed']=len(r['cases'])==12
except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());raise
finally:
 r['finished_epoch']=time.time();(out/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps({'rank':rank,'passed':r['passed'],'cases':len(r['cases']),'error':r.get('error')}),flush=True)
 if 'dist'in globals()and dist.is_initialized():dist.destroy_process_group()

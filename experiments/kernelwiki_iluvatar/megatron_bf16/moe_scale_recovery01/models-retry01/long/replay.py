"""Untimed diagnostic: replay all captured original native expert activations.

This does not alter the frozen admission gate. It localizes candidate arithmetic
and rounding differences on the SAME original upstream gradients.
"""
import argparse,fcntl,hashlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--version',required=True);v=a.parse_args().version
out=ROOT/('diagnostics-'+v+'.json');assert not out.exists()
sys.path.insert(0,'/tmp/kda-bf16-continuation/megatron_training_entry')
from runtime_compat import prepare_local_runtime
prepare_local_runtime(Path('/tmp/kda-bf16-continuation/megatron-native-5be9626'))
import torch
torch.set_num_threads(4)
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
path=ROOT/'candidates'/v/'candidate.py'
spec=importlib.util.spec_from_file_location('weighted_candidate_diagnostic',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r={'schema':'moe-weighted-stage35-heldout-actual-native-replay-v1','candidate_sha256':sha(path),'verifier_sha256':sha(__file__),'atol':3e-4,'rtol':1e-3,'checks':[]}
def compare(l,z):
 ll,zz=l.detach().cpu().to(torch.float64),z.detach().cpu().to(torch.float64)
 diff=(ll-zz).abs();bad=int(torch.count_nonzero(diff>3e-4+1e-3*ll.abs()))
 return {'within_original_tolerance':bad==0,'exact_equal':bool(torch.equal(l.cpu(),z.cpu())),'nonexact_elements':int(torch.count_nonzero(ll!=zz)),
         'bad_elements':bad,'max_abs':float(diff.max()) if diff.numel() else 0.}
with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  for i in range(1,json.loads((ROOT/'contract.json').read_text())['model']['num_layers']*3*json.loads((ROOT/'contract.json').read_text())['model']['num_moe_experts']+1):
   capture=ROOT/'native-admission01/activation'/('capture-%03d.pt'%i)
   value=torch.load(capture,map_location='cpu',weights_only=True)
   x=value['input'].cuda().requires_grad_();w=value['weights'].cuda().requires_grad_();g=value['upstream'].cuda()
   y=weighted_bias_swiglu_impl(x,None,w,False);dx,dw=torch.autograd.grad(y,(x,w),g)
   native_replay={n:compare(value[n],z) for n,z in zip(('output','input_grad','weights_grad'),(y,dx,dw))}
   assert all(z['exact_equal'] for z in native_replay.values()),native_replay
   seen=[]
   y=m.weighted_swiglu(x,w,native=lambda xx,ww:(seen.append(True) or weighted_bias_swiglu_impl(xx,None,ww,False)))
   dx,dw=torch.autograd.grad(y,(x,w),g)
   checks={n:compare(value[n],z) for n,z in zip(('output','input_grad','weights_grad'),(y,dx,dw))}
   row={'index':i,'capture_sha256':sha(capture),'input_shape':list(x.shape),'checks':checks,'native_replay':native_replay,'fallback_calls':len(seen)}
   r['checks'].append(row)
   if not all(z['exact_equal'] for z in checks.values()):print(json.dumps(row),flush=True)
  r['passed']=all(row['fallback_calls']==0 and all(z['within_original_tolerance'] for z in row['checks'].values()) for row in r['checks'])
  r['status']='complete' if r['passed'] else 'rejected'
 except Exception as e:r.update(status='error',error_type=type(e).__name__,error=str(e))
 out.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'status':r['status'],'cases':len(r['checks']),'error':r.get('error'),'nonexact_cases':[z['index'] for z in r['checks'] if not all(t['exact_equal'] for t in z['checks'].values())]}))
raise SystemExit(0 if r['status']=='complete' else 1)

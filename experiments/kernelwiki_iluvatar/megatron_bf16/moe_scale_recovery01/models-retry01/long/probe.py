"""Independent fixed weighted-SwiGLU oracle, actual capture replay and Event microbenchmark."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
NATIVE = Path('/tmp/kda-bf16-continuation/megatron-native-5be9626')
p = argparse.ArgumentParser()
p.add_argument('--version', required=True)
a = p.parse_args()
candidate = ROOT / 'candidates' / a.version / 'candidate.py'
output = ROOT / ('probe-' + a.version + '.json'); assert not output.exists()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
sys.path.insert(0, '/tmp/kda-bf16-continuation/megatron_training_entry')
from runtime_compat import prepare_local_runtime
runtime = prepare_local_runtime(NATIVE)
import torch
from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl
torch.set_num_threads(4)
spec = importlib.util.spec_from_file_location('candidate_weighted', candidate)
m = importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
report = {'schema':'weighted-swiglu-stage34-independent-probe-v1','status':'running','version':a.version,
    'candidate_sha256':sha(candidate),'verifier_sha256':sha(__file__),'runtime':runtime,
    'native_source_sha256':sha(NATIVE/'megatron/core/fusions/fused_bias_swiglu.py'),
    'atol':3e-4,'rtol':1e-3,'comparison':'CPU float64','checks':[],'guards':[],'timing':[]}
fallback_calls = 0
def native(x,w):
    return weighted_bias_swiglu_impl(x,None,w,False)
def fallback(x,w):
    global fallback_calls
    fallback_calls += 1
    return native(x,w)
def run(fn,x,w,g):
    xx=x.detach().clone().requires_grad_(True);ww=w.detach().clone().requires_grad_(True)
    y=fn(xx,ww)
    dx,dw=torch.autograd.grad(y,(xx,ww),g)
    return y.detach(),dx.detach(),dw.detach()
def compare(left,right):
    left,right=left.detach().cpu(),right.detach().cpu()
    valid=left.shape==right.shape and left.dtype==right.dtype
    if not valid:return {'passed':False,'shape_dtype_match':False}
    ll,rr=left.to(torch.float64),right.to(torch.float64)
    finite=bool(torch.isfinite(ll).all() and torch.isfinite(rr).all())
    error=(ll-rr).abs();limit=3e-4+1e-3*ll.abs()
    bad=int(torch.count_nonzero(error>limit).item())
    return {'passed':finite and bad==0,'shape_dtype_match':True,'finite':finite,'bad_elements':bad,
            'max_abs':float(error.max()) if error.numel() else 0.,
            'exact_equal':bool(torch.equal(left,right))}
def check(label,x,w,g,capture=None):
    global fallback_calls
    xx,ww=x.clone(),w.clone();before=fallback_calls
    reference=run(native,x,w,g)
    actual=run(lambda x,w:m.weighted_swiglu(x,w,native=fallback),x,w,g)
    checks={name:compare(l,r) for name,l,r in zip(('output','input_grad','weights_grad'),reference,actual)}
    row={'case':label,'input_shape':list(x.shape),'weights_shape':list(w.shape),
        'dtype':str(x.dtype),'weights_dtype':str(w.dtype),'checks':checks,
        'native_fallback_calls':fallback_calls-before,'inputs_unchanged':torch.equal(x,xx) and torch.equal(w,ww)}
    if capture:
        row['capture_replay']={name:compare(capture[name],ref) for name,ref in
                              zip(('output','input_grad','weights_grad'),reference)}
    row['passed']=(all(v['passed'] for v in checks.values()) and row['native_fallback_calls']==0
        and row['inputs_unchanged'] and (not capture or all(v['passed'] and v['exact_equal']
                                                       for v in row['capture_replay'].values())))
    report['checks'].append(row)
    print(json.dumps({'case':label,'passed':row['passed'],'checks':checks}),flush=True)

with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        torch.manual_seed(34119)
        for n,h in ((0,2048),(1,2048),(17,257),(38,2048),(89,2048),(165,2048),
                    (256,2048),(257,2048),(33,769)):
            for mode in ('zero','one','probability','signed'):
                x=torch.randn(n,2*h,device='cuda',dtype=torch.bfloat16)
                w=torch.rand(n,1,device='cuda',dtype=torch.bfloat16)
                if mode=='zero':w.zero_()
                elif mode=='one':w.fill_(1.)
                elif mode=='signed':w.sub_(.5)
                g=torch.randn(n,h,device='cuda',dtype=torch.bfloat16)
                check(f'{n}-{h}-{mode}',x,w,g)
        for i in range(1,5):
            path=ROOT/'native-admission01/activation'/('capture-%03d.pt'%i)
            value=torch.load(path,map_location='cpu',weights_only=True)
            check('actual-expert-'+str(i),value['input'].cuda(),value['weights'].cuda(),
                  value['upstream'].cuda(),capture=value)
        for n in (17,165):
            x=torch.randn(n*4096+1,device='cuda',dtype=torch.bfloat16)[1:].view(n,4096)
            w=torch.rand(n+1,device='cuda',dtype=torch.bfloat16)[1:].view(n,1)
            g=torch.randn(n,2048,device='cuda',dtype=torch.bfloat16)
            check('offset-'+str(n),x,w,g)
        stream=torch.cuda.Stream()
        for i in range(2):
            with torch.cuda.stream(stream):
                x=torch.randn(81,4096,device='cuda',dtype=torch.bfloat16)
                w=torch.rand(81,1,device='cuda',dtype=torch.bfloat16)
                g=torch.randn(81,2048,device='cuda',dtype=torch.bfloat16)
                check('fresh-input-stream-'+str(i),x,w,g)
            stream.synchronize()
        for device,dtype in (('cpu',torch.bfloat16),('cuda',torch.float32)):
            x=torch.ones(2,4096,device=device,dtype=dtype)
            w=torch.ones(2,1,device=device,dtype=dtype)
            seen=[]
            marker=lambda xx,ww:(seen.append((xx is x,ww is w)) or torch.empty(2,2048,device=device,dtype=dtype))
            y=m.weighted_swiglu(x,w,native=marker)
            report['guards'].append({'device':device,'dtype':str(dtype),'passed':seen==[(True,True)]
                                      and y.shape==(2,2048)})
        passed=all(r['passed'] for r in report['checks']) and all(r['passed'] for r in report['guards'])
        if passed:
            x=torch.randn(128,4096,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            w=torch.rand(128,1,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            g=torch.randn(128,2048,device='cuda',dtype=torch.bfloat16)
            for mode in ('forward','forward-backward'):
                def execute(fn):
                    y=fn(x,w)
                    if mode=='forward-backward':torch.autograd.grad(y,(x,w),g)
                funcs={'native':native,'candidate':lambda x,w:m.weighted_swiglu(x,w,native=fallback)}
                for fn in funcs.values():
                    for _ in range(8):execute(fn)
                torch.cuda.synchronize()
                samples={key:[] for key in funcs}
                for i in range(12):
                    for key in (('native','candidate') if i%2==0 else ('candidate','native')):
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(20):execute(funcs[key])
                        end.record();end.synchronize()
                        samples[key].append(start.elapsed_time(end)/20)
                report['timing'].append({'mode':mode,'shape':[128,4096],'weights_shape':[128,1],
                    'raw_pairs':samples,'native_median_ms':statistics.median(samples['native']),
                    'candidate_median_ms':statistics.median(samples['candidate']),
                    'metric':'CUDA Event API microbenchmark includes host/autograd submission gaps; not formal training gain'})
        report['status']='passed' if passed else 'rejected'
    except Exception as error:
        report.update(status='rejected',error_type=type(error).__name__,error=str(error))
    finally:
        report['total_fallback_calls']=fallback_calls
        output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(report['checks']),'guards':len(report['guards']),
                  'error':report.get('error'),'timing':report['timing']}))
raise SystemExit(0 if report['status']=='passed' else 1)

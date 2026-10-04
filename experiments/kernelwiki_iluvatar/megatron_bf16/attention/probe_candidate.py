"""Independent BF16 native mask+softmax output/backward gate; no training claim."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import torch

BASE = Path('/tmp/kda-attention-stage31')
NATIVE = Path('/tmp/kda-bf16-continuation/megatron-native-5be9626')
p = argparse.ArgumentParser()
p.add_argument('--version', required=True)
p.add_argument('--native-fusion', choices=('enabled', 'fallback'), required=True)
a = p.parse_args()
assert a.version.isdigit()
output = BASE / ('probe-softmax-'+a.version+'.json')
assert not output.exists()
candidate = BASE / ('candidate-'+a.version+'.py')
sys.path.insert(0, '/tmp/kda-bf16-continuation/megatron_training_entry')
from runtime_compat import prepare_local_runtime
runtime = prepare_local_runtime(NATIVE)
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import attention_mask_func
native = FusedScaleMaskSoftmax(False, True, AttnMaskType.causal,
                              a.native_fusion=='enabled', attention_mask_func, False, None)
spec = importlib.util.spec_from_file_location('attention_candidate', candidate)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
rows = []
timing = []
guards = []

def numeric(actual, reference):
    actual, reference = actual.detach().cpu().double(), reference.detach().cpu().double()
    error = (actual-reference).abs()
    tolerance = .0003+.001*reference.abs()
    passed = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()
                  and (error<=tolerance).all())
    return {'passed': passed, 'max_abs': error.max().item(),
            'bad_elements': int((error>tolerance).sum()), 'elements': actual.numel(),
            'bitwise_equal': bool(torch.equal(actual, reference))}

def check(name, scores, mask, upstream):
    x = scores.detach().clone().requires_grad_(True)
    y = scores.detach().clone().requires_grad_(True)
    original = y.detach().clone()
    original_mask = mask.detach().clone()
    original_grad = upstream.detach().clone()
    expected = native(x+0, mask)
    dx = torch.autograd.grad(expected, x, upstream)[0]
    actual = module.masked_softmax(y, mask)
    dy = torch.autograd.grad(actual, y, upstream)[0]
    assert actual.shape==expected.shape and actual.dtype==scores.dtype
    immutable = bool(torch.equal(y.detach(),original) and torch.equal(mask,original_mask)
                     and torch.equal(upstream,original_grad))
    row = {'case': name, 'shape': list(scores.shape), 'mask_shape': list(mask.shape),
           'scores_stride': list(scores.stride()), 'upstream_stride': list(upstream.stride()),
           'checks': {'output': numeric(actual,expected), 'input_gradient': numeric(dy,dx)},
           'inputs_unchanged': immutable}
    row['passed'] = immutable and all(c['passed'] for c in row['checks'].values())
    rows.append(row)
    print(json.dumps({'case':name,'passed':row['passed']}),flush=True)

with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    torch.manual_seed(31112)
    for size in (17,128,256,512):
        shape = (2,8,size,size)
        causal = torch.triu(torch.ones((2,1,size,size),device='cuda',dtype=torch.bool),1)
        for scale in (.01,1.,5.):
            scores = torch.randn(shape,device='cuda',dtype=torch.bfloat16)*scale
            upstream = torch.randn_like(scores)*.05
            check('causal-S%d-scale%g'%(size,scale),scores,causal,upstream)
        scores = torch.randn(shape,device='cuda',dtype=torch.bfloat16)
        for label,mask in [('fully-masked',torch.ones_like(causal)),
                           ('unmasked',torch.zeros_like(causal)),
                           ('broadcast',causal[:1])]:
            check(label+'-S'+str(size),scores,mask,torch.randn_like(scores)*.05)
        backing = torch.randn((2,8,size,size*2),device='cuda',dtype=torch.bfloat16)*.05
        check('noncontiguous-upstream-S'+str(size),scores,causal,backing[...,::2])
    capture = BASE / 'native-capture-01/attention'
    payload = torch.load(capture/'inputs.pt',map_location='cuda',weights_only=False)
    q,k,v = (payload[n] for n in ('query','key','value'))
    s,b,h,d = q.shape
    scores = torch.baddbmm(torch.empty((b*h,s,s),device='cuda',dtype=q.dtype),
                          q.reshape(s,b*h,d).transpose(0,1),
                          k.reshape(s,b*h,d).permute(1,2,0), beta=0.,alpha=.125).view(b,h,s,s)
    check('actual-native-QK-scores',scores,payload['attention_mask'],
          torch.randn_like(scores)*.05)
    for label,x,mask in [
            ('float32',torch.randn((2,8,17,17),device='cuda'),
             torch.zeros((2,1,17,17),device='cuda',dtype=torch.bool)),
            ('cpu',torch.randn((2,8,17,17),dtype=torch.bfloat16),
             torch.zeros((2,1,17,17),dtype=torch.bool))]:
        rejected = False
        try: module.masked_softmax(x,mask)
        except (ValueError,TypeError,NotImplementedError,AssertionError): rejected=True
        guards.append({'case':label,'passed':rejected})
    passed = all(row['passed'] for row in rows) and all(row['passed'] for row in guards)
    if passed:
        x = scores.detach().clone().requires_grad_(True)
        mask = payload['attention_mask']
        g = torch.randn_like(x)*.05
        for fn in (lambda: native(x+0,mask),lambda:module.masked_softmax(x+0,mask)):
            for _ in range(8): torch.autograd.grad(fn(),x,g)
        torch.cuda.synchronize()
        for mode in ('forward','forward-backward'):
            paired = {'native':[],'candidate':[]}
            for pair in range(12):
                order=('native','candidate') if pair%2==0 else ('candidate','native')
                for label in order:
                    fn = (lambda:native(x+0,mask)) if label=='native' else (lambda:module.masked_softmax(x+0,mask))
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(30):
                        y=fn()
                        if mode=='forward-backward':torch.autograd.grad(y,x,g)
                    end.record();end.synchronize()
                    paired[label].append(start.elapsed_time(end)/30)
            timing.append({'mode':mode,'metric':'torch.cuda.Event ms per call; may include submission gaps',
                           'raw_pairs':paired,'native_median_ms':statistics.median(paired['native']),
                           'candidate_median_ms':statistics.median(paired['candidate'])})
    report={'schema':'native-attention-softmax-independent-probe-v1',
            'status':'passed' if passed else 'rejected', 'version':a.version,
            'candidate_sha256':sha(candidate),'verifier_sha256':sha(Path(__file__)),
            'native_fusion':a.native_fusion,'runtime':runtime,
            'native_source_sha256':{name:sha(NATIVE/name) for name in
               ('megatron/core/fusions/fused_softmax.py','megatron/core/transformer/utils.py')},
            'atol':.0003,'rtol':.001,'comparison':'CPU float64',
            'checks':rows,'guard_checks':guards,'timing':timing,
            'scope':'independent BF16 native mask+softmax probe; common x+0 creates fresh nonleaf scores for both Event arms; not formal training performance'}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':len(rows),'guards':guards,
                     'timing':timing,'output':str(output)}),flush=True)
    if not passed:raise SystemExit(1)

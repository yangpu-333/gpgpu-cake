"""Independent full-attention gate versus pinned default native fused backend."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import torch

BASE=Path('/tmp/kda-attention-stage31')
NATIVE=Path('/tmp/kda-bf16-continuation/megatron-native-5be9626')
p=argparse.ArgumentParser();p.add_argument('--version',required=True);a=p.parse_args()
assert a.version.isdigit()
candidate=BASE/('candidate-'+a.version+'.py')
output=BASE/('probe-full-'+a.version+'.json');assert not output.exists()
sys.path.insert(0,'/tmp/kda-bf16-continuation/megatron_training_entry')
from runtime_compat import prepare_local_runtime
runtime=prepare_local_runtime(NATIVE)
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import attention_mask_func
softmax=FusedScaleMaskSoftmax(False,True,AttnMaskType.causal,True,attention_mask_func,False,None)
spec=importlib.util.spec_from_file_location('attention_candidate',candidate)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
rows=[];guards=[];timing=[]
proof=json.loads((BASE/'backend-default.json').read_text())
assert proof['status']=='passed' and proof['requested_backend']=='default-fusion'
assert {r['sequence'] for r in proof['cases']}=={128,256,512}
for row in proof['cases']:
    assert row['route']=='cuda-extension' and row['kernel_available'] is True
    assert row['cuda_extension_forward_completed'] and row['cuda_extension_backward_completed']
    for name in ('scaled_masked_softmax_cuda','scaled_upper_triang_masked_softmax_cuda'):
        extension=importlib.import_module(name)
        assert sha(Path(extension.__file__))==row['extensions_after'][name]['sha256']
for name,expected in proof['native_source_sha256'].items():
    assert sha(NATIVE/name)==expected

def native(q,k,v,scale):
    s,b,h,d=q.shape
    scores=torch.baddbmm(torch.empty((b*h,s,s),device=q.device,dtype=q.dtype),
                       q.reshape(s,b*h,d).transpose(0,1),
                       k.view(s,b*h,d).permute(1,2,0),beta=0.,alpha=scale)
    probabilities=softmax(scores.view(b,h,s,s),None).view(b*h,s,s)
    context=torch.bmm(probabilities,v.view(s,b*h,d).transpose(0,1))
    return context.view(b,h,s,d).permute(2,0,1,3).contiguous().view(s,b,h*d)

def numeric(actual,expected):
    assert actual.shape==expected.shape and actual.dtype==expected.dtype
    assert actual.device==expected.device
    actual,expected=actual.detach().cpu().double(),expected.detach().cpu().double()
    error=(actual-expected).abs();tolerance=.0003+.001*expected.abs()
    return {'passed':bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()
                          and (error<=tolerance).all()),'max_abs':error.max().item(),
            'bad_elements':int((error>tolerance).sum()),'elements':actual.numel(),
            'exact_equal':bool(torch.equal(actual,expected))}

def leaves(values,size,offset=0):
    storage=torch.empty((size*2*8*192+offset,),device='cuda',dtype=torch.bfloat16)
    shared=storage[offset:].view(size,2,8,192)
    for index,value in enumerate(values):shared[...,index*64:(index+1)*64].copy_(value)
    return tuple(shared[...,i*64:(i+1)*64].detach().requires_grad_(True) for i in range(3))

def check(name,qkv,g,group='positive'):
    s=qkv[0].shape[0]
    baseline=leaves(qkv,s)
    actual_inputs=leaves(qkv,s,offset=1 if group=='offset' else 0)
    before=[x.detach().clone() for x in actual_inputs];before_grad=g.detach().clone()
    y=native(*baseline,.125);expected_grads=torch.autograd.grad(y,baseline,g)
    z=module.attention(*actual_inputs,.125)
    assert z.shape==y.shape and z.dtype==torch.bfloat16 and z.is_contiguous()
    assert z.device==actual_inputs[0].device and z.is_cuda and z.requires_grad
    actual_grads=torch.autograd.grad(z,actual_inputs,g)
    checks={'output':numeric(z,y)}
    for key,actual,expected in zip(('query_grad','key_grad','value_grad'),actual_grads,expected_grads):
        checks[key]=numeric(actual,expected)
    immutable=all(torch.equal(x.detach(),copy) for x,copy in zip(actual_inputs,before)) and torch.equal(g,before_grad)
    row={'case':name,'group':group,'shape':list(actual_inputs[0].shape),
         'input_strides':[list(x.stride()) for x in actual_inputs],
         'storage_offsets':[x.storage_offset() for x in actual_inputs],
         'input_storage_shared':len({x.untyped_storage().data_ptr() for x in actual_inputs})==1,
         'current_stream':torch.cuda.current_stream(actual_inputs[0].device).cuda_stream,
         'upstream_stride':list(g.stride()),'checks':checks,'inputs_unchanged':bool(immutable)}
    row['passed']=immutable and all(c['passed'] for c in checks.values())
    rows.append(row);print(json.dumps({'case':name,'passed':row['passed']}),flush=True)

with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    torch.manual_seed(31112)
    capture=BASE/'native-default-capture-02/attention'
    payload=torch.load(capture/'inputs.pt',map_location='cuda',weights_only=False)
    actual_g=torch.load(capture/'upstream-grad.pt',map_location='cuda',weights_only=False)
    actual_qkv=[payload[n] for n in ('query','key','value')]
    qkv=leaves(actual_qkv,128)
    result=native(*qkv,.125);dg=torch.autograd.grad(result,qkv,actual_g)
    replay={'output':numeric(result,torch.load(capture/'output.pt',map_location='cuda',weights_only=False))}
    for name,grad in zip(('query','key','value'),dg):
        replay[name+'_grad']=numeric(grad,torch.load(capture/(name+'-grad.pt'),map_location='cuda',weights_only=False))
    assert all(c['passed'] and c['exact_equal'] for c in replay.values()),'native reference replay must equal captured formal model'
    check('actual-formal-input-and-gradient',actual_qkv,actual_g)
    for size in (128,256,512):
        for scale in (.01,.25,1.):
            qkv=[torch.randn((size,2,8,64),device='cuda',dtype=torch.bfloat16)*scale for _ in range(3)]
            g=torch.randn((size,2,512),device='cuda',dtype=torch.bfloat16)*.05
            check('S%d-random%g'%(size,scale),qkv,g)
        qkv=[torch.randn((size,2,8,64),device='cuda',dtype=torch.bfloat16)*.25 for _ in range(3)]
        backing=torch.randn((size,2,1024),device='cuda',dtype=torch.bfloat16)*.05
        check('S%d-noncontiguous-upstream'%size,qkv,backing[...,::2])
        check('S%d-offset-shared-views'%size,qkv,backing[...,::2],group='offset')
    for index in range(4):
        qkv=[torch.randn((128,2,8,64),device='cuda',dtype=torch.bfloat16)*.25 for _ in range(3)]
        check('fresh-input-'+str(index),qkv,torch.randn((128,2,512),device='cuda',dtype=torch.bfloat16)*.05)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        qkv=[torch.randn((128,2,8,64),device='cuda',dtype=torch.bfloat16)*.25 for _ in range(3)]
        check('nondefault-stream',qkv,torch.randn((128,2,512),device='cuda',dtype=torch.bfloat16)*.05)
    torch.cuda.current_stream().wait_stream(stream)
    for label,device,dtype in [('cpu','cpu',torch.bfloat16),('float32','cuda',torch.float32)]:
        qkv=[torch.randn((128,2,8,64),device=device,dtype=dtype) for _ in range(3)]
        rejected=False;rejection_type=None
        try:module.attention(*qkv,.125)
        except (TypeError,ValueError,NotImplementedError,AssertionError,RuntimeError) as error:
            rejected=True;rejection_type=type(error).__name__
        guards.append({'case':label,'passed':rejected,'error_type':rejection_type})
    # Tail may be supported with its same native automatic fallback or rejected explicitly.
    size=17;qkv=[torch.randn((size,2,8,64),device='cuda',dtype=torch.bfloat16)*.25 for _ in range(3)]
    rejected_tail=False
    try:module.attention(*leaves(qkv,size),.125)
    except (TypeError,ValueError,NotImplementedError,AssertionError) as error:
        rejected_tail=True
        guards.append({'case':'S17-outside-fixed-primary','passed':True,'explicitly_rejected':type(error).__name__})
    if not rejected_tail:
        check('S17-native-automatic-fallback',qkv,torch.randn((size,2,512),device='cuda',dtype=torch.bfloat16)*.05)
    passed=all(r['passed'] for r in rows) and all(r['passed'] for r in guards)
    if passed:
        qkv=leaves(actual_qkv,128);g=actual_g
        funcs={'native':lambda:native(*qkv,.125),'candidate':lambda:module.attention(*qkv,.125)}
        for fn in funcs.values():
            for _ in range(8):torch.autograd.grad(fn(),qkv,g)
        torch.cuda.synchronize()
        for mode in ('forward','forward-backward'):
            values={'native':[],'candidate':[]}
            for pair in range(12):
                for arm in (('native','candidate') if pair%2==0 else ('candidate','native')):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(20):
                        y=funcs[arm]()
                        if mode=='forward-backward':torch.autograd.grad(y,qkv,g)
                    end.record();end.synchronize();values[arm].append(start.elapsed_time(end)/20)
            timing.append({'mode':mode,'metric':'torch.cuda.Event full API milliseconds; submission gaps may be included',
                           'raw_pairs':values,'native_median_ms':statistics.median(values['native']),
                           'candidate_median_ms':statistics.median(values['candidate']),
                           'baseline_scope':'bitwise-validated native functional replay, fresh torch.empty scratch; original module/global-buffer/RNG wrapper cost is excluded; final gate uses actual original formal training'})
    report={'schema':'native-fused-full-attention-independent-probe-v1',
            'status':'passed' if passed else 'rejected','version':a.version,
            'candidate_sha256':sha(candidate),'verifier_sha256':sha(Path(__file__)),
            'native_source_sha256':{name:sha(NATIVE/name) for name in
                ('megatron/core/transformer/dot_product_attention.py','megatron/core/fusions/fused_softmax.py','megatron/core/transformer/utils.py')},
            'backend_proof_sha256':sha(BASE/'backend-default.json'),'runtime':runtime,
            'atol':.0003,'rtol':.001,'comparison':'CPU float64',
            'native_capture_replay':replay,'checks':rows,'guard_checks':guards,'timing':timing,
            'scope':'Full attention API with default native CUDA fused softmax; actual QKV layouts/offsets and all QKV gradients; no training or production claim'}
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'cases':len(rows),'timing':timing,'output':str(output)}),flush=True)
    if not passed:raise SystemExit(1)

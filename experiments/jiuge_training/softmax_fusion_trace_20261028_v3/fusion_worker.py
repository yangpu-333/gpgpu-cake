"""Correctness first, then unprofiled ABBA/BAAB timing and separate traces."""
import argparse,gzip,hashlib,importlib.metadata,json,math,os,signal,statistics,sys,time,traceback
from pathlib import Path
def trace_kinds(kernels,dtype):
 assert len(kernels)==4,'Exactly two protected allocation fills and two fused math launches required'
 names=[k['name']for k in kernels]
 expected='c10::BFloat16'if dtype=='torch.bfloat16'else'float'
 assert 'FillFunctor<float>'in names[0] and names[1]=='_forward'
 assert 'FillFunctor<'+expected+'>'in names[2] and names[3]=='_backward'
 return {'fused_math_launches':2,'protected_fill_launches':2,'observed_total_launches':4}

P=argparse.ArgumentParser();P.add_argument('--phase',choices=('correctness','performance','trace'),required=True);P.add_argument('--device',type=int,required=True);P.add_argument('--seed',type=int,default=63100);P.add_argument('--output',type=Path,required=True);P.add_argument('--contract',type=Path,required=True);P.add_argument('--admission',type=Path)
a=P.parse_args();a.output.mkdir(exist_ok=False)
r={'phase':a.phase,'device':a.device,'seed':a.seed,'scope':'isolated scale/mask/softmax operator; synthetic-input real7B shapes','full_model_training_accepted':False,'training_throughput_claim':False,'cases':[],'boundary_cases':[],'negative_controls':[]}
try:
 contract=json.loads(a.contract.read_text());deadline=contract['absolute_deadline_epoch'];assert time.time()<deadline-30
 signal.alarm(max(1,min(1200,int(deadline-time.time())-20)))
 ROOT=Path('/share/jiuge7b-corex-validation-20261025-01');sys.path.append(str(ROOT))
 for name,want in contract['source_SHA256'].items():assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==want,name
 from corex_process_guard_data import enforce_rank_limit,check_live
 import corex_process_guard_data as guard
 enforce_rank_limit();assert os.getuid()==1001 and a.device in (0,1)
 import numpy as np,torch,triton
 import fused_scale_mask_softmax as fused
 from full_reference import full_check,Metrics
 torch.set_num_threads(8);torch.cuda.set_device(a.device);torch.use_deterministic_algorithms(True,warn_only=False)
 r['strict_allocation_protection']={'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'fill_uninitialized_memory':torch.utils.deterministic.fill_uninitialized_memory};assert all(r['strict_allocation_protection'].values())
 r['runtime']={'torch':torch.__version__,'triton':triton.__version__,'GPU':torch.cuda.get_device_name(a.device),'triton_import':triton.__file__,'device_properties':str(torch.cuda.get_device_properties(a.device))}
 r['source_SHA256']=contract['source_SHA256']
 def emit(row):
  with (a.output/'journal.jsonl').open('a')as f:f.write(json.dumps(row)+'\n');f.flush()
  print(json.dumps(row),flush=True)
 def mask_for(x,mode):
  S=x.shape[-1]
  if mode=='causal':return torch.ones((S,S),device=x.device,dtype=torch.bool).triu(1),None
  if mode=='all-masked':m=torch.ones((1,1,S,S),device=x.device,dtype=torch.bool)
  elif mode=='all-open':m=torch.zeros((1,1,S,S),device=x.device,dtype=torch.bool)
  elif mode=='strided-mask':m=(torch.rand((1,1,S,S*2),device=x.device)<.35)[...,::2]
  else:m=torch.rand((x.shape[0],1,S,S),device=x.device)<.35
  return m,m
 def case(shape,dtype,seed,mode='causal',scale=1/math.sqrt(128),pattern='random',strided_grad=False):
  torch.manual_seed(seed);guard.check_live(128*1024**2)
  x=torch.randn(shape,device='cuda',dtype=dtype,requires_grad=True)
  if pattern=='large-negative':
   with torch.no_grad():x.fill_(-20000.)
  if pattern=='zeros':
   with torch.no_grad():x.zero_()
  g=torch.randn((*shape[:-1],shape[-1]*2),device='cuda',dtype=torch.float32)[...,::2]if strided_grad else torch.randn(shape,device='cuda',dtype=torch.float32)
  m,cm=mask_for(x,mode)
  ny=fused.native_scale_mask_softmax(x,scale,m);nd,=torch.autograd.grad(ny,x,g)
  cy=fused.fused_scale_mask_softmax(x,scale,cm);cd,=torch.autograd.grad(cy,x,g)
  torch.cuda.synchronize();proof=full_check(x,g,m,ny,nd,cy,cd,scale,guard)
  row={'shape':list(shape),'dtype':str(dtype),'seed':seed,'mask_mode':mode,'scale':scale,'pattern':pattern,'noncontiguous_upstream':strided_grad,**proof}
  del x,g,m,cm,ny,nd,cy,cd
  emit({'kind':'correctness','shape':shape,'dtype':str(dtype),'seed':seed,'mask':mode,'passed':proof['passed']})
  return row
 if a.phase=='correctness':
  for seed in contract['seeds']:
   for S in contract['target_lengths']:
    for dt in (torch.bfloat16,torch.float32):
     row=case((1,14,S,S),dt,seed);r['cases'].append(row)
     assert row['passed'],row
  for dt in (torch.bfloat16,torch.float32):
   for S in (1,2,3,31,127,255,257,2047,2049,4095):
    row=case((2,3,S,S)if S<=257 else (1,1,S,S),dt,63103);r['boundary_cases'].append(row);assert row['passed'],row
   for mode in ('all-open','all-masked','random-mask','strided-mask'):
    row=case((2,3,31,31),dt,63104,mode);r['boundary_cases'].append(row);assert row['passed'],row
   for kw in ({'pattern':'large-negative','scale':1.0},{'pattern':'zeros'},{'scale':0.0},{'scale':-.25},{'strided_grad':True}):
    row=case((2,3,31,31),dt,63105,**kw);r['boundary_cases'].append(row);assert row['passed'],row
  # API-contract and deliberately corrupted-output controls reject failures.
  x=torch.randn((1,2,31,31),device='cuda',requires_grad=True)
  bad=[('CPU-input',lambda:fused.fused_scale_mask_softmax(torch.zeros((1,1,3,3)),1.)),('FP16-input',lambda:fused.fused_scale_mask_softmax(x.half(),1.)),('FP64-input',lambda:fused.fused_scale_mask_softmax(x.double(),1.)),('noncontiguous-input',lambda:fused.fused_scale_mask_softmax(x.transpose(-1,-2),1.)),('nonsquare',lambda:fused.fused_scale_mask_softmax(x[...,:30],1.)),('empty',lambda:fused.fused_scale_mask_softmax(x[:,:,:0,:0],1.)),('NaN-scale',lambda:fused.fused_scale_mask_softmax(x,float('nan'))),('infinite-scale',lambda:fused.fused_scale_mask_softmax(x,float('inf'))),('nonboolean-mask',lambda:fused.fused_scale_mask_softmax(x,1.,torch.zeros_like(x))),('CPU-mask',lambda:fused.fused_scale_mask_softmax(x,1.,torch.zeros((31,31),dtype=torch.bool))),('invalid-broadcast',lambda:fused.fused_scale_mask_softmax(x,1.,torch.ones((3,4),dtype=torch.bool,device='cuda')))]
  for name,fn in bad:
   rejected=False
   try:fn()
   except (ValueError,RuntimeError):rejected=True
   r['negative_controls'].append({'name':name,'rejected':rejected});assert rejected,name
  torch.manual_seed(63106);m=torch.ones((31,31),device='cuda',dtype=torch.bool).triu(1);g=torch.randn_like(x)
  y=fused.native_scale_mask_softmax(x,1.,m);d,=torch.autograd.grad(y,x,g)
  ya=y.detach().cpu().numpy();da=d.detach().cpu().numpy()
  for name,corrupt,ref in [('zero-forward',np.zeros_like(ya),ya),('zero-gradient',np.zeros_like(da),da),('reversed-gradient',-da,da),('corrupt-one-element',ya.copy(),ya)]:
   if name=='corrupt-one-element':corrupt.flat[0]+=1
   mm=Metrics();mm.add(corrupt,ref);rejected=not mm.result(1e-5)['passed'];r['negative_controls'].append({'name':name,'rejected':rejected});assert rejected,name
  r['passed']=len(r['cases'])==12 and len(r['boundary_cases'])==38 and len(r['negative_controls'])==15
 else:
  assert a.admission
  admission=json.loads(a.admission.read_text());assert admission['passed'] and admission['source_SHA256']==contract['source_SHA256'] and admission['target_full_tensor_cases']==24 and admission['boundary_cases']==76
  r['correctness_admission_SHA256']=hashlib.sha256(a.admission.read_bytes()).hexdigest()
  for S in contract['target_lengths']:
   for dt in (torch.bfloat16,torch.float32):
    torch.manual_seed(a.seed);guard.check_live(128*1024**2)
    x=torch.randn((1,14,S,S),device='cuda',dtype=dt,requires_grad=True);g=torch.randn(x.shape,device='cuda',dtype=torch.float32);m=torch.ones((S,S),device='cuda',dtype=torch.bool).triu(1)
    def native():
     y=fused.native_scale_mask_softmax(x,contract['scale'],m);dx,=torch.autograd.grad(y,x,g);return y,dx
    def candidate():
     y=fused.fused_scale_mask_softmax(x,contract['scale']);dx,=torch.autograd.grad(y,x,g);return y,dx
    if a.phase=='performance':
     for _ in range(contract['warmup_pairs']):
      z,d=native();del z,d;z,d=candidate();del z,d
     torch.cuda.synchronize();rounds=[]
     for i in range(contract['paired_rounds']):
      order=('native','candidate','candidate','native')if i%2==0 else ('candidate','native','native','candidate')
      timing={'native':[],'candidate':[]}
      for name in order:
       start=time.perf_counter();z,d=(native if name=='native'else candidate)();torch.cuda.synchronize();timing[name].append(time.perf_counter()-start);del z,d
      rounds.append({'round':i,'order':order,'native_seconds':timing['native'],'candidate_seconds':timing['candidate'],'throughput_ratio':statistics.mean(timing['native'])/statistics.mean(timing['candidate'])})
     ns=[t for p in rounds for t in p['native_seconds']];cs=[t for p in rounds for t in p['candidate_seconds']]
     ratios=[p['throughput_ratio']for p in rounds]
     row={'shape':[1,14,S,S],'dtype':str(dt),'seed':a.seed,'rounds':rounds,'native_median_seconds':statistics.median(ns),'candidate_median_seconds':statistics.median(cs),'geomean_paired_ratio':math.exp(statistics.mean(math.log(v)for v in ratios)),'every_ABBA_round_faster':all(v>1 for v in ratios),'native_CV':statistics.pstdev(ns)/statistics.mean(ns),'candidate_CV':statistics.pstdev(cs)/statistics.mean(cs),'timing_scope':'dispatch + allocations + forward/backward + GPU synchronize; precomputed inputs/mask; no profiler or audit in measured calls','max_GPU_reserved_bytes':torch.cuda.max_memory_reserved()}
     r['cases'].append(row);emit({'kind':'paired-performance','device':a.device,'seed':a.seed,'S':S,'dtype':str(dt),'ratio':row['geomean_paired_ratio']})
    else:
     row={'shape':[1,14,S,S],'dtype':str(dt),'seed':a.seed};r['cases'].append(row)
    if a.phase=='trace':
     # Profile after all measurements for this case. No profile timings enter ratios.
     from torch.profiler import profile,ProfilerActivity,record_function
     row['traces']=[]
     for name,fn in (('native',native),('candidate',candidate)):
      with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,profile_memory=False,with_stack=False)as p:
       with record_function(f'scale-mask-softmax-{name}-S{S}-{dt}'):z,d=fn();torch.cuda.synchronize();del z,d
      raw=a.output/f'trace-{name}-S{S}-{str(dt).split(".")[-1]}.json';p.export_chrome_trace(str(raw));buf=raw.read_bytes();events=json.loads(buf)['traceEvents'];kernels=[e for e in events if e.get('cat')=='kernel'and e.get('ph')=='X']
      assert kernels
      if name=='candidate':trace_kinds(kernels,str(dt))
      gz=raw.with_suffix('.json.gz')
      with gzip.open(gz,'wb')as f:f.write(buf)
      group={}
      for e in kernels:group[e['name']]=group.get(e['name'],0)+float(e['dur'])
      row['traces'].append({'arm':name,'fused_math_launches':2 if name=='candidate'else None,'protected_fill_launches':sum('FillFunctor<'in e['name']for e in kernels),'kernel_count':len(kernels),'kernel_duration_sum_us':sum(e['dur']for e in kernels),'GPU_kernels_us':sorted(group.items(),key=lambda v:v[1],reverse=True),'raw_path':str(raw),'gzip_path':str(gz),'raw_SHA256':hashlib.sha256(buf).hexdigest(),'gzip_SHA256':hashlib.sha256(gz.read_bytes()).hexdigest()})
    del x,g,m
  r['passed']=len(r['cases'])==4
 r['compiled_kernels']={}
 for key,k in fused.COMPILED.items():
  asm=getattr(k,'asm',{});r['compiled_kernels'][key]={'assembly_SHA256':{name:hashlib.sha256(v if isinstance(v,bytes)else v.encode()).hexdigest()for name,v in asm.items()if isinstance(v,(str,bytes))},'n_regs':getattr(k,'n_regs',None),'n_spills':getattr(k,'n_spills',None)}
except BaseException as error:r.update(passed=False,error=repr(error),traceback=traceback.format_exc());raise
finally:
 r['finish_epoch']=time.time();(a.output/'result.json').write_text(json.dumps(r,indent=2));print(json.dumps({'phase':a.phase,'device':a.device,'passed':r.get('passed',False),'closed_target_cases':len(r['cases']),'error':r.get('error')}),flush=True)

"""Frozen operator-only contract, shared pair lock and PID/startticks guards."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,re,signal,statistics,subprocess,sys,time,traceback
ROOT=Path('/share/jiuge7b-corex-validation-20261025-01');C=Path(__file__).parent;contract=json.loads((C/'contract.json').read_text());D=contract['absolute_deadline_epoch'];OUT=C/'run-01';OUT.mkdir(exist_ok=False);os.chown(OUT,1001,1001)
PAIR=['GPU-a378f8c5-ada0-5b96-954c-ff19be59f3c0','GPU-0d6156e3-6ce8-5382-8ff3-f37ad849e80a']
env=dict(os.environ,LD_LIBRARY_PATH='/usr/local/iluvatar/lib64:/usr/local/corex/lib64',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=str(C/'triton-cache'))
(C/'triton-cache').mkdir();os.chown(C/'triton-cache',1001,1001)
r={'scope':'isolated operator correctness and paired performance only','start_epoch':time.time(),'deadline_epoch':D,'full_model_training_accepted':False,'workers':[],'stages':[],'passed':False}
def mem():return {str(n):next(int(x.split()[3])*1024 for x in Path(f'/sys/devices/system/node/node{n}/meminfo').read_text().splitlines()if ' MemFree:'in x)for n in (0,1)}
def identity(pid):
 try:
  raw=Path(f'/proc/{pid}/stat').read_text();a=raw[raw.rfind(')')+2:].split();return {'pid':pid,'ppid':int(a[1]),'startticks':int(a[19])}
 except(FileNotFoundError,ProcessLookupError):return None
def gpu_idle():
 p=subprocess.run(['/usr/local/corex/bin/ixsmi','-i',','.join(PAIR)],env=env,capture_output=True,text=True,timeout=20);q=subprocess.run(['/usr/local/corex/bin/ixsmi','-L'],env=env,capture_output=True,text=True,timeout=10)
 assert p.returncode==q.returncode==0
 assert all(f'GPU  {i}: Iluvatar BI-V150 (UUID: {u})'in q.stdout for i,u in enumerate(PAIR))
 assert '96:00.0'in p.stdout and '99:00.0'in p.stdout and 'No running processes found'in p.stdout
 vals=re.findall(r'(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%',p.stdout);assert len(vals)==2 and all(int(m)<=128 and int(t)==32768 and int(u)==0 for m,t,u in vals)
 return {'epoch':time.time(),'GPU':p.stdout,'UUID':q.stdout,'node_MemFree_bytes':mem()}
def save():
 tmp=OUT/'status.tmp';tmp.write_text(json.dumps(r,indent=2));tmp.replace(OUT/'status.json')
child=None;expected=None
lock=(ROOT/'two-GPU-validation.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def on_signal(n,frame):raise SystemExit('operator supervisor signal '+str(n))
signal.signal(signal.SIGTERM,on_signal);signal.signal(signal.SIGINT,on_signal)
try:
 assert time.time()<D-180
 for n,h in contract['source_SHA256'].items():assert hashlib.sha256((C/n).read_bytes()).hexdigest()==h,n
 r['before']=gpu_idle();m=mem();assert m['0']>=252689747968 and m['1']>=16*1024**3,'Original16GiB physical NUMA reserve rejected before trace-only attempt'
 topology=subprocess.run(['lscpu','-p=CPU,NODE,CORE,SOCKET'],capture_output=True,text=True,check=True).stdout;cores={}
 for line in topology.splitlines():
  if not line or line.startswith('#'):continue
  cpu,node,core,socket=map(int,line.split(','))
  if node==0 and cpu in os.sched_getaffinity(0):cores.setdefault((socket,core),cpu)
 candidates=sorted(cores.values());affinity=candidates[:8];assert len(affinity)==8
 if len(candidates)>8:os.sched_setaffinity(0,{candidates[8]})
 r['worker_CPU_affinity']=affinity;save()
 def run(phase,device,seed=63100,round_name='primary'):
  global child,expected
  assert time.time()<D-60
  name=f'{phase}-gpu{device}-seed{seed}-{round_name}';folder=OUT/name
  command=[str(ROOT/'.venv/bin/python'),str(C/'fusion_worker.py'),'--phase',phase,'--device',str(device),'--seed',str(seed),'--output',str(folder),'--contract',str(C/'contract.json')]
  if phase!='correctness':command+=['--admission',str(OUT/'operator-correctness-admission.json')]
  r['current_stage']=name;save()
  out=(OUT/(name+'.stdout.log')).open('x');err=(OUT/(name+'.stderr.log')).open('x')
  def bind():os.sched_setaffinity(0,affinity)
  child=subprocess.Popen(command,env=env,stdout=out,stderr=err,user=1001,group=1001,extra_groups=[],start_new_session=True,preexec_fn=bind);expected=identity(child.pid);assert expected
  r['workers'].append(expected);peak=0;tracked={child.pid:expected['startticks']}
  while child.poll()is None:
   current=identity(child.pid);assert current and current['startticks']==expected['startticks']
   for p in Path('/proc').iterdir():
    if p.name.isdigit():
     z=identity(int(p.name))
     if z and z['ppid']in tracked:tracked[z['pid']]=z['startticks']
   rss=0
   for pid,tick in tracked.items():
    z=identity(pid)
    if z and z['startticks']==tick:
     try:rss+=next(int(v.split()[1])*1024 for v in Path(f'/proc/{pid}/status').read_text().splitlines()if v.startswith('VmRSS:'))
     except(FileNotFoundError,StopIteration):pass
   peak=max(peak,rss);m=mem();host=next(int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines()if v.startswith('MemAvailable:'))
   assert rss<8*1024**3 and host>=8*1024**3 and min(m.values())>=16*1024**3 and time.time()<D,'Original live reserves or tighter8GiB own-RSS rejected'
   time.sleep(.5)
  out.close();err.close();exit_code=child.returncode;child=None;expected=None
  result=json.loads((folder/'result.json').read_text())if(folder/'result.json').exists()else{}
  stage={'name':name,'phase':phase,'device':device,'seed':seed,'round':round_name,'exit_code':exit_code,'result_path':str(folder/'result.json'),'result_SHA256':hashlib.sha256((folder/'result.json').read_bytes()).hexdigest()if result else None,'own_RSS_peak_bytes':peak,'passed':result.get('passed',False)}
  r['stages'].append(stage);r['current_stage']=None;save()
  assert exit_code==0 and result.get('passed'),(name,exit_code,result.get('error'),(OUT/(name+'.stderr.log')).read_text()[-3000:])
  for attempt in range(15):
   try:r.setdefault('post_stage_gpu',[]).append(gpu_idle());break
   except AssertionError:
    if attempt==14:raise
    time.sleep(1)
  return result
 # Replay only missing attribution. All earlier GPU results remain unchanged.
 parent=ROOT/'softmax-fusion-20261028-02';f=json.loads((parent/'run-01/final.json').read_text());pc=json.loads((parent/'contract.json').read_text())
 assert not f['passed'] and 'Two-launch candidate trace contract'in f['error']
 assert len(f['stages'])==9 and all(x['exit_code']==0 and x['passed']for x in f['stages'][:8])
 assert [x['phase']for x in f['stages'][:8]]==['correctness']*2+['performance']*6
 assert f['stages'][8]['phase']=='trace'and f['stages'][8]['exit_code']==1 and not f['confirmation_required']
 for x in f['workers']:assert identity(x['pid'])is None or identity(x['pid'])['startticks']!=x['startticks']
 for n in('fused_scale_mask_softmax.py','full_reference.py','corex_process_guard_data.py'):
  assert contract['source_SHA256'][n]==pc['source_SHA256'][n]==hashlib.sha256((parent/n).read_bytes()).hexdigest()
 a=json.loads((parent/'run-01/operator-correctness-admission.json').read_text());assert a['passed'] and a['target_full_tensor_cases']==24 and a['boundary_cases']==76 and a['negative_controls_rejected']==30
 a['parent_admission_SHA256']=hashlib.sha256((parent/'run-01/operator-correctness-admission.json').read_bytes()).hexdigest();a['source_SHA256']=contract['source_SHA256'];a['reused_only_byte_identical_math_correctness']=True
 (OUT/'operator-correctness-admission.json').write_text(json.dumps(a,indent=2));r['parent_final_SHA256']=hashlib.sha256((parent/'run-01/final.json').read_bytes()).hexdigest();r['trace_accounting_repair_only']=True
 for device in(0,1):run('trace',device,contract['seeds'][-1])
 r['passed']=True
except BaseException as e:r.update(passed=False,error=repr(e),traceback=traceback.format_exc())
finally:
 if child is not None and child.poll()is None:
  current=identity(child.pid)
  if current and expected and current['startticks']==expected['startticks']:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:
    current=identity(child.pid)
    if current and current['startticks']==expected['startticks']:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
 r['finish_epoch']=time.time();r['source_SHA256_end']={n:hashlib.sha256((C/n).read_bytes()).hexdigest()for n in contract['source_SHA256']};save();(OUT/'final.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items()if k not in('post_stage_gpu','before')}),flush=True)
 raise SystemExit(0 if r['passed']else 1)

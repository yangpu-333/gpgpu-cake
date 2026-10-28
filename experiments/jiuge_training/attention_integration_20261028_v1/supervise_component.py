from pathlib import Path
import fcntl,hashlib,json,os,re,signal,subprocess,time,traceback
C=Path(__file__).parent;R=Path('/share/jiuge7b-corex-validation-20261025-01');p=json.loads((C/'contract.json').read_text());deadline=p['deadline_epoch'];r={'scope':'actual MCore component with TP2, no full model weights','passed':False,'start_epoch':time.time(),'deadline_epoch':deadline,'workers':[]}
env=dict(os.environ,JIUGE_DT_ROOT=str(R),JIUGE_DT_CONTROL=str(C),LD_LIBRARY_PATH='/usr/local/iluvatar/lib64:/usr/local/corex/lib64',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='0,1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1',TRITON_CACHE_DIR=str(C/'triton-cache'))
pair='GPU-a378f8c5-ada0-5b96-954c-ff19be59f3c0,GPU-0d6156e3-6ce8-5382-8ff3-f37ad849e80a';lock=(R/'two-GPU-validation.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def gpu():
 q=subprocess.run(['/usr/local/corex/bin/ixsmi','-i',pair],env=env,capture_output=True,text=True,timeout=20);ids=subprocess.check_output(['/usr/local/corex/bin/ixsmi','-L'],env=env,text=True,timeout=10);assert q.returncode==0 and 'No running processes found'in q.stdout and '96:00.0'in q.stdout and '99:00.0'in q.stdout
 for i,u in enumerate(pair.split(',')):assert f'GPU  {i}: Iluvatar BI-V150 (UUID: {u})'in ids
 vals=re.findall(r'(\d+)MiB\s*/\s*(\d+)MiB\s*\|\s*(\d+)%',q.stdout);assert len(vals)==2 and all(int(m)<=128 and int(t)==32768 and int(u)==0 for m,t,u in vals)
 return q.stdout
def identity(pid):
 try:
  v=Path('/proc',str(pid),'stat').read_text().rsplit(') ',1)[1].split();return {'pid':pid,'PPID':int(v[1]),'startticks':int(v[19]),'state':v[0],'RSS_bytes':int(Path('/proc',str(pid),'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')}
 except(OSError,IndexError):return None
def memory():return {str(n):next(int(x.split()[3])*1024 for x in Path(f'/sys/devices/system/node/node{n}/meminfo').read_text().splitlines()if ' MemFree:'in x)for n in(0,1)}
def save():
 q=C/'status.tmp';q.write_text(json.dumps(r,indent=2));q.replace(C/'status.json')
child=None;tracked={}
def stop(n,f):raise TimeoutError('Own component supervisor received stop')
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
try:
 for n,h in p['source_SHA256'].items():assert hashlib.sha256((C/n).read_bytes()).hexdigest()==h
 assert time.time()<deadline-180;r['GPU_before']=gpu();r['NUMA_before']=memory();assert min(r['NUMA_before'].values())>=16*1024**3
 rows=subprocess.check_output(['lscpu','-p=CPU,NODE,CORE,SOCKET'],text=True);cpus={}
 for line in rows.splitlines():
  if line.startswith('#')or not line:continue
  cpu,node,core,socket=map(int,line.split(','))
  if node==0 and cpu in os.sched_getaffinity(0):cpus.setdefault((socket,core),cpu)
 selected=sorted(cpus.values())[:16];assert len(selected)==16;env['JIUGE_PROBE_NODE0_CPUS']=','.join(map(str,selected));(C/'triton-cache').mkdir();os.chown(C/'triton-cache',1001,1001);(C/'actual-attention-probe').mkdir();os.chown(C/'actual-attention-probe',1001,1001)
 with(C/'probe.stdout.log').open('xb')as out,(C/'probe.stderr.log').open('xb')as err:
  child=subprocess.Popen([str(R/'.venv/bin/python'),'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(C/'actual_attention_probe.py')],env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,user=1001,group=1001,extra_groups=[]);r['parent']=identity(child.pid);tracked[child.pid]=r['parent']['startticks'];r['peak_own_RSS_bytes']=0;last=0
  while child.poll()is None:
   allrows=[identity(int(x.name))for x in Path('/proc').iterdir()if x.name.isdigit()];allrows=[x for x in allrows if x]
   for _ in range(4):
    for v in allrows:
     if v['PPID']in tracked:tracked[v['pid']]=v['startticks']
   own=[v for v in allrows if tracked.get(v['pid'])==v['startticks']and v['state']!='Z'];rss=sum(v['RSS_bytes']for v in own);r['peak_own_RSS_bytes']=max(r['peak_own_RSS_bytes'],rss);m=memory();free=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))
   assert rss<8*1024**3 and free>=8*1024**3 and min(m.values())>=16*1024**3 and time.time()<min(deadline,r['start_epoch']+1800),'Own component memory/time reserve rejected'
   if time.time()-last>=15:r.update(current_NUMA=m,workers=own);save();last=time.time()
   time.sleep(.5)
 r['exit_code']=child.returncode;assert child.returncode==0,'Actual MCore component worker failed'
 r['results']={str(rank):json.loads((C/'actual-attention-probe'/f'rank-{rank}'/'result.json').read_text())for rank in(0,1)};assert all(v['passed']and len(v['cases'])==12 for v in r['results'].values());r['passed']=True
except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc())
finally:
 if child is not None and child.poll()is None:
  for sig in(signal.SIGTERM,signal.SIGKILL):
   for pid,t in sorted(tracked.items(),reverse=True):
    v=identity(pid)
    if v and v['startticks']==t and v['state']!='Z':
     try:os.kill(pid,sig)
     except ProcessLookupError:pass
   time.sleep(2)
  child.wait(timeout=20)
 for attempt in range(15):
  try:r['GPU_after']=gpu();break
  except AssertionError:
   if attempt==14:r['GPU_after_error']='Selected pair did not become idle'
   time.sleep(1)
 r['finish_epoch']=time.time();r['source_SHA256_end']={n:hashlib.sha256((C/n).read_bytes()).hexdigest()for n in p['source_SHA256']};save();(C/'final.json').write_text(json.dumps(r,indent=2));print(json.dumps({'passed':r['passed'],'error':r.get('error')}),flush=True)

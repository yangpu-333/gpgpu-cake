
import base64,fcntl,hashlib,json,os,re,signal,subprocess,sys,time,traceback
from pathlib import Path
p=json.load(sys.stdin);root=Path(p['root']);control=Path(p['control']);deadline=p['deadline']
assert p['start']==1793155038 and deadline==1793183838 and p['start']<=time.time()<deadline-600
assert not control.exists();control.mkdir();os.chown(control,1001,1001)
lock=(root/'two-GPU-validation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for name,item in p['sources'].items():
 b=base64.b64decode(item['base64']);assert hashlib.sha256(b).hexdigest()==item['sha256'];compile(b,name,'exec')
 with (control/name).open('xb') as f:f.write(b)
 os.chown(control/name,1001,1001)
for name,h in p['original179'].items():assert hashlib.sha256((root/'original-sources'/name).read_bytes()).hexdigest()==h
for name,h in json.loads((root/'source_manifest.json').read_text())['files'].items():assert hashlib.sha256((root/'megatron-native-5be9626'/name).read_bytes()).hexdigest()==h
for name,value in [('campaign-first-start.epoch',p['start']),('deadline.epoch',deadline)]:
 with (control/name).open('x') as f:f.write(str(value)+'\n')
record={'schema':'datathinking-short-loop-controller-v1','status':'preparing','PID':os.getpid(),'start_epoch':p['start'],
 'deadline_epoch':deadline,'max_points':None,'cost_policy':'coupons_preferred_no_zero_gross_cost_cap','source_SHA':{n:i['sha256'] for n,i in p['sources'].items()},
 'original179_SHA':p['original179'],'original179_and_native600_unchanged':True,'stages':[],
 'raw_originals':'Data Thinking only','local_keys_cap_bytes':67108864,'no_parallel_GPU':True,
 'sampled_RSS_is_not_kernel_cgroup_limit':True,'kernel_per_rank_DATA_limit':144*1024**3}
def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2)
def status():
 record['updated_epoch']=time.time();q=control/'controller.json.tmp';q.write_text(json.dumps(record,indent=2));q.replace(control/'controller.json')
write(control/'component-independent-admission.json',p['component_independent_admission'])
capacity=p['capacity_independent_admission'];assert capacity['capacity_passed'] and capacity['explicit_human_authorization']['user_explicit_answer']=='允许容量实测并据实修订准入'
assert capacity['startup_required_GiB']=={'0':291,'1':28}
write(control/'capacity-independent-admission.json',capacity)
write(control/'authorization.json',p);status()
env=os.environ.copy();env.update(JIUGE_DT_ROOT=str(root),JIUGE_DT_CONTROL=str(control),JIUGE_DT_DEADLINE=str(deadline),
 LD_LIBRARY_PATH='/usr/local/iluvatar/lib64:/usr/local/corex/lib64',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='',
 CUDA_DEVICE_MAX_CONNECTIONS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
 USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=str(root/'cache/huggingface'),HF_MODULES_CACHE=str(root/'cache/huggingface/modules'),
 TORCH_EXTENSIONS_DIR=str(root/'work/torch-extensions'),HOME=str(root/'work'),PATH=str(root/'.venv/bin')+':/usr/local/corex/bin:'+env.get('PATH',''))
pair='GPU-a378f8c5-ada0-5b96-954c-ff19be59f3c0,GPU-0d6156e3-6ce8-5382-8ff3-f37ad849e80a'
def gate():
 text=subprocess.check_output(['/usr/local/corex/bin/ixsmi','-i',pair],env=env,text=True,timeout=30)
 assert 'No running processes found' in text and '96:00.0' in text and '99:00.0' in text,text
 memory=re.findall(r'(\d+)MiB\s*/\s*(\d+)MiB',text)
 assert len(memory)==2 and all(int(a)<=128 and int(b)==32768 for a,b in memory),text
 assert len(re.findall(r'\b0%\s+Default',text))==2,text
 inventory=subprocess.check_output(['/usr/local/corex/bin/ixsmi','-L'],env=env,text=True,timeout=20)
 for i,u in enumerate(pair.split(',')):assert f'GPU  {i}: Iluvatar BI-V150 (UUID: {u})' in inventory
 return {'epoch':time.time(),'selected_pair':text}
def available():return next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))

def processes():
 rows={}
 for path in Path('/proc').iterdir():
  if not path.name.isdigit():continue
  try:
   fields=(path/'stat').read_text().rsplit(') ',1)[1].split()
   rows[int(path.name)]={'ppid':int(fields[1]),'start_ticks':int(fields[19]),'state':fields[0],'RSS_bytes':int((path/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')}
  except (FileNotFoundError,ProcessLookupError,PermissionError,IndexError):pass
 return rows
tracked={}
def sample(child):
 rows=processes();selected={child.pid};changed=True
 while changed:
  prior=len(selected);selected.update(pid for pid,value in rows.items() if value['ppid'] in selected);changed=len(selected)!=prior
 for pid in selected:
  if pid in rows:tracked[pid]=rows[pid]['start_ticks']
 return sum(value['RSS_bytes'] for pid,value in rows.items() if tracked.get(pid)==value['start_ticks'] and value['state']!='Z')
def stop_owned():
 for signum in (signal.SIGTERM,signal.SIGKILL):
  rows=processes()
  for pid,start in sorted(tracked.items(),reverse=True):
   if pid in rows and rows[pid]['start_ticks']==start and rows[pid]['state']!='Z':
    try:os.kill(pid,signum)
    except ProcessLookupError:pass
  time.sleep(2)
def available():
 return next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))


def own_numa_snapshot():
 result=[];live=processes()
 for pid,ticks in tracked.items():
  if pid not in live or live[pid]['start_ticks']!=ticks:continue
  try:
   categories={};page=os.sysconf('SC_PAGE_SIZE')
   for line in Path('/proc',str(pid),'numa_maps').read_text().splitlines():
    fields=line.split();kind='file' if any(x.startswith('file=')for x in fields)else'anonymous_or_driver';size=next((int(x.split('=')[1])*1024 for x in fields if x.startswith('kernelpagesize_kB=')),page)
    for x in fields:
     if re.fullmatch('N[0-9]+=[0-9]+',x):
      n,count=x.split('=');key=kind+'/'+n;categories[key]=categories.get(key,0)+int(count)*size
   result.append({'PID':pid,'startticks':ticks,'mapped_resident_bytes':categories})
  except(FileNotFoundError,ProcessLookupError,PermissionError):pass
 with(control/'own-NUMA-journal.jsonl').open('a')as f:f.write(json.dumps({'epoch':time.time(),'processes':result})+'\n');f.flush()
 return result

child=None
def stage(row):
 global child
 row=dict(row);assert re.fullmatch('[a-zA-Z0-9_-]+',row['name'])
 assert time.time()<deadline-row.get('minimum_remaining',300),'Insufficient shared budget; no reset or shorter standard'
 script=(control/row['script']).resolve(strict=True);assert script.is_relative_to(control.resolve())
 assert hashlib.sha256(script.read_bytes()).hexdigest()==row['source_SHA256']
 tracked.clear();e=env.copy();gpu=row['kind']=='GPU';assert row['kind'] in ('CPU','GPU')
 if gpu:
  assert record.get('fresh_raw_SHA_passed'), 'Complete independent raw SHA precedes new GPU work'
  row['GPU_before']=gate()
  if row.get('needs_full7B'):
   node_free=int(next(x.split()[3] for x in Path('/sys/devices/system/node/node1/meminfo').read_text().splitlines() if ' MemFree:' in x))*1024
   node0=int(next(x.split()[3] for x in Path('/sys/devices/system/node/node0/meminfo').read_text().splitlines() if ' MemFree:' in x))*1024
   row['NUMA_admission']={'node0_MemFree_bytes':node0,'node1_MemFree_bytes':node_free,'required_node0_bytes':capacity['startup_required_MemFree_bytes']['0'],'required_node1_bytes':capacity['startup_required_MemFree_bytes']['1']}
   row['direct_attempt_authorized']=True # Human explicitly requested actual launch, solve failures after launch; retain every live guard.
  e.update(CUDA_VISIBLE_DEVICES='0,1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
 e.update(row.get('environment',{}));assert e['CUDA_VISIBLE_DEVICES']==('0,1' if gpu else '')
 row.update(started_epoch=time.time(),peak_sampled_own_tree_RSS_bytes=0)
 seconds=min(row.get('max_seconds',10800),int(deadline-time.time())-90)
 command=['timeout','--signal=TERM','--kill-after=20s',str(seconds)+'s','setpriv','--reuid=1001','--regid=1001','--clear-groups',str(root/'.venv/bin/python'),str(script),*row.get('args',[])]
 with (control/(row['name']+'.stdout.log')).open('x') as out,(control/(row['name']+'.stderr.log')).open('x') as err:
  child=subprocess.Popen(command,cwd=root,env=e,stdout=out,stderr=err,start_new_session=True)
  row['parent_PID']=child.pid;record.update(status=row['name'],current_stage=row);status();last=0
  try:
   while child.poll() is None:
    rss=sample(child);free=available();row['peak_sampled_own_tree_RSS_bytes']=max(row['peak_sampled_own_tree_RSS_bytes'],rss)
    node_free=int(next(x.split()[3] for x in Path('/sys/devices/system/node/node1/meminfo').read_text().splitlines() if ' MemFree:' in x))*1024 if gpu else 16*1024**3
    node0_free=int(next(x.split()[3] for x in Path('/sys/devices/system/node/node0/meminfo').read_text().splitlines() if ' MemFree:' in x))*1024 if gpu else 16*1024**3
    if rss>292*1024**3 or free<8*1024**3 or time.time()>deadline-90:
     row['safety_stop']={'RSS_bytes':rss,'host_available':free,'node1_MemFree_bytes':node_free,'node0_MemFree_bytes':node0_free,'epoch':time.time()};stop_owned();raise MemoryError('Own descendants memory reserve or immutable deadline')
    if time.time()-last>15:
     row.update(current_sampled_RSS_bytes=rss,current_host_available_bytes=free,current_node0_free=node0_free,current_node1_free=node_free);row['own_NUMA_snapshot']=own_numa_snapshot();status();last=time.time()
    time.sleep(.5)
  finally:
   if child.poll() is None:stop_owned();child.wait(timeout=25)
  row['exit_code']=child.returncode;sample(child);live=processes()
  survivors=[pid for pid,start in tracked.items() if pid in live and live[pid]['start_ticks']==start and live[pid]['state']!='Z']
  if survivors:stop_owned()
  row['owned_survivors_after_parent_exit']=survivors
 if gpu:row['GPU_after']=gate()
 row['finished_epoch']=time.time();record['stages'].append(row);record.pop('current_stage',None);status()
 assert row['exit_code']==0,row['name']+' failed; preserve versioned evidence'
 if row['name']=='CPU-diagnostic-admission':
  proof=json.loads((control/'full-model-diagnostic-admission.json').read_text());assert proof['diagnostic_only_admitted'] and not proof['full_correctness_or_long_loss_or_performance_accepted']
  record.update(fresh_raw_SHA_passed=True,prior_complete_raw_SHA_reference_reused=True,new_window_raw_SHA_rescanned=False,repaired_full7B_correctness_accepted=False);status()

def receive_owned_stop(sig,frame):
 raise TimeoutError('Own controller received stop signal; run original tracked-PID cleanup')
signal.signal(signal.SIGTERM,receive_owned_stop)
signal.signal(signal.SIGINT,receive_owned_stop)
try:
 write(control/'stage-plan-01.json',p['stage_plan'])
 record['GPU_admission']=gate();status()
 stage(p['first_stage'])
 sequence=1
 while True:
  record.update(status='awaiting-reviewed-stage-plan',next_plan_sequence=sequence);status()
  path=control/f'stage-plan-{sequence:02d}.json'
  while not path.exists():
   if time.time()>deadline-120:raise TimeoutError('Shared six-hour deadline; no extension')
   status();time.sleep(10)
  plan=json.loads(path.read_text());assert plan['deadline_epoch']==deadline
  if plan.get('finish'):
   record.update(status='completed',actual_completed_scope=plan['actual_completed_scope']);break
  for row in plan['stages']:stage(row)
  sequence+=1
except BaseException as error:
 record.update(status='failed',error=str(error),traceback=traceback.format_exc())
finally:
 if child is not None and child.poll() is None:stop_owned();child.wait(timeout=25)
 try:record['GPU_final']=gate()
 except BaseException as error:record['GPU_final_error']=str(error)
 record['finished_epoch']=time.time();status();write(control/'final.json',record)
print(json.dumps({'status':record['status'],'error':record.get('error'),'stages':[(s['name'],s['exit_code']) for s in record['stages']]}),flush=True)

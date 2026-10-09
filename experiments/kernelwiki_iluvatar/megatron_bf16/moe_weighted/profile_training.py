"""Separate untimed cProfile diagnostic; never a formal performance result."""
import argparse,fcntl,json,os,pstats,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BF=Path('/tmp/kda-bf16-continuation/megatron_bf16')
NATIVE=BF.parent/'megatron-native-5be9626'
a=argparse.ArgumentParser();a.add_argument('--arm',choices=('native','control','optimized'),required=True);a.add_argument('--version',default='003');v=a.parse_args()
folder=ROOT/('host-profile-'+v.arm+'-'+v.version);folder.mkdir()
command=[sys.executable,'-m','cProfile','-o',str(folder/'profile.pstats'),str(ROOT/'launch_weighted.py'),
 '--weighted-receipt',str(folder/'weighted.json'),'--bf16-root',str(BF),'--gradnorm-receipt',str(folder/'gradnorm.json'),
 '--megatron-root',str(NATIVE),'--contract',str(ROOT/'contract.json'),'--entry-evidence',str(folder/'entry.json')]
if v.arm!='native':command+=['--candidate',str(BF/'candidates/0008/candidate.py'),'--gradnorm-candidate','/tmp/kda-gradnorm-stage26/candidate.py']
if v.arm=='optimized':command+=['--weighted-candidate',str(ROOT/'candidates'/v.version/'candidate.py')]
command+=json.loads((ROOT/'contract.json').read_text())['training_argv']+['--train-iters','20','--seed','34110','--log-interval','10']
env={k:z for k,z in os.environ.items() if 'API_KEY' not in k and 'AUTH_TOKEN' not in k};env.update(CUDA_VISIBLE_DEVICES='0',MASTER_ADDR='127.0.0.1',MASTER_PORT='29941',RANK='0',WORLD_SIZE='1',LOCAL_RANK='0')
with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);start=time.perf_counter()
 with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
  code=subprocess.run(command,cwd=NATIVE,env=env,stdout=out,stderr=err,timeout=600).returncode
stats=pstats.Stats(str(folder/'profile.pstats'));rows=[]
for (file,line,name),(primitive,total,selftime,cumulative,callers) in stats.stats.items():
 if any(z in file for z in ('triton','torch/_dynamo','torch/_inductor','candidate.py','launch_weighted.py','kernel_adapter','autograd')):
  rows.append({'file':file,'line':line,'function':name,'primitive_calls':primitive,'total_calls':total,'self_seconds':selftime,'cumulative_seconds':cumulative})
r={'schema':'stage34-separate-host-profile-v1','arm':v.arm,'version':v.version,'exit_code':code,'whole_process_seconds':time.perf_counter()-start,
 'diagnostic_only':True,'timing_valid_for_formal_gain':False,'profiled_steps':20,'command':command,'top_host_cumulative':sorted(rows,key=lambda z:z['cumulative_seconds'],reverse=True)[:60],
 'top_host_self':sorted(rows,key=lambda z:z['self_seconds'],reverse=True)[:60]}
(folder/'profile.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'arm':v.arm,'exit_code':code,'top':r['top_host_cumulative'][:12]}))
raise SystemExit(0 if code==0 else 1)

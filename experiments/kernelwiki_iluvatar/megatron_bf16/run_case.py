"""Serialize one formal BF16 audit/profile/performance case with raw evidence."""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case')
    parser.add_argument('--mode',choices=['native','optimized'],required=True)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--megatron-root',type=Path,required=True)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--seed',type=int,default=26001)
    parser.add_argument('--audit',action='store_true')
    parser.add_argument('--profile',action='store_true')
    args=parser.parse_args()
    if not args.case.replace('-','').replace('_','').isalnum():
        parser.error('simple immutable case name required')
    if (args.mode=='optimized')!=(args.candidate is not None):
        parser.error('optimized mode requires a candidate; native forbids one')
    if args.audit and args.profile:
        parser.error('numerical audit and profiling are separate runs')
    folder=ROOT/'evidence'/args.case
    folder.mkdir(parents=True,exist_ok=False)
    contract=json.loads(args.contract.read_text())
    command=[sys.executable,'-u',str(ROOT/'launch_training.py'),'--megatron-root',str(args.megatron_root.resolve()),
             '--contract',str(args.contract.resolve()),'--entry-evidence',str(folder/'entry.json')]
    if args.candidate is not None:
        command+=['--candidate',str(args.candidate.resolve())]
    if args.audit:
        command+=['--audit-dir',str(folder/'snapshots')]
    if args.profile:
        command+=['--profile-dir',str(folder/'profile')]
    command+=contract['training_argv']+['--train-iters',str(args.steps),'--seed',str(args.seed),
                                      '--log-interval','1' if args.audit or args.profile else '10']
    if args.audit:
        command+=['--save',str(folder/'snapshots'),'--save-interval',str(args.steps),
                  '--save-wgrads-interval','1','--save-params-interval','1']
    env=os.environ.copy()
    for name in list(env):
        if 'API_KEY' in name or 'AUTH_TOKEN' in name:
            env.pop(name)
    env.update(CUDA_VISIBLE_DEVICES='0',MASTER_ADDR='127.0.0.1',MASTER_PORT='29931',RANK='0',WORLD_SIZE='1',LOCAL_RANK='0')
    record={'case':args.case,'mode':args.mode,'steps':args.steps,'seed':args.seed,'audit':args.audit,'profile':args.profile,
            'command':command,'started_utc':datetime.now(timezone.utc).isoformat(),
            'contract_sha256':hashlib.sha256(args.contract.read_bytes()).hexdigest()}
    # The persistent experiment root is NFS; its flock RPC can block in D state.
    # All runs in this BF16 suite share a Pod-local GPU0 lock instead.
    with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        started=time.perf_counter()
        with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
            try:
                record['exit_code']=subprocess.run(command,cwd=args.megatron_root,env=env,stdout=out,stderr=err,timeout=900).returncode
            except subprocess.TimeoutExpired:
                record.update(exit_code=124,error='900 second timeout')
        record['whole_process_elapsed_seconds']=time.perf_counter()-started
    record['finished_utc']=datetime.now(timezone.utc).isoformat()
    (folder/'command.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({key:record[key] for key in ('case','mode','exit_code','whole_process_elapsed_seconds')}),flush=True)
    return 0 if record['exit_code']==0 else 1


if __name__=='__main__':
    raise SystemExit(main())

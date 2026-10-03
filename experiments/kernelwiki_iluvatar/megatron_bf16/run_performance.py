"""Run ordered BF16 performance pairs only after sealed numerical gates pass."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--megatron-root',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--operator',type=Path,required=True)
    parser.add_argument('--prefix',required=True)
    args=parser.parse_args()
    contract=json.loads(args.contract.read_text())
    audit=json.loads(args.audit.read_text())
    operator=json.loads(args.operator.read_text())
    expected=sha(args.candidate)
    if contract['candidate_sha256']!=expected or audit.get('passed') is not True or audit['check_summary']['failed']!=0:
        parser.error('sealed candidate must pass full three-step native tensor audit')
    inputs=audit['input_files']
    if not any(row['sha256']==sha(args.contract) for row in inputs) or not any(row['sha256']==expected for row in inputs):
        parser.error('audit must verify this exact candidate/contract')
    if operator.get('status')!='passed' or operator.get('candidate_sha256')!=expected or operator.get('passed_cases')!=504:
        parser.error('candidate must pass all 504 fixed BF16 CE cases')
    for index,seed in enumerate(contract['performance']['independent_seeds']):
        order=('native','optimized') if index%2==0 else ('optimized','native')
        for arm in order:
            command=[sys.executable,str(ROOT/'run_case.py'),'%s-%s-perf-%d'%(arm,args.prefix,seed),
                '--mode',arm,'--contract',str(args.contract.resolve()),
                '--megatron-root',str(args.megatron_root.resolve()),'--steps','120','--seed',str(seed)]
            if arm=='optimized':
                command+=['--candidate',str(args.candidate.resolve())]
            subprocess.run(command,check=True)
    folder=ROOT/'evidence'/('%s-dispatch'%args.prefix)
    folder.mkdir(exist_ok=False)
    command=[sys.executable,str(ROOT/'observe_training.py'),'--dispatch-evidence',str(folder/'dispatch.json'),
             '--megatron-root',str(args.megatron_root.resolve()),'--contract',str(args.contract.resolve()),
             '--entry-evidence',str(folder/'entry.json'),'--candidate',str(args.candidate.resolve())]
    command+=contract['training_argv']+['--train-iters','3','--seed','26010','--log-interval','1']
    env=os.environ.copy()
    for name in list(env):
        if 'API_KEY' in name or 'AUTH_TOKEN' in name:
            env.pop(name)
    env.update(CUDA_VISIBLE_DEVICES='0',MASTER_ADDR='127.0.0.1',MASTER_PORT='29931',RANK='0',WORLD_SIZE='1',LOCAL_RANK='0')
    with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (folder/'stdout.log').open('x') as out,(folder/'stderr.log').open('x') as err:
            process=subprocess.run(command,env=env,stdout=out,stderr=err,timeout=900)
    (folder/'command.json').write_text(json.dumps({'command':command,'exit_code':process.returncode,
        'scope':'Separate untimed cached-runner observation, first JIT direct launch excluded.'},indent=2)+'\n')
    if process.returncode:
        return process.returncode
    summary=[sys.executable,str(ROOT/'summarize_performance.py'),'--contract',str(args.contract.resolve()),
             '--candidate',str(args.candidate.resolve()),'--prefix',args.prefix,
             '--output',str(ROOT/'evidence'/('%s-performance.json'%args.prefix))]
    return subprocess.run(summary).returncode


if __name__=='__main__':
    raise SystemExit(main())

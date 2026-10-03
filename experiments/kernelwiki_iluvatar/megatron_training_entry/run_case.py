"""Run one immutable formal-entry case with bounded execution and raw logs."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case')
    parser.add_argument('--mode', choices=['native','optimized'], required=True)
    parser.add_argument('--megatron-root', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=25001)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    if not args.case.replace('-','').replace('_','').isalnum():
        parser.error('simple case name required')
    folder = ROOT/'evidence'/args.case
    folder.mkdir(parents=True, exist_ok=False)
    contract = json.loads((ROOT/'contract.json').read_text(encoding='utf-8'))
    command = [sys.executable,'-u',str(ROOT/'launch_pretrain.py'),
               '--megatron-root',str(args.megatron_root),'--corex42-compat',
               '--entry-evidence',str(folder/'entry.json')]
    if args.mode == 'optimized':
        command += ['--iluvatar-kernels']
    command += contract['training_argv']+['--train-iters',str(args.steps),'--seed',str(args.seed),
                                         '--log-interval','1' if args.audit else '10']
    if args.audit:
        command += ['--entry-tensor-audit-dir',str(folder/'snapshots'),
                    '--save',str(folder/'snapshots'),'--save-interval',str(args.steps),
                    '--save-wgrads-interval','1','--save-params-interval','1']
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0',MASTER_ADDR='127.0.0.1',MASTER_PORT='29921',
               RANK='0',WORLD_SIZE='1',LOCAL_RANK='0')
    record={'case':args.case,'mode':args.mode,'command':command,'started_utc':datetime.now(timezone.utc).isoformat(),
            'contract_sha256':hashlib.sha256((ROOT/'contract.json').read_bytes()).hexdigest(),
            'steps':args.steps,'seed':args.seed,'audit':args.audit}
    with (ROOT.parent/'megatron_cc/.gpu-experiment.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        started=time.perf_counter()
        with (folder/'stdout.log').open('w') as out,(folder/'stderr.log').open('w') as err:
            try:
                record['exit_code']=subprocess.run(command,stdout=out,stderr=err,env=env,
                    cwd=args.megatron_root,timeout=600).returncode
            except subprocess.TimeoutExpired:
                record.update(exit_code=124,error='600 second timeout')
        record['whole_process_elapsed_seconds']=time.perf_counter()-started
    record['finished_utc']=datetime.now(timezone.utc).isoformat()
    (folder/'command.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))
    return 0 if record['exit_code']==0 else 1


if __name__=='__main__':
    raise SystemExit(main())

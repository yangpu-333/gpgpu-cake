"""Run immutable extended candidates serially on the authorized BI-V150 GPU."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate_id')
    parser.add_argument('phase', choices=['smoke', 'screen', 'final'])
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9]{4}', args.candidate_id):
        parser.error('four-digit candidate id required')
    candidate = ROOT / 'candidates' / args.candidate_id / 'candidate.py'
    best = PARENT / 'candidates/0003/candidate.py'
    folder = ROOT / 'evidence' / (args.candidate_id + '-' + args.phase)
    folder.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    native_root = '/private/atrex-megatron/src/megatron-lm'
    driver = [sys.executable, '-u', str(ROOT/'driver.py'), '--megatron-root', native_root]
    residual = [sys.executable, '-u', str(PARENT/'benchmark.py'),
                '--megatron-root', native_root, '--candidate', str(candidate), '--mode', 'operator']
    extended = driver + ['--candidate', str(candidate), '--variant', 'extended']
    control = driver + ['--candidate', str(best), '--variant', 'current-best']
    if args.phase == 'smoke':
        tasks = [('residual-smoke', residual+['--smoke']),
                 ('ce-smoke', extended+['--mode', 'cross-entropy', '--smoke'])]
    elif args.phase == 'screen':
        tasks = [('ce-24001', extended+['--mode', 'cross-entropy', '--seed','24001']),
                 ('primary-24001', extended+['--mode', 'model', '--seed','24001'])]
    else:
        tasks = [(f'residual-{s}', residual+['--seed',str(s)]) for s in (24002,24003,24004)]
        tasks += [(f'ce-{s}', extended+['--mode','cross-entropy','--seed',str(s)]) for s in (24002,24003,24004)]
        for seed in (24002,24003,24004):
            arms = [(f'best-{seed}',control), (f'primary-{seed}',extended)]
            if seed % 2:
                arms.reverse()
            tasks += [(name, command+['--mode','model','--seed',str(seed)]) for name,command in arms]
        tasks += [('holdout-24005', extended+['--mode','model','--seed','24005',
                  '--layers','2','--hidden','256','--heads','4','--sequence','17','--vocab','512']),
                  ('autocast-24006',extended+['--mode','model','--seed','24006','--autocast'])]
    manifest = {'candidate_id':args.candidate_id,'phase':args.phase,
                'started_utc':datetime.now(timezone.utc).isoformat(),'runs':[]}
    with (PARENT/'.gpu-experiment.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, command in tasks:
            command = command + ['--output',str(folder/(name+'.json'))]
            record={'name':name,'command':command,'started_utc':datetime.now(timezone.utc).isoformat()}
            with (folder/(name+'.stdout')).open('w') as out, (folder/(name+'.stderr')).open('w') as err:
                try:
                    record['exit_code']=subprocess.run(command,stdout=out,stderr=err,timeout=600).returncode
                except subprocess.TimeoutExpired:
                    record.update(exit_code=124,error='600 second timeout')
            record['finished_utc']=datetime.now(timezone.utc).isoformat()
            manifest['runs'].append(record)
            (folder/'commands.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
            print(name,record['exit_code'],flush=True)
            if record['exit_code'] != 0:
                print('Stopped on failure; raw evidence preserved.',flush=True)
                return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

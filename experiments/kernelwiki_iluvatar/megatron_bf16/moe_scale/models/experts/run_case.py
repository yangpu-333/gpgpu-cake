"""Fixed native/control/optimized MoE entry, with an external timeout and GPU0 lock."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BF = Path('/tmp/kda-bf16-continuation/megatron_bf16')
NATIVE = BF.parent / 'megatron-native-5be9626'
GRAD = Path('/tmp/kda-gradnorm-stage26')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', required=True)
    p.add_argument('--arm', choices=('native', 'control', 'optimized'), required=True)
    p.add_argument('--steps', type=int, choices=(3, 120), default=3)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--candidate', type=Path)
    p.add_argument('--capture-limit', type=int, default=0)
    a = p.parse_args()
    assert a.case.replace('-', '').isalnum()
    assert (a.arm == 'optimized') == (a.candidate is not None)
    assert not a.capture_limit or a.steps == 3
    folder = ROOT / a.case; folder.mkdir()
    contract_path = ROOT / 'contract.json'
    contract = json.loads(contract_path.read_text())
    command = [sys.executable, '-u', str(ROOT / 'launch_weighted.py'),
        '--weighted-receipt', str(folder / 'weighted.json'), '--bf16-root', str(BF),
        '--gradnorm-receipt', str(folder / 'gradnorm.json'), '--megatron-root', str(NATIVE),
        '--contract', str(contract_path), '--entry-evidence', str(folder / 'entry.json')]
    if a.arm != 'native':
        command += ['--candidate', str(BF / 'candidates/0008/candidate.py'),
                    '--gradnorm-candidate', str(GRAD / 'candidate.py')]
    if a.candidate:
        command += ['--weighted-candidate', str(a.candidate)]
    if a.steps == 3:
        command += ['--audit-dir', str(folder / 'snapshots'), '--route-capture',
                    '--capture-limit', str(a.capture_limit)]
    command += contract['training_argv'] + ['--train-iters', str(a.steps), '--seed', str(a.seed),
                '--log-interval', '1' if a.steps == 3 else '10']
    if a.steps == 3:
        command += ['--save', str(folder / 'snapshots'), '--save-interval', '3',
                    '--save-wgrads-interval', '1', '--save-params-interval', '1']
    env = {k: v for k, v in os.environ.items() if 'API_KEY' not in k and 'AUTH_TOKEN' not in k}
    env.update(CUDA_VISIBLE_DEVICES='0', MASTER_ADDR='127.0.0.1', MASTER_PORT='29941',
               RANK='0', WORLD_SIZE='1', LOCAL_RANK='0')
    utc = lambda: datetime.now(timezone.utc).isoformat()
    record = {'case': a.case, 'stage34_arm': a.arm, 'mode': 'native' if a.arm == 'native' else 'optimized',
              'steps': a.steps, 'seed': a.seed, 'audit': a.steps == 3, 'profile': False,
              'command': command, 'contract_sha256': sha(contract_path),
              'wrapper_sha256': sha(ROOT / 'launch_weighted.py'), 'runner_sha256': sha(__file__),
              'weighted_candidate_sha256': sha(a.candidate) if a.candidate else None, 'started_utc': utc()}
    with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        start = time.perf_counter()
        with (folder / 'stdout.log').open('x') as out, (folder / 'stderr.log').open('x') as err:
            proc = subprocess.Popen(command, cwd=NATIVE, env=env, stdout=out, stderr=err, start_new_session=True)
            try:
                record['exit_code'] = proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                record.update(exit_code=124, error='independent600-second timeout')
        record['whole_process_elapsed_seconds'] = time.perf_counter() - start
    record['finished_utc'] = utc()
    (folder / 'command.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record[k] for k in ('case', 'stage34_arm', 'exit_code', 'whole_process_elapsed_seconds')}))
    return 0 if record['exit_code'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())

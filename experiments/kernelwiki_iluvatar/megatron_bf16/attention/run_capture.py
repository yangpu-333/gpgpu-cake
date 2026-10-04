"""One native formal training step for actual attention semantics, not throughput."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

BASE = Path('/tmp/kda-attention-stage31')
BF = Path('/tmp/kda-bf16-continuation/megatron_bf16')
NATIVE = BF.parent / 'megatron-native-5be9626'
GRAD = Path('/tmp/kda-gradnorm-stage26')
SEALED = Path('/tmp/kda-swiglu-stage29')
folder = BASE / 'native-capture-01'
folder.mkdir(exist_ok=False)
contract = SEALED / 'contracts/main.json'
config = json.loads(contract.read_text())
cmd = [sys.executable, str(BASE / 'launch_capture.py'),
       '--attention-capture', str(folder / 'attention'),
       '--attention-receipt', str(folder / 'attention.json'),
       '--delegate-launcher', str(SEALED / 'launch_swiglu-03.py'),
       '--swiglu-receipt', str(folder / 'swiglu.json'),
       '--gradnorm-launcher', str(GRAD / 'launch_gradnorm-04.py'),
       '--bf16-root', str(BF), '--gradnorm-receipt', str(folder / 'gradnorm.json'),
       '--megatron-root', str(NATIVE), '--contract', str(contract),
       '--entry-evidence', str(folder / 'entry.json')]
cmd += config['training_argv'] + ['--train-iters', '1', '--seed', '31101', '--log-interval', '1']
env = os.environ.copy()
for name in list(env):
    if 'API_KEY' in name or 'AUTH_TOKEN' in name:
        env.pop(name)
env.update(CUDA_VISIBLE_DEVICES='0', MASTER_ADDR='127.0.0.1', MASTER_PORT='29973',
           RANK='0', WORLD_SIZE='1', LOCAL_RANK='0')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc = lambda: datetime.now(timezone.utc).isoformat()
record = {'command': cmd, 'mode': 'native', 'steps': 1, 'seed': 31101,
          'contract_sha256': sha(contract), 'runner_sha256': sha(Path(__file__)),
          'started_utc': utc(), 'scope': 'separate untimed capture; historical stage29 configuration'}
with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    start = time.perf_counter()
    with (folder / 'stdout.log').open('x') as out, (folder / 'stderr.log').open('x') as err:
        child = subprocess.Popen(cmd, env=env, cwd=NATIVE, stdout=out, stderr=err,
                                 start_new_session=True)
        try:
            record['exit_code'] = child.wait(timeout=240)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            record.update(exit_code=124, error='capture timeout')
record.update(finished_utc=utc(), elapsed_seconds=time.perf_counter()-start)
(folder / 'command.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record), flush=True)
assert record['exit_code'] == 0
print((folder / 'attention.json').read_text(), flush=True)

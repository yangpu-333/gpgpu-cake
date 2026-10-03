"""Run untimed compiled-launch observation without changing accepted cases."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: run_dispatch.py MEGATRON_ROOT')
    root = Path(sys.argv[1]).resolve()
    folder = ROOT/'evidence/formal-dispatch-01'
    folder.mkdir(exist_ok=False)
    contract = json.loads((ROOT/'contract.json').read_text())
    command = [sys.executable, '-u', str(ROOT/'observe_pretrain.py'),
        '--dispatch-evidence',str(folder/'dispatch.json'), '--megatron-root',str(root),
        '--corex42-compat', '--iluvatar-kernels', '--entry-evidence',str(folder/'entry.json'),
        *contract['training_argv'], '--train-iters','3','--seed','25001','--log-interval','1']
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0', MASTER_ADDR='127.0.0.1', MASTER_PORT='29921',
               RANK='0', WORLD_SIZE='1', LOCAL_RANK='0')
    started = time.perf_counter()
    with (ROOT.parent/'megatron_cc/.gpu-experiment.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (folder/'stdout.log').open('w') as out, (folder/'stderr.log').open('w') as err:
            result = subprocess.run(command, env=env, cwd=root, stdout=out, stderr=err, timeout=600)
    record = {'command':command, 'exit_code':result.returncode,
        'elapsed_seconds':time.perf_counter()-started, 'finished_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Untimed actual formal-loop dispatch observation; excluded from performance comparison'}
    (folder/'command.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())

"""Stop only this controller's descendants after the verified AdamW step error."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT = PREP / 'pinning-recovery-20261016-03'


def identity(pid):
    process = Path('/proc') / str(pid)
    if process.stat().st_uid != os.getuid():
        raise ValueError('Process belongs to a different user')
    start = process.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
    return start


def main():
    if socket.gethostname() != 'sg-huids25-261016-3d89f-default0-0':
        raise ValueError('Wrong allocation')
    evidence = []
    for rank in (0, 1):
        path = PREP / f'runs/pinning-native-resume-05/rank-{rank}/optimizer-state.json'
        value = json.loads(path.read_text())['optimizers'][0]
        if value['min_step'] != 200 or value['max_step'] != 200 or value['adam_state_entries'] != 199:
            raise ValueError('Expected mismatch evidence absent; no processes stopped')
        evidence.append({'rank': rank, 'observed_step': 200, 'expected_step': 3,
                         'receipt_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    state = json.loads((ROOT / 'controller.json').read_text())
    if state['status'] != 'pinning-optimized-resume-05':
        raise ValueError('Controller is not in the expected untimed candidate restore')
    pid = state['pid']
    identity(pid)
    command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if b'continue_pinning_recovery_v2.py' not in command:
        raise ValueError('Controller identity changed')
    pending, records = [state['active_child_pid']], []
    while pending:
        child = pending.pop()
        try:
            records.append((child, identity(child)))
            pending.extend(map(int, Path(f'/proc/{child}/task/{child}/children').read_text().split()))
        except FileNotFoundError:
            pass
    receipt = {'reason': 'Actual restored native AdamW step200 instead of3; invalid next update',
               'evidence': evidence, 'controller_pid': pid, 'descendant_pids': [p for p, _ in records],
               'original_checkpoint_and_numeric_archives_deleted': False}
    with (ROOT / 'stop-after-native-optimizer-step-mismatch.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    os.kill(pid, signal.SIGINT)
    time.sleep(1)
    for child, start in reversed(records):
        try:
            if identity(child) == start:
                os.kill(child, signal.SIGTERM)
        except (FileNotFoundError, ProcessLookupError):
            pass
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

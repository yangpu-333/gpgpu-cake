"""Stop only this experiment's idle followup 01 before a versioned handoff."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    if socket.gethostname() != 'sg-huids25-261015-0d9b8-default0-0':
        raise ValueError('Wrong allocation')
    root = PREP / 'pinning-followup-20261015-01'
    state = json.loads((root / 'controller.json').read_text())
    if state['status'] != 'waiting_full_numeric_04' or 'active_child_pid' in state:
        raise ValueError('Only an idle waiting followup may be handed off')
    pid = state['pid']
    proc = Path(f'/proc/{pid}')
    argv = [os.fsdecode(part) for part in (proc / 'cmdline').read_bytes().split(b'\0') if part]
    if proc.stat().st_uid != os.getuid() or Path(argv[-1]).name != 'continue_pinning_validation.py':
        raise ValueError('Not this user followup controller')
    receipt = {'schema': 'jiuge7b-followup-idle-handoff-v1', 'utc': datetime.now(timezone.utc).isoformat(),
               'pid': pid, 'argv': argv, 'numeric_audit_signalled': False,
               'reason': 'Add an unchanged native checkpoint-resume reference for dataloader startup CPU RNG',
               'old_originals_deleted': False, 'signal': 'SIGINT'}
    with (root / 'handoff-stop-request.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    os.kill(pid, signal.SIGINT)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

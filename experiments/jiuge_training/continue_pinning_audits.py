"""Continue only the full numeric audits after the current native run completes.

No formal performance run is admitted by this controller.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def remaining_keeper_seconds():
    stat = Path('/proc/1/stat').read_text().rsplit(')', 1)[1].split()
    started = float(stat[19]) / os.sysconf('SC_CLK_TCK')
    elapsed = float(Path('/proc/uptime').read_text().split()[0]) - started
    if Path('/proc/1/cmdline').read_bytes().split(b'\0')[:2] != [b'sleep', b'39600']:
        raise ValueError('Unexpected keeper; no automatic resource extension')
    return 39600 - elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--number', choices=('03', '04'), default='03')
    args = parser.parse_args()
    if socket.gethostname() != 'sg-huids25-261015-0d9b8-default0-0':
        raise ValueError('Wrong allocated host')
    campaign = args.campaign.resolve(strict=True)
    if campaign != PREP / f'pinning-validation-20261015-{args.number}':
        raise ValueError('Wrong immutable campaign')
    receipt = {'schema': 'jiuge-pinning-numeric-campaign-controller-v1',
               'started_utc': datetime.now(timezone.utc).isoformat(), 'runs': [],
               'formal_performance_started': False, 'status': 'waiting_for_native'}
    state = campaign / 'controller.json'
    if state.exists():
        raise ValueError('Controller already registered')
    def save():
        tmp = state.with_suffix('.tmp')
        tmp.write_text(json.dumps(receipt, indent=2) + '\n')
        tmp.replace(state)
    save()
    try:
        native = PREP / f'runs/pinning-native-audit-{args.number}/case.json'
        until = time.monotonic() + 7200
        while not native.is_file():
            if time.monotonic() > until or remaining_keeper_seconds() < 900:
                raise TimeoutError('Native run or free allocation budget exhausted')
            time.sleep(5)
        if json.loads(native.read_text())['exit_code'] != 0:
            raise ValueError('Native full audit did not complete')
        for arm in ('native', 'control', 'optimized'):
            if arm != 'native':
                # Two hours per audit plus a ten-minute shutdown/archive reserve.
                if remaining_keeper_seconds() < 7800:
                    raise TimeoutError('Insufficient remaining allocation for the next complete audit')
                case = f'pinning-{arm}-audit-{args.number}'
                receipt['status'] = 'running_' + arm
                save()
                command = [sys.executable, str(PREP / 'run_pinning_case.py'), '--case', case,
                           '--steps', '3', '--seq-length', '2048', '--pinning-arm', arm,
                           '--pinning-campaign', str(campaign), '--full-state-audit']
                with (campaign / f'{arm}-controller.log').open('x') as stream:
                    process = subprocess.Popen(command, cwd=PREP, stdout=stream, stderr=subprocess.STDOUT)
                    receipt['active_controller_child_pid'] = process.pid
                    save()
                    code = process.wait()
                receipt.pop('active_controller_child_pid', None)
                if code:
                    raise ValueError(f'{arm} audit failed, exit={code}')
            completions = [json.loads((campaign / arm / f'rank-{rank}/completion.json').read_text())
                           for rank in (0, 1)]
            if not all(row['passed'] and row['complete_fields'] and row['steps'] == 3 for row in completions):
                raise ValueError('Incomplete full-state comparison')
            receipt['runs'].append({'arm': arm, 'rank_completions': completions})
            save()
        receipt['status'] = 'numeric_runs_completed_pending_archive_and_resume'
        receipt['local_raw_copy_verified'] = False
        receipt['full_model_checkpoint_resume_verified'] = False
    except BaseException as exc:
        receipt.update(status='stopped', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt['updated_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

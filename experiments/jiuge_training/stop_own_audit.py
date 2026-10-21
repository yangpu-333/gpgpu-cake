"""Stop only the exact current user's named torchrun; preserve every raw file."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal

ROOT = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--reason', choices=('resource-handoff', 'parallel-archive'), default='resource-handoff')
    args = parser.parse_args()
    folder = (ROOT / 'runs' / args.case).resolve(strict=True)
    if folder.parent != (ROOT / 'runs').resolve() or not args.case.startswith('pinning-native-audit-'):
        raise ValueError('Unexpected run namespace')
    matches = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            values = (proc / 'cmdline').read_bytes().split(b'\0')
            if b'torch.distributed.run' in values and os.fsencode(folder) in values:
                matches.append(int(proc.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    if len(matches) != 1:
        raise ValueError('Expected exactly one owned named torchrun: ' + str(matches))
    record = {'case': args.case, 'owned_torchrun_pid': matches[0], 'signal': 'SIGINT',
              'time_utc': datetime.now(timezone.utc).isoformat(),
              'reason': ('Preserve partial raw archive; hand off to the newly requested longer allocation'
                         if args.reason == 'resource-handoff' else
                         'Preserve partial raw archive; rerun with bounded parallel compression and unchanged numeric coverage'),
              'other_processes_signalled': False, 'raw_files_deleted': False}
    with (folder / 'controller-stop.json').open('x') as stream:
        json.dump(record, stream, indent=2)
    os.kill(matches[0], signal.SIGINT)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()

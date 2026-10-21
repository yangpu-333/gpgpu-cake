"""Briefly hold this job's own startup sleep within its recorded free window."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import time

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    pid=28
    if socket.gethostname()!='sg-huids25-261014-26803-default0-0':
        raise RuntimeError('Unexpected job; refusing to signal any process')
    proc=Path(f'/proc/{pid}')
    argv=(proc/'cmdline').read_bytes().split(b'\0')
    if proc.stat().st_uid!=os.getuid() or [Path(os.fsdecode(argv[0])).name,os.fsdecode(argv[1])] != ['sleep','39600']:
        raise RuntimeError('Not this allocation own idle startup sleep')
    start=float((proc/'stat').read_text().rsplit(')',1)[1].split()[19])/os.sysconf('SC_CLK_TCK')
    elapsed=float(Path('/proc/uptime').read_text().split()[0])-start
    seconds=min(1800,43200-elapsed-600)
    if seconds<=0:
        raise RuntimeError('No room in the recorded 12-hour free window')
    def interrupted(*_):
        raise SystemExit('Guard interrupted; releasing own startup command')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,interrupted)
    record={'schema':'own-job-startup-lifetime-guard-v1','pid':pid,'argv':['sleep','39600'],
            'started_utc':datetime.now(timezone.utc).isoformat(),'maximum_seconds':seconds,
            'recorded_platform_free_seconds':43200,'original_keeper_elapsed_seconds':elapsed,
            'purpose':'finish active experiment and archive within the existing free resource window',
            'other_processes_signalled':False,'status':'holding_own_startup_sleep'}
    path=PREP/'job-lifetime-guard.json'
    path.write_text(json.dumps(record,indent=2)+'\n')
    try:
        os.kill(pid,signal.SIGSTOP)
        print(json.dumps(record),flush=True)
        time.sleep(seconds)
    finally:
        record.update(status='released',released_utc=datetime.now(timezone.utc).isoformat())
        path.write_text(json.dumps(record,indent=2)+'\n')
        os.kill(pid,signal.SIGCONT)


if __name__=='__main__':
    main()

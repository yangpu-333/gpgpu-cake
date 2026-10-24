"""Wait for final key SHA, then perform one CPU-only independent review."""
from datetime import datetime, timezone
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
LOCAL = BASE/'results/sft-phased-acceptance-postprocess-20261024-01'
DRIVER = BASE/'results/sft-phased-acceptance-driver-20261024-01/status.json'


def main():
    LOCAL.mkdir(exist_ok=True)
    with (LOCAL/'postprocess.lock').open('a+b') as lock:
        lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        path = LOCAL/'status.json'
        def save(status, **values):
            result = {'pid':os.getpid(),'status':status,'updated_utc':datetime.now(timezone.utc).isoformat(),
                      'GPU_used':False,**values}
            temp = LOCAL/'status.json.tmp'
            temp.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');temp.replace(path)
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))['status'] not in ('waiting-for-final-keys',):
            raise ValueError('Preserve closed or failed CPU postprocessing; do not rerun')
        while True:
            driver = json.loads(DRIVER.read_text(encoding='utf-8')) if DRIVER.exists() else {}
            if driver.get('status')=='remote-stopped':
                save('remote-failed-review-not-started',error=driver.get('error'));return 2
            if driver.get('status')=='completed':
                break
            save('waiting-for-final-keys',driver_status=driver.get('status'))
            time.sleep(30)
        save('independent-CPU-review')
        with (LOCAL/'review.stdout.log').open('x') as out,(LOCAL/'review.stderr.log').open('x') as err:
            code = subprocess.run([sys.executable,str(BASE/'review_sft_phased_acceptance_20261024.py')],
                                  stdout=out,stderr=err,timeout=600).returncode
        if code:
            save('independent-review-failed',exit_code=code);return 3
        result = json.loads((BASE/'results/sft-phased-acceptance-independent-review-20261024-01.json').read_text(encoding='utf-8'))
        save('independent-review-completed-reports-pending',passed=result['passed'],
             formal_performance_passed=result['formal_performance_passed']);return 0


if __name__=='__main__':
    raise SystemExit(main())

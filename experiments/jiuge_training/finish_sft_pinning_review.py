"""Wait locally for the active key-only driver, then review closed evidence."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
DRIVER = BASE / 'results/sft-pinning-driver-20261020-01/status.json'
OUT = BASE / 'results/sft-pinning-review-driver-20261020-01'


def main():
    OUT.mkdir(exist_ok=False)
    state = {'schema': 'jiuge7b-sft-pinning-post-review-driver-v1',
             'started_utc': datetime.now(timezone.utc).isoformat(),
             'raw_originals_downloaded': False}

    def save(status, **values):
        state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
        temporary = OUT / 'status.json.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n', encoding='utf-8')
        temporary.replace(OUT / 'status.json')
        print(json.dumps(state), flush=True)

    save('waiting-for-closed-key-results')
    until = time.monotonic() + 12 * 3600
    while time.monotonic() < until:
        try:
            driver = json.loads(DRIVER.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(20)
            continue
        if driver['status'] in ('remote-stopped', 'authorized-campaign-window-ended'):
            save('validation-incomplete', driver=driver)
            return 2
        if driver['status'] != 'completed':
            time.sleep(20)
            continue
        save('independent-review-running')
        with (OUT / 'stdout.log').open('x', encoding='utf-8') as stdout, \
                (OUT / 'stderr.log').open('x', encoding='utf-8') as stderr:
            result = subprocess.run([sys.executable, '-u',
                                     str(BASE / 'review_sft_pinning_results.py')],
                                    cwd=BASE, stdout=stdout, stderr=stderr,
                                    timeout=1800, check=False)
        if result.returncode:
            save('independent-review-failed', exit_code=result.returncode)
            return result.returncode
        receipt = json.loads((BASE / 'results/sft-pinning-independent-review-20261020.json')
                             .read_text(encoding='utf-8'))
        save('completed', formal_performance_passed=receipt['formal_performance_passed'],
             review='results/sft-pinning-independent-review-20261020.json')
        return 0
    save('review-wait-window-ended')
    return 3


if __name__ == '__main__':
    raise SystemExit(main())

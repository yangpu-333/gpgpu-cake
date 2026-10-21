"""One CPU-only postprocessor; preserve completed keys and failure evidence."""
from datetime import datetime, timezone
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
from drive_sft_pinning_validation import SSH, PREP, run, remote_json

BASE = Path(__file__).resolve().parent
ROOT = BASE/'results/sft-moment-direct-zero-postprocess-20261021-01'
DRIVER = BASE/'results/sft-moment-direct-zero-driver-20261021-01'


def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'postprocess.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'1'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        state = {'schema': 'jiuge7b-sft-numa-CPU-postprocessing-v1', 'pid': os.getpid(),
                 'status': 'waiting_for_verified_keys', 'GPU_used': False, 'raw_originals_downloaded': False}

        def save(status, **extra):
            state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
            tmp = ROOT/'status.tmp'
            tmp.write_text(json.dumps(state, indent=2)+'\n', encoding='utf-8')
            tmp.replace(ROOT/'status.json')

        def local(script, stage):
            save(stage)
            with (ROOT/(stage+'.log')).open('x', encoding='utf-8') as stream:
                subprocess.run([sys.executable, str(BASE/script)], stdout=stream,
                               stderr=subprocess.STDOUT, check=True, timeout=900)

        save('waiting_for_verified_keys')
        try:
            deadline = time.monotonic()+4*3600
            while time.monotonic() < deadline:
                path = DRIVER/'status.json'
                driver = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
                status = driver.get('status')
                if status == 'completed':
                    break
                if status in ('remote-stopped', 'observation-deadline-reached'):
                    raise RuntimeError('Scientific driver stopped: '+json.dumps(driver))
                save('waiting_for_verified_keys', driver_status=status,
                     completed_cases=driver.get('completed_cases'))
                time.sleep(45)
            else:
                raise TimeoutError('CPU postprocessing wait expired; no new resource created')
            local('review_sft_moment_direct_zero_20261021.py', 'independent_review')
            save('ACT_CPU_packaging')
            payload = (BASE/'package_sft_moment_direct_zero_20261021.py').read_bytes()
            import base64
            code = '''from pathlib import Path
import base64,hashlib,json,subprocess,sys
p=Path('''+repr(PREP)+''')
f=p/'package_sft_moment_direct_zero_20261021.py'
payload=base64.b64decode('''+repr(base64.b64encode(payload).decode())+''')
assert not f.exists() or f.read_bytes()==payload
if not f.exists():
    with f.open('xb') as stream:stream.write(payload)
receipt=p/'sft-moment-direct-zero-delivery-20261021-01/verification.json'
if not receipt.exists():subprocess.run([sys.executable,str(f)],check=True)
print(receipt.read_text())
'''
            response = run([*SSH, PREP+'/.venv/bin/python -'], input=code.encode()).decode()
            # The packaging program prints a compact status before the complete
            # JSON receipt. Parse the final full object without downloading zip.
            decoder = json.JSONDecoder()
            objects = []
            cursor = 0
            while cursor < len(response):
                while cursor < len(response) and response[cursor].isspace(): cursor += 1
                if cursor >= len(response): break
                value, used = decoder.raw_decode(response[cursor:])
                objects.append(value); cursor += used
            receipt = objects[-1]
            if not receipt['passed'] or not receipt['compressed_and_decompressed_SHA_passed'] or receipt['bundle_downloaded_to_local']:
                raise ValueError('ACT full raw trace packaging SHA gate failed')
            target = BASE/'results/sft-moment-direct-zero-delivery-verification-20261021.json'
            with target.open('x', encoding='utf-8') as stream:
                json.dump(receipt, stream, indent=2)
            local('plot_sft_moment_direct_zero_20261021.py', 'plotting')
            files = ['review_sft_moment_direct_zero_20261021.py', 'package_sft_moment_direct_zero_20261021.py',
                     'plot_sft_moment_direct_zero_20261021.py', 'finish_sft_moment_direct_zero_20261021.py']
            save('scientific_postprocessing_completed_reports_pending',
                 CPU_source_sha256={n: hashlib.sha256((BASE/n).read_bytes()).hexdigest() for n in files},
                 raw_trace_count=receipt['verified_raw_traces'], ACT_raw_bundle=receipt['bundle'],
                 ACT_raw_bundle_sha256=receipt['bundle_sha256'])
        except BaseException as exc:
            save('failed', error_type=type(exc).__name__, error=str(exc))
            raise


if __name__ == '__main__':
    main()

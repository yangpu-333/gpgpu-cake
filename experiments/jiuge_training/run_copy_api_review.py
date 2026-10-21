"""CPU-only review of closed native/candidate traces; originals stay on ACT."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from pinning_key_results import export

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT = PREP / 'pinning-copy-api-review-20261016-01'


def main():
    if socket.gethostname() != 'sg-huids25-261016-24e4c-default0-0':
        raise ValueError('Wrong analysis allocation')
    formal_path = PREP / 'pinning-formal-20261016-01/controller.json'
    profile_path = PREP / 'pinning-profile-review-20261016-01/controller.json'
    formal = json.loads(formal_path.read_text())
    profile = json.loads(profile_path.read_text())
    if formal['status'] != 'completed_promoted' or profile['status'] != 'completed':
        raise ValueError('Finish all formal measurements and diagnostic captures first')
    cases = ('pinning-native-profile-01', 'pinning-optimized-profile-01')
    if set(profile['cases']) != set(cases):
        raise ValueError('Wrong diagnostic cases')
    for case in cases:
        proof = json.loads((profile_path.parent / f'{case}-primary-archive-verification.json').read_text())
        if not proof['passed'] or proof['verified_file_count'] < 1:
            raise ValueError('Complete primary trace archive verification required')
    cards = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                    '--format=csv,noheader'], text=True)
    if apps.strip() or set(cards.split()) != {
            'GPU-09ab6992-d722-c828-7bca-934ccb46aa54', 'GPU-c9db5c06-61c6-01b7-b9aa-20307600482c'}:
        raise ValueError('Cards changed or GPU compute processes present; do not interfere')
    ROOT.mkdir(exist_ok=False)
    source = ROOT / 'source'
    source.mkdir()
    hashes = {}
    for name in ('run_copy_api_review.py', 'inspect_copy_api.py', 'analyze_trace.py'):
        payload = (PREP / name).read_bytes()
        (source / name).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    state = {'schema': 'jiuge7b-copy-api-review-controller-v1', 'status': 'running',
             'started_utc': datetime.now(timezone.utc).isoformat(), 'hostname': socket.gethostname(),
             'source_sha256': hashes, 'gpu_gate': {'cards': cards, 'compute_apps': apps, 'passed': True},
             'formal_controller_sha256': hashlib.sha256(formal_path.read_bytes()).hexdigest(),
             'profile_controller_sha256': hashlib.sha256(profile_path.read_bytes()).hexdigest(),
             'cases': list(cases), 'cpu_only': True, 'raw_trace_local_download_enabled': False,
             'formal_throughput_evidence': False}
    def save():
        temp = ROOT / 'controller.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(ROOT / 'controller.json')
    save()
    try:
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
        for case in cases:
            with (ROOT / f'{case}.log').open('x') as stream:
                subprocess.run([sys.executable, source / 'inspect_copy_api.py', '--case-root',
                                PREP / 'runs' / case, '--output', ROOT / f'{case}.json'],
                               env=environment, stdout=stream, stderr=subprocess.STDOUT,
                               check=True, timeout=600)
        state.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        bundle = PREP / 'pinning-copy-api-key-results-20261016.zip'
        catalog = export(PREP, [ROOT], bundle)
        print(json.dumps({'status': state['status'], 'key_bundle': str(bundle),
                          'key_bytes': catalog['key_uncompressed_bytes'], 'source_sha256': hashes}), flush=True)
    except BaseException as exc:
        state.update(status='stopped', error_type=type(exc).__name__, error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()

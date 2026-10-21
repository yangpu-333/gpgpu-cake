"""Fixed three-arm benchmark; at most one predeclared unchanged full repeat."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from analyze_pinning_formal import analyze, ARMS, SEEDS
from continue_pinning_validation import remaining_seconds
from run_pinning_formal import check_admission

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
ORDERS = (('native', 'control', 'optimized'), ('optimized', 'control', 'native'), ('control', 'native', 'optimized'))
UUIDS = {'GPU-09ab6992-d722-c828-7bca-934ccb46aa54', 'GPU-c9db5c06-61c6-01b7-b9aa-20307600482c'}
CODE = ('continue_pinning_formal.py', 'run_pinning_formal.py', 'pinning_formal_entry.py', 'analyze_pinning_formal.py',
        'pinning_candidate.py', 'pinning_resume_audit.py', 'pinning_full_audit.py', 'jiuge7b_bridge.py',
        'optimizer_evidence.py', 'run_native_case.py', 'continue_pinning_validation.py', 'pinning.contract.json')


def resources():
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True, timeout=20).stdout
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'],
                          capture_output=True, text=True, check=True, timeout=20).stdout
    rows = [[part.strip() for part in line.split(',')] for line in gpu.strip().splitlines()]
    passed = len(rows) == 2 and {row[0] for row in rows} == UUIDS and not apps.strip()
    passed &= all(int(row[2]) <= 256 and int(row[3]) == 0 for row in rows)
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'gpu_stdout': gpu,
              'compute_apps_stdout': apps, 'passed': passed}
    if not passed:
        raise ValueError('Allocated GPUs changed or other GPU computation present: ' + json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'sg-huids25-261015-0d9b8-default0-0':
        raise ValueError('Wrong allocation')
    check_admission(args.admission)
    root = PREP / 'pinning-formal-20261015-01'
    root.mkdir(exist_ok=False)
    source = root / 'source'
    source.mkdir()
    hashes = {}
    for name in CODE:
        payload = (PREP / name).read_bytes()
        hashes[name] = hashlib.sha256(payload).hexdigest()
        (source / name).write_bytes(payload)
    state = {'schema': 'jiuge7b-pinning-formal-controller-v1', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'source_sha256': hashes, 'status': 'initializing', 'completed_cases': [], 'rounds': [],
             'admission_sha256': hashlib.sha256(args.admission.read_bytes()).hexdigest(), 'formal_performance_passed': False}
    def save():
        temp = root / 'controller.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(root / 'controller.json')
    def run_round(name):
        if remaining_seconds() < 10100:
            raise TimeoutError('Insufficient remaining free allocation for one complete formal round')
        before = resources()
        state['status'] = 'running_' + name
        save()
        for seed, order in zip(SEEDS, ORDERS):
            for arm in order:
                if remaining_seconds() < 1800:
                    raise TimeoutError('Reserve reached; no further formal cases started')
                if any(hashlib.sha256((PREP / key).read_bytes()).hexdigest() != digest for key, digest in hashes.items()):
                    raise ValueError('Formal source changed; no retuning or silent retry allowed')
                resources()
                case = f'pinning-formal-{name}-{seed}-{arm}'
                argv = [sys.executable, str(PREP / 'run_pinning_formal.py'), '--case', case, '--steps', '120',
                        '--seq-length', '2048', '--seed', str(seed), '--pinning-arm', arm,
                        '--pinning-campaign', str(PREP / 'pinning-validation-20261015-05'),
                        '--admission', str(args.admission)]
                with (root / f'{case}.log').open('x') as stream:
                    result = subprocess.run(argv, cwd=PREP, stdout=stream, stderr=subprocess.STDOUT,
                                            timeout=max(1, remaining_seconds() - 600))
                if result.returncode:
                    raise ValueError(f'Formal case failed: {case}; no timing replacement')
                # Outside every retained interval; all workers have exited.
                time.sleep(3)
                resources()
                state['completed_cases'].append(case)
                save()
        after = resources()
        result = analyze(PREP, name)
        result.update(resource_gate_before=before, resource_gate_after=after)
        with (root / f'{name}-result.json').open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        state['rounds'].append({'round': name, **{key: result[key] for key in
            ('round_promotion_passed', 'repeat_required', 'seed_mean_cv', 'gain_vs_control', 'vs_native', 'vs_control')}})
        save()
        return result
    save()
    try:
        primary = run_round('primary')
        confirmation = run_round('confirmation') if primary['repeat_required'] else None
        state['formal_performance_passed'] = primary['round_promotion_passed'] and (
            confirmation is None or confirmation['round_promotion_passed'])
        state['status'] = 'completed_promoted' if state['formal_performance_passed'] else 'completed_not_promoted'
        state['repeat_performed'] = confirmation is not None
    except BaseException as exc:
        state.update(status='stopped_incomplete', error_type=type(exc).__name__, error=str(exc), formal_performance_passed=False)
        raise
    finally:
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({k: v for k, v in state.items() if k != 'source_sha256'}), flush=True)


if __name__ == '__main__':
    main()

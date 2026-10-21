"""Continue saved numerical evidence on the same two physical A100s.

Only independent untimed checkpoint loads use the Torch 2.6 API adapter.
All completed and failed observations from the previous job are preserved.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from continue_pinning_validation import remaining_seconds

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT = PREP / 'pinning-recovery-20261016-01'
SEMANTICS = PREP / 'pinning-semantics-20261016-01'
CODE = ('continue_pinning_recovery.py', 'checkpoint_load_compat.py',
        'pinning_resume_compat_entry.py', 'run_pinning_resume_compat.py',
        'pinning_semantics_entry.py', 'pinning_resume_audit.py', 'run_pinning_semantics.py',
        'verify_pinning_semantics.py', 'verify_pinning_archive.py', 'pinning_key_results.py',
        'pinning_entry.py', 'pinning_candidate.py', 'pinning_full_audit.py',
        'input_evidence.py', 'jiuge7b_bridge.py', 'run_native_case.py', 'native_entry.py')
UUIDS = {'GPU-09ab6992-d722-c828-7bca-934ccb46aa54',
         'GPU-c9db5c06-61c6-01b7-b9aa-20307600482c'}


def file_sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(32 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def resources():
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                    '--format=csv,noheader'], text=True)
    rows = [[v.strip() for v in row.split(',')] for row in gpu.strip().splitlines()]
    if (len(rows) != 2 or {r[0] for r in rows} != UUIDS or apps.strip() or
            any(int(r[2]) > 256 or int(r[3]) for r in rows)):
        raise ValueError('Allocated cards changed or other GPU computation is present')
    return {'gpu': gpu, 'compute_apps': apps, 'cpu_max': Path('/sys/fs/cgroup/cpu.max').read_text(),
            'memory_max': Path('/sys/fs/cgroup/memory.max').read_text(),
            'utc': datetime.now(timezone.utc).isoformat()}


def main():
    if socket.gethostname() != 'sg-huids25-261016-3d89f-default0-0':
        raise ValueError('Wrong recovery allocation')
    ROOT.mkdir(exist_ok=False)
    source = ROOT / 'source'
    source.mkdir()
    hashes = {}
    for name in CODE:
        payload = (PREP / name).read_bytes()
        hashes[name] = hashlib.sha256(payload).hexdigest()
        (source / name).write_bytes(payload)
    state = {'schema': 'jiuge7b-pinning-recovery-controller-v1', 'pid': os.getpid(),
             'status': 'initializing', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'source_sha256': hashes, 'runs': [], 'formal_performance_started': False,
             'originals_location': str(PREP), 'local_large_raw_copy_enabled': False}
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = ROOT / 'controller.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(ROOT / 'controller.json')
    def run(name, argv, reserve=1800):
        if remaining_seconds() < reserve:
            raise TimeoutError('Insufficient remaining free resource; no extension')
        if any(file_sha(PREP / key) != value for key, value in hashes.items()):
            raise ValueError('Frozen recovery source changed')
        state['status'] = name
        save()
        with (ROOT / f'{name}.log').open('x') as stream:
            child = subprocess.Popen([sys.executable, *map(str, argv)], cwd=PREP,
                                     stdout=stream, stderr=subprocess.STDOUT)
            state['active_child_pid'] = child.pid
            save()
            code = child.wait(timeout=max(1, remaining_seconds() - 600))
        state.pop('active_child_pid', None)
        state['runs'].append({'name': name, 'exit_code': code})
        save()
        if code:
            raise ValueError(f'{name} failed with exit code {code}')
    save()
    try:
        state['resource_gate_before'] = resources()
        audit = PREP / 'pinning-validation-20261015-04'
        old = PREP / 'pinning-semantics-20261015-02'
        numeric = json.loads((audit / 'controller.json').read_text())
        if numeric['status'] != 'numeric_runs_completed_pending_archive_and_resume':
            raise ValueError('Existing complete numerical campaign required')
        for rank in (0, 1):
            proof = json.loads((PREP / f'checkpoint-common-compat-gloo-20261016-rank-{rank}.json').read_text())
            if not proof['passed'] or proof['cuda_initialized'] or proof['redirected_calls'] != 1:
                raise ValueError('Actual CPU/Gloo checkpoint compatibility preflight missing')
        SEMANTICS.mkdir(exist_ok=False)
        origins = []
        for name in ('semantics-native', 'semantics-control', 'semantics-optimized', 'checkpoint-native'):
            original, target = old / name, SEMANTICS / name
            shutil.copytree(original, target, copy_function=os.link)
            origins.extend({'source': str(p), 'observed_copy': str(target / p.relative_to(original)),
                            'bytes': p.stat().st_size, 'sha256': file_sha(p)}
                           for p in original.rglob('*') if p.is_file())
        (SEMANTICS / 'baseline-observation-origins.json').write_text(json.dumps({
            'method': 'hardlink complete immutable observations; no checkpoint/tensor-body copies',
            'files': origins}, indent=2) + '\n')
        campaigns = [PREP / f'pinning-validation-20261016-{i:02d}' for i in (1, 2)]
        for index, campaign in enumerate(campaigns):
            run(f'prepare-resume-{index}', [PREP / 'pinning_candidate.py', '--output', campaign])
        checkpoint = old / 'checkpoints/native-step2'
        if (checkpoint / 'latest_checkpointed_iteration.txt').read_text().strip() != '2':
            raise ValueError('Wrong original checkpoint iteration')
        for arm, campaign in zip(('native', 'optimized'), campaigns):
            resources()
            name = f'pinning-{arm}-resume-03'
            argv = [PREP / 'run_pinning_resume_compat.py', '--case', name, '--steps', '3',
                    '--seq-length', '2048', '--pinning-arm', arm, '--pinning-campaign', campaign,
                    '--semantics-root', SEMANTICS / f'resume-{arm}', '--load-dir', checkpoint]
            if arm == 'optimized':
                argv += ['--resume-audit-campaign', audit]
            run(name, argv, reserve=7800 if arm == 'optimized' else 1800)
            resources()
            for rank in (0, 1):
                proof = json.loads((PREP / 'runs' / name / f'rank-{rank}/checkpoint-compat.json').read_text())
                if not proof['run_returned_successfully'] or proof['redirected_calls'] != 1:
                    raise ValueError('Actual real-model checkpoint compatibility was not observed')
        run('verify_semantics', [PREP / 'verify_pinning_semantics.py', '--root', SEMANTICS,
                                '--resume-campaign', audit, '--output', SEMANTICS / 'semantics-verification.json'])
        state['status'] = 'hash_native_checkpoint'
        save()
        checkpoint_files = [{'file': p.relative_to(checkpoint).as_posix(),
                             'bytes': p.stat().st_size, 'sha256': file_sha(p)}
                            for p in sorted(checkpoint.rglob('*')) if p.is_file()]
        (ROOT / 'native-checkpoint-sha256.json').write_text(json.dumps({
            'checkpoint': str(checkpoint), 'files': checkpoint_files,
            'full_checkpoint_body_local_copy': False}, indent=2) + '\n')
        run('verify_primary_raw', [PREP / 'verify_pinning_archive.py', '--root', audit, '--include-resume',
                                   '--output', audit / 'primary-archive-verification.json'], reserve=4200)
        summary = ROOT / 'summary'
        summary.mkdir()
        (summary / 'correctness-completion.json').write_text(json.dumps({
            'full_correctness_resume_and_primary_archive_passed': True,
            'formal_performance_started': False}, indent=2) + '\n')
        # Only closed logs, receipts and source snapshots; large originals stay ACT.
        folders = [audit, SEMANTICS, *campaigns, source, summary]
        folders += [PREP / 'runs' / f'pinning-{arm}-audit-04' for arm in ('native', 'control', 'optimized')]
        folders += [PREP / 'runs' / name for name in ('pinning-semantics-native-02',
                    'pinning-semantics-control-02', 'pinning-semantics-optimized-02',
                    'pinning-native-checkpoint-02', 'pinning-native-resume-02',
                    'pinning-native-resume-03', 'pinning-optimized-resume-03')]
        shutil.copyfile(ROOT / 'native-checkpoint-sha256.json', summary / 'native-checkpoint-sha256.json')
        bundle = PREP / 'pinning-complete-key-results-20261016.zip'
        argv = [PREP / 'pinning_key_results.py', '--root', PREP, '--output', bundle]
        for folder in folders:
            argv += ['--folder', folder]
        run('export_key_results', argv)
        state.update(status='ready_for_local_key_verification', key_results_bundle=str(bundle),
                     full_correctness_resume_and_primary_archive_passed=True)
    except BaseException as exc:
        state.update(status='stopped', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        save()
        print(json.dumps({k: v for k, v in state.items() if k != 'source_sha256'}), flush=True)


if __name__ == '__main__':
    main()

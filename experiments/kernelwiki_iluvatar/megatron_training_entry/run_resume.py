"""Run native and opt-in checkpoint continuations serially through the frozen entry.

Both arms load the same native iteration-three checkpoint and execute original
Megatron iterations four through six. This is a correctness audit, not timing.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
START_ITERATION = 3
FINAL_ITERATION = 6


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def checkpoint_manifest(directory, iteration):
    directory = Path(directory)
    tracker = directory / 'latest_checkpointed_iteration.txt'
    if tracker.read_text(encoding='utf-8').strip() != str(iteration):
        raise ValueError('checkpoint tracker does not identify iteration %d' % iteration)
    required = directory / ('iter_%07d/mp_rank_00/model_optim_rng.pt' % iteration)
    if not required.is_file() or required.stat().st_size == 0:
        raise ValueError('missing nonempty original torch checkpoint: ' + str(required))
    files = [tracker] + sorted((directory / ('iter_%07d' % iteration)).rglob('*'))
    return {str(path.relative_to(directory)).replace('\\', '/'): digest(path)
            for path in files if path.is_file()}


def verify_resume_execution(folder, checkpoint_directory):
    """Require original successful-load text and exact real iteration progress."""
    folder = Path(folder)
    text = (folder / 'stdout.log').read_text(encoding='utf-8', errors='replace')
    pattern = (r'successfully loaded checkpoint from ' + re.escape(str(checkpoint_directory))
               + r'\s+\[[^\n]*\]\s+at iteration\s+3\b')
    load_matches = re.findall(pattern, text)
    progress = [{'iteration': int(step), 'train_iters': int(total), 'consumed_samples': int(samples)}
                for step, total, samples in re.findall(
                    r'iteration\s+(\d+)\s*/\s*(\d+)\s*\|[^\n]*consumed samples:\s*(\d+)', text)]
    expected = [{'iteration': step, 'train_iters': 6, 'consumed_samples': step * 2}
                for step in (4, 5, 6)]
    errors = []
    if len(load_matches) != 1:
        errors.append('expected exactly one successful original checkpoint load at iteration 3')
    if progress != expected:
        errors.append('original training log must report only iterations 4/5/6 and samples 8/10/12')
    try:
        final_files = checkpoint_manifest(folder / 'snapshots', 6)
    except (OSError, ValueError) as error:
        final_files = {}
        errors.append(str(error))
    for label in ('params', 'wgrads'):
        directory = folder / 'snapshots' / label
        observed = {str(path.relative_to(directory)).replace('\\', '/')
                    for path in directory.glob('iter_*/*') if path.is_file()}
        expected_files = {'iter_%07d/mp_rank_00.pth' % step for step in (4, 5, 6)}
        if observed != expected_files:
            errors.append('incorrect native %s continuation file inventory' % label)
    forward_files = {path.name for path in (folder / 'snapshots').glob('model*-forward-*.pt')}
    if forward_files != {'model0-forward-%03d.pt' % index for index in (1, 2, 3)}:
        errors.append('expected exactly three captured real resumed forwards')
    return {'passed': not errors, 'errors': errors, 'successful_load_matches': load_matches,
            'actual_training_progress': progress, 'expected_training_progress': expected,
            'final_checkpoint_files_sha256': final_files,
            'forward_to_training_iteration': {'1': 4, '2': 5, '3': 6}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native_case')
    parser.add_argument('optimized_case')
    parser.add_argument('--megatron-root', type=Path, required=True)
    parser.add_argument('--checkpoint-case', type=Path, required=True)
    args = parser.parse_args()
    names = (args.native_case, args.optimized_case)
    if names[0] == names[1] or any(not name.replace('-', '').replace('_', '').isalnum() for name in names):
        parser.error('two distinct simple case names are required')
    contract = json.loads((ROOT / 'contract.json').read_text(encoding='utf-8'))
    source_case = args.checkpoint_case.resolve()
    source_command = json.loads((source_case / 'command.json').read_text(encoding='utf-8'))
    source_entry = json.loads((source_case / 'entry.json').read_text(encoding='utf-8'))
    if not (source_command.get('mode') == 'native' and source_command.get('audit') is True
            and source_command.get('steps') == 3 and source_command.get('exit_code') == 0
            and source_command.get('contract_sha256') == digest(ROOT / 'contract.json')
            and source_entry.get('status') == 'completed' and source_entry.get('iluvatar_kernels') is False
            and source_entry.get('kernel_adapters') == []):
        parser.error('a completed native three-step audit from the same frozen contract is required')
    seed = source_command.get('seed')
    if not isinstance(seed, int):
        parser.error('native checkpoint audit must record its seed')
    checkpoint_directory = source_case / 'snapshots'
    input_files = checkpoint_manifest(checkpoint_directory, 3)
    input_record = {'case': str(source_case), 'directory': str(checkpoint_directory), 'iteration': 3,
                    'files_sha256': input_files, 'source_command_sha256': digest(source_case / 'command.json'),
                    'source_entry_sha256': digest(source_case / 'entry.json')}
    folders = [ROOT / 'evidence' / name for name in names]
    if any(folder.exists() for folder in folders):
        parser.error('refusing to overwrite either continuation case')
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=False)
    import fcntl
    results = []
    # Hold the common existing experiment lock across both child processes.
    # Other performance/audit runners cannot execute a GPU case between them.
    with (ROOT.parent / 'megatron_cc/.gpu-experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for mode, folder in zip(('native', 'optimized'), folders):
            command = [sys.executable, '-u', str(ROOT / 'launch_pretrain.py'),
                       '--megatron-root', str(args.megatron_root.resolve()), '--corex42-compat',
                       '--entry-evidence', str(folder / 'entry.json'),
                       '--entry-tensor-audit-dir', str(folder / 'snapshots')]
            if mode == 'optimized':
                command.append('--iluvatar-kernels')
            command += contract['training_argv'] + [
                '--train-iters', '6', '--seed', str(seed), '--log-interval', '1',
                '--load', str(checkpoint_directory), '--exit-on-missing-checkpoint',
                '--use-checkpoint-opt-param-scheduler',
                '--save', str(folder / 'snapshots'), '--save-interval', '6',
                '--save-wgrads-interval', '1', '--save-params-interval', '1']
            record = {'case': folder.name, 'mode': mode, 'command': command, 'steps': 3,
                      'resumed_from_iteration': 3, 'total_train_iters': 6, 'seed': seed, 'audit': True,
                      'contract_sha256': digest(ROOT / 'contract.json'),
                      'resume_runner_sha256': digest(__file__), 'source_checkpoint': input_record,
                      'started_utc': datetime.now(timezone.utc).isoformat()}
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='0', MASTER_ADDR='127.0.0.1', MASTER_PORT='29921',
                       RANK='0', WORLD_SIZE='1', LOCAL_RANK='0')
            started = time.perf_counter()
            with (folder / 'stdout.log').open('x') as out, (folder / 'stderr.log').open('x') as err:
                try:
                    record['exit_code'] = subprocess.run(command, stdout=out, stderr=err, env=env,
                        cwd=args.megatron_root, timeout=600).returncode
                except subprocess.TimeoutExpired:
                    record.update(exit_code=124, error='600 second timeout')
                except OSError as error:
                    record.update(exit_code=125, error=str(error))
            record['whole_process_elapsed_seconds'] = time.perf_counter() - started
            record['resume_validation'] = verify_resume_execution(folder, checkpoint_directory)
            record['source_checkpoint_unchanged'] = checkpoint_manifest(checkpoint_directory, 3) == input_files
            record['passed'] = (record['exit_code'] == 0 and record['resume_validation']['passed']
                                and record['source_checkpoint_unchanged'])
            record['finished_utc'] = datetime.now(timezone.utc).isoformat()
            with (folder / 'command.json').open('x', encoding='utf-8') as stream:
                json.dump(record, stream, indent=2)
                stream.write('\n')
            results.append({'case': folder.name, 'passed': record['passed'],
                            'exit_code': record['exit_code'], 'resume_validation': record['resume_validation']})
    print(json.dumps(results, indent=2))
    return 0 if all(result['passed'] for result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())

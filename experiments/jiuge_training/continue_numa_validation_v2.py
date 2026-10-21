"""Reuse completed immutable affinity audit; retry full native checkpoint resume."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from continue_numa_validation import CODE
from continue_pinning_validation import remaining_seconds
from continue_pinning_formal import resources
from numa_allocation_v2 import check

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
EXTRA = ('continue_numa_validation_v2.py', 'numa_allocation_v2.py', 'numa_observation_v2_entry.py',
         'numa_observation_gc_v2.patch', 'run_numa_resume_v2_case.py', 'check_numa_observer_lifetime.py',
         'continue_numa_formal_v2.py', 'curate_cpu_numa_keys_v2.py',
         'cpu-numa-observer-lifetime-regression-20261016.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allocation', type=Path, required=True)
    args = parser.parse_args()
    allocation = check(args.allocation)
    regression = json.loads((PREP / 'cpu-numa-observer-lifetime-regression-20261016.json').read_text())
    if not regression['passed'] or any(hashlib.sha256((PREP / name).read_bytes()).hexdigest() != digest
                                       for name, digest in regression['source_sha256'].items()):
        raise ValueError('Actual weak-reference lifetime regression must match admitted sources')
    if remaining_seconds() < 20000:
        raise TimeoutError('Complete resume, raw and formal time reserve required')
    old = PREP / 'cpu-numa-validation-20261016-01'
    prior = json.loads((old / 'controller.json').read_text())
    # Original source snapshot and actual completed audit are immutable references.
    for name, digest in prior['source_sha256'].items():
        if hashlib.sha256((PREP / name).read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen original source changed: ' + name)
    case = PREP / 'runs/cpu-numa-local8-full-audit-01'
    metadata = json.loads((case / 'case.json').read_text())
    if metadata['exit_code'] or metadata['resource_gate_before'].strip() or metadata['resource_gate_after'].strip():
        raise ValueError('Original affinity audit failed or contended')
    for rank in (0, 1):
        manifest = json.loads((old / f'optimized/rank-{rank}/all-state-manifest.json').read_text())
        done = json.loads((old / f'optimized/rank-{rank}/completion.json').read_text())
        if (len(manifest['tensors']) != 3585 or manifest['comparison_dtype'] != 'CPU float64' or
                manifest['atol'] != 3e-4 or manifest['rtol'] != 1e-3 or not done['passed'] or
                not done['complete_fields'] or done['steps'] != 3 or
                not all(row['passed'] and row['max_abs'] == 0 for row in manifest['tensors'])):
            raise ValueError('Complete successful immutable candidate audit required')
        actual = json.loads((case / f'rank-{rank}/optimizer-state.json').read_text())['optimizers']
        if (len(actual) != 1 or actual[0]['adam_state_entries'] != 199 or actual[0]['min_step'] != 3 or
                actual[0]['max_step'] != 3 or actual[0]['elements_with_both_adam_moments'] != 3834518016):
            raise ValueError('Original actual AdamW update coverage differs')
    root = PREP / 'cpu-numa-validation-20261016-02'
    subprocess.run([sys.executable, PREP / 'pinning_candidate.py', '--output', root], check=True)
    source = root / 'source'; source.mkdir()
    hashes = {}
    for name in (*CODE, *EXTRA):
        data = (PREP / name).read_bytes(); (source / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    shutil.copyfile(args.allocation, root / 'allocation.json')
    state = {'schema': 'jiuge7b-cpu-numa-validation-controller-v2', 'status': 'linking_completed_audit',
             'started_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': hashes, 'runs': [],
             'hostname': socket.gethostname(), 'allocation': allocation, 'formal_performance_passed': False,
             'complete_raw_local_download': False, 'three_step_candidate_audit_rerun': False}
    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temp = root / 'controller.tmp'; temp.write_text(json.dumps(state, indent=2) + '\n'); temp.replace(root / 'controller.json')
    def run(name, argv, reserve=10000):
        if remaining_seconds() < reserve:
            raise TimeoutError('Insufficient remaining free reserve')
        if any(hashlib.sha256((PREP / key).read_bytes()).hexdigest() != digest for key, digest in hashes.items()):
            raise ValueError('Frozen source changed')
        state['status'] = name; save(); print(json.dumps({'event': 'validation_stage_start', 'stage': name}), flush=True)
        with (root / (name + '.log')).open('x') as stream:
            subprocess.run([sys.executable, *map(str, argv)], cwd=PREP, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=max(1, remaining_seconds() - 1800))
        state['runs'].append(name); save(); print(json.dumps({'event': 'validation_stage_complete', 'stage': name}), flush=True)
    save()
    try:
        state['resource_gate_before'] = resources(); save()
        origins = []
        def link(path):
            path = path.resolve(strict=True)
            if not path.is_relative_to(old.resolve(strict=True)) or not path.is_file():
                raise ValueError('Unsafe immutable reference path')
            target = root / path.relative_to(old)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(path, target)
            if (path.stat().st_dev, path.stat().st_ino) != (target.stat().st_dev, target.stat().st_ino):
                raise ValueError('Reference alias must be same immutable inode')
            origins.append({'source': str(path), 'alias': str(target), 'bytes': path.stat().st_size,
                            'same_inode': True, 'body_copied': False})
        for arm in ('native', 'optimized'):
            for rank in (0, 1):
                directory = old / arm / f'rank-{rank}'
                manifest = json.loads((directory / 'all-state-manifest.json').read_text())
                for relative in sorted({chunk['file'] for row in manifest['tensors'] for chunk in row['chunks']}):
                    link(old / relative)
                for name in ('all-state-manifest.json', 'completion.json'):
                    link(directory / name)
        for path in (old / 'semantics-optimized').rglob('*'):
            if path.is_file():
                link(path)
        (root / 'reference-origins.json').write_text(json.dumps({
            'method': 'Same-inode immutable native and completed candidate audit bodies; not a new three-step run',
            'candidate_case': str(case), 'original_controller_sha256': hashlib.sha256((old / 'controller.json').read_bytes()).hexdigest(),
            'frozen_original_source_sha256': prior['source_sha256'],
            'resume_only_change': 'Weak-reference observer releases replaced HDO; binding/math/timing unchanged',
            'files': origins}, indent=2) + '\n')
        resume_candidate = root / 'resume-candidate'
        run('prepare-resume', [PREP / 'pinning_candidate.py', '--output', resume_candidate])
        checkpoint = PREP / 'pinning-semantics-20261015-02/checkpoints/native-step2'
        if (checkpoint / 'latest_checkpointed_iteration.txt').read_text().strip() != '2':
            raise ValueError('Same actual iteration2 checkpoint required')
        resources()
        run('cpu-numa-local8-full-resume-02', [PREP / 'run_numa_resume_v2_case.py', '--case', 'cpu-numa-local8-full-resume-02',
            '--steps', '3', '--seq-length', '2048', '--numa-mode', 'local8', '--validation-stage', 'resume',
            '--pinning-campaign', resume_candidate, '--semantics-root', root / 'resume-optimized',
            '--load-dir', checkpoint, '--resume-audit-campaign', root])
        resources()
        for rank in (0, 1):
            actual = json.loads((PREP / f'runs/cpu-numa-local8-full-resume-02/rank-{rank}/optimizer-state.json').read_text())['optimizers']
            if (len(actual) != 1 or actual[0]['adam_state_entries'] != 199 or actual[0]['min_step'] != 3 or
                    actual[0]['max_step'] != 3 or actual[0]['elements_with_both_adam_moments'] != 3834518016):
                raise ValueError('Actual restored state/update coverage differs')
        run('verify-semantics', [PREP / 'verify_numa_semantics.py', '--root', root, '--output', root / 'semantics-verification.json'])
        run('verify-full-primary-raw', [PREP / 'verify_numa_archive.py', '--root', root, '--output', root / 'full-archive-verification.json'])
        state.update(status='ready_for_local_key_verification', resource_gate_after=resources()); save()
        print(json.dumps({'status': state['status']}), flush=True)
    except BaseException as exc:
        state.update(status='stopped', error_type=type(exc).__name__, error=str(exc)); save(); raise


if __name__ == '__main__':
    main()

"""Admit one full-model update before six fresh cases; preserve frozen93."""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from run_sft_moment_direct_zero_case_20261021 import derive as profile_launcher

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
APPROVAL = PREP/'sft-direct-zero-first-step-approved-20261023.json'


def derive(source, timeout_seconds=1800):
    if not 1 <= timeout_seconds <= 1800:
        raise ValueError('First-update admission has a bounded timeout')
    text = profile_launcher(source)
    changes = (
        ('if args.steps != 35 or args.seq_length != 2048 or not args.capture_windows or args.reference or args.capture_inputs:',
         "if args.steps != 1 or args.seq_length != 2048 or args.capture_windows or args.reference or args.capture_inputs or args.sft_moment_arm != 'local-moments' or args.sft_numa_mode != 'gpu-local8':"),
        ('Fixed 35-step S2048 profile only', 'Fixed one-step S2048 candidate memory admission only'),
        ("str(PREP/'sft_moment_direct_zero_entry_20261021.py')",
         "str(PREP/'sft_direct_zero_first_step_entry_20261023.py')"),
    )
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError('Exact first-update launcher derivation differs: ' + before)
        text = text.replace(before, after)
    # This replacement belongs to the generated native launcher, not model code.
    before = '    changes = ['
    if text.count(before) != 1:
        raise ValueError('Exact native launch change list required')
    text = text.replace(before, before + '\n        ' + repr(('timeout=7200', 'timeout='+str(timeout_seconds))) + ',')
    return text


def admission():
    gate = json.loads(APPROVAL.read_text())
    if (gate['hostname'] != socket.gethostname() or gate['max_charge_points'] != 0
            or gate['startup_limit_seconds'] != 10800):
        raise ValueError('Exact authorized three-hour zero-cost allocation required')
    remaining = int(gate['stop_before_epoch'] - time.time() - 180)
    if remaining < 300:
        raise TimeoutError('Preserve shutdown reserve; no update may start')
    original = json.loads((PREP/'sft-moment-direct-zero-source-manifest-v3-20261021.json').read_text())['files']
    if any(gate['source_files_sha256'].get(name) != sha for name, sha in original.items()):
        raise ValueError('All original93 sources must remain in the new allocation gate')
    for name, sha in gate['source_files_sha256'].items():
        if Path(name).name != name or hashlib.sha256((PREP/name).read_bytes()).hexdigest() != sha:
            raise ValueError('Actual approved source differs: ' + name)
    for name in ('sft_direct_zero_first_step_entry_20261023.py', 'run_sft_direct_zero_first_step_20261023.py'):
        if name not in gate['source_files_sha256']:
            raise ValueError('First-step orchestration must also be frozen')
    runtime = PREP/('sft-moment-direct-zero-runtime-'+socket.gethostname())/'summary.json'
    receipt = json.loads(runtime.read_text())
    if not receipt['passed'] or receipt['GPU_used'] or not receipt['allocation_guard_active']:
        raise ValueError('Live eight-GiB guarded CPU admission must pass first')
    root = Path('/sys/fs/cgroup')
    quota, period = map(int, (root/'cpu.max').read_text().split())
    memory = int((root/'memory.max').read_text())
    if (not 16 <= quota/period < 17 or memory != gate['memory_max_bytes']
            or not 300*1024**3 <= memory <= 300*1024**3+1024**2):
        raise ValueError('Fixed16CPU/300GiB allocation required')
    cards = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=20)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name', '--format=csv,noheader'], text=True, timeout=20)
    if cards.split() != gate['ordered_gpu_uuids'] or len(cards.split()) != 2 or apps.strip():
        raise ValueError('Exact two assigned cards and no other compute process required')
    return min(1800, remaining)


def verify(folder):
    case = json.loads((folder/'case.json').read_text())
    if case['exit_code'] != 0:
        raise ValueError('First update did not exit successfully')
    ranks = {}
    for rank in (0, 1):
        observer = json.loads((folder/f'sft-numa-observer-rank-{rank}.json').read_text())
        events = json.loads((folder/f'sft-local-moments-rank-{rank}.json').read_text())['moment_events']
        progress = [json.loads(row) for row in (folder/f'moment-initialization-memory-rank-{rank}.jsonl').read_text().splitlines()]
        before = json.loads((folder/f'memory-before-first-update-rank-{rank}.json').read_text())
        after = json.loads((folder/f'sft-numa-after-first-update-rank-{rank}.json').read_text())['memory_budget_snapshot']
        objective = json.loads((folder/f'sft-trace-objective-rank-{rank}.json').read_text())
        if observer['steps'] != 1 or not objective['run_returned']:
            raise ValueError('Both native TP ranks must complete exactly one update')
        if objective['data_receipt_sha256'] != '1e30f6af3c21d31ee6fae117bfec9c25120bb1b14fe86d815d4f059fdc7a8d08':
            raise ValueError('Frozen answer-only SFT data receipt differs')
        if len(events) != 398 or any(e['initialization_path'] != 'direct_local_zero'
                                     or e['native_source_moment_created'] or e['copied_source_moment_bytes'] for e in events):
            raise ValueError('Every fresh moment must use the direct-zero path')
        if progress[-1].get('created_moment_objects') != 398:
            raise ValueError('Complete moment initialization telemetry required')
        def oom_count(snapshot):
            return dict(row.split() for row in snapshot['cgroup']['memory.events'].splitlines())
        old, new = oom_count(before), oom_count(after)
        for field in ('oom', 'oom_kill', 'oom_group_kill'):
            if int(new.get(field, '0')) != int(old.get(field, '0')):
                raise ValueError('A cgroup OOM event occurred during the update')
        ranks[str(rank)] = {'updates':1, 'moment_objects':398, 'copied_source_moment_bytes':0,
                            'before':before, 'after':after}
    receipt = {'schema':'jiuge7b-direct-zero-one-update-memory-admission-v1',
               'passed':True, 'ranks':ranks, 'only_first_update_memory_admission':True,
               'full_SFT_correctness_acceptance':False, 'formal_performance_evidence':False,
               'six_fresh_cases_still_required':True, 'raw_storage':'ACT shared Ceph'}
    with (folder/'first-step-memory-admission.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({'passed':True, 'receipt':str(folder/'first-step-memory-admission.json')}))


def main():
    limit = admission()
    text = derive((PREP/'run_sft_trace_case_20261020.py').read_bytes(), limit)
    case_name = sys.argv[sys.argv.index('--case')+1]
    namespace = {'__name__':'<first-step-launcher>', '__file__':__file__}
    exec(compile(text, '<one-update-native-SFT-launcher>', 'exec'), namespace)
    try:
        namespace['generated_main']()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    verify(PREP/'runs'/case_name)


if __name__ == '__main__':
    main()

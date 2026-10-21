"""Screen NUMA policies on fresh pinned SFT runs, never claim formal speedup."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import pinning_key_results as keys
from sft_numa_placement_20261021 import discover, build_plan

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT = PREP/'sft-numa-diagnostic-20261021-01'
GATE = PREP/'sft-numa-approved-20261021.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    approval = json.loads(GATE.read_text())
    if (approval['hostname'] != socket.gethostname() or approval['max_charge_points'] != 0
            or approval['startup_limit_seconds'] != 10800):
        raise ValueError('Exact allocated zero-cost job gate required')
    if time.time() >= approval['stop_before_epoch']-180:
        raise TimeoutError('Expired allocation; do not restart a closed campaign')
    ROOT.mkdir(exist_ok=False)
    source = ROOT/'source'
    source.mkdir()
    for name, sha in approval['source_files_sha256'].items():
        if digest(PREP/name) != sha:
            raise ValueError('Frozen source changed: '+name)
        shutil.copyfile(PREP/name, source/name)
    shutil.copyfile(GATE, ROOT/'approval.json')
    state = {'schema': 'jiuge7b-sft-numa-diagnostic-controller-v1', 'status': 'initializing',
             'started_utc': datetime.now(timezone.utc).isoformat(), 'cases': [],
             'hostname': socket.gethostname(), 'max_charge_points': 0,
             'stop_before_epoch': approval['stop_before_epoch'],
             'formal_performance_evidence': False, 'raw_originals_downloaded': False}

    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        tmp = ROOT/'controller.tmp'
        tmp.write_text(json.dumps(state, indent=2)+'\n')
        tmp.replace(ROOT/'controller.json')

    def run(name, arguments):
        left = int(approval['stop_before_epoch']-time.time()-180)
        if left < 120:
            raise TimeoutError('Independent free-window shutdown reserve reached')
        with (ROOT/(name+'.log')).open('x') as stream:
            subprocess.run([sys.executable, *map(str, arguments)], cwd=PREP,
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=left)

    def resources():
        cards = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu', '--format=csv,noheader'],
            text=True, timeout=20)
        apps = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_memory', '--format=csv,noheader'],
            text=True, timeout=20)
        uuids = [row.split(',')[0].strip() for row in cards.strip().splitlines()]
        if uuids != approval['ordered_gpu_uuids'] or len(uuids) != 2 or apps.strip():
            raise ValueError('Assigned GPU pair changed or foreign compute is present')
        if shutil.disk_usage(PREP).free < 100*1024**3:
            raise ValueError('Preserve persistent storage reserve')
        return {'cards': cards, 'compute_apps': apps, 'utc': datetime.now(timezone.utc).isoformat()}

    save()
    try:
        state['resource_gate_before'] = resources()
        state['cpu_max'] = Path('/sys/fs/cgroup/cpu.max').read_text().strip()
        state['memory_max'] = int(Path('/sys/fs/cgroup/memory.max').read_text())
        quota, period = map(int, state['cpu_max'].split())
        if (not 16 <= quota/period < 17
                or state['cpu_max'] != approval['cpu_max']
                or state['memory_max'] != approval['memory_max_bytes']
                or not 300*1024**3 <= state['memory_max'] <= 300*1024**3+1024**2):
            raise ValueError('Fixed16CPU/300GiB allocation required')
        plan = build_plan(discover())
        plan_path = ROOT/'affinity-plan.json'
        with plan_path.open('x') as stream:
            json.dump(plan, stream, indent=2)
        state['gpu_topology'] = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True)
        order = [('unbound', 'a1'), ('gpu-local8', 'b1')]
        if 'socket-split8' in plan['policies']:
            order += [('socket-split8', 'c1'), ('socket-split8', 'c2')]
        order += [('gpu-local8', 'b2'), ('unbound', 'a2')]
        state['ordered_policies'] = order
        campaign = ROOT/'candidate'
        run('candidate-prepare', [PREP/'pinning_candidate.py', '--output', campaign])
        reference = None
        folders = []
        for policy, tag in order:
            if approval['stop_before_epoch']-time.time() < 1200:
                raise TimeoutError('Insufficient reserve for another complete case')
            for name, sha in approval['source_files_sha256'].items():
                if digest(PREP/name) != sha:
                    raise ValueError('Frozen source changed before launch: '+name)
            gate_before = resources()
            case = f'sft-numa-20261021-{tag}-{policy}'
            folder = PREP/'runs'/case
            isolated = ROOT/('candidate-'+tag)
            isolated.mkdir()
            for name in ('native_hybrid_optimizer.py', 'candidate_hybrid_optimizer.py', 'candidate.patch', 'candidate.json'):
                shutil.copyfile(campaign/name, isolated/name)
            state.update(status='capturing_'+tag+'_'+policy, active_case=case)
            save()
            run(case, [PREP/'run_sft_numa_case_20261021.py', '--case', case,
                       '--steps', '35', '--seq-length', '2048', '--pinning-arm', 'optimized',
                       '--pinning-campaign', isolated, '--capture-windows',
                       '--sft-numa-mode', policy, '--sft-numa-plan', plan_path])
            run(case+'-completion', [PREP/'verify_trace_case_completion.py', '--case-root', folder,
                                    '--steps', '35', '--sft', '--profile'])
            completion = json.loads((folder/'completion-verification.json').read_text())
            rows = json.loads((folder/'sft-supervision-rank-0.json').read_text())['rows']
            losses = completion['native_losses']
            if len(rows) != 35 or len(losses) != 35:
                raise ValueError('All35 paired samples and logged losses required')
            if reference is None:
                reference = (rows, losses)
            elif (rows, losses) != reference:
                raise ValueError('Placement changed paired SFT sample/mask counts or logged losses')
            for rank in (0, 1):
                binding = json.loads((folder/f'sft-numa-binding-rank-{rank}.json').read_text())
                pin = json.loads((isolated/f'optimized-rank-{rank}-actual-pinning-init-1.json').read_text())
                objective = json.loads((folder/f'sft-trace-objective-rank-{rank}.json').read_text())
                rank_rows = json.loads((folder/f'sft-supervision-rank-{rank}.json').read_text())['rows']
                if (binding['allowed_after'] != plan['policies'][policy][rank]
                        or binding['plan_sha256'] != digest(plan_path)
                        or pin['parameter_count'] != 199 or pin['pinned_count'] != 199
                        or not objective['run_returned']
                        or objective['range_call_counts']['jiuge/cpu_adamw_math'] != 35*199
                        or objective['data_receipt_sha256'] != '1e30f6af3c21d31ee6fae117bfec9c25120bb1b14fe86d815d4f059fdc7a8d08'
                        or rank_rows != rows):
                    raise ValueError('Actual pinning, affinity, SFT or AdamW coverage gate failed')
            item = {'case': case, 'policy': policy, 'tag': tag,
                    'gate_before': gate_before, 'gate_after': resources()}
            state['cases'].append(item)
            folders.append(folder)
            save()
        state.update(status='analyzing', active_case=None)
        save()
        for item, folder in zip(state['cases'], folders):
            for script, name in (
                ('review_native_trace.py', 'trace-analysis.json'),
                ('inspect_trace_communication.py', 'communication-analysis.json'),
                ('analyze_sft_paired_trace_20261020.py', 'symmetric-mechanism.json')):
                run(item['case']+'-'+name, [PREP/script, '--case-root', folder, '--output', folder/name])
            run(item['case']+'-archive', [PREP/'archive_case.py', '--case-root', folder, '--create', '--snapshot-analysis'])
            run(item['case']+'-archive-verify', [PREP/'archive_case.py', '--case-root', folder, '--verify',
                                              '--receipt', ROOT/(item['case']+'-archive-verification.json')])
        state.update(status='completed', paired35_samples_masks_logged_losses_equal=True,
                     resource_gate_after=resources(), verified_raw_traces=4*len(folders),
                     finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        # Include frozen shell bootstraps from the start. Never mutate the old
        # exporter file, an old closed archive, or a previous key receipt.
        keys.KEY_SUFFIXES = keys.KEY_SUFFIXES | {'.sh'}
        bundle = PREP/'sft-numa-key-results-20261021.zip'
        catalog = keys.export(PREP, [ROOT, *folders], bundle, 32*1024**2)
        with (ROOT/'final-key-bundle.json').open('x') as stream:
            json.dump({'path': str(bundle), 'bytes': bundle.stat().st_size,
                       'sha256': digest(bundle), 'key_files': catalog['key_file_count'],
                       'key_uncompressed_bytes': catalog['key_uncompressed_bytes'],
                       'extra_key_suffixes': ['.sh']}, stream, indent=2)
    except BaseException as exc:
        state.update(status='stopped', error_type=type(exc).__name__, error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()

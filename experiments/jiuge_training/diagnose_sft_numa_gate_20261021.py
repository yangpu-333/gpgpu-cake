"""Post-hoc CPU review of the preserved successful run and failed controller."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pinning_key_results as keys
from sft_tp_reader_gate_20261021 import check_rank_rows

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
ORIGINAL = PREP/'sft-numa-diagnostic-20261021-01'
ROOT = PREP/'sft-numa-failure-audit-20261021-01'
CASE = PREP/'runs/sft-numa-20261021-a1-unbound'


def read(path):
    return json.loads(path.read_text())


def main():
    ROOT.mkdir(exist_ok=False)
    prior = read(ORIGINAL/'approval.json')
    for name, sha in prior['source_files_sha256'].items():
        if hashlib.sha256((PREP/name).read_bytes()).hexdigest() != sha:
            raise ValueError('Original53 frozen sources changed: '+name)
    native = read(PREP/'source_manifest.json')
    sources = {}
    for name in ('pretrain_gpt.py', 'megatron/core/utils.py'):
        path = PREP/'megatron-native-5be9626'/name
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha != native['files'][name]:
            raise ValueError('Native reader/broadcast source changed')
        sources[name] = {'sha256': sha}
    completion = read(CASE/'completion-verification.json')
    if not completion['passed'] or read(CASE/'case.json')['exit_code']:
        raise ValueError('Original training did not successfully finish')
    reader = read(CASE/'sft-supervision-rank-0.json')['rows']
    checks = []
    for rank in (0, 1):
        row = read(CASE/f'sft-supervision-rank-{rank}.json')['rows']
        checks.append(check_rank_rows(rank, row, reader))
    diagnosis = {'schema': 'jiuge7b-sft-numa-controller-failure-audit-v1',
                 'utc': datetime.now(timezone.utc).isoformat(),
                 'original_controller_status': read(ORIGINAL/'controller.json')['status'],
                 'original_training_exit_code': 0,
                 'actual_non_skipped_steps_each_rank': 35,
                 'original_trace_files': 4, 'original53_sources_unchanged': True,
                 'reason': 'Controller incorrectly required the native TP receiver to have its own35 dataset-read logs',
                 'native_TP_reader_rank': 0, 'native_sources': sources,
                 'metadata_role_checks': checks,
                 'rank0_sample_metadata': reader,
                 'full_receiver_raw_array_equality_checked_by_this_diagnosis': False,
                 'new_GPU_work_started': False, 'formal_performance_passed': False}
    with (ROOT/'diagnosis.json').open('x') as stream:
        json.dump(diagnosis, stream, indent=2)
    for script, name in (('review_native_trace.py', 'trace-analysis.json'),
                         ('inspect_trace_communication.py', 'communication-analysis.json'),
                         ('analyze_sft_paired_trace_20261020.py', 'symmetric-mechanism.json')):
        with (ROOT/(name+'.log')).open('x') as stream:
            subprocess.run([sys.executable, str(PREP/script), '--case-root', str(CASE),
                            '--output', str(CASE/name)], stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=180)
    for label, arguments in [('create', ['--create', '--snapshot-analysis']),
                             ('verify', ['--verify', '--receipt', str(ROOT/'ACT-originals-verification.json')])]:
        with (ROOT/('archive-'+label+'.log')).open('x') as stream:
            subprocess.run([sys.executable, str(PREP/'archive_case.py'), '--case-root', str(CASE), *arguments],
                           stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=180)
    keys.KEY_SUFFIXES = keys.KEY_SUFFIXES | {'.sh'}
    bundle = PREP/'sft-numa-failure-key-results-20261021.zip'
    catalog = keys.export(PREP, [ORIGINAL, ROOT, CASE], bundle, 32*1024**2)
    receipt = {'path': str(bundle), 'bytes': bundle.stat().st_size,
               'sha256': hashlib.sha256(bundle.read_bytes()).hexdigest(),
               'key_files': catalog['key_file_count'], 'key_uncompressed_bytes': catalog['key_uncompressed_bytes'],
               'raw_trace_bodies_included': False, 'GPU_used': False}
    with (ROOT/'key-bundle.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

"""CPU check of preserved build, controller and numerical rejection evidence."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / 'evidence/swiglu-stage29'
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
read = lambda path: json.loads(path.read_text(encoding='utf-8'))


def verify():
    failed = read(BASE / 'cpp-build-002-receipt.json')
    assert failed['status'] == 'failed'
    assert failed['sources'] == {name: sha(BASE / name) for name in ('swiglu-002.cpp', 'swiglu-002.cu')}
    assert failed['build_script_sha256'] == sha(BASE / 'build_cpp.py')
    assert failed['extra_cuda_cflags'] == ['-O3', '--fmad=false']
    assert "unsupported option '--fmad=false'" in (BASE / 'build-002.stdout.log').read_text()
    assert 'NoneType' in (BASE / 'probe-003-v2.stderr.log').read_text()
    assert (BASE / 'probe_candidate-02.py').read_text() == (ROOT / 'swiglu/probe_candidate-02.py').read_text()
    probe = read(BASE / 'probe-003-v3.json')
    assert probe['status'] == 'rejected' and probe['performance'] == []
    assert probe['candidate_sha256'] == sha(BASE / 'candidate-003.py')
    assert probe['verifier_sha256'] == sha(BASE / 'probe_candidate-03.py')
    assert (BASE / 'probe_candidate-03.py').read_text() == (ROOT / 'swiglu/probe_candidate-03.py').read_text()
    assert len(probe['checks']) == 37 and len(probe['guard_checks']) == 4
    assert all(item['passed'] for item in probe['guard_checks'])
    failed_cases = sum(not all(check['passed'] for check in row['checks']) for row in probe['checks'])
    assert failed_cases == 19
    assert all(row['input_unchanged'] and row['upstream_unchanged'] for row in probe['checks'])
    built = read(BASE / 'cpp-build-003-receipt.json')
    assert built['status'] == 'built'
    assert built['sources'] == {name: sha(BASE / name) for name in ('swiglu-003.cpp', 'swiglu-003.cu')}
    assert built['build_script_sha256'] == sha(BASE / 'build_cpp-02.py')
    assert built['extra_cflags'] == built['extra_cuda_cflags'] == ['-O3', '-ffp-contract=off']
    assert probe['extension']['receipt_sha256'] == sha(BASE / 'cpp-build-003-receipt.json')
    assert probe['extension']['artifact_sha256'] == built['artifact_sha256']
    assert probe['extension']['sources'] == built['sources']
    assert probe['extension']['build_script_sha256'] == built['build_script_sha256']
    assert not (BASE / 'audit-summary-main-003-complete.json').exists()
    assert not (BASE / 'performance-main-003-complete.json').exists()
    return {'status': 'verified', 'build002': 'rejected SDK flag',
            'controller02': 'lazy module compatibility error',
            'candidate003': 'numerical rejection', 'failed_cases': failed_cases,
            'no_candidate003_training_or_performance': True, 'gpu_rerun': False}


if __name__ == '__main__':
    print(json.dumps(verify()))

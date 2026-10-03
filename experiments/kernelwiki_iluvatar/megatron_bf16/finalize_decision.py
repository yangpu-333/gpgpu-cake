"""Summarize sealed BF16 GPU receipts without reinterpreting their gates."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'evidence'


def read(name):
    path = EVIDENCE / name
    return json.loads(path.read_text()), hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    target = EVIDENCE / 'decision.json'
    assert not target.exists()
    scopes = {}
    receipts = {}
    for scope in ('main', 'wide', 'deep'):
        r8, h8 = read(f'0008-{scope}-performance.json')
        r9, h9 = read(f'0009-{scope}-performance.json')
        control, hc = read(f'0009-{scope}-controlled.json')
        suffix = '03' if scope == 'main' else '01'
        audit8, ha8 = read(f'0008-{scope}-comparison-01.json')
        audit9, ha9 = read(f'0009-{scope}-comparison-{suffix}.json')
        assert all(r['status'] == 'validated' for r in (r8, r9, control))
        assert audit8['passed'] and audit9['passed']
        assert all(t['exact_equal'] for t in audit9['tensor_comparisons'])
        assert not control['all_seeds_faster_than_control']
        scopes[scope] = {
            'model': r8['model'], 'numerical_verification_passed': True,
            'candidate8_checks': audit8['check_summary'],
            'candidate9_checks': audit9['check_summary'],
            'candidate9_all_compared_tensors_bitwise_equal': True,
            'preferred_candidate': '0008',
            'candidate8_initial_gain_percent': r8['throughput_gain_percent'],
            'candidate9_gain_percent': r9['throughput_gain_percent'],
            'candidate9_vs_control_gain_percent': (control['candidate_vs_control_geomean'] - 1) * 100,
            'throughput_gain_percent': (control['native_geomean'] / control['candidate_vs_control_geomean'] - 1) * 100,
            'throughput_gain_scope': '0008 control vs native in the later same-trial three-arm batch',
            'candidate9_all_seeds_faster_than_control': False,
            'selection_reason': '0009 passes numerical gates but has mixed seed gains and no stable universal improvement over 0008.'
        }
        receipts[scope] = dict(performance8=h8, performance9=h9, controlled=hc, audit8=ha8, audit9=ha9)
    result = {
        'status': 'completed', 'recorded_at': '2026-10-03',
        'completion_scope': 'GPU optimization and sealed correctness/performance protocol; repository delivery and raw-data backup have separate receipts.',
        'baseline': 'Pinned original Megatron pretrain_gpt.py with native MockGPTDataset, local Torch, DDP and SGD on one BI-V150; shared explicit CoreX4.2 compatibility.',
        'precision': 'Actual BF16 model weights, FP32 gradient accumulation and SGD master weights; no active CUDA autocast.',
        'candidate7': {'status': 'rejected', 'operator_cases_passed': 504, 'formal_checks_failed': 5},
        'candidate8': {'status': 'default', 'operator_cases_passed': 504, 'forward': 'Native Torch ordering; TP1 all-reduce identities omitted', 'backward': 'Triton onehot-subtract and upstream multiply'},
        'candidate9': {'status': 'numerically_verified_experimental', 'operator_cases_passed': 504, 'backward': 'Additionally stores BF16 dlogits in the same Triton backward kernel'},
        'scopes': scopes, 'receipt_sha256': receipts,
        'successful_timed_processes': 45, 'steps_per_timed_process': 120,
        'total_timed_steps': 5400, 'discarded_startup_steps_per_process': 20,
        'timing': 'Original GPU-synchronized time.time intervals; ten interval10 means ending30..120; no tensor snapshots, profiling, checkpoint or evaluation while timing.',
        'numerical_scope': 'Separate three-update audits of per-token loss, all gradients, BF16 model parameters and FP32 optimizer master parameters. Not full-tensor audits of all5400 timed steps.',
        'performance_limit': 'Three seeds are observations, not confidence intervals. Initial 0008 two-arm and later 0009 three-arm batches have different storage roots; only same-trial controls compare the implementations.',
        'remaining': ['Real production model/data and convergence', 'AdamW and training/finetuning tasks', 'Multi-GPU/TP>1', 'Native BF16 RMSNorm rounding before fusion', 'Gradient L2 reduction hotspots'],
        'nvidia_provenance': 'Original Hopper/Blackwell pages and hardware claims retain their original architecture scope.'
    }
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'scopes': list(scopes), 'timed_steps': 5400}))


if __name__ == '__main__':
    main()

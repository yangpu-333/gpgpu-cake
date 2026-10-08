"""Seal stage32 decisions from complete original raw archives and the CPU verifier."""
import json
from pathlib import Path
from types import SimpleNamespace
import verify_receipts as verifier


BASE = Path(__file__).resolve().parent / 'evidence'


def write_new(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('immutable sealed record changed: ' + str(path))
    else:
        path.write_bytes(data)


def main():
    rounds = []
    for model in ('wide', 'deep'):
        args = SimpleNamespace(model=model, version='003', base=BASE,
                               audit_suffix='complete02', run_suffix='complete02',
                               foundations_only=False, require_complete=True)
        proof = verifier.verify(args)
        path = BASE / ('cpu-verify-' + model + '-raw.json')
        write_new(path, proof)
        perf = proof['performance']
        rounds.append({'model_name': model, 'version': '003', 'audit_suffix': 'complete02',
                       'run_suffix': 'complete02', 'verification_path': path.name,
                       'verification_sha256': verifier.sha(path),
                       'candidate_sha256': proof['candidate_sha256'],
                       'contract_sha256': proof['contract_sha256'],
                       'performance_sha256': perf['sha256'], 'promoted': perf['promoted'],
                       'vs_control': perf['vs_control'], 'vs_native': perf['vs_native'],
                       'comparisons': proof['comparisons'],
                       'snapshot_files_missing': proof['snapshot_files_missing']})
    accepted = [row['model_name'] for row in rounds if row['promoted']]
    decision = {'schema': 'frozen-attention-stage32-sealed-decision-v1',
                'date': '2026-10-08', 'status': 'completed', 'candidate_versions': ['003'],
                'promoted': bool(accepted), 'promoted_models': accepted, 'rounds': rounds,
                'promotion_rule': 'strictly >1% geometric throughput over same-round control and all three seeds faster',
                'control': 'CE0008 + gradnorm001; native compiled SwiGLU, native RMSNorm, default fused causal softmax=True',
                'optimized': 'same control plus unchanged frozen Attention003',
                'code_retuned': False, 'raw_backups_complete': True,
                'scope': 'Only these two fixed single BI-V150 BF16 mock-data GPT shapes, FP32 gradients/SGD masters, '
                         'TP/PP/CP1,120 timed steps per arm/first20 excluded. Stage31 has a different control including '
                         'SwiGLU004 and remains separately scoped. No production, convergence or multiGPU result.'}
    write_new(BASE / 'decision.json', decision)
    print(json.dumps({'status': decision['status'], 'promoted_models': accepted, 'rounds': rounds}))


if __name__ == '__main__':
    main()

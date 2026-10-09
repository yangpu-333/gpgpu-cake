"""Finalize stage35 only after complete original-byte backup and CPU verification."""
import json
from pathlib import Path
import statistics
import verify_receipts as v

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE/'evidence'


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    output = EVIDENCE/'decision.json'
    v.require(not output.exists(), 'immutable final decision already exists')
    receipt_path = EVIDENCE/'raw-backup-moe-final-receipt.json'
    receipt = v.read(receipt_path)
    v.require(receipt['status']=='verified' and receipt['through_candidate']=='004'
              and receipt['gpu_timing_finished_acknowledged'] is True
              and receipt['remote_deleted'] is False, 'complete final raw backup required')
    v.require(receipt['inventory_sha256']==v.sha(EVIDENCE/'raw-inventory-moe-final.json')
              and receipt['backup_script_sha256']==v.sha(HERE/'backup_raw.py'), 'backup provenance differs')
    inventory=v.read(EVIDENCE/'raw-inventory-moe-final.json')
    v.require(receipt['files']==inventory and receipt['verified_files']==len(inventory)==384,
              'complete immutable inventory differs')
    unique={row['sha256']:row['bytes'] for row in inventory}
    v.require(receipt['raw_bytes']==sum(row['bytes'] for row in inventory)
              and receipt['unique_files']==len(unique)==227
              and receipt['unique_bytes']==sum(unique.values()), 'complete raw byte counts differ')
    for row in receipt['files']:
        path = EVIDENCE/'raw-backup'/row['path']
        v.require(path.stat().st_size==row['bytes'] and v.sha(path)==row['sha256'], 'original raw backup differs')
    models = {}
    proofs = {}
    for model in ('long', 'experts'):
        proof = v.verify(model, require_raw=True)
        base = HERE/'models'/model/'evidence'
        performance = v.read(base/'performance-004.json')
        cv = {arm: statistics.stdev(pair['cases'][arm]['mean_iteration_ms'] for pair in performance['pairs']) /
                   statistics.mean(pair['cases'][arm]['mean_iteration_ms'] for pair in performance['pairs'])
              for arm in ('native', 'control')}
        v.require(not any(value>=0.10 for value in cv.values()), 'predeclared complete repeat is required')
        losses = proof['loss_print_comparisons']
        proof.update(candidate_sha256=v.CANDIDATE_SHA, performance_sha256=v.sha(base/'performance-004.json'),
                     backup_receipt_sha256=v.sha(receipt_path), seed_mean_cv=cv, loss_print_comparisons=losses)
        proofs[model]=proof
        models[model] = {'model': performance['model'],
                         'performance_sha256': v.sha(base/'performance-004.json'),
                         'promoted': proof['promoted'], 'vs_control': proof['vs_control'],
                         'vs_native': proof['vs_native'], 'comparisons': proof['comparisons'],
                         'seed_mean_cv': cv, 'repeat_required': False, 'loss_print_comparisons': losses}
    # Verify both models before creating either final proof.
    for model,proof in proofs.items():
        proof_path=EVIDENCE/('cpu-verify-'+model+'-raw.json')
        write_new(proof_path,proof)
        models[model]['proof_sha256']=v.sha(proof_path)
    result = {'schema': 'moe-heldout-stage35-final-decision-v1', 'date': '2026-10-09', 'status': 'complete',
              'candidate_version': '004', 'candidate_sha256': v.CANDIDATE_SHA, 'retuned': False,
              'models': models, 'controllers_manifest_sha256': v.sha(HERE/'models/controllers.json'),
              'predeclared_plan_sha256': v.sha(HERE/'plan.json'), 'reader_sha256': v.sha(HERE/'verify_receipts.py'),
              'sealer_sha256': v.sha(Path(__file__)), 'raw_backup': {
                  'receipt_sha256': v.sha(receipt_path), 'verified_files': receipt['verified_files'],
                  'raw_bytes': receipt['raw_bytes'], 'unique_files': receipt['unique_files'],
                  'unique_bytes': receipt['unique_bytes'], 'remote_deleted': False},
              'formal_processes': 18, 'timed_steps': 2160, 'retained_steps': 1800,
              'promotion_rule': 'same-round control geometric ratio>1.01 and all3seeds faster',
              'metric': 'unchanged native GPU-synchronized time.time interval10;120steps/first20excluded',
              'scope': 'Single BI-V150,4L/H512/FFN2048/V1024/B2, BF16/FP32-SGD masters, mock/dropout0/TP-PP-CP-EP1. '
                       'Experts changes sequence and expert count jointly;3-update tensor gates and12 printed loss intervals '
                       'do not prove120-step all-tensor equivalence or convergence. No production/GroupedGEMM/multiGPU/AdamW/SFT/RL claim.',
              'gpu_arithmetic_replay_in_cpu_or_ci': False}
    write_new(output, result)
    print(json.dumps({k: result[k] for k in ('status','retuned','models','raw_backup')}))


if __name__ == '__main__':
    main()

"""Seal the fresh collection without replacing any lost old-Pod inventory/result."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    target=HERE/'evidence/decision-retry01-01.json'
    assert not target.exists()
    imported=HERE/'evidence/completed-formal-import-retry01-01.json'
    proof=read(imported)
    assert proof['status']=='exact_metadata_and_full_raw_verified'
    backup_path=HERE/'evidence/raw-http-backup-retry01-01-receipt.json'
    backup=read(backup_path)
    assert backup['status']=='all_384_originals_byte_verified' and backup['complete_raw_backup'] is True
    assert len(backup['files'])==384 and backup['original_bytes']==22291589014
    assert backup['unique_objects']==227 and backup['unique_bytes']==9978190961
    models={}
    processes=0
    for model,result in proof['results'].items():
        assert model in ('long','experts') and result['primary']['original_byte_coverage_complete'] is True
        wrapper_path=HERE/'evidence/retry01-metadata/root'/('browser-formal-'+model+'-retry01-primary-01.json')
        wrapper=read(wrapper_path)
        assert wrapper['status']=='passed' and wrapper['complete_local_raw_proof_sha256']==sha(backup_path)
        performance_path=HERE/'models-retry01'/model/'evidence/performance-004.json'
        performance=read(performance_path)
        assert sha(performance_path)==wrapper['performance_sha256']
        cv={arm:statistics.stdev(p['cases'][arm]['mean_iteration_ms'] for p in performance['pairs'])/
                statistics.mean(p['cases'][arm]['mean_iteration_ms'] for p in performance['pairs'])
            for arm in ('native','control','optimized')}
        assert cv==wrapper['seed_mean_cv']
        assert result['repeat_required']==any(cv[arm]>=.1 for arm in ('native','control'))
        for round_name in ('primary','confirmation') if result['repeat_required'] else ('primary',):
            verified=result[round_name]
            assert verified['original_byte_coverage_complete'] is True
            assert all(r['native_control_equal'] and r['native_optimized_equal'] for r in verified['loss_print_comparisons'])
            processes+=9
        models[model]={'model':performance['model'],'primary':result['primary'],
                       'seed_mean_cv':cv,'repeat_required':result['repeat_required'],
                       'promoted':result['promoted_with_repeat_policy'],
                       'performance_sha256':sha(performance_path),'wrapper_sha256':sha(wrapper_path)}
        if result['repeat_required']:models[model]['confirmation']=result['confirmation']
    assert set(models)=={'long','experts'}
    record={'schema':'moe-heldout-stage35-fresh-retry01-decision-v1','created_utc':datetime.now(timezone.utc).isoformat(),
            'status':'complete','candidate_sha256':'b9a2ca76c294481e38e2422332ccb821224bd64b8bb7cd61d972a5e2a4a9133e',
            'retuned':False,'old_collection_or_inventory_overwritten':False,'models':models,
            'formal_processes':processes,'timed_steps':processes*120,'retained_steps':processes*100,
            'complete_raw_backup':True,'raw_backup_receipt_sha256':sha(backup_path),
            'formal_import_sha256':sha(imported),'reader_sha256':sha(HERE/'verify_receipts_retry01_mount.py'),
            'sealer_sha256':sha(__file__),'scope':'Fresh independent collection on replacement Pod using unchanged native/control/candidate contracts. Single BI-V150,4-layer BF16 MoE with FP32 SGD masters,mock data,dropout0,TP/PP/CP/EP1. Three-step tensor audits and120-step printed losses do not establish convergence or production/SFT/RL/multiGPU performance.'}
    with target.open('x',encoding='utf-8',newline='\n') as stream:json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':record['status'],'formal_processes':processes,'models':models},indent=2))


if __name__=='__main__':main()

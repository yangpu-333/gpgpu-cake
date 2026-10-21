"""Independent local review of compact ACT CPU/NUMA diagnostic evidence."""
import hashlib
import json
from pathlib import Path
import zipfile

BASE=Path(__file__).resolve().parent


def main():
    results=BASE/'results'; bundle=results/'cpu-numa-curated-diagnostic-key-results-20261016.zip'
    receipt=json.loads((results/'cpu-numa-curated-diagnostic-key-results-20261016-verification.json').read_text(encoding='utf-8'))
    if not receipt['passed'] or hashlib.sha256(bundle.read_bytes()).hexdigest()!=receipt['bundle_sha256']:
        raise ValueError('Verified local diagnostic bundle required')
    with zipfile.ZipFile(bundle) as archive:
        names=archive.namelist()
        def payload(suffix):
            found=[name for name in names if name.endswith(suffix)]
            if len(found)!=1:
                raise ValueError('Exactly one key member required: '+suffix)
            return archive.read(found[0])
        def read(suffix):
            return json.loads(payload(suffix))
        summary=read('/cpu-numa-mechanism-20261016-01/mechanism-summary.json')
        if len(summary['cases'])!=3 or not summary['shared_training_sources_unchanged']:
            raise ValueError('Complete source-matched three-case diagnostic required')
        rows=[]; total_original_bytes=0; total_original_files=0
        for case in summary['cases']:
            name=case['case']; case_key='/runs/'+name+'/'
            record=read(case_key+'case.json')
            if record['exit_code'] or record['resource_gate_before'].strip() or record['resource_gate_after'].strip():
                raise ValueError('Training failed or GPU occupancy unresolved')
            manifest_payload=payload(case_key+'archive-manifest.json')
            manifest=json.loads(manifest_payload)
            proof=read('/'+name+'-primary-verification.json')
            if not proof['passed'] or hashlib.sha256(manifest_payload).hexdigest()!=proof['manifest_sha256']:
                raise ValueError('ACT primary original archive proof mismatch')
            total_original_bytes+=proof['total_expected_bytes']; total_original_files+=proof['verified_file_count']
            numa=read('/cpu-numa-mechanism-20261016-01/'+name+'-numa.json')
            after=[item for item in numa['rows'] if item['phase']=='after-training']
            if len(after)!=2 or {item['rank'] for item in after}!={0,1}:
                raise ValueError('Both actual optimizer placement observations required')
            for row in numa['rows']:
                if row['sha256']!=manifest['files'][row['file']]['sha256']:
                    raise ValueError('NUMA analysis from a different actual snapshot')
                if row['page_query_return_code'] or row['intraop_threads']!=8:
                    raise ValueError('Page query incomplete or CPU thread count changed')
            for rank in (0,1):
                entry=read(case_key+f'rank-{rank}/entry.json')
                if entry['status']!='completed' or [s['step'] for s in entry['completed_native_steps']]!=list(range(1,36)) or any(s['skipped'] for s in entry['completed_native_steps']):
                    raise ValueError('35 actual unskipped training steps required')
                optimizers=entry['actual_optimizer_state']['optimizers']
                if len(optimizers)!=1 or optimizers[0]['adam_state_entries']!=199 or optimizers[0]['min_step']!=35 or optimizers[0]['max_step']!=35:
                    raise ValueError('Actual AdamW update coverage incomplete')
                observed=next(item for item in after if item['rank']==rank)
                for field in ('master','gradient','exp_avg','exp_avg_sq'):
                    coverage=observed['categories'][field]
                    if coverage['tensors']!=199 or coverage['elements']!=3834518016 or coverage['elements_with_valid_samples']!=3834518016:
                        raise ValueError('Actual buffer sample coverage incomplete')
                binding=read(case_key+f'numa-binding-rank-{rank}.json')
                for source,digest in binding['source_sha256'].items():
                    if hashlib.sha256((BASE/source).read_bytes()).hexdigest()!=digest:
                        raise ValueError('Local placement source differs from diagnostic')
                if binding['mode']!=case['mode']:
                    raise ValueError('Wrong placement mode')
            if len(case['windows'])!=4 or any(w['fused_adamw_calls']!=995 or w['gpu']['activity_record_counts']['collective']!=922 for w in case['windows']):
                raise ValueError('Both five-step rank windows and actual AdamW/collective activity required')
            rows.append({'case':name,'mode':case['mode'],'diagnostic_seconds_per_step':case['diagnostic_mean_seconds_per_step'],
                         'fused_adamw_cpu_seconds_per_rank_step':case['fused_adamw_host_seconds_per_rank_step'],
                         'final_actual_buffer_numa_samples':after})
    candidate,=[row for row in rows if row['mode']=='local8']; baselines=[row for row in rows if row['mode']=='unbound']
    result={'schema':'jiuge7b-cpu-numa-independent-diagnostic-review-v1','passed':True,'cases':rows,
            'original_files_verified_at_ACT':total_original_files,'original_bytes_at_ACT':total_original_bytes,
            'diagnostic_candidate_faster_than_both_unbound_runs':all(candidate['diagnostic_seconds_per_step']<row['diagnostic_seconds_per_step'] for row in baselines),
            'requires_new_full_numeric_restore_and_formal':True,'formal_performance_passed':False,
            'full_raw_local_copy_verified':False,'local_key_bundle_sha256':receipt['bundle_sha256']}
    (results/'cpu-numa-independent-diagnostic-review-20261016.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:value for key,value in result.items() if key!='cases'},indent=2))


if __name__=='__main__':
    main()

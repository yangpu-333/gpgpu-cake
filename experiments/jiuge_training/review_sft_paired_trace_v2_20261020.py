"""Independently aggregate verified, bounded keys; never load raw local traces."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics

BASE=Path(__file__).resolve().parent
DRIVER=BASE/'results/sft-paired-trace-driver-20261020-01'
KEYS=DRIVER/'keys'
CAMPAIGN=KEYS/'sft-paired-tracing-20261020-01'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    checked=read(DRIVER/'local-key-verification.json')
    if not checked['passed']:
        raise ValueError('Bounded key SHA verification required')
    state=read(CAMPAIGN/'controller.json')
    if state['status']!='completed' or state['verified_raw_traces']!=16:
        raise ValueError('Complete four-case capture required')
    if [(x['arm'],x['tag']) for x in state['cases']] != [('native','a1'),('optimized','b1'),('optimized','b2'),('native','a2')]:
        raise ValueError('ABBA order differs')
    approval=read(CAMPAIGN/'approval.json')
    candidate=read(CAMPAIGN/'candidate/candidate.json')
    if (approval['max_charge_points']!=0 or approval['startup_limit_seconds']!=10800
        or candidate['candidate_sha256']!='8c83ed05b1333b6775abea19bb2ff07f5966938764654ab133613fd52573ca88'
        or candidate['native_source_modified']):
        raise ValueError('Accepted isolated patch or zero-cost resource gate differs')
    manifest=read(BASE/'results/sft-paired-trace-source-manifest-20261020.json')['files']
    for name,sha in manifest.items():
        source_path=CAMPAIGN/'source'/name
        if not source_path.exists():
            if not name.endswith('.sh'):
                raise ValueError('Unexpected missing frozen source: '+name)
            source_path=BASE/'results/sft-paired-trace-source-supplement-20261020-01/source'/name
        if hashlib.sha256(source_path.read_bytes()).hexdigest()!=sha:
            raise ValueError('ACT frozen source/key differs: '+name)
    cases=[]
    reference=None
    windows=[]
    raw_files=[]
    for item in state['cases']:
        root=KEYS/'runs'/item['case']
        case=read(root/'case.json')
        completion=read(root/'completion-verification.json')
        archive=read(CAMPAIGN/(item['case']+'-archive-verification.json'))
        for gate in ('gate_before','gate_after'):
            observed=item[gate]
            actual=[line.split(',')[0].strip() for line in observed['cards'].strip().splitlines()]
            if actual!=approval['ordered_gpu_uuids'] or observed['compute_apps'].strip():
                raise ValueError('Pair/foreign compute gate differs')
        if case['exit_code']!=0 or not completion['passed'] or not archive['passed']:
            raise ValueError('Execution/completion/ACT original SHA gate failed')
        samples=read(root/'sft-supervision-rank-0.json')['rows']
        losses=completion['native_losses']
        if len(losses)!=35 or len(samples)!=35:
            raise ValueError('All35 actual updates required')
        if reference is None: reference=(samples,losses)
        elif (samples,losses)!=reference: raise ValueError('Paired SFT samples/masks/losses differ')
        for rank in (0,1):
            pin=read(CAMPAIGN/('candidate-'+item['tag'])/(item['arm']+f'-rank-{rank}-actual-pinning-init-1.json'))
            objective=read(root/f'sft-trace-objective-rank-{rank}.json')
            entry=read(root/f'rank-{rank}/entry.json')
            if (pin['parameter_count']!=199 or pin['pinned_count']!=(199 if item['arm']=='optimized' else 0)
                or not objective['run_returned'] or objective['range_call_counts']['jiuge/cpu_adamw_math']!=6965
                or entry['native_files_checked']!=600 or len(entry['completed_native_steps'])!=35
                or any(x['skipped'] for x in entry['completed_native_steps'])):
                raise ValueError('Actual full-update/pinning/range gate differs')
        mechanism=read(root/'symmetric-mechanism.json')
        comm=read(root/'communication-analysis.json')
        trace_review=read(root/'trace-analysis.json')
        for row in mechanism['windows']:
            h=row['host_ranges']
            for name in ('jiuge/cpu_adamw_math','jiuge/optimizer_update','jiuge/offload_d2h_submit'):
                if h[name]['count']!=(995 if name.endswith('math') else 5):
                    raise ValueError('Incomplete per-window AdamW observation')
            copies=[v for v in row['copy_api_review']['copy_groups'] if 'HtoD' in v['copy_name']]
            if not copies:
                raise ValueError('Actual HtoD copies absent')
            seconds=row['window_seconds']
            gpu=row['gpu_union_us']
            match=next(x for x in trace_review['windows'] if x['rank']==row['rank'] and x['window']==row['window'])
            messages=next(x for x in comm['ranks'] if x['rank']==row['rank'] and x['window']==row['window'])['device_message_groups']
            large=[x for x in messages if x['input_elements']==7340032 and x['dtype']=='BFloat16'
                   and str(x['collective']).lower().replace('_','')=='allreduce']
            capture=read(root/row['window']/f'rank-{row["rank"]}-capture.json')
            paired=next(x for x in comm['cross_rank_pair_candidates'] if x['window']==row['window'])
            # Kineto does not expose its drop counter in this Torch runtime.
            # Preserve that known uncertainty; reject every other coverage issue.
            issues=set(match['analysis']['issues'])
            if issues-{'zero dropped events is not established'}:
                raise ValueError('Additional whole-window coverage issues: '+str(issues))
            if len(match['analysis']['capture_window']['devices'])!=1:
                raise ValueError('Per-rank aggregation requires exactly one actual GPU device')
            window={
                'arm':item['arm'],'case':item['case'],'tag':item['tag'],'rank':row['rank'],'window':row['window'],
                'profiled_seconds_per_rank_step':seconds/5,
                'CPU_optimizer_seconds_per_rank_step':h['jiuge/optimizer_update']['seconds_per_rank_step'],
                'CPU_forward_backward_seconds_per_rank_step':h['jiuge/forward_backward']['seconds_per_rank_step'],
                'CPU_AdamW_math_seconds_per_rank_step':h['jiuge/cpu_adamw_math']['seconds_per_rank_step'],
                'CPU_D2H_submit_seconds_per_rank_step':h['jiuge/offload_d2h_submit']['seconds_per_rank_step'],
                'CPU_optimizer_outside_math_seconds_per_rank_step':row['optimizer_outside_adamw_math_union_us']/5e6,
                'CPU_HtoD_API_seconds_per_rank_step':sum(v['cpu_api_union_clipped_us'] for v in copies)/5e6,
                'GPU_HtoD_seconds_per_rank_step':gpu.get('HtoD',0)/5e6,
                'GPU_memcpy_seconds_per_rank_step':gpu.get('memcpy',0)/5e6,
                'GPU_compute_seconds_per_rank_step':gpu.get('compute',0)/5e6,
                'GPU_NCCL_seconds_per_rank_step':gpu.get('collective',0)/5e6,
                'GPU_idle_proxy_fraction':row['gpu_idle_proxy_us']/(seconds*1e6),
                'NCCL_overlap_compute_fraction':row['compute_collective_overlap_us']/gpu['collective'],
                'NCCL_overlap_memcpy_fraction':row['collective_memcpy_overlap_us']/gpu['collective'],
                'GPU_collective_memcpy_overlap_seconds_per_rank_step':row['collective_memcpy_overlap_us']/5e6,
                'memcpy_overlap_compute_fraction':row['compute_memcpy_overlap_us']/gpu['memcpy'],
                'CPU_AdamW_overlap_HtoD_seconds_per_rank_step':row['cpu_adamw_gpu_overlap_us']['HtoD']/5e6,
                'CPU_cudaEventSynchronize_inside_optimizer_seconds_per_rank_step':row['synchronization_inside_optimizer_us'].get('cudaEventSynchronize',0)/5e6,
                'HtoD_GiB_per_rank_step':sum(v['recorded_bytes'] for v in copies)/5/1024**3,
                'HtoD_copy_names':[v['copy_name'] for v in copies],
                'NCCL_kernel_count_per_five_steps':capture['gpu_collective_kernel_events'],
                'large_BF16_allreduce_count_per_five_steps':sum(x['count'] for x in large),
                'large_allreduce_logical_input_GiB_per_rank_step':sum(x['logical_input_bytes_sum'] for x in large)/5/1024**3,
                'maximum_collective_start_skew_candidate_ms':max([x['start_skew_us']/1000 for x in paired['largest_start_skew_candidates']],default=0),
                'ordered_collective_signatures_match_across_ranks':all(x['ordered_type_dtype_length_signatures_match'] for x in paired['group_checks']),
                'HtoD_unique_API_matching_complete':all(v['unmatched_gpu_copy_events']==0 and v['ambiguous_gpu_copy_events']==0 and v['api_correlations_shared_with_other_copy_names']==0 for v in copies),
                'trace_sha256':row['trace_sha256'],'decompressed_sha256':row['decompressed_sha256']}
            windows.append(window)
            raw_files.append({'case':item['case'],'window':row['window'],'rank':row['rank'],
                              'ACT_path':f'/home/huids25/kda-jiuge-prep-20261014/runs/{item["case"]}/{row["window"]}/rank-{row["rank"]}.json.gz',
                              'sha256':row['trace_sha256'],'decompressed_sha256':row['decompressed_sha256']})
        cases.append({'case':item['case'],'arm':item['arm'],'raw_archive_SHA_passed':archive['passed'],
                      'actual_updates_each_rank':35,'cross_rank_collective_candidates':comm['cross_rank_pair_candidates']})
    numeric=[key for key,value in windows[0].items() if isinstance(value,(int,float)) and not isinstance(value,bool) and key!='rank']
    arms={}
    for arm in ('native','optimized'):
        rows=[r for r in windows if r['arm']==arm]
        if len(rows)!=8: raise ValueError('Two runs x two windows x two ranks per arm required')
        arms[arm]={key:{'mean':statistics.mean(r[key] for r in rows),
                        'median':statistics.median(r[key] for r in rows),
                        'min':min(r[key] for r in rows),'max':max(r[key] for r in rows),
                        'descriptive_rank_window_count':len(rows)} for key in numeric}
    formal=read(BASE/'results/sft-pinning-independent-review-20261020-continuation-v2.json')
    if not formal['formal_performance_passed']: raise ValueError('Prior formal SFT acceptance missing')
    formal_ratio=formal['rounds'][0]['vs_native']['geomean_throughput_ratio']
    result={'schema':'jiuge7b-sft-paired-trace-independent-review-v2','utc':datetime.now(timezone.utc).isoformat(),
            'passed':True,'source_supplement_SHA_verified':True,'original_failed_review_preserved':True,'verified_raw_traces':16,'cases':cases,'windows':windows,'arms':arms,'raw_trace_index':raw_files,
            'paired_all35_SFT_sample_IDs_supervision_counts_and_logged_losses_equal':True,
            'all35_raw_token_and_mask_arrays_exported_in_this_profile':False,
            'original_SFT_formal_throughput_speedup_vs_native_percent':(formal_ratio-1)*100,
            'new_formal_speedup_measured':False,'critical_path_proven':False,
            'native_collective_sequence_numbers_verified':False,'collector_drop_count_verified':False,
            'statistical_scope':'8 rank-window observations per arm; same seed, two runs; not8 independent seeds',
            'CPU_math_scope':'torch.optim.adamw.adamw functional call; excludes AdamW step post-hooks and HtoD staging',
            'CPU_API_scope':'unique CUPTI-correlated runtime spans; not independently proven exposed wait',
            'categories_are_additive':False,'raw_originals_location':'ACT Ceph; no local trace bodies'}
    output=BASE/'results/sft-paired-trace-independent-review-v2-20261020.json'
    with output.open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    print(json.dumps({'passed':True,'raw_traces':16,'arms':arms},indent=2))


if __name__=='__main__':
    main()

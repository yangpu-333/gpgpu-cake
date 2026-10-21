"""Summarize verified same-allocation CPT/SFT traces without a speedup claim."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT=PREP/'sft-tracing-20261017-01'


def read(path):
    return json.loads(path.read_text())


def main():
    allocation=read(ROOT/'allocation.json')
    tasks=[]
    for objective,name in (('CPT','cpt-native-profile-20261017-01'),
                           ('SFT','sft-native-profile-20261017-01')):
        folder=PREP/'runs'/name
        close=read(folder/'completion-verification.json')
        if not close['passed'] or close['actual_non_skipped_updates_each_rank']!=35:
            raise ValueError('Both 35-step native tasks must pass completion checks')
        archive=read(ROOT/(name+'-archive-verification.json'))
        if not archive['passed']:
            raise ValueError('ACT primary originals must pass full-case SHA verification')
        analysis=read(folder/'trace-analysis.json')
        communication=read(folder/'communication-analysis.json')
        supervision=(read(folder/'sft-supervision-rank-0.json')['rows'] if objective=='SFT'
                     else [{'input_slots':2048,'supervised_answer_and_EOD_targets':2048} for _ in range(35)])
        windows=[]
        for row in analysis['windows']:
            window,rank=row['window'],row['rank']
            receipt=read(folder/window/f'rank-{rank}-capture.json')
            gpu=row['analysis']['capture_window']['devices'][0]
            width=gpu['window_us']
            steps=receipt['stable_step_ids']
            input_slots=sum(supervision[step-1]['input_slots'] for step in steps)
            targets=sum(supervision[step-1]['supervised_answer_and_EOD_targets'] for step in steps)
            host=row['cpu_phase_ranges']
            comm=next(item for item in communication['ranks'] if item['window']==window and item['rank']==rank)
            windows.append({'window':window,'rank':rank,'step_ids':steps,
                'profiled_five_step_window_seconds':width/1e6,
                'host_forward_backward_seconds_per_step':host['jiuge/forward_backward']['host_range_sum_us']/5e6,
                'host_optimizer_offload_seconds_per_step':host['jiuge/optimizer_update']['host_range_sum_us']/5e6,
                'host_optimizer_offload_window_fraction':host['jiuge/optimizer_update']['host_window_fraction'],
                'GPU_compute_seconds_per_step':gpu['interval_union_us']['compute']/5e6,
                'GPU_NCCL_seconds_per_step':gpu['interval_union_us']['collective']/5e6,
                'GPU_memcpy_seconds_per_step':gpu['interval_union_us']['memcpy']/5e6,
                'compute_NCCL_overlap_seconds_per_step':gpu['compute_collective_overlap_us']/5e6,
                'NCCL_overlap_with_compute_fraction':gpu['collective_overlap_fraction'],
                'NCCL_without_compute_overlap_seconds_per_step':gpu['collective_nonoverlap_with_compute_proxy_us']/5e6,
                'GPU_idle_proxy_fraction':gpu['gpu_idle_proxy_fraction'],
                'GPU_compute_window_fraction':gpu['interval_union_us']['compute']/width,
                'GPU_busy_window_fraction':gpu['gpu_busy_union_us']/width,
                'input_slots_in_window':input_slots,'supervised_targets_in_window':targets,
                'profiled_input_slots_per_second':input_slots/(width/1e6),
                'profiled_supervised_targets_per_second':targets/(width/1e6),
                'GPU_activity_record_counts':gpu['activity_record_counts'],
                'recorded_memcpy_bytes_by_name':comm['recorded_memcpy_bytes_by_name'],
                'trace_path':str(folder/window/f'rank-{rank}.json.gz'),
                'compressed_sha256':receipt['trace_sha256'],
                'decompressed_sha256':receipt['decompressed_json_sha256'],
                'clock_review':row['clock_review']})
        metrics=('host_forward_backward_seconds_per_step','host_optimizer_offload_seconds_per_step',
                 'host_optimizer_offload_window_fraction','GPU_compute_seconds_per_step','GPU_NCCL_seconds_per_step',
                 'GPU_memcpy_seconds_per_step','compute_NCCL_overlap_seconds_per_step',
                 'NCCL_overlap_with_compute_fraction','NCCL_without_compute_overlap_seconds_per_step',
                 'GPU_idle_proxy_fraction','GPU_compute_window_fraction','GPU_busy_window_fraction')
        summary={key:{'min':min(row[key] for row in windows),'max':max(row[key] for row in windows),
                      'mean_across_two_windows_two_ranks_not_independent_replicates':statistics.mean(row[key] for row in windows)}
                 for key in metrics}
        tasks.append({'objective':objective,'case':name,'raw_root':str(folder),'windows':windows,
                      'descriptive_summary':summary,'communication_pair_candidates':communication['cross_rank_pair_candidates'],
                      'ACT_full_original_archive_verified':True,'actual_optimizer_steps_each_rank':35,
                      'warmup_steps_excluded_from_capture':20})
    result={'schema':'jiuge7b-CPT-SFT-tracing-summary-v1','utc':datetime.now(timezone.utc).isoformat(),
            'passed':True,'allocation':allocation,'actual_same_input_admission':read(ROOT/'cpt-sft-input-pair-verification.json'),
            'SFT_original_document_admission':read(PREP/'runs/sft-native-admission-20261017-01/sft-admission-verification.json'),
            'data':read(PREP/'datasets/alpaca-sft-answer-v1-receipt.json'),'tasks':tasks,
            'formal_speedup_evidence':False,'optimization_promoted_from_this_capture':False,
            'scope':'28-layer actual7B full-parameter training; two stable whole-step windows per task, not a full startup-to-shutdown recording',
            'limitations':['Profiler overhead is present; these are diagnostic windows, not formal throughput acceptance.',
                           'Task objectives differ; fewer supervised targets do not reduce all dense forward/backward compute.',
                           'CPU/GPU ranges are separate overlapping accounts and must not be added.',
                           'NCCL duration can include peer waiting; no-compute overlap and idle are descriptive proxies.',
                           'Cross-rank matching uses group/order/type/dtype/length; native collective sequence is unverified.',
                           'Shared host CPU/memory-bandwidth interference cannot be ruled out.',
                           'Collector drop count unavailable; no zero-drop or complete critical-path claim.',
                           'Single-node TP2/DP1; no multi-node or data-parallel scaling validation.'],
            'summary_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (ROOT/'tracing-summary.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({'passed':True,'tasks':[task['case'] for task in tasks],'raw_trace_windows':8,
                      'formal_speedup_evidence':False}),flush=True)


if __name__=='__main__':
    main()

"""CPU-only review of symmetric AdamW annotations and real GPU interval unions."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from analyze_trace import load_trace, interval_length, intersection_length, _gpu_kind
from inspect_copy_api import inspect


def analyze_case(root):
    rows = []
    for folder in sorted(root.glob('window-*')):
        for rank in (0, 1):
            receipt = json.loads((folder/f'rank-{rank}-capture.json').read_text())
            path = folder/f'rank-{rank}.json.gz'
            if hashlib.sha256(path.read_bytes()).hexdigest() != receipt['trace_sha256']:
                raise ValueError('Compressed trace SHA differs')
            trace, source = load_trace(path, max_bytes=512*1024**2)
            if source['decompressed_json_sha256'] != receipt['decompressed_json_sha256']:
                raise ValueError('Decompressed trace SHA differs')
            outer = [e for e in trace['traceEvents'] if e.get('ph') == 'X'
                     and e.get('cat') == 'user_annotation' and e.get('name') == receipt['window_name']]
            if len(outer) != 1:
                raise ValueError('One whole-window enclosing annotation required')
            start, end = outer[0]['ts'], outer[0]['ts']+outer[0]['dur']
            host, gpu, sync, host_counts = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(int)
            for e in trace['traceEvents']:
                if e.get('ph') != 'X' or 'ts' not in e or 'dur' not in e:
                    continue
                a, b = max(start, e['ts']), min(end, e['ts']+e['dur'])
                if b <= a:
                    continue
                name, cat = str(e.get('name', '')), e.get('cat')
                if cat == 'user_annotation' and name.startswith('jiuge/'):
                    host[name].append((a, b))
                    host_counts[name] += 1
                kind = _gpu_kind(e)
                if kind:
                    gpu[kind].append((a,b))
                if cat == 'gpu_memcpy' and 'HtoD' in name:
                    gpu['HtoD'].append((a,b))
                if cat == 'cuda_runtime' and 'Synchronize' in name:
                    sync[name].append((a,b))
            expected = {'jiuge/cpu_adamw_math':995, 'jiuge/optimizer_update':5,
                        'jiuge/offload_d2h_submit':5, 'jiuge/offload_group_sync':5,
                        'jiuge/offload_state_sync':5, 'jiuge/forward_backward':5}
            if any(host_counts[key] != count for key,count in expected.items()):
                raise ValueError('Complete symmetric range coverage differs: '+str(dict(host_counts)))
            math_ranges, opt = host['jiuge/cpu_adamw_math'], host['jiuge/optimizer_update']
            groups = {key: {'count':host_counts[key], 'union_us':interval_length(host[key]),
                            'seconds_per_rank_step':interval_length(host[key])/5e6,
                            'window_fraction':interval_length(host[key])/(end-start)} for key in expected}
            total_gpu = gpu['compute']+gpu['collective']+gpu['memcpy']+gpu['memset']+gpu['unknown_gpu']
            rows.append({'window':folder.name, 'rank':rank, 'steps':receipt['stable_step_ids'],
                'window_seconds':(end-start)/1e6, 'host_ranges':groups,
                'optimizer_outside_adamw_math_union_us':interval_length(opt)-intersection_length(opt,math_ranges),
                'gpu_union_us':{k:interval_length(v) for k,v in gpu.items()},
                'gpu_idle_proxy_us':(end-start)-interval_length(total_gpu),
                'compute_collective_overlap_us':intersection_length(gpu['compute'],gpu['collective']),
                'compute_memcpy_overlap_us':intersection_length(gpu['compute'],gpu['memcpy']),
                'collective_memcpy_overlap_us':intersection_length(gpu['collective'],gpu['memcpy']),
                'cpu_adamw_gpu_overlap_us':{key:intersection_length(math_ranges,gpu[key]) for key in ('compute','collective','memcpy','HtoD')},
                'synchronization_inside_optimizer_us':{key:intersection_length(opt,value) for key,value in sync.items()},
                'copy_api_review':inspect(trace,receipt['window_name']),
                'trace_sha256':receipt['trace_sha256'], 'decompressed_sha256':source['decompressed_json_sha256']})
    if len(rows) != 4:
        raise ValueError('Exactly both windows and both ranks required')
    return {'schema':'jiuge7b-sft-symmetric-mechanism-trace-v1', 'case':root.name, 'windows':rows,
            'formal_performance_evidence':False, 'critical_path_proven':False,
            'cpu_math_excludes_adamw_optimizer_post_hooks':True,
            'cpu_api_is_exposed_wait':False, 'host_and_device_categories_are_additive':False,
            'collector_drop_count_verified':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = analyze_case(args.case_root)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({'case':args.case_root.name,'windows':len(result['windows']),'passed':True}))

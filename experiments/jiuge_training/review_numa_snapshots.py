"""Summarize actual own-process sampled page locations; not a full inventory."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    case=json.loads((args.case_root/'case.json').read_text())
    if case['exit_code']!=0:
        raise ValueError('Completed actual training case required')
    rows=[]
    for path in sorted(args.case_root.glob('numa-*-rank-*.json')):
        if path.name.startswith('numa-binding-'):
            continue
        data=json.loads(path.read_text())
        rank=int(path.stem.rsplit('-',1)[1])
        binding=json.loads((args.case_root/f'numa-binding-rank-{rank}.json').read_text())
        query=data['page_queries']
        categories={}
        for tensor in query.get('tensors',[]):
            name=tensor['field']; label='master' if name.startswith('master/') else name.rsplit('/',1)[-1]
            group=categories.setdefault(label,{'tensors':0,'elements':0,'pinned_tensors':0,'sample_nodes':Counter(),
                                               'element_weighted_local_sample_fraction_sum':0.,'elements_with_valid_samples':0})
            group['tensors']+=1; group['elements']+=tensor['numel']; group['pinned_tensors']+=int(tensor['is_pinned'])
            if query['return_code']<0:
                continue
            start=tensor['sample_start']; count=tensor['sample_count']; nodes=query['sample_nodes'][start:start+count]
            group['sample_nodes'].update(map(str,nodes))
            valid=[node for node in nodes if node>=0]
            if valid:
                group['elements_with_valid_samples']+=tensor['numel']
                group['element_weighted_local_sample_fraction_sum']+=tensor['numel']*sum(
                    node==binding['gpu_local_numa_node'] for node in valid)/len(valid)
        for group in categories.values():
            group['sample_nodes']=dict(group['sample_nodes'])
            denominator=group['elements_with_valid_samples']
            group['element_weighted_local_sample_fraction']=group.pop('element_weighted_local_sample_fraction_sum')/denominator if denominator else None
        rows.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'phase':data['phase'],'rank':rank,'mode':binding['mode'],
                     'selected_cpus':binding['allowed_after'],'intraop_threads':data['torch_intraop_threads'],
                     'page_query_return_code':query['return_code'],'page_query_errno':query['errno'],
                     'categories':categories,
                     'thread_affinity_groups':dict(Counter(','.join(map(str,value)) for value in data['thread_affinity'].values()))})
    if not rows:
        raise ValueError('Actual optimizer placement observations missing')
    output={'schema':'jiuge7b-actual-buffer-numa-sample-review-v1','case':case['case'],'rows':rows,
            'limitations':['Own-process sample locations, not a complete per-page inventory.',
                           'Element-weighted fractions extrapolate each tensor sample; not a full byte-placement measurement.',
                           'Affinity can affect subsequent default-policy allocations; no existing pages were migrated.',
                           'Not a formal throughput result or NUMA critical-path proof.']}
    with args.output.open('x') as stream:
        json.dump(output,stream,indent=2); stream.write('\n')
    print(json.dumps({'case':case['case'],'snapshots':len(rows)}))


if __name__=='__main__':
    main()

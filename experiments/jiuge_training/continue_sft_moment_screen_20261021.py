"""Versioned three-arm screen, preserving all original scientific sources."""
import hashlib
import json
from pathlib import Path

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')


def moment_gate(folder,rank,arm,binding):
    record=json.loads((folder/f'sft-local-moments-rank-{rank}.json').read_text())
    if record['arm']!=arm:raise ValueError('Moment arm receipt differs')
    events=record['moment_events']
    if arm!='local-moments':
        if events:raise ValueError('Control must retain native moment allocation')
        return
    nodes=binding['selected_cpu_nodes']
    if len(nodes)!=1 or len(events)<398:raise ValueError('All398 final moments require a single CPU node')
    node=nodes[0]
    if any(e['target_node']!=node or e['mbind_flags']!=0 or e['migrated_existing_pages']
           or not e['private_mapping'] or any(n!=node for n in e['sampled_nodes']) for e in events):
        raise ValueError('Candidate placement receipt failed')
    for phase in ('after-first-update','after-training'):
        query=json.loads((folder/f'sft-numa-{phase}-rank-{rank}.json').read_text())['page_queries']
        if query['return_code']!=0:raise ValueError('Actual moment page query failed')
        for field in ('exp_avg','exp_avg_sq'):
            values=[t for t in query['tensors'] if t['field'].endswith('/'+field)]
            if len(values)!=199 or sum(t['numel'] for t in values)!=3834518016:
                raise ValueError('Full final moment-object coverage required')
            for t in values:
                begin,count=t['sample_start'],t['sample_count']
                locations=query['sample_nodes'][begin:begin+count]
                if (t['dtype']!='torch.float32' or t['is_pinned'] or count<1 or len(locations)!=count
                        or any(n!=node for n in locations)):
                    raise ValueError('Actual final moment samples differ from the CPU node')


def derive(source):
    text=source.decode()
    old_order="""        order = [('unbound', 'a1'), ('gpu-local8', 'b1')]
        if 'socket-split8' in plan['policies']:
            order += [('socket-split8', 'c1'), ('socket-split8', 'c2')]
        order += [('gpu-local8', 'b2'), ('unbound', 'a2')]"""
    new_order="""        order = [('baseline','a1'),('affinity-control','b1'),('local-moments','c1'),
                 ('local-moments','c2'),('affinity-control','b2'),('baseline','a2')]"""
    changes=[
        ("sft-numa-diagnostic-20261021-02","sft-moment-screen-20261021-01"),
        ("sft-numa-approved-v2-20261021.json","sft-moment-screen-approved-20261021.json"),
        ("jiuge7b-sft-numa-diagnostic-controller-v2","jiuge7b-sft-moment-screen-controller-v1"),
        ("    ROOT.mkdir(exist_ok=False)","    runtime_path=PREP/('sft-moment-runtime-'+socket.gethostname())/'summary.json'\n    runtime=json.loads(runtime_path.read_text())\n    if not runtime['passed'] or runtime['GPU_used'] or runtime['hostname']!=socket.gethostname():\n        raise ValueError('Live CPU placement and resume gate required')\n    ROOT.mkdir(exist_ok=False)\n    shutil.copyfile(runtime_path,ROOT/'runtime-probe.json')"),
        ("        plan = build_plan(discover())", "        plan = build_plan(discover())\n        plan['arms']={'baseline':plan['policies']['unbound'],'affinity-control':plan['policies']['gpu-local8'],'local-moments':plan['policies']['gpu-local8']}"),
        (old_order,new_order),
        ("        for policy, tag in order:","        for policy, tag in order:\n            mode='unbound' if policy=='baseline' else 'gpu-local8'"),
        ("sft-numa-v2-20261021-","sft-moment-screen-20261021-"),
        ("run_sft_numa_case_20261021.py","run_sft_moment_case_20261021.py"),
        ("'--sft-numa-mode', policy, '--sft-numa-plan', plan_path", "'--sft-numa-mode', mode, '--sft-numa-plan', plan_path, '--sft-moment-arm', policy"),
        ("plan['policies'][policy][rank]","plan['arms'][policy][rank]"),
        ("            item = {'case': case, 'policy': policy, 'tag': tag,", "            for moment_rank in (0,1):\n                moment_binding=json.loads((folder/f'sft-numa-binding-rank-{moment_rank}.json').read_text())\n                moment_gate(folder,moment_rank,policy,moment_binding)\n            item = {'case': case, 'policy': policy, 'tag': tag,"),
        ("sft-numa-key-results-v2-20261021.zip","sft-moment-screen-key-results-20261021.zip")]
    for before,after in changes:
        if text.count(before)!=1:raise ValueError('Exact controller derivation failed: '+before)
        text=text.replace(before,after)
    return text


def main():
    manifest=json.loads((PREP/'sft-moment-screen-source-manifest-20261021.json').read_text())['files']
    source=(PREP/'continue_sft_numa_v2_20261021.py').read_bytes()
    if hashlib.sha256(source).hexdigest()!=manifest['continue_sft_numa_v2_20261021.py']:
        raise ValueError('Frozen prior controller changed')
    text=derive(source)
    exec(compile(text,str(PREP/'continue_sft_moment_screen_20261021.generated.py'),'exec'),
         {'__name__':'__main__','__file__':__file__,'moment_gate':moment_gate})


if __name__=='__main__':main()

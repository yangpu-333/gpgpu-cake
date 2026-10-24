"""Strict resume-aware CPU review of closed evidence; no GPU or source changes."""
from pathlib import Path
import ast
import collections
import copy
import hashlib
import json

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT=PREP/'sft-phased-acceptance-20261024-02'


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(policy,entry,objective,resume,rank):
    assert entry['status']=='completed' and entry['native_files_checked']==600
    assert entry['completed_native_steps']==[{'step':1,'skipped':0}]
    assert entry['weight_loading']['all_loaded_shard_bytes_verified']
    assert objective['run_returned_successfully'] and len(objective['supervision_rows'])==(1 if rank==0 else 0)
    leaves=entry['actual_optimizer_state']['optimizers']
    assert len(leaves)==1 and leaves[0]['adam_state_entries']==199
    assert leaves[0]['min_step']==leaves[0]['max_step']==3
    assert leaves[0]['elements_with_both_adam_moments']==3834518016
    assert policy['steps']==2 and len(policy['moment_events'])==796
    node=policy['binding']['selected_cpu_nodes'][0]
    for events,path in ((policy['moment_events'][:398],'direct_local_zero'),(policy['moment_events'][398:],'native_resume_copy')):
        assert all(x['initialization_path']==path and x['target_node']==node for x in events)
        for field in ('exp_avg','exp_avg_sq'):
            rows=[x for x in events if x['field']==field]
            assert len(rows)==199 and sum(x['elements'] for x in rows)==3834518016
            assert all(x['copied_source_moment_bytes']==(0 if path=='direct_local_zero' else x['elements']*4) for x in rows)
            assert all(not x['migrated_existing_pages'] and all(n==node for n in x['sampled_nodes']) for x in rows)
    page=policy['final_pages']['page_queries'];assert page['return_code']==0 and page['sample_nodes'] is not None
    for field in ('exp_avg','exp_avg_sq'):
        rows=[x for x in page['tensors'] if x['field'].endswith('/'+field)]
        assert len(rows)==199 and sum(x['numel'] for x in rows)==3834518016
        assert all(n==node for x in rows for n in page['sample_nodes'][x['sample_start']:x['sample_start']+x['sample_count']])
    assert resume['passed'] and resume['native_checkpoint_loader_used'] and resume['restored_iteration']==2
    assert resume['completed_steps']==[3]
    assert [(x['arm'],x['fields'],x['expected_fields'],x['complete'],x['numerical_passed']) for x in resume['runs']]==[
        ('resume-restored',995,995,True,True),('resume-next-step',1195,1195,True,True)]
    return {'rank':rank,'native_initialization_plus_real_update':2,'actual_training_updates':1,
            'real_optimizer_iteration':3,'initialization_moment_events':398,'resume_copy_events':398,
            'final_moment_objects':398,'final_sampled_moment_pages_local':True,
            'resume_max_abs':max(x['max_abs'] for x in resume['runs']),'passed':True}


def review():
    controller=read(ROOT/'controller.json')
    assert controller['status']=='stopped' and controller['error']=='Actual policy/update/page proof incomplete'
    assert not controller['formal_performance_started'] and all(x['exit_code']==0 for x in controller['commands'])
    sources=read(PREP/'sft-phased-acceptance-source-manifest-20261024-02.json')['files']
    assert len(sources)==169
    for n,h in sources.items():assert sha(PREP/n)==sha(ROOT/'source'/n)==h,n
    native=read(PREP/'source_manifest.json');assert len(native['files'])==600
    native_root=PREP/'megatron-native-5be9626'
    excerpts={}
    for name,method in (('megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py','dummy_step'),('megatron/core/optimizer/distrib_optimizer.py','load_state_dict')):
        path=native_root/name;assert sha(path)==native['files'][name]
        text=path.read_text();tree=ast.parse(text)
        functions=[x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name==method]
        selected=[x for x in functions if any(isinstance(y,ast.Call) and isinstance(y.func,ast.Attribute) and y.func.attr==('step' if method=='dummy_step' else 'dummy_step') for y in ast.walk(x))]
        assert len(selected)==1
        excerpts[name]={'sha256':sha(path),'method':method,'line':selected[0].lineno,'source':ast.get_source_segment(text,selected[0])}
    case=PREP/'runs/phased-resume-optimized-20261024-02';record=read(case/'case.json')
    assert record['exit_code']==0 and not record['resource_gate_before'].strip() and not record['resource_gate_after'].strip()
    result=[];rejected=0
    for rank in (0,1):
        args=[read(case/f'phased-policy-rank-{rank}.json'),read(case/f'rank-{rank}/entry.json'),read(case/f'sft-objective-rank-{rank}.json'),read(ROOT/'numeric'/f'resume-rank-{rank}-completion.json'),rank]
        assert args[1]['source_manifest_sha256']==sha(PREP/'source_manifest.json')
        result.append(validate(*args))
        for kind in ('extra-call','missing-event','extra-training','bad-step','failed-values','foreign-page'):
            changed=copy.deepcopy(args)
            if kind=='extra-call':changed[0]['steps']=3
            elif kind=='missing-event':changed[0]['moment_events'].pop()
            elif kind=='extra-training':changed[1]['completed_native_steps'].append({'step':2,'skipped':0})
            elif kind=='bad-step':changed[1]['actual_optimizer_state']['optimizers'][0]['max_step']=4
            elif kind=='failed-values':changed[3]['passed']=False
            else:
                q=changed[0]['final_pages']['page_queries'];row=next(x for x in q['tensors'] if x['field'].endswith('/exp_avg'));q['sample_nodes'][row['sample_start']]=1-changed[0]['binding']['selected_cpu_nodes'][0]
            try:validate(*changed)
            except AssertionError:rejected+=1
            else:raise ValueError('Invalid evidence was accepted: '+kind)
    return {'passed':True,'GPU_used':False,'scientific169_sources_modified':False,'original_controller_failure_preserved':True,
            'gate_correction':'separate native checkpoint initialization from one real update; require both398-event phases',
            'ranks':result,'negative_gate_regressions_rejected':rejected,'native_source_provenance':excerpts,
            'full_semantics_and_raw_SHA_pending':True,'formal_performance_passed':False}


if __name__=='__main__':print(json.dumps(review()))

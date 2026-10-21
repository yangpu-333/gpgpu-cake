"""Exercise the fully generated launch and reject unsafe scientific receipts."""
import hashlib
import json
from pathlib import Path
import tempfile
from run_sft_moment_direct_zero_case_20261021 import derive as runner
from sft_moment_direct_zero_entry_20261021 import derive as entry
from continue_sft_moment_direct_zero_20261021 import derive as controller,moment_gate
from review_sft_moment_direct_zero_20261021 import generated as review
from sft_moment_memory_guard_20261021 import check_values,RESERVE_BYTES

BASE=Path(__file__).resolve().parent


def main():
    source=(BASE/'run_sft_trace_case_20261020.py').read_bytes()
    scope={'__name__':'probe','__file__':str(BASE/'run_sft_moment_direct_zero_case_20261021.py')}
    exec(compile(runner(source),'<direct-zero-outer-launcher>','exec'),scope)
    scope['PREP']=BASE;generated=[]
    scope['compile']=lambda text,*args:text
    scope['exec']=lambda text,*args:generated.append(text)
    scope['generated_main']()
    assert len(generated)==1
    compile(generated[0],'<actual-direct-zero-native-launch>','exec')
    for needed in ['sft_moment_direct_zero_entry_20261021.py','sft_moment_memory_guard_20261021.py',
                   "'--sft-moment-arm',args.sft_moment_arm",'Fixed 35-step S2048 profile only',
                   "OMP_NUM_THREADS='8'",'datasets/alpaca-sft-answer-v1']:
        assert needed in generated[0],needed
    original=(BASE/'sft_numa_entry_20261021.py').read_bytes()
    compile(entry(original,hashlib.sha256(original).hexdigest()),'<actual-guarded-direct-zero-entry>','exec')
    text=controller((BASE/'continue_sft_numa_v2_20261021.py').read_bytes())
    compile(text,'<six-fresh-direct-zero-cases>','exec')
    assert "'--sft-moment-arm', policy" in text and 'sft-moment-direct-zero-key-results-20261021.zip' in text
    compile(review(),'<actual-direct-zero-independent-review>','exec')
    assert RESERVE_BYTES==8*1024**3
    check_values(2*1024**3,16*1024**3,1024**3)
    try:check_values(10*1024**3,16*1024**3,1024)
    except MemoryError:pass
    else:raise AssertionError('Insufficient production headroom accepted')
    rejected=[]
    with tempfile.TemporaryDirectory(dir=BASE,prefix='.moment-v3-gate-') as temporary:
        root=Path(temporary)
        events=[{'target_node':1,'mbind_flags':0,'migrated_existing_pages':False,
                 'private_mapping':True,'sampled_nodes':[1],'initialization_path':'direct_local_zero',
                 'native_source_moment_created':False,'copied_source_moment_bytes':0} for _ in range(398)]
        receipt={'arm':'local-moments','moment_events':events}
        config=root/'sft-local-moments-rank-0.json'
        def save():config.write_text(json.dumps(receipt))
        save()
        (root/'moment-initialization-memory-rank-0.jsonl').write_text(json.dumps(
            {'created_moment_objects':398,'copied_source_moment_bytes':0})+'\n')
        tensors=[]
        for field in ('exp_avg','exp_avg_sq'):
            for i in range(199):
                tensors.append({'field':f'cpu/{i}/{field}','numel':3834518016-198 if i==198 else 1,
                    'dtype':'torch.float32','is_pinned':False,'sample_start':len(tensors),'sample_count':1})
        snap={'page_queries':{'return_code':0,'tensors':tensors,'sample_nodes':[1]*398}}
        for phase in ('after-first-update','after-training'):
            (root/f'sft-numa-{phase}-rank-0.json').write_text(json.dumps(snap))
        moment_gate(root,0,'local-moments',{'selected_cpu_nodes':[1]})
        for label,updates in [('wrong-node',{'sampled_nodes':[0]}),('migration',{'mbind_flags':2}),
                              ('old-copy-path',{'initialization_path':'native_resume_copy'}),
                              ('source-copy-bytes',{'copied_source_moment_bytes':4})]:
            old=events[0].copy();events[0].update(updates);save()
            try:moment_gate(root,0,'local-moments',{'selected_cpu_nodes':[1]})
            except ValueError:rejected.append(label)
            else:raise AssertionError('Unsafe state admitted: '+label)
            events[0]=old;save()
    print(json.dumps({'passed':True,'GPU_used':False,'Torch_imported':False,
                      'fully_generated_launch_entry_controller_review_compile':True,
                      'production_reserve_bytes':RESERVE_BYTES,
                      'insufficient_memory_rejected_before_allocation':True,
                      'rejected_invalid_live_receipts':rejected,
                      'new_7B_SFT_acceptance':False,'formal_performance_evidence':False}))


if __name__=='__main__':main()

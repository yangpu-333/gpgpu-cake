"""Check generated programs and costly-run gates without Torch or GPU work."""
import hashlib
import json
from pathlib import Path
import tempfile
from run_sft_moment_case_20261021 import derive as runner
from sft_moment_entry_20261021 import derive as entry
from continue_sft_moment_screen_20261021 import derive as controller,moment_gate
from review_sft_moment_screen_20261021 import derive as review

BASE=Path(__file__).resolve().parent


def main():
    source=(BASE/'run_sft_trace_case_20261020.py').read_bytes()
    scope={'__name__':'probe','__file__':str(BASE/'run_sft_moment_case_20261021.py')}
    exec(compile(runner(source),'<outer runner>','exec'),scope)
    scope['PREP']=BASE;generated=[]
    scope['compile']=lambda text,*args:text
    scope['exec']=lambda text,*args:generated.append(text)
    scope['generated_main']()
    assert len(generated)==1
    compile(generated[0],'<actual launch>','exec')
    for required in ('sft_moment_entry_20261021.py',"'--sft-moment-arm',args.sft_moment_arm",'Fixed 35-step S2048 profile only',"OMP_NUM_THREADS='8'",'datasets/alpaca-sft-answer-v1'):
        assert required in generated[0],required
    old=(BASE/'sft_numa_entry_20261021.py').read_bytes()
    compile(entry(old,hashlib.sha256(old).hexdigest()),'<actual entry>','exec')
    text=controller((BASE/'continue_sft_numa_v2_20261021.py').read_bytes())
    compile(text,'<actual controller>','exec')
    assert "'--sft-moment-arm', policy" in text and "plan['arms'][policy][rank]" in text
    compile(review((BASE/'review_sft_numa_v2_20261021.py').read_bytes()),'<actual independent review>','exec')
    # Reject incomplete live moment coverage, wrong node and migration claims.
    with tempfile.TemporaryDirectory(dir=BASE,prefix='.moment-gate-fixture-') as temporary:
        root=Path(temporary).resolve()
        assert root.parent==BASE and root.name.startswith('.moment-gate-fixture-')
        events=[{'target_node':1,'mbind_flags':0,'migrated_existing_pages':False,
                 'private_mapping':True,'sampled_nodes':[1]} for _ in range(398)]
        receipt={'arm':'local-moments','moment_events':events}
        config=root/'sft-local-moments-rank-0.json'
        config.write_text(json.dumps(receipt))
        tensors=[]
        for label in ('exp_avg','exp_avg_sq'):
            for i in range(199):
                tensors.append({'field':f'cpu/{i}/{label}','numel':3834518016-198 if i==198 else 1,
                                'dtype':'torch.float32','is_pinned':False,'sample_start':len(tensors),'sample_count':1})
        snapshot={'page_queries':{'return_code':0,'tensors':tensors,'sample_nodes':[1]*398}}
        for phase in ('after-first-update','after-training'):
            (root/f'sft-numa-{phase}-rank-0.json').write_text(json.dumps(snapshot))
        binding={'selected_cpu_nodes':[1]}
        moment_gate(root,0,'local-moments',binding)
        rejected=[]
        for label,mutate in (('wrong-node',lambda:events[0].update(sampled_nodes=[0])),
                             ('migration-flag',lambda:events[0].update(mbind_flags=2))):
            mutate();config.write_text(json.dumps(receipt))
            try:moment_gate(root,0,'local-moments',binding)
            except ValueError:rejected.append(label)
            else:raise AssertionError('Invalid receipt accepted: '+label)
            events[0].update(sampled_nodes=[1],mbind_flags=0)
        config.write_text(json.dumps(receipt))
        tensors.pop()
        (root/'sft-numa-after-training-rank-0.json').write_text(json.dumps(snapshot))
        try:moment_gate(root,0,'local-moments',binding)
        except ValueError:rejected.append('missing-final-moment')
        else:raise AssertionError('Incomplete final state accepted')
    print(json.dumps({'passed':True,'Torch_imported':False,'GPU_used':False,
                      'fully_generated_launch_entry_controller_review_compile':True,
                      'rejected_invalid_live_state_fixtures':rejected,
                      'real_7B_SFT_acceptance':False}))


if __name__=='__main__':main()

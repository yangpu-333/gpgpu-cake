"""Compare new affinity candidate with immutable native/pinned observations."""
import argparse
import hashlib
import json
from pathlib import Path
from verify_pinning_semantics import load_observation, compare, compare_observations

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    old=PREP/'pinning-semantics-20261016-04'
    rows=[]
    for rank in (0,1):
        baseline=load_observation(old/'semantics-optimized'/f'rank-{rank}',[1,2,3],0)
        candidate=load_observation(args.root/'semantics-optimized'/f'rank-{rank}',[1,2,3],0)
        compare(baseline[1]['after-setup-rng.pt'],candidate[1]['after-setup-rng.pt'])
        rows.append({'rank':rank,'stage':'continuous',**compare_observations(baseline,candidate,[1,2,3])})
        saved=load_observation(old/'checkpoint-native'/f'rank-{rank}',[1,2],0)
        prior_restart=load_observation(old/'resume-native'/f'rank-{rank}',[3],2)
        restored=load_observation(args.root/'resume-optimized'/f'rank-{rank}',[3],2)
        compare(saved[1]['after-step-2-rng.pt'],restored[1]['after-setup-rng.pt'])
        compare(saved[1]['after-step-2-scheduler.pt'],restored[1]['after-setup-scheduler.pt'])
        rows.append({'rank':rank,'stage':'resume-vs-same-native-restart',**compare_observations(prior_restart,restored,[3])})
        rows.append({'rank':rank,'stage':'resume-vs-continuous-input-loss-scheduler',
                     **compare_observations(baseline,restored,[3],compare_rng=False)})
        completion=json.loads((args.root/f'resume-rank-{rank}-completion.json').read_text())
        if not completion['passed'] or not completion['native_checkpoint_loader_used']:
            raise ValueError('Full new candidate restore/next-update audit missing')
    peers=[load_observation(args.root/'semantics-optimized'/f'rank-{rank}',[1,2,3],0) for rank in (0,1)]
    for step in (1,2,3):
        compare(peers[0][1][f'actual-input-{step}.pt'],peers[1][1][f'actual-input-{step}.pt'])
    result={'schema':'jiuge7b-cpu-numa-semantics-verification-v1','passed':True,'runs':rows,
            'tp_actual_inputs_equal':True,'full_model_native_checkpoint_resume_verified':True,
            'baseline_observation_root':str(old),'baseline_prior_semantic_proof_sha256':hashlib.sha256(
                (old/'semantics-verification.json').read_bytes()).hexdigest(),
            'comparison_dtype':'CPU float64','atol':3e-4,'rtol':1e-3,
            'rng_restart_cpu_dataloader_difference_not_forcibly_reset':True,'formal_performance_passed':False}
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()

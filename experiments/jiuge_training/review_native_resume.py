"""CPU review of the completed real-model native checkpoint restart."""
import json
import os
from pathlib import Path


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from verify_pinning_semantics import load_observation, compare_observations, compare
    import torch
    root = Path('/home/huids25/kda-jiuge-prep-20261014/pinning-semantics-20261016-03')
    results = []
    for rank in (0, 1):
        baseline = load_observation(root / 'semantics-native' / f'rank-{rank}', [1, 2, 3], 0)
        saved = load_observation(root / 'checkpoint-native' / f'rank-{rank}', [1, 2], 0)
        resumed = load_observation(root / 'resume-native' / f'rank-{rank}', [3], 2)
        compare(saved[1]['after-step-2-rng.pt'], resumed[1]['after-setup-rng.pt'])
        compare(saved[1]['after-step-2-scheduler.pt'], resumed[1]['after-setup-scheduler.pt'])
        results.append({'rank': rank, 'saved_rng_restored_exact': True,
                        'saved_scheduler_restored_exact': True,
                        **compare_observations(baseline, resumed, [3], compare_rng=False)})
    result = {'schema': 'jiuge7b-native-resume-semantic-review-v1', 'runs': results,
              'passed': True, 'cuda_initialized': torch.cuda.is_initialized(),
              'optimized_full_state_resume_passed': False,
              'formal_performance_passed': False}
    if result['cuda_initialized']:
        raise ValueError('CPU verification unexpectedly initialized CUDA')
    with (root / 'native-resume-semantic-review.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

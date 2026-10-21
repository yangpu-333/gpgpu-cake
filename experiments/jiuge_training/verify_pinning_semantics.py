"""CPU verification of actual batches, losses, RNG, scheduler and native resume."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def compare(left, right, *, tolerant=False, path='root'):
    import torch
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError('Tensor structure differs: ' + path)
        if tolerant and left.is_floating_point():
            x, y = left.double(), right.double()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all() or not torch.allclose(x, y, atol=3e-4, rtol=1e-3):
                raise ValueError('Nonfinite or numerical difference: ' + path)
            return float((x-y).abs().max()) if x.numel() else 0.0
        if not torch.equal(left, right):
            raise ValueError('Exact tensor differs: ' + path)
        return 0.0
    if type(left) is not type(right):
        raise ValueError('State type differs: ' + path)
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise ValueError('State keys differ: ' + path)
        return max((compare(left[key], right[key], tolerant=tolerant, path=f'{path}/{key}') for key in left), default=0.0)
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError('State length differs: ' + path)
        return max((compare(x, y, tolerant=tolerant, path=f'{path}/{index}')
                    for index, (x, y) in enumerate(zip(left, right))), default=0.0)
    if isinstance(left, float) and (not math.isfinite(left) or not math.isfinite(right)):
        raise ValueError('Nonfinite state: ' + path)
    if left != right:
        raise ValueError('Exact state differs: ' + path)
    return 0.0


def load_observation(folder, expected_steps, initial_iteration):
    import torch
    record = json.loads((folder / 'semantics.json').read_text())
    if not record['run_returned_successfully'] or record['completed_native_steps'] != expected_steps:
        raise ValueError('Incomplete actual training semantics: ' + str(folder))
    if record['initial_iteration'] != initial_iteration or record['initial_consumed_train_samples'] != initial_iteration:
        raise ValueError('Unexpected iteration or consumed sample position')
    if len(record['actual_inputs']) != len(expected_steps) or [r['step'] for r in record['loss_calls']] != expected_steps:
        raise ValueError('Missing actual input or per-token loss coverage')
    files = {}
    for item in [*record['files'], *({'file': r['raw_file'], 'sha256': r['raw_sha256']} for r in record['actual_inputs'])]:
        name = item['file']
        if Path(name).name != name or name in files:
            raise ValueError('Duplicate or unsafe raw semantic file')
        path = folder / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Actual semantic original SHA differs: ' + str(path))
        files[name] = torch.load(path, map_location='cpu', weights_only=True)
    for item in record['actual_inputs']:
        if not item['loss_mask_all_ones'] or not item['labels_shift_matches_tokens_inside_sequence'] or item['valid_loss_tokens'] != 2048:
            raise ValueError('Not the fixed full-document CPT input contract')
        if not {'tokens', 'labels', 'position_ids', 'loss_mask', 'attention_mask'} <= set(files[item['raw_file']]):
            raise ValueError('Complete actual input/causal mask required')
    return record, files


def compare_observations(native, observed, steps, compare_rng=True):
    before, after = native[1], observed[1]
    errors = []
    reference_indices = {step: index for index, step in enumerate(native[0]['completed_native_steps'], 1)}
    for index, step in enumerate(steps, 1):
        ref_index = reference_indices[step]
        compare(before[f'actual-input-{ref_index}.pt'], after[f'actual-input-{index}.pt'])
        errors.append(compare(before[f'loss-call-{ref_index}.pt'], after[f'loss-call-{index}.pt'], tolerant=True))
        if compare_rng:
            for when in ('before', 'after'):
                compare(before[f'{when}-step-{step}-rng.pt'], after[f'{when}-step-{step}-rng.pt'])
        compare(before[f'after-step-{step}-scheduler.pt'], after[f'after-step-{step}-scheduler.pt'])
    return {'native_steps': steps, 'per_token_loss_max_abs': max(errors, default=0.0),
            'inputs_masks_scheduler_exact': True, 'rng_exact_checked': compare_rng, 'passed': True}


def verify(root, resume_campaign):
    results = []
    for rank in (0, 1):
        baseline = load_observation(root / 'semantics-native' / f'rank-{rank}', [1, 2, 3], 0)
        for arm in ('control', 'optimized'):
            observation = load_observation(root / f'semantics-{arm}' / f'rank-{rank}', [1, 2, 3], 0)
            compare(baseline[1]['after-setup-rng.pt'], observation[1]['after-setup-rng.pt'])
            results.append({'arm': arm, 'rank': rank, **compare_observations(baseline, observation, [1, 2, 3])})
        saved = load_observation(root / 'checkpoint-native' / f'rank-{rank}', [1, 2], 0)
        results.append({'arm': 'native-checkpoint-save', 'rank': rank, **compare_observations(baseline, saved, [1, 2])})
        native_resumed = load_observation(root / 'resume-native' / f'rank-{rank}', [3], 2)
        resumed = load_observation(root / 'resume-optimized' / f'rank-{rank}', [3], 2)
        # Restored RNG must match the saved native RNG before iterator startup.
        # After startup, compare the two unchanged restart paths exactly.
        for observation in (native_resumed, resumed):
            compare(saved[1]['after-step-2-rng.pt'], observation[1]['after-setup-rng.pt'])
            compare(saved[1]['after-step-2-scheduler.pt'], observation[1]['after-setup-scheduler.pt'])
        results.append({'arm': 'native-resume-vs-continuous', 'rank': rank,
                        **compare_observations(baseline, native_resumed, [3], compare_rng=False),
                        'reason': 'CPU DataLoader iterator startup is separately observed; exact RNG restoration and same-restart RNG remain required'})
        results.append({'arm': 'optimized-vs-native-resume', 'rank': rank,
                        **compare_observations(native_resumed, resumed, [3])})
        receipt = json.loads((resume_campaign / f'resume-rank-{rank}-completion.json').read_text())
        if not receipt['passed'] or not receipt['native_checkpoint_loader_used']:
            raise ValueError('Full real-model native checkpoint restore/next update did not pass')
    # TP peers must consume identical actual input bytes, independently of candidate comparisons.
    peers = [load_observation(root / 'semantics-native' / f'rank-{rank}', [1, 2, 3], 0) for rank in (0, 1)]
    for step in (1, 2, 3):
        compare(peers[0][1][f'actual-input-{step}.pt'], peers[1][1][f'actual-input-{step}.pt'])
    return {'schema': 'jiuge7b-pinning-semantics-verification-v1', 'root': str(root.resolve()),
            'runs': results, 'passed': True, 'full_model_native_checkpoint_resume_verified': True,
            'tp_actual_inputs_equal': True, 'comparison_dtype': 'CPU float64', 'atol': 3e-4, 'rtol': 1e-3,
            'formal_performance_passed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--resume-campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.root, args.resume_campaign)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)

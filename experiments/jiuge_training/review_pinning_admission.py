"""Review the saved real-model pinning admission; never promote partial audits."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(1048576):
            value.update(data)
    return value.hexdigest()


def review(root):
    import torch
    baseline = root / 'native-smoke-s2048-01'
    current = root / 'pinning-optimized-admission-01'
    campaign = root / 'pinning-validation-20261015-02'
    archive = json.loads((root / 'archive-verification-pinning-optimized-admission-01.json').read_text())
    if not archive['passed']:
        raise ValueError('Admission originals not archived')
    rows, ranks = [], []
    for rank in (0, 1):
        pin = json.loads((campaign / f'optimized-rank-{rank}-actual-pinning-init-2.json').read_text())
        if pin['parameter_count'] != 199 or pin['pinned_count'] != 199 or pin['pinned_elements'] != 3834518016:
            raise ValueError('Actual master parameter pinning coverage differs')
        entry = json.loads((current / f'rank-{rank}/entry.json').read_text())
        if entry['status'] != 'completed' or entry['completed_native_steps'] != [
                {'step': i, 'skipped': 0} for i in (1, 2, 3)]:
            raise ValueError('Three real update steps incomplete')
        for opt in entry['actual_optimizer_state']['optimizers']:
            if opt['adam_state_entries'] != 199 or opt['min_step'] != 3 or opt['max_step'] != 3:
                raise ValueError('AdamW state update incomplete')
        old = torch.load(baseline / f'rank-{rank}/optimizer-state.samples.pt', map_location='cpu', weights_only=True)
        new = torch.load(current / f'rank-{rank}/optimizer-state.samples.pt', map_location='cpu', weights_only=True)
        if len(old) != len(new):
            raise ValueError('Selected state sample count differs')
        for ordinal, (a, b) in enumerate(zip(old, new)):
            if set(a) != set(b):
                raise ValueError('Selected sample state keys differ')
            for field in a:
                x, y = a[field].double(), b[field].double()
                rows.append({'rank': rank, 'sample': ordinal, 'field': field, 'numel': x.numel(),
                             'passed': bool(torch.allclose(x, y, atol=3e-4, rtol=1e-3, equal_nan=False)),
                             'max_abs': float((x-y).abs().max())})
        ranks.append({'rank': rank, 'actual_cpu_master_parameters_pinned': 199,
                      'actual_cpu_master_elements_pinned': pin['pinned_elements'],
                      'completed_steps': 3, 'gpu_peak_allocated_bytes': entry['gpu_peak_allocated_bytes']})
    probe_dir = root / 'pinning-validation-20261015-01/optimizer-diagnostic-03'
    probe = json.loads((probe_dir / 'receipt.json').read_text())
    for name, expected in probe['files_sha256'].items():
        if digest(probe_dir / name) != expected:
            raise ValueError('Small diagnostic raw file differs')
    losses = {}
    for label, case in (('native', baseline), ('optimized', current)):
        losses[label] = re.findall(r'lm loss:\s*([0-9.Ee+\-]+)', (case / 'stdout.log').read_text())
    return {'schema': 'jiuge7b-pinning-admission-review-v1', 'real_model_ranks': ranks,
            'selected_state_samples_compared_on_cpu_float64': rows,
            'selected_samples_passed': all(r['passed'] for r in rows),
            'rounded_stdout_losses': losses, 'rounded_stdout_losses_equal': losses['native'] == losses['optimized'],
            'rounded_logs_are_not_full_precision_loss_audit': True,
            'small_actual_hdo_three_step_and_resume_passed': probe['passed'],
            'small_diagnostic_raw_copies_sha256_verified': True,
            'admission_archive': archive,
            'full_model_all_gradient_and_optimizer_state_equivalence_passed': False,
            'full_model_checkpoint_resume_tested': False, 'formal_performance_passed': False,
            'partial_raw_audits_are_not_promoted': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.results_root)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'selected_state_samples_compared_on_cpu_float64'}), flush=True)

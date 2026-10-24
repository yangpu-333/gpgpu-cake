"""Recompute unchanged native timers with the incremental pinned arm identities."""
from sft_phased_acceptance_20261024 import PREP, read, replace_exact, evaluate_incremental


def analyze(round_label):
    source = replace_exact((PREP/'analyze_pinning_formal.py').read_text(), [
        ("(199 if arm == 'optimized' else 0)", '199'),
    ])
    scope = {'__name__': 'incremental_formal_analysis'}
    exec(compile(source, '<incremental-pinned-formal-analysis>', 'exec'), scope)
    scope['evaluate'] = evaluate_incremental
    result = scope['analyze'](PREP, round_label)
    # Bind all 120 actual samples/mask counts and every logged loss, not just timers.
    for seed in (63100,63101,63102):
        samples, losses = [], []
        for arm in ('native','control','optimized'):
            root = PREP/'runs'/f'pinning-formal-{round_label}-{seed}-{arm}'
            obj = [read(root/f'sft-objective-rank-{rank}.json') for rank in (0,1)]
            assert len(obj[0]['supervision_rows']) == 120 and not obj[1]['supervision_rows']
            samples.append(obj[0]['supervision_rows'])
            import re
            rows = re.findall(r'iteration\s+(\d+)\s*/\s*120.*?lm loss:\s*([\d.Ee+\-]+)', (root/'stdout.log').read_text())
            assert len(rows) == 13 and len({x[0] for x in rows}) == 13
            losses.append(rows)
        assert samples[0] == samples[1] == samples[2]
        assert losses[0] == losses[1] == losses[2]
    result.update(incremental_baseline='accepted pinned, unbound',
                  actual_120_sample_and_mask_rows_equal=True, all_logged_loss_rows_equal=True,
                  candidate_requires_own_admission=True)
    return result

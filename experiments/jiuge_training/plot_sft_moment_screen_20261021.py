"""Standalone diagnostic figure; no formal-performance claim."""
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent


def main():
    value = json.loads((BASE/'results/sft-moment-screen-independent-review-v2-20261021.json').read_text(encoding='utf-8'))
    if not value['passed'] or value['formal_performance_passed']:
        raise ValueError('Verified diagnostic-only result required')
    names = list(value['policies'])
    colors = ['#64748b', '#2563eb', '#0d9488'][:len(names)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    metrics = ['diagnostic_slow_rank_mean_seconds_per_step', 'CPU_AdamW_seconds_per_rank_step']
    titles = ['Profiled training window (slow rank)', 'CPU AdamW functional range']
    for index, (ax, metric, title) in enumerate(zip(axes, metrics, titles)):
        samples = [[r[metric] for r in (value['cases'] if index == 0 else value['windows']) if r['policy'] == n] for n in names]
        means = [statistics.mean(r) for r in samples]
        ax.bar(names, means, color=colors, alpha=.85, width=.6)
        ax.errorbar(range(len(names)), means,
                    yerr=[[m-min(r) for m, r in zip(means, samples)],
                          [max(r)-m for m, r in zip(means, samples)]],
                    fmt='none', ecolor='#172554', capsize=5)
        for x, (mean, rows) in enumerate(zip(means, samples)):
            for j, observation in enumerate(rows):
                offset = ((j/(len(rows)-1))-.5)*.22 if len(rows) > 1 else 0
                ax.scatter(x+offset, observation, color='#172554', s=15, zorder=3)
            ax.text(x, max(rows)+max(means)*.055, f'{mean:.3f}s', ha='center', fontsize=10)
        ax.set_ylim(0, max(max(r) for r in samples)*1.23)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel('Seconds per step')
        ax.spines[['right', 'top']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('Jiuge 7B SFT: CPU-local AdamW moment storage', fontsize=14)
    fig.supxlabel('Diagnostic only; 2 runs / policy, seed63100. Whiskers: observed min-max, not confidence intervals.', fontsize=9)
    output = BASE/'results/sft-moment-screen-comparison-20261021'
    for suffix in ('.png', '.svg'):
        fig.savefig(output.with_suffix(suffix), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()

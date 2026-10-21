"""Plot separate host/device interval accounts from real saved trace analysis."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    labels, host_rows, gpu_rows = [], [], []
    for seq, case in ((128, 'native-profile-256g-01'), (2048, 'native-profile-s2048-01')):
        value = json.loads((args.results / case / 'trace-analysis.json').read_text())
        for row in value['windows']:
            device = row['analysis']['capture_window']['devices'][0]
            width = device['window_us']
            fb = row['cpu_phase_ranges']['jiuge/forward_backward']['host_range_sum_us']
            opt = row['cpu_phase_ranges']['jiuge/optimizer_update']['host_range_sum_us']
            host_rows.append(np.array([fb, opt, max(0, width - fb - opt)]) / width * 100)
            compute = device['interval_union_us']['compute']
            comm = device['interval_union_us']['collective']
            overlap = device['compute_collective_overlap_us']
            kernel_union = compute + comm - overlap
            busy = device['gpu_busy_union_us']
            gpu_rows.append(np.array([compute - overlap, overlap, comm - overlap,
                                      max(0, busy - kernel_union), max(0, width - busy)]) / width * 100)
            labels.append(f'S{seq}  W{row["window"][-1]}  r{row["rank"]}  ({width / 1e6:.2f}s)')
    host, gpu = np.array(host_rows), np.array(gpu_rows)
    if not np.allclose(host.sum(axis=1), 100) or not np.allclose(gpu.sum(axis=1), 100):
        raise ValueError('Interval partitions must sum to one window')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.6), sharey=True)
    y = np.arange(len(labels))
    for axis, values, names, colors in (
        (axes[0], host, ['Forward/backward range', 'Optimizer/offload range', 'Other host ranges'],
         ['#4c87bd', '#e8a34d', '#d9dee5']),
        (axes[1], gpu, ['Compute only', 'Compute + NCCL', 'NCCL only',
                        'Copy/memset outside kernels', 'No recorded activity'],
         ['#4385b9', '#3a9d88', '#9577b8', '#e8a34d', '#d9dee5'])):
        left = np.zeros(len(labels))
        for column, name, color in zip(values.T, names, colors):
            axis.barh(y, column, left=left, height=.64, color=color, label=name)
            left += column
        axis.set_xlim(0, 100)
        axis.set_xlabel('Share of the same 5-step window (%)')
        axis.set_axisbelow(True)
        axis.grid(axis='x', alpha=.2)
        axis.legend(loc='upper left', bbox_to_anchor=(0, -.12), frameon=False, fontsize=9, ncol=1)
        axis.axhline(3.5, color='#aab2bd', linewidth=.8)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_title('Host wall-time phase ranges')
    axes[1].set_title('Device activity interval partition')
    fig.suptitle('Jiuge 7B continued pretraining | 2 x A100 | TP2 | BF16 | CPU optimizer offload',
                 fontsize=13, x=.53, y=.98)
    fig.text(.53, .925, 'Separate host and device accounts; do not add the panels. Profiled data is not speedup acceptance.',
             ha='center', fontsize=9, color='#485260')
    fig.text(.17, .015, 'GPU annotation spans are excluded from device work. NCCL duration can include peer waiting. Collector drop count is unknown.',
             fontsize=8, color='#485260')
    fig.subplots_adjust(left=.17, right=.985, top=.86, bottom=.29, wspace=.12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor='white')
    fig.savefig(args.output.with_suffix('.svg'), facecolor='white')
    plt.close(fig)
    print(str(args.output.resolve()))


if __name__ == '__main__':
    main()

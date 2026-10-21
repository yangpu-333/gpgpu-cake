"""Standalone scientific plots from bounded aggregate and binned trace keys."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE=Path(__file__).resolve().parent
RESULTS=BASE/'results'


def main():
    review=json.loads((RESULTS/'sft-paired-trace-independent-review-v2-20261020.json').read_text(encoding='utf-8'))
    arms=review['arms']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    groups=[('Host ranges / CUDA API spans (s per rank-step)',[
                ('CPU optimizer','CPU_optimizer_seconds_per_rank_step'),
                ('CPU AdamW math','CPU_AdamW_math_seconds_per_rank_step'),
                ('HtoD CPU API','CPU_HtoD_API_seconds_per_rank_step')]),
            ('GPU interval unions (s per rank-step)',[
                ('Compute','GPU_compute_seconds_per_rank_step'),('Memcpy','GPU_memcpy_seconds_per_rank_step'),
                ('NCCL','GPU_NCCL_seconds_per_rank_step')]),
            ('Overlap and captured GPU gaps (%)',[
                ('NCCL overlapping compute','NCCL_overlap_compute_fraction'),
                ('Memcpy overlapping compute','memcpy_overlap_compute_fraction'),
                ('Captured GPU gap proxy','GPU_idle_proxy_fraction')])]
    fig,axes=plt.subplots(1,3,figsize=(15,5.2),layout='constrained')
    for panel,(title,metrics) in zip(axes,groups):
        positions=np.arange(len(metrics))
        for offset,arm,color,label in [(-.18,'native','#64748b','Native'),(.18,'optimized','#087e8b','Pinned repair')]:
            factor=100 if metrics[0][1].endswith('fraction') else 1
            means=np.array([arms[arm][key]['mean']*factor for _,key in metrics])
            low=means-np.array([arms[arm][key]['min']*factor for _,key in metrics])
            high=np.array([arms[arm][key]['max']*factor for _,key in metrics])-means
            bars=panel.barh(positions+offset,means,height=.32,color=color,label=label,
                            xerr=np.stack((low,high)),error_kw={'elinewidth':1,'capsize':2,'alpha':.7})
            panel.bar_label(bars,labels=[f'{v:.4f}' if 0<v<.01 else f'{v:.2f}' for v in means],padding=3,fontsize=8)
        panel.set_yticks(positions,[label for label,_ in metrics])
        panel.invert_yaxis()
        panel.set_title(title,fontsize=10,pad=12)
        panel.set_xlim(left=0,right=panel.get_xlim()[1]*1.13)
        panel.grid(axis='x',alpha=.16)
        panel.set_axisbelow(True)
    axes[0].legend(loc='lower right',frameon=False)
    fig.suptitle('Jiuge 7B SFT | paired native vs pinned-master repair | 2 x A100',fontsize=15)
    fig.text(.5,-.035,'ABBA: 2 runs/arm, same seed; 2 five-step windows x 2 ranks/run. Bars: mean; whiskers: observed min/max.\nHost and GPU categories overlap and must not be added. Profiling is separate from formal throughput acceptance.',ha='center',fontsize=9,color='#475569')
    fig.savefig(RESULTS/'sft-paired-trace-comparison-20261020.png',dpi=160,bbox_inches='tight')
    fig.savefig(RESULTS/'sft-paired-trace-comparison-20261020.svg',bbox_inches='tight')
    plt.close(fig)
    data=json.loads((RESULTS/'sft-paired-trace-lanes-20261020.json').read_text(encoding='utf-8'))
    rows=[r for r in data['rows'] if r['window']=='window-02' and r['tag'] in ('a1','b1')]
    rows.sort(key=lambda r:({'native':0,'optimized':1}[r['arm']],r['rank']))
    maximum=max(r['window_seconds'] for r in rows)
    fig,axes=plt.subplots(4,1,figsize=(13,8),sharex=True,layout='constrained')
    labels=['CPU optimizer','CPU AdamW math','GPU compute','GPU memcpy','GPU NCCL']
    for ax,row in zip(axes,rows):
        values=np.array([row['lanes'][k] for k in labels])
        edges=np.minimum(np.arange(values.shape[1]+1)*data['bin_seconds'],row['window_seconds'])
        im=ax.pcolormesh(edges,np.arange(6)-.5,values,cmap='Blues',vmin=0,vmax=1,shading='flat')
        ax.set_ylim(4.5,-.5)
        ax.set_yticks(range(5),labels,fontsize=9)
        ax.set_title(f'{row["arm"].capitalize()} | rank {row["rank"]} | updates 31-35 | {row["window_seconds"]:.2f}s',loc='left',fontsize=10)
        ax.set_xlim(0,maximum)
    axes[-1].set_xlabel('Seconds from each panel capture start (display alignment only)')
    fig.colorbar(im,ax=axes,label='Fraction active within each 100 ms bin',shrink=.7,pad=.02)
    fig.suptitle('Observed training-window lanes | host and device activity can overlap',fontsize=14)
    fig.savefig(RESULTS/'sft-paired-trace-lanes-20261020.png',dpi=160,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__': main()

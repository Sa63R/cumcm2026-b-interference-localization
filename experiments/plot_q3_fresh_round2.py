"""Standalone scientific figure from completed, audited paired results."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    report = json.loads(args.analysis.read_text(encoding='utf-8'))
    with args.csv.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    groups = [('nominal','Nominal: 64 new cases'), ('pressure','Pressure: 32 new cases'),
              ('reconstruction','Reconstruction: 12 new groups')]
    fig, axes = plt.subplots(1,3,figsize=(13.2,4.8),sharey=True)
    for ax, (group,title) in zip(axes,groups):
        selected = [r for r in rows if r['suite_group']==group and r['strategy'] in ('B','CR')
                    and r['audit_passed']=='True' and r['success']=='True'
                    and r['deduplication_status'] not in ('conflict','consistent_duplicate')]
        baseline = {r['case_id']:r for r in selected if r['strategy']=='B'}
        paired = defaultdict(list)
        for r in selected:
            if r['strategy']!='CR' or r['case_id'] not in baseline:
                continue
            b = baseline[r['case_id']]
            key = 'observation_robust_guarantee_lower_bound_s' if group=='reconstruction' else 'guarantee_lower_bound_s'
            paired[r['original_case_group']].append((float(b['virtual_time_s'])/float(b[key]),
                float(r['virtual_time_s'])-float(b['virtual_time_s']),int(r['source_total'])))
        points = [tuple(statistics.mean(p[i] for p in values) for i in range(3)) for values in paired.values()]
        if not points:
            raise ValueError('No complete paired data for '+group)
        x,y,n = zip(*points)
        ax.scatter(x,y,c=n,cmap='viridis',norm=Normalize(10,16),s=44,edgecolors='white',linewidths=.55,zorder=3)
        pair = report['groups'][group]['paired']['B_vs_CR']
        mean = pair['mean_group_delta_s']; low,high = pair['paired_group_bootstrap_95_delta_s']
        ax.axhspan(low,high,color='#246BCE',alpha=.09)
        ax.axhline(mean,color='#246BCE',lw=1.4,label='Paired mean and 95% interval')
        ax.axhline(0,color='#333333',lw=1,linestyle='--')
        ax.set_title(title,fontsize=11,fontweight='bold')
        ax.text(.04,.97,f'Mean {mean:+.1f} s\n95% [{low:+.1f}, {high:+.1f}]',transform=ax.transAxes,
                va='top',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
        ax.set_xlabel('B / guarantee lower bound'+ ('*' if group=='reconstruction' else ''),fontsize=9)
        ax.grid(alpha=.14)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('CR - B completion time (s)\nNegative values favour CR',fontsize=10)
    fig.subplots_adjust(left=.08,right=.90,bottom=.20,top=.86,wspace=.12)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(10,16),cmap='viridis'),cax=fig.add_axes([.93,.24,.015,.52]))
    cb.set_label('Number of sources',fontsize=9);cb.set_ticks(range(10,17))
    fig.suptitle('Frozen candidate CR: paired outcomes and individual regressions',fontsize=14,fontweight='bold',y=.97)
    fig.text(.08,.055,'Each dot is one original case group. Reconstruction averages its two radius variants.\n'
             '* Reconstruction uses the observation-robust guarantee bound. Intervals bootstrap original groups; research distributions only.',fontsize=8,color='#444444')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out,dpi=180,facecolor='white')
    fig.savefig(args.out.with_suffix('.svg'),facecolor='white')
    plt.close(fig)
    print('saved',args.out)


if __name__=='__main__':
    main()

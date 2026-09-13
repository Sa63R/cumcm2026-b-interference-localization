"""Export scientific plots; all statistics come from saved ablation results."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

s=json.loads((ROOT/'summary.json').read_text())
m=s['groups']['new_practice']['methods']
dest=ROOT/'figures';dest.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
names=['no_route','no_transit','no_guard','no_rollout','no_rb']
labels=['Learned route ranking','Extra transit sensing','Early four-station guard',
        'Local posterior rollout','Posterior shared-sensing score']
fig,ax=plt.subplots(figsize=(10,4.8))
for i,name in enumerate(names):
    item=m[name];lo,hi=item['component_benefit_ci99_bonferroni5'];value=item['change_vs_full_s']
    color='#087e8b' if lo>0 else ('#bd433e' if hi<0 else '#63748b')
    ax.errorbar(value,i,xerr=[[max(0,value-lo)],[max(0,hi-value)]],fmt='o',color=color,
                capsize=5,markersize=7,elinewidth=2)
    ax.annotate(f'{value:+.2f}',(value,i),xytext=(0,12),textcoords='offset points',ha='center',fontsize=10)
ax.axvline(0,color='#8d96a3',linewidth=1,linestyle='--')
ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set_ylim(4.6,-.65)
ax.set_xlabel('Additional task time after removing the component (seconds/source)')
ax.set_title('V6 ablation: 300 new paired scenarios',loc='left',pad=18,fontweight='bold')
ax.grid(axis='x',alpha=.2)
fig.text(.5,.015,'Bars: 99% paired bootstrap intervals; Bonferroni family coverage 95% for five components.\nPositive values favor retaining the component. Local reconstruction only.',ha='center',fontsize=9,color='#566171')
fig.tight_layout(rect=[0,.1,1,1]);fig.savefig(dest/'component_effects.png',dpi=180);fig.savefig(dest/'component_effects.pdf');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4.7))
panels=[(axes[0],[['v5','no_route'],['no_transit','full']],
         'Transit sensing','Learned route ranking','Route and transit interaction (guard on)'),
        (axes[1],[['no_rollout_rb','no_rollout'],['no_rb','full']],
         'Posterior shared-sensing score','Local rollout','Rollout and sharing interaction')]
for ax,keys,xlabel,ylabel,title in panels:
    values=np.array([[m[key]['mean_seconds_per_source'] for key in row] for row in keys])
    ax.imshow(values,cmap='Blues',vmin=values.min()-.1,vmax=values.max()+.1)
    for i in range(2):
        for j in range(2):
            ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=20,
                    color='white' if values[i,j]>(values.min()+values.max())/2 else '#152b49')
    ax.set_xticks([0,1],['Off','On']);ax.set_yticks([0,1],['Off','On'])
    ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.set_title(title,fontsize=12,pad=15)
fig.suptitle('Complete task seconds per source (lower is better)',fontweight='bold',y=.98)
fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(dest/'factorial_means.png',dpi=180);fig.savefig(dest/'factorial_means.pdf');plt.close(fig)
print(dest)

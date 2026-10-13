"""Standalone beyond-horizon relay figure; source-counterfactual performance included."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/attention-pilot-matplotlib')
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_recurrent import check_pairs
ROOT=Path(__file__).parent
parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,default=ROOT/'results-recurrent')
args=parser.parse_args()
D=json.loads((args.input_dir/'runs.json').read_text());cfg=D['config'];rows=D['results']
check_pairs(rows,cfg,complete=True)
seeds=cfg['seeds'];distances=np.array(cfg['distances'])
def select(c):return [next(r for r in rows if r['task']=='probe' and r['condition']==c and r['seed']==s) for s in seeds]
colors={'full-128':'#707070','sliding-8':'#2863b5','recurrent-8':'#dc7028'}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(14,4.8))
for ax,field,title in [(axes[0],'accuracy','Held-out retrieval'),(axes[2],'counterfactual_accuracy','Changed-source retrieval')]:
    ax.axvspan(56,84,color='#eeeeee',zorder=0)
    ax.axvline(56,color='#999999',lw=1,ls=':')
    ax.axhline(12.5,color='#999999',lw=1,ls='--')
    for c in ('full-128','sliding-8','recurrent-8'):
        ys=np.array([[r['rolling_metrics'][str(d)][field]*100 for d in distances] for r in select(c)])
        for y in ys:ax.plot(distances,y,color=colors[c],alpha=.13,lw=.8)
        ax.plot(distances,ys.mean(0),color=colors[c],lw=2.3,marker='o',label=c)
    ax.set(title=title,xlabel='Source-to-query distance (tokens)',ylabel='Accuracy (%)',ylim=(-3,103),xlim=(4,84),xticks=distances)
    ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=9)
a=np.array([r['rolling_metrics']['80']['accuracy']*100 for r in select('sliding-8')]);b=np.array([r['rolling_metrics']['80']['accuracy']*100 for r in select('recurrent-8')])
for i,(v,w) in enumerate(zip(a,b)):
    axes[1].plot([0,1],[v,w],color='#555555',alpha=.4,lw=1)
    axes[1].scatter([0,1],[v,w],color=[colors['sliding-8'],colors['recurrent-8']],s=35,zorder=3)
axes[1].scatter([0,1],[a.mean(),b.mean()],marker='D',color=['#123968','#9f4914'],s=90,zorder=4)
axes[1].axhline(12.5,color='#999999',ls='--',lw=1)
axes[1].set(title='Distance 80: paired seeds',ylabel='Accuracy (%)',xticks=[0,1],xticklabels=['Sliding-8','Recurrent-8'],ylim=(-3,103),xlim=(-.25,1.25));axes[1].grid(axis='y',alpha=.15)
fig.suptitle('Same-layer recurrence: testing relay beyond the sliding-window horizon',fontsize=16)
fig.text(.5,.88,'128-token sequences · 8 layers · W=8 baseline horizon: 56 · 6 paired seeds · unchanged training budgets',ha='center',fontsize=10)
fig.text(.5,.025,'Thin lines/points: individual seeds. Thick lines/diamonds: means. Dashed horizontal lines: 12.5% chance.\nShaded distance region exceeds the baseline horizon. Changed-source answers must follow the modified value.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.12,1,.86))
out=ROOT/'figures';out.mkdir(exist_ok=True)
for ext in ('png','pdf','svg'):fig.savefig(out/f'recurrent-beyond-horizon.{ext}',dpi=200,facecolor='white')
print('Saved figures/recurrent-beyond-horizon.png, .pdf, .svg')

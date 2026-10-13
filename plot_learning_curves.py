"""Export standalone V1 learning curves, with paired seed variability."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/attention-pilot-matplotlib')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).parent
D=json.loads((ROOT/'results-curves/curves.json').read_text())
seeds=D['config']['seeds']; assert len(D['results'])==12
rows={c:[next(r for r in D['results'] if r['condition']==c and r['seed']==s) for s in seeds] for c in ('full-128','sliding-32')}
assert all(r['exact_v1_reproduction'] for rs in rows.values() for r in rs)
x=np.array([p['processed_tokens'] for p in rows['full-128'][0]['curve']])/1000
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
fig,axes=plt.subplots(1,3,figsize=(14,4.5),gridspec_kw={'width_ratios':[1,1,1.05]})
colors={'full-128':'#2863b5','sliding-32':'#dc7028'}
labels={'full-128':'Full attention (128)','sliding-32':'Sliding window (32)'}
for ax,field,title in zip(axes[:2],('train_eval_nll','validation_nll'),('Training-set loss','Held-out validation loss')):
    for c in rows:
        y=np.array([[p[field] for p in r['curve']] for r in rows[c]])
        mean=y.mean(0); sd=y.std(0,ddof=1)
        ax.plot(x,mean,label=labels[c],color=colors[c],lw=2.3)
        ax.fill_between(x,mean-sd,mean+sd,color=colors[c],alpha=.15,lw=0)
    ax.set(title=title,xlabel='Processed training tokens (thousands)',ylabel='Next-character NLL (nats)')
    ax.grid(alpha=.2); ax.set_xlim(0,x[-1]); ax.legend(frameon=False,fontsize=9)
full=np.array([[p['validation_nll'] for p in r['curve']] for r in rows['full-128']])
sliding=np.array([[p['validation_nll'] for p in r['curve']] for r in rows['sliding-32']])
delta=sliding-full; mean=delta.mean(0); sd=delta.std(0,ddof=1)
axes[2].axhline(0,color='#777777',lw=1,ls='--')
axes[2].plot(x,mean,color='#6b4799',lw=2.3)
axes[2].fill_between(x,mean-sd,mean+sd,color='#6b4799',alpha=.18,lw=0)
axes[2].set(title='Paired validation difference',xlabel='Processed training tokens (thousands)',ylabel='Sliding − full NLL (nats)',xlim=(0,x[-1]))
axes[2].grid(alpha=.2)
axes[2].text(.96,.84,f'Final: {mean[-1]:.4f} ± {sd[-1]:.4f}\nLower favors sliding',transform=axes[2].transAxes,ha='right',fontsize=10)
fig.suptitle('V1: full attention vs. sliding-window training',fontsize=17,y=.99)
fig.text(.5,.90,'128-token sequences · 8 layers · 6 paired seeds · identical initialization, batches and token budget',ha='center',fontsize=11)
fig.text(.5,.025,'Lines: six-seed means. Shading: ±1 sample SD across seeds. Fixed 512-sequence sets scored every 20 updates.\nInference KV storage: 381 KiB (full) vs. 93 KiB (sliding); training-memory savings are not measured.',ha='center',fontsize=10,color='#444444')
fig.tight_layout(rect=(0,.12,1,.88))
out=ROOT/'figures';out.mkdir(exist_ok=True)
for ext in ('png','pdf','svg'):fig.savefig(out/f'v1-learning-curves.{ext}',dpi=200)
print('Saved figures/v1-learning-curves.png, .pdf, .svg')

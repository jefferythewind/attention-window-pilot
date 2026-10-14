"""Retrieval figure: new architecture with preserved control metrics."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/attention-pilot-matplotlib')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent
rows=[]
for directory,names in [('results-w32-horizon',{'sliding-32'}),('results-lateral-slots',{'recurrent-1slot-diagonal','recurrent-2slot-diagonal'}),('results-three-slots',{'recurrent-3slot-diagonal'}),('results-two-sweep',{'two-sweep-top-recurrent-w32'})]:
    rows += [r for r in json.loads((ROOT/directory/'runs.json').read_text())['results'] if r['condition'] in names]
labels=['Sliding','Interleaved 1','Interleaved 2','Interleaved 3','Two-sweep top']
colors=['#777777','#2863b5','#dc7028','#35845b','#8847ad']
fig,axes=plt.subplots(1,2,figsize=(11,4.8));x=np.arange(4);distances=[8,24,48,280]
for ax,field,title in zip(axes,['accuracy','counterfactual_accuracy'],['Held-out retrieval','Changed-source retrieval']):
    for i,r in enumerate(rows):
        ax.bar(x+(i-2)*.15,[100*r['rolling_metrics'][str(d)][field] for d in distances],width=.14,color=colors[i],label=labels[i])
    ax.axhline(12.5,color='#888888',ls='--');ax.axvspan(2.55,3.5,color='#eeeeee',alpha=.3,zorder=0)
    ax.set(title=title,xlabel='Source-to-query distance',ylabel='Accuracy (%)',xticks=x,xticklabels=distances,ylim=(0,105));ax.grid(axis='y',alpha=.15)
    ax.spines[['top','right']].set_visible(False)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.85),ncol=5,frameon=False,fontsize=9)
fig.suptitle('Separate two-sweep top-layer recurrence',fontsize=15)
fig.text(.5,.9,'One matched seed · 320-token sequences · W=32 · 400 updates · Only the new condition trained',ha='center',fontsize=10)
fig.text(.5,.025,'Dashed line: 12.5% chance. Shaded distance280 exceeds the ordinary horizon248.\nPreserved controls; one seed does not establish reliable long-term retention.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.1,1,.80))
for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'two-sweep.{ext}',dpi=200,facecolor='white')

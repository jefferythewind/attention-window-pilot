"""Matched diagonal one/two-slot pair; no seed error bars claimed."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/attention-pilot-matplotlib')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent
D=json.loads((ROOT/'results-lateral-slots/runs.json').read_text());assert len(D['results'])==2
rows=sorted(D['results'],key=lambda r:r['recurrent_attention_slots']);ds=D['config']['distances'];x=np.arange(4)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(10,4.5))
for ax,field,title in zip(axes,('accuracy','counterfactual_accuracy'),('Held-out retrieval','Changed-source retrieval')):
    for i,r in enumerate(rows):
        y=[100*r['rolling_metrics'][str(d)][field] for d in ds]
        bars=ax.bar(x+(i-.5)*.32,y,width=.3,label=f'{r["recurrent_attention_slots"]} slot'+('s' if r['recurrent_attention_slots']>1 else ''),color=['#2863b5','#dc7028'][i])
        for bar,v in zip(bars,y):ax.text(bar.get_x()+bar.get_width()/2,v+1.5,f'{v:.1f}'.removesuffix('.0'),ha='center',fontsize=8)
    ax.axhline(12.5,color='#888888',ls='--',lw=1)
    ax.set(title=title,xlabel='Source-to-query distance (tokens)',ylabel='Accuracy (%)',xticks=x,xticklabels=[str(d) for d in ds],ylim=(0,125),yticks=[0,25,50,75,100]);ax.grid(axis='y',alpha=.15);ax.legend(frameon=False,fontsize=9)
    ax.axvspan(2.55,3.5,color='#eeeeee',alpha=.35,zorder=0)
fig.suptitle('One versus two recurrent slots: diagonal execution',fontsize=15)
fig.text(.5,.87,'One paired seed (515843) · 320-token training sequences · 8 layers · 400 retrieval updates',ha='center',fontsize=10)
fig.text(.5,.025,'Same initialization, sampled batches and labels. Dashed line: 12.5% chance.\nShaded group: beyond the baseline horizon. One seed does not establish reliability.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.12,1,.85))
for ext in ('png','pdf'):fig.savefig(ROOT/'figures'/f'lateral-slots.{ext}',dpi=200,facecolor='white')

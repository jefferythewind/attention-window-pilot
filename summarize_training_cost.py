"""Training-cost report and standalone figure; CPU costs separated from inference KV."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/attention-pilot-matplotlib')
import json
from pathlib import Path
import statistics as stats
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent
D=json.loads((ROOT/'results-training-cost/benchmark.json').read_text());rs=D['results'];assert len(rs)==12
seeds=D['seeds'];groups={w:[next(r for r in rs if r['seed']==s and r['window']==w) for s in seeds] for w in (128,32)}
metrics=[('Mean training update','mean_step_seconds',1000,'Milliseconds / update'),('Peak process memory','peak_process_rss_bytes',1/2**20,'MiB (includes Python / libraries)'),('Autograd saved storage','autograd_saved_nonparameter_storage_bytes',1/2**20,'MiB (unique nonparameter storages)')]
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(12,4.5))
for ax,(title,field,scale,ylabel) in zip(axs,metrics):
    ys=[[r[field]*scale for r in groups[w]] for w in (128,32)]
    means=[stats.mean(y) for y in ys];sds=[stats.stdev(y) for y in ys]
    ax.bar([0,1],means,color=['#2863b5','#dc7028'],width=.55,alpha=.85,yerr=sds,capsize=5)
    for pair in zip(*ys):ax.plot([0,1],pair,color='#444444',lw=.7,alpha=.35,marker='o',markersize=3)
    ax.set(title=title,ylabel=ylabel,xticks=[0,1],xticklabels=['Full 128','Sliding 32']);ax.set_ylim(0,max(max(y) for y in ys)*1.18);ax.grid(axis='y',alpha=.2)
    for i,v in enumerate(means):ax.text(i,v+max(means)*.05,f'{v:.2f}',ha='center')
fig.suptitle('V1 training costs: dense-mask CPU implementation',fontsize=16)
fig.text(.5,.88,'Batch 16 × 128 tokens · 8 layers · float32 · 1 CPU thread · 6 isolated paired trials',ha='center',fontsize=10)
fig.text(.5,.025,'Timing: 10 warmup + 50 measured forward/backward/clip/AdamW updates per trial. Error bars: sample SD.\nCPU memory is distinct from inference KV storage; saved-storage counts are not total peak memory.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.12,1,.86))
out=ROOT/'figures';out.mkdir(exist_ok=True)
for ext in ('png','pdf','svg'):fig.savefig(out/f'v1-training-cost.{ext}',dpi=200,facecolor='white')
lines=['# V1 training time and CPU memory','','Six isolated paired trials, full-128 versus sliding-32. This benchmarks the current dense-mask implementation; it does not benchmark an optimized sparse attention kernel.','','| Measurement | Full 128 | Sliding 32 |','|---|---:|---:|']
for title,field,scale,unit in metrics:
    values=[[r[field]*scale for r in groups[w]] for w in (128,32)]
    lines.append('| '+title+' ('+unit+') | '+' | '.join(f'{stats.mean(v):.2f} ± {stats.stdev(v):.2f}' for v in values)+' |')
ratios=[b['mean_step_seconds']/a['mean_step_seconds'] for a,b in zip(groups[128],groups[32])]
lines+=['',f'Mean paired sliding/full update-time ratio: **{stats.mean(ratios):.3f}** (SD {stats.stdev(ratios):.3f}).', '',
        '## How to read this comparison','',
        'A smaller attention window currently changes a dense boolean mask. Every layer still processes the complete 128-token sequence, and the SDPA call receives the same Q/K/V shapes. The implementation therefore does not guarantee reduced attention work or reduced training memory. Timing differences are specific to the current CPU kernel, machine, and small shapes.', '',
        'Peak process RSS includes Python, imported libraries, the corpus, all 60 prebuilt batches, weights, optimizer state, activations and allocator behavior. Each condition gets a fresh process, avoiding contamination from the other condition, but RSS remains a coarse process-level measurement. Baseline and post-warmup RSS are recorded in raw JSON. It is not a measurement of GPU VRAM.', '',
        'Autograd saved storage is deduplicated nonparameter tensor storage captured during one forward/backward step. It includes inputs and excludes weights; it is a diagnostic of tensors retained for backward, not total memory, peak activation memory or cumulative allocation traffic. Weight and optimizer tensor sizes match exactly across paired conditions.', '',
        'Inference KV storage remains 93 KiB per sequence for W=32 versus 381 KiB for W=128 (24.4%). That saving does not imply the same reduction in training memory.', '',
        '## Protocol','',D['method'],'',
        f'PyTorch {D["torch_version"]}; CPU: {D["cpu"]}. Both conditions use identical initialization and prebuilt text batches for each seed, verified by hashes. Timing is serial, alternating condition order across seeds. All threads are set to one. The benchmark holds learning rate at 0.002; it is not a new training-quality experiment.', '',
        'Times are means over 50 steady-state updates; reported dispersion is across the six process trials. No statistical speedup claim is made. For performance per second, use the learning curves with appropriately measured end-to-end training times, rather than assuming kernel timing equals the full pipeline.', '',
        'Reproduce: `.venv/bin/python benchmark_training.py`, then `.venv/bin/python summarize_training_cost.py`. Raw results: [results-training-cost/benchmark.json](results-training-cost/benchmark.json).', '',
        '![Training cost comparison](figures/v1-training-cost.png)']
(ROOT/'RESULTS-TRAINING-COST.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:15]))

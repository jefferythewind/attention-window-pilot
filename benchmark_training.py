"""Isolated CPU training-step cost benchmark for the two V1 attention conditions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).parent

def child(seed,window):
    import torch
    from model import WindowConfig,WindowLM
    from run import digest
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
    c=json.loads((ROOT/'v1_config.json').read_text())
    chars=sorted(set((ROOT/'shakespeare.txt').read_text())); lookup={ch:i for i,ch in enumerate(chars)}
    corpus=torch.tensor([lookup[ch] for ch in (ROOT/'shakespeare.txt').read_text()]);train=corpus[:int(len(corpus)*.9)]
    model=WindowLM(WindowConfig(vocab_size=len(chars),width=48,layers=8,heads=4,window=window,max_positions=512)).train()
    initial=digest(model.state_dict());opt=c['optimizer']
    optimizer=torch.optim.AdamW([{'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':.1},{'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0}],lr=.002,betas=(.9,.95))
    g=torch.Generator().manual_seed(seed+1000)
    batches=[]
    for i in range(60):
        starts=torch.randint(0,len(train)-128,(16,),generator=g)
        batches.append(torch.stack([train[s:s+128] for s in starts]))
    batch_hash=hashlib.sha256(b''.join(x.numpy().tobytes() for x in batches)).hexdigest()
    def rss():
        fields=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
        return int(fields['VmRSS'].split()[0])*1024
    setup_rss=rss();setup_hwm=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    params={p.untyped_storage().data_ptr() for p in model.parameters()}
    saved={}
    def pack(t):
        storage=t.untyped_storage()
        if storage.data_ptr() not in params:saved[storage.data_ptr()]=storage.nbytes()
        return t
    def update(x):
        optimizer.zero_grad(set_to_none=True)
        loss=model(x,labels=x).loss;loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
    # Only record storage metadata; do not retain additional tensors or time this hook.
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):update(batches[0])
    for x in batches[1:10]:update(x)
    warm_rss=rss();times=[]
    for x in batches[10:]:
        start=time.perf_counter();update(x);times.append(time.perf_counter()-start)
    return {'seed':seed,'window':window,'initialization_sha256':initial,'batches_sha256':batch_hash,
            'warmup_steps':10,'measured_steps':50,'mean_step_seconds':statistics.mean(times),
            'median_step_seconds':statistics.median(times),'step_times_seconds':times,
            'tokens_per_second':2048/statistics.mean(times),'setup_rss_bytes':setup_rss,
            'setup_high_water_bytes':setup_hwm,'post_warmup_rss_bytes':warm_rss,
            'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'autograd_saved_nonparameter_storage_bytes':sum(saved.values()),
            'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
            'optimizer_tensor_bytes':sum(v.numel()*v.element_size() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child',action='store_true');parser.add_argument('--seed',type=int);parser.add_argument('--window',type=int);args=parser.parse_args()
    if args.child:print(json.dumps(child(args.seed,args.window)));return
    cfg=json.loads((ROOT/'v1_config.json').read_text()); rows=[]
    for i,seed in enumerate(cfg['seeds']):
        for window in ((128,32) if i%2==0 else (32,128)):
            p=subprocess.run([sys.executable,__file__,'--child','--seed',str(seed),'--window',str(window)],capture_output=True,text=True,check=True)
            row=json.loads(p.stdout);rows.append(row)
            print(f'{len(rows)}/12 seed={seed} W={window}: {row["mean_step_seconds"]*1000:.2f} ms/update',flush=True)
    for seed in cfg['seeds']:
        pair=[r for r in rows if r['seed']==seed]
        for field in ('initialization_sha256','batches_sha256','parameter_bytes','optimizer_tensor_bytes'):assert pair[0][field]==pair[1][field],field
    out=ROOT/'results-training-cost';out.mkdir(exist_ok=True)
    data={'device':'cpu','threads':1,'dtype':'float32','batch':16,'sequence_length':128,'layers':8,'width':48,'heads':4,
          'conditions':[128,32],'seeds':cfg['seeds'],'torch_version':__import__('torch').__version__,
          'cpu':next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'unknown'),
          'method':'Fresh process per seed/condition, serial alternating order, 10 warmup + 50 timed updates. Prebuilt identical text batches; forward/backward/clip/AdamW timed. No evaluation, IO or data sampling in timed region. Fixed peak learning rate; microbenchmark is not a retraining result.',
          'memory_method':'Linux process peak RSS includes imports, model, optimizer, activations, allocator and all prebuilt batches; not attributable GPU memory. Saved nonparameter tensor storages deduplicated by storage pointer during one forward/backward; includes inputs, excludes parameters, is not total or peak memory.',
          'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('model.py','benchmark_training.py')},'results':rows}
    (out/'benchmark.json').write_text(json.dumps(data,indent=2))

if __name__=='__main__':main()

"""Fresh-process cost measurements on saved checkpoints; no model updates.

One training forward/backward measures RSS and saved tensors. Prefill and
single-token decoding are timed separately under no_grad. No training-control
experiments are rerun and no checkpoint parameters are modified.
"""
import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
import torch
from model import WindowLM
from run import probe_batch
from two_sweep import ordinary_sweep
from recurrent_evaluation import state_bytes
ROOT=Path(__file__).parent


def worker(checkpoint):
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    model=WindowLM.from_pretrained(checkpoint)
    cfg=json.loads((ROOT/'two_sweep_config.json').read_text())
    gen=torch.Generator().manual_seed(cfg['seeds'][0]+cfg['training_data_seed_offset'])
    x,labels,_=probe_batch(16,gen,sequence_length=320,distances=cfg['distances'])
    initial={n:p.detach().clone() for n,p in model.named_parameters()}
    saved={}
    def pack(t):
        storage=t.untyped_storage();saved[storage.data_ptr()]=storage.nbytes()
        return t
    model.train();start=time.perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        loss=model(x,labels=labels).loss
    saved_bytes=sum(saved.values());loss.backward()
    training_forward_backward_seconds=time.perf_counter()-start
    peak_training_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    model.zero_grad(set_to_none=True);model.eval()
    # Logical completed-U grid, separately from full autograd saved storage.
    ordinary_grid_bytes=None
    zigzag_u_volume=None;zigzag_r_volume=None;zigzag_u_row=None
    if model.config.two_sweep_recurrent:
        with torch.no_grad():
            embedded=model.token(x)+model.position(torch.arange(320))[None]
            grid,_=ordinary_sweep(model,embedded,torch.arange(320))
            ordinary_grid_bytes=sum(u.numel()*u.element_size() for u in grid)
            del grid,embedded
    if model.config.zigzag_recurrent:
        from zigzag import grid_sweep
        with torch.no_grad():
            embedded=model.token(x)+model.position(torch.arange(320))[None]
            _,_,_,ug,rg=grid_sweep(model,embedded,torch.arange(320),record_grid=True)
            zigzag_u_volume=sum(t.numel()*t.element_size() for t in ug)
            zigzag_r_volume=sum(t.numel()*t.element_size() for t in rg)
            zigzag_u_row=ug[0].numel()*ug[0].element_size()
            del ug,rg,embedded
    timings={}
    with torch.no_grad():
        for batch in (1,16):
            prompt=x[:batch,:319];token=x[:batch,319:320]
            for _ in range(2):model(prompt,use_cache=True)
            prefill=[]
            for _ in range(5):
                start=time.perf_counter();out=model(prompt,use_cache=True);prefill.append(time.perf_counter()-start)
            cache=out.past_key_values
            # Identical frozen prefix state reused: token320 is never repeatedly appended.
            for _ in range(2):model(token,past_key_values=cache,use_cache=True,position_ids=torch.tensor([319]))
            decode=[]
            for _ in range(20):
                start=time.perf_counter();model(token,past_key_values=cache,use_cache=True,position_ids=torch.tensor([319]));decode.append(time.perf_counter()-start)
            timings[str(batch)]={'prefill_length':319,'prefill_median_seconds':statistics.median(prefill),'prefill_seconds':prefill,
                'decode_median_seconds':statistics.median(decode),'decode_seconds':decode,**state_bytes(cache)}
    for n,p in model.named_parameters():torch.testing.assert_close(p,initial[n],atol=0,rtol=0)
    # Hook storage includes weights and inputs; it is not peak resident memory.
    return {'checkpoint':str(checkpoint),'threads':2,'training_batch':16,'training_sequence_length':320,
        'training_forward_backward_seconds':training_forward_backward_seconds,
        'fixed_batch_training_peak_process_rss_bytes':peak_training_rss,'autograd_unique_saved_storage_bytes':saved_bytes,
        'first_sweep_completed_grid_logical_bytes':ordinary_grid_bytes,
        'first_sweep_completed_grid_retained_for_lateral':bool(model.config.two_sweep_recurrent),
        'zigzag_U_row_logical_bytes':zigzag_u_row,'zigzag_U_all_rows_logical_volume_bytes':zigzag_u_volume,
        'zigzag_R_all_rows_logical_volume_bytes':zigzag_r_volume,
        'latencies_and_state_by_batch':timings,'torch':torch.__version__,'python':platform.python_version()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.checkpoint:
        print(json.dumps(worker(args.checkpoint)));return
    out=args.output or ROOT/'results-two-sweep/costs.json'
    sources=[('results-w32-horizon','sliding-32'),('results-lateral-slots','recurrent-1slot-diagonal'),
             ('results-lateral-slots','recurrent-2slot-diagonal'),('results-three-slots','recurrent-3slot-diagonal'),
             ('results-two-sweep','two-sweep-top-recurrent-w32')]
    rows=[]
    for directory,name in sources:
        data=json.loads((ROOT/directory/'runs.json').read_text())
        row=next(r for r in data['results'] if r['condition']==name)
        checkpoint=ROOT/directory/row['checkpoint']
        result=json.loads(subprocess.check_output([sys.executable,__file__,'--checkpoint',str(checkpoint)],text=True))
        result['condition']=name;rows.append(result)
        out.write_text(json.dumps({'method':'Fresh process per checkpoint; one fixed-batch forward/backward (no optimizer step), two latency warmups, five prefill and twenty frozen-prefix decode timings. No model updates. RSS includes runtime/weights; autograd storage deduplicates allocations and includes parameters/inputs. Grid volume is not total activation memory.','results':rows},indent=2)+'\n')
        print(name,'prefill/decode batch1:',result['latencies_and_state_by_batch']['1']['prefill_median_seconds'],result['latencies_and_state_by_batch']['1']['decode_median_seconds'],flush=True)

if __name__=='__main__':main()

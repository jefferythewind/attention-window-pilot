"""Measure logical and backing inference-cache tensor bytes, batch one, chunk-16 prefill."""
import argparse
import json
from pathlib import Path
import torch
from model import WindowConfig,WindowLM
from recurrent_evaluation import state_bytes
ROOT=Path(__file__).parent
torch.set_num_threads(1)
config=json.loads((ROOT/'recurrent_config.json').read_text());rows=[]
with torch.no_grad():
    for c in config['conditions']:
        if c['mode']=='chunk':continue
        model=WindowLM(WindowConfig(vocab_size=32,width=48,layers=8,heads=4,window=c['window'],recurrent_attention_slot=c.get('recurrent_attention_slot',False))).eval()
        prompt=torch.zeros(1,127,dtype=torch.long);cache=None
        for start in range(0,127,16):
            out=model(prompt[:,start:start+16],past_key_values=cache,use_cache=True)
            cache=out.past_key_values
        rows.append({'condition':c['name'],**state_bytes(cache)})
parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'results-recurrent')
args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
path=args.output_dir/'state-bytes.json'
path.write_text(json.dumps({'batch_size':1,'processed_tokens':127,'prefill_chunk_size':16,'dtype':'float32','description':'Tensor storage only; excludes model weights, allocator overhead and temporary activations. Logical dimensions and deduplicated backing allocations are distinct. No training required; contents do not affect shape/storage bytes.','results':rows},indent=2)+'\n')
print(json.dumps(rows,indent=2))

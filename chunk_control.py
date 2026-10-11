"""Paired independent-chunk controls for both existing pilots; no overwritten results."""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import transformers
from model import WindowConfig, WindowLM
from run import digest, evaluate, probe_batch

ROOT = Path(__file__).parent


def chunk_logits(model, input_ids, chunk_size):
    """Batched full causal attention per independent block, original absolute positions."""
    batch, length = input_ids.shape
    if length % chunk_size:
        raise ValueError('Sequence length must be divisible by chunk size.')
    chunks = input_ids.reshape(-1, chunk_size)
    positions = torch.arange(length, device=input_ids.device).reshape(-1, chunk_size).repeat(batch, 1)
    return model(chunks, position_ids=positions).logits.reshape(batch, length, -1)


def next_token_loss(logits, labels):
    # Last logits of non-final blocks predict the first token of the next block.
    # Labels supervise outputs; they never become inputs to the previous block.
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                           labels[:, 1:].reshape(-1), ignore_index=-100)


@torch.no_grad()
def evaluate_chunked(model, task, validation, chunk_size):
    model.eval()
    if task == 'text':
        losses = [float(next_token_loss(chunk_logits(model, x, chunk_size), x)) for x in validation]
        loss = float(np.mean(losses))
        return {'heldout_nll': loss, 'perplexity': math.exp(loss)}
    result = {}
    for distance, (x, labels, source) in validation.items():
        # Query logits are causal; the following answer input cannot influence them.
        logits = chunk_logits(model, x, chunk_size)[:, -2]
        answer = x[:, -1]
        changed = x.clone()
        changed[torch.arange(len(x)), source] = (answer+1) % 8
        altered = chunk_logits(model, changed, chunk_size)[:, -2]
        result[str(distance)] = {
            'accuracy': float((logits.argmax(-1)==answer).float().mean()),
            'answer_nll': float(F.cross_entropy(logits, answer)),
            'counterfactual_accuracy': float((altered.argmax(-1)==(answer+1)%8).float().mean()),
            'max_source_change_logit_difference':float((altered-logits).abs().max()),
            'source_in_query_chunk':bool(torch.all(source//chunk_size == (x.shape[1]-2)//chunk_size)),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilots', nargs='+', choices=['initial', 'doubled'], default=['initial','doubled'])
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    raw = (ROOT/'shakespeare.txt').read_bytes()
    text = raw.decode('utf-8'); chars = sorted(set(text)); lookup = {c:i for i,c in enumerate(chars)}
    data = torch.tensor([lookup[c] for c in text])
    split = int(len(data)*.9); train, heldout = data[:split], data[split:]
    for pilot in args.pilots:
        source_dir = ROOT/('results' if pilot=='initial' else 'results-2x')
        original = json.loads((source_dir/'runs.json').read_text())
        out = ROOT/('results-chunk' if pilot=='initial' else 'results-2x-chunk')
        out.mkdir(exist_ok=True)
        (out/'tokenizer.json').write_text((source_dir/'tokenizer.json').read_text())
        length = original['sequence_length']; layers = original.get('layers',4)
        distances = original.get('distances',[4,12,24,40]); windows=original['args']['windows']
        assert hashlib.sha256(raw).hexdigest()==original['dataset_sha256']
        assert split==original['split_index']
        metadata = {'pilot':pilot,'source_results':str(source_dir.relative_to(ROOT)/'runs.json'),
                    'torch':torch.__version__,'transformers':transformers.__version__,
                    'python':platform.python_version(),'threads':args.threads,'device':'cpu',
                    'sequence_length':length,'layers':layers,'width':48,'heads':4,'batch_size':16,
                    'distances':distances,'windows':windows,'seeds':original['args']['seeds'],
                    'position_policy':'original absolute positions preserved in every chunk',
                    'target_policy':'all original shifted targets, including chunk-boundary predictions',
                    'dataset_sha256':original['dataset_sha256'],'split_index':split,
                    'source_results_sha256':hashlib.sha256((source_dir/'runs.json').read_bytes()).hexdigest(),
                    'results':[]}
        for task in ('text','probe'):
            steps = original['args'][task+'_steps']
            vg=torch.Generator().manual_seed(2026)
            if task=='text':
                starts=torch.randint(0,len(heldout)-length,(32,16),generator=vg)
                validation=[torch.stack([heldout[s:s+length] for s in row]) for row in starts]
            else:
                validation={d:probe_batch(256,vg,d,length,distances) for d in distances}
            for seed in original['args']['seeds']:
                torch.manual_seed(seed)
                base=WindowLM(WindowConfig(vocab_size=len(chars) if task=='text' else 32,
                                          width=48,layers=layers,heads=4))
                initial={k:v.detach().clone() for k,v in base.state_dict().items()}
                for window in windows:
                    reference=next(r for r in original['results'] if r['task']==task and r['seed']==seed and r['window']==window)
                    assert digest(initial)==reference['initialization_sha256']
                    target_count=steps*16*(length-1 if task=='text' else 1)
                    assert target_count==reference['supervised_targets']
                    name=f'{task}-seed{seed}-w{window}'
                    sliding=WindowLM.from_pretrained(source_dir/name).eval()
                    sliding_native=evaluate_chunked(sliding,task,validation,window)
                    if window==length:
                        # One chunk is the existing full-attention baseline; reuse it.
                        if task=='text':
                            assert abs(sliding_native['heldout_nll']-reference['metrics']['heldout_nll'])<1e-5
                        else:
                            for d in distances:
                                assert abs(sliding_native[str(d)]['answer_nll']-reference['metrics'][str(d)]['answer_nll'])<1e-5
                        row={**reference,'reused_full_baseline':True,'chunked_metrics':sliding_native,
                             'rolling_metrics':reference['metrics'],'sliding_trained_chunked_metrics':sliding_native}
                    else:
                        model=WindowLM(WindowConfig(vocab_size=base.config.vocab_size,width=48,
                                                    layers=layers,heads=4,window=window))
                        model.load_state_dict(initial); model.train()
                        optimizer=torch.optim.AdamW([
                            {'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':.1},
                            {'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0.}],
                            lr=.002,betas=(.9,.95))
                        g=torch.Generator().manual_seed(seed+1000); h=hashlib.sha256()
                        history=[]; start_time=time.perf_counter()
                        for step in range(steps):
                            if task=='text':
                                starts=torch.randint(0,len(train)-length,(16,),generator=g)
                                x=torch.stack([train[s:s+length] for s in starts]); labels=x
                            else:
                                x,labels,_=probe_batch(16,g,sequence_length=length,distances=distances)
                            h.update(x.numpy().tobytes())
                            warmup=min(20,max(1,steps//10))
                            scale=(step+1)/warmup if step<warmup else .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup-1)))
                            for group in optimizer.param_groups:group['lr']=.002*scale
                            optimizer.zero_grad(set_to_none=True)
                            loss=next_token_loss(chunk_logits(model,x,window),labels)
                            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                            optimizer.step()
                            if step%50==0 or step==steps-1:history.append({'step':step+1,'loss':float(loss.detach())})
                        elapsed=time.perf_counter()-start_time
                        assert h.hexdigest()==reference['training_sequences_sha256']
                        native=evaluate_chunked(model,task,validation,window)
                        rolling=evaluate(model,task,validation)
                        if task=='probe':
                            for d in distances:
                                if not native[str(d)]['source_in_query_chunk']:
                                    assert native[str(d)]['max_source_change_logit_difference']==0
                        model.save_pretrained(out/name,safe_serialization=False)
                        row={'task':task,'seed':seed,'window':window,
                             'parameters':sum(p.numel() for p in model.parameters()),
                             'initialization_sha256':digest(initial),'training_sequences_sha256':h.hexdigest(),
                             'processed_tokens':steps*16*length,'supervised_targets':target_count,
                             'optimizer_updates':steps,'chunks_per_original_sequence':length//window,
                             'training_seconds':elapsed,'tokens_per_second':steps*16*length/elapsed,
                             'history':history,'reused_full_baseline':False,'chunked_metrics':native,
                             'rolling_metrics':rolling,'sliding_trained_chunked_metrics':sliding_native}
                    metadata['results'].append(row)
                    (out/'runs.json').write_text(json.dumps(metadata,indent=2))
                    print(pilot,name,'reused' if row['reused_full_baseline'] else f'{row["training_seconds"]:.1f}s',
                          'chunked',json.dumps(row['chunked_metrics']),'rolling',json.dumps(row['rolling_metrics']),flush=True)


if __name__=='__main__':
    main()

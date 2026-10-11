"""Small paired character-LM experiment and separate answer-supervised relay probe."""
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import torch
import transformers
from model import WindowConfig, WindowLM

ROOT = Path(__file__).parent


def digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def probe_batch(n, generator, distance=None):
    x = torch.randint(16, 32, (n, 64), generator=generator)
    distances = torch.tensor([4, 12, 24, 40])
    ds = distances[torch.randint(0, 4, (n,), generator=generator)] if distance is None else torch.full((n,), distance)
    value = torch.randint(0, 8, (n,), generator=generator)
    source = 62-ds
    x[torch.arange(n), source-1] = 8  # STORE marker
    x[torch.arange(n), source] = value
    x[:, 62] = 9  # QUERY marker; answer is the next token
    x[:, 63] = value
    labels = torch.full_like(x, -100)
    labels[:, 63] = value
    return x, labels, source


@torch.no_grad()
def evaluate(model, task, validation):
    model.eval()
    if task == 'text':
        losses = [float(model(x, labels=x).loss) for x in validation]
        return {'heldout_nll': float(np.mean(losses)), 'perplexity': math.exp(float(np.mean(losses)))}
    result = {}
    for distance, (x, labels, source) in validation.items():
        prompt, answer = x[:, :63], x[:, 63]
        output = model(prompt).logits[:, -1]
        cache = None
        for start in range(0, 63, 16):
            cached = model(prompt[:, start:start+16], past_key_values=cache, use_cache=True,
                           position_ids=torch.arange(start, min(start+16, 63)))
            cache = cached.past_key_values
        torch.testing.assert_close(output, cached.logits[:, -1], atol=3e-6, rtol=3e-5)
        w = model.config.window
        recent = prompt[:, -w:]
        recomputed = model(recent, position_ids=torch.arange(63-recent.shape[1], 63)).logits[:, -1]
        corrupted = prompt.clone()
        corrupted[torch.arange(len(x)), source] = (answer+1) % 8
        altered = model(corrupted).logits[:, -1]
        result[str(distance)] = {
            'accuracy': float((output.argmax(-1) == answer).float().mean()),
            'answer_nll': float(torch.nn.functional.cross_entropy(output, answer)),
            'recomputed_window_accuracy': float((recomputed.argmax(-1) == answer).float().mean()),
            'counterfactual_accuracy': float((altered.argmax(-1) == (answer+1)%8).float().mean()),
            'cache_kv_bytes': sum(k.numel()*k.element_size()+v.numel()*v.element_size() for k,v,p in cache),
            'cache_batch_size': len(x),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-steps', type=int, default=200)
    parser.add_argument('--probe-steps', type=int, default=400)
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 29])
    parser.add_argument('--windows', type=int, nargs='+', default=[64, 16, 8, 4])
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    out = ROOT/'results'; out.mkdir(exist_ok=True)
    raw = (ROOT/'shakespeare.txt').read_bytes()
    text = raw.decode('utf-8')
    chars = sorted(set(text)); lookup = {c:i for i,c in enumerate(chars)}
    data = torch.tensor([lookup[c] for c in text])
    split = int(len(data)*.9); train, heldout = data[:split], data[split:]
    (out/'tokenizer.json').write_text(json.dumps({'type':'character', 'characters':chars}))
    metadata = {'args': vars(args), 'torch':torch.__version__, 'transformers':transformers.__version__,
                'python': platform.python_version(), 'device':'cpu', 'sequence_length':64,
                'batch_size':16, 'dataset_sha256':hashlib.sha256(raw).hexdigest(),
                'dataset_url':'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
                'split_index':split, 'training_dtype':'float32', 'results':[]}
    path = out/'runs.json'
    for task in ('text', 'probe'):
        steps = args.text_steps if task == 'text' else args.probe_steps
        vg = torch.Generator().manual_seed(2026)
        if task == 'text':
            starts = torch.randint(0, len(heldout)-64, (32,16), generator=vg)
            validation = [torch.stack([heldout[s:s+64] for s in row]) for row in starts]
        else:
            validation = {d:probe_batch(256, vg, d) for d in (4,12,24,40)}
        for seed in args.seeds:
            torch.manual_seed(seed)
            base = WindowLM(WindowConfig(vocab_size=len(chars) if task=='text' else 32,
                                         width=48, layers=4, heads=4))
            initial = {k:v.detach().clone() for k,v in base.state_dict().items()}
            for window in args.windows:
                m = WindowLM(WindowConfig(vocab_size=base.config.vocab_size, width=48,
                                           layers=4, heads=4, window=window))
                m.load_state_dict(initial); m.train()
                optimizer = torch.optim.AdamW([
                    {'params':[p for p in m.parameters() if p.ndim>=2], 'weight_decay':0.1},
                    {'params':[p for p in m.parameters() if p.ndim<2], 'weight_decay':0.0}],
                    lr=0.002, betas=(0.9,0.95))
                generator = torch.Generator().manual_seed(seed+1000)
                data_hash = hashlib.sha256(); start_time = time.perf_counter()
                history = []
                for step in range(steps):
                    if task == 'text':
                        starts = torch.randint(0, len(train)-64, (16,), generator=generator)
                        x = torch.stack([train[s:s+64] for s in starts]); labels=x
                    else:
                        x, labels, _ = probe_batch(16, generator)
                    data_hash.update(x.numpy().tobytes())
                    warmup = min(20, max(1, steps//10))
                    scale = (step+1)/warmup if step<warmup else .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup-1)))
                    for group in optimizer.param_groups: group['lr'] = .002*scale
                    optimizer.zero_grad(set_to_none=True)
                    loss = m(x, labels=labels).loss
                    loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    optimizer.step()
                    if step%50 == 0 or step==steps-1:
                        history.append({'step':step+1,'loss':float(loss.detach())})
                elapsed = time.perf_counter()-start_time
                metrics = evaluate(m, task, validation)
                name = f'{task}-seed{seed}-w{window}'
                m.save_pretrained(out/name, safe_serialization=False)
                row = {'task':task,'seed':seed,'window':window,'parameters':sum(p.numel() for p in m.parameters()),
                       'initialization_sha256':digest(initial),'training_sequences_sha256':data_hash.hexdigest(),
                       'processed_tokens':steps*16*64,'supervised_targets':steps*16*(63 if task=='text' else 1),
                       'training_seconds':elapsed,'tokens_per_second':steps*16*64/elapsed,
                       'history':history,'metrics':metrics}
                metadata['results'].append(row)
                path.write_text(json.dumps(metadata, indent=2))
                print(name, f'{elapsed:.1f}s', json.dumps(metrics), flush=True)
    print('Saved', path, flush=True)


if __name__ == '__main__':
    main()

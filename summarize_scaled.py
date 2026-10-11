"""Summarize the scaled sweep without overwriting the first pilot's report."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
data = json.loads((ROOT/'results-2x/runs.json').read_text())
old = json.loads((ROOT/'results/runs.json').read_text())
groups = defaultdict(list)
for row in data['results']:
    groups[row['task'], row['window']].append(row)
for task in ('text', 'probe'):
    for seed in data['args']['seeds']:
        rows = [r for r in data['results'] if r['task']==task and r['seed']==seed]
        assert len(rows) == len(data['args']['windows']), 'Incomplete sweep'
        for field in ('initialization_sha256', 'training_sequences_sha256', 'processed_tokens'):
            assert len({r[field] for r in rows}) == 1, (task, seed, field)

layers, length = data['layers'], data['sequence_length']
distances = data['distances']
lines = ['# Doubled-depth and doubled-context CPU experiment', '',
         f'{layers} layers, {length}-token sequences, width {data["width"]}, four attention heads.',
         'Two separately trained tasks and two paired seeds, following the first pilot.', '',
         '**Headline:** Restricted windows retain substantial information relay, but the fourfold',
         'reduction does not reliably match full-attention retrieval at this budget. W=32 achieves',
         '100% at distance 80 in seed 17 and 60.9% in seed 29, versus 100% for full attention in both.',
         'Separately trained text models still show lower early held-out loss for smaller windows.', '',
         '| Window | Text NLL, mean (seed range) | '+
         ' | '.join(f'Recall d={d}' for d in distances)+' | KV KiB/sequence* |',
         '|---:|---:|'+ '---:|'*(len(distances)+1)]
for window in data['args']['windows']:
    losses = [r['metrics']['heldout_nll'] for r in groups['text', window]]
    recalls = [statistics.mean(r['metrics'][str(d)]['accuracy'] for r in groups['probe',window])*100 for d in distances]
    kv = 2*layers*(window-1)*data['width']*4/1024
    lines.append(f'| {window} | {statistics.mean(losses):.4f} ({min(losses):.4f}–{max(losses):.4f}) | '+
                 ' | '.join(f'{v:.1f}%' for v in recalls)+f' | {kv:.1f} |')
lines += ['', '*Steady-state float32 KV tensors; excludes metadata, activations and transient allocations.', '',
          'Text NLL is nats per character. Text and retrieval columns use separately trained models.',
          'Training does not cache or detach states: all complete sequences retain their graph.', '',
          '## Retrieval controls and variation across seeds', '',
          '| Window | Distance | Accuracy per seed | Answer NLL per seed | Final-window recomputation per seed | Changed-source accuracy per seed |',
          '|---:|---:|---|---|---|---|']
for window in data['args']['windows']:
    for distance in distances:
        rows = groups['probe',window]
        values = [' / '.join(f"{r['metrics'][str(distance)][field]*100:.1f}%" for r in rows)
                  for field in ('accuracy', 'recomputed_window_accuracy', 'counterfactual_accuracy')]
        nll = ' / '.join(f"{r['metrics'][str(distance)]['answer_nll']:.4f}" for r in rows)
        lines.append(f'| {window} | {distance} | {values[0]} | {nll} | {values[1]} | {values[2]} |')
lines += ['', 'Seed order: '+', '.join(str(s) for s in data['args']['seeds'])+'.', '',
          'The weaker W=32 seed also scores only 63.3% at distance 8, inside its direct window.',
          'Its errors are therefore not confined to long-range relay. These results suggest a',
          'training/optimization limitation may contribute, but do not establish its cause.', '',
          '## Comparison with the first pilot', '',
          '| Setting | First pilot | Doubled experiment |',
          '|---|---:|---:|',
          '| Layers | 4 | 8 |', '| Sequence length | 64 | 128 |',
          '| Windows | 64, 16, 8, 4 | 128, 32, 16, 8 |',
          '| Retrieval distances | 4, 12, 24, 40 | 8, 24, 48, 80 |',
          '| Updates (text / retrieval) | 200 / 400 | 200 / 400 |',
          '| Processed tokens per run (text / retrieval) | 204,800 / 409,600 | 409,600 / 819,200 |',
          '| Supervised targets per run (text / retrieval) | 201,600 / 6,400 | 406,400 / 6,400 |',
          f'| Timed training, all runs | {sum(r["training_seconds"] for r in old["results"]):.1f}s | '+
          f'{sum(r["training_seconds"] for r in data["results"]):.1f}s |', '',
          'Width, heads, tokenizer, corpus split, batch size, seeds, optimizer and learning-rate',
          'schedule remain unchanged. Models are trained from scratch, not continued from the first pilot.',
          'The new sweep uses one CPU thread instead of four to reduce threading overhead; timing',
          'comparisons across pilots also reflect this execution-setting difference.',
          'Matching initialization, sequence order and budgets was verified within each new task and seed.', '',
          'This jointly changes depth, sequence length, windows, retrieval distances and processed-token',
          'budget. It is not an isolated test of depth or a token-matched comparison with the first pilot.',
          'The text evaluations also differ in sequence length and sampled positions, so cross-pilot',
          'loss differences are descriptive rather than controlled effects.', '',
          'Maximum structural reach is L(W-1): '+
          ', '.join(f'W={w}: {layers*(w-1)}' for w in data['args']['windows'])+' positions.',
          'In particular, distance 80 is within the W=16 horizon (120), but outside W=8 (56).',
          'Reachability permits retrieval; it does not guarantee it is learned in this training budget.', '',
          'Cached and full-sequence logits were checked for equivalence on every retrieval evaluation.',
          'Two seeds and a simple single-value probe do not establish general capability equivalence.',
          'Matching exact-match accuracy need not mean matching answer NLL; both are reported above.',
          'Dense attention masking does not establish sparse-kernel training savings.', '',
          'Raw metrics: [results-2x/runs.json](results-2x/runs.json).',
          'Protocol and reproduction command: [DESIGN.md](DESIGN.md#doubled-depth-and-context-experiment).']
path = ROOT/'RESULTS-2X.md'
path.write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))

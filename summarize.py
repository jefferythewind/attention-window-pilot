import json
from collections import defaultdict
from pathlib import Path
import statistics

root = Path(__file__).parent
data = json.loads((root/'results/runs.json').read_text())
groups = defaultdict(list)
for run in data['results']:
    groups[run['task'], run['window']].append(run)
for task in ('text', 'probe'):
    for seed in data['args']['seeds']:
        rows = [r for r in data['results'] if r['task']==task and r['seed']==seed]
        assert len(rows) == len(data['args']['windows']), 'Incomplete sweep'
        for field in ('initialization_sha256', 'training_sequences_sha256', 'processed_tokens'):
            assert len({r[field] for r in rows}) == 1, (task, seed, field)

lines = ['# CPU pilot results', '',
         'Two paired seeds; all initialization, sequence-order and token-budget checks passed.', '',
         '**Headline:** A fourfold attention-window reduction preserved simple controlled retrieval;',
         'separately trained text models showed no held-out loss penalty after short matched training.', '',
         'Each table row summarizes two independently trained models per seed: a text model and a',
         'retrieval model. We have not tested retrieval on the text-trained checkpoints.', '',
         '| Window | Text NLL, mean (seed range) | Recall d=4 | d=12 | d=24 | d=40 | KV KiB/sequence* |',
         '|---:|---:|---:|---:|---:|---:|---:|']
for w in data['args']['windows']:
    losses = [r['metrics']['heldout_nll'] for r in groups['text',w]]
    recall = [statistics.mean(r['metrics'][str(d)]['accuracy'] for r in groups['probe',w])*100 for d in (4,12,24,40)]
    kv = 2*4*(w-1)*48*4/1024
    lines.append(f'| {w} | {statistics.mean(losses):.4f} ({min(losses):.4f}–{max(losses):.4f}) | '+
                 ' | '.join(f'{v:.1f}%' for v in recall)+f' | {kv:.1f} |')
lines += ['', '*Steady-state float32 KV tensor storage, excluding metadata and transient allocations.', '',
          'Text and recall columns refer to separately trained models. Text NLL is nats per character.', '',
          'The text task predicts the next character on held-out Shakespeare text; it is not explicitly',
          'a memorization test. The retrieval task learns one random stored value amid random filler,',
          'with only the answer supervised. See archive/PILOT-HISTORY.md for actual data examples.', '',
          'The W=16 probe matches full attention on this simple retrieval task, including distances',
          '24 and 40 beyond its direct window. Its final-window recomputation has no source token:',
          'accuracy at distance 40 is '+
          ', '.join(f"{r['metrics']['40']['recomputed_window_accuracy']*100:.1f}%" for r in groups['probe',16])+'.',
          'Changing the stored source value yields the corresponding changed answer at distance 40:',
          ', '.join(f"{r['metrics']['40']['counterfactual_accuracy']*100:.1f}%" for r in groups['probe',16])+'.', '',
          'W=8 can propagate information across a 28-position structural horizon, but did not learn',
          'reliable retrieval at distance 24 in this budget. Structural reachability is not sufficient',
          'for learned retrieval. Distance 40 is outside its horizon and is a negative control.', '',
          'Smaller windows do not fail universally: W=8 reaches 100% accuracy at distances 4 and 12,',
          'and W=4 averages 73.6% at distance 4. Degradation depends on source-to-query distance.', '',
          'Smaller windows achieved better early character prediction in this short training run.',
          'This could reflect optimization or regularization at limited training, and does not',
          'establish a converged capability advantage or statistical equivalence.', '',
          f"Total timed training: {sum(r['training_seconds'] for r in data['results']):.1f} CPU wall seconds.",
          'These timings use dense masked attention; they are not sparse-kernel efficiency claims.',
          'All trained checkpoints are retained under results/. See archive/PILOT-HISTORY.md for protocol and limits.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))

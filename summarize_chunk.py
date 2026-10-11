"""Produce factual paired comparisons for the independent-chunk controls."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).parent
lines=['# Independently chunk-trained controls', '',
       'Two controls serve different questions: final-window recomputation tests the value of retained',
       'states at inference; independently chunked training tests connected versus disconnected',
       'training. The new control is paired with both existing pilots.', '',
       'Every chunk uses full causal attention internally and original absolute positions. Models',
       'share initialization, ordered original sequences, all shifted targets (including boundaries),',
       'processed-token counts, optimizer steps, and optimizer settings with their sliding-trained pair.',
       'Maximum attention span is matched; average attention coverage and training compute are not.', '',
       'Four evaluations separate training and inference: sliding-trained/rolling inference,',
       'sliding-trained/independent chunks, chunk-trained/independent chunks, and chunk-trained/rolling.',
       'Full-window rows reuse the existing full-attention checkpoints; the one-chunk equivalence was checked.', '',
       'Text and retrieval results use separately trained models. Individual seed results are',
       'descriptive, not confidence intervals. This remains a small, short-training pilot.', '']

additional_time=0
for pilot, original_dir, control_dir in [('Initial: 4 layers, 64 tokens','results','results-chunk'),
                                        ('Doubled: 8 layers, 128 tokens','results-2x','results-2x-chunk')]:
    original=json.loads((ROOT/original_dir/'runs.json').read_text())
    control=json.loads((ROOT/control_dir/'runs.json').read_text())
    assert hashlib.sha256((ROOT/original_dir/'runs.json').read_bytes()).hexdigest()==control['source_results_sha256']
    rows=control['results']
    assert len(rows)==len(control['seeds'])*len(control['windows'])*2, 'Incomplete controls'
    for row in rows:
        ref=next(r for r in original['results'] if all(r[k]==row[k] for k in ('task','seed','window')))
        for field in ('initialization_sha256','training_sequences_sha256','processed_tokens','supervised_targets','parameters'):
            assert row[field]==ref[field],(pilot,row['task'],row['seed'],row['window'],field)
        if not row['reused_full_baseline']:
            assert row['optimizer_updates']==original['args'][row['task']+'_steps']
        if not row['reused_full_baseline']:additional_time+=row['training_seconds']
    lines += [f'## {pilot}', '',
              '### Held-out next-character loss', '',
              '| Window | Sliding trained, rolling | Sliding trained, chunks | Chunk trained, chunks | Chunk trained, rolling |',
              '|---:|---:|---:|---:|---:|']
    for w in control['windows']:
        rs=[r for r in rows if r['task']=='text' and r['window']==w]
        old=[r for r in original['results'] if r['task']=='text' and r['window']==w]
        values=[statistics.mean(r['metrics']['heldout_nll'] for r in old)]+[
            statistics.mean(r[field]['heldout_nll'] for r in rs)
            for field in ('sliding_trained_chunked_metrics','chunked_metrics','rolling_metrics')]
        lines.append(f'| {w} | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    lines += ['', 'All columns score the same original next-character targets, including chunk-boundary',
              'predictions. Lower is better; units are nats per character.', '',
              '### Retrieval accuracy, mean across seeds', '',
              '| Window | Distance | Sliding trained, rolling | Sliding trained, chunks | Chunk trained, chunks | Chunk trained, rolling |',
              '|---:|---:|---:|---:|---:|---:|']
    for w in control['windows']:
        for d in control['distances']:
            rs=[r for r in rows if r['task']=='probe' and r['window']==w]
            old=[r for r in original['results'] if r['task']=='probe' and r['window']==w]
            values=[statistics.mean(r['metrics'][str(d)]['accuracy'] for r in old)]+[
                statistics.mean(r[field][str(d)]['accuracy'] for r in rs)
                for field in ('sliding_trained_chunked_metrics','chunked_metrics','rolling_metrics')]
            lines.append(f'| {w} | {d} | '+' | '.join(f'{v*100:.1f}%' for v in values)+' |')
    far=max(control['distances']); quarter=control['sequence_length']//4
    lines += ['', f'### Quarter-window (W={quarter}), distance {far}: individual seeds', '',
              '| Seed | Sliding/rolling accuracy | Chunk/chunks accuracy | Chunk/rolling accuracy | Sliding/rolling answer NLL | Chunk/chunks answer NLL | Chunk/rolling answer NLL |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for seed in control['seeds']:
        r=next(r for r in rows if r['task']=='probe' and r['window']==quarter and r['seed']==seed)
        ref=next(r for r in original['results'] if r['task']=='probe' and r['window']==quarter and r['seed']==seed)
        ms=[ref['metrics'][str(far)],r['chunked_metrics'][str(far)],r['rolling_metrics'][str(far)]]
        lines.append(f'| {seed} | '+' | '.join(f'{m["accuracy"]*100:.1f}%' for m in ms)+' | '+
                     ' | '.join(f'{m["answer_nll"]:.4f}' for m in ms)+' |')
    lines += ['', f'Raw results: [{control_dir}/runs.json]({control_dir}/runs.json).', '']

lines += ['## Interpretation and limits', '',
          'Independent chunks cannot recover a random source value outside the query chunk. That',
          'structural limitation predicts chance-level retrieval (12.5% among eight values), rather',
          'than indicating an implementation failure. The changed-source control verifies exact',
          'invariance of native chunk outputs whenever the source lies in a different chunk.', '',
          'Sliding/rolling versus chunk/chunks measures the practical difference between the two',
          'training-and-inference systems. Sliding/rolling versus chunk/rolling holds inference',
          'semantics fixed and tests the contribution of connected training. Chunk-trained models',
          'may relay information when switched to rolling inference without having trained on that',
          'cross-chunk pathway; the transfer evaluation is intentionally outside their training regime.', '',
          'In the answer-only probe, earlier independent chunks receive no gradient, including their',
          'learned absolute position embeddings. Rolling transfer therefore also uses positions outside',
          'the chunk-trained probe\'s supervised region. Fixed positions or RoPE would help separate',
          'that positional effect from learning to relay in a future experiment.', '',
          'Any successful chunk/rolling result limits claims that connected training is necessary',
          'for relay. Average text loss, task accuracy, and answer NLL can favor different conditions.',
          'These results do not establish a converged capability advantage or general statistical equivalence.', '',
          f'Additional timed training: {additional_time:.1f} seconds (24 new runs; reused full baselines excluded).',
          'The controls use one CPU thread, with the original optimizer schedule and update budgets.',
          'Training timing excludes evaluation and checkpoint writes. No sparse-kernel efficiency claim.', '',
          'Reproduction and protocol: [DESIGN.md](DESIGN.md#independently-chunk-trained-control).']
(ROOT/'RESULTS-CHUNK.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))

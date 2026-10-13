"""Six-seed V1 summary; all conditions, seed variation, and paired comparisons."""
import json
import math
from pathlib import Path
import statistics
from run_v1 import check_pairs

ROOT=Path(__file__).parent
data=json.loads((ROOT/'results-v1/runs.json').read_text()); config=data['config']; rows=data['results']
check_pairs(rows,config,complete=True)
seeds=config['seeds']; conditions=config['conditions']; distances=config['distances']; far=max(distances)


def select(task, condition):
    selected=[next(r for r in rows if r['task']==task and r['seed']==s and r['condition']==condition) for s in seeds]
    assert len(selected)==len(seeds)
    return selected


def native(row):
    return row['chunked_metrics'] if row['training_mode']=='chunk' else row['rolling_metrics']


def ms(values, percent=False):
    scale=100 if percent else 1
    values=[v*scale for v in values]
    precision=1 if percent else 4
    return f'{statistics.mean(values):.{precision}f} ± {statistics.stdev(values):.{precision}f}'+('%' if percent else '')


lines=['# V1 hello world: attention-window experiment', '',
       'One setting: 128-token sequences, eight layers, width 48, four heads; six new paired seeds.',
       'Full attention is the baseline. Sliding attention uses windows 32/16/8. The independent-chunk',
       'training control is evaluated only at W=32. Text prediction and retrieval train separate models.', '',
       '## Primary results', '',
       'Entries are mean ± sample standard deviation across seeds. Each condition uses its native',
       'inference rule: rolling attention for full/sliding, disconnected blocks for chunk-trained.', '',
       '| Condition | Text NLL ↓ | Recall d=8 | d=24 | d=48 | d=80 |',
       '|---|---:|---:|---:|---:|---:|']
for c in conditions:
    texts=select('text',c['name']); probes=select('probe',c['name'])
    values=[ms([native(r)['heldout_nll'] for r in texts])]+[
        ms([native(r)[str(d)]['accuracy'] for r in probes],True) for d in distances]
    lines.append('| '+c['name']+' | '+' | '.join(values)+' |')
lines+=['', 'NLL is nats per character; recall is exact-match answer accuracy. Chance is 12.5% among eight values.',
        'Seed variation, not the number of held-out examples, defines the dispersion reported above.', '',
        '## W=32 training/inference comparison', '',
        'The same two sets of W=32 trained checkpoints are tested with both inference rules.',
        'This is an evaluation matrix, not four separately trained conditions.', '',
        '| Training / inference | Text NLL ↓ | Recall d=8 | d=24 | d=48 | d=80 |',
        '|---|---:|---:|---:|---:|---:|']
for condition,label in [('sliding-32','Sliding trained'),('chunk-32','Chunk trained')]:
    for field,inference in [('rolling_metrics','rolling'),('chunked_metrics','chunks')]:
        texts=select('text',condition); probes=select('probe',condition)
        values=[ms([r[field]['heldout_nll'] for r in texts])]+[
            ms([r[field][str(d)]['accuracy'] for r in probes],True) for d in distances]
        lines.append('| '+label+' / '+inference+' | '+' | '.join(values)+' |')

lines+=['', '## Individual retrieval seeds', '',
        '| Seed | Full d=80 | Sliding32 d=8 | Sliding32 d=80 | Sliding16 d=80 | Sliding8 d=80 | Chunk32 native d=8 | Chunk32 native d=80 | Chunk32 rolling d=80 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for s in seeds:
    rs={r['condition']:r for r in rows if r['task']=='probe' and r['seed']==s}
    values=[rs['full-128']['rolling_metrics']['80']['accuracy'],
            rs['sliding-32']['rolling_metrics']['8']['accuracy'],rs['sliding-32']['rolling_metrics']['80']['accuracy'],
            rs['sliding-16']['rolling_metrics']['80']['accuracy'],rs['sliding-8']['rolling_metrics']['80']['accuracy'],
            rs['chunk-32']['chunked_metrics']['8']['accuracy'],rs['chunk-32']['chunked_metrics']['80']['accuracy'],
            rs['chunk-32']['rolling_metrics']['80']['accuracy']]
    lines.append(f'| {s} | '+' | '.join(f'{v*100:.1f}%' for v in values)+' |')

lines+=['', '## Paired differences', '',
        'Each difference subtracts the other condition from sliding-32 within the same seed.',
        'Positive accuracy differences favor sliding-32; negative NLL differences favor sliding-32.', '',
        '| Paired comparison | Text NLL difference | d=80 accuracy difference (percentage points) |',
        '|---|---:|---:|']
sl_text=select('text','sliding-32'); sl_probe=select('probe','sliding-32')
for other,field,label in [('full-128','rolling_metrics','Sliding32 − full'),
                         ('chunk-32','chunked_metrics','Sliding32 rolling − chunk32 native'),
                         ('chunk-32','rolling_metrics','Sliding32 rolling − chunk32 rolling')]:
    texts=select('text',other); probes=select('probe',other)
    td=[a['rolling_metrics']['heldout_nll']-b[field]['heldout_nll'] for a,b in zip(sl_text,texts)]
    pd=[100*(a['rolling_metrics']['80']['accuracy']-b[field]['80']['accuracy']) for a,b in zip(sl_probe,probes)]
    lines.append(f'| {label} | {ms(td)} | {statistics.mean(pd):.1f} ± {statistics.stdev(pd):.1f} |')

lines+=['', '## Answer loss and source counterfactuals at distance 80', '',
        '| Condition / inference | Answer NLL ↓ | Changed-source accuracy |',
        '|---|---:|---:|']
for c in conditions:
    rs=select('probe',c['name']); metrics=[native(r)['80'] for r in rs]
    lines.append(f'| {c["name"]} native | {ms([m["answer_nll"] for m in metrics])} | '+
                 ms([m['counterfactual_accuracy'] for m in metrics],True)+' |')
rs=select('probe','chunk-32'); metrics=[r['rolling_metrics']['80'] for r in rs]
lines.append('| chunk-32 rolling transfer | '+ms([m['answer_nll'] for m in metrics])+' | '+
             ms([m['counterfactual_accuracy'] for m in metrics],True)+' |')

lines+=['', '## KV storage and budgets', '',
        '| Rolling attention window | Retained KV KiB per sequence, float32 |', '|---:|---:|']
for w in (128,32,16,8):
    expected=2*config['layers']*(w-1)*config['width']*4
    for r in rows:
        if r['task']=='probe' and r['window']==w:
            for m in r['rolling_metrics'].values():assert m['cache_kv_bytes']/m['cache_batch_size']==expected
    lines.append(f'| {w} | {expected/1024:.1f} |')
lines+=['', 'W=32 retains 31 prior keys/values per layer versus 127 for full attention: 24.4% of KV',
        'tensor storage. This is not a claim of quarter total memory or quarter training cost.',
        'Position metadata, weights, activations, allocator overhead and transient prefill are excluded.', '',
        f'- Seeds, selected before training: {", ".join(str(s) for s in seeds)}.',
        '- Text: 200 updates × 16 sequences × 128 tokens = 409,600 processed tokens; 406,400 targets/run.',
        '- Retrieval: 400 updates × 16 sequences × 128 tokens = 819,200 processed tokens; 6,400 targets/run.',
        '- Same held-out examples across every condition and seed: 512 text sequences, 256 retrieval examples/distance.',
        f'- Completed models: {len(rows)}; three CPU workers, one PyTorch thread per worker.',
        f'- Recorded end-to-end wall time: {data["wall_seconds"]:.1f}s.',
        f'- Sum of overlapping per-model timed training durations: {sum(r["training_seconds"] for r in rows):.1f}s.', '',
        'All input, label, initialization, parameter-count, token, target and optimizer-update checks passed.',
        'Cached/full-sequence equivalence and native chunk source invariance passed in evaluation.', '',
        '## Interpretation', '',
        'Local and distant failures may indicate a broader learning/optimization limitation, but the',
        'seed changes both initialization and training-batch sampling; initializer quality is not isolated.',
        'No seeds are excluded or selected based on performance. Six seeds remain a preliminary check,',
        'not a demonstration of converged general capability equivalence.', '',
        'The native chunk control cannot recover values in other chunks. Its rolling transfer evaluation',
        'holds inference fixed to test training adaptation, but earlier chunks and their learned position',
        'embeddings receive no gradient under answer-only supervision. RoPE or fixed embeddings would',
        'help investigate the contribution of positional learning. Exact-match and answer NLL may differ.', '',
        'Reach is finite: eight layers give horizons 248, 120, 56 for W=32,16,8. Distance 80 is structurally',
        'unreachable for W=8. Text and retrieval columns refer to different trained models, not one model',
        'demonstrating both capabilities. Dense masks do not establish sparse-attention training savings.', '',
        'Raw data: [results-v1/runs.json](results-v1/runs.json). Protocol: [DESIGN.md](DESIGN.md).']
(ROOT/'RESULTS-V1.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))


readme = [
    '# Attention-window pilot — V1 hello world', '',
    'A six-seed CPU experiment testing whether local attention trained over connected sequences supports retrieval beyond its window. A follow-up to Timothy DeLise and Seth Cromelin’s [Thinking Outside the Box: Retention and Transmission of Information in Sliding-Window KV Inference](https://arxiv.org/abs/2609.34049).', '',
    'One setting: **128-token sequences, eight layers, width 48, four heads**. Full attention over the whole sequence is the baseline. Sliding models train on the same complete sequences with windows 32, 16, or 8. The W=32 chunk control instead trains on four independent blocks of the same data. Within each seed, initialization, batches, targets, and training budgets match.', '',
    '## Results', '',
    'Text prediction and synthetic retrieval use separate models. Values are mean ± sample standard deviation across six new seeds, using each condition’s native inference rule (rolling attention or independent chunks). Retrieval asks for a stored value 80 tokens before the query; chance is 12.5%.', '',
    '| Condition | Held-out text NLL ↓ | Retrieval at distance 80 ↑ |',
    '|---|---:|---:|',
]
for c in conditions:
    readme.append('| '+c['name']+' | '+ms([native(r)['heldout_nll'] for r in select('text',c['name'])])+' | '+ms([native(r)['80']['accuracy'] for r in select('probe',c['name'])], True)+' |')
sliding_success=sum(r['rolling_metrics']['80']['accuracy']==1 for r in select('probe','sliding-32'))
full_success=sum(r['rolling_metrics']['80']['accuracy']==1 for r in select('probe','full-128'))
readme += [
    '',
    f'Sliding-32 reaches 100% far-distance retrieval in {sliding_success}/6 seeds; full attention does so in {full_success}/6. Giving chunk-trained models the same rolling inference yields '+ms([r['rolling_metrics']['80']['accuracy'] for r in select('probe','chunk-32')],True)+'. This supports connected training on this task; it does not establish general capability equivalence. Seeds vary both initialization and training batches, and none are excluded. Learned positional coverage also differs under answer-only chunk supervision.', '',
    'W=32 retains **93 KiB versus 381 KiB** of KV tensors for the full baseline (24.4%, approximately one quarter). This measures retained KV memory, not total model memory or training cost. Eight layers impose a finite information horizon; the implementation uses dense attention masks.', '',
    '## Reproduce', '',
    'The corpus is included. No GPU or account is required. Use a new output directory to rerun all 60 models; published metrics exclude checkpoints. The summary below reads the preserved V1 metrics.', '',
    '```bash',
    'git clone https://github.com/jefferythewind/attention-window-pilot.git',
    'cd attention-window-pilot',
    'python -m venv .venv',
    'source .venv/bin/activate',
    'pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu',
    'pip install transformers==4.57.6 numpy==2.5.4',
    'python checks.py',
    'python checks_chunk.py',
    'python run_v1.py --output-dir results-v1-reproduction',
    'python summarize_v1.py',
    '```', '',
    '[Design and reproducibility](DESIGN.md) · [Full results and every seed](RESULTS-V1.md) · [Configuration](v1_config.json) · [Raw metrics](results-v1/runs.json) · [Earlier pilots](archive/PILOT-HISTORY.md)',
]
if (ROOT/'RESULTS-TRAINING-COST.md').exists():
    readme+=['', 'Additional V1 measurements: [learning curves](figures/v1-learning-curves.png) and [training time/memory](RESULTS-TRAINING-COST.md).']
if (ROOT/'RESULTS-RECURRENT.md').exists():
    readme+=['', '**Same-layer recurrent slot:** the default-off feature adds no parameters and retains one completed output per layer. In 36 matched new runs, recurrent-8 lowered text loss to 2.3035 ± 0.0056, but distance-80 recall remained near chance (14.3% ± 3.1%). The verified beyond-horizon path did not learn reliable retrieval at this budget. See [all seeds, timings and state sizes](RESULTS-RECURRENT.md) and [reproduction protocol](RECURRENT-DESIGN.md).']
if (ROOT/'RESULTS-W32-HORIZON.md').exists():
    readme+=['', 'A [single-seed W=32 test on 320-token sequences](RESULTS-W32-HORIZON.md) puts the source 280 tokens away, beyond the sliding horizon of 248. Recurrent recall is 27.0% versus sliding’s 14.8%, with source-sensitive predictions and strong local recall: partial learned relay, not reliable exact retrieval.']
if (ROOT/'RESULTS-LATERAL-SLOTS.md').exists():
    readme+=['', 'A [diagonal one-versus-two-slot comparison](RESULTS-LATERAL-SLOTS.md) scores 36.7% versus 28.1% at distance280, with changed-source accuracy39.5% versus24.2%. This is improved partial relay in one paired seed. [Architecture and correctness checks](DIAGONAL-LATERAL-DESIGN.md).']
if (ROOT/'RESULTS-THREE-SLOTS.md').exists():
    readme+=['', 'The [three-slot extension](RESULTS-THREE-SLOTS.md), trained without rerunning controls, scores 22.7% at distance 280 versus two slots’ 36.7%. Local recall also declines; adding t−3 did not improve this one-seed, fixed-budget trial.']
(ROOT/'README.md').write_text('\n'.join(readme)+'\n')

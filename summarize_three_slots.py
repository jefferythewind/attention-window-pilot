"""Three-slot result compared with preserved one/two-slot controls."""
import json
from pathlib import Path
ROOT=Path(__file__).parent
D=json.loads((ROOT/'results-three-slots/runs.json').read_text());assert len(D['results'])==1
old=json.loads((ROOT/'results-lateral-slots/runs.json').read_text())
rows=sorted(old['results']+D['results'],key=lambda r:r['recurrent_attention_slots'])
for field in ('initialization_sha256','training_sequences_sha256','training_labels_sha256','parameters','processed_tokens','supervised_targets','optimizer_updates'):
    assert len({r[field] for r in rows})==1,field
lines=['# Three recurrent connections: one new training run','',
       'Only the three-slot condition was newly trained. The one- and two-slot controls below are preserved measurements from the previous experiment, with no retraining or checkpoint re-evaluation. All three use diagonal execution, seed 515843, eight layers, width 48, four heads, ordinary W=32, batch 16, 320-token sequences and 400 retrieval updates. Initialization, ordered training inputs, labels, parameter count and budgets match by hashes. Each model processes 2,048,000 tokens and 6,400 answer targets.', '',
       'Three slots attend jointly to completed outputs from the same layer at t−1, t−2 and t−3. Existing ln1 and K/V projections, including biases, are reused; there are no added parameters or positional embeddings. W remains 32 ordinary positions including the current token. Missing history is masked at startup. Full-sequence backpropagation is retained, and ordinary K/Vs remain separate from recurrent outputs in streaming state. Diagonal wavefronts remain valid: the added t−3 dependency arrives from three diagonals earlier.', '',
       'Training mixes distances 8/24/48/280 uniformly in expectation. Distance 280 exceeds the nonrecurrent horizon 8 × 31 = 248. Evaluation uses the same 256 held-out examples per distance (seed 2026), excludes the answer token from the prompt, and checks full-sequence/cached equivalence. Chance is 12.5%.', '',
       '| Slots | Distance | Accuracy | Answer NLL | Changed-source accuracy | Prediction changes | Max source logit difference |',
       '|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    for distance in D['config']['distances']:
        m=r['rolling_metrics'][str(distance)]
        lines.append(f'| {r["recurrent_attention_slots"]} | {distance} | {100*m["accuracy"]:.1f}% | {m["answer_nll"]:.4f} | {100*m["counterfactual_accuracy"]:.1f}% | {100*m["prediction_changed_fraction"]:.1f}% | {m["max_changed_source_logit_difference"]:.6g} |')
a,b=[r['rolling_metrics']['280'] for r in rows[1:]]
lines+=['',f'At distance 280, three slots score **{100*b["accuracy"]:.1f}%**, versus **{100*a["accuracy"]:.1f}%** with two ({100*(b["accuracy"]-a["accuracy"]):+.1f} percentage points). Changed-source accuracy is {100*b["counterfactual_accuracy"]:.1f}% versus {100*a["counterfactual_accuracy"]:.1f}%; answer NLL is {b["answer_nll"]:.4f} versus {a["answer_nll"]:.4f}. Changed-source evaluation substitutes value+1 modulo8 and checks that new answer. NLL is in nats.', '',
        ('Three slots have higher observed distant recall in this trial.' if b['accuracy']>a['accuracy'] else 'Adding t−3 does not improve distant recall in this trial. This is a learning outcome under this configuration and budget, not proof that the connection cannot support relay.')+' This is one paired seed; no between-seed standard deviation is available, and the result does not establish reliable learning across seeds.', '',
        'Local recall also declines: three slots score85.2–89.5% across distances8/24/48, versus100% with two. The degradation is therefore broader than the beyond-horizon task alone.', '', '## Timing and retained state','',
        '| Slots | Training seconds | Amortized logical KiB/sequence | Amortized backing KiB/sequence |','|---|---:|---:|---:|']
for r in rows:
    m=r['rolling_metrics']['280'];batch=m['cache_batch_size']
    lines.append(f'| {r["recurrent_attention_slots"]} | {r["training_seconds"]:.1f} | {m["cache_total_logical_bytes"]/batch/1024:.3f} | {m["cache_backing_storage_bytes"]/batch/1024:.3f} |')
lines+=['',f'New three-slot end-to-end wall time: {D["wall_seconds"]:.1f}s. All timings use two PyTorch CPU threads; the earlier runs were measured separately. Training timings include batch creation, forward/backward, clipping and optimizer updates, excluding evaluation/save. Cache totals are amortized over evaluation batch 256; shared position metadata is divided by that batch size. Weights, training activations and allocator/Python overhead are excluded.', '',
        'At batch one, three slots retain 93 KiB of ordinary float32 K/Vs, 4.5 KiB of completed outputs and 2 KiB of logical position/cursor metadata: **99.5KiB** total logical tensor state. This is inference-state accounting, not a training-memory reduction.', '',
        '## Reproduce','',
        '```bash','.venv/bin/python checks_diagonal.py','.venv/bin/python run_three_slots.py --output-dir results-three-slots-reproduction','.venv/bin/python summarize_three_slots.py','.venv/bin/python plot_three_slots.py','```','',
        'The runner trains only one model and verifies its data/initialization/budget hashes against the preserved controls. A new output directory retrains; summary and plot read the preserved default results. Tests include one/two/three-slot agreement with an independent joint-softmax reference (outputs and parameter gradients), full/token/chunk consistency, startup masking, future-token causality, resets, cache bounds, save/load, an isolated t−3 forward/gradient dependency and a controlled beyond-horizon source path.', '',
        '[Training source snapshot](https://github.com/jefferythewind/attention-window-pilot/tree/f072388156954c6f7858778b89b689443e6f9c6e) · [Configuration](three_slots_config.json) · [New raw results and source hashes](results-three-slots/runs.json) · [Preserved controls](results-lateral-slots/runs.json) · [Diagonal execution design](DIAGONAL-LATERAL-DESIGN.md)', '',
        '![One/two/three-slot retrieval comparison](figures/three-slots.png)']
(ROOT/'RESULTS-THREE-SLOTS.md').write_text('\n'.join(lines)+'\n')
print(lines[0]);print(lines[23])

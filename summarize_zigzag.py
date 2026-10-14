"""Intended all-layer zigzag trial; preserve top-only as a separately named variant."""
import json
from pathlib import Path
ROOT=Path(__file__).parent
D=json.loads((ROOT/'results-zigzag/runs.json').read_text());assert len(D['results'])==1
new=D['results'][0];rows=[]
for directory,names in [('results-w32-horizon',{'sliding-32'}),('results-lateral-slots',{'recurrent-1slot-diagonal','recurrent-2slot-diagonal'}),('results-three-slots',{'recurrent-3slot-diagonal'}),('results-two-sweep',{'two-sweep-top-recurrent-w32'})]:
    old=json.loads((ROOT/directory/'runs.json').read_text())
    rows += [dict(r,threads=old['config']['threads_per_worker']) for r in old['results'] if r['condition'] in names]
rows.append(dict(new,threads=2))
for field in ('initialization_sha256','training_sequences_sha256','training_labels_sha256','parameters','processed_tokens','supervised_targets','optimizer_updates'):
    assert len({r[field] for r in rows})==1,field
pct=lambda x:f'{100*x:.6f}'.rstrip('0').rstrip('.')+'%'
lines=['# Intended all-layer full-window zigzag: one exploratory seed','',
       'Only the all-layer zigzag condition was newly trained. The completed top-layer-only two-sweep trial is a separate variant; its saved results are included for comparison. Ordinary sliding and earlier interleaved recurrent conditions are also preserved, without retraining or re-evaluating their retrieval checkpoints.', '',
       'Each ordinary layer consumes the preceding corrected row R, produces U after its ordinary attention/residual/MLP, then receives a full-window same-layer recurrent correction. Its corrected R feeds the next ordinary layer. Historical lateral K/Vs come from this layer’s completed R; the current temporary self slot comes from U. Reused LN1/Q/K/V/output weights and biases add no parameter, second MLP, gate or positional embedding. Every corrected row has a verified path to the final head and loss.', '',
       'The protocol remains 320-token complete sequences, eight layers, width 48, four heads, ordinary and lateral W=32, batch 16, seed 515843, 400 retrieval updates, distances 8/24/48/280, the original optimizer and two CPU threads. Initialization, training inputs/labels and token/target/update budgets match by hashes. Training retains the complete autograd graph. Evaluation uses the same 256 held-out examples per distance, seed 2026, with the answer token excluded from the prompt. Chance is 12.5%; exact percentages below have denominator 256.', '',
       '## Main comparison at distance 280','',
       '| Condition | Exact accuracy | Answer NLL | Changed-source accuracy | Prediction changes | Max source logit difference |','|---|---:|---:|---:|---:|---:|']
for r in rows:
    m=r['rolling_metrics']['280'];lines.append(f'| {r["condition"]} | {pct(m["accuracy"])} | {m["answer_nll"]:.6f} | {pct(m["counterfactual_accuracy"])} | {pct(m["prediction_changed_fraction"])} | {m["max_changed_source_logit_difference"]:.6g} |')
m=new['rolling_metrics']['280']
lines+=['',f'Zigzag recalls {round(256*m["accuracy"])}/256 original values and {round(256*m["counterfactual_accuracy"])}/256 changed-source values. Source substitution changes predictions on {round(256*m["prediction_changed_fraction"])}/256 examples; mean absolute source-induced logit difference is {m["mean_changed_source_logit_difference"]:.6g}. Changed-source evaluation changes only the stored value to value+1 modulo8 and checks that new answer. Answer NLL is in nats.', '',
        'Zigzag reaches 100% recall at distances 8/24/48 but remains near chance at 280, with no source-induced prediction changes. This configuration failed to learn useful distant retrieval under 400 updates; it is not evidence that the verified connection cannot support relay.', '',
        'Distance 280 exceeds the original ordinary horizon 8 × 31 = 248. However, U-only local history in two attention operations per layer would have finite reach 2 × 8 × 31 = 496. Thus successful source-sensitive retrieval at 280 is beyond the original backbone’s horizon but cannot isolate recurrent-history relay from the added finite attention depth. A short controlled source test beyond both corresponding small-model bounds verifies the recurrent dependency path. These are separate claims; graph connectivity alone does not establish learned unbounded memory. One seed has no between-seed standard deviation and cannot establish reliable learning across seeds.', '',
        '## All distances','', '| Condition | Distance | Exact accuracy | Answer NLL | Changed-source accuracy |','|---|---:|---:|---:|---:|']
for r in rows:
    for d in D['config']['distances']:
        m=r['rolling_metrics'][str(d)];lines.append(f'| {r["condition"]} | {d} | {pct(m["accuracy"])} | {m["answer_nll"]:.6f} | {pct(m["counterfactual_accuracy"])} |')
lines+=['','## Training durations','', '| Condition | Threads | Training seconds |','|---|---:|---:|']
for r in rows:lines.append(f'| {r["condition"]} | {r["threads"]} | {r["training_seconds"]:.1f} |')
lines+=['',f'New zigzag wall time including evaluation/save: {D["wall_seconds"]:.1f}s. Full-run peak training-worker process RSS before evaluation: **{new["training_peak_process_rss_bytes"]/2**20:.1f}MiB**. RSS includes runtime, model, optimizer, inputs and allocator storage. Training durations include sampling, forward/backward, clipping and updates, excluding final evaluation/save. Earlier durations were measured separately; ordinary sliding used one thread. They are historical measurements, not a controlled training-speedup benchmark.', '',
        '## Common checkpoint measurements','']
p=ROOT/'results-zigzag/costs.json'
if p.exists():
    fresh=json.loads(p.read_text());assert len(fresh['results'])==1
    old=json.loads((ROOT/'results-two-sweep/costs.json').read_text());costs=old['results']+fresh['results']
    lines+=['Profiles use the same CPU and two threads. Older checkpoint profiles are preserved; only zigzag is newly profiled. Each uses a fresh process, one fixed batch 16 × 320 forward/backward pass with no optimizer update, and bitwise-unchanged parameters. Prefill/decode use two warmups, five 319-token prefills and twenty one-token calls from the same frozen prefix. Medians are separate. A cold single forward/backward timing is diagnostic, not a steady-state training benchmark.', '',
            '| Condition | Forward/backward ms | Peak training-pass RSS MiB | Unique autograd saved storage MiB |','|---|---:|---:|---:|']
    for r in costs:lines.append(f'| {r["condition"]} | {r["training_forward_backward_seconds"]*1000:.1f} | {r["fixed_batch_training_peak_process_rss_bytes"]/2**20:.1f} | {r["autograd_unique_saved_storage_bytes"]/2**20:.1f} |')
    lines+=['','RSS includes process/runtime/weights; unique autograd storage deduplicates saved allocations and includes parameters/inputs. Those are different measures from the actual full-run peak above.','',
            '| Condition | Batch | Prefill ms | Cached decode ms |','|---|---:|---:|---:|']
    for r in costs:
        for b,m in r['latencies_and_state_by_batch'].items():lines.append(f'| {r["condition"]} | {b} | {m["prefill_median_seconds"]*1000:.3f} | {m["decode_median_seconds"]*1000:.3f} |')
    lines+=['','## Retained streaming state and temporary activations','',
            'Batch-one tensor state after 319-token prefill. Old output-state recurrence and new corrected-history KV recurrence have different budgets. Older ordinary cache slices may retain full prefill allocations; compact returned caches in the new modes avoid that backing-storage effect. Weights, activations, allocator and Python overhead are excluded from retained-state totals.','',
            '| Condition | Ordinary KV KiB | Recurrent KV KiB | Recurrent output KiB | Metadata KiB | Logical total KiB | Backing KiB |','|---|---:|---:|---:|---:|---:|---:|']
    for r in costs:
        m=r['latencies_and_state_by_batch']['1'];v=[m['cache_kv_bytes'],m.get('cache_recurrent_kv_bytes',0),m.get('cache_recurrent_output_bytes',0),m['cache_metadata_bytes'],m['cache_total_logical_bytes'],m['cache_backing_storage_bytes']]
        lines.append('| '+r['condition']+' | '+' | '.join(f'{x/1024:.3f}' for x in v)+' |')
    r=fresh['results'][0]
    lines+=['',f'Diagnostic grid instrumentation counts a zigzag U row at **{r["zigzag_U_row_logical_bytes"]/2**20:.3f}MiB** for batch 16 × 320 × width 48. U across all layers has logical volume **{r["zigzag_U_all_rows_logical_volume_bytes"]/2**20:.3f}MiB**; R across all layers has volume **{r["zigzag_R_all_rows_logical_volume_bytes"]/2**20:.3f}MiB**. Default inference does not retain lists of all these rows; these totals are not simultaneous peak temporary allocations. Each batch-one 319-token U row is 59.8125KiB. Training autograd retains tensors required for backward, measured separately above. Grid/saved-storage counts overlap with process RSS and are not disjoint sums. No total-memory reduction is assumed.']
lines+=['','## Acceptance checks and reproduction','',
        'Independent explicit all-layer reference values and parameter gradients passed. Tests verify corrected previous-row inputs, a real nonzero gradient to every R row, completed same-row R-history intervention, causal/full/token/chunk consistency, cache/window/startup/reset/independent-example/absolute-position handling, save/load, and a controlled source beyond both the ordinary and U-only finite test horizons. Top-only and baseline regression checks also passed. No auxiliary supervision or silent cross-row lateral attention was introduced.', '',
        '```bash','.venv/bin/python checks_zigzag.py','.venv/bin/python checks_two_sweep.py','.venv/bin/python run_zigzag.py --output-dir results-zigzag-reproduction','.venv/bin/python measure_two_sweep.py --checkpoint PATH_TO_ZIGZAG_CHECKPOINT','.venv/bin/python summarize_zigzag.py','.venv/bin/python plot_zigzag.py','```','',
        'Only zigzag is trained in the new output directory. Comparison retrieval/cost metrics are preserved; checkpoint binaries are not published. The cost worker prints JSON for the supplied checkpoint. Reports/plots read the preserved default results. Plot dependencies are in requirements-plots.txt.', '',
        f'Training revision: [{D["source_git_commit"][:7]}](https://github.com/jefferythewind/attention-window-pilot/tree/{D["source_git_commit"]}). [Configuration](zigzag_config.json) · [Raw zigzag results and source hashes](results-zigzag/runs.json) · [New cost profile](results-zigzag/costs.json) · [Intended architecture and checks](ZIGZAG-DESIGN.md) · [Separate top-only variant](RESULTS-TWO-SWEEP.md).', '',
        '![All-layer zigzag with preserved variants](figures/zigzag.png)']
(ROOT/'RESULTS-ZIGZAG.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:22]))

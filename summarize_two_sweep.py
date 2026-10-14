"""Two-sweep retrieval and separate training/prefill/decode/state measurements."""
import json
from pathlib import Path
ROOT=Path(__file__).parent
D=json.loads((ROOT/'results-two-sweep/runs.json').read_text());assert len(D['results'])==1
new=D['results'][0];rows=[]
for path,conditions in [('results-w32-horizon',{'sliding-32'}),('results-lateral-slots',{'recurrent-1slot-diagonal','recurrent-2slot-diagonal'}),('results-three-slots',{'recurrent-3slot-diagonal'})]:
    data=json.loads((ROOT/path/'runs.json').read_text())
    for r in data['results']:
        if r['condition'] in conditions:rows.append(dict(r,historical_threads=data['config']['threads_per_worker']))
rows.append(dict(new,historical_threads=2))
for field in ('initialization_sha256','training_sequences_sha256','training_labels_sha256','parameters','processed_tokens','supervised_targets','optimizer_updates'):
    assert len({r[field] for r in rows})==1,field
pct=lambda x:f'{100*x:.6f}'.rstrip('0').rstrip('.')+'%'
lines=['# Top-layer-only two-sweep attention: separate variant','',
       'This is the separately retained top-only variant, not the intended all-layer zigzag experiment. Only the top-only model was trained for these measurements. The ordinary sliding and interleaved one/two/three-slot measurements are preserved controls, without retraining. All use seed 515843, eight layers, width 48, four heads, ordinary W=32, batch 16, 320-token full sequences, 400 retrieval updates and the same AdamW recipe. Initialization, ordered inputs, labels, parameter counts, processed tokens, answer targets and update budgets match by hashes. The new lateral window is also 32. No text models were trained.', '',
       'The ordinary backbone runs first without alteration. A separate top-only sweep then attends to completed corrected-state history K/Vs and the current ordinary U self slot, reusing the existing top block’s LN1/Q/K/V/output weights and biases. It adds no MLP, gate, embedding or parameter. Both sweeps retain the full training graph. Lower corrected rows are deliberately absent: without extra supervision or vertical integration they would not reach the existing top-only head. This is a new architecture, not an equivalent implementation of the interleaved recurrence.', '',
       '## Beyond-horizon retrieval','',
       'Distance 280 exceeds the ordinary horizon 8 × (32−1) = 248. Evaluation uses the same 256 examples per distance, validation seed 2026; exact percentages below have denominator 256. Chance is 12.5%. The answer token is excluded from the prompt. Changed-source evaluation substitutes value+1 modulo8 while leaving the other prompt tokens fixed and scores the new answer.', '',
       '| Condition | Accuracy d280 | Answer NLL | Changed-source accuracy | Prediction changes | Max source logit difference |',
       '|---|---:|---:|---:|---:|---:|']
for r in rows:
    m=r['rolling_metrics']['280'];lines.append(f'| {r["condition"]} | {pct(m["accuracy"])} | {m["answer_nll"]:.6f} | {pct(m["counterfactual_accuracy"])} | {pct(m["prediction_changed_fraction"])} | {m["max_changed_source_logit_difference"]:.6g} |')
m=new['rolling_metrics']['280']
lines+=['',f'The new model retrieves {round(256*m["accuracy"])}/256 original values and {round(256*m["counterfactual_accuracy"])}/256 changed-source values. Source substitution changes predictions on {round(256*m["prediction_changed_fraction"])}/256 examples. Mean absolute source-induced logit difference is {m["mean_changed_source_logit_difference"]:.6g}. These task scores assess learned source-sensitive relay; the separate controlled gradient test only establishes a dependency path. One seed has no between-seed standard deviation and cannot establish reliable or arbitrary-distance memory.', '',
        'The new model reaches 100% at distances 8/24/48 but is near chance at 280, with no source-induced prediction changes. This configuration failed to learn useful beyond-horizon retrieval within 400 updates. The controlled graph checks establish capacity for a dependency path, not successful learning or proof that the architecture cannot relay information.', '', '## All retrieval distances','',
        '| Condition | Distance | Exact accuracy | Answer NLL | Changed-source accuracy |','|---|---:|---:|---:|---:|']
for r in rows:
    for distance in D['config']['distances']:
        m=r['rolling_metrics'][str(distance)];lines.append(f'| {r["condition"]} | {distance} | {pct(m["accuracy"])} | {m["answer_nll"]:.6f} | {pct(m["counterfactual_accuracy"])} |')
lines+=['','## Complete training durations','',
        '| Condition | Threads | Training seconds |','|---|---:|---:|']
for r in rows:lines.append(f'| {r["condition"]} | {r["historical_threads"]} | {r["training_seconds"]:.1f} |')
lines+=['',f'New end-to-end wall time: {D["wall_seconds"]:.1f}s. New training worker peak process RSS before final evaluation: **{new["training_peak_process_rss_bytes"]/2**20:.1f}MiB**. This includes runtime, weights, optimizer, inputs and allocator storage, not only activations. Full training timings include sampling, forward/backward, clipping and optimizer steps, but exclude final evaluation/save. Existing controls were measured separately; the older ordinary sliding run used one thread. Do not infer a controlled architectural speedup from those historical durations.', '',
        '## Common checkpoint cost profiles','']
cost_path=ROOT/'results-two-sweep/costs.json'
if cost_path.exists():
    costs=json.loads(cost_path.read_text());assert len(costs['results'])==5
    lines+=['Each saved checkpoint is measured in a fresh process, on the same CPU with two threads. No optimizer updates occur and weights remain bitwise unchanged. Training forward/backward is one fixed batch 16 × 320 pass, without warmup or optimizer; its timing is a diagnostic, not a converged steady-state training benchmark. Prefill and decoding use two warmups, five timed full-prefill calls and twenty timed one-token calls from a frozen prefix. Median latencies are reported separately.', '',
            '| Condition | Fixed-batch forward/backward ms | Peak process RSS MiB | Autograd saved storage MiB |','|---|---:|---:|---:|']
    for r in costs['results']:lines.append(f'| {r["condition"]} | {r["training_forward_backward_seconds"]*1000:.1f} | {r["fixed_batch_training_peak_process_rss_bytes"]/2**20:.1f} | {r["autograd_unique_saved_storage_bytes"]/2**20:.1f} |')
    lines+=['','RSS is the process high-water mark during the training pass, including imports/weights/inputs. Autograd storage deduplicates saved allocations and includes parameters/inputs; it is not peak process memory. These differ from the full-run worker peak above.','',
            '| Condition | Batch | 319-token prefill ms | Cached one-token decode ms |','|---|---:|---:|---:|']
    for r in costs['results']:
        for b,m in r['latencies_and_state_by_batch'].items():lines.append(f'| {r["condition"]} | {b} | {m["prefill_median_seconds"]*1000:.3f} | {m["decode_median_seconds"]*1000:.3f} |')
    baseline=costs['results'][1]['latencies_and_state_by_batch']['1'];current=costs['results'][-1]['latencies_and_state_by_batch']['1']
    lines+=['',f'Batch-one two-sweep prefill is {baseline["prefill_median_seconds"]/current["prefill_median_seconds"]:.2f}× faster than the saved one-slot interleaved model in this measurement; cached decoding is {baseline["decode_median_seconds"]/current["decode_median_seconds"]:.2f}× faster. These are local latency comparisons between different architectures and learned behaviors, not evidence of equal capability or general hardware speedup.','', '## Retained streaming tensors and temporary grid','',
            'Batch-one state immediately after 319-token prefill. Ordinary and recurrent KV are distinct; older interleaved models instead retain completed output states. Actual backing allocations can exceed logical window sizes because the older paths return slices; the two-sweep path compacts its returned cache. This allocation-policy difference is separate from the architectural state budget.','',
            '| Condition | Ordinary KV KiB | Recurrent KV KiB | Recurrent output KiB | Metadata KiB | Logical total KiB | Backing KiB |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for r in costs['results']:
        m=r['latencies_and_state_by_batch']['1'];v=[m['cache_kv_bytes'],m.get('cache_recurrent_kv_bytes',0),m.get('cache_recurrent_output_bytes',0),m['cache_metadata_bytes'],m['cache_total_logical_bytes'],m['cache_backing_storage_bytes']]
        lines.append('| '+r['condition']+' | '+' | '.join(f'{x/1024:.3f}' for x in v)+' |')
    r=costs['results'][-1];grid=r['first_sweep_completed_grid_logical_bytes']
    lines+=['',f'The new model explicitly materializes the completed first-sweep grid U[1..8]: **{grid/2**20:.3f}MiB** for training batch 16 × 320 × width 48, excluding embeddings and other activations. The corresponding batch-one 319-token prefill grid is **{8*319*48*4/1024:.3f}KiB**. This grid is temporary; it is not retained streaming state. Autograd also retains other tensors, captured separately by the saved-storage measurement. The parallel lateral Q/K/V projection is another 2.8125MiB at training shape, not included in the grid count. RSS already includes these tensors; the grid and saved-storage counts describe overlapping parts of memory. These volumes are not total or peak activation memory.', '',
            'No total-memory reduction is assumed: a top-row recurrent KV history adds 31 × two × width 48 × four bytes = 11.625KiB at batch one, on top of ordinary KV storage.']
else:lines+=['Cost profiling has not completed.']
lines+=['','## Correctness and reproduction','',
        'Acceptance checks passed for baseline-mode exact outputs/gradients, ordinary first-sweep/grid/cache equivalence, independent corrected-history reference values/gradients, top-corrected readout and lower-backbone gradients, causal future/answer exclusion, full/token/chunk consistency, startup/reset/cache/absolute-position handling, completed-R intervention, a controlled beyond-horizon forward/gradient dependency, no new parameters and save/load. No optional all-row mode is implemented.', '',
        '```bash','.venv/bin/python checks_two_sweep.py','.venv/bin/python run_two_sweep.py --output-dir results-two-sweep-reproduction','.venv/bin/python measure_two_sweep.py','.venv/bin/python summarize_two_sweep.py','.venv/bin/python plot_two_sweep.py','```','',
        'Use a new output directory to train only the new condition. Cost comparisons require the locally saved historical checkpoint binaries; they are excluded from Git. On a fresh clone, measure the new saved checkpoint with `measure_two_sweep.py --checkpoint PATH`. Report/plot read the preserved default output. Plotting dependencies are in requirements-plots.txt.', '',
        f'Training revision: [{D["source_git_commit"][:7]}](https://github.com/jefferythewind/attention-window-pilot/tree/{D["source_git_commit"]}). [Raw retrieval results and exact source hashes](results-two-sweep/runs.json) · [Raw costs](results-two-sweep/costs.json) · [Configuration](two_sweep_config.json) · [Architecture, measurement definitions and checks](TWO-SWEEP-DESIGN.md).', '',
        '![Two-sweep and preserved-control retrieval](figures/two-sweep.png)']
(ROOT/'RESULTS-TWO-SWEEP.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:20]))

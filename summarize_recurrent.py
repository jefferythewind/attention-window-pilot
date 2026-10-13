"""All-seed recurrent-slot results, paired V1 references, timing and retained state."""
import argparse
import json
from pathlib import Path
import statistics as S
from run_recurrent import check_pairs

ROOT=Path(__file__).parent
parser=argparse.ArgumentParser();parser.add_argument('--input-dir',type=Path,default=ROOT/'results-recurrent')
args=parser.parse_args()
D=json.loads((args.input_dir/'runs.json').read_text());cfg=D['config'];rows=D['results']
check_pairs(rows,cfg,complete=True)
seeds=cfg['seeds'];distances=cfg['distances'];conditions=cfg['conditions']

def select(task,name):return [next(r for r in rows if r['task']==task and r['condition']==name and r['seed']==s) for s in seeds]
def metrics(r):return r['chunked_metrics'] if r['training_mode']=='chunk' else r['rolling_metrics']
def fmt(v,percentage=False):
    v=[x*(100 if percentage else 1) for x in v];precision=1 if percentage else 4
    return f'{S.mean(v):.{precision}f} ± {S.stdev(v):.{precision}f}'+('%' if percentage else '')

lines=['# One-slot same-layer recurrent attention: results','','V1 is preserved; 36 new recurrent models use the same six initializations, original batches, labels and update budgets. Means ± sample SD across six seeds. Text and retrieval are separately trained models.','','## Text loss and retrieval accuracy','','| Condition | Text NLL ↓ | Recall d=8 | d=24 | d=48 | d=80 |','|---|---:|---:|---:|---:|---:|']
for c in conditions:
    name=c['name'];t=select('text',name);p=select('probe',name)
    lines.append('| '+name+' | '+fmt([metrics(r)['heldout_nll'] for r in t])+' | '+' | '.join(fmt([metrics(r)[str(d)]['accuracy'] for r in p],True) for d in distances)+' |')
lines+=['','V1 chunk-32 uses native disconnected-block evaluation in this table; other conditions use their full/rolling inference. Chance is 12.5% among eight values.','','## Retrieval answer loss','','| Condition | d=8 NLL | d=24 NLL | d=48 NLL | d=80 NLL |','|---|---:|---:|---:|---:|']
for c in conditions:
    p=select('probe',c['name']);lines.append('| '+c['name']+' | '+' | '.join(fmt([metrics(r)[str(d)]['answer_nll'] for r in p]) for d in distances)+' |')
lines+=['','## Changed-source accuracy','','The source value is incremented modulo eight; success requires predicting the new value.','','| Condition | d=8 | d=24 | d=48 | d=80 |','|---|---:|---:|---:|---:|']
for c in conditions:
    p=select('probe',c['name']);lines.append('| '+c['name']+' | '+' | '.join(fmt([metrics(r)[str(d)]['counterfactual_accuracy'] for r in p],True) for d in distances)+' |')
lines+=['','## Primary comparison: distance 80 beyond horizon 56','','| Seed | Sliding8 recall | Recurrent8 recall | Sliding8 NLL | Recurrent8 NLL | Sliding8 changed source | Recurrent8 changed source | Recurrent8 prediction changes |','|---:|---:|---:|---:|---:|---:|---:|---:|']
base=select('probe','sliding-8');rec=select('probe','recurrent-8')
for a,b in zip(base,rec):
    ma,mb=metrics(a)['80'],metrics(b)['80']
    lines.append(f'| {a["seed"]} | {ma["accuracy"]*100:.1f}% | {mb["accuracy"]*100:.1f}% | {ma["answer_nll"]:.4f} | {mb["answer_nll"]:.4f} | {ma["counterfactual_accuracy"]*100:.1f}% | {mb["counterfactual_accuracy"]*100:.1f}% | {mb["prediction_changed_fraction"]*100:.1f}% |')
diffs=[metrics(b)['80']['accuracy']-metrics(a)['80']['accuracy'] for a,b in zip(base,rec)]
lines+=['',f'Paired recall advantage, recurrent8 − sliding8: {fmt(diffs,True).replace('%',' percentage points')}.',f'Recurrent8 seeds with at least 90% recall at d=80: {sum(metrics(r)["80"]["accuracy"]>=.9 for r in rec)}/6; with at least 90% changed-source accuracy: {sum(metrics(r)["80"]["counterfactual_accuracy"]>=.9 for r in rec)}/6. These counts are descriptive, not a predeclared success threshold.']
far_accuracy=[metrics(r)['80']['accuracy'] for r in rec]
far_changed=[metrics(r)['80']['counterfactual_accuracy'] for r in rec]
if all(a>=.9 and b>=.9 for a,b in zip(far_accuracy,far_changed)):
    conclusion='All six seeds achieve at least 90% original and changed-source accuracy at distance 80; this supports reliable relay in this specific pilot.'
else:
    conclusion='The target of reliable source-sensitive distance-80 retrieval is not met across these six seeds under the specified 400-update budget. This configuration failed to learn reliable relay; it is not evidence that the connection cannot support relay.'
lines+=['',conclusion,'',f'At distance 80, predictions change after source substitution in {sum(metrics(r)["80"]["prediction_changed_fraction"]>0 for r in rec)}/6 recurrent8 seeds. Some source dependence can exist without accurate value recovery. Controlled forward/gradient tests establish the path, not successful task learning.']
lines+=['','## Every new seed and distance']
for c in conditions:
    if not c.get('recurrent_attention_slot'):continue
    name=c['name'];lines+=['',f'### {name}: text','','| Seed | Held-out NLL | Training seconds |','|---:|---:|---:|']
    for r in select('text',name):lines.append(f'| {r["seed"]} | {metrics(r)["heldout_nll"]:.4f} | {r["training_seconds"]:.1f} |')
    for field,label in [('accuracy','Recall'),('answer_nll','Answer NLL'),('counterfactual_accuracy','Changed-source accuracy')]:
        lines+=['',f'**{label}**','','| Seed | d=8 | d=24 | d=48 | d=80 |','|---:|---:|---:|---:|---:|']
        for r in select('probe',name):
            vs=[metrics(r)[str(d)][field] for d in distances]
            vals=[f'{v:.4f}' for v in vs] if field=='answer_nll' else [f'{100*v:.1f}%' for v in vs]
            lines.append('| '+str(r['seed'])+' | '+' | '.join(vals)+' |')
lines+=['','## Timings','','Training durations overlap across three workers; each uses one CPU thread. New recurrent timings exclude evaluation and checkpoint writes. V1 rows retain their historical measurements. They are contextual observations, not a new controlled speed benchmark.','','| Condition | Text training seconds | Retrieval training seconds |','|---|---:|---:|']
for c in conditions:
    lines.append('| '+c['name']+' | '+' | '.join(fmt([r['training_seconds'] for r in select(task,c['name'])]) for task in ('text','probe'))+' |')
new=[r for r in rows if not r.get('reused_v1')]
lines+=['',f'End-to-end new-run wall time: {D["wall_seconds"]:.1f}s. Sum of overlapping new-model training durations: {sum(r["training_seconds"] for r in new):.1f}s.','','## Retained inference state','','The following logical tensor sizes use batch size one, float32 K/V and completed outputs, int64 positions. W counts ordinary positions including current; only W−1 ordinary positions are retained after a token. Recurrent slot K/V are recomputed from the one completed output rather than stored.','','| Condition | Ordinary KV KiB | Recurrent output KiB | Position metadata KiB | Total logical KiB |','|---|---:|---:|---:|---:|']
for c in conditions:
    if c['mode']=='chunk':continue
    w=c['window'];flag=c.get('recurrent_attention_slot',False)
    kv=2*8*(w-1)*48*4;output=8*48*4 if flag else 0;meta=8*(w-1)*8+(8*8 if flag else 0)
    if flag:
        for r in select('probe',c['name']):
            for m in metrics(r).values():
                assert m['cache_kv_bytes']/m['cache_batch_size']==kv
                assert m['cache_recurrent_output_bytes']/m['cache_batch_size']==output
                assert m['cache_metadata_bytes']==meta
                assert m['cache_total_logical_bytes']==(kv+output)*m['cache_batch_size']+meta
    lines.append(f'| {c["name"]} | {kv/1024:.3f} | {output/1024:.3f} | {meta/1024:.3f} | {(kv+output+meta)/1024:.3f} |')
state_measurements=json.loads((args.input_dir/'state-bytes.json').read_text())
lines+=['','### Actual cache backing allocations','','Batch one, 127 processed tokens, chunk-16 prefill; same input shape and external chunk policy. The baseline vectorizes each prefill chunk; recurrence internally processes single tokens. Logical tensor bytes differ from backing allocations when eviction uses slices; backing sizes therefore depend on execution/chunking, not only architecture.','','| Condition | Total logical KiB | Deduplicated backing KiB |','|---|---:|---:|']
for r in state_measurements['results']:
    lines.append(f'| {r["condition"]} | {r["cache_total_logical_bytes"]/1024:.3f} | {r["cache_backing_storage_bytes"]/1024:.3f} |')
lines+=['','Raw JSON also records actual batched-cache logical bytes and deduplicated backing-storage bytes. Position metadata is shared across the batch. Sliced K/V views may retain larger backing allocations; weights, allocator overhead, inference temporaries and the training graph are excluded from this table. No training-memory reduction is claimed.','','## Interpretation','','The old sliding8 architecture has no input-to-query path at distance 80; its horizon is 56. The recurrent version has a path, verified independently by controlled forward and gradient tests. Useful relay still requires accurate, source-sensitive held-out retrieval across seeds. A connected graph or a tiny nonzero logit change alone does not establish it.','','Recurrent8 also struggles at the short distance 8 (15.5% mean accuracy), so this failure does not isolate long-range transmission from broader learning/optimization difficulty. This configuration failed to learn reliable relay under the initial budget; it does not prove the connection cannot support relay. No seeds are excluded, and seeds change both initialization and training sampling. The finite training budget, absolute positional embeddings and single-value synthetic task limit generalization claims.','','Protocol: [RECURRENT-DESIGN.md](RECURRENT-DESIGN.md). Config: [recurrent_config.json](recurrent_config.json). Raw metrics: [results-recurrent/runs.json](results-recurrent/runs.json).']
(ROOT/'RESULTS-RECURRENT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:20]))

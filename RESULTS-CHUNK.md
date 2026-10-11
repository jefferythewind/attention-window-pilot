# Independently chunk-trained controls

Two controls serve different questions: final-window recomputation tests the value of retained
states at inference; independently chunked training tests connected versus disconnected
training. The new control is paired with both existing pilots.

Every chunk uses full causal attention internally and original absolute positions. Models
share initialization, ordered original sequences, all shifted targets (including boundaries),
processed-token counts, optimizer steps, and optimizer settings with their sliding-trained pair.
Maximum attention span is matched; average attention coverage and training compute are not.

Four evaluations separate training and inference: sliding-trained/rolling inference,
sliding-trained/independent chunks, chunk-trained/independent chunks, and chunk-trained/rolling.
Full-window rows reuse the existing full-attention checkpoints; the one-chunk equivalence was checked.

Text and retrieval results use separately trained models. Individual seed results are
descriptive, not confidence intervals. This remains a small, short-training pilot.

## Initial: 4 layers, 64 tokens

### Held-out next-character loss

| Window | Sliding trained, rolling | Sliding trained, chunks | Chunk trained, chunks | Chunk trained, rolling |
|---:|---:|---:|---:|---:|
| 64 | 2.5255 | 2.5255 | 2.5255 | 2.5255 |
| 16 | 2.4937 | 2.4871 | 2.4621 | 2.5233 |
| 8 | 2.4673 | 2.4885 | 2.4475 | 2.5257 |
| 4 | 2.3949 | 2.5181 | 2.4513 | 2.5072 |

All columns score the same original next-character targets, including chunk-boundary
predictions. Lower is better; units are nats per character.

### Retrieval accuracy, mean across seeds

| Window | Distance | Sliding trained, rolling | Sliding trained, chunks | Chunk trained, chunks | Chunk trained, rolling |
|---:|---:|---:|---:|---:|---:|
| 64 | 4 | 100.0% | 100.0% | 100.0% | 100.0% |
| 64 | 12 | 100.0% | 100.0% | 100.0% | 100.0% |
| 64 | 24 | 100.0% | 100.0% | 100.0% | 100.0% |
| 64 | 40 | 100.0% | 100.0% | 100.0% | 100.0% |
| 16 | 4 | 100.0% | 100.0% | 100.0% | 100.0% |
| 16 | 12 | 100.0% | 100.0% | 100.0% | 100.0% |
| 16 | 24 | 100.0% | 12.5% | 11.7% | 19.9% |
| 16 | 40 | 100.0% | 11.9% | 13.3% | 13.3% |
| 8 | 4 | 100.0% | 100.0% | 100.0% | 100.0% |
| 8 | 12 | 100.0% | 12.9% | 14.3% | 15.2% |
| 8 | 24 | 29.3% | 11.1% | 15.0% | 15.0% |
| 8 | 40 | 13.1% | 12.3% | 11.7% | 11.7% |
| 4 | 4 | 73.6% | 13.3% | 13.7% | 13.7% |
| 4 | 12 | 19.1% | 11.7% | 13.7% | 13.7% |
| 4 | 24 | 11.7% | 11.1% | 10.9% | 10.9% |
| 4 | 40 | 13.3% | 14.5% | 12.5% | 12.5% |

### Quarter-window (W=16), distance 40: individual seeds

| Seed | Sliding/rolling accuracy | Chunk/chunks accuracy | Chunk/rolling accuracy | Sliding/rolling answer NLL | Chunk/chunks answer NLL | Chunk/rolling answer NLL |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0% | 14.1% | 14.1% | 0.0079 | 2.0855 | 2.0856 |
| 29 | 100.0% | 12.5% | 12.5% | 0.0023 | 2.0977 | 2.0980 |

Raw results: [results-chunk/runs.json](results-chunk/runs.json).

## Doubled: 8 layers, 128 tokens

### Held-out next-character loss

| Window | Sliding trained, rolling | Sliding trained, chunks | Chunk trained, chunks | Chunk trained, rolling |
|---:|---:|---:|---:|---:|
| 128 | 2.5433 | 2.5433 | 2.5433 | 2.5433 |
| 32 | 2.5124 | 2.5059 | 2.4634 | 2.5475 |
| 16 | 2.4775 | 2.4841 | 2.4358 | 2.5309 |
| 8 | 2.4414 | 2.4751 | 2.4189 | 2.5256 |

All columns score the same original next-character targets, including chunk-boundary
predictions. Lower is better; units are nats per character.

### Retrieval accuracy, mean across seeds

| Window | Distance | Sliding trained, rolling | Sliding trained, chunks | Chunk trained, chunks | Chunk trained, rolling |
|---:|---:|---:|---:|---:|---:|
| 128 | 8 | 100.0% | 100.0% | 100.0% | 100.0% |
| 128 | 24 | 100.0% | 100.0% | 100.0% | 100.0% |
| 128 | 48 | 100.0% | 100.0% | 100.0% | 100.0% |
| 128 | 80 | 100.0% | 100.0% | 100.0% | 100.0% |
| 32 | 8 | 81.6% | 81.6% | 84.6% | 84.6% |
| 32 | 24 | 80.5% | 80.5% | 88.3% | 88.3% |
| 32 | 48 | 81.6% | 11.1% | 11.3% | 13.7% |
| 32 | 80 | 80.5% | 10.4% | 13.7% | 13.7% |
| 16 | 8 | 53.1% | 47.1% | 69.9% | 69.5% |
| 16 | 24 | 31.6% | 14.3% | 12.9% | 11.9% |
| 16 | 48 | 28.7% | 13.3% | 11.3% | 11.3% |
| 16 | 80 | 27.1% | 11.5% | 13.7% | 13.7% |
| 8 | 8 | 17.6% | 13.1% | 15.4% | 15.4% |
| 8 | 24 | 18.8% | 12.3% | 11.7% | 11.7% |
| 8 | 48 | 11.3% | 11.3% | 10.5% | 10.5% |
| 8 | 80 | 12.7% | 12.7% | 12.9% | 12.9% |

### Quarter-window (W=32), distance 80: individual seeds

| Seed | Sliding/rolling accuracy | Chunk/chunks accuracy | Chunk/rolling accuracy | Sliding/rolling answer NLL | Chunk/chunks answer NLL | Chunk/rolling answer NLL |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 100.0% | 16.4% | 16.4% | 0.1278 | 2.0997 | 2.1084 |
| 29 | 60.9% | 10.9% | 10.9% | 0.9208 | 2.0941 | 2.0940 |

Raw results: [results-2x-chunk/runs.json](results-2x-chunk/runs.json).

## Interpretation and limits

Independent chunks cannot recover a random source value outside the query chunk. That
structural limitation predicts chance-level retrieval (12.5% among eight values), rather
than indicating an implementation failure. The changed-source control verifies exact
invariance of native chunk outputs whenever the source lies in a different chunk.

Sliding/rolling versus chunk/chunks measures the practical difference between the two
training-and-inference systems. Sliding/rolling versus chunk/rolling holds inference
semantics fixed and tests the contribution of connected training. Chunk-trained models
may relay information when switched to rolling inference without having trained on that
cross-chunk pathway; the transfer evaluation is intentionally outside their training regime.

In the answer-only probe, earlier independent chunks receive no gradient, including their
learned absolute position embeddings. Rolling transfer therefore also uses positions outside
the chunk-trained probe's supervised region. Fixed positions or RoPE would help separate
that positional effect from learning to relay in a future experiment.

Any successful chunk/rolling result limits claims that connected training is necessary
for relay. Average text loss, task accuracy, and answer NLL can favor different conditions.
These results do not establish a converged capability advantage or general statistical equivalence.

Additional timed training: 585.8 seconds (24 new runs; reused full baselines excluded).
The controls use one CPU thread, with the original optimizer schedule and update budgets.
Training timing excludes evaluation and checkpoint writes. No sparse-kernel efficiency claim.

Reproduction and protocol: [DESIGN.md](DESIGN.md#independently-chunk-trained-control).

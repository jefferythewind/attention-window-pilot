# One-slot same-layer recurrent attention

Follow-up to V1 and [Thinking Outside the Box](https://arxiv.org/abs/2609.34049). V1 results remain unchanged in results-v1/. The new output directory is results-recurrent/.

## Architecture

`WindowConfig(recurrent_attention_slot=True)` enables the connection at every block. The default is false and retains the original vectorized code path. No parameters are introduced; the state dict is unchanged.

For each token, each block computes ordinary Q/K/V from `ln1(x_t)`. Ordinary attention includes the current token and at most W−1 earlier block inputs. If a previous completed block output exists, its `ln1` normalization is projected by the existing K and V rows of the same QKV matrix, including their biases. These K/V vectors are appended to the ordinary attention inputs and participate in the same softmax. The slot gets no additional positional embedding. The first token omits it entirely.

The state carried forward is the completed output after both attention and MLP residual additions. Ordinary K/V cache entries are always derived from block inputs; output-derived recurrent KVs never enter that cache. There is no additive temporal residual or separate attention branch.

Reference execution loops over tokens, then layers, parallelizing only the batch. Training retains the entire graph with no detaching or truncation; state updates use new tuples rather than in-place tensor writes. Causal next-token target alignment remains unchanged. Performance optimization is outside this experiment.

## Streaming state

Each layer returns `(ordinary_K, ordinary_V, ordinary_positions, completed_output, next_position)` when caching is enabled. K/V contain at most W−1 ordinary entries; the completed output has shape batch × 1 × width. The scalar next-position tensor also permits automatic streaming continuation when W=1 and ordinary K/V are empty. Cached chunks must continue at the next absolute position. A forward call with no supplied state resets both state types for every example. Training rejects supplied or returned caches.

Raw evaluation records ordinary K/V, recurrent-output, position-metadata, total logical, and deduplicated backing-storage bytes separately. Sliced tensors can retain a larger underlying allocation; logical bytes must not be confused with allocation size. Float32 recurrent outputs add 8 × 48 × 4 = 1,536 bytes (1.5 KiB) per sequence, independent of W. Positions are int64 and shared across the inference batch; raw byte counts specify batch size.

## Experiment

Preserve all 60 V1 results as reference conditions and train 36 new models: six seeds × recurrent windows 32/16/8 × separate text/retrieval tasks. Configuration is [recurrent_config.json](recurrent_config.json). Sequence length 128, eight layers, width 48, four heads, batch 16, float32; 200 text updates and 400 answer-supervised retrieval updates. Seeds, optimizer, learning-rate schedule, original ordered batches, every label, and update budgets match V1. Initialization hashes, input hashes, label hashes, parameter counts, token/target counts and update counts verify matching as results finish. Wall-clock time is not matched.

The primary comparison is recurrent-8 versus sliding-8 at distance 80. The baseline horizon is 8 × (8−1) = 56. Recurrence supplies a longer graph path, but success requires learned source-dependent predictions, not just graph connectivity. Report answer accuracy and NLL at distances 8/24/48/80, changed-source accuracy (correct answer changes to value+1 modulo 8), prediction-change fraction, and logit changes. Fresh held-out examples and changed-source pairs match V1.

The legacy full/sliding/chunk reference rows are copied without recomputing or overwriting their measurements. Only new recurrent conditions are trained. Native chunk-control training remains limited to V1 window 32; no new chunk-trained recurrent condition is added.

## Correctness and reproduction

```bash
.venv/bin/python checks_recurrent.py
.venv/bin/python checks.py
.venv/bin/python checks_chunk.py
.venv/bin/python run_recurrent.py
.venv/bin/python measure_retained_state.py
.venv/bin/python summarize_recurrent.py
```

Published results exclude checkpoint binaries. On a fresh checkout, regenerate V1 in a new directory (the V1 runner treats published metrics as completed jobs and needs their binaries), then place its checkpoint directories alongside the unchanged published V1 metrics. Use a new recurrent output directory as well:

```bash
.venv/bin/python run_v1.py --output-dir results-v1-reproduction
.venv/bin/python - <<'PYCODE'
from pathlib import Path
import shutil
for path in Path('results-v1-reproduction').iterdir():
    if path.is_dir():
        shutil.copytree(path, Path('results-v1')/path.name, dirs_exist_ok=True)
PYCODE
.venv/bin/python run_recurrent.py --output-dir results-recurrent-reproduction
.venv/bin/python measure_retained_state.py --output-dir results-recurrent-reproduction
.venv/bin/python summarize_recurrent.py --input-dir results-recurrent-reproduction
.venv/bin/pip install -r requirements-plots.txt
.venv/bin/python plot_recurrent.py --input-dir results-recurrent-reproduction
```

 The default runner reuses V1 checkpoints/results, requires their configuration and corpus hashes to match, and resumes completed recurrent jobs only when all recorded source hashes still match. Interrupted incomplete models restart from the original initialization; no partial optimizer-state resume is claimed.

The checks compare flag-off outputs and gradients directly with published V1 commit ac7dfe0; validate the recurrent connection against an independent explicit joint-softmax reference with nonzero projection biases; compare full-sequence/tokenwise/chunked evaluation; test causality including answer-token leakage, cache bounds, W=1 continuation, independent examples and HF checkpoint restoration; and verify both forward influence and nonzero gradient on a controlled distance-4 example with a baseline horizon of 2. The independent reference also matches parameter gradients, including the shared normalization and biased K/V projections. The controlled projections avoid numerical underflow. The baseline gives exactly zero influence and gradient.

Training timing covers batch creation, hashing, full forward/backward, clipping and optimizer updates, but excludes final evaluation and checkpoint writes. Three worker processes run concurrently with one PyTorch thread each; aggregate training durations overlap. End-to-end wall time is reported separately. No sparse compute or training-memory improvement is claimed.

Failure at this budget is failure of this particular configuration to learn useful relay, not proof that the connection cannot support relay. Text loss and synthetic retrieval belong to separately trained models; neither establishes broad capability equivalence. Learned absolute positions and the single-value task remain V1 limitations.
